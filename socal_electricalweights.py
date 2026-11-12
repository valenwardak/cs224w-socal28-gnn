# =============================================================================
# File:        socal_electricalweights.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Helper functions for the electrical edge weights, the second
#              alternative to the current meter graph.
#
#              Why it exists: the course staff's milestone feedback asked
#              whether the model uses the graph. In the current meter graph
#              every edge has weight 1.0, so the graph says which meters
#              are connected but not how strongly. A pair of meters across
#              one large transformer and a pair separated by two
#              transformers and two lines count the same. These functions
#              compute, for each period of the record, how strongly each
#              connected pair of meters is coupled through the electrical
#              network, so that the weight can replace the 1.0.
#
#              The construction, as the reviewers specified it:
#                1. Terminal blocks. graph.json gives every line and
#                   transformer a 6x6 block in siemens: the currents at the
#                   three phases of its two ends as a function of the
#                   voltages at the two ends. The block holds everything
#                   the element does: its series admittance, its shunt
#                   admittance, and for a transformer its voltage ratio
#                   and the phase shift of its winding connection. The
#                   complete block is used. Nothing is taken out of it.
#                2. Positive sequence. Each of the four 3x3 parts of the
#                   block (end 1 to end 1, end 1 to end 2, end 2 to end 1,
#                   end 2 to end 2) is reduced to one complex number by the
#                   symmetrical-components transform. That assumes a
#                   balanced element. A coupling test measures how well
#                   the assumption holds, and an element that fails it
#                   stops the construction. It is not approximated.
#                3. Per unit. Every value is put on one power base
#                   (s_base) and on the voltage base of the bus at each
#                   end, so that values at 277 V, 2.4 kV and 16.5 kV can
#                   be combined: y_ab(per unit) = y_ab(siemens) * V_a * V_b
#                   / s_base, with V_a and V_b the nominal line-to-line
#                   voltages of the two ends. The network file stores the
#                   nominal line-to-neutral voltage, so V = sqrt(3) V_LN.
#                   A transformer whose rated voltage differs from the
#                   nominal voltage of its bus keeps that ratio in its
#                   block, where it belongs.
#                4. The network of a period. Buses joined by a closed
#                   switching element (switch, breaker, multi-position
#                   switch, fault interrupter, fuse) or by a line that has
#                   no impedance block in graph.json are one node. The
#                   switch states come from the network file, or from the
#                   switch's status file at the period's midpoint. Every
#                   element's four numbers are added into the nodal
#                   admittance matrix at its two nodes.
#                5. Reduction. The nodal admittance matrix of the period
#                   is reduced onto the nodes that hold a meter (Kron
#                   reduction): the nodes without a meter are eliminated.
#                   Parallel paths and branching are handled by the
#                   reduction itself, which a series formula along one
#                   path would get wrong. The reduction refuses to run on
#                   a badly conditioned block, and checks its own solve.
#                6. Weight. Only here does the complex result become one
#                   number: the weight of a meter pair is the magnitude
#                   of their entry in the reduced matrix.
#                7. Scaling. Within one edge set the weights are divided by
#                   the largest weight, so they lie in (0, 1]. Two meters
#                   on the same node have no impedance between them. They
#                   are marked (their raw weight is None, not a number)
#                   and get 1.0.
#
#              A second construction is kept only for comparison, and is
#              labeled as an approximation wherever it appears: lines by
#              their series part alone, transformers from the nameplate
#              (percent impedance and rated power) with no ratio and no
#              phase shift.
#
#              What the weight is not: it is not an electrical equivalent
#              of the network. It is positive-sequence only, and a single
#              magnitude cannot carry the angle of the reduced entry. It
#              is a scalar measure of how strongly two metered nodes are
#              coupled.
#
# Used by:     main_electricalweights.py (writes the weighted graph file and
#              the coefficient report) and checks/main_electricalcheck.py.
#
# Usage:       Imported, not run. For example:
#                branches, notes = branches_per_unit(net, graph, s_base)
#                root, unresolved = merged_nodes(net, records, when, set(branches))
#                Y, order = nodal_matrix(branches, root)
#
# Functions:
#   Per unit and sequence
#     sequence_matrix()                       the symmetrical-components transform
#     positive_sequence(block, limit)         positive-sequence entry of a 3x3 block, with the coupling test
#     base_voltage(v_ln)                      nominal line-to-line voltage from the line-to-neutral voltage
#     terminal_per_unit(Y6, v_ln_1, v_ln_2, s_base, limit)
#                                             the four per-unit positive-sequence numbers of a 6x6 block
#     transformer_per_unit(kva, percent_z, s_base, x_over_r)
#                                             nameplate series admittance of a transformer (approximation and cross-check)
#     series_terminal(y)                      the four numbers of a plain series branch
#     terminal_summary(t)                     ratio, phase shift, X/R and ideal-transformer test of four numbers
#   Network of a period
#     switch_state(status, records, when)     NC or NO of one switching position at a time
#     merged_nodes(net, records, when, impedance_elements)
#                                             bus -> node representative under the switch states at a time
#     branches_per_unit(net, graph, s_base, rating_index, limit, construction, x_over_r)
#                                             every line and transformer with a block, per unit, with notes
#     nodal_matrix(branches, root)            the per-unit nodal admittance matrix of the merged network
#   Reduction and weights
#     kron_reduce(Y, keep, condition_limit, residual_limit)
#                                             eliminate every node not in keep that is connected to a kept node
#     pair_weights(Y_red, position, pairs, asymmetry)
#                                             weight per meter pair from the reduced matrix, None for one node
#     check_weights(raw)                      refuse a weight that is not a positive finite number or None
#     scale_weights(raw)                      divide by the largest weight; same-node pairs get 1.0
#     same_weights(a, b, tolerance)           whether two weight lists agree
#     weighted_edge_sets(period_graph, raws, n_graphs, tolerance)
#                                             split an edge set whose periods carry different weights
#     gcn_coefficients(n, edges, weights)     self-loop and neighbor coefficients of the GCN normalization
#   One period, end to end
#     period_reduction(net, records, timeline, index, branches, tolerance, condition_limit, residual_limit)
#                                             rebuild, check against the timeline, reduce, weigh the period's edges
# =============================================================================

# Third party: numerics
import numpy as np
# Third party: timestamps of the periods
import pandas as pd

# Element kinds of the network file that join two buses with no impedance when closed
SWITCHING_KINDS = ("Switch", "CB", "SwitchMultiPosition", "VFI", "Fuse")
# A positive-sequence entry below this share of the block's largest entry is treated as zero and reported, not used
NEAR_ZERO = 1e-9
# The two constructions: the complete terminal blocks, and the series-only approximation kept for comparison
CONSTRUCTIONS = ("terminal", "series")


def sequence_matrix():
    """
    The symmetrical-components transform T = (1/3) [[1, 1, 1], [1, a, a^2],
    [1, a^2, a]] with a = 1 at an angle of 120 degrees. A phase-domain 3x3
    block B becomes T B T^-1 in sequence order zero, positive, negative,
    so the positive-sequence entry is at index 1.
    """
    # The 120 degree rotation
    a = np.exp(2j * np.pi / 3)
    # The transform
    return np.array([[1, 1, 1], [1, a, a * a], [1, a * a, a]], dtype=complex) / 3.0


def positive_sequence(block, limit=0.01):
    """
    The positive-sequence entry of a 3x3 phase-domain block, with the
    coupling test. Returns (value, coupling, status). coupling is the
    largest off-sequence entry in the positive row or column divided by
    the positive entry: a balanced block gives 0. status is "ok", or
    "coupled" when coupling exceeds limit, or "near zero" when the positive
    entry is below NEAR_ZERO times the block's largest entry, in which case
    the value is not usable and coupling is None.
    """
    # The transform
    T = sequence_matrix()
    # The block in sequence coordinates
    seq = T @ np.asarray(block, dtype=complex) @ np.linalg.inv(T)
    # The positive-sequence entry
    value = seq[1, 1]
    # The block's largest entry, for the near-zero test
    top = float(np.abs(np.asarray(block)).max())
    # An empty or vanishing positive-sequence entry cannot be used
    if top == 0.0 or abs(value) < NEAR_ZERO * top:
        # Report
        return value, None, "near zero"
    # Largest coupling between the positive sequence and the other two
    coupling = float(max(abs(seq[1, 0]), abs(seq[1, 2]), abs(seq[0, 1]), abs(seq[2, 1])) / abs(value))
    # Done
    return value, coupling, ("ok" if coupling <= limit else "coupled")


def base_voltage(v_ln):
    """The nominal line-to-line voltage of a bus, in volts, from its nominal line-to-neutral voltage: sqrt(3) times v_ln."""
    # Line-to-line from line-to-neutral
    return np.sqrt(3.0) * float(v_ln)


def terminal_per_unit(Y6, v_ln_1, v_ln_2, s_base, limit=0.01):
    """
    The complete terminal block of one element in per unit, positive
    sequence. Y6 is the 6x6 block in siemens: rows and columns are end 1
    phases a b c, then end 2 phases a b c. v_ln_1 and v_ln_2 are the
    nominal line-to-neutral voltages of the buses at the two ends, in
    volts. Each of the four 3x3 parts is reduced to its positive-sequence
    entry and put per unit with the voltage bases of its own two ends:
        y_ab = y_ab(siemens) * V_a * V_b / s_base
    Nothing is left out: the shunt part, a voltage ratio and a phase shift
    all stay in the four numbers. Returns a dictionary with y (a 2x2
    complex array, rows and columns end 1 then end 2), coupling (the
    largest of the four coupling measures) and status ("ok" only when all
    four parts pass the coupling test).
    """
    # The block as complex numbers
    Y = np.asarray(Y6, dtype=complex)
    # Line-to-line base voltages of the two ends
    V = (base_voltage(v_ln_1), base_voltage(v_ln_2))
    # Outputs: the four numbers, the worst coupling, the worst status
    y, worst, status = np.zeros((2, 2), dtype=complex), 0.0, "ok"
    # Each of the four parts
    for a in range(2):
        # Second index
        for b in range(2):
            # Positive-sequence entry of the 3x3 part
            value, coupling, part = positive_sequence(Y[3 * a:3 * a + 3, 3 * b:3 * b + 3], limit)
            # Per unit on the bases of its own two ends
            y[a, b] = value * V[a] * V[b] / float(s_base)
            # Keep the first status that is not ok
            status = part if status == "ok" else status
            # Keep the largest coupling
            worst = max(worst, coupling if coupling is not None else 0.0)
    # Done
    return {"y": y, "coupling": worst, "status": status}


def transformer_per_unit(kva, percent_z, s_base, x_over_r=None):
    """
    Per-unit series admittance of a two-winding transformer from its
    nameplate. With z = percent_z / 100 on the rated power, the impedance
    on the common base is z * s_base / s_rated, so the admittance is
    (s_rated / s_base) / z. The nameplate gives no resistance: with
    x_over_r None the admittance is a pure reactance, -j times that
    magnitude. With a ratio of reactance to resistance, the same magnitude
    is split into resistance and reactance. This is the approximation
    kept for comparison, and the cross-check of the terminal blocks. It
    has no voltage ratio and no phase shift.
    """
    # The inputs must be positive numbers
    if not (float(kva) > 0 and float(percent_z) > 0 and float(s_base) > 0) or (x_over_r is not None and not float(x_over_r) > 0):
        # Refuse
        raise ValueError(f"refused: transformer nameplate needs positive values, got kva {kva}, percent_z {percent_z}, s_base {s_base}, x_over_r {x_over_r}")
    # Magnitude on the common base
    size = (float(kva) * 1e3 / float(s_base)) / (float(percent_z) / 100.0)
    # No resistance given: pure reactance
    if x_over_r is None:
        # -j times the magnitude
        return complex(0.0, -size)
    # Direction of the impedance R + jX with X / R as given, of length 1
    z_dir = complex(1.0, float(x_over_r)) / abs(complex(1.0, float(x_over_r)))
    # Admittance of that impedance, with the nameplate magnitude
    return size / z_dir


def series_terminal(y):
    """The four numbers of a plain series branch of admittance y between two ends: [[y, -y], [-y, y]]. No shunt, no ratio, no phase shift."""
    # The two-terminal pattern
    return complex(y) * np.array([[1, -1], [-1, 1]], dtype=complex)


def terminal_summary(t):
    """
    What the four per-unit numbers of an element say, for the report and
    the checks. For an ideal transformer with series admittance y (seen
    from end 2), ratio r at end 1 and phase shift p, the four numbers are
    y / r^2, -(y / r) e^{jp}, -(y / r) e^{-jp} and y. So:
      ratio        sqrt(|y22| / |y11|), 1 when the rated voltages equal the bases
      shift_deg    the angle of -y12 / y22, in degrees
      x_over_r     -imag(y22) / real(y22), None when the real part is zero
      residual     |y11 y22 - y12 y21| / |y11 y22|: 0 for an ideal transformer
                   or a plain series branch, larger when the block holds a
                   shunt part
    """
    # The four numbers
    y11, y12, y21, y22 = t[0, 0], t[0, 1], t[1, 0], t[1, 1]
    # Done
    return {"ratio": float(np.sqrt(abs(y22) / abs(y11))), "shift_deg": float(np.degrees(np.angle(-y12 / y22))),
            "x_over_r": (float(-y22.imag / y22.real) if y22.real != 0 else None), "residual": float(abs(y11 * y22 - y12 * y21) / abs(y11 * y22))}


def switch_state(status, records, when):
    """
    The state, "NC" (closed) or "NO" (open), of one switching position at
    the time `when`. status is the entry of the network file: "NC", "NO",
    or "file:<name>-tbus_status.csv", which points to the switch's status
    records. records is the dictionary from
    socal_metergraph.read_switch_records. The state in force is the last
    record at or before `when`. Returns None when the file has no record
    that early or is not among the records.
    """
    # A fixed state from the network file
    if not str(status).startswith("file:"):
        # As listed
        return status
    # The switch name inside "file:<name>-tbus_status.csv"
    name = str(status)[len("file:"):-len("-tbus_status.csv")]
    # No records for this switch
    if name not in records:
        # Unresolved
        return None
    # The last state at or before the time
    state = None
    # Records are in time order
    for t, s in records[name]:
        # At or before
        if t <= when:
            # Latest so far
            state = s
    # Done
    return state


def merged_nodes(net, records, when, impedance_elements):
    """
    The nodes of the electrical network at the time `when`: buses joined
    by a closed switching element, or by a line that is not among
    impedance_elements (a line with no impedance block in graph.json is a
    plain connection), are one node. Returns (root, unresolved): root maps
    every bus name to the name of its node's representative bus, and
    unresolved lists the switching positions whose state could not be read
    (they are treated as open).
    """
    # Every bus starts as its own node
    parent = {b["name"]: b["name"] for b in net["Bus"]}

    # Find the representative of a bus, shortening the chain on the way
    def find(x):
        # Walk up
        while parent[x] != x:
            # Shorten
            parent[x] = parent[parent[x]]
            # Step
            x = parent[x]
        # Representative
        return x

    # Positions whose state could not be read
    unresolved = []
    # Closed switching elements join their two buses
    for kind in SWITCHING_KINDS:
        # Each element of the kind
        for e in net.get(kind, []):
            # Each position of the element
            for k, tb in enumerate(e["tbus"]):
                # State at the time
                state = switch_state(tb.get("status"), records, when)
                # Unknown state: record it and treat it as open
                if state not in ("NC", "NO"):
                    # Keep
                    unresolved.append(f"{e['name']} position {k} ({tb.get('status')})")
                # Closed: one node
                if state == "NC":
                    # Join
                    parent[find(e["fbus"])] = find(tb["name"])
    # Lines without an impedance block are plain connections
    for line in net.get("Line", []):
        # Only those without impedance
        if line["name"] not in impedance_elements:
            # Each far end
            for tb in line["tbus"]:
                # Join
                parent[find(line["fbus"])] = find(tb["name"])
    # Representative of every bus
    return {b: find(b) for b in parent}, unresolved



def branches_per_unit(net, graph, s_base, rating_index=0, limit=0.01, construction="terminal", x_over_r=None):
    """
    Every line and transformer that graph.json gives a block, per unit on
    s_base. Returns (branches, notes). branches maps the element name to
    {"kind", "from", "to", "y"} with y the 2x2 complex array of its four
    per-unit numbers, rows and columns in the order from, to.

    construction "terminal" (the one used for the weights) takes the four
    numbers from the complete block. construction "series" is the
    approximation kept for comparison: a line by the series part of its
    block alone, a transformer from its nameplate (x_over_r as in
    transformer_per_unit), both as plain series branches.

    An element that cannot be put per unit is left out of branches and
    named in notes["unresolved"]: a block that fails the coupling test, a
    bus without a nominal voltage, a block whose ends do not match the
    element's buses, a transformer with other than two windings. The
    caller must stop on any such entry. notes also holds one record per
    line and per transformer for the report, and the transformers whose
    rated voltage differs from the nominal voltage of their bus
    (notes["off_nominal"]).
    """
    # Only the two known constructions
    if construction not in CONSTRUCTIONS:
        # Refuse
        raise ValueError(f"refused: construction must be one of {CONSTRUCTIONS}, got {construction}")
    # The power base must be a positive number
    if not float(s_base) > 0 or not np.isfinite(float(s_base)):
        # Refuse
        raise ValueError(f"refused: s_base must be a positive finite number, got {s_base}")
    # Nominal line-to-neutral voltage of every bus
    v_ln = {b["name"]: b.get("nominal_voltage") for b in net["Bus"]}
    # Lines by name
    lines = {x["name"]: x for x in net.get("Line", [])}
    # Transformers by name
    transformers = {x["name"]: x for x in net.get("Transformer", [])}
    # Outputs
    branches, notes = {}, {"unresolved": [], "transformers": [], "off_nominal": [], "lines": []}
    # Each element with a block
    for e in graph["edges"]:
        # Its name and its complex block
        name, Y6 = e["element"], np.array(e["Y_real"], dtype=float) + 1j * np.array(e["Y_imag"], dtype=float)
        # Its entry in the network file
        item = lines.get(name) if e["type"] == "Line" else transformers.get(name) if e["type"] == "Transformer" else None
        # Not found, or more than one far end
        if item is None or len(item["tbus"]) != 1:
            # Report and leave out
            notes["unresolved"].append(f"{name}: type {e['type']} not found among the network file's lines and transformers, or not a two-ended element")
            # Next
            continue
        # Its two buses
        fb, tb = item["fbus"], item["tbus"][0]["name"]
        # The block's end 1 must be the from bus: graph.json's source node must hold it, and the target node the to bus
        if fb not in graph["nodes"][e["source"]]["buses"] or tb not in graph["nodes"][e["target"]]["buses"]:
            # Report and leave out
            notes["unresolved"].append(f"{name}: the ends of its block in graph.json do not match its buses {fb} and {tb}")
            # Next
            continue
        # Both buses need a nominal voltage
        if not v_ln.get(fb) or not v_ln.get(tb):
            # Report and leave out
            notes["unresolved"].append(f"{name}: no nominal voltage for {fb} or {tb}, no voltage base")
            # Next
            continue
        # The complete block per unit
        full = terminal_per_unit(Y6, v_ln[fb], v_ln[tb], s_base, limit)
        # A block that is not balanced enough is not used in either construction
        if full["status"] != "ok":
            # Report and leave out
            notes["unresolved"].append(f"{name}: positive-sequence reduction {full['status']} (coupling {full['coupling']})")
            # Next
            continue
        # What the four numbers say
        summary = terminal_summary(full["y"])
        # A line
        if e["type"] == "Line":
            # The series part alone: minus the end-to-end number
            series = series_terminal(-full["y"][0, 1])
            # Keep the construction asked for
            branches[name] = {"kind": "Line", "from": fb, "to": tb, "y": full["y"] if construction == "terminal" else series}
            # Record: the shunt part is what the four numbers add up to, relative to the series part
            notes["lines"].append({"name": name, "v_ln": float(v_ln[fb]), "y_series_abs": float(abs(full["y"][0, 1])), "coupling": full["coupling"], "x_over_r": summary["x_over_r"],
                                   "shunt_ratio": float(abs(full["y"].sum()) / abs(full["y"][0, 1])), "ends_differ": float(abs(v_ln[fb] - v_ln[tb]) / v_ln[fb])})
        # A transformer
        else:
            # Only two windings are handled
            if int(item.get("num_windings", 2)) != 2:
                # Report and leave out
                notes["unresolved"].append(f"{name}: not a two-winding transformer")
                # Next
                continue
            # The rating in use when several are listed
            kva = item["kva"][min(int(rating_index), len(item["kva"]) - 1)]
            # Rated voltages of the from and to windings, in kilovolts
            kv_from, kv_to = float(item["kv"][0][0]), float(item["kv"][0][1])
            # Nameplate series admittance
            plate = transformer_per_unit(kva, item["percent_z"][0], s_base, x_over_r)
            # Keep the construction asked for
            branches[name] = {"kind": "Transformer", "from": fb, "to": tb, "y": full["y"] if construction == "terminal" else series_terminal(plate)}
            # The block's own series admittance on the transformer's rated voltage, to set beside the nameplate: end 2 number, moved from the bus base to the rated base
            on_rated = full["y"][1, 1] * (kv_to * 1e3 / base_voltage(v_ln[tb])) ** 2
            # Record
            notes["transformers"].append({"name": name, "conn": list(item.get("conn", [])), "kva_listed": list(item["kva"]), "kva_used": float(kva), "percent_z": float(item["percent_z"][0]), "kv": [kv_from, kv_to],
                                          "nameplate_abs": float((float(kva) * 1e3 / float(s_base)) / (float(item["percent_z"][0]) / 100.0)), "block_on_rated": complex(on_rated), "y11_abs": float(abs(full["y"][0, 0])), "y22_abs": float(abs(full["y"][1, 1])),
                                          "ratio": summary["ratio"], "shift_deg": summary["shift_deg"], "x_over_r": summary["x_over_r"], "residual": summary["residual"]})
            # Rated voltage against the nominal voltage of the bus, each winding
            for rated, bus in ((kv_from, fb), (kv_to, tb)):
                # The bus level in kilovolts
                level = base_voltage(v_ln[bus]) / 1e3
                # More than half a percent apart
                if abs(rated - level) > 0.005 * level:
                    # Report
                    notes["off_nominal"].append(f"{name}: winding rated {rated} kV on a {level:.3f} kV bus (ratio {rated / level:.4f})")
    # Done
    return branches, notes


def nodal_matrix(branches, root):
    """
    The per-unit nodal admittance matrix of the merged network: one row
    per node. Each element adds its four numbers at the rows and columns
    of its two nodes, so its shunt part, ratio and phase shift all enter
    the matrix. An element whose two ends are the same node (joined by a
    closed switch) adds all four numbers to that node's diagonal entry.
    Returns (Y, order) with order the sorted list of node representatives,
    the row order.
    """
    # Nodes in a fixed order
    order = sorted(set(root.values()))
    # Row of each node
    row = {r: i for i, r in enumerate(order)}
    # Empty matrix
    Y = np.zeros((len(order), len(order)), dtype=complex)
    # Each element
    for b in branches.values():
        # The rows of its two ends
        at = (row[root[b["from"]]], row[root[b["to"]]])
        # Each of its four numbers
        for a in range(2):
            # Second index
            for c in range(2):
                # Add at the rows of the two ends
                Y[at[a], at[c]] += b["y"][a, c]
    # Done
    return Y, order


def kron_reduce(Y, keep, condition_limit=1e12, residual_limit=1e-9):
    """
    Kron reduction of a nodal admittance matrix onto the rows in keep:
    Y_red = Y_kk - Y_ke inv(Y_ee) Y_ek, where e are the eliminated rows.
    Only rows that are connected, through the matrix, to a kept row are
    eliminated. A part of the network with no kept row is left out: it
    cannot couple two kept rows.

    The reduction refuses (ValueError) instead of returning a doubtful
    result:
      - when the matrix holds a value that is not finite,
      - before solving, when a connected part of the eliminated block has
        a condition number above condition_limit, or one that is not
        finite (the block is checked part by part, since unrelated parts
        of different size would inflate one overall number),
      - after solving, when the solution does not satisfy its own
        equations to within residual_limit (relative), or the reduced
        matrix holds a value that is not finite.
    Returns (Y_red, info) with Y_red in the order of keep and info holding
    the rows eliminated, the rows left out, the number of connected parts
    of the eliminated block, the largest condition number among them and
    the relative residual of the solve (0 when nothing is eliminated).
    """
    # The limits must be positive numbers
    if not float(condition_limit) > 0 or not float(residual_limit) > 0:
        # Refuse
        raise ValueError(f"refused: condition_limit and residual_limit must be positive, got {condition_limit} and {residual_limit}")
    # The matrix as complex numbers
    Y = np.asarray(Y, dtype=complex)
    # Every value must be finite
    if not np.isfinite(Y).all():
        # Refuse
        raise ValueError("refused: the nodal matrix holds a value that is not finite")
    # Size
    n = Y.shape[0]
    # Kept rows as a list
    keep = list(keep)
    # Kept rows as a set
    kept = set(keep)
    # Neighbors of every row: a nonzero entry in either direction
    nb = [(set(np.nonzero(np.abs(Y[i]) > 0)[0].tolist()) | set(np.nonzero(np.abs(Y[:, i]) > 0)[0].tolist())) - {i} for i in range(n)]
    # Rows reachable from a kept row
    seen, stack = set(keep), list(keep)
    # Search
    while stack:
        # Next row
        u = stack.pop()
        # Its neighbors
        for v in nb[u]:
            # New
            if v not in seen:
                # Mark
                seen.add(v)
                # Expand
                stack.append(v)
    # Rows to eliminate: reachable and not kept
    elim = [i for i in range(n) if i in seen and i not in kept]
    # Rows left out: not reachable from any kept row
    left_out = n - len(seen)
    # Nothing to eliminate: the reduced matrix is the kept block itself
    if not elim:
        # Kept block
        return Y[np.ix_(keep, keep)].copy(), {"eliminated": 0, "left_out": left_out, "parts": 0, "condition": 0.0, "residual": 0.0}
    # Connected parts of the eliminated block, through its own entries
    parts, placed = [], set()
    # Each eliminated row starts a part unless it is already in one
    for start in elim:
        # Already placed
        if start in placed:
            # Next
            continue
        # Grow the part
        part, stack = {start}, [start]
        # Search inside the eliminated rows
        while stack:
            # Next row
            u = stack.pop()
            # Its neighbors that are eliminated rows
            for v in nb[u]:
                # New and eliminated
                if v not in part and v not in kept and v in seen:
                    # Mark
                    part.add(v)
                    # Expand
                    stack.append(v)
        # Keep the part
        parts.append(sorted(part))
        # Mark its rows
        placed |= part
    # Condition number of each part, before any solve
    worst = 0.0
    # Each part
    for part in parts:
        # Its condition number
        c = float(np.linalg.cond(Y[np.ix_(part, part)]))
        # Refuse a part that is singular or close to it
        if not np.isfinite(c) or c > float(condition_limit):
            # Refuse
            raise ValueError(f"refused: a connected part of {len(part)} eliminated rows has condition number {c:.3e}, above the limit {float(condition_limit):.1e}")
        # Largest so far
        worst = max(worst, c)
    # Blocks
    Ykk, Yke, Yee, Yek = Y[np.ix_(keep, keep)], Y[np.ix_(keep, elim)], Y[np.ix_(elim, elim)], Y[np.ix_(elim, keep)]
    # Solve Y_ee X = Y_ek, not by inverting
    X = np.linalg.solve(Yee, Yek)
    # How well the solution satisfies its equations, relative to the right-hand side
    residual = float(np.abs(Yee @ X - Yek).max() / max(np.abs(Yek).max(), 1e-300))
    # Refuse a solve that does not check
    if not np.isfinite(residual) or residual > float(residual_limit):
        # Refuse
        raise ValueError(f"refused: the solve leaves a relative residual of {residual:.3e}, above the limit {float(residual_limit):.1e}")
    # Reduce
    Y_red = Ykk - Yke @ X
    # The result must be finite
    if not np.isfinite(Y_red).all():
        # Refuse
        raise ValueError("refused: the reduced matrix holds a value that is not finite")
    # Done
    return Y_red, {"eliminated": len(elim), "left_out": left_out, "parts": len(parts), "condition": worst, "residual": residual}


def pair_weights(Y_red, position, pairs, asymmetry=1e-9):
    """
    The raw weight of every meter pair from the reduced matrix. position
    maps a meter name to its row in Y_red (two meters on one node share a
    row). pairs is a list of (meter, meter). The weight is the mean of the
    magnitudes of the two directed entries. A phase shift makes the two
    entries differ in angle, not in magnitude. A pair on one node gets
    None: no impedance lies between them, so there is no number to give.
    None is a marker, never a numerical result. Returns (weights, notes)
    with notes listing pairs whose two directed magnitudes differ by more
    than `asymmetry`, relative.
    """
    # Outputs
    weights, notes = [], []
    # Each pair
    for a, b in pairs:
        # Rows
        i, j = position[a], position[b]
        # Same node
        if i == j:
            # The marker
            weights.append(None)
            # Next
            continue
        # The two directed magnitudes
        wij, wji = abs(Y_red[i, j]), abs(Y_red[j, i])
        # They should agree
        if abs(wij - wji) > asymmetry * max(wij, wji, 1e-300):
            # Report
            notes.append(f"{a} - {b}: |Y_red| {wij} against {wji}")
        # Mean of the two
        weights.append(float(0.5 * (wij + wji)))
    # Done
    return weights, notes


def check_weights(raw):
    """
    Refuse (ValueError) a list of raw weights that holds anything but a
    positive finite number or the same-node marker None. A zero, a
    negative number, an infinity or a not-a-number is a numerical failure
    or a pair that is not coupled. None of them is ever turned into a
    weight.
    """
    # Each weight
    for k, w in enumerate(raw):
        # The same-node marker is allowed
        if w is None:
            # Next
            continue
        # Anything else must be a positive finite number
        if isinstance(w, bool) or not isinstance(w, (int, float, np.integer, np.floating)) or not np.isfinite(w) or not w > 0:
            # Refuse
            raise ValueError(f"refused: weight {k} is {w!r}; a weight must be a positive finite number, or None for two meters on one node")


def scale_weights(raw):
    """
    Scale the raw weights of one edge set to (0, 1]: divide by the largest
    weight. A pair on one node (marker None) gets 1.0, the same as the
    strongest pair. Refuses an invalid weight (see check_weights).
    Returns (scaled, top) with top the largest raw weight. An empty list
    gives ([], None). A list of same-node pairs only gives all ones and a
    top of None.
    """
    # Refuse anything that is not a positive finite number or the marker
    check_weights(raw)
    # The numbers among them
    numbers = [float(w) for w in raw if w is not None]
    # No number: an empty set, or same-node pairs only
    if not numbers:
        # All ones, no scale
        return [1.0 for _ in raw], None
    # Largest weight
    top = max(numbers)
    # Scaled
    return [1.0 if w is None else float(w) / top for w in raw], float(top)


def same_weights(a, b, tolerance):
    """Whether two lists of raw weights agree: same-node markers in the same places, numbers within the relative tolerance."""
    # Same length
    if len(a) != len(b):
        # Different
        return False
    # Entry by entry
    for x, y in zip(a, b):
        # A marker on one side only
        if (x is None) != (y is None):
            # Different
            return False
        # Two numbers that are apart
        if x is not None and abs(x - y) > tolerance * max(abs(x), abs(y), 1e-300):
            # Different
            return False
    # Same
    return True


def weighted_edge_sets(period_graph, raws, n_graphs, tolerance):
    """
    Decide which weighted edge sets the file needs. period_graph gives,
    per period, the number of its edge set in the source file. raws gives,
    per period, the raw weights of that edge set's edges in that period's
    network. Periods that share an edge set usually have the same
    weights. When they do not (same connections, different network), the
    edge set is split so that every period keeps the weights of its own
    network and no period boundary moves.

    Returns (sets, new_period_graph). sets is a list of dictionaries
    {"source", "variant", "raw", "periods"}: first one entry per source
    edge set, in source order, so those keep their numbers; then the
    further variants. An edge set no period uses has raw None.
    new_period_graph gives each period's number in that list.
    """
    # First variants in source order, further variants after them
    first, extra = [], []
    # Each source edge set
    for g in range(int(n_graphs)):
        # Distinct weight lists among its periods, each with its periods
        variants = []
        # Each period that uses it
        for i in [k for k, pg in enumerate(period_graph) if int(pg) == g]:
            # A known variant with the same weights
            known = next((v for v in variants if same_weights(v["raw"], raws[i], tolerance)), None)
            # New
            if known is None:
                # Keep
                variants.append({"source": g, "variant": len(variants), "raw": raws[i], "periods": [i]})
            # Known
            else:
                # Add the period
                known["periods"].append(i)
        # An edge set no period uses
        if not variants:
            # Placeholder
            variants.append({"source": g, "variant": 0, "raw": None, "periods": []})
        # The first variant keeps the edge set's number
        first.append(variants[0])
        # The others go after all source edge sets
        extra.extend(variants[1:])
    # Final list
    sets = first + extra
    # Each period's number in the list
    new_period_graph = [None] * len(period_graph)
    # Each set
    for index, s in enumerate(sets):
        # Each of its periods
        for i in s["periods"]:
            # Its number
            new_period_graph[i] = index
    # Done
    return sets, new_period_graph


def gcn_coefficients(n, edges, weights):
    """
    The coefficients the GCN layer applies with these weights and a
    self-loop of 1: d_i = 1 + the sum of the weights of node i's edges,
    self-loop coefficient 1 / d_i, and neighbor coefficient
    w_ij / sqrt(d_i d_j) for each edge. They are coefficients, not shares:
    a node's self-loop coefficient and its neighbor coefficients do not
    add up to 1. edges is a list of (i, j) node pairs, undirected.
    Refuses a weight that is not a finite number of at least 0.
    Returns (degree (n,), self (n,), neighbor (len(edges),)).
    """
    # Every weight must be a finite number that is not negative
    if any(w is None or not np.isfinite(w) or w < 0 for w in weights):
        # Refuse
        raise ValueError("refused: GCN coefficients need finite weights of at least 0")
    # Degrees start at the self-loop
    d = np.ones(n, dtype=float)
    # Add each edge's weight to both ends
    for (i, j), w in zip(edges, weights):
        # One end
        d[i] += w
        # Other end
        d[j] += w
    # Neighbor coefficient per edge
    nbr = np.array([w / np.sqrt(d[i] * d[j]) for (i, j), w in zip(edges, weights)], dtype=float)
    # Done
    return d, 1.0 / d, nbr


def period_reduction(net, records, timeline, index, branches, tolerance=1e-9, condition_limit=1e12, residual_limit=1e-9):
    """
    One period of the time-indexed meter graph, end to end: rebuild the
    electrical network at the period's midpoint, check it against what the
    timeline file records for that period, reduce it onto the metered
    nodes, and weigh the edges of the period's edge set.

    timeline is the body of graph_meters_periods.json. Returns a
    dictionary with:
      pairs      the meter pairs of the period's edge set, in file order
      raw        their raw weights (None for two meters on one node)
      info       what kron_reduce reports
      problems   a list of text lines, empty when every check holds:
                 a switching position with no readable state, a meter
                 whose node holds other buses than the timeline records,
                 a connected pair that is not an edge, an edge that is not
                 connected, two directed weights that disagree, or a
                 reduction that was refused
    The checks are what make the rebuilt network trustworthy: it must
    reproduce the bus sets and the connections that the timeline file was
    built with by the original topology code.
    """
    # The period and its edge set
    period = timeline["periods"][index]
    # Edge set
    graph = timeline["graphs"][int(period["graph"])]
    # Meter names in node order
    names = [n["meter"] for n in timeline["nodes"]]
    # The meter pairs of the edge set
    pairs = [(names[e["source"]], names[e["target"]]) for e in graph["edges"]]
    # Midpoint of the period
    when = pd.Timestamp(period["start"]) + (pd.Timestamp(period["end"]) - pd.Timestamp(period["start"])) / 2
    # Merged nodes at that time
    root, unresolved = merged_nodes(net, records, when, set(branches))
    # Problems found
    problems = [f"period {index}: no readable state for {u}" for u in unresolved]
    # Buses of every node
    members = {}
    # Collect
    for bus, r in root.items():
        # Under its representative
        members.setdefault(r, []).append(bus)
    # Each meter's node must hold exactly the buses the timeline records
    for m in names:
        # The meter's voltage bus
        bus = timeline["meters"][m]["voltage_bus"]
        # Buses here
        mine = sorted(members[root[bus]])
        # Buses in the timeline file
        theirs = sorted(period["topology"][m]["node_buses"])
        # Different
        if mine != theirs:
            # Report
            problems.append(f"period {index}: {m} sits on {len(mine)} buses here and {len(theirs)} in the timeline file")
    # Nodal matrix
    Y, order = nodal_matrix(branches, root)
    # Row of each node
    row = {r: i for i, r in enumerate(order)}
    # Row of each meter's node
    meter_row = {m: row[root[timeline["meters"][m]["voltage_bus"]]] for m in names}
    # Kept rows: one per metered node
    keep = sorted(set(meter_row.values()))
    # Reduce; a refusal becomes a problem of this period, with no weights
    try:
        # Reduction with its checks
        Y_red, info = kron_reduce(Y, keep, condition_limit, residual_limit)
    # Refused
    except ValueError as err:
        # Report and stop here
        return {"pairs": pairs, "raw": None, "info": None, "problems": problems + [f"period {index}: {err}"], "when": str(when)}
    # Position of each meter in the reduced matrix
    position = {m: keep.index(meter_row[m]) for m in names}
    # The edges as unordered pairs
    have = {frozenset(p) for p in pairs}
    # Largest entry, for the relative zero test
    top = float(np.abs(Y_red).max()) if Y_red.size else 0.0
    # Every meter pair: coupled exactly when it is an edge
    for i, a in enumerate(names):
        # Later meters
        for b in names[i + 1:]:
            # Coupled: same node, or a nonzero reduced entry
            connected = position[a] == position[b] or abs(Y_red[position[a], position[b]]) > tolerance * max(top, 1e-300)
            # Coupled but not an edge
            if connected and frozenset((a, b)) not in have:
                # Report
                problems.append(f"period {index}: {a} and {b} are coupled in the reduced network but are not an edge")
            # An edge but not coupled
            if not connected and frozenset((a, b)) in have:
                # Report
                problems.append(f"period {index}: {a} and {b} are an edge but are not coupled in the reduced network")
    # Weights of the edges
    raw, notes = pair_weights(Y_red, position, pairs)
    # Disagreeing directions
    problems += [f"period {index}: {t}" for t in notes]
    # Done
    return {"pairs": pairs, "raw": raw, "info": info, "problems": problems, "when": str(when)}
