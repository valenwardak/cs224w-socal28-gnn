# =============================================================================
# File:        checks/main_electricalcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Test the electrical edge weights before any model is trained
#              on them.
#
#              Why it exists: the weights come from a chain of steps
#              (terminal blocks, sequence reduction, per unit, network
#              rebuild, Kron reduction, scaling). A mistake in any step
#              would still give plausible numbers between 0 and 1. So
#              every step is tested against a number that was worked out
#              another way: by hand, by a second route, or by a second
#              implementation written in this file. The steps must also
#              refuse bad input instead of turning it into a weight, and
#              the builder must never write over a file. Both are tested.
#
#              The cases:
#              Steps in isolation
#                V1  One transformer nameplate by two routes: 1,000 kVA,
#                    5 percent, 480 V, on 10 MVA gives 2.0 per unit from
#                    the formula and 2.0 through ohms.
#                V6  The sequence reduction: a balanced block, an
#                    unbalanced block (flagged), an empty block (flagged).
#                T1  The complete block of a line with a shunt part, per
#                    unit, against the four numbers worked by hand.
#                T2  The complete block of a delta-wye transformer, built
#                    here from three single-phase units and the winding
#                    connections, against the four complex numbers worked
#                    by hand: the series admittance, a 30 degree phase
#                    shift, and an off-nominal voltage ratio.
#              Reduction
#                V2  Two branches in a row, 4 and 4, middle removed: 2.0.
#                V3  A star of 2, 3 and 6, center removed: 6/11, 12/11 and
#                    18/11, and NOT the pair-alone series values.
#                K1  A mesh with a shunt at the removed node: 2.0 by hand,
#                    against 2.2 without the shunt.
#                K2  Nothing to remove: the matrix comes back unchanged.
#                K3  A part of the network with no meter is left out and
#                    does not disturb the result.
#                K4  A phase shift changes the angle of the reduced entry
#                    and not its magnitude.
#                K5  Two steps equal one step, and a second implementation
#                    (one node at a time) gives the same matrix, on a
#                    complex network with shunts.
#                K6  Refusals: a badly conditioned part (1e14), a value
#                    that is not finite, a solve that does not meet its
#                    residual limit, limits that are not positive.
#              Weights and scaling
#                S1  Scaling, with a same-node pair.
#                S2  An empty set, and a set of same-node pairs only.
#                S3  Refusals: zero, negative, not-a-number, plus and
#                    minus infinity, text, a truth value. None of them
#                    becomes a weight.
#                S4  No pairs gives no weights; GCN coefficients refuse a
#                    weight that is not a number.
#                S5  Splitting an edge set whose periods carry different
#                    weights, on a made-up case, since the real data never
#                    need it.
#              The real network
#                N1  The rebuilt network has the nodes of graph.json;
#                    every element is per unit; every transformer block is
#                    an ideal transformer with a 30 degree shift whose
#                    complex series admittance equals the nameplate value
#                    at the block's own X/R.
#                N2  Every period passes its own checks.
#                N3  A second implementation, with its own sequence
#                    formula, per-unit conversion, assembly and
#                    elimination, gives the same weights in every period.
#                N4  Periods that share an edge set have the same weights.
#                N5  The scaled weights do not depend on the power base.
#                N6  The GCN coefficients equal those of the normalized
#                    matrix built directly.
#                N7  Reported, not judged: how far the series-only
#                    approximation is from the complete-block weights,
#                    every period.
#                V4  A weaker transformer gives a weaker edge.
#                V5  The edge set with the bus tie closed against the
#                    reference edge set.
#              The builder's refusals (run as a separate process, in a
#              temporary folder)
#                R1  An existing graph file, an existing report file: not
#                    written over, byte for byte.
#                R2  One path for both outputs.
#                R3  An output that is an input: by another spelling of
#                    its path, through a symbolic link, through a hard
#                    link. The input is unchanged byte for byte.
#                R4  Options that are not positive finite numbers.
#                R5  A dry run finishes with exit code 0 and writes
#                    nothing, which shows the refusals above are not the
#                    builder failing for another reason.
#              The written file, when it is present
#                F1  Same nodes, meters, periods and connections as the
#                    source file; only the weights differ.
#                F2  Every weight is in (0, 1], every edge set has a
#                    largest weight of exactly 1.
#                F3  An edge is marked same_node exactly when its two
#                    meters sit on the same buses, and then carries 1.0.
#                F4  The stored weights equal the second implementation's.
#                F5  The file records the identities of every input in
#                    use, the switch record files included, and the source
#                    file still has every weight at 1.0.
#
#              Nothing is trained or scored. The script needs no torch.
#
# Usage (from the repo root):
#   python checks/main_electricalcheck.py
#   python checks/main_electricalcheck.py --weighted graph_meters_periods_pu.json
#   Options: python checks/main_electricalcheck.py --help
#
# Exit code:   0 when every check passes and none was skipped, 1 otherwise.
#
# Functions:
#   parse_args()                          command-line options with defaults
#   Tally                                 counts and prints PASS and FAIL lines
#   close(a, b, tolerance)                relative agreement of two numbers
#   refuses(call)                         whether a call raises ValueError
#   eliminate_one_by_one(Y, keep)         second implementation of the reduction
#   second_weights(net, graph, records, timeline, index, s_base)
#                                         a period's weights from the second implementation
#   delta_wye_block(y, turns)             6x6 block of a delta-wye transformer from first principles
#   case_steps(t, args)                   V1, V6, T1, T2
#   case_reduction(t, args)               V2, V3, K1 to K6
#   case_scaling(t, args)                 S1 to S5
#   case_network(t, args, data)           N1 to N7, V4, V5
#   case_refusals(t, args)                R1 to R5
#   case_file(t, args, data)              F1 to F5
#   main()                                run everything, print the tally, exit code
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: copy branches before changing one
import copy
# Standard library: file identities
import hashlib
# Standard library: JSON
import json
# Standard library: paths
import os
# Standard library: copy a file into the temporary folder
import shutil
# Standard library: run the builder as a separate process
import subprocess
# Standard library: exit code and import path
import sys
# Standard library: a temporary folder for the refusal tests
import tempfile
# Third party: numerics
import numpy as np
# Third party: times
import pandas as pd

# The repo root, one level above this file, so the project modules import from the checks folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Project: the switch records reader
import socal_metergraph as smg
# Project: the module under test
import socal_electricalweights as ew


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Test the electrical edge weights")
    # The electrical graph
    parser.add_argument("--graph", default="graph.json", help="Electrical graph at the reference time, with the terminal blocks")
    # The source timeline
    parser.add_argument("--timeline", default="graph_meters_periods.json", help="Time-indexed meter graph with unit weights")
    # The weighted file
    parser.add_argument("--weighted", default="graph_meters_periods_pu.json", help="Weighted graph file written by main_electricalweights.py; the file checks are skipped, and counted as not run, when it is absent")
    # Topology folder
    parser.add_argument("--topology-dir", default="topology", help="Caltech's topology folder")
    # Network file
    parser.add_argument("--network", default=os.path.join("topology", "network_files", "circuit3", "2023-08-01T00h00m00.000000s.json"), help="Network file")
    # The builder script
    parser.add_argument("--builder", default="main_electricalweights.py", help="The script that writes the weighted graph, for the refusal tests")
    # Power base
    parser.add_argument("--s-base", type=float, default=1e6, help="Common power base in volt-amperes")
    # Second power base
    parser.add_argument("--other-s-base", type=float, default=1e7, help="Second power base, for the test that scaled weights do not depend on it")
    # Resistance variant
    parser.add_argument("--variant-x-over-r", type=float, default=6.0, help="Ratio of reactance to resistance for the second series-only comparison")
    # Agreement
    parser.add_argument("--tolerance", type=float, default=1e-9, help="Relative difference below which two numbers count as equal")
    # Conditioning limit under test
    parser.add_argument("--condition-limit", type=float, default=1e12, help="Condition number limit of the reduction")
    # A condition number that must be refused
    parser.add_argument("--bad-condition", type=float, default=1e14, help="K6: condition number of the made-up badly conditioned part")
    # V1 inputs: rated power
    parser.add_argument("--v1-kva", type=float, default=1000.0, help="V1: transformer rated power in kVA")
    # V1 inputs: impedance
    parser.add_argument("--v1-percent-z", type=float, default=5.0, help="V1: transformer impedance in percent")
    # V1 inputs: voltage
    parser.add_argument("--v1-volts", type=float, default=480.0, help="V1: line-to-line voltage in volts")
    # V1 inputs: base
    parser.add_argument("--v1-s-base", type=float, default=1e7, help="V1: power base in volt-amperes")
    # V1 expected
    parser.add_argument("--v1-expected", type=float, default=2.0, help="V1: expected per-unit admittance, worked by hand")
    # V4: the transformer that is an edge by itself
    parser.add_argument("--v4-direct", default="tr_117", help="V4: transformer whose admittance is halved")
    # V4: the edge that is that transformer alone
    parser.add_argument("--v4-direct-edge", default="egauge_17,egauge_19", help="V4: meter pair joined by that transformer alone")
    # V4: a transformer on a longer path
    parser.add_argument("--v4-path", default="tr_125", help="V4: second transformer whose admittance is halved")
    # V4: an edge whose path crosses it
    parser.add_argument("--v4-path-edge", default="egauge_3,egauge_9", help="V4: meter pair whose path crosses the second transformer")
    # V5: periods
    parser.add_argument("--v5-reference-period", type=int, default=9, help="V5: a period on the reference edge set")
    # V5: tie period
    parser.add_argument("--v5-tie-period", type=int, default=13, help="V5: the period with the bus tie closed")
    # Done
    return parser.parse_args()


class Tally:
    """Counts checks and prints one line per check."""

    def __init__(self):
        """Start with nothing counted."""
        # Passed, failed, not run
        self.passed, self.failed, self.skipped = 0, 0, 0

    def check(self, ok, text):
        """Record one check and print it."""
        # Count
        if ok:
            # Passed
            self.passed += 1
        # Otherwise
        else:
            # Failed
            self.failed += 1
        # Print
        print(f"  {'PASS' if ok else 'FAIL'}  {text}", flush=True)

    def skip(self, text):
        """Record a check that could not run."""
        # Count
        self.skipped += 1
        # Print
        print(f"  NOT RUN  {text}", flush=True)

    def info(self, text):
        """Print a reported figure that is not judged."""
        # Print
        print(f"  INFO  {text}", flush=True)


def close(a, b, tolerance):
    """Whether two numbers (real or complex) agree within the relative tolerance."""
    # Relative to the larger
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1e-300)


def refuses(call):
    """Whether calling `call` raises ValueError, which is how the module refuses. Any other outcome, a returned value included, is not a refusal."""
    # Try the call
    try:
        # Run it
        call()
    # The module's refusal
    except ValueError:
        # Refused
        return True
    # It returned
    return False


def eliminate_one_by_one(Y, keep):
    """
    Second implementation of the reduction, written without the module:
    remove one node at a time. Removing node k changes every remaining
    entry by Y_ij - Y_ik * Y_kj / Y_kk. Only nodes that reach a kept node
    are removed; a part of the network with no kept node is dropped, as it
    cannot couple two kept nodes. Returns the reduced matrix in the order
    of keep.
    """
    # Work on a copy
    Y = np.array(Y, dtype=complex)
    # Size
    n = Y.shape[0]
    # Nodes reached from a kept node through nonzero entries
    reached, frontier = set(keep), list(keep)
    # Search
    while frontier:
        # Next
        u = frontier.pop()
        # Its neighbors
        for v in range(n):
            # Linked in either direction, and new
            if v != u and (Y[u, v] != 0 or Y[v, u] != 0) and v not in reached:
                # Mark
                reached.add(v)
                # Expand
                frontier.append(v)
    # Nodes still in the matrix
    alive = sorted(reached)
    # Remove each node that is not kept
    for k in [i for i in alive if i not in set(keep)]:
        # The others
        rest = [i for i in alive if i != k]
        # Update the remaining entries
        Y[np.ix_(rest, rest)] -= np.outer(Y[rest, k], Y[k, rest]) / Y[k, k]
        # Drop the node
        alive = rest
    # Kept block
    return Y[np.ix_(list(keep), list(keep))]


def second_weights(net, graph, records, timeline, index, s_base):
    """
    A period's raw weights from the second implementation. Nothing of the
    module's electrical steps is used: the positive-sequence number of each
    3x3 part is taken by the direct formula (1/3) v^H B v with
    v = (1, a^2, a), the per-unit conversion and the assembly are written
    here from graph.json's blocks, and the reduction removes one node at a
    time. Only the node merging is the module's, which N1 and N2 test
    against graph.json and the timeline file. Returns a dictionary from a
    frozen meter pair to its weight, None for two meters on one node.
    """
    # The period
    period = timeline["periods"][index]
    # Meter names
    names = [n["meter"] for n in timeline["nodes"]]
    # Midpoint of the period
    when = pd.Timestamp(period["start"]) + (pd.Timestamp(period["end"]) - pd.Timestamp(period["start"])) / 2
    # Merged nodes, with every element of graph.json as an impedance element
    root, _ = ew.merged_nodes(net, records, when, {e["element"] for e in graph["edges"]})
    # Row of every node
    row = {r: i for i, r in enumerate(sorted(set(root.values())))}
    # Empty matrix
    Y = np.zeros((len(row), len(row)), dtype=complex)
    # The 120 degree rotation
    a = np.exp(2j * np.pi / 3)
    # The positive-sequence set of three phase voltages
    v = np.array([1, a * a, a])
    # Line-to-line voltage of every bus
    volts = {b["name"]: np.sqrt(3.0) * b["nominal_voltage"] for b in net["Bus"] if b.get("nominal_voltage")}
    # The elements of the network file by name
    items = {x["name"]: x for kind in ("Line", "Transformer") for x in net.get(kind, [])}
    # Assemble each element's block
    for e in graph["edges"]:
        # Its complex 6x6 block
        B = np.array(e["Y_real"]) + 1j * np.array(e["Y_imag"])
        # Its two buses
        ends = (items[e["element"]]["fbus"], items[e["element"]]["tbus"][0]["name"])
        # Each of the four parts
        for p in range(2):
            # Second index
            for q in range(2):
                # Positive-sequence number of the part, per unit on the voltages of its two ends
                value = (v.conj() @ B[3 * p:3 * p + 3, 3 * q:3 * q + 3] @ v) / 3.0 * volts[ends[p]] * volts[ends[q]] / s_base
                # Add at the rows of the two ends
                Y[row[root[ends[p]]], row[root[ends[q]]]] += value
    # Each meter's row
    at = {m: row[root[timeline["meters"][m]["voltage_bus"]]] for m in names}
    # Kept rows
    keep = sorted(set(at.values()))
    # Reduce
    R = eliminate_one_by_one(Y, keep)
    # Weights of the period's edges
    out = {}
    # Each edge
    for e in timeline["graphs"][int(period["graph"])]["edges"]:
        # Its meters
        m1, m2 = names[e["source"]], names[e["target"]]
        # Same node: the marker; otherwise the magnitude of the reduced entry
        out[frozenset((m1, m2))] = None if at[m1] == at[m2] else float(abs(R[keep.index(at[m1]), keep.index(at[m2])]))
    # Done
    return out


def delta_wye_block(y, turns):
    """
    The 6x6 phase-domain block of a delta-wye transformer, from first
    principles: three single-phase units, each with series admittance y
    seen from its wye-side winding and a turns ratio `turns` (delta
    winding to wye winding). Each unit alone is
    [[y / turns^2, -y / turns], [-y / turns, y]]. The delta windings sit
    between phases a-b, b-c and c-a, the wye windings between each phase
    and ground. Rows and columns: delta side a b c, then wye side a b c.
    """
    # Winding voltages of the delta side from its phase voltages: a-b, b-c, c-a
    A = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]], dtype=complex)
    # The four parts
    return np.block([[A.T @ A * (y / turns ** 2), -A.T * (y / turns)], [-A * (y / turns), y * np.eye(3)]])


def case_steps(t, args):
    """V1, V6, T1 and T2: the per-unit and sequence steps in isolation, against numbers worked by hand."""
    # Heading
    print("\nSteps in isolation, against numbers worked by hand")
    # V1 by the nameplate formula
    y_name = abs(ew.transformer_per_unit(args.v1_kva, args.v1_percent_z, args.v1_s_base))
    # V1 through ohms: impedance in ohms on the transformer's own rating
    z_ohm = (args.v1_percent_z / 100.0) * args.v1_volts ** 2 / (args.v1_kva * 1e3)
    # Base impedance of the level
    z_base = args.v1_volts ** 2 / args.v1_s_base
    # Per-unit admittance through ohms
    y_ohm = z_base / z_ohm
    # Judge both
    t.check(close(y_name, args.v1_expected, args.tolerance) and close(y_ohm, args.v1_expected, args.tolerance), f"V1 transformer {args.v1_kva:.0f} kVA, {args.v1_percent_z}%, {args.v1_volts:.0f} V on {args.v1_s_base / 1e6:.0f} MVA: nameplate {y_name:.12f}, through ohms {y_ohm:.12f} (Z {z_ohm:.6f} ohm, Z_base {z_base:.6f} ohm), expected {args.v1_expected}")
    # A transformer with resistance keeps the same magnitude
    t.check(close(abs(ew.transformer_per_unit(args.v1_kva, args.v1_percent_z, args.v1_s_base, args.variant_x_over_r)), args.v1_expected, args.tolerance), f"V1 the same transformer with X/R {args.variant_x_over_r}: the magnitude stays {args.v1_expected}")
    # A nameplate that is not positive is refused
    t.check(refuses(lambda: ew.transformer_per_unit(0.0, 5.0, 1e6)) and refuses(lambda: ew.transformer_per_unit(1000.0, -5.0, 1e6)) and refuses(lambda: ew.transformer_per_unit(1000.0, 5.0, 1e6, 0.0)), "V1 a rated power of 0, a negative impedance and an X/R of 0 are refused")
    # V6: a balanced block, self term 5 - 2j and mutual term 1 + 0.5j
    s, m = 5 - 2j, 1 + 0.5j
    # The block
    block = np.array([[s, m, m], [m, s, m], [m, m, s]])
    # Reduce
    value, coupling, status = ew.positive_sequence(block)
    # Judge: self minus mutual, no coupling
    t.check(abs(value - (s - m)) < args.tolerance and coupling < args.tolerance and status == "ok", f"V6 balanced block: positive sequence {value:.6f}, expected self minus mutual {s - m:.6f}; coupling {coupling:.1e}")
    # An unbalanced block: one phase twice as strong
    value, coupling, status = ew.positive_sequence(np.diag([2.0, 1.0, 1.0]))
    # Judge: flagged, not accepted
    t.check(status == "coupled", f"V6 unbalanced block (one phase twice the others): flagged '{status}', coupling {coupling:.3f}")
    # An all-zero block
    value, coupling, status = ew.positive_sequence(np.zeros((3, 3)))
    # Judge: flagged
    t.check(status == "near zero", f"V6 empty block: flagged '{status}'")
    # T1: a line with series admittance 0.5 - 1.5j S and a shunt of 0.002j S, half at each end, at 2,400 V on 1 MVA
    ys, ysh = 0.5 - 1.5j, 0.002j
    # Its 6x6 block: each end sees the series part plus half the shunt
    Y6 = np.block([[(ys + ysh / 2) * np.eye(3), -ys * np.eye(3)], [-ys * np.eye(3), (ys + ysh / 2) * np.eye(3)]])
    # Per unit
    rec = ew.terminal_per_unit(Y6, 2400.0 / np.sqrt(3.0), 2400.0 / np.sqrt(3.0), 1e6)
    # By hand: siemens times V^2 / S = 2400^2 / 1e6 = 5.76
    want = 5.76 * np.array([[ys + ysh / 2, -ys], [-ys, ys + ysh / 2]])
    # Judge the four numbers, and that their sum is the shunt
    t.check(np.abs(rec["y"] - want).max() < args.tolerance * np.abs(want).max() and close(rec["y"].sum(), 5.76 * ysh, 1e-6) and rec["status"] == "ok", f"T1 line with a shunt, 2,400 V on 1 MVA: y11 {rec['y'][0, 0]:.6f}, y12 {rec['y'][0, 1]:.6f}, expected {want[0, 0]:.6f} and {want[0, 1]:.6f}; the four numbers add up to the shunt {rec['y'].sum():.6f}, expected {5.76 * ysh:.6f}")
    # T2: a delta-wye transformer, 2,400 V delta to 480 V wye, series admittance 1 - 6j S seen from a wye winding
    y = 1 - 6j
    # Turns ratio of one unit: the delta winding carries 2,400 V, the wye winding 480 / sqrt(3) V
    turns = 2400.0 / (480.0 / np.sqrt(3.0))
    # Its block from first principles
    Y6 = delta_wye_block(y, turns)
    # Per unit on the rated voltages
    rec = ew.terminal_per_unit(Y6, 2400.0 / np.sqrt(3.0), 480.0 / np.sqrt(3.0), 1e6)
    # By hand: y per unit = y * 480^2 / 1e6 = 0.2304 y at both ends; the end-to-end numbers are minus that, turned by 30 degrees one way and the other
    ypu = y * 0.2304
    # The rotation by 30 degrees
    turn = np.exp(1j * np.pi / 6)
    # The four numbers by hand
    want = np.array([[ypu, -ypu / turn], [-ypu * turn, ypu]])
    # What the module reads from the four numbers
    summary = ew.terminal_summary(rec["y"])
    # Judge the complex numbers and the summary
    t.check(np.abs(rec["y"] - want).max() < args.tolerance * abs(ypu) and close(summary["ratio"], 1.0, args.tolerance) and close(abs(summary["shift_deg"]), 30.0, args.tolerance) and close(summary["x_over_r"], 6.0, args.tolerance) and summary["residual"] < args.tolerance and rec["status"] == "ok",
            f"T2 delta-wye transformer, complex: y11 {rec['y'][0, 0]:.6f}, y12 {rec['y'][0, 1]:.6f}, y21 {rec['y'][1, 0]:.6f}, y22 {rec['y'][1, 1]:.6f}; expected {want[0, 0]:.6f}, {want[0, 1]:.6f}, {want[1, 0]:.6f}, {want[1, 1]:.6f}; ratio {summary['ratio']:.6f}, shift {summary['shift_deg']:.3f} degrees, X/R {summary['x_over_r']:.3f}, ideal test {summary['residual']:.1e}")
    # The same transformer on a bus whose nominal voltage is 2,300 V: an off-nominal ratio of 2400 / 2300
    rec = ew.terminal_per_unit(Y6, 2300.0 / np.sqrt(3.0), 480.0 / np.sqrt(3.0), 1e6)
    # By hand: the delta-side number shrinks by (2300 / 2400)^2, the end-to-end numbers by 2300 / 2400, the wye-side number stays
    k = 2300.0 / 2400.0
    # The four numbers by hand
    want = np.array([[ypu * k * k, -ypu * k / turn], [-ypu * k * turn, ypu]])
    # Judge, with the ratio the module reads back
    t.check(np.abs(rec["y"] - want).max() < args.tolerance * abs(ypu) and close(ew.terminal_summary(rec["y"])["ratio"], 2400.0 / 2300.0, args.tolerance), f"T2 the same transformer on a 2,300 V bus: y11 {rec['y'][0, 0]:.6f}, expected {want[0, 0]:.6f}; ratio read back {ew.terminal_summary(rec['y'])['ratio']:.6f}, expected {2400.0 / 2300.0:.6f}")


def case_reduction(t, args):
    """V2, V3 and K1 to K6: the reduction, against numbers worked by hand, and its refusals."""
    # Heading
    print("\nReduction, against numbers worked by hand, and its refusals")
    # V2: nodes 0 - 1 - 2 with branches 4 and 4, node 1 removed
    Y = np.array([[4, -4, 0], [-4, 8, -4], [0, -4, 4]], dtype=complex)
    # Reduce onto the two ends
    R, _ = ew.kron_reduce(Y, [0, 2])
    # Judge: 4 * 4 / (4 + 4) = 2
    t.check(close(abs(R[0, 1]), 2.0, args.tolerance), f"V2 two branches in a row, 4 and 4, middle removed: {abs(R[0, 1]):.12f}, expected 2")
    # V3: a star, center node 3, branches 2, 3 and 6 to nodes 0, 1 and 2
    Y = np.zeros((4, 4), dtype=complex)
    # Assemble
    for i, v in enumerate([2.0, 3.0, 6.0]):
        # Branch i to the center
        Y[np.ix_([i, 3], [i, 3])] += v * np.array([[1, -1], [-1, 1]])
    # Reduce onto the three ends
    R, _ = ew.kron_reduce(Y, [0, 1, 2])
    # Expected: product over the sum of all three, worked by hand
    want = {(0, 1): 6.0 / 11.0, (0, 2): 12.0 / 11.0, (1, 2): 18.0 / 11.0}
    # Series values of each pair alone, which are wrong for a star
    wrong = {(0, 1): 6.0 / 5.0, (0, 2): 3.0 / 2.0, (1, 2): 2.0}
    # Judge the right values
    t.check(all(close(abs(R[i, j]), w, args.tolerance) for (i, j), w in want.items()), "V3 star 2, 3, 6 with the center removed: " + ", ".join(f"{abs(R[i, j]):.6f}" for (i, j) in want) + " expected 6/11, 12/11, 18/11 = " + ", ".join(f"{w:.6f}" for w in want.values()))
    # Judge that the series values do not come out
    t.check(all(not close(abs(R[i, j]), w, 1e-3) for (i, j), w in wrong.items()), "V3 the pair-alone series values 6/5, 3/2, 2 do not come out")
    # K1: a mesh of three nodes, branches 0-1 = 1, 0-2 = 2, 1-2 = 3, and a shunt of 1 at node 2, which is removed
    def mesh(shunt):
        """The nodal matrix of the mesh with the given shunt at node 2."""
        # Branch values off the diagonal, their sums on it, the shunt added at node 2
        return np.array([[3, -1, -2], [-1, 4, -3], [-2, -3, 5 + shunt]], dtype=complex)
    # With the shunt
    R, _ = ew.kron_reduce(mesh(1.0), [0, 1])
    # Without it
    R0, _ = ew.kron_reduce(mesh(0.0), [0, 1])
    # By hand: -1 - (2 * 3) / (2 + 3 + 1) = -2 with the shunt, -1 - 6 / 5 = -2.2 without
    t.check(close(R[0, 1], -2.0, args.tolerance) and close(R0[0, 1], -2.2, args.tolerance), f"K1 mesh with a shunt at the removed node: {R[0, 1].real:.12f}, expected -2 by hand; without the shunt {R0[0, 1].real:.12f}, expected -2.2")
    # The diagonal by hand too: 3 - 4 / 6 and 4 - 9 / 6
    t.check(close(R[0, 0], 3 - 4.0 / 6.0, args.tolerance) and close(R[1, 1], 4 - 9.0 / 6.0, args.tolerance), f"K1 its diagonal entries: {R[0, 0].real:.6f} and {R[1, 1].real:.6f}, expected {3 - 4.0 / 6.0:.6f} and {4 - 9.0 / 6.0:.6f}")
    # K2: nothing to remove
    R, info = ew.kron_reduce(mesh(1.0), [0, 1, 2])
    # Judge: the same matrix, nothing eliminated
    t.check(np.array_equal(R, mesh(1.0)) and info["eliminated"] == 0 and info["parts"] == 0, f"K2 every node kept: the matrix comes back unchanged, {info['eliminated']} rows eliminated")
    # K3: the mesh plus two more nodes joined only to each other, with no kept node among them
    Y = np.zeros((5, 5), dtype=complex)
    # The mesh
    Y[:3, :3] = mesh(1.0)
    # The separate pair, a branch of 7 with no shunt, which alone is a singular block
    Y[3:, 3:] = 7 * np.array([[1, -1], [-1, 1]])
    # Reduce
    R, info = ew.kron_reduce(Y, [0, 1])
    # Judge: the same result as without them, two rows left out, no refusal
    t.check(close(R[0, 1], -2.0, args.tolerance) and info["left_out"] == 2 and info["eliminated"] == 1, f"K3 a part of the network with no kept node: {info['left_out']} rows left out, {info['eliminated']} eliminated, result {R[0, 1].real:.12f} unchanged")
    # K4: kept node 0, an element with a 30 degree shift to node 1, a plain branch from node 1 to kept node 2; both of size 4 as reactances
    y = -4j
    # The rotation
    turn = np.exp(1j * np.pi / 6)
    # Assemble: the shifted element between 0 and 1, the plain branch between 1 and 2
    Y = np.array([[y, -y / turn, 0], [-y * turn, y + y, -y], [0, -y, y]], dtype=complex)
    # Reduce onto the two ends
    R, _ = ew.kron_reduce(Y, [0, 2])
    # By hand: magnitude 4 * 4 / (4 + 4) = 2 in both directions, with the angle turned by 30 degrees
    t.check(close(abs(R[0, 1]), 2.0, args.tolerance) and close(abs(R[1, 0]), 2.0, args.tolerance) and close(abs(np.degrees(np.angle(R[0, 1] / R[1, 0]))), 60.0, 1e-6), f"K4 a 30 degree phase shift on the path: magnitudes {abs(R[0, 1]):.12f} and {abs(R[1, 0]):.12f}, expected 2 and 2; the two directions differ in angle by {abs(np.degrees(np.angle(R[0, 1] / R[1, 0]))):.3f} degrees, expected 60")
    # K5: a random connected complex network of 9 nodes with a shunt at every node
    rng = np.random.default_rng(0)
    # Random complex branch values on every pair
    W = np.triu(rng.uniform(0.5, 5.0, (9, 9)) - 1j * rng.uniform(0.5, 5.0, (9, 9)), 1)
    # Symmetric
    W = W + W.T
    # Nodal matrix: minus the branch values off the diagonal, their sum plus a shunt on it
    Y = np.diag(W.sum(axis=1) + 1j * rng.uniform(0.01, 0.1, 9)) - W
    # One step onto nodes 0, 1, 2
    one, _ = ew.kron_reduce(Y, [0, 1, 2])
    # Two steps: first onto 0 to 5
    half, _ = ew.kron_reduce(Y, [0, 1, 2, 3, 4, 5])
    # Then onto 0, 1, 2
    two, _ = ew.kron_reduce(half, [0, 1, 2])
    # Judge
    t.check(np.abs(one - two).max() < args.tolerance * np.abs(one).max(), f"K5 reduction in two steps equals one step, complex network with shunts: largest difference {np.abs(one - two).max():.1e}")
    # The second implementation agrees on the same network
    t.check(np.abs(eliminate_one_by_one(Y, [0, 1, 2]) - one).max() < args.tolerance * np.abs(one).max(), f"K5 removing one node at a time gives the same reduced matrix: largest difference {np.abs(eliminate_one_by_one(Y, [0, 1, 2]) - one).max():.1e}")
    # K6: a badly conditioned part. Kept node 0, removed nodes 1 and 2, linked, with diagonal entries 1 and 1 / bad
    Y = np.array([[2, -1, -1], [-1, 1, 0], [-1, 0, 1.0 / args.bad_condition]], dtype=complex)
    # Link the two removed nodes weakly so they are one connected part
    Y[1, 2] = Y[2, 1] = -1e-3 / args.bad_condition
    # Its condition number, measured here
    measured = np.linalg.cond(Y[1:, 1:])
    # Judge: refused at the limit under test, accepted when the limit is raised above it
    t.check(measured > args.condition_limit and refuses(lambda: ew.kron_reduce(Y, [0], args.condition_limit)) and not refuses(lambda: ew.kron_reduce(Y, [0], measured * 10)), f"K6 a connected part with condition number {measured:.1e} is refused at the limit {args.condition_limit:.0e}, and accepted only when the limit is raised above it")
    # A singular part: two removed nodes joined to each other and to the kept node by entries that cancel exactly
    Y = np.array([[1, -1, 0], [-1, 1, -1], [0, -1, 1]], dtype=complex)
    # Make the removed block [[1, -1], [-1, 1]], which is exactly singular
    t.check(refuses(lambda: ew.kron_reduce(Y, [0])), "K6 an exactly singular connected part is refused")
    # A value that is not finite
    Y = mesh(1.0)
    # Put a not-a-number in it
    Y[2, 2] = np.nan
    # Judge
    t.check(refuses(lambda: ew.kron_reduce(Y, [0, 1])), "K6 a matrix holding a not-a-number is refused")
    # A residual limit no solve can meet
    t.check(refuses(lambda: ew.kron_reduce(np.diag(W.sum(axis=1) + 1j * rng.uniform(0.01, 0.1, 9)) - W, [0, 1, 2], args.condition_limit, 1e-300)), "K6 a solve that does not meet its residual limit is refused (limit set to 1e-300)")
    # Limits that are not positive
    t.check(refuses(lambda: ew.kron_reduce(mesh(1.0), [0, 1], 0.0)) and refuses(lambda: ew.kron_reduce(mesh(1.0), [0, 1], args.condition_limit, -1.0)), "K6 a condition limit of 0 and a negative residual limit are refused")


def case_scaling(t, args):
    """S1 to S5: weights, scaling and edge-set splitting, with their refusals."""
    # Heading
    print("\nWeights, scaling and edge-set splitting")
    # S1: the largest weight becomes 1, a same-node pair becomes 1
    scaled, top = ew.scale_weights([2.0, None, 8.0, 4.0])
    # Judge
    t.check(scaled == [0.25, 1.0, 1.0, 0.5] and top == 8.0, f"S1 scaling 2, same node, 8, 4: {scaled}, scale {top}")
    # S2: an empty set
    empty = ew.scale_weights([])
    # Same-node pairs only
    only = ew.scale_weights([None, None])
    # Judge
    t.check(empty == ([], None) and only == ([1.0, 1.0], None), f"S2 an empty set gives {empty}; a set of same-node pairs only gives {only}")
    # S3: values that must never become a weight
    bad = {"zero": 0.0, "a negative number": -1.0, "not-a-number": float("nan"), "plus infinity": float("inf"), "minus infinity": float("-inf"), "text": "2", "a truth value": True}
    # Each must be refused, alone and beside a good weight
    results = {name: refuses(lambda v=v: ew.scale_weights([v])) and refuses(lambda v=v: ew.scale_weights([v, 2.0])) for name, v in bad.items()}
    # Judge
    t.check(all(results.values()), "S3 refused as a weight: " + ", ".join(f"{name} ({'refused' if ok else 'ACCEPTED'})" for name, ok in results.items()))
    # S4: no pairs
    weights, notes = ew.pair_weights(np.zeros((2, 2), dtype=complex), {}, [])
    # Judge, with the coefficient refusals
    t.check(weights == [] and notes == [] and refuses(lambda: ew.gcn_coefficients(2, [(0, 1)], [None])) and refuses(lambda: ew.gcn_coefficients(2, [(0, 1)], [float("nan")])) and refuses(lambda: ew.gcn_coefficients(2, [(0, 1)], [-0.5])), "S4 no pairs give no weights; GCN coefficients refuse a marker, a not-a-number and a negative weight")
    # A same-node pair gets the marker, not a number
    weights, _ = ew.pair_weights(np.array([[1, -1], [-1, 1]], dtype=complex), {"a": 0, "b": 0, "c": 1}, [("a", "b"), ("a", "c")])
    # Judge
    t.check(weights[0] is None and weights[1] == 1.0, f"S4 two meters on one node get the marker None, never a number: {weights}")
    # S5: four periods on two source edge sets. Periods 0, 1 and 3 use set 0; period 3 has different weights. Period 2 uses set 1.
    sets, where = ew.weighted_edge_sets([0, 0, 1, 0], [[1.0, 2.0], [1.0, 2.0 * (1 + 1e-12)], [5.0], [1.0, 3.0]], 2, args.tolerance)
    # Judge: set 0 and set 1 keep their numbers, the different period gets a new set 2, no period moves
    t.check([(s["source"], s["variant"], s["periods"]) for s in sets] == [(0, 0, [0, 1]), (1, 0, [2]), (0, 1, [3])] and where == [0, 0, 1, 2] and sets[2]["raw"] == [1.0, 3.0], f"S5 an edge set whose periods carry different weights is split: sets (source, variant, periods) {[(s['source'], s['variant'], s['periods']) for s in sets]}, periods now on sets {where}")
    # A same-node marker in one period and a number in the other also splits
    sets, where = ew.weighted_edge_sets([0, 0], [[None, 2.0], [1.0, 2.0]], 1, args.tolerance)
    # Judge
    t.check(len(sets) == 2 and where == [0, 1], f"S5 a pair that shares a node in one period only also splits the edge set: periods on sets {where}")
    # An edge set no period uses is marked, not weighted
    sets, where = ew.weighted_edge_sets([0], [[1.0]], 2, args.tolerance)
    # Judge
    t.check(sets[1]["raw"] is None and sets[1]["periods"] == [], "S5 an edge set no period uses gets no weights (the builder then stops)")


def case_network(t, args, data):
    """N1 to N7, V4 and V5: the real network."""
    # Heading
    print("\nThe real network")
    # Shorthand
    net, graph, timeline, records, branches = data["net"], data["graph"], data["timeline"], data["records"], data["branches"]
    # N1: rebuild at the reference time
    root, unresolved = ew.merged_nodes(net, records, pd.Timestamp(graph["reference_time"]), set(branches))
    # Bus sets of the rebuilt nodes
    mine = {}
    # Collect
    for bus, r in root.items():
        # Under its representative
        mine.setdefault(r, set()).add(bus)
    # Judge: the same sets of buses as graph.json's nodes
    t.check(not unresolved and {frozenset(v) for v in mine.values()} == {frozenset(n["buses"]) for n in graph["nodes"]}, f"N1 network rebuilt at {graph['reference_time']}: {len(mine)} nodes, graph.json has {len(graph['nodes'])}, same bus sets")
    # Nothing unresolved among the elements
    t.check(not data["notes"]["unresolved"] and len(branches) == len(graph["edges"]), f"N1 every element with a block is per unit from its complete block: {len(branches)} of {len(graph['edges'])}")
    # Transformers
    trs = data["notes"]["transformers"]
    # Complex comparison: the block's series admittance on the rated voltage against the nameplate value at the block's own X/R
    worst = max(abs(x["block_on_rated"] - ew.transformer_per_unit(x["kva_used"], x["percent_z"], args.s_base, x["x_over_r"])) / x["nameplate_abs"] for x in trs)
    # Judge, with the ideal-transformer test and the 30 degree shift
    t.check(worst < args.tolerance and all(x["residual"] < args.tolerance for x in trs) and all(close(abs(x["shift_deg"]), 30.0, args.tolerance) for x in trs),
            f"N1 all {len(trs)} transformer blocks: complex series admittance equals the nameplate value at the block's X/R (largest relative difference {worst:.1e}), ideal-transformer test at most {max(x['residual'] for x in trs):.1e}, phase shift 30 degrees, X/R {min(x['x_over_r'] for x in trs):.2f} to {max(x['x_over_r'] for x in trs):.2f}")
    # N2: each period
    periods = [ew.period_reduction(net, records, timeline, i, branches, args.tolerance, args.condition_limit) for i in range(len(timeline["periods"]))]
    # Problems
    problems = [x for p in periods for x in p["problems"]]
    # Judge
    t.check(not problems, f"N2 all {len(periods)} periods: every meter's node holds the buses the timeline records, the coupled pairs are exactly the edges, the two directions of every weight agree" + ("" if not problems else "; " + "; ".join(problems[:5])))
    # Without weights the remaining cases cannot run
    if problems:
        # Count them as not run
        t.skip("N3 to N7, V4, V5: not run, since a period did not pass")
        # Stop here
        return
    # Conditioning and solve of every period
    t.check(all(p["info"]["condition"] < args.condition_limit and p["info"]["residual"] < args.tolerance for p in periods), f"N2 conditioning and solve: largest condition number of a connected part {max(p['info']['condition'] for p in periods):.1f} (limit {args.condition_limit:.0e}), largest relative residual {max(p['info']['residual'] for p in periods):.1e}")
    # N3: the second implementation, every period
    worst, count = 0.0, 0
    # Each period
    for i, p in enumerate(periods):
        # Its second weights
        second = second_weights(net, graph, records, timeline, i, args.s_base)
        # Each edge
        for pair, w in zip(p["pairs"], p["raw"]):
            # The other value
            v = second[frozenset(pair)]
            # Count
            count += 1
            # Markers must agree; numbers by their relative difference
            worst = max(worst, 0.0 if (w is None and v is None) else (np.inf if (w is None) != (v is None) else abs(w - v) / max(abs(w), abs(v))))
    # Judge
    t.check(worst < args.tolerance, f"N3 second implementation (own sequence formula, per unit, assembly and one-node-at-a-time elimination): {count} edge weights over {len(periods)} periods, largest relative difference {worst:.1e}")
    # N4: periods that share an edge set
    sets, where = ew.weighted_edge_sets([int(p["graph"]) for p in timeline["periods"]], [p["raw"] for p in periods], len(timeline["graphs"]), args.tolerance)
    # Edge sets used by more than one period
    shared = sum(1 for g in range(len(timeline["graphs"])) if sum(1 for p in timeline["periods"] if int(p["graph"]) == g) > 1)
    # Judge: no split, numbering unchanged
    t.check(len(sets) == len(timeline["graphs"]) and where == [int(p["graph"]) for p in timeline["periods"]], f"N4 periods that share an edge set carry the same weights: {shared} edge sets used by more than one period, {len(sets) - len(timeline['graphs'])} splits")
    # N5: another power base
    other, _ = ew.branches_per_unit(net, graph, args.other_s_base)
    # Reference period under the first base
    a = periods[args.v5_reference_period]
    # Under the other base
    b = ew.period_reduction(net, records, timeline, args.v5_reference_period, other, args.tolerance, args.condition_limit)
    # Scaled weights
    sa, sb = ew.scale_weights(a["raw"])[0], ew.scale_weights(b["raw"])[0]
    # Ratio of raw weights
    ratios = [x / y for x, y in zip(a["raw"], b["raw"]) if x is not None]
    # Judge
    t.check(max(abs(x - y) for x, y in zip(sa, sb)) < args.tolerance and all(close(r, args.other_s_base / args.s_base, args.tolerance) for r in ratios), f"N5 power base {args.s_base:.0f} against {args.other_s_base:.0f} VA: raw weights differ by the factor {ratios[0]:.6f}, scaled weights by at most {max(abs(x - y) for x, y in zip(sa, sb)):.1e}")
    # N6: GCN coefficients against the normalized matrix built directly
    names = [n["meter"] for n in timeline["nodes"]]
    # Node pairs of the reference period's edges
    pairs = [(names.index(x), names.index(y)) for x, y in a["pairs"]]
    # From the module
    d, self_c, nbr = ew.gcn_coefficients(len(names), pairs, sa)
    # Directly: A with the weights, plus the identity
    A = np.eye(len(names))
    # Fill
    for (i, j), w in zip(pairs, sa):
        # Both directions
        A[i, j] = A[j, i] = w
    # D^-1/2 A D^-1/2
    N = A / np.sqrt(np.outer(A.sum(axis=1), A.sum(axis=1)))
    # Judge
    t.check(np.abs(np.diag(N) - self_c).max() < args.tolerance and max(abs(N[i, j] - c) for (i, j), c in zip(pairs, nbr)) < args.tolerance, f"N6 GCN coefficients equal the normalized matrix built directly: {len(names)} self-loop coefficients, {len(pairs)} neighbor coefficients")
    # N7: the series-only approximation, pure reactance and with resistance, every period
    for label, ratio in (("pure reactance", None), (f"X/R {args.variant_x_over_r}", args.variant_x_over_r)):
        # Its branches
        approx, _ = ew.branches_per_unit(net, graph, args.s_base, 0, 0.01, "series", ratio)
        # Largest relative difference of raw and scaled weights over all periods
        raw_d, scaled_d, same = 0.0, 0.0, True
        # Each period
        for i, p in enumerate(periods):
            # The approximation's reduction
            q = ew.period_reduction(net, records, timeline, i, approx, args.tolerance, args.condition_limit)
            # Same coupled pairs
            same = same and not q["problems"]
            # Scaled weights of both
            sp, sq = ew.scale_weights(p["raw"])[0], ew.scale_weights(q["raw"])[0]
            # Raw difference
            raw_d = max(raw_d, max(abs(y - x) / x for x, y in zip(p["raw"], q["raw"]) if x is not None))
            # Scaled difference
            scaled_d = max(scaled_d, max(abs(y - x) / x for x, y, r in zip(sp, sq, p["raw"]) if r is not None))
        # Report only
        t.info(f"N7 series-only approximation (transformers from the nameplate, {label}; no shunt, ratio or phase shift) against the complete blocks, all {len(periods)} periods: raw weights differ by at most {100 * raw_d:.2f}%, scaled weights by at most {100 * scaled_d:.2f}%; same coupled pairs: {same}")
    # V4: weights by name in the reference period
    base = {frozenset(p): w for p, w in zip(a["pairs"], a["raw"])}
    # The two transformers to weaken, each with its edge
    for label, element, edge in (("V4 the edge that is one transformer alone", args.v4_direct, args.v4_direct_edge), ("V4 an edge whose path crosses the transformer", args.v4_path, args.v4_path_edge)):
        # The edge
        pair = frozenset(edge.split(","))
        # Copy the branches
        weak = copy.deepcopy(branches)
        # Halve all four numbers of the element
        weak[element]["y"] = weak[element]["y"] / 2.0
        # Reduce
        w = ew.period_reduction(net, records, timeline, args.v5_reference_period, weak, args.tolerance, args.condition_limit)
        # By name
        after = {frozenset(p): x for p, x in zip(w["pairs"], w["raw"])}
        # The path of that edge in the file
        path = next(e["path_elements"] for e in timeline["graphs"][int(timeline["periods"][args.v5_reference_period]["graph"])]["edges"] if frozenset((names[e["source"]], names[e["target"]])) == pair)
        # Direct edge: exactly half. Longer path: lower, and by less than half.
        ok = close(after[pair], base[pair] / 2.0, args.tolerance) if path == [element] else (base[pair] / 2.0 < after[pair] < base[pair])
        # Judge, with the pattern unchanged
        t.check(ok and element in path and not w["problems"], f"{label}: {element} halved, {edge.replace(',', ' - ')} (path {' '.join(path)}) goes from {base[pair]:.4f} to {after[pair]:.4f}; coupled pairs unchanged")
    # V5: the tie period, by name
    tied = {frozenset(p): w for p, w in zip(periods[args.v5_tie_period]["pairs"], periods[args.v5_tie_period]["raw"])}
    # Added edges
    added = [p for p in tied if p not in base]
    # Added edges with a number
    numbers = [tied[p] for p in added if tied[p] is not None]
    # Largest change on the common edges
    moved = max((abs(tied[p] - base[p]) / base[p] for p in base if p in tied and base[p] is not None and tied[p] is not None), default=0.0)
    # Judge
    t.check(set(base) <= set(tied) and len(added) > 0 and all(w > 0 for w in numbers), f"V5 period {args.v5_tie_period} ({'; '.join(c['switch'] + ' ' + c['from'] + ' to ' + c['to'] for c in timeline['periods'][args.v5_tie_period]['changes'])}) against period {args.v5_reference_period}: all {len(base)} reference edges present, {len(added)} edges added, "
            f"{len(added) - len(numbers)} of them between meters on one node, the others with raw weights {min(numbers):.3f} to {max(numbers):.3f}")
    # Report how far the common edges move
    t.info(f"V5 weights of the common edges move by at most {100 * moved:.2e}% between the two periods")
    # Keep the periods for the file checks
    data["periods"] = periods


def case_refusals(t, args):
    """R1 to R5: the builder never writes over a file. It is run as a separate process with its outputs in a temporary folder."""
    # Heading
    print("\nThe builder's refusals (separate process, outputs in a temporary folder)")
    # The builder must be there
    if not os.path.exists(args.builder):
        # Count as not run
        t.skip(f"R1 to R5: {args.builder} is not present")
        # Done
        return
    # The inputs by absolute path
    graph, timeline, network, topology = (os.path.abspath(p) for p in (args.graph, args.timeline, args.network, args.topology_dir))

    def run(extra):
        """Run the builder with the real inputs and the extra options. Returns (exit code, combined output)."""
        # The command
        command = [sys.executable, os.path.abspath(args.builder), "--graph", graph, "--timeline", timeline, "--network", network, "--topology-dir", topology] + extra
        # Run it and collect what it prints
        done = subprocess.run(command, capture_output=True, text=True)
        # Exit code and text
        return done.returncode, done.stdout + done.stderr

    def digest(path):
        """SHA-256 of a file's bytes."""
        # Hash the bytes
        with open(path, "rb") as f:
            # Digest
            return hashlib.sha256(f.read()).hexdigest()

    # A temporary folder that is removed afterwards
    with tempfile.TemporaryDirectory() as tmp:
        # A file that stands for an earlier result
        old = os.path.join(tmp, "earlier.json")
        # Its content
        with open(old, "w") as f:
            # Write
            f.write("an earlier result that must survive\n")
        # Its digest
        before = digest(old)
        # Free paths
        free_out, free_report = os.path.join(tmp, "new.json"), os.path.join(tmp, "new.csv")
        # R1: the graph file exists
        code, text = run(["--out", old, "--report", free_report])
        # Judge: refused, the file unchanged, the other output not created
        t.check(code != 0 and "refused" in text and digest(old) == before and not os.path.exists(free_report), f"R1 an existing graph file is not written over: exit code {code}, file unchanged byte for byte, no report created")
        # R1: the report file exists
        code, text = run(["--out", free_out, "--report", old])
        # Judge
        t.check(code != 0 and "refused" in text and digest(old) == before and not os.path.exists(free_out), f"R1 an existing report file is not written over: exit code {code}, file unchanged byte for byte, no graph file created")
        # R2: one path for both outputs
        code, text = run(["--out", free_out, "--report", free_out])
        # Judge
        t.check(code != 0 and "same file" in text and not os.path.exists(free_out), f"R2 one path for both outputs is refused: exit code {code}, nothing created")
        # R2: the same file through another spelling of the path
        code, text = run(["--out", free_out, "--report", os.path.join(tmp, ".", "new.json")])
        # Judge
        t.check(code != 0 and "same file" in text and not os.path.exists(free_out), f"R2 the same output file spelled two ways is refused: exit code {code}")
        # R3: a copy of the timeline stands for an input, so the test cannot touch the real one
        stand_in = os.path.join(tmp, "input_timeline.json")
        # Copy
        shutil.copyfile(timeline, stand_in)
        # Its digest
        input_before = digest(stand_in)
        # The aliases to try: another spelling, a symbolic link, a hard link
        aliases = {"another spelling of its path": os.path.join(tmp, "sub", "..", "input_timeline.json"), "a symbolic link": os.path.join(tmp, "link_sym.json"), "a hard link": os.path.join(tmp, "link_hard.json")}
        # A folder for the spelling with ".."
        os.mkdir(os.path.join(tmp, "sub"))
        # Each alias
        for label, alias in aliases.items():
            # Create the link; a file system that cannot link is reported as not run
            try:
                # Symbolic link
                if label == "a symbolic link":
                    # Create
                    os.symlink(stand_in, alias)
                # Hard link
                if label == "a hard link":
                    # Create
                    os.link(stand_in, alias)
            # No links here
            except OSError as err:
                # Not run
                t.skip(f"R3 an output that is an input through {label}: the file system cannot create it ({err})")
                # Next
                continue
            # Run with the stand-in as the timeline input and the alias as the output
            code, text = run(["--timeline", stand_in, "--out", alias, "--report", free_report])
            # Judge: refused as an input, the input unchanged
            t.check(code != 0 and "is the input file" in text and digest(stand_in) == input_before and not os.path.exists(free_report), f"R3 an output that is an input through {label} is refused: exit code {code}, input unchanged byte for byte")
        # R4: options that are not positive finite numbers
        bad = [["--s-base", "0"], ["--s-base", "nan"], ["--s-base", "inf"], ["--condition-limit", "-1"], ["--residual-limit", "0"], ["--coupling-limit", "nan"], ["--weight-tolerance", "0"], ["--comparison-x-over-r", "0"], ["--rating-index", "-1"]]
        # Each must stop the builder before anything is created
        codes = [run(option + ["--out", free_out, "--report", free_report])[0] for option in bad]
        # Judge
        t.check(all(c != 0 for c in codes) and not os.path.exists(free_out) and not os.path.exists(free_report), f"R4 {len(bad)} options that are not positive finite numbers are refused ({', '.join(' '.join(o) for o in bad)}): exit codes {codes}, nothing created")
        # R5: a dry run with the same inputs finishes and writes nothing
        code, text = run(["--dry-run", "--out", free_out, "--report", free_report])
        # Judge
        t.check(code == 0 and "nothing written" in text and not os.path.exists(free_out) and not os.path.exists(free_report), f"R5 a dry run with the same inputs finishes with exit code {code} and writes nothing, so the refusals above are not the builder failing for another reason")
        # The earlier file once more, after everything
        t.check(digest(old) == before, "R1 after all runs the earlier file is still unchanged byte for byte")


def case_file(t, args, data):
    """F1 to F5: the written file."""
    # Heading
    print(f"\nThe written file {args.weighted}")
    # Absent
    if not os.path.exists(args.weighted):
        # Count as not run
        t.skip(f"F1 to F5: {args.weighted} is not present; run main_electricalweights.py first")
        # Done
        return
    # The network cases must have passed
    if "periods" not in data:
        # Count as not run
        t.skip("F1 to F5: not run, since the network cases did not pass")
        # Done
        return
    # Read it
    with open(args.weighted) as f:
        # Parse
        body = json.load(f)
    # Shorthand
    timeline = data["timeline"]
    # Meter names
    names = [n["meter"] for n in timeline["nodes"]]
    # F1: the parts that must be untouched
    same = body["nodes"] == timeline["nodes"] and body["meters"] == timeline["meters"] and len(body["periods"]) == len(timeline["periods"])
    # Period fields other than the edge set number
    same = same and all({k: v for k, v in p.items() if k != "graph"} == {k: v for k, v in q.items() if k != "graph"} for p, q in zip(body["periods"], timeline["periods"]))

    def bare(g):
        """An edge set's connections without any weight field."""
        # Drop the weight fields
        return [{k: v for k, v in e.items() if k not in ("admittance_abs", "weight_raw_pu", "same_node")} for e in g["edges"]]

    # Compare period by period, through each file's own edge set number
    same = same and all(bare(body["graphs"][int(p["graph"])]) == bare(timeline["graphs"][int(q["graph"])]) for p, q in zip(body["periods"], timeline["periods"]))
    # Judge
    t.check(same, f"F1 same nodes, meters, periods and connections as {args.timeline}: {len(body['periods'])} periods, {len(body['graphs'])} edge sets with {[len(g['edges']) for g in body['graphs']]} edges")
    # F2: range
    every = [e["admittance_abs"] for g in body["graphs"] for e in g["edges"]]
    # Judge
    t.check(all(isinstance(w, float) and np.isfinite(w) and 0.0 < w <= 1.0 for w in every) and all(max(e["admittance_abs"] for e in g["edges"]) == 1.0 for g in body["graphs"]), f"F2 all {len(every)} weights are finite numbers in (0, 1], smallest {min(every):.4f}; every edge set's largest weight is 1")
    # F3: same-node marks, period by period
    wrong, marked = 0, 0
    # Each period
    for p in body["periods"]:
        # Its edges
        for e in body["graphs"][int(p["graph"])]["edges"]:
            # Whether the two meters sit on the same buses in this period
            together = p["topology"][names[e["source"]]]["node_buses"] == p["topology"][names[e["target"]]]["node_buses"]
            # Count the marks
            marked += int(e["same_node"])
            # The mark must say the same; a marked edge carries 1.0 and no raw weight; an unmarked edge has a raw weight
            wrong += int(together != e["same_node"] or (e["same_node"] and (e["admittance_abs"] != 1.0 or e["weight_raw_pu"] is not None)) or (not e["same_node"] and e["weight_raw_pu"] is None))
    # Judge
    t.check(wrong == 0, f"F3 same_node is set exactly for meters on the same buses, with weight 1.0 and no raw weight: {marked} marked edges over the periods, {wrong} wrong")
    # F4: against the second implementation, at the file's own power base
    settings = body["weights"]
    # Largest difference
    worst, count = 0.0, 0
    # Each period
    for i, p in enumerate(body["periods"]):
        # Second implementation
        second = second_weights(data["net"], data["graph"], data["records"], timeline, i, settings["s_base_va"])
        # Largest value, for the scaling
        top = max(w for w in second.values() if w is not None)
        # Each edge
        for e in body["graphs"][int(p["graph"])]["edges"]:
            # Its value
            w = second[frozenset((names[e["source"]], names[e["target"]]))]
            # Count
            count += 1
            # Scaled the same way
            worst = max(worst, abs(e["admittance_abs"] - (1.0 if w is None else w / top)))
    # Judge
    t.check(worst < args.tolerance, f"F4 stored weights equal the second implementation's: {count} comparisons over {len(body['periods'])} periods, largest difference {worst:.1e}")

    def identity(path):
        """File name and the first 16 hex digits of the SHA-256 of its bytes."""
        # Hash the bytes
        with open(path, "rb") as f:
            # Name and digest
            return f"{os.path.basename(path)}:{hashlib.sha256(f.read()).hexdigest()[:16]}"

    # The switch record files in use
    folder = os.path.join(args.topology_dir, "parameter_timeseries")
    # Their identities, in name order
    switches = [identity(os.path.join(folder, name)) for name in sorted(os.listdir(folder)) if name.endswith("-tbus_status.csv")]
    # All inputs
    now = {"graph": identity(args.graph), "timeline": identity(args.timeline), "network": identity(args.network), "switch_records": switches}
    # Judge
    t.check(settings["inputs"] == now, f"F5 the file records the inputs in use: {settings['inputs']['timeline']}, {settings['inputs']['graph']}, {settings['inputs']['network']}, and {len(settings['inputs'].get('switch_records', []))} switch record files")
    # The source file is untouched: every weight still 1.0
    t.check(all(e["admittance_abs"] == 1.0 for g in timeline["graphs"] for e in g["edges"]), f"F5 {args.timeline} still has every weight at 1.0")


def main():
    """Run every case, print the tally, return the exit code."""
    # Options
    args = parse_args()
    # Tally
    t = Tally()
    # Say what runs
    print(f"electrical weight checks: graph {args.graph}, timeline {args.timeline}, weighted file {args.weighted}, power base {args.s_base:.0f} VA, tolerance {args.tolerance}")
    # Steps in isolation
    case_steps(t, args)
    # Reduction
    case_reduction(t, args)
    # Weights and scaling
    case_scaling(t, args)
    # Inputs for the network cases
    data = {}
    # The electrical graph
    with open(args.graph) as f:
        # Parse
        data["graph"] = json.load(f)
    # The timeline
    with open(args.timeline) as f:
        # Parse
        data["timeline"] = json.load(f)
    # The network file
    with open(args.network) as f:
        # Parse
        data["net"] = json.load(f)
    # Switch records
    data["records"] = smg.read_switch_records(os.path.join(args.topology_dir, "parameter_timeseries"))
    # Per-unit terminal blocks
    data["branches"], data["notes"] = ew.branches_per_unit(data["net"], data["graph"], args.s_base)
    # The real network
    case_network(t, args, data)
    # The builder's refusals
    case_refusals(t, args)
    # The written file
    case_file(t, args, data)
    # Tally
    print(f"\n{t.passed} of {t.passed + t.failed} checks PASS, {t.failed} FAIL, {t.skipped} not run")
    # Exit code: failures, or checks that could not run, are not a pass
    return 0 if t.failed == 0 and t.skipped == 0 else 1


# Run only when executed directly
if __name__ == "__main__":
    # Exit with the code
    sys.exit(main())
