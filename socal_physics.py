# =============================================================================
# File:        socal_physics.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-20
#
# Purpose:     Helper functions for the physics baseline: a least-squares
#              state estimator with no learning. Imported by main_physics.py.
#
#              For every ten-second slot it takes the measured voltage and
#              current phasors, the admittance matrix of a FIXED reference
#              wiring (the normal switch state), and solves for the bus
#              voltages that best fit the circuit equations. The leftover
#              gap, the residual, is the detection signal: when a switch or
#              breaker changes state, the readings stop fitting the fixed
#              wiring and the residual shifts.
#
#              The model follows Caltech's bus injection model (BIM) in
#              code_examples/state_estimation_phasor.py of their repo,
#              including its objective, normalization, and injection map.
#              Section VI-A of the paper (arXiv 2504.06588) reports a 0.851
#              percent average normalized residual on 228 points over one
#              normal hour of the A-side circuit. That window is the first
#              check for this code.
#
# Caltech code: The graph and admittance matrix come from Caltech's own
#              utils folder, copied unchanged into this repo as caltech_utils/.
#              Their global_param.py expects that folder to be named utils/
#              and the topology to sit under sample_dataset/topology/. We do
#              not edit their files. setup_caltech() redirects those paths at
#              startup, so their code reads our caltech_utils/ and topology/.
#
# Functions:
#   Caltech setup and reference model
#     setup_caltech(caltech_dir, topology_dir)   import Caltech's modules, redirect paths
#     load_config(path)                          read physics_config.json
#     build_reference_model(ct, topology_dir, config)
#                                                electrical graph, node list, Y matrix
#   Measurement plan
#     register_lookup(net_data)                  element.phase -> (meter, channel, unit, rating)
#     bus_to_node(node_list)                     bus name -> electrical node name
#     measurement_plan(model, config)            which readings feed which equations
#   Reading the processed data
#     utc_timestamp(text)                        time string -> UTC-aware Timestamp
#     months_in_range(start_utc, end_utc)        "YYYY-MM" folders a window touches
#     load_block(processed_dir, meters, start_utc, end_utc)
#                                                one table of the needed columns
#     aligned_phasor(table, meter, channel)      complex phasor shifted to slot time
#   Assembling and solving
#     assemble(model, plan, table)               measured vectors, nominals, row masks
#     solve_slot_numpy(...)                      weighted least squares, closed form
#     solve_slot_cvxpy(...)                      Caltech's objective via cvxpy and ECOS
#     run_block(model, plan, table, solver, weight, step, progress)
#                                                residual per slot for one window
#     summarize(results)                         mean, percentiles, valid count
# =============================================================================

# Standard library: JSON for the config file
import json
# Standard library: file and folder operations
import os
# Standard library: module path handling
import sys
# Standard library: simple namespace to hold Caltech's modules
from types import SimpleNamespace
# Third party: numerics
import numpy as np
# Third party: tables and Parquet reading
import pandas as pd
# Third party: Parquet schema reading, to pick columns before loading
import pyarrow.parquet as pq
# Third party: graph library, used to pick the connected component
import networkx as nx

# Separator used in processed column names
SEP = "|"
# Name of the per-meter capture-offset column
TOFF = "toff"
# Value columns the baseline reads for every channel
NEEDED_VALUES = ("rms", "ang0", "freq")
# Phase letters in matrix order
PHASES = ("a", "b", "c")


# -----------------------------------------------------------------------------
# Caltech setup and reference model
# -----------------------------------------------------------------------------

def setup_caltech(caltech_dir, topology_dir):
    """
    Make Caltech's utils importable and point their file paths at our
    folders. Returns a namespace with their modules: utils, math_utils,
    topology, phasor_utils, global_param.

    Their global_param.py builds paths from its own location assuming the
    folder is named utils/ and the data sits under sample_dataset/. We
    overwrite the entries in its FILE_PATHS dictionary before importing
    the other modules, which read that same dictionary.
    """
    # Absolute path of the Caltech folder
    caltech_dir = os.path.abspath(caltech_dir)
    # Absolute path of our topology folder
    topology_dir = os.path.abspath(topology_dir)
    # Fail early with a clear message
    if not os.path.exists(os.path.join(caltech_dir, "topology.py")):
        # Raise so the caller sees the path tried
        raise FileNotFoundError(f"Caltech utils not found in {caltech_dir}")
    # Their modules import each other by bare name, so the folder must be on sys.path
    if caltech_dir not in sys.path:
        # Put it first so "import utils" finds their utils.py
        sys.path.insert(0, caltech_dir)
    # Their path table
    import global_param
    # Element inheritance file lives next to their code
    global_param.FILE_PATHS["element_inheritance"] = os.path.join(caltech_dir, "element_inheritance.json")
    # Cable table lives next to their code
    global_param.FILE_PATHS["cable_info"] = os.path.join(caltech_dir, "cable_info.csv")
    # Network JSON folder is ours
    global_param.FILE_PATHS["net_files"] = os.path.join(topology_dir, "network_files")
    # Breaker and switch status files are ours
    global_param.FILE_PATHS["net_params"] = os.path.join(topology_dir, "parameter_timeseries")
    # Change logs are ours
    global_param.FILE_PATHS["net_metadata"] = os.path.join(topology_dir, "metadata")
    # Now import the rest, which read FILE_PATHS at import time
    import utils
    # Matrix helpers
    import math_utils
    # Graph and admittance code
    import topology
    # Phasor helpers
    import phasor_utils
    # Hand back one object holding all of them
    return SimpleNamespace(utils=utils, math_utils=math_utils, topology=topology, phasor_utils=phasor_utils, global_param=global_param)


def load_config(path):
    """Read physics_config.json and return it as a dictionary."""
    # Fail with a clear message if missing
    if not os.path.exists(path):
        # Raise so the caller sees the path tried
        raise FileNotFoundError(f"Config not found: {path}")
    # Read and parse
    with open(path) as f:
        # Dictionary
        return json.load(f)


def build_reference_model(ct, topology_dir, config):
    """
    Build the fixed reference model at config["reference_time"].

    Steps, all with Caltech's code:
      1. load the network JSON and apply the switch states at that time
      2. physical graph: buses as nodes, every element as an edge
      3. electrical graph: open switches removed, closed zero-impedance
         elements collapsed so their two buses become one node
      4. keep the connected component holding config["component_anchor_bus"]
      5. drop the nodes listed in config["prune_nodes"]
      6. admittance matrix Y, 3N x 3N complex, three phases per node

    Returns a dictionary: net_data, node_list, Y, edges, n_nodes.
    """
    # Folder with the circuit's JSON files
    net_folder = os.path.join(os.path.abspath(topology_dir), "network_files", "circuit3")
    # Element type inheritance, needed by Caltech's graph code
    parents = ct.utils.dict_parents(ct.utils.load_json(ct.global_param.FILE_PATHS["element_inheritance"], raise_err=True))
    # Step 1: network at the reference time, switch states applied
    net_data = ct.topology.load_net_files([net_folder], t=config["reference_time"], timeseries_dir=ct.global_param.FILE_PATHS["net_params"])
    # Step 2: physical graph
    nodes, edges, _ = ct.topology.net2graph(net_data, parents, fix_implicit=False, connected_edges=True, print_warning=False)
    # Refuse duplicate edges, as Caltech's code does
    if ct.topology.check_duplicate_edge(edges):
        # Raise
        raise RuntimeError("Duplicate edge in the physical graph")
    # Step 3: electrical graph
    nodes_e, edges_e, _ = ct.topology.graph2electrical_graph(nodes, edges, {}, net_data, parents)
    # Step 4: connected component holding the anchor bus
    graph = nx.Graph()
    # Every node, including isolated ones
    graph.add_nodes_from(nodes_e.keys())
    # Every impedance edge
    graph.add_edges_from([(e["source"], e["target"]) for e in edges_e.values()])
    # Node that contains the anchor bus
    anchor_node = bus_to_node(list(nodes_e.keys())).get(config["component_anchor_bus"])
    # Must exist
    if anchor_node is None:
        # Raise
        raise ValueError(f"Anchor bus {config['component_anchor_bus']} not in the electrical graph")
    # Nodes in that component
    keep = nx.node_connected_component(graph, anchor_node)
    # Step 5: prune by name
    keep = keep - set(config.get("prune_nodes", []))
    # Node list in a fixed order
    node_list = sorted(keep)
    # Edges with both ends kept
    edges_kept = {k: e for k, e in edges_e.items() if e["source"] in keep and e["target"] in keep}
    # Step 6: admittance matrix
    Y = ct.topology.network_Y_matrix(edges_kept, node_list, net_data)
    # Hand back
    return {"net_data": net_data, "node_list": node_list, "Y": Y, "edges": edges_kept, "n_nodes": len(node_list)}


# -----------------------------------------------------------------------------
# Measurement plan
# -----------------------------------------------------------------------------

def register_lookup(net_data):
    """
    From the EgaugeMeter entries build {element.phase: (meter, channel, unit, rating)}.
    Example: "cb_146.a" -> ("egauge_7", "S1", "A", 3200)
             "bus_1119.ag" -> ("egauge_7", "L1", "V", None)
    """
    # Result
    lookup = {}
    # Loop over meters
    for meter in net_data.get("EgaugeMeter", {}).values():
        # Loop over registers
        for reg in meter.get("registers", []):
            # Element the register measures
            element = reg.get("element")
            # Skip registers that measure nothing
            if not element:
                # Next
                continue
            # Voltage registers carry the channel in "name", current registers in "value"
            channel = reg.get("name") if reg.get("unit") == "V" else reg.get("value")
            # Keep only the phasor channels we downloaded
            if channel not in ("L1", "L2", "L3", "S1", "S2", "S3"):
                # Next
                continue
            # Store
            lookup[element] = (meter["name"], channel, reg.get("unit"), reg.get("I_rating"))
    # Return
    return lookup


def bus_to_node(node_list):
    """Map every bus name to the electrical node (possibly combined) that holds it."""
    # Result
    mapping = {}
    # Loop over nodes
    for node in node_list:
        # A combined node is bus names joined by |
        for bus in node.split(SEP):
            # Store
            mapping[bus] = node
    # Return
    return mapping


def measurement_plan(model, config):
    """
    Decide which readings feed which equations.

    Returns a dictionary with:
      voltages:   list of {meter, channel, node_idx, phase_idx, nominal}
                  one per metered bus phase inside the kept component
      injections: list of {node_idx, terms: [{meter, channel per phase, sign, rating}]}
                  from config["measured_injections"]
      zeros:      list of node_idx from config["zero_injection_nodes"]
      meters:     sorted list of meters the plan reads
    Voltages measured line-to-line (egauge_17, egauge_18) are skipped, as
    Caltech's code does, since the model works in line-to-ground phasors.
    """
    # Network and nodes
    net_data = model["net_data"]
    # Node order
    node_list = model["node_list"]
    # Bus to node
    b2n = bus_to_node(node_list)
    # Node to index
    n2i = {n: i for i, n in enumerate(node_list)}
    # Register table
    regs = register_lookup(net_data)
    # Voltage equations
    voltages = []
    # Loop over registers
    for element, (meter, channel, unit, rating) in regs.items():
        # Voltages only
        if unit != "V":
            # Next
            continue
        # Split "bus_1119.ag"
        bus, _, suffix = element.partition(".")
        # Only line-to-ground phases a, b, c
        if suffix not in ("ag", "bg", "cg"):
            # Next
            continue
        # Node holding the bus, if kept
        node = b2n.get(bus)
        # Skip buses outside the kept component
        if node is None:
            # Next
            continue
        # Nominal voltage of the bus
        nominal = net_data["Bus"][bus].get("nominal_voltage")
        # Skip if unknown, as Caltech's code does
        if nominal is None:
            # Next
            continue
        # Store
        voltages.append({"meter": meter, "channel": channel, "node_idx": n2i[node], "phase_idx": PHASES.index(suffix[0]), "nominal": float(nominal)})
    # Injection equations
    injections = []
    # Loop over config entries
    for node, terms in config.get("measured_injections", {}).items():
        # Skip nodes not in the kept component
        if node not in n2i:
            # Next
            continue
        # Terms for this node
        plan_terms = []
        # Loop over elements
        for term in terms:
            # Channel per phase, from the registers
            channels = {}
            # Rating per phase
            ratings = []
            # Loop over phases
            for p in PHASES:
                # Register for element.phase
                hit = regs.get(f"{term['element']}.{p}")
                # Need a current register with a rating
                if hit is None or hit[2] != "A" or hit[3] is None:
                    # Mark incomplete
                    channels = None
                    # Stop
                    break
                # Store channel
                channels[p] = hit[1]
                # Store rating
                ratings.append(float(hit[3]))
                # Meter name is the same for all phases
                meter = hit[0]
            # Skip incomplete elements
            if not channels:
                # Next
                continue
            # Store
            plan_terms.append({"meter": meter, "channels": channels, "sign": float(term["sign"]), "rating": float(np.mean(ratings))})
        # Keep nodes with at least one usable term
        if plan_terms:
            # Store
            injections.append({"node_idx": n2i[node], "terms": plan_terms})
    # Zero-injection nodes present in the component
    zeros = [n2i[n] for n in config.get("zero_injection_nodes", []) if n in n2i]
    # Meters needed
    meters = sorted({v["meter"] for v in voltages} | {t["meter"] for inj in injections for t in inj["terms"]})
    # Soft nominal for zero-injection rows, Caltech uses 0.1
    zero_nominal = float(config.get("zero_injection_nominal", 0.1))
    # Return
    return {"voltages": voltages, "injections": injections, "zeros": zeros, "meters": meters, "zero_injection_nominal": zero_nominal}


# -----------------------------------------------------------------------------
# Reading the processed data
# -----------------------------------------------------------------------------

def utc_timestamp(text):
    """
    Turn a time string into a UTC-aware pandas Timestamp. A string with
    no zone, such as 2024-11-14T15:00:00, is taken as UTC. The processed
    tables carry a UTC-aware index, and pandas refuses to compare an
    aware index with a naive value.
    """
    # Parse
    ts = pd.Timestamp(text)
    # Label as UTC if no zone was given
    if ts.tzinfo is None:
        # Localize
        ts = ts.tz_localize("UTC")
    # Otherwise convert whatever zone was given to UTC
    else:
        # Convert
        ts = ts.tz_convert("UTC")
    # Return
    return ts


def months_in_range(start_utc, end_utc):
    """Return the "YYYY-MM" folder names a UTC window touches, in order."""
    # First month
    current = utc_timestamp(start_utc).tz_localize(None).to_period("M")
    # Last month, end is exclusive so step back one microsecond
    last = (utc_timestamp(end_utc) - pd.Timedelta(microseconds=1)).tz_localize(None).to_period("M")
    # Collect
    months = []
    # Loop
    while current <= last:
        # Label
        months.append(current.strftime("%Y-%m"))
        # Advance
        current += 1
    # Return
    return months


def load_block(processed_dir, meters, start_utc, end_utc):
    """
    Read the needed columns for the given meters over a UTC window and
    return one table indexed by slot time. Missing files or columns are
    filled with NaN so the caller only has to check masks.
    """
    # Pieces per month
    month_tables = []
    # Loop over months
    for month in months_in_range(start_utc, end_utc):
        # Pieces per meter
        meter_tables = []
        # Loop over meters
        for meter in meters:
            # File path
            path = os.path.join(processed_dir, month, f"{meter}.parquet")
            # Skip missing files, they show up as NaN after the join
            if not os.path.exists(path):
                # Next
                continue
            # Column names on disk, read from the file schema without loading data
            names = pq.read_schema(path).names
            # Columns we want: values, masks, offset
            wanted = [c for c in names if c.startswith(f"{meter}{SEP}") and (c.endswith(tuple(f"{SEP}{v}" for v in NEEDED_VALUES)) or c == f"{meter}{SEP}{TOFF}")]
            # Mask columns
            wanted += [c for c in names if c.startswith(f"mask{SEP}{meter}{SEP}")]
            # Read
            meter_tables.append(pd.read_parquet(path, columns=wanted))
        # Nothing for this month
        if not meter_tables:
            # Next
            continue
        # Join meters side by side
        month_tables.append(pd.concat(meter_tables, axis=1))
    # Nothing at all
    if not month_tables:
        # Empty table
        return pd.DataFrame()
    # Stack months
    table = pd.concat(month_tables, axis=0)
    # Window bounds as UTC-aware timestamps
    start_ts, end_ts = utc_timestamp(start_utc), utc_timestamp(end_utc)
    # Clip to the window, end excluded
    table = table[(table.index >= start_ts) & (table.index < end_ts)]
    # Return
    return table


def aligned_phasor(table, meter, channel):
    """
    Complex phasor for one meter channel at every slot, shifted to the
    slot time: magnitude = rms, angle = ang0 - 360 * freq * toff degrees.
    Slots without a reading are NaN.
    """
    # Column names
    rms_col = f"{meter}{SEP}{channel}{SEP}rms"
    # Angle
    ang_col = f"{meter}{SEP}{channel}{SEP}ang0"
    # Frequency
    freq_col = f"{meter}{SEP}{channel}{SEP}freq"
    # Offset
    toff_col = f"{meter}{SEP}{TOFF}"
    # Any column missing means no data
    for c in (rms_col, ang_col, freq_col, toff_col):
        # Check
        if c not in table.columns:
            # All NaN
            return np.full(len(table), np.nan, dtype=np.complex128)
    # Magnitude
    rms = table[rms_col].to_numpy(dtype=np.float64)
    # Angle at capture, degrees
    ang = table[ang_col].to_numpy(dtype=np.float64)
    # Frequency
    freq = table[freq_col].to_numpy(dtype=np.float64)
    # Capture offset
    toff = table[toff_col].to_numpy(dtype=np.float64)
    # Angle at the slot time, degrees
    ang_slot = ang - 360.0 * freq * toff
    # Complex phasor
    return rms * np.exp(1j * np.deg2rad(ang_slot))


# -----------------------------------------------------------------------------
# Assembling and solving
# -----------------------------------------------------------------------------

def assemble(model, plan, table):
    """
    Turn the table into measurement arrays.

    Returns a dictionary:
      V_meas     (MV, T) complex, one row per measured bus phase
      V_idx      (MV,) int, row of the state vector each measurement reads
      V_nom      (MV,) float, nominal voltage per row
      I_meas     (MI, T) complex, one row per injection phase (zeros for zero-injection nodes)
      I_rows     (MI,) int, row of Y each injection equation uses
      I_nom      (MI,) float, nominal current per row
      V_ok       (MV, T) bool, reading present
      I_ok       (MI, T) bool, all readings behind the row present
    """
    # Number of slots
    T = len(table)
    # Voltage rows
    V_meas, V_idx, V_nom = [], [], []
    # Loop over planned voltages
    for v in plan["voltages"]:
        # Phasor
        V_meas.append(aligned_phasor(table, v["meter"], v["channel"]))
        # State row: node index times three plus phase
        V_idx.append(3 * v["node_idx"] + v["phase_idx"])
        # Nominal
        V_nom.append(v["nominal"])
    # Injection rows
    I_meas, I_rows, I_nom = [], [], []
    # Loop over measured injections
    for inj in plan["injections"]:
        # Loop over phases
        for p_i, p in enumerate(PHASES):
            # Sum of signed terms
            total = np.zeros(T, dtype=np.complex128)
            # Ratings
            ratings = []
            # Loop over terms
            for term in inj["terms"]:
                # Signed phasor
                total = total + term["sign"] * aligned_phasor(table, term["meter"], term["channels"][p])
                # Rating
                ratings.append(term["rating"])
            # Store
            I_meas.append(total)
            # Y row
            I_rows.append(3 * inj["node_idx"] + p_i)
            # Nominal is the mean rating
            I_nom.append(float(np.mean(ratings)))
    # Zero-injection rows
    for node_idx in plan["zeros"]:
        # Loop over phases
        for p_i in range(3):
            # Zero current
            I_meas.append(np.zeros(T, dtype=np.complex128))
            # Y row
            I_rows.append(3 * node_idx + p_i)
            # Soft nominal from the config, Caltech uses 0.1
            I_nom.append(float(plan.get("zero_injection_nominal", 0.1)))
    # Stack
    V_meas = np.array(V_meas) if V_meas else np.zeros((0, T), dtype=np.complex128)
    # Stack
    I_meas = np.array(I_meas) if I_meas else np.zeros((0, T), dtype=np.complex128)
    # Presence
    V_ok = ~np.isnan(V_meas)
    # Presence
    I_ok = ~np.isnan(I_meas)
    # Return
    return {"V_meas": V_meas, "V_idx": np.array(V_idx, dtype=int), "V_nom": np.array(V_nom, dtype=float), "I_meas": I_meas, "I_rows": np.array(I_rows, dtype=int), "I_nom": np.array(I_nom, dtype=float), "V_ok": V_ok, "I_ok": I_ok}


def objective_value(Y, V_hat, V_meas, V_idx, V_nom, I_meas, I_rows, I_nom, weight):
    """
    Caltech's objective at a given state: weight * sum |I residual| / I_nom
    plus sum |V residual| / V_nom. Returns (value, count of terms).
    """
    # Current residuals
    I_res = np.abs(I_meas - Y[I_rows] @ V_hat) / I_nom
    # Voltage residuals
    V_res = np.abs(V_meas - V_hat[V_idx]) / V_nom
    # Value and count
    return float(weight * I_res.sum() + V_res.sum()), int(len(I_res) + len(V_res))


def solve_slot_numpy(Y, V_meas, V_idx, V_nom, I_meas, I_rows, I_nom, weight, n_state):
    """
    Weighted least squares in closed form: minimize the sum of squared
    normalized residuals. Fast, used for long windows. Returns V_hat.
    """
    # Current block of the measurement matrix, scaled by weight and nominal
    A_I = (np.sqrt(weight) / I_nom)[:, None] * Y[I_rows]
    # Voltage block: selection rows scaled by nominal
    A_V = np.zeros((len(V_idx), n_state), dtype=np.complex128)
    # One entry per row
    A_V[np.arange(len(V_idx)), V_idx] = 1.0 / V_nom
    # Stack the matrix
    A = np.vstack([A_I, A_V])
    # Stack the right-hand side
    b = np.concatenate([np.sqrt(weight) * I_meas / I_nom, V_meas / V_nom])
    # Least squares
    V_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    # Return
    return V_hat


def solve_slot_cvxpy(Y, V_meas, V_idx, V_nom, I_meas, I_rows, I_nom, weight, n_state):
    """
    Caltech's objective exactly: minimize the sum of absolute normalized
    residuals, solved with cvxpy and ECOS. Slower. Returns V_hat or None.
    """
    # Import here so the numpy path works without cvxpy installed
    import cvxpy as cp
    # Complex state vector
    V = cp.Variable(n_state, complex=True)
    # Current residual magnitudes
    I_res = cp.abs(I_meas - Y[I_rows] @ V)
    # Voltage residual magnitudes
    V_res = cp.abs(V_meas - V[V_idx])
    # Objective
    obj = cp.Minimize(weight * cp.sum(I_res / I_nom) + cp.sum(V_res / V_nom))
    # Problem
    prob = cp.Problem(obj)
    # Solve
    try:
        # ECOS as in Caltech's code
        prob.solve(solver=cp.ECOS)
    # Any solver failure
    except Exception:
        # No answer
        return None
    # No value means failure
    if V.value is None:
        # No answer
        return None
    # Return
    return np.asarray(V.value).flatten()


def run_block(model, plan, table, solver="cvxpy", weight=0.1, step=1, progress=True):
    """
    Solve every step-th slot in the table and return a DataFrame indexed
    by slot time with columns:
      residual        objective value (sum of normalized residuals)
      residual_mean   objective value divided by the number of terms
      n_voltage       voltage terms used in that slot
      n_current       current terms used in that slot (zero-injection rows included)
      status          "ok", "no_data", "too_few", or "solver_failed"
    Slots with no usable readings are marked, never scored as zero.
    """
    # Admittance matrix as complex128 for the solvers
    Y = model["Y"].astype(np.complex128)
    # State size
    n_state = 3 * model["n_nodes"]
    # Arrays
    m = assemble(model, plan, table)
    # Slot times
    times = table.index[::step]
    # Row positions
    positions = np.arange(len(table))[::step]
    # Progress bar if asked
    if progress:
        # Import here so the module works without tqdm
        from tqdm import tqdm
        # Wrap
        iterator = tqdm(positions, desc="slots")
    # Plain
    else:
        # No bar
        iterator = positions
    # Output rows
    rows = []
    # Loop over slots
    for pos in iterator:
        # Voltage rows present
        v_ok = m["V_ok"][:, pos]
        # Current rows present
        i_ok = m["I_ok"][:, pos]
        # Counts
        n_v, n_i = int(v_ok.sum()), int(i_ok.sum())
        # No readings at all
        if n_v == 0 and (n_i == 0 or np.all(m["I_meas"][i_ok, pos] == 0)):
            # Mark
            rows.append({"residual": np.nan, "residual_mean": np.nan, "n_voltage": n_v, "n_current": n_i, "status": "no_data"})
            # Next
            continue
        # Need at least one voltage to pin the magnitude scale
        if n_v == 0:
            # Mark
            rows.append({"residual": np.nan, "residual_mean": np.nan, "n_voltage": n_v, "n_current": n_i, "status": "too_few"})
            # Next
            continue
        # Pick the solver
        solve = solve_slot_cvxpy if solver == "cvxpy" else solve_slot_numpy
        # Solve with the rows present
        V_hat = solve(Y, m["V_meas"][v_ok, pos], m["V_idx"][v_ok], m["V_nom"][v_ok], m["I_meas"][i_ok, pos], m["I_rows"][i_ok], m["I_nom"][i_ok], weight, n_state)
        # Solver failure
        if V_hat is None:
            # Mark
            rows.append({"residual": np.nan, "residual_mean": np.nan, "n_voltage": n_v, "n_current": n_i, "status": "solver_failed"})
            # Next
            continue
        # Objective at the solution, same formula for both solvers
        value, count = objective_value(Y, V_hat, m["V_meas"][v_ok, pos], m["V_idx"][v_ok], m["V_nom"][v_ok], m["I_meas"][i_ok, pos], m["I_rows"][i_ok], m["I_nom"][i_ok], weight)
        # Store
        rows.append({"residual": value, "residual_mean": value / count, "n_voltage": n_v, "n_current": n_i, "status": "ok"})
    # Table
    out = pd.DataFrame(rows, index=times)
    # Name the index
    out.index.name = "t_utc"
    # Return
    return out


def summarize(results):
    """Return a small dictionary of summary numbers for a results table."""
    # Scored slots
    ok = results[results["status"] == "ok"]
    # Nothing scored
    if ok.empty:
        # Empty summary
        return {"slots": int(len(results)), "scored": 0}
    # Summary
    return {
        # All slots
        "slots": int(len(results)),
        # Scored slots
        "scored": int(len(ok)),
        # Mean objective
        "residual_mean_of_objective": float(ok["residual"].mean()),
        # Mean per-term residual, in percent
        "residual_mean_per_term_percent": float(100.0 * ok["residual_mean"].mean()),
        # Median per-term residual, in percent
        "residual_median_per_term_percent": float(100.0 * ok["residual_mean"].median()),
        # 99th percentile per-term residual, in percent
        "residual_p99_per_term_percent": float(100.0 * ok["residual_mean"].quantile(0.99)),
        # Status counts
        "status_counts": results["status"].value_counts().to_dict(),
    }
