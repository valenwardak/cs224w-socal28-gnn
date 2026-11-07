# =============================================================================
# File:        main_topologyaudit.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-07
#
# Purpose:     Audit of the graph the models use against the dataset's
#              time-varying topology. graph.json is one snapshot, built at
#              the reference time in physics_config.json. The switch
#              records in topology/parameter_timeseries/ change the wiring
#              over the record. This script establishes, from the source
#              files and Caltech's own topology code, when the real wiring
#              differed from the snapshot and what the difference was.
#              The committed graphs are first checked against fresh builds
#              (graph.json at its reference time, graph_meters.json from
#              graph.json), so a stale file is not mistaken for a topology
#              difference. Then seven parts:
#                1. switch records: every state change with its time in the
#                   files' fixed Pacific Standard Time and in UTC, the two
#                   buses the switch joins, and the gap to the previous
#                   change; repeated unchanged records are collapsed; the
#                   lookup rule is verified by reading the state through
#                   Caltech's loader one second before, at, and one second
#                   after a change, and the initial-state rule is checked
#                2. topology periods: half-open intervals between groups of
#                   simultaneous changes, from the first record to the end
#                   of the data; for each, the electrical graph is built
#                   with Caltech's code and compared with the reference
#                   graph by bus identity and element name: merged bus sets,
#                   edges, components, and the node each meter sits on
#                3. meter graph per period: rebuilt with socal_metergraph
#                   and compared with graph_meters.json by meter pair, with
#                   physical hops and shared-node relationships both shown
#                4. overlay on the splits: for every block of every splits
#                   file, the hours in a period whose complete topology
#                   differs from the reference, and the hours in a period
#                   where connectivity among the meters reporting in that
#                   month differs (reporting taken from coverage.csv)
#                5. meter mapping: each meter's registers read independently
#                   from the network file (voltage bus, channel-to-phase
#                   map, current elements, ratings) compared with graph.json
#                   channel by channel, with reversed phase labels and
#                   line-to-line voltage channels noted, the current
#                   element's two end buses, and the current sign as
#                   configured in physics_config.json (copied from Caltech's
#                   example) or "unknown"; no sign is verified here
#                6. loader check: on graph.json and graph_meters.json, the
#                   training code's meter_node_index and build_edges give the
#                   node per meter, every edge in both directions, and each
#                   directed edge's weight equal to admittance_abs over the
#                   file maximum
#                7. the reference time read both ways (files' time base and
#                   UTC) against the last restoration, so the state the
#                   snapshot reflects is explicit
#              Every failed check is collected and printed at the end, and
#              the exit code is 1 on any failure or incomplete check.
#              A topology lookup applying a change at time t says when the
#              wiring changed, not when a forecasting model may use it; that
#              follows the pre-target rule and is decided from this output.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root. Needs the
#              repo's caltech_utils/ and topology/ folders and the packages
#              socal_physics imports.
#
# Usage:
#   python checks/main_topologyaudit.py
#   python checks/main_topologyaudit.py --coverage data/processed/coverage.csv --csv topology_periods.csv
#   python checks/main_topologyaudit.py --splits splits.json splits_v2.json
#
# Functions:
#   parse_args()                    command-line options with defaults
#   to_utc(t_std, offset_hours)     fixed-PST timestamp to UTC
#   read_switch_records(ts_dir)     every switch's records, collapsed
#   switch_buses(net_file)          the two buses each switch joins, and the nominal status
#   verify_lookup(ct, topology_dir, changes, net_folder)
#                                   the state one second before, at, and after each change
#   build_periods(changes, start, end)
#                                   half-open topology periods between change groups
#   graph_at(ct, topology_dir, config, t, path)
#                                   graph.json-style dict built at time t
#   node_key(node)                  frozenset of buses, the identity of a node
#   compare_graphs(ref, g)          differences by bus set, element, component, meter node
#   physical_hops(g)                hop distance between every meter pair in a graph dict
#   meter_graph_diff(ref_mg, mg)    edges added and removed by meter pair
#   reporting_meters(coverage, month)
#                                   meters with any reading in a month
#   overlay(periods, splits, offset_hours, coverage, diffs, mdiffs)
#                                   hours per block in differing topology
#   meter_mapping(graph, net_file, config)
#                                   the mapping table rows
#   loader_check(graph_path, manifest)
#                                   the training code's view of graph.json
#   main()                          run the seven parts
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: CSV reading and writing
import csv
# Standard library: JSON
import json
# Standard library: paths, temporary files, exit
import os
import sys
import tempfile
# Standard library: breadth-first search
from collections import deque
# Third party: timestamps
import pandas as pd

# The repo root, one level above checks/, so the project modules import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project: Caltech setup and the graph export
import socal_physics as sp
# Project: the meter graph builder
import socal_metergraph as smg


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Audit graph.json against the dataset's time-varying topology")
    # Folders and files
    parser.add_argument("--topology-dir", default="topology", help="Caltech's topology folder in the repo")
    parser.add_argument("--caltech-dir", default="caltech_utils", help="Caltech's utils folder in the repo")
    parser.add_argument("--config", default="physics_config.json", help="Physics config with the reference time and the configured current signs")
    parser.add_argument("--graph", default="graph.json", help="The committed reference graph")
    parser.add_argument("--meter-graph", default="graph_meters.json", help="The committed meter graph")
    parser.add_argument("--splits", nargs="+", default=["splits.json", "splits_v2.json"], help="Splits files to overlay")
    parser.add_argument("--coverage", default=os.path.join("data", "processed", "coverage.csv"), help="Per-month coverage report, for which meters report")
    parser.add_argument("--manifest", default=os.path.join("data", "features", "manifest.json"), help="features manifest.json, for the loader check")
    # Record bounds, in the files' fixed-PST time base
    parser.add_argument("--record-end", default="2025-09-04T00:00:00", help="End of the record, fixed PST, exclusive")
    # Time base
    parser.add_argument("--utc-offset-hours", type=int, default=8, help="Hours added to the files' standard time to get UTC")
    # CSV
    parser.add_argument("--csv", default=None, help="Write the period table to this CSV as well")
    # Parse
    return parser.parse_args()


def to_utc(t_std, offset_hours):
    """Fixed-PST timestamp (naive) to a UTC-aware timestamp."""
    # Add the offset and label
    return (pd.Timestamp(t_std) + pd.Timedelta(hours=offset_hours)).tz_localize("UTC")


def read_switch_records(ts_dir):
    """
    Every switch's records from <switch>-tbus_status.csv, as a dict
    switch -> list of (time, state) with repeated unchanged records
    collapsed. Times are naive timestamps in the files' time base.
    """
    # Result
    out = {}
    # Loop over files
    for fn in sorted(os.listdir(ts_dir)):
        # Only status files
        if not fn.endswith("-tbus_status.csv"):
            # Next
            continue
        # Switch name
        switch = fn[:-len("-tbus_status.csv")]
        # Read
        with open(os.path.join(ts_dir, fn)) as f:
            # Rows after the header
            rows = [r for r in csv.reader(f)][1:]
        # Records
        recs = []
        # Loop
        for t, state in rows:
            # Collapse repeats
            if recs and recs[-1][1] == state:
                # Skip
                continue
            # Keep
            recs.append((pd.Timestamp(t), state))
        # Store
        out[switch] = recs
    # Done
    return out


def switch_buses(net_file):
    """
    The two buses each switch joins and its nominal status, from the
    network file: switch name -> (fbus, tbus, nominal_status, element type).
    A multi-position switch has several tbus entries; the one whose status
    is a file reference is the recorded one.
    """
    # Read
    with open(net_file) as f:
        # Parse
        net = json.load(f)
    # Result
    out = {}
    # Switch-like categories
    for cat in ("CB", "Switch", "SwitchMultiPosition", "Fuse"):
        # Records
        for rec in net.get(cat, []):
            # Each tbus entry
            for tb in rec.get("tbus", []):
                # Status source
                status = tb.get("status")
                # Recorded in a file
                if isinstance(status, str) and status.startswith("file:"):
                    # Name used by the file
                    key = status[len("file:"):].replace("-tbus_status.csv", "")
                    # Store
                    out[key] = (rec["fbus"], tb["name"], tb.get("nominal_status"), cat, rec["name"])
    # Done
    return out


def verify_lookup(ct, topology_dir, changes, net_folder):
    """
    For each change (time, switch, new state), read the switch's status
    through Caltech's loader one second before, at, and one second after
    the change. Returns rows (time, switch, before, at, after, expected).
    """
    # Rows
    rows = []
    # Loop
    for t, switch, old, new, _ in changes:
        # The element name and tbus of this switch
        states = []
        # Three times
        for dt in (-1, 0, 1):
            # Time
            tt = (pd.Timestamp(t) + pd.Timedelta(seconds=dt)).to_pydatetime()
            # Load
            net = ct.topology.load_net_files([net_folder], t=tt, timeseries_dir=ct.global_param.FILE_PATHS["net_params"])
            # Find the status
            found = None
            # The element and tbus this switch file belongs to
            elem, tbus = switch_elem.get(switch), switch_tbus.get(switch)
            # Categories
            for cat in ("CB", "Switch", "SwitchMultiPosition", "Fuse"):
                # Elements by name
                for name, rec in net.get(cat, {}).items():
                    # Only that element
                    if name != elem:
                        # Next
                        continue
                    # Each tbus
                    for tb in rec.get("tbus", []):
                        # Only that tbus
                        if tb.get("name") == tbus:
                            # Status now a plain value
                            found = tb.get("status")
            # Keep
            states.append(found)
        # Row
        rows.append((t, switch, states[0], states[1], states[2], new))
    # Done
    return rows


# Filled by main: switch file name -> tbus name and -> element name, for verify_lookup
switch_tbus = {}
switch_elem = {}


def build_periods(changes, start, end):
    """
    Half-open topology periods [t0, t1) from the record start to the end,
    cut at every distinct change time. Each period carries the group of
    changes that opened it (empty for the first).
    """
    # Bounds must be in order
    if not end > start:
        # Raise
        raise ValueError(f"record end {end} is not after record start {start}")
    # Distinct change times strictly inside the record
    times = sorted({t for t, *_ in changes if start < t < end})
    # Bounds
    bounds = [start] + times + [end]
    # Periods
    out = []
    # Loop
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        # Changes at t0
        group = [c for c in changes if c[0] == t0]
        # Keep
        out.append({"start": t0, "end": t1, "changes": group})
    # Done
    return out


def graph_at(ct, topology_dir, config, t, path):
    """The graph.json-style dictionary built at time t (files' time base), through socal_physics.export_graph."""
    # Config with the time replaced
    cfg = dict(config, reference_time=pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S"))
    # Export to the path
    sp.export_graph(ct, topology_dir, cfg, path)
    # Read back
    with open(path) as f:
        # Parse
        return json.load(f)


def node_key(node):
    """The identity of a node: the frozenset of buses it merges."""
    # Set
    return frozenset(node["buses"])


def compare_graphs(ref, g):
    """
    Differences between two graph dicts by identity, not index: node bus
    sets added and removed, edges (by element name) added and removed,
    components (as frozensets of buses) added and removed, and meters
    whose node bus set changed.
    """
    # Node sets
    rn, gn = {node_key(n) for n in ref["nodes"]}, {node_key(n) for n in g["nodes"]}
    # Edges by element with their end bus sets
    def edge_map(x):
        # Element -> (bus set, bus set)
        return {e["element"]: frozenset([node_key(x["nodes"][e["source"]]), node_key(x["nodes"][e["target"]])]) for e in x["edges"]}
    # Maps
    re, ge = edge_map(ref), edge_map(g)
    # Components as bus sets
    def comps(x):
        # Component id -> buses
        c = {}
        # Loop
        for n in x["nodes"]:
            # Add
            c.setdefault(n["component"], set()).update(n["buses"])
        # Frozen
        return {frozenset(v) for v in c.values()}
    # Sets
    rc, gc = comps(ref), comps(g)
    # Meter node bus sets
    def meter_nodes(x):
        # Meter -> bus set of its node
        return {m: (node_key(x["nodes"][rec["node"]]) if rec.get("node") is not None else None) for m, rec in x["meters"].items()}
    # Maps
    rm, gm = meter_nodes(ref), meter_nodes(g)
    # Differences
    return {
        # Nodes
        "nodes_added": sorted("|".join(sorted(k)) for k in gn - rn), "nodes_removed": sorted("|".join(sorted(k)) for k in rn - gn),
        # Edges: present in one only, or with different ends
        "edges_added": sorted(e for e in ge if e not in re), "edges_removed": sorted(e for e in re if e not in ge),
        "edges_moved": sorted(e for e in re if e in ge and re[e] != ge[e]),
        # Components
        "components_changed": len(rc ^ gc) > 0, "n_components": len(gc),
        # Meters whose node changed
        "meters_moved": sorted(m for m in rm if rm[m] != gm.get(m)),
    }


def physical_hops(g):
    """Hop distance between every pair of meters in a graph dict, as {(a, b): hops}, with 0 for a shared node and None when unreachable."""
    # Adjacency
    adj = {i: set() for i in range(len(g["nodes"]))}
    # Edges
    for e in g["edges"]:
        # Both ways
        adj[e["source"]].add(e["target"]); adj[e["target"]].add(e["source"])
    # Meter nodes
    node_of = {m: rec["node"] for m, rec in g["meters"].items() if rec.get("node") is not None}
    # Result
    out = {}
    # Loop over meters
    for a, na in node_of.items():
        # Breadth-first
        dist = {na: 0}
        # Queue
        q = deque([na])
        # Search
        while q:
            # Pop
            u = q.popleft()
            # Neighbors
            for v in adj[u]:
                # New
                if v not in dist:
                    # Record
                    dist[v] = dist[u] + 1
                    # Push
                    q.append(v)
        # Pairs
        for b, nb in node_of.items():
            # Ordered once
            if a < b:
                # Record
                out[(a, b)] = dist.get(nb)
    # Done
    return out


def meter_graph_diff(ref_mg, mg):
    """Edges added and removed between two meter graphs, by meter pair."""
    # Pairs
    def pairs(x):
        # Set of sorted name pairs
        return {tuple(sorted((x["nodes"][e["source"]]["meter"], x["nodes"][e["target"]]["meter"]))) for e in x["edges"]}
    # Sets
    rp, gp = pairs(ref_mg), pairs(mg)
    # Differences
    return sorted(gp - rp), sorted(rp - gp)


def reporting_meters(coverage, month):
    """Meters with any reading in a month, from the coverage report; None when the report is missing."""
    # No report
    if coverage is None:
        # Unknown
        return None
    # Rows of the month with a reading
    return {r["meter"] for r in coverage if r["month"] == month and float(r["fraction"]) > 0}


def month_pieces(t0, t1, offset_hours):
    """
    Cut the UTC interval [t0, t1) at fixed-PST month boundaries. Yields
    (month label, piece start, piece end) in UTC.
    """
    # Cursor
    cur = t0
    # Loop
    while cur < t1:
        # The PST month of the cursor
        pst = cur - pd.Timedelta(hours=offset_hours)
        # Start of the next PST month, back in UTC
        nxt = (pd.Timestamp(year=pst.year + (pst.month == 12), month=(pst.month % 12) + 1, day=1, tz="UTC") + pd.Timedelta(hours=offset_hours))
        # Piece end
        end = min(t1, nxt)
        # Yield
        yield pst.strftime("%Y-%m"), cur, end
        # Advance
        cur = end


def reporting_differs(mdiff, rep):
    """Whether any meter-graph edge added or removed in a period joins two meters of the reporting set (every meter when rep is None)."""
    # Pairs
    for a, b in mdiff["added"] + mdiff["removed"]:
        # Both reporting
        if rep is None or (a in rep and b in rep):
            # Yes
            return True
    # No
    return False


def overlay(periods, splits, offset_hours, coverage, diffs, mdiffs):
    """
    For every block of a splits file, the hours inside periods whose
    complete topology differs from the reference, and the hours inside
    periods where the meter graph among the meters reporting in the
    month differs. Each block-period overlap is cut at fixed-PST month
    boundaries and judged with that month's reporting set, taken from the
    coverage report (any reading in the month: monthly availability, not
    simultaneous reporting). Returns rows (split, block label, hours,
    full-diff hours, reporting-diff hours).
    """
    # Rows
    rows = []
    # Loop over splits
    for split in ("train", "validation", "test_events", "test_normal"):
        # Blocks
        for b in splits.get(split, []):
            # Bounds in UTC
            b0, b1 = pd.Timestamp(b["start_utc"]), pd.Timestamp(b["end_utc_exclusive"])
            # Totals
            total = (b1 - b0).total_seconds() / 3600.0
            # Differing hours
            full, rep_h = 0.0, 0.0
            # Loop over periods
            for i, p in enumerate(periods):
                # Period bounds in UTC
                p0, p1 = to_utc(p["start"], offset_hours), to_utc(p["end"], offset_hours)
                # Overlap
                o0, o1 = max(b0, p0), min(b1, p1)
                # None
                if o0 >= o1:
                    # Next
                    continue
                # Pieces by month
                for month, q0, q1 in month_pieces(o0, o1, offset_hours):
                    # Hours
                    h = (q1 - q0).total_seconds() / 3600.0
                    # Complete topology differs
                    if diffs[i]["differs"]:
                        # Add
                        full += h
                    # Meter graph among that month's reporting meters differs
                    if reporting_differs(mdiffs[i], reporting_meters(coverage, month)):
                        # Add
                        rep_h += h
            # Row
            rows.append((split, f"{b['start']} to {b['end']}", total, full, rep_h))
    # Done
    return rows


def registers_from_network(net):
    """
    Each meter's registers read independently from the network file's
    EgaugeMeter records: meter -> {voltage_bus, phase {channel: suffix},
    current_elements [..], current_rating {channel: rating}, extra [..]}.
    The six downloaded phasor channels (L1 L2 L3 S1 S2 S3) are compared
    with graph.json; every other register with an element (neutral CTs,
    further CTs on other elements, derived voltage channels) is listed in
    "extra" so nothing the export left out goes unnoticed.
    """
    # Result
    out = {}
    # Loop over meters
    for rec in net.get("EgaugeMeter", []):
        # Record
        m = {"voltage_bus": None, "phase": {}, "current_elements": [], "current_rating": {}, "extra": []}
        # Registers
        for reg in rec.get("registers", []):
            # Element and suffix
            base, _, suffix = str(reg.get("element") or "").partition(".")
            # Channel is the register value name
            channel = reg.get("value")
            # Registers outside the six downloaded channels, or with no element, are listed and not compared
            if channel not in ("L1", "L2", "L3", "S1", "S2", "S3") or not base:
                # Keep a short description of voltage and current registers only (power registers are derived)
                if reg.get("unit") in ("V", "A"):
                    # Note
                    m["extra"].append(f"{reg.get('name')}={reg.get('value')}@{reg.get('element') or 'no element'}")
                # Next
                continue
            # Voltage
            if reg.get("unit") == "V":
                # Bus
                m["voltage_bus"] = base
                # Phase suffix as written
                m["phase"][channel] = suffix
            # Current
            elif reg.get("unit") == "A":
                # Element
                if base not in m["current_elements"]:
                    # Add
                    m["current_elements"].append(base)
                # Phase letter
                m["phase"][channel] = suffix[:1]
                # Rating
                m["current_rating"][channel] = reg.get("I_rating")
        # Keep
        out[rec["name"]] = m
    # Done
    return out


def meter_mapping(graph, net_file, config):
    """
    Rows: meter, voltage bus, current element, its two end buses, phase
    map, configured sign or unknown, and whether the graph's record
    matches the registers read independently from the network file
    (voltage bus, phase map, current elements, ratings), with the
    mismatch named. Also returns the list of mismatched meters.
    """
    # Network
    with open(net_file) as f:
        # Parse
        net = json.load(f)
    # Independent registers
    regs = registers_from_network(net)
    # Element -> (fbus, tbus names) for every category with fbus and tbus
    ends = {}
    # Loop over categories
    for cat, recs in net.items():
        # Lists of records
        if not isinstance(recs, list):
            # Next
            continue
        # Loop
        for rec in recs:
            # Has ends
            if isinstance(rec, dict) and "fbus" in rec and "name" in rec:
                # tbus may be a list of dicts or a string
                tb = rec.get("tbus")
                # Names
                tnames = [x["name"] if isinstance(x, dict) else x for x in (tb if isinstance(tb, list) else [tb])] if tb is not None else []
                # Store
                ends[rec["name"]] = (cat, rec["fbus"], tnames)
    # Configured signs: node -> element -> sign
    signs = {}
    # Loop
    for node, items in config.get("measured_injections", {}).items():
        # Each element
        for it in items:
            # Store
            signs[it["element"]] = (it["sign"], node)
    # Rows and mismatches
    rows, bad = [], []
    # Loop over meters by number
    for m in sorted(graph["meters"], key=lambda x: int(x.split("_")[1])):
        # Record
        rec = graph["meters"][m]
        # Independent record
        ind = regs.get(m)
        # Compare field by field
        problems = []
        # Missing
        if ind is None:
            # Record
            problems.append("not in the network file")
        else:
            # Voltage bus
            if ind["voltage_bus"] != rec.get("voltage_bus"):
                # Record
                problems.append(f"voltage bus {rec.get('voltage_bus')} vs registers {ind['voltage_bus']}")
            # Phase map
            if ind["phase"] != rec.get("phase"):
                # Record
                problems.append(f"phase map {rec.get('phase')} vs registers {ind['phase']}")
            # Current elements
            if sorted(ind["current_elements"]) != sorted(rec.get("current_elements", [])):
                # Record
                problems.append(f"current elements {rec.get('current_elements')} vs registers {ind['current_elements']}")
            # Ratings
            if ind["current_rating"] != rec.get("current_rating"):
                # Record
                problems.append(f"ratings {rec.get('current_rating')} vs registers {ind['current_rating']}")
        # Keep
        if problems:
            # Record
            bad.append(m)
        # Phase notes to check explicitly: reversed labels and line-to-line voltages
        notes = []
        # Registers the export left out, listed for the record
        if ind and ind["extra"]:
            # Note
            notes.append(f"registers not downloaded: {', '.join(ind['extra'])}")
        # Phase map
        ph = rec.get("phase", {})
        # Voltage or current channel 1 not on phase a: labels follow the network file, not the channel order
        if ph.get("L1") not in (None, "ag", "ab") or ph.get("S1") not in (None, "a"):
            # Note
            notes.append("reversed phase labels")
        # Line-to-line voltage channel
        if any(v in ("ab", "bc", "ca") for v in ph.values()):
            # Note
            notes.append("line-to-line voltage")
        # Each current element
        for el in rec.get("current_elements", []) or [None]:
            # Ends
            cat, fb, tbs = ends.get(el, ("?", "?", []))
            # Sign
            sg = signs.get(el)
            # Row
            rows.append((m, rec.get("voltage_bus"), el, cat, fb, ",".join(tbs), json.dumps(rec.get("phase")), f"{sg[0]:+d} configured in physics_config.json (copied from Caltech's example), node {sg[1][:30]}" if sg else "unknown", "MATCH" if not problems else "MISMATCH: " + "; ".join(problems), ", ".join(notes)))
    # Done
    return rows, bad


def loader_check(graph_path, manifest_path):
    """
    The training code's view of a graph file: node per meter, every file
    edge in both directions, and each directed edge's weight equal to the
    file's admittance_abs divided by the file's maximum. Returns (ok, note)
    or (None, note) when torch or the manifest is missing.
    """
    # Imports here so the rest runs without torch
    try:
        # Model side
        import socal_models as sm
    # Missing
    except ImportError as exc:
        # Skip
        return None, f"import failed: {exc}"
    # Manifest
    if not os.path.exists(manifest_path):
        # Skip
        return None, f"no manifest at {manifest_path}"
    # Read
    with open(manifest_path) as f:
        # Parse
        manifest = json.load(f)
    # Graph
    g = sm.load_graph(graph_path)
    # Node index per meter
    idx = sm.meter_node_index(g, manifest)
    # Must match the file
    for mi, m in enumerate(manifest["meters"]):
        # Compare
        if int(idx[mi]) != int(g["meters"][m]["node"]):
            # Fail
            return False, f"meter_node_index gives node {int(idx[mi])} for {m}, file says {g['meters'][m]['node']}"
    # Edges
    ei, ew = sm.build_edges(g, "true", "admittance", 0)
    # Both directions
    if ei.shape[1] != 2 * len(g["edges"]):
        # Fail
        return False, f"build_edges gives {ei.shape[1]} directed edges for {len(g['edges'])} undirected"
    # Every file edge present both ways
    pairs = {(int(a), int(b)) for a, b in zip(ei[0].tolist(), ei[1].tolist())}
    # Check
    for e in g["edges"]:
        # Both
        if (e["source"], e["target"]) not in pairs or (e["target"], e["source"]) not in pairs:
            # Fail
            return False, f"edge {e['element']} missing a direction"
    # Expected weight per file edge: admittance_abs over the file maximum
    top = max(float(e["admittance_abs"]) for e in g["edges"])
    # Weight per directed pair as built
    built = {(int(a), int(b)): float(w) for a, b, w in zip(ei[0].tolist(), ei[1].tolist(), ew.tolist())}
    # Compare every edge both ways
    for e in g["edges"]:
        # Expected
        want = float(e["admittance_abs"]) / top
        # Both directions
        for key in ((e["source"], e["target"]), (e["target"], e["source"])):
            # Off
            if abs(built[key] - want) > 1e-6:
                # Fail
                return False, f"edge {e['element']} weight {built[key]:.6f} differs from expected {want:.6f}"
    # Done
    return True, f"{len(manifest['meters'])} meters on their file nodes; {ei.shape[1]} directed edges, every file edge both ways, every weight equal to admittance_abs / max ({top:.4f})"


def main():
    """Run the seven parts."""
    # Options
    args = parse_args()
    # Config
    config = sp.load_config(args.config)
    # Caltech modules with paths redirected
    ct = sp.setup_caltech(args.caltech_dir, args.topology_dir)
    # Network folder and its one file
    net_folder = os.path.join(os.path.abspath(args.topology_dir), "network_files", "circuit3")
    # The network file
    net_file = os.path.join(net_folder, sorted(fn for fn in os.listdir(net_folder) if fn.endswith(".json"))[0])
    # Switch records
    records = read_switch_records(os.path.join(args.topology_dir, "parameter_timeseries"))
    # Buses per switch
    buses = switch_buses(net_file)
    # tbus and element per switch file, for the lookup check
    switch_tbus.update({k: v[1] for k, v in buses.items()})
    # Element names
    switch_elem.update({k: v[4] for k, v in buses.items()})
    # Reference graph
    with open(args.graph) as f:
        # Parse
        ref = json.load(f)
    # Reference meter graph
    with open(args.meter_graph) as f:
        # Parse
        ref_mg = json.load(f)
    # Coverage report, optional
    coverage = None
    # Present
    if os.path.exists(args.coverage):
        # Read
        with open(args.coverage) as f:
            # Rows
            coverage = list(csv.DictReader(f))
    # Offset
    off = args.utc_offset_hours
    # Failures and incomplete checks, reported at the end and in the exit code
    failures, incomplete = [], []

    # ---- 7 first, since everything hangs on it: the reference time both ways
    print("=== 7. the reference time ===")
    # As given
    ref_t = pd.Timestamp(config["reference_time"])
    # Both readings
    print(f"  reference_time in physics_config.json: {ref_t} (naive)")
    # As files' time base
    print(f"  read as the files' fixed PST (UTC-{off}): {ref_t} PST = {to_utc(ref_t, off)}")
    # As UTC
    print(f"  read as UTC: {ref_t} UTC = {ref_t - pd.Timedelta(hours=off)} PST")
    # Caltech's loader compares naive timestamps against the switch files, which are in the files' time base
    print("  Caltech's loader compares the time as given against the switch files' timestamps (naive, same base), so the first reading applies")

    # ---- 0. the committed graphs against fresh builds
    print("\n=== 0. committed graphs against fresh builds ===")
    # Temp folder for the exports
    tmp = tempfile.mkdtemp(prefix="topology_audit_")
    # Saved reference time must be the configured one
    if ref.get("reference_time") != config["reference_time"]:
        # Record
        failures.append(f"graph.json reference_time {ref.get('reference_time')} differs from physics_config.json {config['reference_time']}")
        # Say
        print(f"  FAIL: graph.json reference_time {ref.get('reference_time')} differs from the config's {config['reference_time']}")
    else:
        # Say
        print(f"  PASS: graph.json reference_time matches the config ({config['reference_time']})")
    # Fresh build at the reference time
    fresh = graph_at(ct, args.topology_dir, config, config["reference_time"], os.path.join(tmp, "g_ref.json"))
    # Compare by identity
    d0 = compare_graphs(ref, fresh)
    # Any difference
    if d0["nodes_added"] or d0["nodes_removed"] or d0["edges_added"] or d0["edges_removed"] or d0["edges_moved"] or d0["meters_moved"]:
        # Record
        failures.append("graph.json differs from a fresh build at its reference time")
        # Say
        print(f"  FAIL: graph.json differs from a fresh build at its reference time: {d0}")
    else:
        # Say
        print(f"  PASS: graph.json equals a fresh build at its reference time ({len(fresh['nodes'])} nodes, {len(fresh['edges'])} edges, {len(fresh['meters'])} meters)")
    # Node order and index agreement too, since the models use indices
    if [n["name"] for n in ref["nodes"]] != [n["name"] for n in fresh["nodes"]] or {m: r["node"] for m, r in ref["meters"].items()} != {m: r["node"] for m, r in fresh["meters"].items()}:
        # Record
        failures.append("graph.json node order or meter node indices differ from a fresh build")
        # Say
        print("  FAIL: node order or meter node indices differ from the fresh build")
    else:
        # Say
        print("  PASS: node order and meter node indices match the fresh build")
    # Meter graph rebuilt from the committed graph.json
    mg0 = smg.build_meter_graph(ref)
    # Diff by pair
    a0, r0 = meter_graph_diff(ref_mg, mg0)
    # Node order
    same_nodes = [n["meter"] for n in ref_mg["nodes"]] == [n["meter"] for n in mg0["nodes"]]
    # Report
    if a0 or r0 or not same_nodes:
        # Record
        failures.append("graph_meters.json differs from a rebuild from graph.json")
        # Say
        print(f"  FAIL: graph_meters.json differs from a rebuild from graph.json: added {a0}, removed {r0}, node order same {same_nodes}")
    else:
        # Say
        print(f"  PASS: graph_meters.json equals a rebuild from graph.json ({len(mg0['edges'])} edges, same node order)")

    # ---- 1. switch records
    print("\n=== 1. switch records (time in fixed PST and UTC, repeated unchanged records collapsed) ===")
    # All changes after the initial record: (time, switch, old, new, buses)
    changes = []
    # Initial states
    initial = {}
    # Loop over switches
    for switch, recs in sorted(records.items()):
        # Buses
        fb, tb, nominal, cat, elem = buses.get(switch, ("?", "?", "?", "?", switch))
        # Initial
        initial[switch] = (recs[0][0], recs[0][1], nominal)
        # Changes
        for (t0, s0), (t1, s1) in zip(recs[:-1], recs[1:]):
            # Record
            changes.append((t1, switch, s0, s1, (fb, tb)))
    # Sort by time
    changes.sort()
    # Initial state table
    print(f"  {'switch':18s} {'first record (PST)':20s} {'initial':8s} {'nominal':8s} {'joins':30s}")
    # Loop
    for switch, (t, s, nominal) in sorted(initial.items()):
        # Buses
        fb, tb = buses.get(switch, ("?", "?", None, None, None))[:2]
        # Line
        print(f"  {switch:18s} {str(t):20s} {s:8s} {str(nominal):8s} {fb}-{tb}" + ("" if s == nominal else "   <- initial state is not the nominal state"))
    # Changes table
    print(f"\n  {'time PST':20s} {'time UTC':26s} {'gap to previous':16s} {'switch':18s} {'change':9s} {'joins':24s}")
    # Previous time
    prev = None
    # Loop
    for t, switch, s0, s1, (fb, tb) in changes:
        # Gap
        gap = "" if prev is None else str(t - prev)
        # Line
        print(f"  {str(t):20s} {str(to_utc(t, off)):26s} {gap:16s} {switch:18s} {s0}->{s1:5s} {fb}-{tb}")
        # Keep
        prev = t
    # Lookup rule
    print("\n  lookup rule through Caltech's loader, one second before / at / one second after each change:")
    # Old state per change, for the strict comparison
    old_of = {(t, sw): s0 for t, sw, s0, s1, _ in changes}
    # Verify
    for t, switch, before, at, after, expected in verify_lookup(ct, args.topology_dir, changes, net_folder):
        # The recorded old state
        old = old_of[(t, switch)]
        # Strict: before equals the old state, at and after equal the new one
        good = before == old and at == expected and after == expected
        # Record
        if not good:
            # Keep
            failures.append(f"lookup rule at {t} {switch}: before {before} (old {old}), at {at}, after {after} (new {expected})")
        # Line
        print(f"    {str(t):20s} {switch:18s} before {before} (recorded old {old}), at {at}, after {after} (new {expected})" + ("" if good else "   <- UNEXPECTED"))
    # Initial-state rule: a lookup before the first record
    first = min(t for t, *_ in (v[0] for v in records.values()))
    # Try
    try:
        # Load one second before the first record
        ct.topology.load_net_files([net_folder], t=(first - pd.Timedelta(seconds=1)).to_pydatetime(), timeseries_dir=ct.global_param.FILE_PATHS["net_params"])
        # Did not raise: the rule is not the documented one
        print(f"  a lookup before the first record ({first}) returns a state instead of raising")
        # Record
        incomplete.append("initial-state rule: a lookup before the first record returned a state; check which")
    # Raised
    except AssertionError as exc:
        # The documented behavior
        print(f"  a lookup before the first record ({first}) raises: {str(exc)[:90]}; no state is defined before it")

    # ---- 2. periods and graph differences
    print("\n=== 2. topology periods and their differences from the reference graph (by bus set and element name) ===")
    # Record bounds
    start, end = first, pd.Timestamp(args.record_end)
    # Periods
    periods = build_periods(changes, start, end)
    # Differences and meter graphs per period
    diffs, mdiffs, hops_ref = [], [], physical_hops(ref)
    # Reference meter graph pairs for the hop report
    loop_i = 0
    # Loop over periods
    for p in periods:
        # A time inside the period
        mid = p["start"] + (p["end"] - p["start"]) / 2
        # Graph
        g = graph_at(ct, args.topology_dir, config, mid, os.path.join(tmp, f"g_{loop_i}.json"))
        # Differences
        d = compare_graphs(ref, g)
        # Whether anything differs
        d["differs"] = bool(d["nodes_added"] or d["nodes_removed"] or d["edges_added"] or d["edges_removed"] or d["edges_moved"] or d["meters_moved"])
        # Keep
        diffs.append(d)
        # Meter graph for the period
        mg = smg.build_meter_graph(g)
        # Diff
        added, removed = meter_graph_diff(ref_mg, mg)
        # Hops in this period for the changed pairs
        hp = physical_hops(g)
        # Months the period touches, in fixed PST
        months = sorted({mo for mo, _, _ in month_pieces(to_utc(p["start"], off), to_utc(p["end"], off), off)})
        # Keep; the reporting judgement is made per month in the overlay
        mdiffs.append({"added": added, "removed": removed, "hops": hp, "months": months})
        # Print the period
        hours = (p["end"] - p["start"]).total_seconds() / 3600.0
        # Header line
        print(f"\n  period {loop_i}: [{p['start']} PST, {p['end']} PST) = [{to_utc(p['start'], off)}, {to_utc(p['end'], off)}), {hours:.1f} h")
        # Opening changes
        print("    opened by: " + ("record start" if not p["changes"] else "; ".join(f"{c[1]} {c[2]}->{c[3]} ({c[4][0]}-{c[4][1]})" for c in p["changes"])))
        # Differences
        if not d["differs"]:
            # Same
            print("    complete topology: same as the reference graph")
        else:
            # List
            print(f"    nodes added {d['nodes_added']}")
            print(f"    nodes removed {d['nodes_removed']}")
            print(f"    edges added {d['edges_added']}, removed {d['edges_removed']}, moved to other nodes {d['edges_moved']}")
            print(f"    components: {d['n_components']} ({'changed' if d['components_changed'] else 'same bus sets'}); meters on a different node: {d['meters_moved']}")
        # Meter graph
        print(f"    meter graph: {len(mg['edges'])} edges; added {added}; removed {removed}")
        # Hops for changed pairs, both quantities
        for pr in added + removed:
            # Hops now and in the reference
            print(f"      {pr[0]}--{pr[1]}: physical hops {hp.get(pr)} now, {hops_ref.get(pr)} in the reference (0 = shared node)")
        # Reporting, month by month
        if coverage is not None:
            # Loop
            for mo in months:
                # That month's set
                rep = reporting_meters(coverage, mo)
                # Pairs among reporting meters
                ra = [pr for pr in added if pr[0] in rep and pr[1] in rep]
                # Removed
                rr = [pr for pr in removed if pr[0] in rep and pr[1] in rep]
                # Say
                print(f"    among meters with any reading in {mo} (monthly availability): edges added {ra}, removed {rr}")
        # Next
        loop_i += 1

    # ---- 4. overlay on splits
    print("\n=== 4. hours per block in a differing topology ===")
    # Loop over splits files
    for path in args.splits:
        # Missing
        if not os.path.exists(path):
            # Skip
            print(f"  {path}: not found, skipped")
            # Next
            continue
        # Read
        with open(path) as f:
            # Parse
            splits = json.load(f)
        # Header
        print(f"\n  {path}")
        # Column header
        print(f"    {'split':12s} {'block':26s} {'hours':>8s} {'full-topology differs':>22s} {'reporting-meter graph differs':>30s}")
        # Rows
        for split, label, total, full, rep in overlay(periods, splits, off, coverage, diffs, mdiffs):
            # Line
            print(f"    {split:12s} {label:26s} {total:8.1f} {full:22.1f} {rep:30.1f}")
    # Totals over the record, month by month for the reporting judgement
    full_h, rep_h = 0.0, 0.0
    # Loop
    for p, d, m in zip(periods, diffs, mdiffs):
        # Pieces
        for mo, q0, q1 in month_pieces(to_utc(p["start"], off), to_utc(p["end"], off), off):
            # Hours
            h = (q1 - q0).total_seconds() / 3600.0
            # Full
            if d["differs"]:
                # Add
                full_h += h
            # Reporting
            if reporting_differs(m, reporting_meters(coverage, mo)):
                # Add
                rep_h += h
    # Record length
    rec_h = (end - start).total_seconds() / 3600.0
    # Print
    print(f"\n  over the record ({rec_h:.0f} h): complete topology differs from the reference for {full_h:.1f} h ({100 * full_h / rec_h:.2f}%); the meter graph among meters with any reading in the month differs for {rep_h:.1f} h ({100 * rep_h / rec_h:.2f}%)" + ("" if coverage is not None else " [no coverage report: reporting = every meter]"))

    # ---- 5. meter mapping
    print("\n=== 5. meter mapping: graph.json against the registers in the network file (signs are as configured, not verified) ===")
    # Rows
    rows, bad = meter_mapping(ref, net_file, config)
    # Header
    print(f"  {'meter':10s} {'voltage bus':12s} {'current element':16s} {'type':6s} {'fbus':10s} {'tbus':12s} {'phase map':48s} {'registers':10s} notes / sign")
    # Rows
    for m, vb, el, cat, fb, tb, phase, sign, match, notes in rows:
        # Line
        print(f"  {m:10s} {str(vb):12s} {str(el):16s} {cat:6s} {fb:10s} {tb:12s} {phase:48s} {match:10s} {notes + '; ' if notes else ''}{sign}")
    # Mismatches
    if bad:
        # Record
        failures.append(f"meter records in graph.json differ from the network file registers: {bad}")
        # Say
        print(f"  FAIL: {len(bad)} meters differ from the registers: {bad}")
    else:
        # Say
        print(f"  PASS: every meter's voltage bus, phase map, current elements, and ratings match the registers")

    # ---- 6. loader check, both graph files
    print("\n=== 6. loader check ===")
    # Both files
    for path in (args.graph, args.meter_graph):
        # Run
        ok, note = loader_check(path, args.manifest)
        # Record
        if ok is None:
            # Incomplete
            incomplete.append(f"loader check on {path}: {note}")
        elif not ok:
            # Failure
            failures.append(f"loader check on {path}: {note}")
        # Report
        print(f"  {path}: {'SKIP' if ok is None else ('PASS' if ok else 'FAIL')}: {note}")

    # ---- CSV
    if args.csv:
        # Write
        with open(args.csv, "w", newline="") as f:
            # Writer
            w = csv.writer(f)
            # Header
            w.writerow(["period", "start_pst", "end_pst", "start_utc", "end_utc", "hours", "opened_by", "full_topology_differs", "meters_moved", "meter_graph_edges_added", "meter_graph_edges_removed", "reporting_meter_graph_differs_hours"])
            # Rows
            for i, (p, d, m) in enumerate(zip(periods, diffs, mdiffs)):
                # Row
                w.writerow([i, p["start"], p["end"], to_utc(p["start"], off), to_utc(p["end"], off), (p["end"] - p["start"]).total_seconds() / 3600.0,
                            "; ".join(f"{c[1]} {c[2]}->{c[3]}" for c in p["changes"]) or "record start", d["differs"], " ".join(d["meters_moved"]),
                            " ".join(f"{a}-{b}" for a, b in m["added"]), " ".join(f"{a}-{b}" for a, b in m["removed"]),
                            # Hours of the period in which the meter graph among that month's reporting meters differs, same monthly logic as the totals
                            sum((q1 - q0).total_seconds() / 3600.0 for mo, q0, q1 in month_pieces(to_utc(p["start"], off), to_utc(p["end"], off), off) if reporting_differs(m, reporting_meters(coverage, mo)))])
        # Say so
        print(f"\nwrote {args.csv}")

    # ---- verdict
    print("\n=== result ===")
    # Failures
    for f_ in failures:
        # Line
        print(f"  FAIL: {f_}")
    # Incomplete
    for i_ in incomplete:
        # Line
        print(f"  INCOMPLETE: {i_}")
    # Summary
    print(f"  {len(failures)} failed check(s), {len(incomplete)} incomplete; " + ("the committed graphs and mappings passed every check that ran" if not failures else "see the FAIL lines"))
    # Exit code
    sys.exit(1 if failures or incomplete else 0)


# Run when called as a script
if __name__ == "__main__":
    # Go
    main()
