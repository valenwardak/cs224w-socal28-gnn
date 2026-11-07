# =============================================================================
# File:        main_metergraphscheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-07
#
# Purpose:     One-off check of graph_meters_periods.json, the time-indexed
#              meter graph written by main_metergraphs.py, before any model
#              reads it:
#                1. fixed parts: the 22 meter nodes and their order equal
#                   graph_meters.json, the meter records equal, graph 0's
#                   edge set equals graph_meters.json, the graphs are
#                   pairwise distinct in edge set, and every period carries
#                   topology meta (merged bus set and component) for every
#                   meter
#                2. periods: half-open, contiguous from the record start to
#                   the record end with no gap or overlap, UTC bounds equal
#                   the fixed-PST bounds plus the offset, every graph index
#                   valid, and every switch change time from the source
#                   records opens exactly one period that lists every switch
#                   changing at that time (simultaneous changes grouped)
#                3. lookup: period_index_at gives the period holding a time
#                   (found independently by scanning the bounds),
#                   the earlier period one second before each boundary,
#                   raises before the record start and at or after the end,
#                   and agrees between scalar and array calls
#                4. the pre-target rule on the slot grid: for every switch
#                   time s and the horizon, the graph used for a window is
#                   the one in force at its last history slot H = target
#                   minus horizon slots. The expected period for H is found
#                   independently by scanning the period bounds (so closely
#                   spaced switches are handled), and the lookup must match
#                   it. The first target whose H is at or after s is the
#                   first to use the wiring opened at s. Printed for the
#                   targets around each switch
#                5. blocks: for every block of every splits file, which
#                   graphs are in force over it and for how many hours;
#                   blocks expected to sit in one period (both calibration
#                   blocks, the design block, the quiet block, and the
#                   September 20 to 23, 2023 train block) must show exactly
#                   one graph, whichever it is; nothing assumes graph 0
#                6. month boundaries: the first and last slot of every month
#                   in the record resolves to a period
#                7. with --rebuild, every period's graph is rebuilt with
#                   Caltech's code at the period midpoint; its edge set must
#                   equal the graph the file assigns to that period and its
#                   topology meta must equal the period's
#              Prints PASS or FAIL per check and exits with 1 on any failure.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root.
#
# Usage:
#   python checks/main_metergraphscheck.py
#   python checks/main_metergraphscheck.py --rebuild
#   python checks/main_metergraphscheck.py --horizon 1 --splits splits.json splits_v2.json
#
# Functions:
#   parse_args()            command-line options with defaults
#   report(name, ok, note)  print one PASS or FAIL line, return ok
#   slot(t)                 the slot grid time at or below t, seconds
#   holding(periods, t)     index of the period with start_utc <= t < end, by scan
#   main()                  run the checks
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON
import json
# Standard library: paths, exit, temporary files
import os
import sys
import tempfile
# Third party: numerics and timestamps
import numpy as np
import pandas as pd

# The repo root, one level above checks/, so the project modules import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project: meter graph helpers
import socal_metergraph as smg

# Seconds per slot
SLOT = 10


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Check graph_meters_periods.json before a model reads it")
    # Files
    parser.add_argument("--periods-graph", default="graph_meters_periods.json", help="The time-indexed meter graph")
    parser.add_argument("--meter-graph", default="graph_meters.json", help="The fixed meter graph graph 0 must equal")
    parser.add_argument("--topology-dir", default="topology", help="Caltech's topology folder")
    parser.add_argument("--caltech-dir", default="caltech_utils", help="Caltech's utils folder")
    parser.add_argument("--config", default="physics_config.json", help="Physics config")
    parser.add_argument("--splits", nargs="+", default=["splits.json", "splits_v2.json"], help="Splits files to overlay")
    # Window
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in slots, as the models use")
    # Rebuild
    parser.add_argument("--rebuild", action="store_true", help="Rebuild every period's graph with Caltech's code and compare")
    # Parse
    return parser.parse_args()


def report(name, ok, note=""):
    """Print one PASS or FAIL line and return ok."""
    # Line
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + note if note else ''}")
    # Pass through
    return ok


def slot(t):
    """The slot grid time at or below t, in seconds."""
    # Floor
    return (int(t) // SLOT) * SLOT


def holding(periods, t):
    """Index of the period with start_utc <= t < end_utc_exclusive, by a plain scan, or None."""
    # Scan
    for i, p in enumerate(periods):
        # Inside
        if p["start_utc"] <= t < p["end_utc_exclusive"]:
            # Found
            return i
    # None
    return None


def main():
    """Run the checks."""
    # Options
    args = parse_args()
    # Files
    with open(args.periods_graph) as f:
        # Parse
        pg = json.load(f)
    # Fixed
    with open(args.meter_graph) as f:
        # Parse
        fixed = json.load(f)
    # Every check's result
    ok = True
    # Short names
    graphs, periods, off = pg["graphs"], pg["periods"], int(pg["utc_offset_hours"])

    # 1. fixed parts
    same_nodes = [n["meter"] for n in pg["nodes"]] == [n["meter"] for n in fixed["nodes"]]
    # Meters: equal to graph_meters.json once its reference-only physical_node field is dropped, and that field must be absent
    stripped = {m: {k: v for k, v in rec.items() if k != "physical_node"} for m, rec in fixed["meters"].items()}
    # Compare
    same_meters = pg["meters"] == stripped and not any("physical_node" in rec for rec in pg["meters"].values())
    # Graph 0 edges: rebuild a signature from the file's edges with the file's node order
    def sig(edges):
        # Names
        names = [n["meter"] for n in pg["nodes"]]
        # Pairs
        return tuple(sorted(tuple(sorted((names[e["source"]], names[e["target"]]))) for e in edges))
    # Compare with the fixed graph
    g0 = sig(graphs[0]["edges"]) == smg.graph_signature(fixed)
    # Report
    ok &= report("1a. nodes, order, and meter records (without physical_node) equal graph_meters.json", same_nodes and same_meters, f"{len(pg['nodes'])} nodes")
    # Graph 0
    ok &= report("1b. graph 0 edge set equals graph_meters.json", g0, f"{graphs[0]['n_edges']} edges")
    # Pairwise distinct by edge set
    keys = [sig(g["edges"]) for g in graphs]
    # Report
    ok &= report("1c. graphs pairwise distinct in edge set", len(set(keys)) == len(keys), f"{len(graphs)} graphs")
    # Every edge valid
    bad = [g["index"] for g in graphs for e in g["edges"] if not (0 <= e["source"] < len(pg["nodes"]) and 0 <= e["target"] < len(pg["nodes"]) and e["source"] != e["target"])]
    # Report
    ok &= report("1d. every edge joins two different valid nodes", not bad, f"bad graphs {bad}" if bad else "")

    # Topology meta per period, every meter, with bus set and component
    meter_names = [n["meter"] for n in pg["nodes"]]
    # Check
    topo_ok = all(sorted(p["topology"].keys()) == sorted(meter_names) and all(isinstance(v.get("node_buses"), list) and v["node_buses"] and "component" in v for v in p["topology"].values()) for p in periods)
    # Report
    ok &= report("1e. every period carries node_buses and component for every meter", topo_ok)
    # 2. periods
    starts = [pd.Timestamp(p["start"]) for p in periods]
    # Ends
    ends = [pd.Timestamp(p["end"]) for p in periods]
    # Contiguous
    contiguous = all(ends[i] == starts[i + 1] for i in range(len(periods) - 1)) and starts[0] == pd.Timestamp(pg["record_start"]) and ends[-1] == pd.Timestamp(pg["record_end"]) and all(e > s for s, e in zip(starts, ends))
    # Report
    ok &= report("2a. periods half-open and contiguous from record start to record end, no gap or overlap", contiguous, f"{len(periods)} periods")
    # UTC bounds
    utc_ok = all(p["start_utc"] == int((s + pd.Timedelta(hours=off)).tz_localize("UTC").timestamp()) and p["end_utc_exclusive"] == int((e + pd.Timedelta(hours=off)).tz_localize("UTC").timestamp()) for p, s, e in zip(periods, starts, ends))
    # Report
    ok &= report("2b. UTC bounds equal fixed-PST bounds plus the offset", utc_ok)
    # Graph indices
    ok &= report("2c. every period names a valid graph", all(0 <= p["graph"] < len(graphs) for p in periods))
    # Simultaneous changes from the source records
    records = smg.read_switch_records(os.path.join(args.topology_dir, "parameter_timeseries"))
    # Change time -> set of switches
    by_time = {}
    # Loop
    for sw, recs in records.items():
        # Pairs
        for (t0, s0), (t1, s1) in zip(recs[:-1], recs[1:]):
            # Add
            by_time.setdefault(t1, set()).add(sw)
    # Each inside the record must open exactly one period with the same switch set
    problems = []
    # Loop
    for t, sws in sorted(by_time.items()):
        # Inside
        if not (starts[0] < t < ends[-1]):
            # Next
            continue
        # Periods opened at t
        opened = [p for p, s in zip(periods, starts) if s == t]
        # Exactly one with the same switches
        if len(opened) != 1 or {c["switch"] for c in opened[0]["changes"]} != sws:
            # Record
            problems.append(str(t))
    # Report
    ok &= report("2d. every change time opens exactly one period listing every switch changing then", not problems, f"problems at {problems}" if problems else f"{len(by_time)} change times")

    # 3. lookup
    lookup_ok = True
    # Notes
    notes = []
    # Boundaries, each compared with the independent scan
    for i, p in enumerate(periods):
        # Times to test: the start, one second before, one second after, one second before the end
        for t in (p["start_utc"], p["start_utc"] - 1, p["start_utc"] + 1, p["end_utc_exclusive"] - 1):
            # Expected by scan
            want = holding(periods, t)
            # Outside the record: skip here, tested below
            if want is None:
                # Next
                continue
            # Lookup
            if int(smg.period_index_at(periods, t)) != want:
                # Fail
                lookup_ok = False; notes.append(f"t={t} near period {i}")
    # Before the record
    try:
        # Must raise
        smg.period_index_at(periods, periods[0]["start_utc"] - 1)
        # Did not
        lookup_ok = False; notes.append("no raise before the record start")
    except ValueError:
        # Good
        pass
    # At the record end
    try:
        # Must raise
        smg.period_index_at(periods, periods[-1]["end_utc_exclusive"])
        # Did not
        lookup_ok = False; notes.append("no raise at the record end")
    except ValueError:
        # Good
        pass
    # Array agrees with scalars
    sample = np.array([p["start_utc"] + 5 for p in periods] + [p["end_utc_exclusive"] - 1 for p in periods])
    # Compare
    if list(smg.period_index_at(periods, sample)) != [int(smg.period_index_at(periods, int(t))) for t in sample]:
        # Fail
        lookup_ok = False; notes.append("array and scalar lookups differ")
    # Report
    ok &= report("3. lookup equals the scanned period at every boundary, raises outside, array equals scalar", lookup_ok, "; ".join(notes[:5]))

    # 4. the pre-target rule around every switch, on the slot grid
    print(f"\n  pre-target rule, horizon {args.horizon} slot(s): a window's graph is the period in force at its last history slot H = target - {args.horizon * SLOT} s")
    # Header
    print(f"  {'switch time (UTC)':22s} {'target slot (UTC)':22s} {'last history slot':22s} {'period':>7s} {'graph':>6s}  note")
    # Rule check
    rule_ok = True
    # Loop over periods opened by a change
    for pi_new, p in enumerate(periods[1:], start=1):
        # Switch time in seconds
        s = p["start_utc"]
        # Slot holding the switch
        base = slot(s)
        # First target expected to use the wiring opened at s
        first_new = None
        # Targets from two slots before to three slots after
        for k in range(-2, 4):
            # Target
            T = base + k * SLOT
            # Last history slot
            H = T - args.horizon * SLOT
            # Expected period: the one holding H, found by scan
            want = holding(periods, H)
            # Outside the record (a switch close to the record start): skip
            if want is None:
                # Next
                continue
            # Lookup
            got = int(smg.period_index_at(periods, H))
            # Must agree
            good = got == want
            # Also: H before s must give an earlier period than this one, H at or after s this one or a later one
            good &= (want < pi_new) if H < s else (want >= pi_new)
            # First new
            if H >= s and first_new is None:
                # Record
                first_new = T
            # Record
            rule_ok &= good
            # Note
            note = "old wiring" if H < s else ("first target with the new wiring" if T == first_new else "new wiring")
            # Line
            print(f"  {str(pd.Timestamp(s, unit='s', tz='UTC')):22s} {str(pd.Timestamp(T, unit='s', tz='UTC')):22s} {str(pd.Timestamp(H, unit='s', tz='UTC')):22s} {got:7d} {periods[got]['graph']:6d}  {note}" + ("" if good else "   <- WRONG"))
    # Report
    ok &= report("4. pre-target rule: lookup at each last history slot equals the scanned period, old wiring before the switch, new from the first slot at or after it", rule_ok)

    # 5. blocks
    print()
    # Expected single-period blocks, by their fixed-PST start date
    single = {"2024-01-08", "2025-02-03", "2023-11-01", "2025-04-01", "2023-09-20"}
    # Loop over splits files
    for path in args.splits:
        # Missing
        if not os.path.exists(path):
            # Say
            print(f"  {path}: not found, skipped")
            # Next
            continue
        # Read
        with open(path) as f:
            # Parse
            splits = json.load(f)
        # Header
        print(f"  {path}: graphs in force per block (hours)")
        # Loop
        for split in ("train", "validation", "test_events", "test_normal"):
            # Blocks
            for b in splits.get(split, []):
                # Bounds in seconds
                b0, b1 = int(pd.Timestamp(b["start_utc"]).timestamp()), int(pd.Timestamp(b["end_utc_exclusive"]).timestamp())
                # Hours per graph
                hours = {}
                # Loop over periods
                for p in periods:
                    # Overlap
                    o0, o1 = max(b0, p["start_utc"]), min(b1, p["end_utc_exclusive"])
                    # Any
                    if o0 < o1:
                        # Add
                        hours[p["graph"]] = hours.get(p["graph"], 0.0) + (o1 - o0) / 3600.0
                # Text
                text = ", ".join(f"graph {g}: {h:.1f} h" for g, h in sorted(hours.items()))
                # Single-period expectation
                must_single = b["start"] in single
                # Check
                good = (len(hours) == 1) if must_single else True
                # Record
                ok &= good
                # Line
                print(f"    {split:12s} {b['start']} to {b['end']}: {text}" + ("   (must be one graph)" if must_single else "") + ("" if good else "   <- FAIL"))
    # Report
    ok &= report("5. blocks expected in one period show exactly one graph", ok)

    # 6. month boundaries
    month_ok = True
    # First month start
    cur = pd.Timestamp(pg["record_start"]).to_period("M")
    # Last
    last = (pd.Timestamp(pg["record_end"]) - pd.Timedelta(seconds=1)).to_period("M")
    # Loop
    while cur <= last:
        # Month bounds in UTC seconds, clipped to the record
        m0 = max(int((cur.start_time + pd.Timedelta(hours=off)).tz_localize("UTC").timestamp()), periods[0]["start_utc"])
        # End
        m1 = min(int((cur.end_time.floor("s") + pd.Timedelta(hours=off)).tz_localize("UTC").timestamp()), periods[-1]["end_utc_exclusive"] - 1)
        # Resolve both
        try:
            # First and last slot
            smg.period_index_at(periods, np.array([slot(m0), slot(m1)]))
        except ValueError:
            # Fail
            month_ok = False
        # Next month
        cur += 1
    # Report
    ok &= report("6. first and last slot of every month resolves to a period", month_ok)

    # 7. rebuild
    if args.rebuild:
        # Caltech
        import socal_physics as sp
        # Config
        config = sp.load_config(args.config)
        # Modules
        ct = sp.setup_caltech(args.caltech_dir, args.topology_dir)
        # Temp
        tmp = tempfile.mkdtemp(prefix="metergraphs_check_")
        # Problems
        problems = []
        # Loop over periods
        for p in periods:
            # Midpoint in the files' time base
            mid = pd.Timestamp(p["start"]) + (pd.Timestamp(p["end"]) - pd.Timestamp(p["start"])) / 2
            # Export
            cfg = dict(config, reference_time=mid.strftime("%Y-%m-%dT%H:%M:%S"))
            # Path
            path = os.path.join(tmp, f"g_{p['index']}.json")
            # Build
            sp.export_graph(ct, args.topology_dir, cfg, path)
            # Read
            with open(path) as f:
                # Parse
                physical = json.load(f)
            # Meter graph
            mg = smg.build_meter_graph(physical)
            # Compare with the assigned graph and the period's topology
            g = graphs[p["graph"]]
            # Edge set and topology
            if smg.graph_signature(mg) != sig(g["edges"]) or smg.topology_meta(mg, physical) != p["topology"]:
                # Record
                problems.append(p["index"])
        # Report
        ok &= report("7. every period's graph equals a fresh build at its midpoint", not problems, f"periods differing: {problems}" if problems else f"{len(periods)} rebuilt")
    else:
        # Skipped
        print("SKIP  7. rebuild check (run with --rebuild)")
    # Exit code
    sys.exit(0 if ok else 1)


# Run when called as a script
if __name__ == "__main__":
    # Go
    main()
