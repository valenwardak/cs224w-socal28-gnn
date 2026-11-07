# =============================================================================
# File:        main_metergraphs.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-07
#
# Purpose:     Write graph_meters_periods.json, the time-indexed meter
#              graph: the same 22 meter nodes in the same order throughout,
#              with one edge set per distinct wiring over the record. The
#              topology audit (checks/main_topologyaudit.py) showed that
#              graph.json is the nominal wiring and that the switch records
#              change it: building transfers in September and November to
#              December 2023, the A-side feed change on November 13, 2024,
#              and the open breakers of egauge_3 and egauge_4 before
#              February 1, 2024. This file lets a model use the wiring in
#              force at each window's last history slot (the pre-target
#              rule, applied by the data side, not here).
#
#              How it is built: the switch records are read, the record is
#              cut into half-open periods at every group of simultaneous
#              changes, the meter graph is built with Caltech's topology
#              code and socal_metergraph inside each period (at its
#              midpoint), and the distinct edge sets are kept. Graph 0 is
#              the graph at the reference time in physics_config.json and
#              must equal graph_meters.json, the fixed baseline. Graphs are
#              distinct by edge set. Per period the file stores the rest of
#              what changes with the topology: each meter's merged bus set
#              and component, read from the physical graph. What does not
#              change is stored once: the meter nodes, their order,
#              registers, and ratings. graph.json, graph_meters.json, and
#              physics_config.json are read and never written.
#
# Usage:
#   python main_metergraphs.py
#   python main_metergraphs.py --out graph_meters_periods.json --record-end 2025-09-04T00:00:00
#   Options: python main_metergraphs.py --help
#
# Output:
#   graph_meters_periods.json with:
#     reference_time, utc_offset_hours, record_start, record_end   as in the switch files' time base
#     nodes, meters            the 22 meter nodes and meter records, from graph 0, fixed throughout
#     graphs                   list; each has edges (graph.json edge layout, binary weight, physical path), built_at
#     periods                  list in time order; each has start, end (files' time base),
#                              start_utc, end_utc_exclusive (seconds since 1970), changes that opened it,
#                              graph (index into graphs), and topology (per meter: node_buses, component)
#
# Functions:
#   parse_args()   command-line options with defaults
#   main()         read the records, build the periods and graphs, write the file, print a summary
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON
import json
# Standard library: paths and temporary files
import os
import tempfile
# Third party: timestamps
import pandas as pd
# Project: Caltech setup and the graph export
import socal_physics as sp
# Project: meter graph helpers
import socal_metergraph as smg


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Write the time-indexed meter graph graph_meters_periods.json")
    # Inputs
    parser.add_argument("--topology-dir", default="topology", help="Caltech's topology folder in the repo")
    parser.add_argument("--caltech-dir", default="caltech_utils", help="Caltech's utils folder in the repo")
    parser.add_argument("--config", default="physics_config.json", help="Physics config with the reference time")
    parser.add_argument("--meter-graph", default="graph_meters.json", help="The fixed meter graph, which graph 0 must equal")
    parser.add_argument("--physical-graph", default="graph.json", help="The fixed physical graph, protected from being written over")
    # Record bounds in the files' time base
    parser.add_argument("--record-end", default="2025-09-04T00:00:00", help="End of the record, fixed PST, exclusive")
    # Time base
    parser.add_argument("--utc-offset-hours", type=int, default=8, help="Hours added to the files' standard time to get UTC")
    # Output
    parser.add_argument("--out", default="graph_meters_periods.json", help="Where to write the time-indexed meter graph")
    # Parse
    args = parser.parse_args()
    # Never write over an input
    for path in (args.meter_graph, args.physical_graph, args.config):
        # Same file
        if os.path.abspath(path) == os.path.abspath(args.out) or (os.path.exists(args.out) and os.path.samefile(path, args.out)):
            # Refuse
            parser.error(f"--out must not be an input file: {args.out}")
    # Done
    return args


def main():
    """Read the records, build the periods and graphs, write the file, print a summary."""
    # Options
    args = parse_args()
    # Config and Caltech
    config = sp.load_config(args.config)
    # Modules with paths redirected
    ct = sp.setup_caltech(args.caltech_dir, args.topology_dir)
    # Switch records
    records = smg.read_switch_records(os.path.join(args.topology_dir, "parameter_timeseries"))
    # Record start: the first record of any switch
    start = min(recs[0][0] for recs in records.values())
    # End
    end = pd.Timestamp(args.record_end)
    # Periods
    periods = smg.build_periods(records, start, end)
    # Temporary folder for the exports
    tmp = tempfile.mkdtemp(prefix="metergraphs_")
    # Counter for file names
    count = [0]

    # The meter graph in force at a naive time, through the same export the fixed graph used
    def build_fn(t):
        # Config with the time replaced
        cfg = dict(config, reference_time=pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S"))
        # Path
        path = os.path.join(tmp, f"g_{count[0]}.json")
        # Next
        count[0] += 1
        # Export the physical graph
        sp.export_graph(ct, args.topology_dir, cfg, path)
        # Read it back
        with open(path) as f:
            # Parse
            physical = json.load(f)
        # Meter graph and the physical graph it came from
        return smg.build_meter_graph(physical), physical

    # Graphs and the period index
    graphs, index, topology = smg.build_period_graphs(build_fn, periods, config["reference_time"])
    # Graph 0 must equal the fixed meter graph
    with open(args.meter_graph) as f:
        # Parse
        fixed = json.load(f)
    # Compare edge sets and node order
    if smg.graph_signature(fixed) != graphs[0]["signature"] or [n["meter"] for n in fixed["nodes"]] != [n["meter"] for n in graphs[0]["nodes"]]:
        # Raise
        raise ValueError(f"graph 0 built at {config['reference_time']} differs from {args.meter_graph}")
    # Offset
    off = args.utc_offset_hours
    # UTC seconds for a naive time
    utc_s = lambda t: int((pd.Timestamp(t) + pd.Timedelta(hours=off)).tz_localize("UTC").timestamp())
    # Invariant meter records: drop the reference-only physical node index
    meters = {m: {k: v for k, v in rec.items() if k != "physical_node"} for m, rec in fixed["meters"].items()}
    # File body
    body = {
        # Time base
        "note": ("Time-indexed meter graph. nodes and meters are fixed throughout (the 22 meters in meter-number order, "
                 "as in graph_meters.json). graphs holds one edge set per distinct wiring; graph 0 is the reference "
                 "(nominal) wiring and equals graph_meters.json. periods are half-open in time; each names its graph "
                 "and carries each meter's merged bus set and component in that period. "
                 "Times in start and end are in the switch files' fixed-PST base; start_utc and end_utc_exclusive are "
                 "seconds since 1970 UTC. The wiring in force at a time is the period holding it. Which graph a forecast "
                 "window uses is decided by the data side (the graph at its last history slot)."),
        "reference_time": config["reference_time"], "utc_offset_hours": off,
        "record_start": str(start), "record_end": str(end),
        # Fixed parts, from graph 0
        "nodes": [{"name": n["name"], "meter": n["meter"], "nominal_voltage": n["nominal_voltage"]} for n in graphs[0]["nodes"]],
        "meters": meters,
        # Changing parts
        "graphs": [{"index": i, "built_at": g["built_at"], "n_edges": len(g["edges"]), "edges": g["edges"]} for i, g in enumerate(graphs)],
        # Periods
        "periods": [{"index": i, "start": str(p["start"]), "end": str(p["end"]), "start_utc": utc_s(p["start"]), "end_utc_exclusive": utc_s(p["end"]),
                     "changes": [{"switch": sw, "from": s0, "to": s1} for sw, s0, s1 in p["changes"]], "graph": index[i], "topology": topology[i]} for i, p in enumerate(periods)],
    }
    # Write
    with open(args.out, "w") as f:
        # Dump
        json.dump(body, f, indent=1)
    # Summary
    print(f"wrote {args.out}: {len(graphs)} distinct meter graphs over {len(periods)} periods, {start} to {end} (files' time base)")
    # Per graph
    for g in body["graphs"]:
        # Periods using it
        using = [p["index"] for p in body["periods"] if p["graph"] == g["index"]]
        # Hours
        hours = sum((pd.Timestamp(p["end"]) - pd.Timestamp(p["start"])).total_seconds() / 3600.0 for p in body["periods"] if p["graph"] == g["index"])
        # Line
        print(f"  graph {g['index']}: {g['n_edges']} edges, built at {g['built_at']}, used by periods {using}, {hours:.1f} h")
    # Per period
    for p in body["periods"]:
        # Line
        print(f"  period {p['index']:2d}: [{p['start']}, {p['end']}) graph {p['graph']}, opened by " + ("record start" if not p["changes"] else "; ".join(f"{c['switch']} {c['from']}->{c['to']}" for c in p["changes"])))


# Run when called as a script
if __name__ == "__main__":
    # Go
    main()
