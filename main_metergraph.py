# =============================================================================
# File:        main_metergraph.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-05
#
# Purpose:     Write graph_meters.json, the meter graph built from graph.json:
#              one node per meter, an edge between two meters that share a
#              physical node or are joined by a path of unmetered nodes only.
#              Built for the masked-node objective, where every hidden meter
#              needs a metered neighbor one hop away. graph.json is read and
#              never changed. The helpers are in socal_metergraph.py.
#
#              graph_meters.json keeps the layout of graph.json, so it can be
#              passed to main_gcn.py with --graph graph_meters.json. Like
#              graph.json it is small and is committed to the repo.
#
# Usage:
#   python main_metergraph.py
#   python main_metergraph.py --graph graph.json --out graph_meters.json
#   Options: python main_metergraph.py --help
#
# Output:
#   graph_meters.json   nodes (one per meter), edges (meter pairs with the
#                       physical path each came from, binary weight), meters
#                       (records from graph.json with the new node index).
#
# Functions:
#   parse_args()   command-line options with defaults
#   main()         refuse to overwrite the input, read graph.json, build the
#                  meter graph, write it, print counts
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON read and write
import json
# Standard library: path comparison, so the input is never overwritten
import os
# Project helpers
import socal_metergraph as smg


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser with the file's purpose as description
    parser = argparse.ArgumentParser(description="Write the meter graph graph_meters.json from graph.json")
    # Input graph
    parser.add_argument("--graph", default="graph.json", help="Physical graph written by main_graph.py")
    # Output path
    parser.add_argument("--out", default="graph_meters.json", help="Where to write the meter graph")
    # Parse
    return parser.parse_args()


def main():
    """Build the meter graph and print what it contains."""
    # Options
    args = parse_args()
    # Never write over the input: refuse when both paths name the same file,
    # by path, or by identity when the output already exists (catches symlinks)
    same = os.path.abspath(args.graph) == os.path.abspath(args.out)
    # Identity check, only possible when the output exists
    if not same and os.path.exists(args.out):
        # Same inode
        same = os.path.samefile(args.graph, args.out)
    # Refuse
    if same:
        # Raise with both paths
        raise ValueError(f"--out must not be the input file: --graph {args.graph}, --out {args.out}")
    # Read the physical graph
    with open(args.graph) as f:
        # Parse
        graph = json.load(f)
    # Build
    meter_graph = smg.build_meter_graph(graph)
    # Write, readable
    with open(args.out, "w") as f:
        # Dump
        json.dump(meter_graph, f, indent=1)
    # Summary
    summary = smg.summarize(meter_graph)
    # Header line
    print(f"wrote {args.out}: {len(meter_graph['nodes'])} nodes (meters), {len(meter_graph['edges'])} edges, from {args.graph} with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
    # Per component
    for comp, rec in sorted(summary["components"].items()):
        # One line
        print(f"  component {comp}: {len(rec['meters'])} meters, {rec['edges']} edges: {', '.join(rec['meters'])}")
    # Degrees
    print("  degree per meter: " + ", ".join(f"{m} {d}" for m, d in summary["degree"].items()))
    # Isolated
    print(f"  isolated meters (no edge, excluded from a masked loss): {', '.join(summary['isolated']) or 'none'}")
    # Hop histogram
    hops = {}
    # Count
    for e in meter_graph["edges"]:
        # Tally
        hops[e["hops"]] = hops.get(e["hops"], 0) + 1
    # Print
    print("  edges by physical hops: " + ", ".join(f"{h} hops: {n}" for h, n in sorted(hops.items())))


# Run when called as a script
if __name__ == "__main__":
    # Go
    main()
