# =============================================================================
# File:        main_graph.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-22
#
# Purpose:     Write the grid's electrical graph to graph.json so every
#              learned model loads the same graph from one file instead of
#              rebuilding it from Caltech's topology code. The graph is the
#              one the physics baseline builds (see socal_physics.py), but
#              for the whole grid, every connected component, nothing pruned.
#
#              graph.json is small and is committed to the repo, like
#              splits.json. Rerun this script only when physics_config.json
#              changes the reference time.
#
# Usage:
#   python main_graph.py
#   python main_graph.py --out graph.json --config physics_config.json
#   Options: python main_graph.py --help
#
# Output:
#   graph.json   nodes (buses joined by closed zero-impedance elements),
#                edges (lines and transformers with admittance blocks and
#                one scalar weight), meters (meter -> node index, buses,
#                elements, phase letters). Layout described inside the file.
#
# Functions:
#   parse_args()   command-line options with defaults
#   main()         set up Caltech's code, export the graph, print counts
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON to read back and summarize
import json
# Project helpers
import socal_physics as sp


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser with the file's purpose as description
    parser = argparse.ArgumentParser(description="Write the grid's electrical graph to graph.json")
    # Topology folder
    parser.add_argument("--topology-dir", default="topology", help="Caltech topology folder copied into the repo")
    # Caltech utils folder
    parser.add_argument("--caltech-dir", default="caltech_utils", help="Caltech utils folder copied into the repo")
    # Config file, for the reference time
    parser.add_argument("--config", default="physics_config.json", help="Config holding reference_time")
    # Output path
    parser.add_argument("--out", default="graph.json", help="Where to write the graph")
    # Parse
    return parser.parse_args()


def main():
    """Export the graph and print what it contains."""
    # Options
    args = parse_args()
    # Config
    config = sp.load_config(args.config)
    # Caltech modules with paths redirected
    ct = sp.setup_caltech(args.caltech_dir, args.topology_dir)
    # Export
    n_nodes, n_edges, n_meters = sp.export_graph(ct, args.topology_dir, config, args.out)
    # Read back for the summary
    with open(args.out) as f:
        # Parse
        g = json.load(f)
    # Component sizes
    sizes = {}
    # Count nodes per component
    for node in g["nodes"]:
        # Tally
        sizes[node["component"]] = sizes.get(node["component"], 0) + 1
    # Edge types
    types = {}
    # Count edges per type
    for edge in g["edges"]:
        # Tally
        types[edge["type"]] = types.get(edge["type"], 0) + 1
    # Meters with a node
    placed = sum(1 for m in g["meters"].values() if m["node"] is not None)
    # Report
    print(f"Wrote {args.out} at reference time {g['reference_time']}")
    # Counts
    print(f"  nodes {n_nodes}, edges {n_edges} {types}, meters {n_meters} ({placed} placed on a node)")
    # Components
    print(f"  component sizes, largest first: {[sizes[k] for k in sorted(sizes)]}")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
