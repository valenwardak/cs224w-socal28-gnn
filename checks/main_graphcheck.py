# =============================================================================
# File:        main_graphcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-22
#
# Purpose:     One-off check of graph.json, the electrical graph every GNN
#              loads. Confirms the file is internally consistent and agrees
#              with the physics baseline's view of the grid:
#                1. every edge joins two valid node indices, not the same one
#                2. every edge has a positive admittance and a 6x6 block
#                3. every meter sits on a node that contains its voltage bus
#                4. the A-side nodes named in physics_config.json all exist
#                5. component sizes, largest first, for a glance
#              Prints PASS or FAIL per check and exits with 1 on any failure.
#              Also draws the graph with networkx and matplotlib and saves it
#              as a PNG next to graph.json: nodes colored by connected
#              component, metered nodes labeled with their meter names,
#              lines drawn solid and transformers dashed.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root.
#
# Usage:
#   python checks/main_graphcheck.py
#   python checks/main_graphcheck.py --graph graph.json --config physics_config.json --png graph.png
#   python checks/main_graphcheck.py --no-draw
#
# Functions:
#   parse_args()            command-line options with defaults
#   report(name, ok, note)  print one PASS or FAIL line, return ok
#   draw_graph(g, path)     draw the graph and save a PNG
#   main()                  run the checks, then draw
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON reading
import json
# Standard library: exit code
import sys
# Third party: graph library, to rebuild the graph for drawing
import networkx as nx


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Check graph.json for consistency")
    # Graph file
    parser.add_argument("--graph", default="graph.json", help="Graph file written by main_graph.py")
    # Config with the A-side node names
    parser.add_argument("--config", default="physics_config.json", help="Physics config, for the A-side node names")
    # Picture path
    parser.add_argument("--png", default="graph.png", help="Where to save the drawing")
    # Skip the drawing
    parser.add_argument("--no-draw", action="store_true", help="Run the checks only, no picture")
    # Parse
    return parser.parse_args()


def report(name, ok, note=""):
    """Print one line for a check and hand back its result."""
    # Status word
    status = "PASS" if ok else "FAIL"
    # Print
    print(f"{status}  {name}" + (f"  ({note})" if note else ""))
    # Return
    return ok


def draw_graph(g, path):
    """
    Rebuild the graph in networkx from graph.json and save a picture.
    Nodes are colored by connected component. Metered nodes carry their
    meter names. Lines are solid, transformers dashed. Node labels show
    the first bus of each node so the picture stays readable.
    """
    # Import here so the checks run even without matplotlib
    import matplotlib
    # Draw to a file, no window
    matplotlib.use("Agg")
    # Plotting
    import matplotlib.pyplot as plt
    # Empty graph
    G = nx.Graph()
    # Nodes with their component
    for i, node in enumerate(g["nodes"]):
        # Add
        G.add_node(i, component=node["component"], first_bus=node["buses"][0])
    # Edges with their type
    for e in g["edges"]:
        # Add
        G.add_edge(e["source"], e["target"], kind=e["type"])
    # Meter names per node
    meters_at = {}
    # Loop over meters
    for name, m in g["meters"].items():
        # Skip unplaced
        if m["node"] is None:
            # Next
            continue
        # Short name, egauge_7 -> e7
        meters_at.setdefault(m["node"], []).append(name.replace("egauge_", "e"))
    # Layout: each component on its own, then placed side by side so the halves do not overlap
    pos = {}
    # Components, largest first
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    # Horizontal offset for the next component
    x_offset = 0.0
    # Loop over components
    for comp in comps:
        # Its own spring layout, fixed seed for the same picture each run
        sub_pos = nx.spring_layout(G.subgraph(comp), seed=0, k=1.2)
        # Width scales with size so small ones stay compact
        width = max(1.0, len(comp) ** 0.5)
        # Place
        for n, (x, y) in sub_pos.items():
            # Shift right by the running offset
            pos[n] = (x * width + x_offset, y * width)
        # Advance past this component with a gap
        x_offset += 2 * width + 1.0
    # Figure
    plt.figure(figsize=(16, 12))
    # One color per component
    colors = [G.nodes[i]["component"] for i in G.nodes]
    # Node size: bigger when metered
    sizes = [900 if i in meters_at else 350 for i in G.nodes]
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.tab10, node_size=sizes, linewidths=0.5, edgecolors="black")
    # Edges by type
    lines = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "Line"]
    # Transformers
    transformers = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "Transformer"]
    # Draw lines solid
    nx.draw_networkx_edges(G, pos, edgelist=lines, style="solid", width=1.5)
    # Draw transformers dashed
    nx.draw_networkx_edges(G, pos, edgelist=transformers, style="dashed", width=1.5)
    # Labels: first bus, plus meters when present
    labels = {i: G.nodes[i]["first_bus"].replace("bus_", "b") + ("\n" + ",".join(meters_at[i]) if i in meters_at else "") for i in G.nodes}
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)
    # Title
    plt.title(f"Electrical graph at {g['reference_time']}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges. Solid = line, dashed = transformer. Big nodes carry meters. Color = component.")
    # No axes
    plt.axis("off")
    # Save
    plt.savefig(path, dpi=150, bbox_inches="tight")
    # Close
    plt.close()


def main():
    """Run every check, draw the picture, and exit with 1 if any check failed."""
    # Options
    args = parse_args()
    # Graph
    with open(args.graph) as f:
        # Parse
        g = json.load(f)
    # Config
    with open(args.config) as f:
        # Parse
        config = json.load(f)
    # Pieces
    nodes, edges, meters = g["nodes"], g["edges"], g["meters"]
    # Number of nodes
    n = len(nodes)
    # Node names in index order
    names = [node["name"] for node in nodes]
    # Collect results
    results = []
    # Check 1: edge endpoints
    bad_ends = [e["element"] for e in edges if not (0 <= e["source"] < n and 0 <= e["target"] < n) or e["source"] == e["target"]]
    # Report
    results.append(report("edges join two different valid nodes", not bad_ends, f"{len(edges)} edges" if not bad_ends else f"bad: {bad_ends}"))
    # Check 2: admittance and block shape
    bad_adm = [e["element"] for e in edges if not (e["admittance_abs"] > 0) or len(e["Y_real"]) != 6 or any(len(r) != 6 for r in e["Y_real"])]
    # Report
    results.append(report("edges have positive admittance and a 6x6 block", not bad_adm, "" if not bad_adm else f"bad: {bad_adm}"))
    # Check 3: meter placement
    bad_meters = []
    # Loop over meters
    for name, m in meters.items():
        # Must have a node and a bus
        if m["node"] is None or m["voltage_bus"] is None:
            # Record
            bad_meters.append(name)
            # Next
            continue
        # The node must contain the bus
        if m["voltage_bus"] not in nodes[m["node"]]["buses"]:
            # Record
            bad_meters.append(name)
    # Report
    results.append(report("every meter sits on a node containing its voltage bus", not bad_meters, f"{len(meters)} meters" if not bad_meters else f"bad: {bad_meters}"))
    # Check 4: A-side nodes from the config exist
    wanted = list(config.get("measured_injections", {}).keys()) + list(config.get("zero_injection_nodes", [])) + list(config.get("prune_nodes", []))
    # Missing ones
    missing = [w for w in wanted if w not in names]
    # Report
    results.append(report("A-side nodes from physics_config.json exist in the graph", not missing, f"{len(wanted)} named" if not missing else f"missing: {[m[:40] for m in missing]}"))
    # Check 5: component sizes, informational
    sizes = {}
    # Tally
    for node in nodes:
        # Count
        sizes[node["component"]] = sizes.get(node["component"], 0) + 1
    # Ordered list
    ordered = [sizes[k] for k in sorted(sizes)]
    # Report, always passes, the numbers are for the eye
    results.append(report("component sizes, largest first", True, str(ordered)))
    # Summary
    print(f"\n{sum(results)} of {len(results)} checks passed")
    # Picture unless skipped
    if not args.no_draw:
        # Draw
        draw_graph(g, args.png)
        # Say where
        print(f"Drawing saved to {args.png}")
    # Exit code
    sys.exit(0 if all(results) else 1)


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
