# =============================================================================
# File:        main_metergraphcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-05
#
# Purpose:     One-off check of graph_meters.json, the meter graph written by
#              main_metergraph.py, against graph.json it was built from:
#                1. one node per meter, every meter of graph.json present,
#                   each on its own node, in meter-number order
#                2. every edge joins two different valid nodes, no pair twice,
#                   both ends in the same component, admittance_abs 1.0
#                3. every edge's physical path is real: consecutive path
#                   nodes are joined in graph.json by the named element, the
#                   path ends on the two meters' physical nodes, and every
#                   inner node is unmetered
#                4. expected counts: 22 nodes, 36 edges, and per component
#                   A-side 19, B-side 14, the three-meter component 3,
#                   egauge_24's component 0
#                5. reach: every meter but egauge_24 has at least one
#                   neighbor, so a one-layer GCN reaches a metered neighbor
#                6. the training code loads it: socal_models.load_graph,
#                   meter_node_index, build_edges (true, random, none) and
#                   socal_modeldata.channel_fallback all accept the file
#              Prints PASS, FAIL, or SKIP per check and exits with 1 on any
#              failure or skip, so a skipped loader check never reads as a
#              passed one.
#              Also draws the meter graph with networkx and matplotlib to a
#              PNG next to the file, nodes colored by component.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root.
#
# Usage:
#   python checks/main_metergraphcheck.py
#   python checks/main_metergraphcheck.py --graph graph.json --meter-graph graph_meters.json --png graph_meters.png
#   python checks/main_metergraphcheck.py --manifest data/features/manifest.json
#   python checks/main_metergraphcheck.py --no-draw
#
# Functions:
#   parse_args()            command-line options with defaults
#   report(name, ok, note, skipped)
#                           print one PASS, FAIL, or SKIP line, return ok
#   check_loaders(args, mg) run the training code's loaders on the file
#   draw_graph(mg, path)    draw the meter graph and save a PNG
#   main()                  run the checks, then draw
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON reading
import json
# Standard library: repo root on the import path, exit code
import os
import sys

# The repo root, one level above checks/, so the project modules import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Check graph_meters.json against graph.json")
    # Physical graph
    parser.add_argument("--graph", default="graph.json", help="Physical graph written by main_graph.py")
    # Meter graph
    parser.add_argument("--meter-graph", default="graph_meters.json", help="Meter graph written by main_metergraph.py")
    # Manifest, for the loader check; without it the meter order is graph.json's meters by number
    parser.add_argument("--manifest", default=None, help="features manifest.json, for the real meter and channel order; default uses graph.json's meters")
    # Expected counts
    parser.add_argument("--expect-edges", type=int, default=36, help="Expected edge count")
    parser.add_argument("--expect-isolated", nargs="*", default=["egauge_24"], help="Meters expected to have no edge")
    parser.add_argument("--expect-component-edges", nargs="*", default=["0:14", "1:19", "2:3", "3:0"], help="Expected edges per component as comp:count")
    # Drawing
    parser.add_argument("--png", default="graph_meters.png", help="Where to save the drawing")
    parser.add_argument("--no-draw", action="store_true", help="Skip the drawing")
    # Parse
    return parser.parse_args()


def report(name, ok, note="", skipped=False):
    """Print one PASS, FAIL, or SKIP line and return ok (a skipped check returns False)."""
    # Word
    word = "SKIP" if skipped else ("PASS" if ok else "FAIL")
    # Line
    print(f"{word}  {name}{': ' + note if note else ''}")
    # A skipped check does not count as passed
    return ok and not skipped


def check_loaders(args, mg):
    """Run the training code's graph loaders on the meter graph. Returns (ok, note)."""
    # The loaders need torch; say so if it is missing rather than fail
    try:
        # Model side
        import socal_models as sm
        # Data side
        import socal_modeldata as smd
    # Missing dependency: report as skipped, which fails the run
    except ImportError as exc:
        # Skip
        return None, f"import failed: {exc}"
    # The real manifest when given
    if args.manifest:
        # Read
        with open(args.manifest) as f:
            # Parse
            manifest = json.load(f)
    # Otherwise a stand-in: the meters of graph.json in meter-number order, the six channels
    else:
        # Physical graph
        with open(args.graph) as f:
            # Parse
            g = json.load(f)
        # Stand-in with the keys the loaders read
        manifest = {"meters": sorted(g["meters"], key=lambda m: int(m.split("_")[1])), "channels": ["L1", "L2", "L3", "S1", "S2", "S3"]}
    # Meter list
    meters = manifest["meters"]
    # Load through the model side
    graph = sm.load_graph(args.meter_graph)
    # Node index per meter
    idx = sm.meter_node_index(graph, manifest)
    # Every meter must map to its own node
    if len(set(idx.tolist())) != len(meters):
        # Fail
        return False, "meter_node_index did not give one node per meter"
    # Edge tensors for the three ablation modes
    notes = []
    # Loop
    for mode in ("true", "random", "none"):
        # Build
        edge_index, edge_weight = sm.build_edges(graph, mode, "binary", 0)
        # Record shape
        notes.append(f"{mode} {edge_index.shape[1]} directed edges")
    # True wiring must give both directions of every edge
    if sm.build_edges(graph, "true", "binary", 0)[0].shape[1] != 2 * len(mg["edges"]):
        # Fail
        return False, "build_edges(true) edge count is not twice the file's"
    # Admittance weights must all be 1.0 after scaling
    w = sm.build_edges(graph, "true", "admittance", 0)[1]
    # Check
    if float(w.min()) != 1.0 or float(w.max()) != 1.0:
        # Fail
        return False, "admittance weights are not all 1.0"
    # Channel fallback must find the nominal voltage through the new node
    fallback = smd.channel_fallback(manifest, args.meter_graph, 1.0)
    # Fallback must be a filled array
    if fallback.shape != (len(meters), len(manifest["channels"])):
        # Fail
        return False, f"channel_fallback shape {fallback.shape}"
    # Done
    return True, "; ".join(notes) + f"; channel_fallback ok, voltage fallback egauge_3 L1 = {fallback[meters.index('egauge_3'), 0]:.0f}"


def draw_graph(mg, path):
    """Draw the meter graph colored by component and save a PNG."""
    # Imports here so the checks run without them
    import networkx as nx
    # Plotting without a display
    import matplotlib
    # Backend
    matplotlib.use("Agg")
    # Plot
    import matplotlib.pyplot as plt
    # Graph
    g = nx.Graph()
    # Nodes
    for i, n in enumerate(mg["nodes"]):
        # Add with component
        g.add_node(i, meter=n["meter"], component=n["component"])
    # Edges
    for e in mg["edges"]:
        # Add with hop count
        g.add_edge(e["source"], e["target"], hops=e["hops"])
    # Layout
    pos = nx.spring_layout(g, seed=0)
    # Colors by component
    comps = sorted({n["component"] for n in mg["nodes"]})
    # Color per node
    colors = [comps.index(mg["nodes"][i]["component"]) for i in g.nodes]
    # Figure
    plt.figure(figsize=(9, 7))
    # Draw
    nx.draw_networkx_nodes(g, pos, node_color=colors, cmap="tab10", node_size=500)
    # Edges
    nx.draw_networkx_edges(g, pos, alpha=0.6)
    # Labels
    nx.draw_networkx_labels(g, pos, {i: mg["nodes"][i]["meter"].replace("egauge_", "") for i in g.nodes}, font_size=9)
    # Edge labels with hop counts
    nx.draw_networkx_edge_labels(g, pos, {(e["source"], e["target"]): e["hops"] for e in mg["edges"]}, font_size=7)
    # Title
    plt.title("Meter graph: nodes are meters, edge labels are physical hops")
    # No axes
    plt.axis("off")
    # Save
    plt.savefig(path, dpi=150, bbox_inches="tight")
    # Close
    plt.close()


def main():
    """Run the checks, then draw."""
    # Options
    args = parse_args()
    # Read both files
    with open(args.graph) as f:
        # Physical
        g = json.load(f)
    # Meter graph
    with open(args.meter_graph) as f:
        # Parse
        mg = json.load(f)
    # Every check's result
    ok = True
    # Meter number for sorting
    num = lambda m: int(m.split("_")[1])

    # 1. one node per meter
    meters = sorted(g["meters"], key=num)
    # Node meters in order
    node_meters = [n["meter"] for n in mg["nodes"]]
    # Meter records point at their own node
    own = all(mg["meters"][m]["node"] == i for i, m in enumerate(node_meters))
    # Report
    ok &= report("1. one node per meter, in order", node_meters == meters and own and len(mg["meters"]) == len(meters), f"{len(node_meters)} nodes")

    # 2. edges valid
    n = len(mg["nodes"])
    # Seen pairs
    pairs = set()
    # Problems
    bad = []
    # Loop
    for e in mg["edges"]:
        # Ends
        s, t = e["source"], e["target"]
        # Valid, distinct
        if not (0 <= s < n and 0 <= t < n) or s == t:
            # Record
            bad.append(f"{e['element']} ends")
        # Pair
        key = (min(s, t), max(s, t))
        # Duplicate
        if key in pairs:
            # Record
            bad.append(f"{e['element']} duplicate")
        # Keep
        pairs.add(key)
        # Same component
        if mg["nodes"][s]["component"] != mg["nodes"][t]["component"]:
            # Record
            bad.append(f"{e['element']} crosses components")
        # Binary weight
        if e["admittance_abs"] != 1.0:
            # Record
            bad.append(f"{e['element']} weight {e['admittance_abs']}")
    # Report
    ok &= report("2. edges valid, no duplicates, same component, binary", not bad, "; ".join(bad[:5]))

    # 3. paths real
    metered_nodes = {int(r["node"]) for r in g["meters"].values()}
    # Physical adjacency with element names
    joined = {}
    # Fill
    for pe in g["edges"]:
        # Both directions
        joined[(pe["source"], pe["target"])] = pe["element"]
        joined[(pe["target"], pe["source"])] = pe["element"]
    # Problems
    bad = []
    # Loop
    for e in mg["edges"]:
        # Path
        nodes, elements = e["path_nodes"], e["path_elements"]
        # Physical nodes of the two meters
        a, b = mg["nodes"][e["source"]]["physical_node"], mg["nodes"][e["target"]]["physical_node"]
        # Ends must be the two meters' physical nodes, either order
        if {nodes[0], nodes[-1]} != {a, b} and not (a == b and nodes == [a]):
            # Record
            bad.append(f"{e['element']} ends {nodes[0]},{nodes[-1]} vs {a},{b}")
        # Hop count matches
        if len(elements) != e["hops"] or len(nodes) != e["hops"] + 1:
            # Record
            bad.append(f"{e['element']} hop count")
        # Each step is a real element
        for u, v, el in zip(nodes[:-1], nodes[1:], elements):
            # Must exist with that name
            if joined.get((u, v)) != el:
                # Record
                bad.append(f"{e['element']} step {u}-{v} {el}")
        # Inner nodes unmetered
        for inner in nodes[1:-1]:
            # Must not carry a meter
            if inner in metered_nodes:
                # Record
                bad.append(f"{e['element']} inner node {inner} is metered")
    # Report
    ok &= report("3. every edge's physical path is real and passes only unmetered nodes", not bad, "; ".join(bad[:5]))

    # 4. counts
    per_comp = {}
    # Count edges per component
    for e in mg["edges"]:
        # Tally
        c = mg["nodes"][e["source"]]["component"]
        per_comp[c] = per_comp.get(c, 0) + 1
    # Expected component edge counts, by meter count
    sizes = {}
    # Meters per component
    for nd in mg["nodes"]:
        # Tally
        sizes[nd["component"]] = sizes.get(nd["component"], 0) + 1
    # Expected edges per component, parsed from "comp:count" pairs
    expect_comp = {int(k): int(v) for k, v in (item.split(":") for item in args.expect_component_edges)}
    # Every expected component must match
    comps_ok = all(per_comp.get(c, 0) == v for c, v in expect_comp.items())
    # Note
    note = f"{len(mg['edges'])} edges; per component " + ", ".join(f"{c}: {sizes[c]} meters {per_comp.get(c, 0)} edges" for c in sorted(sizes))
    # Report
    ok &= report("4. expected counts", len(mg["nodes"]) == 22 and len(mg["edges"]) == args.expect_edges and comps_ok, note)

    # 5. reach
    degree = [0] * n
    # Count
    for e in mg["edges"]:
        # Both ends
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    # Isolated
    isolated = [mg["nodes"][i]["meter"] for i in range(n) if degree[i] == 0]
    # Report
    ok &= report("5. every meter but the expected ones has a neighbor one hop away", isolated == args.expect_isolated, f"isolated: {', '.join(isolated) or 'none'}")

    # 6. loaders; None means the imports failed, reported as SKIP and counted as a failure
    good, note = check_loaders(args, mg)
    # Report
    ok &= report("6. training code loads the file", bool(good), note, skipped=good is None)

    # Draw
    if not args.no_draw:
        # Try, since the plotting libraries may be missing
        try:
            # Draw
            draw_graph(mg, args.png)
            # Say
            print(f"drew {args.png}")
        # Missing library
        except ImportError as exc:
            # Say
            print(f"drawing skipped: {exc}")
    # Exit code
    sys.exit(0 if ok else 1)


# Run when called as a script
if __name__ == "__main__":
    # Go
    main()
