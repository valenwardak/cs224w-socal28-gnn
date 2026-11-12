# =============================================================================
# File:        main_islandcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Read-only trace of why a meter is isolated in the meter
#              graph, from the source evidence in the repository: the
#              electrical graph (graph.json), the network file, and the
#              switch status files. For the meter it names (egauge_24 by
#              default) it prints the meter's node and component, every
#              other node of that component, every element in the network
#              file that touches the component's buses, the status and
#              nominal status of each switching element among them, whether
#              that element has a status time series, and the other meters
#              that would be reached if each normally open element were
#              closed. It then says in plain words what the files can and
#              cannot settle. It infers nothing, adds no edge and writes no
#              file.
#
# Usage (from the repo root):
#   python checks/main_islandcheck.py
#   python checks/main_islandcheck.py --meter egauge_24
#
# Functions:
#   parse_args()                       command-line options with defaults
#   touching_elements(net, buses)      network elements with a bus in the set
#   element_status(e)                  (status, nominal status) of a switching element, or None
#   reach_if_closed(graph, node, extra_pairs)
#                                      meters reachable from the node when the given bus pairs are joined
#   main()                             print the trace
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON
import json
# Standard library: paths
import os
# Third party: graph search
import networkx as nx

# Element kinds that are switching devices in the network file
SWITCH_KINDS = ("Switch", "CB", "SwitchMultiPosition", "VFI")
# Element kinds with two ends
TWO_ENDED = ("Line", "Transformer", "Switch", "CB", "SwitchMultiPosition", "VFI")
# Element kinds with one end
ONE_ENDED = ("Load", "Generator", "Inverter", "PV", "FuelCell")


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Read-only trace of a meter's isolation in the meter graph from the source files")
    # Meter
    parser.add_argument("--meter", default="egauge_24", help="Meter to trace")
    # Files
    parser.add_argument("--graph", default="graph.json", help="Electrical graph at the reference time")
    # Network file
    parser.add_argument("--network", default=os.path.join("topology", "network_files", "circuit3", "2023-08-01T00h00m00.000000s.json"), help="Network file")
    # Switch records
    parser.add_argument("--topology-dir", default="topology", help="Caltech's topology folder")
    # Parse
    return parser.parse_args()


def touching_elements(net, buses):
    """Every element of the network file with an end on one of the buses: list of (kind, record)."""
    # Result
    out = []
    # Kinds
    for kind in TWO_ENDED + ONE_ENDED:
        # Records
        for e in net.get(kind, []):
            # Ends
            ends = [e.get("fbus"), e.get("bus")] + [t.get("name") for t in e.get("tbus", [])]
            # Touching
            if any(b in buses for b in ends if b):
                # Keep
                out.append((kind, e))
    # Done
    return out


def element_status(e):
    """(status, nominal status) of a switching element from its tbus record, or None when it has none."""
    # First end record
    t = e.get("tbus", [{}])[0] if e.get("tbus") else {}
    # Status fields
    if "status" in t or "nominal_status" in t:
        # Pair
        return t.get("status"), t.get("nominal_status")
    # None
    return None


def reach_if_closed(graph, node, extra_pairs):
    """Meters reachable from the node in the electrical graph when the given (bus, bus) pairs are additionally joined."""
    # Bus to node
    where = {b: i for i, n in enumerate(graph["nodes"]) for b in n["buses"]}
    # Graph
    G = nx.Graph()
    # Nodes
    G.add_nodes_from(range(len(graph["nodes"])))
    # Edges
    G.add_edges_from((e["source"], e["target"]) for e in graph["edges"])
    # Extra joins, only when both buses are in the graph
    for a, b in extra_pairs:
        # Known
        if a in where and b in where:
            # Join
            G.add_edge(where[a], where[b])
    # Component
    comp = nx.node_connected_component(G, node)
    # Meters there
    return sorted((m for m, r in graph["meters"].items() if r["node"] in comp), key=lambda s: int(s.split("_")[1]))


def main():
    """Print the trace."""
    # Options
    args = parse_args()
    # Files
    with open(args.graph) as f:
        # Graph
        graph = json.load(f)
    # Network
    with open(args.network) as f:
        # Net
        net = json.load(f)
    # Status files
    ts_dir = os.path.join(args.topology_dir, "parameter_timeseries")
    # Switches with time series
    with_series = {fn[:-len("-tbus_status.csv")] for fn in os.listdir(ts_dir) if fn.endswith("-tbus_status.csv")}
    # Meter record
    rec = graph["meters"].get(args.meter)
    # Must exist
    if rec is None:
        # Stop
        raise SystemExit(f"{args.meter} is not in {args.graph}")
    # Node and component
    node = int(rec["node"])
    # Component id
    cid = graph["nodes"][node]["component"]
    # Nodes of the component
    comp_nodes = [i for i, n in enumerate(graph["nodes"]) if n["component"] == cid]
    # Buses
    buses = {b for i in comp_nodes for b in graph["nodes"][i]["buses"]}
    # Meters in the component
    meters_here = sorted((m for m, r in graph["meters"].items() if graph["nodes"][r["node"]]["component"] == cid), key=lambda s: int(s.split("_")[1]))
    # Header
    print(f"{args.meter}: voltage bus {rec.get('voltage_bus')}, current elements {rec.get('current_elements')}, node {node}, component {cid}")
    # Component
    print(f"component {cid}: {len(comp_nodes)} node(s), buses {sorted(buses)}, meters {meters_here}")
    # Graph edges inside the component
    print("electrical-graph edges in the component: " + ", ".join(f"{e['element']} ({e['type']}) {e['source']}-{e['target']}" for e in graph["edges"] if e["source"] in comp_nodes or e["target"] in comp_nodes))
    # Elements touching the buses
    print("\nnetwork elements touching the component's buses:")
    # Candidates for an outward link
    outward = []
    # Loop
    for kind, e in touching_elements(net, buses):
        # Ends
        ends = [e.get("fbus"), e.get("bus")] + [t.get("name") for t in e.get("tbus", [])]
        # Clean
        ends = [b for b in ends if b]
        # Outside buses
        outside = [b for b in ends if b not in buses]
        # Status
        st = element_status(e) if kind in SWITCH_KINDS else None
        # Series
        series = "time series" if e.get("name") in with_series else "no time series"
        # Line
        print(f"  {kind:<20} {e.get('name'):<12} ends {ends}" + (f"  status {st[0]} nominal {st[1]}  {series}" if st else "") + ("  leaves the component" if outside else ""))
        # Outward link through a switching element
        if outside and kind in SWITCH_KINDS:
            # Keep
            outward.append((e.get("name"), ends, st, e.get("name") in with_series))
    # What closing each outward element would reach
    print("\nif an outward switching element were closed, the meters reachable from this meter would be:")
    # Loop
    for name, ends, st, has_series in outward:
        # Pairs of its ends
        pairs = [(ends[0], b) for b in ends[1:]]
        # Reach
        print(f"  {name} (status {st[0] if st else '?'}, {'time series' if has_series else 'no time series'}): {reach_if_closed(graph, node, pairs)}")
    # Plain words
    print("\nwhat the files settle: the component's isolation at the reference time follows the network file's switch statuses listed above; an element with a time series is tracked over the record, one without is held at the listed status for the whole record.")
    # Limits
    print("what the files cannot settle: whether an element without a time series was in fact in its listed status during the record; the meter's own readings cannot decide a switch status by correlation alone, since similar voltages can arise on either side of an open tie fed from the same substation. Resolving it needs operator records or a direct measurement across the element. No edge is added by this check.")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
