# =============================================================================
# File:        socal_metergraph.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-05
#
# Purpose:     Helper functions that turn the physical graph (graph.json,
#              43 nodes, 24 of them without a meter) into a meter graph:
#              one node per meter, and an edge between two meters when they
#              share a physical node or when their physical nodes are joined
#              by a path whose inner nodes are all unmetered. A path stops at
#              the first metered node it reaches, so the meter graph keeps
#              the order of the wiring instead of joining every meter in a
#              component to every other.
#
#              Why: a reach check on graph.json showed that with one GCN
#              layer 13 of the 22 meters receive nothing from any other
#              meter, only the learned constants of unmetered nodes, and the
#              masked-node objective needs every hidden meter to have a
#              metered neighbor one hop away. The meter graph gives that to
#              every meter except egauge_24, which is alone in its component.
#
#              The output keeps the layout of graph.json (nodes, edges,
#              meters) so socal_models.load_graph, meter_node_index and
#              build_edges, and socal_modeldata.channel_fallback read it
#              unchanged. Edge weights are binary (admittance_abs = 1.0);
#              each edge records the physical path it came from, so an
#              electrical weight can be added later without rebuilding.
#
# Functions:
#   adjacency(graph)                   neighbor sets per physical node, with
#                                      the element that joins each pair
#   meters_by_node(graph)              physical node index -> meter names
#   unmetered_paths(graph, start, adj, by_node)
#                                      breadth-first search from one metered
#                                      node through unmetered nodes only;
#                                      returns the metered nodes reached and
#                                      the shortest path to each
#   meter_edges(graph)                 the undirected meter-to-meter edges
#                                      with their path provenance
#   build_meter_graph(graph)           the full meter graph dictionary
#   summarize(meter_graph)             counts per component, isolated meters
# =============================================================================

# Standard library: breadth-first search queue
from collections import deque


def adjacency(graph):
    """Neighbor sets per physical node index, each entry (neighbor, element name, element type)."""
    # One set per node
    adj = {i: set() for i in range(len(graph["nodes"]))}
    # Loop over physical edges
    for e in graph["edges"]:
        # Both directions
        adj[e["source"]].add((e["target"], e["element"], e["type"]))
        # Reverse
        adj[e["target"]].add((e["source"], e["element"], e["type"]))
    # Done
    return adj


def meters_by_node(graph):
    """Physical node index -> sorted list of meter names placed on it."""
    # Lookup
    out = {}
    # Loop over meters
    for meter, rec in graph["meters"].items():
        # Append under its node
        out.setdefault(int(rec["node"]), []).append(meter)
    # Sort each list by meter number for stable output
    for node in out:
        # Sort
        out[node].sort(key=meter_number)
    # Done
    return out


def meter_number(meter):
    """Integer after the underscore in a meter name, for sorting (egauge_3 -> 3)."""
    # Split and convert
    return int(meter.split("_")[1])


def unmetered_paths(graph, start, adj, by_node):
    """
    Breadth-first search from the metered physical node `start` through
    unmetered nodes only. A metered node is recorded when reached and not
    expanded further. Returns {metered node: (path node list, path element list)}
    with the shortest path found to each.
    """
    # Shortest path to each node seen, as (nodes, elements)
    path = {start: ([start], [])}
    # Queue of nodes to expand
    queue = deque([start])
    # Metered nodes reached
    reached = {}
    # Search
    while queue:
        # Next node
        u = queue.popleft()
        # Its neighbors
        for v, element, _ in sorted(adj[u], key=lambda t: (t[0], t[1])):
            # Already seen
            if v in path:
                # Skip
                continue
            # Record the path
            path[v] = (path[u][0] + [v], path[u][1] + [element])
            # A metered node ends the path
            if v in by_node:
                # Record, do not expand
                reached[v] = path[v]
            # An unmetered node is expanded
            else:
                # Queue
                queue.append(v)
    # Done
    return reached


def meter_edges(graph):
    """
    Undirected meter-to-meter edges. Each edge is a dictionary with the two
    meter names (sorted), the physical path (node indices and element names),
    and the hop count (0 for meters sharing a physical node).
    """
    # Adjacency and placement
    adj, by_node = adjacency(graph), meters_by_node(graph)
    # Edges keyed by the sorted meter pair, keeping the shortest path
    edges = {}
    # Meters sharing a physical node
    for node, meters in by_node.items():
        # Every pair on the node
        for i in range(len(meters)):
            # Partner
            for j in range(i + 1, len(meters)):
                # Record with hop 0
                edges[(meters[i], meters[j])] = {"path_nodes": [node], "path_elements": [], "hops": 0}
    # Paths through unmetered nodes
    for node, meters in by_node.items():
        # Search from this node
        reached = unmetered_paths(graph, node, adj, by_node)
        # Loop over metered nodes reached
        for other, (nodes, elements) in reached.items():
            # Every meter here with every meter there
            for a in meters:
                # Partner
                for b in by_node[other]:
                    # Sorted pair
                    key = tuple(sorted((a, b), key=meter_number))
                    # Hop count
                    hops = len(elements)
                    # Keep the shortest path per pair
                    if key not in edges or hops < edges[key]["hops"]:
                        # Record
                        edges[key] = {"path_nodes": nodes, "path_elements": elements, "hops": hops}
    # As a sorted list
    out = []
    # Loop in meter-number order
    for (a, b), rec in sorted(edges.items(), key=lambda kv: (meter_number(kv[0][0]), meter_number(kv[0][1]))):
        # Edge record
        out.append({"meter_a": a, "meter_b": b, **rec})
    # Done
    return out


def build_meter_graph(graph):
    """
    The meter graph in the layout of graph.json. Nodes: one per meter in
    meter-number order, carrying the meter name, its physical node index,
    its voltage bus and that bus's nominal voltage, and the component id.
    Edges: source and target node indices, element "meter_path", type
    "meter_path", admittance_abs 1.0, and the physical path. Meters: the
    records from graph.json with the node index replaced.
    """
    # Meter order
    meters = sorted(graph["meters"], key=meter_number)
    # Node index per meter
    index = {m: i for i, m in enumerate(meters)}
    # Nodes
    nodes = []
    # Loop over meters
    for m in meters:
        # Physical record
        rec = graph["meters"][m]
        # Physical node
        pnode = graph["nodes"][int(rec["node"])]
        # Voltage bus
        bus = rec.get("voltage_bus")
        # Node record
        nodes.append({
            # Name is the meter
            "name": m,
            # The bus the meter's voltages are measured on
            "buses": [bus] if bus else [],
            # Its nominal voltage, read from the physical node
            "nominal_voltage": {bus: pnode.get("nominal_voltage", {}).get(bus)} if bus else {},
            # Component id carried over
            "component": pnode.get("component"),
            # Where this meter sat in graph.json
            "physical_node": int(rec["node"]),
            # The meter name again, explicit
            "meter": m,
        })
    # Edges
    edges = []
    # Loop over meter pairs
    for e in meter_edges(graph):
        # Edge record
        edges.append({
            # Ends
            "source": index[e["meter_a"]],
            "target": index[e["meter_b"]],
            # Labels in the style of graph.json
            "element": f"{e['meter_a']}--{e['meter_b']}",
            "type": "meter_path",
            # Binary weight
            "admittance_abs": 1.0,
            # Provenance
            "hops": e["hops"],
            "path_nodes": e["path_nodes"],
            "path_elements": e["path_elements"],
        })
    # Meter records with the new node index
    meter_recs = {}
    # Loop over meters
    for m in meters:
        # Copy
        rec = dict(graph["meters"][m])
        # New node
        rec["node"] = index[m]
        # Keep the physical node too
        rec["physical_node"] = int(graph["meters"][m]["node"])
        # Store
        meter_recs[m] = rec
    # The graph
    return {
        # Reference time carried over
        "reference_time": graph.get("reference_time"),
        # What this file is
        "note": ("Meter graph built from graph.json by main_metergraph.py: one node per meter; an edge "
                 "between two meters that share a physical node or whose physical nodes are joined by "
                 "a path with only unmetered nodes in between (the search stops at the first metered "
                 "node). admittance_abs is 1.0 on every edge (binary). Each edge records its physical "
                 "path. Node index order is the nodes list order."),
        # Where it came from
        "source_graph_nodes": len(graph["nodes"]),
        "source_graph_edges": len(graph["edges"]),
        # Contents
        "nodes": nodes,
        "edges": edges,
        "meters": meter_recs,
    }


def summarize(meter_graph):
    """Per-component meter and edge counts, degree per meter, and the isolated meters."""
    # Degree per node
    degree = [0] * len(meter_graph["nodes"])
    # Count
    for e in meter_graph["edges"]:
        # Both ends
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    # Per component
    comps = {}
    # Loop over nodes
    for i, n in enumerate(meter_graph["nodes"]):
        # Component record
        c = comps.setdefault(n["component"], {"meters": [], "edges": 0})
        # Meter
        c["meters"].append(n["meter"])
    # Edges per component (both ends share a component by construction)
    for e in meter_graph["edges"]:
        # Count under the source's component
        comps[meter_graph["nodes"][e["source"]]["component"]]["edges"] += 1
    # Isolated meters
    isolated = [n["meter"] for i, n in enumerate(meter_graph["nodes"]) if degree[i] == 0]
    # Done
    return {"components": comps, "degree": {n["meter"]: degree[i] for i, n in enumerate(meter_graph["nodes"])}, "isolated": isolated}
