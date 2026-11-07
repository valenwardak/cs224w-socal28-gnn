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
# Time-indexed meter graphs (graph_meters_periods.json, added 2026-11-07
#              after the topology audit): the switch records in
#              topology/parameter_timeseries/ change the wiring over the
#              record, and one snapshot is wrong for the transferred
#              periods and for the months before February 2024 when the
#              buses of egauge_3 and egauge_4 were open. The functions
#              below cut the record into half-open periods at the switch
#              changes, build the meter graph inside each period, keep the
#              distinct edge sets with graph 0 reserved for the verified
#              reference graph, and record per period what else changes
#              with the topology (each meter's merged bus set and
#              component, read from the physical graph) apart from what
#              does not (the 22 meter nodes, their order, registers, and
#              ratings). Graphs are distinct by edge set only.
#              period_index_at maps a time to the period in force (not to a
#              graph index; the period names its graph), for the pre-target
#              rule applied by the data side: a window's graph is the one
#              in force at its last history slot.
#
# Functions:
#   read_switch_records(ts_dir)        every switch's state records, collapsed
#   build_periods(changes, start, end) half-open periods between change groups
#   graph_signature(meter_graph)       the edge set as a sorted tuple of meter pairs
#   topology_meta(meter_graph, physical)
#                                      per-meter merged bus set and component, from the physical graph
#   build_period_graphs(build_fn, periods, reference_time)
#                                      distinct meter graphs over the periods, graph 0 the reference,
#                                      with the topology meta of every period
#   period_index_at(periods, t_utc)    index of the period in force at a UTC second
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
# Standard library: CSV switch records
import csv
# Standard library: paths
import os
# Third party: timestamps
import pandas as pd


def read_switch_records(ts_dir):
    """
    Every switch's records from <switch>-tbus_status.csv in ts_dir, as a
    dict switch -> list of (naive timestamp, state) with repeated unchanged
    records collapsed. Times are in the files' own time base (fixed PST).
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


def build_periods(records, start, end):
    """
    Half-open topology periods [t0, t1) in the files' time base from
    `start` to `end`, cut at every distinct change time strictly inside
    the record. Simultaneous changes form one group. Each period carries
    the changes that opened it as (switch, old, new); the first is empty.
    """
    # Bounds in order
    if not end > start:
        # Raise
        raise ValueError(f"record end {end} is not after record start {start}")
    # Every change
    changes = []
    # Loop over switches
    for switch, recs in sorted(records.items()):
        # Pairs
        for (t0, s0), (t1, s1) in zip(recs[:-1], recs[1:]):
            # Keep
            changes.append((t1, switch, s0, s1))
    # Distinct times inside the record
    times = sorted({t for t, *_ in changes if start < t < end})
    # Bounds
    bounds = [start] + times + [end]
    # Periods
    out = []
    # Loop
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        # Group at t0
        group = [(sw, s0, s1) for t, sw, s0, s1 in changes if t == t0]
        # Keep
        out.append({"start": t0, "end": t1, "changes": group})
    # Done
    return out


def graph_signature(meter_graph):
    """The edge set of a meter graph as a sorted tuple of (meter, meter) name pairs: equal signatures mean the same wiring between meters."""
    # Names
    names = [n["meter"] for n in meter_graph["nodes"]]
    # Pairs
    return tuple(sorted(tuple(sorted((names[e["source"]], names[e["target"]]))) for e in meter_graph["edges"]))


def topology_meta(meter_graph, physical):
    """
    What changes with the topology, per meter: the full bus set merged into
    the meter's physical node and that node's component id, both read from
    the physical graph the meter graph was built from. Generated node
    indices are not stored: they are not stable between builds.
    """
    # Per meter
    out = {}
    # Loop over meter nodes
    for n in meter_graph["nodes"]:
        # The physical node this meter sat on
        pnode = physical["nodes"][n["physical_node"]]
        # Record
        out[n["meter"]] = {"node_buses": sorted(pnode["buses"]), "component": pnode.get("component")}
    # Done
    return out


def build_period_graphs(build_fn, periods, reference_time):
    """
    Distinct meter graphs over the periods. build_fn(t) returns the pair
    (meter graph, physical graph) in force at naive time t. Graph 0 is the
    graph at reference_time, the verified reference; every period's graph
    is built at the period's midpoint and kept only when its edge set
    differs from every graph already kept. Returns (graphs, period_index,
    period_topology): graphs is a list of {"signature", "edges", "built_at",
    "nodes"}, period_index[i] is the graph index of period i, and
    period_topology[i] is topology_meta for period i. Every graph must
    have the same nodes in the same order as graph 0, else this raises.
    """
    # Reference graph
    ref, ref_physical = build_fn(pd.Timestamp(reference_time))
    # Kept graphs
    graphs = [{"signature": graph_signature(ref), "edges": ref["edges"], "built_at": str(pd.Timestamp(reference_time)), "nodes": ref["nodes"]}]
    # Period index and topology
    index, topology = [], []
    # Loop
    for p in periods:
        # Midpoint
        mid = p["start"] + (p["end"] - p["start"]) / 2
        # Build
        g, physical = build_fn(mid)
        # Nodes must match graph 0
        if [n["meter"] for n in g["nodes"]] != [n["meter"] for n in ref["nodes"]]:
            # Raise
            raise ValueError(f"meter graph at {mid} has different nodes from the reference")
        # Signature
        sig = graph_signature(g)
        # Known by edge set
        known = next((i for i, k in enumerate(graphs) if k["signature"] == sig), None)
        # New
        if known is None:
            # Keep
            graphs.append({"signature": sig, "edges": g["edges"], "built_at": str(mid), "nodes": g["nodes"]})
            # Index
            known = len(graphs) - 1
        # Record
        index.append(known)
        # Topology of this period
        topology.append(topology_meta(g, physical))
    # Done
    return graphs, index, topology


def period_index_at(periods, t_utc):
    """
    Index of the PERIOD in force at UTC second(s) t_utc, from a periods
    list whose entries carry start_utc and end_utc_exclusive as seconds.
    This is not a graph index: the graph is periods[index]["graph"].
    Half-open: a period holds t when start_utc <= t < end_utc_exclusive.
    Raises when a time lies outside every period. Accepts a scalar or a
    numpy array and returns the same shape.
    """
    # Numpy for the vector case
    import numpy as np
    # Bounds
    starts = np.array([p["start_utc"] for p in periods], dtype=np.int64)
    # Ends
    ends = np.array([p["end_utc_exclusive"] for p in periods], dtype=np.int64)
    # Times
    t = np.asarray(t_utc, dtype=np.int64)
    # Period index: last start <= t
    idx = np.searchsorted(starts, t, side="right") - 1
    # Outside: before the first start or at or after the matching end
    bad = (idx < 0) | (t >= ends[np.clip(idx, 0, len(ends) - 1)])
    # Raise with the first offender
    if np.any(bad):
        # Raise
        raise ValueError(f"time {int(np.asarray(t).reshape(-1)[np.argmax(np.asarray(bad).reshape(-1))])} s UTC lies outside every topology period")
    # Done
    return idx


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
