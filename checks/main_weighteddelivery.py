# =============================================================================
# File:        checks/main_weighteddelivery.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Check that the model really receives the electrical edge
#              weights of the weighted meter graph, before any model is
#              trained on it.
#
#              Why it exists: the course staff's milestone feedback asked
#              whether the model uses the graph. The weighted graph file
#              was built and tested by itself. That says nothing about
#              what the model does with it. The model code could read the
#              file and still pass messages with weight 1.0 on every edge,
#              or pick the wrong edge set for a period, and a training run
#              would look normal. A result from such a run would be
#              reported as "electrical weights" while being the old graph.
#              This check closes that gap. It builds no weights and trains
#              nothing. It only follows the weights from the file into the
#              model's message passing.
#
#              The cases:
#                I   Identity. The weighted file is the reviewed one (its
#                    identity equals the expected one), the identity the
#                    training script would record for it is that same
#                    identity, it differs from the uniform graph's, and
#                    the uniform graph is still the accepted file.
#                P   Periods. The model's period table sends each of the
#                    periods to the edge set the file names. For every
#                    change of topology, the windows before and after the
#                    change get the edge set of their own period, found
#                    here by a separate lookup.
#                E   Edge sets. For every edge set, the model holds
#                    exactly the file's connections, in both directions,
#                    and each directed edge carries the file's stored
#                    weight (within the declared tolerance for the
#                    conversion to 32-bit numbers). The weights are not
#                    all 1.0.
#                N   Normalization. For every edge set, the coefficients
#                    the GCN layer applies equal the ones computed
#                    separately from the file: once through the layer's
#                    own normalization function, and once through the
#                    layer itself, fed one node at a time.
#                B   Binary and uniform. A model built with binary
#                    weights on the weighted file, and a model built on
#                    the uniform graph, hold the same connections as the
#                    weighted model and different weights.
#                F   Forecasts. With identical learned parameters and
#                    nonzero test heads, the weighted model's learned
#                    correction differs from the binary model's and from
#                    the uniform-graph model's, for every edge set. A
#                    positive control shows the correction responds at
#                    all. A negative control shows that the binary model
#                    on the weighted file and the model on the uniform
#                    graph agree exactly, so the difference comes from the
#                    weights and from nothing else.
#                S   No silent substitute. The training script's default
#                    for --edge-weights is "admittance", the setting that
#                    reads the stored weights.
#
#              The untrained model's heads start at zero in this project,
#              so its forecast would ignore the graph entirely. As in
#              checks/main_graphdelivery.py, the heads get nonzero test
#              weights, the inputs are all-valid synthetic inputs, and the
#              quantity compared is the learned correction (forecast minus
#              the persistence base).
#
#              checks/main_graphdelivery.py, the accepted check of the
#              uniform graph, is not changed. This file imports its
#              helpers and adds the weighted cases beside it.
#
#              What this check does not show: that a finished training run
#              recorded the weighted graph in its saved options. That
#              needs a run, and no run is authorized yet.
#
# Usage (from the repo root, where torch and torch_geometric are installed):
#   python checks/main_weighteddelivery.py --features-dir /content/features
#   Options: python checks/main_weighteddelivery.py --help
#
# Exit code:   0 when every case passes, 1 otherwise.
#
# Functions:
#   parse_args()                              command-line options with defaults
#   stored_weights(graph)                     directed (source, target) -> stored weight of one edge set
#   dense_coefficients(n, graph)              the GCN coefficient matrix computed from the file, without torch
#   layer_matrix(conv, n, ei, ew)             the matrix the model's GCN layer applies, read through the layer
#   copy_parameters(src, dst)                 give dst the learned parameters of src
#   main()                                    run every case, print the tally, exit code
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: paths
import os
# Standard library: exit code and import path
import sys
# Third party: numerics
import numpy as np
# Third party: tensors
import torch

# This folder, so the accepted delivery check can be imported for its helpers
HERE = os.path.dirname(os.path.abspath(__file__))
# The repo root, one level above checks/
ROOT = os.path.dirname(HERE)
# Import from the root first
sys.path.insert(0, ROOT)
# Then from this folder
sys.path.insert(1, HERE)

# Project: model, graph loading, dataset
import socal_models as sm
# Project: feature count and manifest
import socal_modeldata as sd
# Project: the independent coefficient calculation (NumPy only)
import socal_electricalweights as ew
# Project: the training script, for the identity it records and its defaults
import main_gnn
# Project: helpers of the accepted uniform-graph delivery check
import main_graphdelivery as gd


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Check that the model receives the electrical edge weights of the weighted meter graph")
    # The weighted graph
    parser.add_argument("--graph", default="graph_meters_periods_pu.json", help="Weighted time-indexed meter graph")
    # The uniform graph
    parser.add_argument("--uniform-graph", default="graph_meters_periods.json", help="Uniform time-indexed meter graph, the current one")
    # Meter and channel order
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with manifest.json (meter and channel order)")
    # Expected identity of the weighted graph
    parser.add_argument("--expected-graph", default="graph_meters_periods_pu.json:836420b19ea7e9d4", help="Identity the weighted graph file must have: the reviewed file")
    # Expected identity of the uniform graph
    parser.add_argument("--expected-uniform", default="graph_meters_periods.json:39977b2b42a72afd", help="Identity the uniform graph file must have: the accepted file")
    # Stored weight against delivered weight
    parser.add_argument("--weight-tolerance", type=float, default=1e-6, help="Largest absolute difference between a stored weight or coefficient and the model's 32-bit copy of it")
    # Forecast comparisons
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Largest absolute change of a learned correction that counts as unchanged")
    # History
    parser.add_argument("--history", type=int, default=60, help="History slots, as trained")
    # Gap rule
    parser.add_argument("--max-gap", type=int, default=12, help="Gap rule of the window dataset, as trained")
    # Seed
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random weights and inputs")
    # Done
    return parser.parse_args()


def stored_weights(graph):
    """The stored weight of every directed edge of one edge set of the file: a dictionary from (source, target) to admittance_abs, both directions of every edge."""
    # Outputs
    out = {}
    # Each edge
    for e in graph["edges"]:
        # One direction
        out[(int(e["source"]), int(e["target"]))] = float(e["admittance_abs"])
        # The other direction
        out[(int(e["target"]), int(e["source"]))] = float(e["admittance_abs"])
    # Done
    return out


def dense_coefficients(n, graph):
    """
    The matrix of GCN coefficients for one edge set, computed from the
    file's stored weights without torch: entry (i, i) is the self-loop
    coefficient 1 / d_i, entry (i, j) is w_ij / sqrt(d_i d_j) for an edge
    and 0 otherwise, with d_i = 1 + the sum of the weights of node i's
    edges. Returns a NumPy array (n, n).
    """
    # Node pairs
    pairs = [(int(e["source"]), int(e["target"])) for e in graph["edges"]]
    # Coefficients from the independent calculation
    _, self_c, nbr = ew.gcn_coefficients(n, pairs, [float(e["admittance_abs"]) for e in graph["edges"]])
    # Self-loop coefficients on the diagonal
    N = np.diag(self_c)
    # Neighbor coefficients
    for (i, j), c in zip(pairs, nbr):
        # Both directions
        N[i, j] = N[j, i] = c
    # Done
    return N


def layer_matrix(conv, n, ei, ew_tensor):
    """
    The matrix the model's own GCN layer applies with these edges and
    weights, read through the layer: its linear map is set to the identity
    and its bias to zero, and it is fed the identity matrix, one node
    switched on per column. The output entry (i, j) is then the
    coefficient with which node j's state reaches node i. The layer's
    input and output size must be n. Returns a tensor (n, n).
    """
    # No gradients, and the layer is changed only for this reading
    with torch.no_grad():
        # Linear map: identity
        conv.lin.weight.copy_(torch.eye(n))
        # Bias: zero
        if conv.bias is not None:
            # Zero
            conv.bias.zero_()
        # One node switched on per column
        return conv(torch.eye(n), ei, edge_weight=ew_tensor)


def copy_parameters(src, dst):
    """Give dst the learned parameters of src, name by name. Buffers, which hold the graph, are not touched. Stops when the two models do not have the same parameters."""
    # No gradients
    with torch.no_grad():
        # Parameter by parameter
        for (name, p_src), (name_d, p_dst) in zip(src.named_parameters(), dst.named_parameters()):
            # The names must line up
            if name != name_d:
                # Stop
                raise SystemExit(f"parameter order differs between the two models: {name} against {name_d}")
            # Copy
            p_dst.copy_(p_src)


def main():
    """Run every case and return 0 when all pass."""
    # Options
    args = parse_args()
    # Seed
    torch.manual_seed(args.seed)
    # Counters
    passed, total = 0, 0

    def check(name, ok, note=""):
        """Count one case and print its PASS or FAIL line."""
        # Counters of main
        nonlocal passed, total
        # Count
        total += 1
        # Count a pass
        passed += int(bool(ok))
        # Print through the accepted check's reporter
        gd.report(name, ok, note)

    # Manifest: meter and channel order
    manifest = sd.load_manifest(args.features_dir)
    # The weighted graph
    body = sm.load_graph(args.graph)
    # The uniform graph
    body_u = sm.load_graph(args.uniform_graph)
    # Validated timelines
    timeline, timeline_u = sm.load_timeline(body), sm.load_timeline(body_u)
    # Sizes
    M, C, F = len(manifest["meters"]), len(manifest["channels"]), sd.n_features()
    # Number of nodes and of edge sets
    n, G = len(body["nodes"]), len(body["graphs"])
    # Periods of the file
    periods = timeline["periods"]

    # ---- I: identity ----
    print("\nI  Identity of the files")
    # The identity the training script records for the weighted file
    recorded = main_gnn.graph_identity(args.graph)
    # It must be the reviewed file
    check("the weighted graph is the reviewed file, and this is the identity a training run would record for it", recorded == args.expected_graph, f"{recorded}, expected {args.expected_graph}")
    # The uniform graph must be the accepted file
    recorded_u = main_gnn.graph_identity(args.uniform_graph)
    # Judge
    check("the uniform graph is still the accepted file", recorded_u == args.expected_uniform, f"{recorded_u}, expected {args.expected_uniform}")
    # The two identities differ, so a run on one cannot be taken for a run on the other
    check("the two graphs have different identities", recorded != recorded_u)
    # The weighted file names the uniform file as its source
    check("the weighted file records the uniform graph as its source", body.get("weights", {}).get("inputs", {}).get("timeline") == recorded_u, f"recorded source {body.get('weights', {}).get('inputs', {}).get('timeline')}")
    # Meter order
    check("meter order of the manifest equals the node order of the weighted graph", list(manifest["meters"]) == [x["meter"] for x in body["nodes"]] and n == M, f"{M} meters, {n} nodes")

    # ---- models ----
    # The weighted model: stored weights
    model = sm.make_model("gcn", manifest, edges="true", edge_weights="admittance", seed=args.seed, timeline=timeline)
    # Binary weights on the weighted file
    model_b = sm.make_model("gcn", manifest, edges="true", edge_weights="binary", seed=args.seed, timeline=timeline)
    # The uniform graph, as trained so far
    model_u = sm.make_model("gcn", manifest, edges="true", edge_weights="admittance", seed=args.seed, timeline=timeline_u)

    # ---- P: periods ----
    print("\nP  Periods select their edge sets")
    # The model's period table against the file
    table = model.period_graph.tolist()
    # Judge: one entry per period, each the file's edge set number
    check(f"the model's period table sends each of the {len(periods)} periods to the edge set the file names", table == [int(p["graph"]) for p in body["periods"]], f"table {table}")
    # The table equals the uniform graph's, since no edge set was split
    check("the period table equals the uniform graph's (no edge set was split, no period boundary moved)", table == model_u.period_graph.tolist() and [(p["start_utc"], p["end_utc_exclusive"]) for p in body["periods"]] == [(p["start_utc"], p["end_utc_exclusive"]) for p in body_u["periods"]])
    # Each change of topology: windows on an aligned grid around it
    seen, bad, windows = set(), 0, 0
    # Every period but the first starts with a change
    for p in periods[1:]:
        # Aligned ten-second slots around the change
        t = gd.aligned_grid(int(p["start_utc"]), args.history)
        # Empty features and all-valid flags: only the times matter here
        X, valid = np.zeros((len(t), M, C, F), dtype=np.float32), np.ones((len(t), M, C), dtype=bool)
        # The window dataset, which attaches a period to every window
        ds = sm.GraphWindowDataset(X, valid, valid, args.history, 1, args.max_gap, 1, t=t, timeline=timeline)
        # The last history slot of each window
        last = t[ds.starts + args.history - 1]
        # The edge set the model uses for each window
        got = model.graph_ids(torch.as_tensor(np.asarray(ds.period), dtype=torch.long)).tolist()
        # The edge set of the period holding that slot, by a separate lookup in the file
        want = [int(body["periods"][gd.lookup_period(periods, v)]["graph"]) for v in last]
        # Count disagreements
        bad += sum(1 for a, b in zip(got, want) if a != b)
        # Count windows
        windows += len(got)
        # Periods reached
        seen |= set(np.asarray(ds.period).tolist())
    # Periods too short to hold an aligned slot cannot be selected by any window
    no_slot = [q["index"] for q in periods if ((int(q["start_utc"]) + gd.SLOT - 1) // gd.SLOT) * gd.SLOT >= int(q["end_utc_exclusive"])]
    # Judge
    check(f"around all {len(periods) - 1} topology changes, every window gets the edge set of the period holding its last history slot", bad == 0 and windows > 0, f"{windows} windows, {bad} disagreements; periods reached {sorted(seen)}; periods holding no aligned slot: {no_slot if no_slot else 'none'}")
    # Every period that can be selected was selected
    check("every period that holds an aligned slot was reached by some window", set(q["index"] for q in periods if q["index"] not in no_slot) <= seen)

    # ---- E: edge sets ----
    print("\nE  Edge sets reach the model with their stored weights")
    # Totals over the edge sets
    worst, non_unit, directed = 0.0, 0, 0
    # Each edge set
    for g, graph in enumerate(body["graphs"]):
        # What the model holds
        ei, w = model.edge_set(g)
        # What the file stores, per directed edge
        want = stored_weights(graph)
        # The model's directed edges
        got = [(int(a), int(b)) for a, b in ei.t().tolist()]
        # Connections: exactly the file's, both directions, no duplicate, in range
        wiring = set(got) == set(want) and len(got) == len(set(got)) == 2 * len(graph["edges"]) and int(ei.max()) < n and int(ei.min()) >= 0
        # Largest difference between a delivered and a stored weight
        diff = max(abs(float(x) - want[k]) for k, x in zip(got, w.tolist())) if wiring else float("inf")
        # Totals
        worst, directed = max(worst, diff), directed + len(got)
        # Delivered weights that are not 1.0
        non_unit += int((w < 1.0 - args.weight_tolerance).sum())
        # Judge
        check(f"edge set {g}: {len(graph['edges'])} stored edges delivered as {len(got)} directed edges, both directions, nothing else; each carries its stored weight", wiring and diff <= args.weight_tolerance, f"largest difference {diff:.1e}, delivered weights {float(w.min()):.4f} to {float(w.max()):.4f}, file {min(want.values()):.4f} to {max(want.values()):.4f}")
    # The weights are not all 1.0
    check("the delivered weights are not all 1.0", non_unit > 0, f"{non_unit} of {directed} directed edges carry a weight below 1.0; largest difference from the file over all edge sets {worst:.1e} (tolerance {args.weight_tolerance:.0e}, for the conversion to 32-bit numbers)")

    # ---- N: normalization ----
    print("\nN  The GCN layer's coefficients equal the separate calculation")
    # The layer's own normalization function
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    # A model whose state size equals the number of nodes, so the layer can be read with one node switched on at a time
    model_n = sm.make_model("gcn", manifest, edges="true", edge_weights="admittance", hidden=n, seed=args.seed, timeline=timeline)
    # Largest differences
    worst_fn, worst_layer = 0.0, 0.0
    # Each edge set
    for g, graph in enumerate(body["graphs"]):
        # Edges and weights as the model holds them
        ei, w = model_n.edge_set(g)
        # The separate calculation from the file
        want = torch.as_tensor(dense_coefficients(n, graph), dtype=torch.float32)
        # Through the normalization function, with the self-loops the layer adds
        ni, nw = gcn_norm(ei, w, n, add_self_loops=True)
        # As a matrix: row = receiver, column = sender
        fn = torch.zeros(n, n).index_put((ni[1], ni[0]), nw, accumulate=True)
        # Through the layer itself
        layer = layer_matrix(model_n.convs[0], n, ei, w)
        # Differences
        worst_fn, worst_layer = max(worst_fn, float((fn - want).abs().max())), max(worst_layer, float((layer - want).abs().max()))
    # Judge the function
    check(f"all {G} edge sets: the layer's normalization function gives the separately calculated self-loop and neighbor coefficients", worst_fn <= args.weight_tolerance, f"largest difference {worst_fn:.1e}")
    # Judge the layer
    check(f"all {G} edge sets: the model's GCN layer applies those same coefficients", worst_layer <= args.weight_tolerance, f"largest difference {worst_layer:.1e}")

    # ---- B: binary and uniform ----
    print("\nB  Binary weights and the uniform graph: same connections, different weights")
    # Same connections in every edge set
    same_wiring = all(torch.equal(model.edge_set(g)[0], model_b.edge_set(g)[0]) and torch.equal(model.edge_set(g)[0], model_u.edge_set(g)[0]) for g in range(G)) and model_u.n_graphs == G
    # Judge
    check("the weighted model, the binary model and the uniform-graph model hold identical connections in every edge set", same_wiring)
    # Weight buffers differ from the binary model's in every edge set
    differ_b = [not torch.equal(model.edge_set(g)[1], model_b.edge_set(g)[1]) for g in range(G)]
    # And from the uniform-graph model's
    differ_u = [not torch.equal(model.edge_set(g)[1], model_u.edge_set(g)[1]) for g in range(G)]
    # Judge
    check("the weighted model's weight buffers differ from the binary model's and from the uniform-graph model's in every edge set", all(differ_b) and all(differ_u), f"{sum(differ_b)} and {sum(differ_u)} of {G} edge sets differ")
    # The binary model on the weighted file and the uniform-graph model hold all ones
    ones = all(bool((model_b.edge_set(g)[1] == 1.0).all()) and bool((model_u.edge_set(g)[1] == 1.0).all()) for g in range(G))
    # Judge
    check("the binary model and the uniform-graph model carry weight 1.0 on every edge", ones)

    # ---- F: forecasts ----
    print("\nF  Forecasts with identical learned parameters and nonzero test heads")
    # Nonzero test heads on the weighted model
    gd.randomize_heads(model, args.seed)
    # The graph buffers of the other two models before the copy
    before = {id(m): {name: buf.clone() for name, buf in m.named_buffers()} for m in (model_b, model_u)}
    # The same learned parameters, heads included, in the other two models
    copy_parameters(model, model_b)
    # And the uniform-graph model
    copy_parameters(model, model_u)
    # The copy must not have touched a graph buffer
    untouched = all(torch.equal(before[id(m)][name], buf) for m in (model_b, model_u) for name, buf in m.named_buffers())
    # Judge
    check("the parameter copy left every graph buffer of the binary and uniform-graph models unchanged", untouched)
    # Evaluation mode for all three
    for m in (model, model_b, model_u):
        # No dropout
        m.eval()
    # What is compared
    print("note: forecast heads given nonzero test weights (the project zero-initializes them); inputs are all-valid synthetic inputs; every case reads the learned correction, forecast minus the persistence base")
    # One example per edge set
    x = gd.draw_inputs((G, args.history, M, C * F), C)
    # Example g passes messages over edge set g
    gid = torch.arange(G)
    # Learned corrections of the three models
    ya, yb, yu = gd.correction(model, x, gid), gd.correction(model_b, x, gid), gd.correction(model_u, x, gid)
    # Positive control: the correction responds to a meter's own earlier history with the persistence base held fixed
    base_same, own = gd.own_history_control(model, x, gid, 0, 0)
    # Judge
    check("positive control: the weighted model's learned correction responds to a meter's own earlier history with the persistence base held fixed", base_same and own > args.tolerance, f"base fixed {base_same}, correction change {own:.2e}")
    # Negative control: binary on the weighted file against the uniform graph, same connections and all ones
    check("negative control: the binary model on the weighted file and the uniform-graph model give identical corrections", gd.max_change(yb, yu) <= args.tolerance, f"largest difference {gd.max_change(yb, yu):.2e}")
    # Per edge set, how far the weighted model is from the binary model
    per_b = [gd.max_change(ya[g], yb[g]) for g in range(G)]
    # And from the uniform-graph model
    per_u = [gd.max_change(ya[g], yu[g]) for g in range(G)]
    # Judge
    check(f"in every one of the {G} edge sets the weighted model's corrections differ from the binary model's", all(v > args.tolerance for v in per_b), "largest difference per edge set " + ", ".join(f"{v:.2e}" for v in per_b))
    # Judge
    check(f"in every one of the {G} edge sets the weighted model's corrections differ from the uniform-graph model's", all(v > args.tolerance for v in per_u), "largest difference per edge set " + ", ".join(f"{v:.2e}" for v in per_u))

    # ---- S: no silent substitute ----
    print("\nS  The training script reads the stored weights by default")
    # The training script's own source
    with open(os.path.join(ROOT, "main_gnn.py")) as f:
        # Lines that define the --edge-weights option
        lines = [line.strip() for line in f if line.strip().startswith("parser.add_argument(\"--edge-weights\"")]
    # Judge: one definition, with the default that reads the stored weights
    check("main_gnn.py defines --edge-weights once, with the default \"admittance\" (the stored weights), and \"binary\" only on request", len(lines) == 1 and "default=\"admittance\"" in lines[0], lines[0][:140] if lines else "no definition found")
    # What a run must be started with
    print(f"a training run on this graph is started with --graph {args.graph} and without --edge-weights binary; its saved options must then show graph_id {recorded} and edge_weights admittance.")
    # What this check cannot show
    print("not shown here: that a finished run saved those options. That needs a run, and checks/main_runcompare.py --expected-graph " + recorded + " is the check for it.")

    # Tally
    print(f"\n{passed} of {total} cases PASS, {total - passed} FAIL")
    # Exit code
    return 0 if passed == total else 1


# Run only when executed directly
if __name__ == "__main__":
    # Exit with the code
    sys.exit(main())
