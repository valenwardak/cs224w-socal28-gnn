# =============================================================================
# File:        main_dynamicgraphcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-07
#
# Purpose:     One-off check of the time-indexed graph integration in
#              socal_models.py before any training run reads
#              graph_meters_periods.json:
#                1. dataset selection: GraphWindowDataset windows built on
#                   the real slot grid around every switch in the timeline
#                   and around every month boundary carry the period
#                   holding their last history slot, t[start + history - 1],
#                   found independently by scanning the period bounds;
#                   closely spaced switches included. The period-to-graph
#                   table gives the graph the file assigns to that period.
#                   A window whose last history slot lies before the record
#                   raises at dataset construction
#                2. mixed batches: in evaluation mode, a batch whose
#                   examples use different graphs gives the same forecast
#                   for every example as running that example alone with
#                   its graph, within tolerance
#                3. fixed-graph equivalence: a model built from the fixed
#                   graph_meters.json and a model built from the timeline,
#                   given the same weights and inputs in evaluation mode,
#                   agree when every example uses graph 0 (graph_ids all
#                   zero) and when graph_ids is None; the timeline model's
#                   state_dict holds the fixed model's keys plus only the
#                   extra edge sets
#                4. normalization: forwards over graph a, then graph b, then
#                   graph a again give identical results for a both times
#                   and equal a fresh model's result with the same weights,
#                   so nothing stale is reused between edge sets; every
#                   GCN layer has cached False and no stored normalization.
#                   Whether two graphs give different forecasts is printed
#                   as a NOTE only, since equal values are possible
#                5. edge assembly: for a mixed batch, the edges and weights
#                   batch_edges assembles for each example, with the node
#                   offset removed, equal the stored edge set of that
#                   example's graph, and no edge joins two examples; with
#                   --edges true and with --edges none (empty edge sets)
#                6. checkpoints: a checkpoint saved without the
#                   graph_selection key, as an older fixed-graph run wrote
#                   it, is accepted by load_checkpoint for a fixed-graph
#                   run with otherwise equal options and refused for a run
#                   asking for last_history_slot
#                7. refusals: masked mode with a timeline is refused by
#                   make_model, period indices outside the table are
#                   refused by graph_ids, and graph_ids of the wrong shape
#                   are refused by forward
#              Prints PASS or FAIL per check and exits with 1 on any failure.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root. Needs
#              torch and torch_geometric. Random inputs are used: the
#              checks are about index handling and equivalence, not about
#              forecast quality.
#
# Usage:
#   python checks/main_dynamicgraphcheck.py
#   python checks/main_dynamicgraphcheck.py --history 60 --horizon 1 --hidden 16
#
# Functions:
#   parse_args()            command-line options with defaults
#   report(name, ok, note)  print one PASS or FAIL line, return ok
#   holding(periods, t)     index of the period with start_utc <= t < end, by scan
#   manifest_for(timeline, features_dir)
#                           manifest.json if present, else one built from the timeline
#   synthetic_piece(t, M, C, rng)
#                           random features and validity on the slot grid t
#   edge_assembly_ok(model, graph_ids)
#                           per-example assembled edges equal the stored sets, no cross-example edge
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
# Third party: deep learning
import torch

# The repo root, one level above checks/, so the project modules import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project: data side, models, period lookup
import socal_modeldata as sd
import socal_models as sm
import socal_metergraph as smg

# Seconds per slot
SLOT = sd.SLOT_SECONDS


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Check the time-indexed graph integration before a training run")
    # Files
    parser.add_argument("--periods-graph", default="graph_meters_periods.json", help="The time-indexed meter graph")
    parser.add_argument("--meter-graph", default="graph_meters.json", help="The fixed meter graph")
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with manifest.json; a manifest is built from the timeline when absent")
    # Window, as the models use
    parser.add_argument("--history", type=int, default=60, help="History length in slots")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in slots")
    parser.add_argument("--max-gap", type=int, default=12, help="Gap rule")
    # Model size, small for speed
    parser.add_argument("--hidden", type=int, default=16, help="GRU hidden size")
    parser.add_argument("--gnn-layers", type=int, default=1, help="Message passing layers")
    # Tolerance
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for equivalence checks")
    # Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    # Parse
    return parser.parse_args()


def report(name, ok, note=""):
    """Print one PASS or FAIL line and return ok."""
    # Line
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + note if note else ''}")
    # Pass through
    return ok


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


def manifest_for(timeline, features_dir):
    """manifest.json from the features folder when present, else a manifest built from the timeline's meters and the channel names of the first meter record."""
    # Real manifest
    path = os.path.join(features_dir, "manifest.json")
    # Present
    if os.path.exists(path):
        # Read
        with open(path) as f:
            # Parse
            return json.load(f), "manifest.json"
    # Meters in node order
    meters = [n["meter"] for n in timeline["graphs"][0]["nodes"]]
    # Channels from the first meter's phase keys
    channels = list(timeline["graphs"][0]["meters"][meters[0]]["phase"].keys())
    # Built
    return {"meters": meters, "channels": channels}, "built from the timeline"


def synthetic_piece(t, M, C, rng):
    """Random scaled features (T, M, C, F) with every slot valid, and validity arrays (T, M, C), on the slot grid t."""
    # Length
    T = len(t)
    # Features: magnitude, cos, sin, and the flags, all random but in range
    X = rng.standard_normal((T, M, C, sd.F)).astype(np.float32)
    # Validity everywhere
    mag_valid = np.ones((T, M, C), dtype=bool)
    # Angle validity
    ang_valid = np.ones((T, M, C), dtype=bool)
    # Done
    return X, mag_valid, ang_valid


def edge_assembly_ok(model, graph_ids):
    """
    For a batch with the given graph ids, assemble the edges with
    model.batch_edges and check, per example, that the edges whose ends
    both lie in that example's node range, with the offset removed and
    sorted, equal that example's stored edge set with its weights, and
    that no edge has its two ends in different examples. Returns (ok, note).
    """
    # Batch size
    B = len(graph_ids)
    # Assemble
    ei, ew = model.batch_edges(B, graph_ids, graph_ids.device)
    # Example of each end
    ex_src, ex_dst = ei[0] // model.n_nodes, ei[1] // model.n_nodes
    # Cross-example edges
    cross = int((ex_src != ex_dst).sum())
    # Any
    if cross:
        # Fail
        return False, f"{cross} edges join two examples"
    # Per example
    for b in range(B):
        # Edges of this example
        rows = ex_src == b
        # Offset removed, as rows (source, target, weight rounded to 1e-6), sorted
        got = np.stack([(ei[0][rows] - b * model.n_nodes).numpy(), (ei[1][rows] - b * model.n_nodes).numpy(), np.round(ew[rows].numpy() * 1e6).astype(np.int64)], axis=1)
        # Stored set
        sei, sew = model.edge_set(int(graph_ids[b]))
        # Same form
        want = np.stack([sei[0].numpy(), sei[1].numpy(), np.round(sew.numpy() * 1e6).astype(np.int64)], axis=1)
        # Sort both by (source, target, weight)
        got = got[np.lexsort((got[:, 2], got[:, 1], got[:, 0]))] if len(got) else got
        # Sort
        want = want[np.lexsort((want[:, 2], want[:, 1], want[:, 0]))] if len(want) else want
        # Compare
        if got.shape != want.shape or not np.array_equal(got, want):
            # Fail
            return False, f"example {b} (graph {int(graph_ids[b])}): assembled {len(got)} edges, stored {len(want)}, or contents differ"
    # Good
    return True, f"{int(ei.shape[1])} edges over {B} examples"


def main():
    """Run the checks."""
    # Options
    args = parse_args()
    # Seeds
    torch.manual_seed(args.seed)
    # Generator
    rng = np.random.default_rng(args.seed)
    # Timeline
    body = sm.load_graph(args.periods_graph)
    # Must be a timeline
    if sm.graph_kind(body) != "timeline":
        # Raise
        raise ValueError(f"{args.periods_graph} is not a time-indexed graph file")
    # Validated
    timeline = sm.load_timeline(body)
    # Fixed graph
    fixed = sm.load_graph(args.meter_graph)
    # Manifest
    manifest, source = manifest_for(timeline, args.features_dir)
    # Sizes
    M, C = len(manifest["meters"]), len(manifest["channels"])
    # Periods
    periods = timeline["periods"]
    # Say
    print(f"timeline {args.periods_graph}: {len(timeline['graphs'])} graphs, {len(periods)} periods; manifest {source}: {M} meters, {C} channels; history {args.history}, horizon {args.horizon}")
    # Every check's result
    ok = True

    # 1. dataset selection on the real slot grid
    span = args.history + args.horizon
    # Times to build around: every switch time and every month start inside the record
    anchors = [p["start_utc"] for p in periods[1:]]
    # Month starts, in the files' fixed-PST base shifted to UTC
    cur = pd.Timestamp(body["record_start"]).to_period("M") + 1
    # Last month
    last = (pd.Timestamp(body["record_end"]) - pd.Timedelta(seconds=1)).to_period("M")
    # Loop
    while cur <= last:
        # UTC seconds of the month start
        anchors.append(int((cur.start_time + pd.Timedelta(hours=int(body["utc_offset_hours"]))).tz_localize("UTC").timestamp()))
        # Next
        cur += 1
    # Check every anchor
    bad, n_windows, n_new = [], 0, 0
    # Loop
    for a in anchors:
        # Slot grid from span + 5 slots before the anchor to 5 slots after, so windows end on both sides of it
        first = (a // SLOT) * SLOT - (span + 5) * SLOT
        # Grid
        t = first + SLOT * np.arange(2 * span + 10)
        # Keep inside the record
        t = t[(t >= periods[0]["start_utc"]) & (t < periods[-1]["end_utc_exclusive"])]
        # Too short
        if len(t) < span:
            # Next
            continue
        # Synthetic data
        X, mv, av = synthetic_piece(t, M, C, rng)
        # Dataset with the timeline
        ds = sm.GraphWindowDataset(X, mv, av, args.history, args.horizon, args.max_gap, 1, t=t, timeline=timeline)
        # Batch of everything
        x, y, y_valid, extra = ds.get_batch(np.arange(len(ds)))
        # Shape
        if extra.ndim != 2 or extra.shape[1] != 2:
            # Record
            bad.append(f"anchor {a}: extra shape {extra.shape}")
            # Next
            continue
        # Every window
        for i, s in enumerate(ds.starts):
            # Last history slot time
            H = int(t[s + args.history - 1])
            # Expected period by scan
            want = holding(periods, H)
            # Target slot
            target = s + args.history + args.horizon - 1
            # Compare target and period
            if int(extra[i, 0]) != target or int(extra[i, 1]) != want:
                # Record
                bad.append(f"anchor {a}: window {i} target {int(extra[i, 0])} vs {target}, period {int(extra[i, 1])} vs {want}")
            # Graph through the table
            if int(timeline["period_graph"][extra[i, 1]]) != int(periods[want]["graph"]):
                # Record
                bad.append(f"anchor {a}: window {i} graph table disagrees with the file")
            # Count
            n_windows += 1
            # Windows whose last history slot is at or after the anchor
            n_new += int(H >= a)
    # Report
    ok &= report("1a. every window around every switch and month boundary carries the period of its last history slot", not bad, (f"{n_windows} windows over {len(anchors)} anchors, {n_new} with the last history slot at or after the anchor") if not bad else "; ".join(bad[:5]))
    # A window before the record must raise
    t0 = periods[0]["start_utc"] - span * SLOT + SLOT * np.arange(2 * span)
    # Data
    X, mv, av = synthetic_piece(t0, M, C, rng)
    # Try
    try:
        # Build
        sm.GraphWindowDataset(X, mv, av, args.history, args.horizon, args.max_gap, 1, t=t0, timeline=timeline)
        # No raise
        raised = False
    except ValueError:
        # Raised
        raised = True
    # Report
    ok &= report("1b. a window whose last history slot precedes the record raises at dataset construction", raised)
    # Without a timeline the extra stays one-dimensional
    ds_plain = sm.GraphWindowDataset(X, mv, av, args.history, args.horizon, args.max_gap, 1)
    # Batch
    _, _, _, extra_plain = ds_plain.get_batch(np.arange(len(ds_plain)))
    # Report
    ok &= report("1c. without a timeline the batch extra is the (B,) target slot, as before", extra_plain.ndim == 1)

    # Models for the remaining checks, small and in evaluation mode
    dyn = sm.make_model("gcn", manifest, None, hidden=args.hidden, gnn_layers=args.gnn_layers, seed=args.seed, timeline=timeline).eval()
    # Random weights are fine, but the heads start at zero with predict-delta; perturb every parameter so the graph path matters
    with torch.no_grad():
        # Loop
        for p in dyn.parameters():
            # Add noise
            p.add_(0.1 * torch.randn_like(p))
    # Random input batch: one example per graph, then a few repeats
    G = len(timeline["graphs"])
    # Batch size
    B = G + 4
    # Input
    x = torch.randn(B, args.history, M, C * sd.F)
    # Graph ids: every graph once, then random
    gids = torch.cat([torch.arange(G), torch.randint(0, G, (4,))])

    # 2. mixed batch equals per-example inference
    with torch.no_grad():
        # Mixed
        y_mixed = dyn(x, graph_ids=gids)
        # One at a time
        y_single = torch.cat([dyn(x[b:b + 1], graph_ids=gids[b:b + 1]) for b in range(B)])
    # Difference
    diff = float((y_mixed - y_single).abs().max())
    # Report
    ok &= report("2. mixed-graph batch equals per-example inference in evaluation mode", diff <= args.atol, f"max abs diff {diff:.2e}, {G} graphs in the batch")

    # 3. fixed-graph equivalence
    fixed_model = sm.make_model("gcn", manifest, fixed, hidden=args.hidden, gnn_layers=args.gnn_layers, seed=args.seed).eval()
    # Keys
    fixed_keys, dyn_keys = set(fixed_model.state_dict().keys()), set(dyn.state_dict().keys())
    # Extra keys must be the numbered edge sets only
    extra_keys = dyn_keys - fixed_keys
    # Check
    keys_ok = fixed_keys <= dyn_keys and all(k.startswith("edge_index_") or k.startswith("edge_weight_") for k in extra_keys)
    # Report
    ok &= report("3a. timeline model state_dict holds the fixed model's keys plus only the extra edge sets", keys_ok, f"{len(extra_keys)} extra keys")
    # Graph 0 edge tensors must equal the fixed graph's, compared before any weights are copied
    edges_equal = fixed_model.edge_index.shape == dyn.edge_index.shape and torch.equal(fixed_model.edge_index, dyn.edge_index) and torch.allclose(fixed_model.edge_weight, dyn.edge_weight)
    # Copy the timeline model's weights into the fixed model
    missing, unexpected = fixed_model.load_state_dict(dyn.state_dict(), strict=False)
    # Only the numbered edge sets may be unexpected
    copy_ok = not missing and all(k.startswith("edge_index_") or k.startswith("edge_weight_") for k in unexpected)
    # Report
    ok &= report("3b. weights copy across and graph 0 edge tensors equal the fixed graph's", copy_ok and edges_equal)
    # Outputs
    with torch.no_grad():
        # Fixed
        y_fixed = fixed_model(x)
        # Timeline with graph 0 everywhere
        y_zero = dyn(x, graph_ids=torch.zeros(B, dtype=torch.long))
        # Timeline with no ids
        y_none = dyn(x)
    # Differences
    d0, d1 = float((y_fixed - y_zero).abs().max()), float((y_fixed - y_none).abs().max())
    # Report
    ok &= report("3c. timeline model with graph 0 (ids all zero, and ids None) equals the fixed-graph model", max(d0, d1) <= args.atol, f"max abs diff {d0:.2e} and {d1:.2e}")

    # 4. normalization
    convs = [c for c in dyn.convs]
    # Cached flag off everywhere
    cached_off = all(getattr(c, "cached", False) is False for c in convs)
    # Nothing stored
    nothing_stored = all(getattr(c, "_cached_edge_index", None) is None and getattr(c, "_cached_adj_t", None) is None for c in convs)
    # Graphs a and b: the two with the most different edge counts
    counts = [len(g["edges"]) for g in timeline["graphs"]]
    # Pick
    ga, gb = int(np.argmin(counts)), int(np.argmax(counts))
    # Ids
    ida, idb = torch.full((B,), ga, dtype=torch.long), torch.full((B,), gb, dtype=torch.long)
    # Forwards in sequence
    with torch.no_grad():
        # a
        ya1 = dyn(x, graph_ids=ida)
        # b
        yb = dyn(x, graph_ids=idb)
        # a again
        ya2 = dyn(x, graph_ids=ida)
        # Fresh model with the same weights
        fresh = sm.make_model("gcn", manifest, None, hidden=args.hidden, gnn_layers=args.gnn_layers, seed=args.seed, timeline=timeline).eval()
        # Copy
        fresh.load_state_dict(dyn.state_dict())
        # a on the fresh model
        ya3 = fresh(x, graph_ids=ida)
    # Differences
    da = max(float((ya1 - ya2).abs().max()), float((ya1 - ya3).abs().max()))
    # Report
    ok &= report("4a. graph a, then b, then a again gives identical results for a, equal to a fresh model's", da <= args.atol, f"max abs diff {da:.2e}, graphs {ga} ({counts[ga]} edges) and {gb} ({counts[gb]} edges)")
    # Stored state still empty after the forwards
    nothing_after = all(getattr(c, "_cached_edge_index", None) is None and getattr(c, "_cached_adj_t", None) is None for c in convs)
    # Report
    ok &= report("4b. every GCN layer has cached False and stores no normalization before or after forwards", cached_off and nothing_stored and nothing_after)
    # Different graphs usually change the forecast; equal values are possible, so this is a note, not a check
    dab = float((ya1 - yb).abs().max())
    # Print
    print(f"NOTE  graphs {ga} and {gb} on the same input differ by max abs {dab:.2e} (diagnostic only)")

    # 5. edge assembly, true edges and empty edges
    with torch.no_grad():
        # True edges, the mixed ids from check 2
        ok_true, note_true = edge_assembly_ok(dyn, gids)
        # A model with no edges at all
        none_model = sm.make_model("gcn", manifest, None, edges="none", hidden=args.hidden, gnn_layers=args.gnn_layers, seed=args.seed, timeline=timeline).eval()
        # Assembly
        ok_none, note_none = edge_assembly_ok(none_model, gids)
        # Its forward must run on empty edge sets
        y_none_edges = none_model(x, graph_ids=gids)
    # Report
    ok &= report("5a. assembled edges per example equal the stored edge set of its graph, no cross-example edge (--edges true)", ok_true, note_true)
    # Report
    ok &= report("5b. the same with --edges none: empty edge sets assemble and forward runs", ok_none and tuple(y_none_edges.shape) == (B, M, C, 3), note_none)

    # 6. checkpoint compatibility
    tmp = tempfile.mkdtemp(prefix="dyncheck_")
    # Options as an older fixed-graph run saved them: every experiment key except graph_selection
    old_options = {k: None for k in sm.EXPERIMENT_KEYS if k != "graph_selection"}
    # Fill the keys the fixed model needs to match
    old_options.update({"kind": "gcn", "masked": False, "maskable_meters": None, "graph_id": "graph_meters.json:test"})
    # Path
    ckpt = os.path.join(tmp, "checkpoint.pt")
    # Save as the training loop does
    sm.save_checkpoint(ckpt, fixed_model, torch.optim.Adam(fixed_model.parameters()), old_options, 1, 1.0, 0, np.random.default_rng(0))
    # A fixed-graph run with the same options must load it
    try:
        # Load
        sm.load_checkpoint(ckpt, fixed_model, dict(old_options, graph_selection="fixed"))
        # Accepted
        fixed_accepted = True
    except ValueError:
        # Refused
        fixed_accepted = False
    # A time-indexed run must be refused, even with the same graph id
    try:
        # Load
        sm.load_checkpoint(ckpt, dyn, dict(old_options, graph_selection="last_history_slot"))
        # Accepted, wrong
        dyn_refused = False
    except ValueError:
        # Refused, right
        dyn_refused = True
    # Report
    ok &= report("6a. an older checkpoint without graph_selection is accepted by a fixed-graph run", fixed_accepted)
    # Report
    ok &= report("6b. the same checkpoint is refused by a last_history_slot run", dyn_refused)

    # 7. refusals
    refusals = []
    # Masked with a timeline
    try:
        # Build
        sm.make_model("gcn", manifest, None, hidden=args.hidden, timeline=timeline, masked=True, maskable_meters=manifest["meters"][:1])
        # No raise
        refusals.append("masked with timeline accepted")
    except ValueError:
        # Good
        pass
    # Period outside the table
    try:
        # Map
        dyn.graph_ids(torch.tensor([len(periods)]))
        # No raise
        refusals.append("period index past the table accepted")
    except ValueError:
        # Good
        pass
    # Wrong shape
    try:
        # Forward
        with torch.no_grad():
            # Ids of the wrong length
            dyn(x, graph_ids=torch.zeros(B + 1, dtype=torch.long))
        # No raise
        refusals.append("graph_ids of the wrong shape accepted")
    except ValueError:
        # Good
        pass
    # Report
    ok &= report("7. refusals: masked with a timeline, period index past the table, graph_ids of the wrong shape", not refusals, "; ".join(refusals))
    # Exit code
    sys.exit(0 if ok else 1)


# Run when called as a script
if __name__ == "__main__":
    # Go
    main()
