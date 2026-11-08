# =============================================================================
# File:        main_gnn.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-23
#
# Purpose:     Train and score the first graph model: the Graph
#              Convolutional Network (GCN) forecaster. The same GRU encoder
#              as the no-graph baseline turns each meter's history into a
#              vector, those vectors are placed on the nodes of the grid
#              graph from graph.json, GCN layers pass messages along the
#              lines and transformers, and a head forecasts each meter's
#              next reading. All the work is in socal_models.py and
#              socal_modeldata.py; this file parses options, computes or
#              loads the scale, builds the model, trains, and writes
#              residuals for the requested splits.
#
#              This file is main_gru.py with a graph model and four graph
#              options added (it was main_gcn.py until November 7, 2026;
#              main_gcn.py now calls this file with --kind gcn, so earlier
#              commands keep working): --edges, --edge-weights, --gnn-layers, and
#              --min-meters. --kind chooses the message-passing layer:
#              gcn (default, weighted sum with the GCN normalization), gat
#              (attention over neighbors, --heads heads, the edge weight
#              given as a one-number edge feature), or sage (mean of the
#              neighbors, weights unused). The three share the encoder,
#              head, loss, data path, time-indexed graph handling, and
#              checkpoint safeguards; only the layer differs. The run
#              records kind and, for gat, heads, so a resume with another
#              kind is refused. A new model (for example an R-GCN) gets a
#              model class in socal_models.py and a line in make_model.
#
#              By default the model predicts the change from the last valid
#              reading (--predict-delta), so persistence is its starting
#              point and it only has to learn what persistence misses.
#              --no-predict-delta makes it forecast the reading directly.
#
#              Masked objective (--masked, added 2026-11-05): one meter per
#              training window is hidden and forecast from its neighbors
#              through the graph; see socal_models.py. Meant for the meter
#              graph: --graph graph_meters.json, where every meter but
#              egauge_24 has a metered neighbor one hop away, and
#              --splits splits_v2.json, which holds out a 2023 calibration
#              block. The maskable meters are read from the graph file given
#              (every meter with at least one edge) and are the same for
#              every --edges control. Predict-delta is off in masked mode,
#              whatever --predict-delta says, and the effective value is
#              recorded in the options. The output layout is unchanged, so
#              main_score.py scores a masked run like any other.
#
#              Time-indexed graph (added 2026-11-07): --graph may name
#              graph_meters_periods.json from main_metergraphs.py instead
#              of a fixed graph. The file's layout decides which path runs.
#              With the timeline, every window passes messages over the
#              wiring in force at its last history slot (the pre-target
#              rule), the model holds one edge set per distinct wiring, and
#              the --edges control is applied to each of them with the same
#              seed. The run records graph_selection = "last_history_slot"
#              (a fixed graph records "fixed") and the file's identity, so
#              a resume or --score-only with the other kind of file is
#              refused. --masked with a timeline is refused for now.
#
#              Ablations (the proposal's "does the graph help" test):
#                --edges true     the real wiring            (default)
#                --edges random   same number of edges, random node pairs
#                --edges none     no edges, each node sees only itself
#                --edge-weights admittance | binary
#
#              Batches are synchronized windows: one example is all meters
#              at one time slot, so a batch of 64 holds 64 x 22 histories.
#              The default batch size is therefore smaller than the GRU's.
#
# Usage:
#   Smoke test, one month of train, two epochs, tiny model:
#     python main_gnn.py --train-months 2024-10 --epochs 2 --hidden 16 --run-name smoke_gcn --score-splits validation
#   Full training with the defaults, then scoring on validation and the test splits:
#     python main_gnn.py --run-name gcn_v1
#   Ablations:
#     python main_gnn.py --run-name gcn_random --edges random
#     python main_gnn.py --run-name gcn_none --edges none
#   Score only, from a finished run:
#     python main_gnn.py --run-name gcn_v1 --score-only
#   Masked model on the meter graph with the held-out 2023 block:
#     python main_gnn.py --run-name gcn_masked --masked --graph graph_meters.json --splits splits_v2.json --angle-weight 0.01
#   Time-indexed meter graph, smoke test on September 2023 (holds two transfers):
#     python main_gnn.py --train-months 2023-09 --epochs 2 --hidden 16 --angle-weight 0.01 --run-name smoke_dyn --graph graph_meters_periods.json --splits splits_v2.json --score-splits validation
#   Time-indexed meter graph, full run:
#     python main_gnn.py --run-name gcn_dyn --graph graph_meters_periods.json --splits splits_v2.json --angle-weight 0.01
#   GAT and GraphSAGE with the same settings:
#     python main_gnn.py --kind gat --run-name gat_dyn_aw001 --graph graph_meters_periods.json --splits splits_v2.json --angle-weight 0.01
#     python main_gnn.py --kind sage --run-name sage_dyn_aw001 --graph graph_meters_periods.json --splits splits_v2.json --angle-weight 0.01
#   Options: python main_gnn.py --help
#
# Requires torch_geometric (pip install torch_geometric) for the GCN layers.
#
# Output under <out_dir>/<run_name>/ (default data/processed/models/<run_name>/):
#   scale.json          the per-channel scale used, from the train blocks, with
#                       the channels that had no train data and got a nominal value
#   checkpoint.pt       last epoch: weights, optimizer, options, progress, random states
#   checkpoint_best.pt  weights of the best validation epoch
#   train_log.txt       one line per epoch, and everything printed
#   <split>_<label>.parquet / .json   residual tables and summaries per block
#
# Functions:
#   parse_args()        command-line options with defaults
#   pick_device(name)   cuda, mps, or cpu
#   restrict_months(blocks, months)
#                       cut train blocks down to the chosen months
#   graph_identity(path) the graph file's name and a hash of its contents, saved with the run
#   maskable_from_graph(graph)
#                       meters with at least one edge in the graph file, in meter-number order
#   main()              scale, graph, model, train, score
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: content hash of the graph file, for its identity
import hashlib
# Standard library: paths
import os
# Third party: numerics
import numpy as np
# Third party: timestamps for the month cut
import pandas as pd
# Third party: deep learning
import torch
# Project: data side
import socal_modeldata as sd
# Project: models and training
import socal_models as sm


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser with the file's purpose as description
    parser = argparse.ArgumentParser(description="Train and score the GCN forecaster on the grid graph")
    # Data
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with <YYYY-MM>.npz and manifest.json")
    # Splits
    parser.add_argument("--splits", default="splits.json", help="Split definition written by main_splits.py")
    # Graph: the wiring the model passes messages over, and the clamp ratings and nominal voltages behind the scale
    parser.add_argument("--graph", default="graph.json", help="Graph file: graph.json or graph_meters.json (fixed), or graph_meters_periods.json (time-indexed)")
    # Which edges to use, the ablation switch
    parser.add_argument("--edges", choices=["true", "random", "none"], default="true", help="true = real wiring, random = same count between random nodes, none = no edges")
    # How to weight the edges
    parser.add_argument("--edge-weights", choices=["admittance", "binary"], default="admittance", help="admittance = scaled admittance_abs from graph.json, binary = all ones")
    # Message passing layer
    parser.add_argument("--kind", choices=["gcn", "gat", "sage"], default="gcn", help="gcn = GCN normalization with edge weights, gat = attention over neighbors, sage = mean of neighbors")
    # Attention heads, used by gat only
    parser.add_argument("--heads", type=int, default=4, help="Attention heads for --kind gat (ignored otherwise)")
    # Message passing depth
    parser.add_argument("--gnn-layers", type=int, default=1, help="Number of message passing layers (hops)")
    # Window rule for synchronized batches
    parser.add_argument("--min-meters", type=int, default=1, help="Keep a time slot only if at least this many meters pass the gap rule")
    # Masked objective
    parser.add_argument("--masked", action="store_true", help="Masked objective: hide one meter per window and forecast it from its neighbors; see socal_models.py")
    # Meters per masked evaluation pass
    parser.add_argument("--mask-chunk", type=int, default=7, help="Masked evaluation and scoring: meters hidden per forward pass (the batch is repeated this many times)")
    # Output
    parser.add_argument("--out-dir", default=os.path.join("data", "processed", "models"), help="Parent folder for runs")
    # Run name
    parser.add_argument("--run-name", default="gcn", help="Subfolder for this run's files")
    # Restrict training months, for smoke tests
    parser.add_argument("--train-months", nargs="*", default=None, help="Only train on these YYYY-MM months (smoke tests); default all train blocks")
    # Window
    parser.add_argument("--history", type=int, default=60, help="History length in slots (60 = 10 minutes)")
    # Horizon
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in slots")
    # Gap rule, must be at most the history or it can never fire
    parser.add_argument("--max-gap", type=int, default=12, help="A window is dropped if its history holds a run of this many absent slots (12 = 2 minutes); must not exceed --history")
    # Model
    parser.add_argument("--hidden", type=int, default=64, help="GRU hidden size")
    # Embedding
    parser.add_argument("--embed", type=int, default=8, help="Meter embedding size")
    # Layers
    parser.add_argument("--layers", type=int, default=1, help="GRU layers")
    # Dropout
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout between GRU layers, when layers > 1")
    # Predict the change from the last reading
    parser.add_argument("--predict-delta", action=argparse.BooleanOptionalAction, default=True, help="Forecast = last valid reading + a learned correction (default); --no-predict-delta forecasts the reading directly")
    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs")
    # Batch
    parser.add_argument("--batch-size", type=int, default=64, help="Time slots per batch; each slot carries every meter")
    # Learning rate
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    # Weight decay
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    # Angle weight
    parser.add_argument("--angle-weight", type=float, default=1.0, help="Weight of the angle term in the loss")
    # Early stopping
    parser.add_argument("--patience", type=int, default=3, help="Epochs without validation improvement before stopping")
    # Minimum improvement
    parser.add_argument("--min-delta", type=float, default=1e-5, help="Validation improvement that counts")
    # Scale
    parser.add_argument("--scale-percentile", type=float, default=95.0, help="Percentile of valid train readings used as the scale")
    # Scale floor
    parser.add_argument("--scale-floor", type=float, default=1.0, help="Scale floor when no graph gives a clamp rating")
    # Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    # Device
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto")
    # Scoring splits
    parser.add_argument("--score-splits", nargs="*", default=["validation", "test_events", "test_normal"], help="Splits to write residuals for")
    # Score only
    parser.add_argument("--score-only", action="store_true", help="Skip training, load the best checkpoint, and score")
    # Parse
    args = parser.parse_args()
    # GAT needs at least one head
    if args.kind == "gat" and args.heads < 1:
        # Refuse
        parser.error("--heads must be at least 1 for --kind gat")
    # The masked evaluation chunk must hide at least one meter per pass
    if args.mask_chunk < 1:
        # Refuse
        parser.error(f"--mask-chunk must be at least 1, got {args.mask_chunk}")
    # Done
    return args


def pick_device(name):
    """Choose the compute device: cuda on Colab, mps on Apple silicon, cpu otherwise."""
    # Explicit
    if name != "auto":
        # Use it
        return torch.device(name)
    # GPU
    if torch.cuda.is_available():
        # CUDA
        return torch.device("cuda")
    # Apple
    if torch.backends.mps.is_available():
        # MPS
        return torch.device("mps")
    # Fallback
    return torch.device("cpu")


def restrict_months(blocks, months):
    """
    Cut each block down to the parts that lie inside the chosen months,
    in the feature files' fixed-PST month convention. Returns new block
    dictionaries with the same keys, so the rest of the code is unchanged.
    """
    # Output
    out = []
    # Loop over blocks
    for block in blocks:
        # Bounds
        b_start, b_end = sd.block_seconds(block)
        # Loop over chosen months
        for month in months:
            # Month bounds
            m_start, m_end = sd.month_bounds(month)
            # Intersection
            start, end = max(b_start, m_start), min(b_end, m_end)
            # Empty
            if start >= end:
                # Next
                continue
            # UTC strings
            start_utc = pd.Timestamp(start, unit="s", tz="UTC").isoformat()
            # End
            end_utc = pd.Timestamp(end, unit="s", tz="UTC").isoformat()
            # Labels in the files' time base
            start_label = pd.Timestamp(start - sd.UTC_OFFSET_SECONDS, unit="s").strftime("%Y-%m-%d")
            # End label, last day inside
            end_label = pd.Timestamp(end - 1 - sd.UTC_OFFSET_SECONDS, unit="s").strftime("%Y-%m-%d")
            # New block
            out.append({"start": start_label, "end": end_label, "start_utc": start_utc, "end_utc_exclusive": end_utc})
    # Return
    return out


def graph_identity(path):
    """The graph file's name and the first 16 hex digits of the SHA-256 of its bytes, so two files with the same counts but different wiring differ."""
    # Hash the bytes
    with open(path, "rb") as f:
        # Digest
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    # Name and digest
    return f"{os.path.basename(path)}:{digest}"


def maskable_from_graph(graph):
    """
    The meters that may be hidden in masked mode: every meter whose node
    has at least one edge in the graph file as written, before any --edges
    control is applied. In meter-number order. egauge_24 has no edge in
    graph_meters.json and is left out.
    """
    # Nodes with an edge
    touched = set()
    # Loop over edges
    for e in graph["edges"]:
        # Both ends
        touched.add(int(e["source"])); touched.add(int(e["target"]))
    # Meters on those nodes
    names = [m for m, rec in graph["meters"].items() if int(rec["node"]) in touched]
    # Sorted by number
    return sorted(names, key=lambda m: int(m.split("_")[1]))


def main():
    """Scale, graph, model, train, score."""
    # Options
    args = parse_args()
    # Seeds
    torch.manual_seed(args.seed)
    # NumPy
    np.random.seed(args.seed)
    # Device
    device = pick_device(args.device)
    # Run folder
    run_dir = os.path.join(args.out_dir, args.run_name)
    # Make it
    os.makedirs(run_dir, exist_ok=True)
    # Log file
    log_path = os.path.join(run_dir, "train_log.txt")

    # Log: print and append to the file
    def log(text):
        # Screen
        print(text, flush=True)
        # File
        with open(log_path, "a") as f:
            # Line
            f.write(text + "\n")

    # Manifest and splits
    manifest = sd.load_manifest(args.features_dir)
    # Splits, overlap checked
    splits = sd.load_splits(args.splits)
    # The gap rule must be able to fire
    if args.max_gap > args.history:
        # Raise
        raise ValueError(f"--max-gap {args.max_gap} exceeds --history {args.history}, so no window could ever be dropped")
    # Restrict train blocks to the chosen months, for smoke tests
    if args.train_months:
        # Each month once, in order
        months = sorted(set(args.train_months))
        # Cut every block to the parts inside those months
        splits["train"] = restrict_months(splits["train"], months)
        # Nothing left is an error, not a run on an untrained model
        if not splits["train"]:
            # Raise
            raise ValueError(f"no train block overlaps months {months}")
        # Say so
        log(f"training restricted to months {months}: {len(splits['train'])} block pieces")
    # Identity of the training data, checked on resume
    train_blocks = [f"{b['start_utc']}/{b['end_utc_exclusive']}" for b in splits["train"]]
    # Graph file, required for a GNN, loaded before the options so its identity and maskable meters are recorded
    body = sm.load_graph(args.graph)
    # Fixed graph or time-indexed timeline, from the file's layout
    kind = sm.graph_kind(body)
    # Time-indexed: validate and keep the timeline; the fixed graph is graph 0 for the scale and the report
    timeline = sm.load_timeline(body) if kind == "timeline" else None
    # The graph for the fixed path, or graph 0 of the timeline
    graph = body if timeline is None else timeline["graphs"][0]
    # How each window's graph is chosen, recorded with the run
    graph_selection = "fixed" if timeline is None else "last_history_slot"
    # Masked mode with a timeline is refused for now
    if args.masked and timeline is not None:
        # Raise
        raise ValueError("--masked with a time-indexed graph is not supported yet; use a fixed graph file")
    # Identity of the graph file, checked on resume: its name and a hash of its contents, not its path, so Colab and local runs agree
    graph_id = graph_identity(args.graph)
    # Maskable meters, from the file as written, the same for every --edges control
    maskable_meters = maskable_from_graph(graph) if args.masked else None
    # Masked mode forecasts the absolute value, never a correction to persistence
    predict_delta = args.predict_delta and not args.masked
    # Say so when the flag was overridden
    if args.masked and args.predict_delta:
        # Note
        log("masked mode: predict-delta turned off, the hidden meter has no persistence value to correct")
    # Masked mode needs at least one maskable meter
    if args.masked and not maskable_meters:
        # Raise
        raise ValueError(f"no meter in {args.graph} has an edge, nothing can be masked")
    # Options that define the experiment, saved with the checkpoint
    options = {
        # Model kind
        "kind": args.kind,
        # Window
        "history": args.history, "horizon": args.horizon, "max_gap": args.max_gap,
        # Model
        "hidden": args.hidden, "embed": args.embed, "layers": args.layers, "dropout": args.dropout, "gnn_layers": args.gnn_layers, "heads": args.heads if args.kind == "gat" else 0, "predict_delta": predict_delta,
        # Graph options
        "edges": args.edges, "edge_weights": args.edge_weights, "min_meters": args.min_meters,
        # Graph identity, how each window's graph is chosen, and the masked objective
        "graph_id": graph_id, "graph_selection": graph_selection, "masked": args.masked, "maskable_meters": maskable_meters, "mask_chunk": args.mask_chunk,
        # Training
        "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "angle_weight": args.angle_weight,
        # Stopping
        "epochs": args.epochs, "patience": args.patience, "min_delta": args.min_delta,
        # Seed
        "seed": args.seed,
        # Training data identity
        "train_blocks": train_blocks, "splits_file": os.path.abspath(args.splits),
    }
    # Report
    log(f"device {device}, run {run_dir}, options {options}")
    # Scale: load if present, else compute from the train blocks and save
    scale_path = os.path.join(run_dir, "scale.json")
    # Existing
    if os.path.exists(scale_path):
        # Load, checked against the manifest
        scale = sd.load_scale(scale_path, manifest)
        # Say so
        log(f"loaded scale from {scale_path}")
    # Compute
    else:
        # From train blocks, with the nominal fallback for channels that have none
        scale, count, fallback = sd.compute_scale(args.features_dir, splits, manifest, args.graph, args.scale_percentile, args.scale_floor)
        # Save
        sd.save_scale(scale_path, scale, count, manifest, args.scale_percentile, args.scale_floor, fallback)
        # Say so, naming the channels that used the fallback
        used = [f"{manifest['meters'][mi]}|{manifest['channels'][ci]}" for mi, ci in zip(*np.nonzero(fallback))]
        # Report
        log(f"computed scale from {int(count.sum())} valid train readings, saved to {scale_path}; nominal fallback used for {len(used)} channels" + (f": {used}" if used else ""))
    # Report the graph
    if timeline is None:
        # Fixed
        log(f"graph {args.graph}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges, edges={args.edges}, edge_weights={args.edge_weights}, selection fixed")
    else:
        # Time-indexed
        log(f"graph {args.graph}: time-indexed, {len(graph['nodes'])} nodes, {len(timeline['graphs'])} edge sets ({[len(g['edges']) for g in timeline['graphs']]} edges) over {len(timeline['periods'])} periods, edges={args.edges}, edge_weights={args.edge_weights}, selection last_history_slot")
    # Report the masked setup
    if args.masked:
        # Names
        log(f"masked objective: {len(maskable_meters)} maskable meters {maskable_meters}; not maskable: {[m for m in manifest['meters'] if m not in maskable_meters]}; {args.mask_chunk} meters per evaluation pass")
    # Model
    model = sm.make_model(args.kind, manifest, graph, edges=args.edges, edge_weights=args.edge_weights, hidden=args.hidden, embed=args.embed, layers=args.layers, dropout=args.dropout, heads=args.heads, seed=args.seed, gnn_layers=args.gnn_layers, predict_delta=predict_delta, masked=args.masked, maskable_meters=maskable_meters, timeline=timeline).to(device)
    # Parameter count
    log(f"model {args.kind} with {sum(p.numel() for p in model.parameters())} parameters" + (f", {args.heads} attention heads" if args.kind == "gat" else ""))
    # Checkpoint path
    checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
    # Train unless scoring only
    if not args.score_only:
        # Train, best weights restored at the end
        best = sm.train(model, args.features_dir, splits, scale, manifest, device, options, checkpoint_path, log, timeline=timeline)
        # Report
        log(f"training done, best validation loss {best:.5f}")
    # Scoring only: check the saved options against the current ones, then load the best weights
    else:
        # Full checkpoint must exist
        if not os.path.exists(checkpoint_path):
            # Raise
            raise FileNotFoundError(f"no checkpoint at {checkpoint_path}")
        # Option check through the same path a resume uses; raises on a mismatch
        epoch, best, _ = sm.load_checkpoint(checkpoint_path, model, options, device=device)
        # Best path
        best_path = checkpoint_path.replace(".pt", "_best.pt")
        # Must exist
        if not os.path.exists(best_path):
            # Raise
            raise FileNotFoundError(f"no best checkpoint at {best_path}")
        # Load the best weights over the checked ones
        model.load_state_dict(torch.load(best_path, map_location=device))
        # Say so
        log(f"options match the checkpoint (epoch {epoch}, best {best:.5f}); loaded best weights from {best_path}")
    # Score each requested split
    for split_name in args.score_splits:
        # Skip unknown
        if split_name not in splits:
            # Say so
            log(f"split {split_name} not in {args.splits}, skipped")
            # Next
            continue
        # Announce
        log(f"scoring {split_name}")
        # Write residuals
        sm.score_split(model, args.features_dir, splits, split_name, scale, manifest, device, options, run_dir, log, timeline=timeline)
    # Done
    log("done")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
