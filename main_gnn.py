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
#              Sender gating (--sender-gate, added 2026-11-12): the course
#              staff's milestone feedback asked whether nodes without
#              readings hurt. The diagnostics showed that meters without
#              usable history stay connected and keep sending messages.
#              With --sender-gate such a meter sends no message in that
#              window; it keeps its self-loop and still receives. The rule
#              is the gap rule of --max-gap on the history only. The graph
#              file and the edge sets are unchanged. GCN only. Off by
#              default, so every earlier command trains and scores the same
#              model it did before. The run records sender_gate, so a
#              resume or --score-only with the other setting is refused,
#              and a checkpoint from before the option compares as off.
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
#   Sender gating on the time-indexed meter graph, matched to gcn_dyn_aw001 at seed 0:
#     python main_gnn.py --kind gcn --run-name gcn_dyn_aw001_gate --sender-gate --graph graph_meters_periods.json --splits splits_v2.json --angle-weight 0.01 --seed 0
#   GAT and GraphSAGE with the same settings:
#     python main_gnn.py --kind gat --run-name gat_dyn_aw001 --graph graph_meters_periods.json --splits splits_v2.json --angle-weight 0.01
#     python main_gnn.py --kind sage --run-name sage_dyn_aw001 --graph graph_meters_periods.json --splits splits_v2.json --angle-weight 0.01
#   With the physics inputs (after main_physicsfeatures.py has written <YYYY-MM>_physics.npz next to the features), the
#   matched pair at one seed, true edges and no edges, both receiving the same four added features:
#     python main_gnn.py --kind gcn --run-name gcn_dyn_aw001_phys --graph graph_meters_periods.json --splits splits_v2.json --angle-weight 0.01 --seed 0 --physics-dir /content/features
#     python main_gnn.py --kind gcn --run-name gcn_dyn_aw001_none_phys --edges none --graph graph_meters_periods.json --splits splits_v2.json --angle-weight 0.01 --seed 0 --physics-dir /content/features
#   With the neighbor angle inputs (experiment 2; after main_latefeatures.py has written <YYYY-MM>_late.npz next to the
#   features), the matched pair at one seed, physics off, both arms receiving the same three added features computed
#   from the true neighbors:
#     python main_gnn.py --kind gcn --run-name gcn_dyn_aw001_ang --graph graph_meters_periods.json --splits splits_v2.json --angle-weight 0.01 --seed 0 --angdiff-dir /content/features
#     python main_gnn.py --kind gcn --run-name gcn_dyn_aw001_none_ang --edges none --graph graph_meters_periods.json --splits splits_v2.json --angle-weight 0.01 --seed 0 --angdiff-dir /content/features
#   Options: python main_gnn.py --help
#
# Requires torch_geometric (pip install torch_geometric) for the GCN layers.
#
# Output under <out_dir>/<run_name>/ (default data/processed/models/<run_name>/):
#   scale.json          the per-channel scale used, from the train blocks, with
#                       the channels that had no train data and got a nominal value,
#                       and the two physics divisors when --physics-dir is given
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
# Project: the neighbor angle helpers, for the defaults
import socal_angdiff as sa
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
    # Sender gating, off by default: every earlier command builds the same model it did before
    parser.add_argument("--sender-gate", action="store_true", help="Sender gating: a meter without usable history (the --max-gap rule on the history only) sends no message in that window; it keeps its self-loop and still receives; gcn only")
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
    # Physics inputs, off by default: every earlier command builds the same inputs it did before
    parser.add_argument("--physics-dir", default=None, help="Folder with <YYYY-MM>_physics.npz and physics_manifest.json (main_physicsfeatures.py); turns the four physics input features on, for the true-edge and no-edge models alike")
    # Neighbor angle inputs
    parser.add_argument("--angdiff-dir", default=None, help="Folder with <YYYY-MM>_late.npz and late_manifest.json (main_latefeatures.py); turns the three neighbor angle difference inputs on, computed from the true neighbors of --graph for the true-edge and no-edge models alike; exclusive with --physics-dir")
    # Resultant cutoff
    parser.add_argument("--angdiff-min-resultant", type=float, default=sa.MIN_RESULTANT, help="Minimum mean resultant length of the neighbor mean for the direction to count")
    # Late threshold, recorded; the late files were written with their own threshold, which must agree
    parser.add_argument("--late-threshold", type=float, default=sa.LATE_THRESHOLD, help="Capture-offset cutoff in seconds the late files must have been written with")
    # Parse
    args = parser.parse_args()
    # GAT needs at least one head
    if args.kind == "gat" and args.heads < 1:
        # Refuse
        parser.error("--heads must be at least 1 for --kind gat")
    # Sender gating is defined for the GCN layer only
    if args.sender_gate and args.kind != "gcn":
        # Refuse
        parser.error("--sender-gate is defined for --kind gcn only")
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


def angdiff_identity(angdiff_dir, late_threshold):
    """
    Identity of the neighbor angle inputs for the run's options: the late
    manifest's name and the first 16 hex digits of the SHA-256 of its
    bytes. The manifest's threshold must be present, finite, positive and
    equal to --late-threshold, so a run cannot silently use late files
    written with another cutoff or with none recorded.
    """
    # Read and validate: threshold present, finite, positive and equal to --late-threshold; meter list present
    _, raw = sa.read_late_manifest(angdiff_dir, late_threshold)
    # Digest
    digest = hashlib.sha256(raw).hexdigest()[:16]
    # Name and digest
    return f"late_manifest.json:{digest}"


def physics_identity(physics_dir):
    """
    Identity of the physics inputs for the run's options: the physics
    manifest's name and the first 16 hex digits of the SHA-256 of its
    bytes, so two folders with different result settings differ.
    """
    # Manifest path
    path = os.path.join(physics_dir, "physics_manifest.json")
    # Must exist
    if not os.path.exists(path):
        # Raise
        raise FileNotFoundError(f"no physics_manifest.json in {physics_dir}; run main_physicsfeatures.py first")
    # Hash
    with open(path, "rb") as f:
        # Digest
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    # Name and digest
    return f"physics_manifest.json:{digest}"


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

    # Not both kinds of added input
    if args.physics_dir is not None and args.angdiff_dir is not None:
        # Raise
        raise ValueError("--physics-dir and --angdiff-dir cannot both be given")
    # Physics inputs on, before anything reads a month: identity recorded, scale set after it is computed or loaded
    physics_id = "none"
    # Asked for
    if args.physics_dir is not None:
        # Identity
        physics_id = physics_identity(args.physics_dir)
        # Turn the inputs on for this process
        sd.configure_physics(args.physics_dir)
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
    # Neighbor angle inputs on, before anything reads a month; the graph body gives the neighbors and the phase map
    angdiff_id, angdiff_settings = "none", None
    # Asked for
    if args.angdiff_dir is not None:
        # Identity, threshold checked against the late manifest
        angdiff_id = angdiff_identity(args.angdiff_dir, args.late_threshold)
        # Turn the inputs on for this process
        angdiff_settings = sd.configure_angdiff(args.angdiff_dir, body, manifest, args.angdiff_min_resultant, args.late_threshold)
        # The graph identity is part of the feature's identity
        angdiff_settings["graph_id"] = graph_id
        # Refuse to train on all-zero columns: counted inside the history inputs of the training windows the
        # selection would retain (history, horizon, gap rule, minimum meters, --train-months), here before the scale
        # is computed or loaded from scale.json, so neither path can skip the check
        ang_valid, ang_slots, ang_windows = sd.angdiff_training_valid(args.features_dir, splits["train"], manifest, args.history, args.horizon, args.max_gap, args.min_meters)
        # Nothing valid inside the retained histories is a configuration fault, not an experiment
        if ang_valid == 0:
            # Raise
            raise ValueError(f"the neighbor angle feature is valid in none of the {ang_slots} history slots of the {ang_windows} training windows that would be retained; check the late files, the graph periods and the phase map (main_latefeatures.py --table-only) before training")
        # Say
        log(f"neighbor angle feature valid in {ang_valid} meter-channel values of the {ang_slots} history slots of the {ang_windows} retained training windows")
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
        # Sender gating: whether meters without usable history are silenced as senders; its gap rule is max_gap above
        "sender_gate": args.sender_gate,
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
        # Physics inputs: "none", or the physics manifest's identity; the features per channel the model reads, in order;
        # and the physics divisors, filled in below once the scale is computed or loaded (None without physics inputs)
        "physics": physics_id, "n_features": sd.n_features(), "features": list(sd.FEATURES) + sd.extra_features(), "physics_scale": None,
        # Neighbor angle inputs: "none", or the late manifest's identity, and the settings (cutoffs, phase map, graph identity)
        "angdiff": angdiff_id, "angdiff_settings": angdiff_settings,
    }
    # Report
    log(f"device {device}, run {run_dir}, options {options}")
    # Scale: load if present, else compute from the train blocks and save
    scale_path = os.path.join(run_dir, "scale.json")
    # Existing
    if os.path.exists(scale_path):
        # Load, checked against the manifest
        scale = sd.load_scale(scale_path, manifest)
        # Physics divisors recorded in the file, None for a run without them
        physics_scale = sd.load_physics_scale(scale_path)
        # A run with physics inputs needs them, a run without must not have them
        if (physics_scale is None) != (args.physics_dir is None):
            # Raise
            raise ValueError(f"{scale_path} {'has no' if physics_scale is None else 'has'} physics divisors but this run {'uses' if args.physics_dir is not None else 'does not use'} physics inputs; use a new run folder")
        # Set
        if physics_scale is not None:
            # Configure with the loaded divisors
            sd.configure_physics(args.physics_dir, physics_scale)
            # Record them with the experiment, so a checkpoint made with other divisors is refused
            options["physics_scale"] = [float(x) for x in physics_scale]
        # Say so
        log(f"loaded scale from {scale_path}" + (f", physics divisors {physics_scale.tolist()}" if physics_scale is not None else ""))
    # Compute
    else:
        # From train blocks, with the nominal fallback for channels that have none
        scale, count, fallback = sd.compute_scale(args.features_dir, splits, manifest, args.graph, args.scale_percentile, args.scale_floor)
        # Physics divisors from the train blocks, same percentile; stops if a side has no valid training value
        physics = None
        # With physics inputs
        if args.physics_dir is not None:
            # Compute
            pscale, pcount = sd.compute_physics_scale(args.features_dir, splits, args.scale_percentile)
            # Configure with them
            sd.configure_physics(args.physics_dir, pscale)
            # Record them with the experiment
            options["physics_scale"] = [float(x) for x in pscale]
            # For the file
            physics = {"scale": pscale, "count": pcount, "features_dir": os.path.abspath(args.physics_dir)}
        # Save
        sd.save_scale(scale_path, scale, count, manifest, args.scale_percentile, args.scale_floor, fallback, physics)
        # Say so, naming the channels that used the fallback
        used = [f"{manifest['meters'][mi]}|{manifest['channels'][ci]}" for mi, ci in zip(*np.nonzero(fallback))]
        # Report
        log(f"computed scale from {int(count.sum())} valid train readings, saved to {scale_path}; nominal fallback used for {len(used)} channels" + (f": {used}" if used else ""))
        # Physics report
        if physics is not None:
            # Line
            log(f"physics divisors (percentile {args.scale_percentile} of valid train residuals): A {float(pscale[0]):.6f} from {int(pcount[0])} values, B {float(pscale[1]):.6f} from {int(pcount[1])} values")
    # Physics inputs report
    if args.physics_dir is not None:
        # Line
        log(f"physics inputs on from {args.physics_dir} ({physics_id}): {sd.n_features()} features per channel in the order {options['features']}, divisors {options['physics_scale']}; the four physics features are shared by every meter and channel and zeroed in every masked forward")
    # Neighbor angle inputs report
    if args.angdiff_dir is not None:
        # Line
        log(f"neighbor angle inputs on from {args.angdiff_dir} ({angdiff_id}): {sd.n_features()} features per channel in the order {options['features']}; equal-weight neighbor mean from {args.graph} (phase map {angdiff_settings['phase_map']}), each slot's own period, late threshold {args.late_threshold} s, minimum resultant {args.angdiff_min_resultant}; the three added features are zeroed in every masked forward")
    # Report the graph
    if timeline is None:
        # Fixed
        log(f"graph {args.graph}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges, edges={args.edges}, edge_weights={args.edge_weights}, selection fixed")
    else:
        # Time-indexed
        log(f"graph {args.graph}: time-indexed, {len(graph['nodes'])} nodes, {len(timeline['graphs'])} edge sets ({[len(g['edges']) for g in timeline['graphs']]} edges) over {len(timeline['periods'])} periods, edges={args.edges}, edge_weights={args.edge_weights}, selection last_history_slot")
    # Report sender gating
    if args.sender_gate:
        # Line
        log(f"sender gating on: a meter with a run of {args.max_gap} or more absent slots in its {args.history}-slot history sends no message in that window; self-loops and incoming messages are kept; graph file and edge sets unchanged")
    # Report the masked setup
    if args.masked:
        # Names
        log(f"masked objective: {len(maskable_meters)} maskable meters {maskable_meters}; not maskable: {[m for m in manifest['meters'] if m not in maskable_meters]}; {args.mask_chunk} meters per evaluation pass")
    # Model
    model = sm.make_model(args.kind, manifest, graph, edges=args.edges, edge_weights=args.edge_weights, hidden=args.hidden, embed=args.embed, layers=args.layers, dropout=args.dropout, heads=args.heads, seed=args.seed, gnn_layers=args.gnn_layers, predict_delta=predict_delta, masked=args.masked, maskable_meters=maskable_meters, timeline=timeline, n_features=sd.n_features(), sender_gate=args.sender_gate, gate_max_gap=args.max_gap).to(device)
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
