# =============================================================================
# File:        main_gru.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-22
#
# Purpose:     Train and score the no-graph baseline: the shared,
#              meter-conditioned GRU forecaster. Each meter forecasts its
#              next reading from its own history only. All the work is in
#              socal_models.py and socal_modeldata.py; this file parses
#              options, computes or loads the scale, builds the model,
#              trains, and writes residuals for the requested splits.
#
#              By default the model predicts the change from the last valid
#              reading (--predict-delta), so persistence is its starting
#              point and it only has to learn what persistence misses.
#              --no-predict-delta makes it forecast the reading directly.
#
#              The same options and steps apply to main_gcn.py,
#              main_gat.py, and main_sage.py, which differ only in the
#              model kind and the graph options.
#
# Usage:
#   Smoke test, one month of train, two epochs, tiny model:
#     python main_gru.py --train-months 2024-10 --epochs 2 --hidden 16 --run-name smoke
#   Full training with the defaults, then scoring on validation and the test splits:
#     python main_gru.py --run-name gru_v1
#   Score only, from a finished run:
#     python main_gru.py --run-name gru_v1 --score-only
#   Options: python main_gru.py --help
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
#   main()              scale, model, train, score
# =============================================================================

# Standard library: command-line options
import argparse
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
    parser = argparse.ArgumentParser(description="Train and score the no-graph GRU baseline")
    # Data
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with <YYYY-MM>.npz and manifest.json")
    # Splits
    parser.add_argument("--splits", default="splits.json", help="Split definition written by main_splits.py")
    # Graph, used only for the scale in the GRU: clamp ratings for the floor, nominal values for channels with no train data
    parser.add_argument("--graph", default="graph.json", help="Graph file, for the clamp ratings and nominal voltages behind the scale")
    # Output
    parser.add_argument("--out-dir", default=os.path.join("data", "processed", "models"), help="Parent folder for runs")
    # Run name
    parser.add_argument("--run-name", default="gru", help="Subfolder for this run's files")
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
    parser.add_argument("--batch-size", type=int, default=512, help="Examples per batch")
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
    return parser.parse_args()


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


def main():
    """Scale, model, train, score."""
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
    # Options that define the experiment, saved with the checkpoint
    options = {
        # Model kind
        "kind": "gru",
        # Window
        "history": args.history, "horizon": args.horizon, "max_gap": args.max_gap,
        # Model
        "hidden": args.hidden, "embed": args.embed, "layers": args.layers, "dropout": args.dropout, "gnn_layers": 0, "heads": 0, "predict_delta": args.predict_delta,
        # Graph options, none for the GRU
        "edges": "none", "edge_weights": "none", "min_meters": 1,
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
        scale, count, fallback = sd.compute_scale(args.features_dir, splits, manifest, args.graph if os.path.exists(args.graph) else None, args.scale_percentile, args.scale_floor)
        # Save
        sd.save_scale(scale_path, scale, count, manifest, args.scale_percentile, args.scale_floor, fallback)
        # Say so, naming the channels that used the fallback
        used = [f"{manifest['meters'][mi]}|{manifest['channels'][ci]}" for mi, ci in zip(*np.nonzero(fallback))]
        # Report
        log(f"computed scale from {int(count.sum())} valid train readings, saved to {scale_path}; nominal fallback used for {len(used)} channels" + (f": {used}" if used else ""))
    # Model
    model = sm.make_model("gru", manifest, hidden=args.hidden, embed=args.embed, layers=args.layers, dropout=args.dropout, seed=args.seed, predict_delta=args.predict_delta).to(device)
    # Parameter count
    log(f"model gru with {sum(p.numel() for p in model.parameters())} parameters")
    # Checkpoint path
    checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
    # Train unless scoring only
    if not args.score_only:
        # Train, best weights restored at the end
        best = sm.train(model, args.features_dir, splits, scale, manifest, device, options, checkpoint_path, log)
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
        sm.score_split(model, args.features_dir, splits, split_name, scale, manifest, device, options, run_dir, log)
    # Done
    log("done")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
