# =============================================================================
# File:        main_forecastcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-29
#
# Purpose:     One-off look at what a trained model actually forecasts for
#              one meter and channel over a few chosen windows, in amps or
#              volts, next to the reading and the persistence forecast.
#              The residual tables hold only the size of the error, so this
#              check loads the run's best weights and runs the model again.
#              It was written for the July 29 to August 5, 2025 period when
#              egauge_3 and egauge_20 carried far higher current than usual
#              and the model residuals were twenty times persistence's.
#
#              For every chosen window it prints, per slot (thinned by
#              --every):
#                actual        the reading
#                pers          persistence forecast, the last valid reading
#                model         the model's forecast
#                corr          model minus pers, the signed learned
#                              correction; positive means the model expects
#                              the value to rise
#                last_in       the last valid history magnitude divided by
#                              the scale: what the model actually received
#                              as its most recent input
#                target_sc     the reading being forecast divided by the
#                              scale, to place it against the training range
#              and a window summary: means of the four, the mean absolute
#              error of model and persistence, the share of slots where the
#              correction points toward the training median, and how many
#              of the window's slots the scorer would have kept under its
#              gap rule (max_gap), since this check forecasts every slot.
#
#              It also prints the channel's training distribution, from the
#              checkpoint's own train_blocks (the effective training blocks,
#              after any --train-months cut), valid magnitudes only: median,
#              95th and 99th percentile, maximum, and the scale.
#
#              The graph models forecast every meter at once from every
#              meter's history, so inference runs on all meters and only
#              the chosen channel is printed. The GRU forecasts the chosen
#              meter alone.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root. Needs torch,
#              the features, and the run folder; meant for Colab.
#
# Usage:
#   python checks/main_forecastcheck.py --run-dir /content/drive/MyDrive/socal_runs/gcn_full_aw001 --features-dir /content/features --meter egauge_3 --channel S1
#   python checks/main_forecastcheck.py --run-dir ... --meter egauge_20 --channel S1 --windows 2025-07-28T20:00 2025-08-09T00:00 --hours 2
#   python checks/main_forecastcheck.py --run-dir ... --meter egauge_3 --channel S1 --every 1
#
# Functions:
#   parse_args()                       command-line options with defaults
#   pick_device()                      cuda if present, else cpu
#   load_run(run_dir, manifest, graph_path, device)
#                                      options, model with best weights, scale
#   month_of(ts)                       fixed-PST feature month holding a UTC timestamp
#   window_arrays(features_dir, manifest, scale, start, hours, history)
#                                      feature rows covering a window plus its history
#   forecasts(model, options, X, mag_valid, ang_valid, mi, history, horizon, device)
#                                      per-slot model and persistence magnitude forecasts (scaled)
#   eligible(mag_valid, mi, history, horizon, max_gap)
#                                      which target slots the scorer's gap rule would keep
#   training_distribution(features_dir, manifest, options, mi, ci)
#                                      percentiles of the channel over the effective train blocks
#   main()                             run the windows and print
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: paths
import os
# Standard library: so the project modules are found when run from the repo root
import sys
# Third party: numerics
import numpy as np
# Third party: timestamps
import pandas as pd
# Third party: deep learning
import torch

# Repo root on the path, this file lives in checks/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Project: data side
import socal_modeldata as sd
# Project: models
import socal_models as sm


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="What a trained model forecasts for one channel over chosen windows, in physical units")
    # Run folder
    parser.add_argument("--run-dir", required=True, help="Run folder with checkpoint.pt, checkpoint_best.pt, scale.json")
    # Features
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with <YYYY-MM>.npz and manifest.json")
    # Graph
    parser.add_argument("--graph", default="graph.json", help="Graph file, needed for the graph models")
    # Meter
    parser.add_argument("--meter", required=True, help="Meter name, for example egauge_3")
    # Channel
    parser.add_argument("--channel", default="S1", help="Channel, for example S1")
    # Windows
    parser.add_argument("--windows", nargs="+", default=["2025-07-27T02:00", "2025-07-29T02:00", "2025-08-01T02:00", "2025-08-09T02:00", "2025-08-12T02:00"], help="Window start times, UTC, YYYY-MM-DDTHH:MM")
    # Window length
    parser.add_argument("--hours", type=float, default=1.0, help="Length of each window in hours")
    # Thinning
    parser.add_argument("--every", type=int, default=6, help="Print every this many slots (6 = one line per minute)")
    # Parse
    return parser.parse_args()


def pick_device():
    """cuda when present, else cpu; mps is avoided since this is meant for Colab."""
    # Choose
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_run(run_dir, manifest, graph_path, device):
    """
    Read the checkpoint's options, build the model they describe, load
    the best weights, and read the scale. Returns (options, model, scale).
    """
    # Checkpoint with the options
    body = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location=device, weights_only=False)
    # Options
    options = body["options"]
    # Graph for the graph models
    graph = sm.load_graph(graph_path) if options["kind"] != "gru" else None
    # Model as trained
    model = sm.make_model(options["kind"], manifest, graph, edges=options["edges"], edge_weights=options["edge_weights"], hidden=options["hidden"], embed=options["embed"], layers=options["layers"], dropout=options["dropout"], heads=options["heads"], seed=options["seed"], gnn_layers=options["gnn_layers"], predict_delta=options["predict_delta"]).to(device)
    # Best weights
    model.load_state_dict(torch.load(os.path.join(run_dir, "checkpoint_best.pt"), map_location=device))
    # Evaluation mode
    model.eval()
    # Scale, checked against the manifest
    scale = sd.load_scale(os.path.join(run_dir, "scale.json"), manifest)
    # Return
    return options, model, scale


def month_of(ts):
    """The fixed-PST feature month holding a UTC timestamp: the month of (ts - 8 hours)."""
    # Shifted
    return (ts - pd.Timedelta(seconds=sd.UTC_OFFSET_SECONDS)).strftime("%Y-%m")


def window_arrays(features_dir, manifest, scale, start, hours, history):
    """
    Feature rows from `history` slots before the window start to the
    window end, with the model input built. Loads the one or two months
    involved. Returns (t, X, mag_valid, ang_valid, rms) or None when the
    window has no rows.
    """
    # Bounds in seconds, history included
    start_s = int(start.timestamp()) - history * sd.SLOT_SECONDS
    # End
    end_s = int(start.timestamp()) + int(hours * 3600)
    # Months that can hold the rows
    months = sorted({month_of(pd.Timestamp(start_s, unit="s", tz="UTC")), month_of(pd.Timestamp(end_s, unit="s", tz="UTC"))})
    # Pieces
    parts = []
    # Loop over months
    for m in months:
        # Load, missing allowed
        arrays = sd.load_month(features_dir, m, allow_missing=True, manifest=manifest)
        # Skip missing
        if arrays is None:
            # Next
            continue
        # Rows inside
        part = sd.slice_block(arrays, start_s, end_s)
        # Keep
        if part is not None:
            # Add
            parts.append(part)
    # Nothing
    if not parts:
        # None
        return None
    # Join months
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # Model input
    X, mag_valid, ang_valid = sd.build_features(arrays, scale)
    # Return
    return arrays["t"], X, mag_valid, ang_valid, arrays["rms"]


def forecasts(model, options, X, mag_valid, ang_valid, mi, history, horizon, device):
    """
    Model and persistence magnitude forecasts (scaled units) for every
    target slot that has a full history in X, for meter mi, all channels.
    Returns (targets, model_mag, pers_mag, pers_ok) with targets the slot
    indices into X. The graph models run on every meter; the GRU on mi.
    """
    # Sizes
    T, M, C, _ = X.shape
    # Window starts with a full history and a target inside
    starts = np.arange(0, T - history - horizon + 1)
    # Target slots
    targets = starts + history + horizon - 1
    # Nothing
    if len(starts) == 0:
        # Empty
        return targets, np.zeros((0, C)), np.zeros((0, C)), np.zeros((0, C), dtype=bool)
    # Outputs
    model_mag, pers_mag, pers_ok = [], [], []
    # Batches of starts, bounded memory
    with torch.no_grad():
        # Loop
        for b0 in range(0, len(starts), 256):
            # Batch starts
            s = starts[b0:b0 + 256]
            # History rows: (B, history)
            rows = s[:, None] + np.arange(history)[None, :]
            # Graph model: every meter
            if options["kind"] != "gru":
                # (B, history, M, C*F)
                x = torch.from_numpy(X[rows].reshape(len(s), history, M, -1)).to(device)
                # Forecast (B, M, C, 3), keep the meter
                out = model(x)[:, mi]
                # Persistence on the same histories, keep the meter
                base, ok = sm.persistence_forecast(x, C)
                # Meter
                base, ok = base[:, mi], ok[:, mi]
            # GRU: this meter only
            else:
                # (B, history, C*F)
                x = torch.from_numpy(X[rows, mi].reshape(len(s), history, -1)).to(device)
                # Meter ids
                meter = torch.full((len(s),), mi, dtype=torch.long, device=device)
                # Forecast (B, C, 3)
                out = model(x, meter)
                # Persistence
                base, ok = sm.persistence_forecast(x, C)
            # Magnitudes
            model_mag.append(out[..., 0].cpu().numpy()); pers_mag.append(base[..., 0].cpu().numpy()); pers_ok.append(ok[..., 0].cpu().numpy())
    # Join
    return targets, np.concatenate(model_mag), np.concatenate(pers_mag), np.concatenate(pers_ok)


def eligible(mag_valid, mi, history, horizon, max_gap):
    """
    Which window starts the scorer's gap rule keeps for meter mi: the same
    rule as WindowDataset, a run of max_gap absent slots inside the history
    drops the window, and the target must be present. Returns the kept
    start indices.
    """
    # Any reading in the slot for this meter
    present_any = mag_valid[:, mi, :].any(axis=1)
    # Starts kept
    return sd.window_starts(present_any, history, horizon, max_gap, present_any)


def training_distribution(features_dir, manifest, options, mi, ci):
    """
    Percentiles of one channel's valid magnitudes over the checkpoint's
    effective training blocks (options["train_blocks"], "start/end"
    strings). Returns (n, median, p95, p99, max) in physical units.
    """
    # Values
    vals = []
    # Loop over blocks
    for span in options["train_blocks"]:
        # Bounds
        start_utc, end_utc = span.split("/")
        # Block dictionary in the splits format
        block = {"start_utc": start_utc, "end_utc_exclusive": end_utc}
        # Seconds
        start_s, end_s = sd.block_seconds(block)
        # Months
        for m in sd.months_in_block(block):
            # Load; a missing training month would make the distribution incomplete, so it raises
            arrays = sd.load_month(features_dir, m, allow_missing=False, manifest=manifest)
            # Rows inside
            part = sd.slice_block(arrays, start_s, end_s)
            # Skip
            if part is None:
                # Next
                continue
            # Valid magnitudes of the channel
            v = part["rms"][:, mi, ci][part["mag_valid"][:, mi, ci]]
            # Keep
            vals.append(v.astype(np.float32))
    # Join, empty when every piece was empty
    v = np.concatenate(vals) if vals else np.zeros(0, dtype=np.float32)
    # No valid reading at all
    if len(v) == 0:
        # Empty
        return 0, None, None, None, None
    # Stats
    return len(v), float(np.median(v)), float(np.percentile(v, 95)), float(np.percentile(v, 99)), float(v.max())


def main():
    """Run the windows and print."""
    # Options
    args = parse_args()
    # Device
    device = pick_device()
    # Manifest
    manifest = sd.load_manifest(args.features_dir)
    # Indices
    mi, ci = manifest["meters"].index(args.meter), manifest["channels"].index(args.channel)
    # Run
    options, model, scale = load_run(args.run_dir, manifest, args.graph, device)
    # Scale of the channel
    sc = float(scale[mi, ci])
    # Unit
    unit = "V" if args.channel.startswith("L") else "A"
    # Header
    print(f"{args.meter} {args.channel}, run {args.run_dir}, model {options['kind']}, history {options['history']} slots, horizon {options['horizon']}, predict_delta {options['predict_delta']}, device {device}")
    # Training distribution
    n, med, p95, p99, mx = training_distribution(args.features_dir, manifest, options, mi, ci)
    # Print, or say there was nothing
    if n == 0:
        # None
        print(f"training distribution over {len(options['train_blocks'])} effective train blocks: no valid reading for this channel (scale {sc:.2f} {unit} came from the fallback)")
    else:
        # Stats
        print(f"training distribution over {len(options['train_blocks'])} effective train blocks, {n} valid readings: median {med:.2f} {unit}, p95 {p95:.2f}, p99 {p99:.2f}, max {mx:.2f}; scale {sc:.2f} {unit} (so max input {mx / sc:.2f})")
    # Loop over windows
    for w in args.windows:
        # Start
        start = pd.Timestamp(w, tz="UTC")
        # Rows
        got = window_arrays(args.features_dir, manifest, scale, start, args.hours, options["history"])
        # Nothing
        if got is None:
            # Say so
            print(f"\n=== window {w} + {args.hours} h: no feature rows ===")
            # Next
            continue
        # Unpack
        t, X, mag_valid, ang_valid, rms = got
        # Forecasts
        targets, model_mag, pers_mag, pers_ok = forecasts(model, options, X, mag_valid, ang_valid, mi, options["history"], options["horizon"], device)
        # Title
        print(f"\n=== window {w} + {args.hours} h: {len(targets)} target slots ===")
        # Starts the scorer's gap rule would keep, for the flag
        kept = set(eligible(mag_valid, mi, options["history"], options["horizon"], options["max_gap"]).tolist())
        # Column header
        print(f"{'time_utc':<20} {'actual_' + unit:>9} {'pers':>9} {'model':>9} {'corr':>9} {'last_in':>8} {'target_sc':>9} {'gap':>4}")
        # Collect for the summary
        rows = []
        # Loop over targets
        for k, ti in enumerate(targets):
            # Actual in units, NaN when not valid
            actual = float(rms[ti, mi, ci]) if mag_valid[ti, mi, ci] else float("nan")
            # Persistence in units, NaN when nothing to copy
            pers = float(pers_mag[k, ci]) * sc if pers_ok[k, ci] else float("nan")
            # Model in units
            mod = float(model_mag[k, ci]) * sc
            # Correction
            corr = mod - pers
            # The model's most recent input: the last valid history magnitude over the scale, which is the persistence value scaled
            last_in = float(pers_mag[k, ci]) if pers_ok[k, ci] else float("nan")
            # The target over the scale, to place it against the training range
            target_sc = actual / sc
            # Whether the scorer's gap rule keeps this window
            keep = (ti - options["history"] - options["horizon"] + 1) in kept
            # Keep
            rows.append((actual, pers, mod, corr, float(keep)))
            # Thinned print
            if k % args.every == 0:
                # Format
                f = lambda v, w=9: " " * (w - 3) + "nan" if not np.isfinite(v) else f"{v:>{w}.2f}"
                # Print
                print(f"{pd.Timestamp(int(t[ti]), unit='s', tz='UTC').strftime('%Y-%m-%d %H:%M:%S'):<20} {f(actual)} {f(pers)} {f(mod)} {f(corr)} {f(last_in, 8)} {f(target_sc)} {'ok' if keep else 'drop':>4}")
        # No target slot had a full history: nothing to summarize
        if not rows:
            # Say so
            print("no target slot with a full history in this window")
            # Next window
            continue
        # Summary
        a = np.array(rows, dtype=float)
        # Valid rows
        ok = np.isfinite(a[:, :4]).all(axis=1)
        # Any
        if ok.any():
            # Means
            m = a[ok].mean(axis=0)
            # Share of corrections pointing toward the training median
            toward = float(np.mean(np.sign(a[ok, 3]) == np.sign(med - a[ok, 1]))) if med is not None else float("nan")
            # Print
            print(f"summary over {int(ok.sum())} slots: actual {m[0]:.2f} {unit}, pers {m[1]:.2f}, model {m[2]:.2f}, corr {m[3]:+.2f}; |model-actual| {np.abs(a[ok, 2] - a[ok, 0]).mean():.2f} vs |pers-actual| {np.abs(a[ok, 1] - a[ok, 0]).mean():.2f}; corrections toward training median {toward:.2f}; slots the scorer's gap rule keeps {int(a[:, 4].sum())} of {len(a)}")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
