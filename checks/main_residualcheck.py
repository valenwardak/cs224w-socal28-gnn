# =============================================================================
# File:        main_residualcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-26
#
# Purpose:     One-off look at the residual distributions a trained run
#              wrote for one validation block, so a model's residuals can
#              be compared with persistence beyond the two medians the
#              training log prints. For every meter and channel, and for
#              the whole block, it prints:
#                median and 99th percentile of the magnitude residual,
#                  model and persistence
#                median and 99th percentile of the angle residual,
#                  model and persistence
#                the share of slots where the model's magnitude residual is
#                  below persistence's
#              Only slots where both the model and persistence have a valid
#              residual are compared, so the two sides see the same slots.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root. Reads the
#              Parquet tables written by score_split in socal_models.py.
#
# Usage:
#   python checks/main_residualcheck.py --run-dir data/processed/models/gru_full_aw001
#   python checks/main_residualcheck.py --run-dir /content/drive/MyDrive/socal_runs/gcn_full_aw001 --label validation_2024-01-08_2024-01-21
#   python checks/main_residualcheck.py --run-dir ... --csv residuals_gcn.csv
#
# Functions:
#   parse_args()                 command-line options with defaults
#   stats(model, pers)           median, p99, and win share for one pair of residual arrays
#   per_channel(table, meters, channels, kind)
#                                one row of stats per meter and channel
#   overall(table, kind)         stats over every meter and channel at once
#   main()                       read the table, print the rows, write the CSV if asked
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: paths
import os
# Third party: numerics
import numpy as np
# Third party: tables
import pandas as pd


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Compare a run's residuals with persistence on one validation block")
    # Run folder with the Parquet tables
    parser.add_argument("--run-dir", required=True, help="Run folder written by main_gru.py or main_gcn.py")
    # Block label, the model-selection block by default
    parser.add_argument("--label", default="validation_2024-01-08_2024-01-21", help="Block label, the Parquet file name without the extension")
    # Optional CSV
    parser.add_argument("--csv", default=None, help="Write the per-channel rows to this CSV as well")
    # Parse
    return parser.parse_args()


def stats(model, pers):
    """
    Median, 99th percentile, and win share for one pair of residual
    arrays. Only positions where both are finite count. Returns a
    dictionary, with None where nothing was comparable.
    """
    # Positions valid on both sides
    both = np.isfinite(model) & np.isfinite(pers)
    # Nothing to compare
    if not both.any():
        # Empty
        return {"n": 0, "model_median": None, "model_p99": None, "pers_median": None, "pers_p99": None, "win_share": None}
    # Comparable values
    m, p = model[both], pers[both]
    # Stats
    return {
        # Count
        "n": int(both.sum()),
        # Model median
        "model_median": float(np.median(m)),
        # Model 99th percentile
        "model_p99": float(np.percentile(m, 99)),
        # Persistence median
        "pers_median": float(np.median(p)),
        # Persistence 99th percentile
        "pers_p99": float(np.percentile(p, 99)),
        # Share of slots where the model is strictly better
        "win_share": float(np.mean(m < p)),
    }


def per_channel(table, meters, channels, kind):
    """
    One row per meter and channel for one residual kind, "mag" or "ang":
    the stats of the model column against the persistence column.
    """
    # Rows
    rows = []
    # Loop over meters
    for meter in meters:
        # Loop over channels
        for ch in channels:
            # Model column
            mcol = f"{meter}|{ch}|{kind}_res"
            # Persistence column
            pcol = f"{meter}|{ch}|pers_{kind}_res"
            # Skip channels the table does not have
            if mcol not in table.columns or pcol not in table.columns:
                # Next
                continue
            # Stats
            row = stats(table[mcol].to_numpy(), table[pcol].to_numpy())
            # Label
            row.update({"meter": meter, "channel": ch, "kind": kind})
            # Keep
            rows.append(row)
    # Table
    return pd.DataFrame(rows)


def overall(table, kind):
    """Stats over every meter and channel at once for one residual kind."""
    # Model columns
    mcols = [c for c in table.columns if c.endswith(f"|{kind}_res")]
    # Persistence columns in the same order
    pcols = [c.replace(f"|{kind}_res", f"|pers_{kind}_res") for c in mcols]
    # Flatten both sides the same way
    return stats(table[mcols].to_numpy().ravel(), table[pcols].to_numpy().ravel())


def main():
    """Read the table, print the per-channel rows and the totals, write the CSV if asked."""
    # Options
    args = parse_args()
    # Path
    path = os.path.join(args.run_dir, f"{args.label}.parquet")
    # Must exist
    if not os.path.exists(path):
        # Raise
        raise FileNotFoundError(f"no residual table at {path}")
    # Read
    table = pd.read_parquet(path)
    # Meters and channels from the column names, in first-seen order
    meters, channels = [], []
    # Loop over columns
    for col in table.columns:
        # Split "<meter>|<channel>|<name>"
        meter, ch, _ = col.split("|")
        # Collect
        if meter not in meters:
            # New meter
            meters.append(meter)
        # Collect
        if ch not in channels:
            # New channel
            channels.append(ch)
    # Header
    print(f"{args.label}: {len(table)} slots, {len(meters)} meters, {len(channels)} channels\n")
    # Both kinds
    frames = []
    # Loop over kinds
    for kind, unit in (("mag", "scaled units"), ("ang", "degrees")):
        # Rows
        df = per_channel(table, meters, channels, kind)
        # Keep for the CSV
        frames.append(df)
        # Title
        print(f"=== {kind} residual ({unit}): model vs persistence, per meter and channel ===")
        # Column header
        print(f"{'meter':<10} {'ch':<3} {'n':>8} {'model_med':>10} {'pers_med':>10} {'model_p99':>10} {'pers_p99':>10} {'win':>6}")
        # Rows
        for _, r in df.iterrows():
            # Skip empty
            if r["n"] == 0:
                # Say so
                print(f"{r['meter']:<10} {r['channel']:<3} {0:>8}  (no comparable slots)")
                # Next
                continue
            # Print
            print(f"{r['meter']:<10} {r['channel']:<3} {r['n']:>8} {r['model_median']:>10.5f} {r['pers_median']:>10.5f} {r['model_p99']:>10.5f} {r['pers_p99']:>10.5f} {r['win_share']:>6.2f}")
        # Totals
        tot = overall(table, kind)
        # Print
        print(f"{'ALL':<10} {'':<3} {tot['n']:>8} {tot['model_median']:>10.5f} {tot['pers_median']:>10.5f} {tot['model_p99']:>10.5f} {tot['pers_p99']:>10.5f} {tot['win_share']:>6.2f}\n")
    # CSV if asked
    if args.csv:
        # Stack both kinds
        pd.concat(frames).to_csv(args.csv, index=False)
        # Say so
        print(f"wrote {args.csv}")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
