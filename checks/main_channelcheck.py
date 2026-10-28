# =============================================================================
# File:        main_channelcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-28
#
# Purpose:     One-off look at one meter over one test block, day by day,
#              to tell three things apart that the alarm diagnostic cannot:
#              a change in the measured current level, a data-quality
#              change (dropouts, rejected angles, odd frequency), and a
#              change in forecasting error. For each chosen channel it
#              prints one line per day with:
#                from the features    median RMS value (volts or amps),
#                                     share of slots present, share of
#                                     present slots with the angle rejected,
#                                     median frequency
#                from the residuals   median model residual, median
#                                     persistence residual (scaled units),
#                                     share of valid slots above the fixed
#                                     calibration threshold
#              Days inside --mark-start/--mark-end are flagged with "*" so a
#              suspected transition (late July 2025) stands out.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root. Reads the
#              feature files directly and reuses socal_score.py for the
#              thresholds, with the same settings check as the alarm check.
#
# Usage:
#   python checks/main_channelcheck.py --run-dir data/processed/models/gcn_full_aw001 --meter egauge_3
#   python checks/main_channelcheck.py --run-dir ... --meter egauge_20 --channels S1 S2 S3 L1 --features-dir /content/features
#   python checks/main_channelcheck.py --run-dir ... --meter egauge_3 --step-days 7
#
# Functions:
#   parse_args()                 command-line options with defaults
#   check_settings(args)         compare the options with the run's score.json settings
#   block_months(label)          feature months covering a block label
#   load_features(features_dir, months, mi, ci)
#                                per-slot RMS, present, angle_ok, freq for one channel
#   daily(t, values, mask, fn)   one number per day from masked per-slot values
#   main()                       print the day-by-day tables
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON for the saved settings
import json
# Standard library: paths
import os
# Standard library: so the project module is found when run from the repo root
import sys
# Third party: numerics
import numpy as np
# Third party: tables and timestamps
import pandas as pd

# Repo root on the path, this file lives in checks/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Project: the scoring method, for thresholds
import socal_score as ss


def parse_args():
    """Read the command-line options. Every setting has a default matching the scoring."""
    # Parser
    parser = argparse.ArgumentParser(description="One meter over one block, day by day: current level, data quality, forecasting error")
    # Run folder
    parser.add_argument("--run-dir", required=True, help="Run folder with the residual tables and score.json")
    # Features
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with <YYYY-MM>.npz and manifest.json")
    # Meter
    parser.add_argument("--meter", required=True, help="Meter name, for example egauge_3")
    # Channels
    parser.add_argument("--channels", nargs="+", default=["S1", "S2", "S3"], help="Channels to print")
    # Block
    parser.add_argument("--label", default="test_normal_2025-04-01_2025-08-31", help="Block label to inspect")
    # Calibration block
    parser.add_argument("--calib-label", default="validation_2025-02-03_2025-02-16", help="Calibration block, as in the scoring")
    # Source
    parser.add_argument("--source", choices=["model", "persistence"], default="model", help="Whose residuals to inspect")
    # Alpha
    parser.add_argument("--alpha", type=float, default=1e-4, help="Conformal level, as in the scoring")
    # Common slots
    parser.add_argument("--common", action=argparse.BooleanOptionalAction, default=True, help="Mask to slots valid for both model and persistence, as in the scoring")
    # Saved settings
    parser.add_argument("--score", default="score.json", help="The scoring output whose settings this check must match")
    # Day step
    parser.add_argument("--step-days", type=int, default=1, help="Days per printed line (1 = daily, 7 = weekly)")
    # Marked range
    parser.add_argument("--mark-start", default="2025-07-20", help="First day of the range to flag")
    # End
    parser.add_argument("--mark-end", default="2025-08-05", help="Last day of the range to flag")
    # Parse
    return parser.parse_args()


def check_settings(args):
    """Read the run's score.json and refuse to run when alpha, calib_label, or common differ from this check's options."""
    # Path
    path = os.path.join(args.run_dir, args.score)
    # Must exist
    if not os.path.exists(path):
        # Raise
        raise FileNotFoundError(f"no {args.score} in {args.run_dir}; run main_score.py first")
    # Read
    with open(path) as f:
        # Parse
        saved = json.load(f)["settings"]
    # Differences
    diffs = {k: (saved.get(k), getattr(args, k)) for k in ("alpha", "calib_label", "common") if saved.get(k) != getattr(args, k)}
    # Any difference is a different experiment
    if diffs:
        # Raise
        raise ValueError(f"options differ from {path} (saved, given): {diffs}")


def block_months(label):
    """Feature months that can hold a block's slots: from the label's start month to the month after its end."""
    # Dates from "<prefix>_<start>_<end>"
    start, end = label.split("_")[-2], label.split("_")[-1]
    # Month range, one past the end since a fixed-PST month file runs to 08:00 UTC of the next month's first day
    months = pd.period_range(pd.Period(start[:7], "M"), pd.Period(end[:7], "M") + 1, freq="M")
    # Strings
    return [str(m) for m in months]


def load_features(features_dir, months, mi, ci):
    """
    Per-slot feature values of one meter and channel over the months:
    t (int64 seconds), rms, present, angle_ok, freq. Months without a
    file are skipped.
    """
    # Pieces
    t, rms, present, angle_ok, freq = [], [], [], [], []
    # Loop over months
    for m in months:
        # Path
        path = os.path.join(features_dir, f"{m}.npz")
        # Skip missing
        if not os.path.exists(path):
            # Next
            continue
        # Read
        d = np.load(path)
        # Keep this channel
        t.append(d["t"]); rms.append(d["rms"][:, mi, ci].astype(np.float32)); present.append(d["mask"][:, mi, ci]); angle_ok.append(d["angle_ok"][:, mi, ci]); freq.append(d["freq"][:, mi].astype(np.float32))
    # Nothing
    if not t:
        # Raise
        raise FileNotFoundError(f"no feature file for months {months} in {features_dir}")
    # Join
    return np.concatenate(t), np.concatenate(rms), np.concatenate(present), np.concatenate(angle_ok), np.concatenate(freq)


def daily(days, values, mask, fn):
    """One number per day label: fn over the masked values of that day, NaN where the day has none."""
    # Frame
    df = pd.DataFrame({"day": days, "v": np.where(mask, values, np.nan)})
    # Group and reduce
    return df.groupby("day")["v"].agg(lambda s: fn(s.dropna()) if s.notna().any() else np.nan)


def main():
    """Print the day-by-day tables."""
    # Options
    args = parse_args()
    # Settings must match the scoring
    check_settings(args)
    # Manifest
    with open(os.path.join(args.features_dir, "manifest.json")) as f:
        # Parse
        manifest = json.load(f)
    # Meter index
    mi = manifest["meters"].index(args.meter)
    # Source prefix and the other
    prefix, other = ("", "pers_") if args.source == "model" else ("pers_", "")
    # Calibration table and thresholds
    cal = ss.load_table(args.run_dir, args.calib_label)
    # Axes
    meters, channels = ss.table_axes(cal)
    # Residuals
    carr = ss.residual_arrays(cal, meters, channels, prefix)
    # Common
    if args.common:
        # Mask
        ss.common_mask(carr, ss.residual_arrays(cal, meters, channels, other))
    # Thresholds
    thr = ss.calibrate(carr, args.alpha)
    # Block table
    table = ss.load_table(args.run_dir, args.label)
    # Residuals
    arr = ss.residual_arrays(table, meters, channels, prefix)
    # Common
    if args.common:
        # Mask
        ss.common_mask(arr, ss.residual_arrays(table, meters, channels, other))
    # Residual timestamps as whole seconds since the epoch, independent of pandas' time resolution
    rt = ((pd.to_datetime(arr["t"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy().astype(np.int64)
    # Block span in seconds
    t0, t1 = int(rt[0]), int(rt[-1])
    # Months
    months = block_months(args.label)
    # Marked range
    mark0, mark1 = pd.Timestamp(args.mark_start, tz="UTC"), pd.Timestamp(args.mark_end, tz="UTC") + pd.Timedelta(days=1)
    # Header
    print(f"{args.meter}, block {args.label}, {args.source} residuals, thresholds from {args.calib_label} at alpha {args.alpha}; * marks {args.mark_start} to {args.mark_end}; days are UTC, {args.step_days} day(s) per line")
    # Meter index in the residual table
    rmi = meters.index(args.meter)
    # Loop over channels
    for ch in args.channels:
        # Indices
        ci, rci = manifest["channels"].index(ch), channels.index(ch)
        # Features for this channel over the block
        ft, rms, present, angle_ok, freq = load_features(args.features_dir, months, mi, ci)
        # Keep the block's span
        keep = (ft >= t0) & (ft <= t1)
        # Cut
        ft, rms, present, angle_ok, freq = ft[keep], rms[keep], present[keep], angle_ok[keep], freq[keep]
        # Day labels, grouped by step
        fdays = pd.to_datetime(ft, unit="s", utc=True).floor(f"{args.step_days}D")
        # Residual day labels
        rdays = pd.to_datetime(rt, unit="s", utc=True).floor(f"{args.step_days}D")
        # Feature summaries
        med_rms = daily(fdays, rms, present & np.isfinite(rms), np.median)
        # Present share
        pres = pd.Series(present.astype(float)).groupby(fdays).mean()
        # Angle rejected share among present
        rej = daily(fdays, (~angle_ok).astype(float), present, np.mean)
        # Frequency median
        med_f = daily(fdays, freq, present & np.isfinite(freq), np.median)
        # Residual summaries
        mres, mval = arr["mag"][:, rmi, rci], arr["mag_valid"][:, rmi, rci]
        # Persistence on the same slots
        parr = ss.residual_arrays(table, meters, channels, other)
        # Persistence residual
        pres_res, pval = parr["mag"][:, rmi, rci], parr["mag_valid"][:, rmi, rci]
        # Model median
        med_m = daily(rdays, mres, mval, np.median)
        # Persistence median
        med_p = daily(rdays, pres_res, pval & mval, np.median)
        # Threshold
        th = thr["thr_mag"][rmi, rci]
        # Exceedance share among valid
        exc = daily(rdays, (mres > th).astype(float) if np.isfinite(th) else np.full(len(mres), np.nan), mval, np.mean)
        # Unit
        unit = "V" if ch.startswith("L") else "A"
        # Title
        print(f"\n=== {args.meter} {ch}: fixed mag threshold {th:.5f} (scaled) ===")
        # Column header
        print(f"{'day':<12} {'rms_med_' + unit:>10} {'present':>8} {'ang_rej':>8} {'freq_med':>9} {'res_med':>9} {'pers_med':>9} {'exceed':>7}")
        # Rows over the union of days
        for d in sorted(set(med_rms.index) | set(med_m.index)):
            # Mark
            flag = "*" if mark0 <= d < mark1 else " "
            # Values with NaN as blanks
            f = lambda s, w, p: (f"{s[d]:>{w}.{p}f}" if d in s.index and np.isfinite(s[d]) else " " * (w - 1) + "-")
            # Print
            print(f"{d.strftime('%Y-%m-%d')}{flag:<2} {f(med_rms, 10, 2)} {f(pres, 8, 3)} {f(rej, 8, 3)} {f(med_f, 9, 3)} {f(med_m, 9, 5)} {f(med_p, 9, 5)} {f(exc, 7, 3)}")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
