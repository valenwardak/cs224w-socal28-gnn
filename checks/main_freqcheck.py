# =============================================================================
# File:        main_freqcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-21
#
# Purpose:     One-off check: which channels report a frequency that
#              disagrees with the meter's L1 voltage channel, by how much,
#              and whether those slots carry a small current. Reads one
#              month of processed Parquet for every meter.
#
# Usage:
#   python checks/main_freqcheck.py
#   python checks/main_freqcheck.py --month 2024-09 --tolerance 0.01
#
# Functions:
#   parse_args()   command-line options with defaults
#   main()         print one line per meter and channel
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
    parser = argparse.ArgumentParser(description="Check channel frequency disagreements against L1")
    # Processed folder
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"), help="Folder with <YYYY-MM>/<meter>.parquet")
    # Month to read
    parser.add_argument("--month", default="2024-09", help="Month folder to read")
    # Meter list
    parser.add_argument("--meters-file", default="meters.txt", help="Meter names, one per line")
    # Tolerance
    parser.add_argument("--tolerance", type=float, default=0.01, help="Hertz; larger differences count as disagreements")
    # Parse
    return parser.parse_args()


def main():
    """For each meter and channel, report how often and how far the frequency differs from L1."""
    # Options
    args = parse_args()
    # Meters
    with open(args.meters_file) as f:
        # Names, skipping blanks and comments
        meters = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    # Header
    print(f"{'meter':10s} {'ch':3s} {'slots':>7s} {'disagree':>8s} {'frac':>6s} {'median_abs_diff':>15s} {'max_abs_diff':>12s} {'median_rms_disagree':>19s} {'median_rms_agree':>16s}")
    # Loop over meters
    for meter in meters:
        # File
        path = os.path.join(args.processed_dir, args.month, f"{meter}.parquet")
        # Skip missing
        if not os.path.exists(path):
            # Next
            continue
        # Read
        df = pd.read_parquet(path)
        # Reference frequency column
        ref_col = f"{meter}|L1|freq"
        # Skip meters without L1
        if ref_col not in df.columns:
            # Say so
            print(f"{meter:10s} no L1 frequency")
            # Next
            continue
        # Reference
        ref = df[ref_col].to_numpy(dtype=np.float64)
        # Loop over channels
        for ch in ("L2", "L3", "S1", "S2", "S3"):
            # Columns
            fcol, rcol = f"{meter}|{ch}|freq", f"{meter}|{ch}|rms"
            # Skip absent channels
            if fcol not in df.columns:
                # Next
                continue
            # Values
            freq = df[fcol].to_numpy(dtype=np.float64)
            # Magnitude
            rms = df[rcol].to_numpy(dtype=np.float64)
            # Slots where both are present
            both = ~np.isnan(freq) & ~np.isnan(ref)
            # Nothing to compare
            if both.sum() == 0:
                # Next
                continue
            # Absolute difference
            diff = np.abs(freq[both] - ref[both])
            # Disagreement flags
            bad = diff > args.tolerance
            # Numbers
            n, nb = int(both.sum()), int(bad.sum())
            # Median and max difference among disagreements
            med = float(np.median(diff[bad])) if nb else 0.0
            # Max
            mx = float(diff.max())
            # Median rms on disagreeing and agreeing slots
            rms_bad = float(np.nanmedian(rms[both][bad])) if nb else float("nan")
            # Agreeing
            rms_good = float(np.nanmedian(rms[both][~bad])) if nb < n else float("nan")
            # Print
            print(f"{meter:10s} {ch:3s} {n:7d} {nb:8d} {nb / n:6.3f} {med:15.3f} {mx:12.3f} {rms_bad:19.1f} {rms_good:16.1f}")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
