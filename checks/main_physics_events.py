# =============================================================================
# File:        main_physics_events.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-20
#
# Purpose:     One-off check: does the physics baseline residual move at
#              the logged switching events? Reads the per-slot residual
#              tables written by main_physics.py and the event table from
#              preprocessing, then prints, for every episode, the residual
#              level before, during, and after, plus the slot-by-slot
#              residual around each logged change.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root.
#
# Usage:
#   python checks/main_physics_events.py
#   python checks/main_physics_events.py --window-minutes 30
#
# Functions:
#   parse_args()                       command-line options with defaults
#   load_residuals(physics_dir, prefix) stack every <prefix>*.parquet result
#   level(res, start, end)             median and max residual in a window
#   around(res, t, minutes)            residual slots within minutes of t
#   main()                             print the comparison per episode
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: file listing
import glob
# Standard library: paths
import os
# Third party: tables
import pandas as pd


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Compare physics residuals against logged events")
    # Folder with main_physics.py results
    parser.add_argument("--physics-dir", default=os.path.join("data", "processed", "physics"), help="Folder with <label>.parquet residual tables")
    # Which result files to read
    parser.add_argument("--prefix", default="test_events", help="Only result files whose name starts with this")
    # Event table
    parser.add_argument("--events", default=os.path.join("data", "processed", "events.csv"), help="Event table from preprocessing")
    # Window length before and after an episode
    parser.add_argument("--window-minutes", type=int, default=60, help="Minutes before and after an episode used for the normal level")
    # Minutes of slot-by-slot detail around each change
    parser.add_argument("--detail-minutes", type=int, default=2, help="Minutes on each side of a change to print slot by slot")
    # Parse
    return parser.parse_args()


def load_residuals(physics_dir, prefix):
    """Read every matching result file and return one table sorted by time."""
    # Matching files
    paths = sorted(glob.glob(os.path.join(physics_dir, f"{prefix}*.parquet")))
    # Need at least one
    if not paths:
        # Raise with the folder tried
        raise FileNotFoundError(f"No {prefix}*.parquet in {physics_dir}")
    # Read and stack
    res = pd.concat([pd.read_parquet(p) for p in paths]).sort_index()
    # Keep scored slots only, in percent per term
    res = res[res["status"] == "ok"].copy()
    # Percent
    res["pct"] = 100.0 * res["residual_mean"]
    # Return
    return res


def level(res, start, end):
    """Median and max residual (percent) for slots in [start, end), or NaN if none."""
    # Slice
    part = res[(res.index >= start) & (res.index < end)]["pct"]
    # Empty
    if part.empty:
        # No data
        return float("nan"), float("nan"), 0
    # Numbers
    return float(part.median()), float(part.max()), int(len(part))


def around(res, t, minutes):
    """Residual slots within minutes of time t, as a Series indexed by time."""
    # Half window
    half = pd.Timedelta(minutes=minutes)
    # Slice
    return res[(res.index >= t - half) & (res.index <= t + half)]["pct"]


def main():
    """Print the residual level before, during, and after each episode."""
    # Options
    args = parse_args()
    # Residuals
    res = load_residuals(args.physics_dir, args.prefix)
    # Events
    ev = pd.read_csv(args.events)
    # UTC times as aware timestamps
    ev["t_utc"] = pd.to_datetime(ev["t_utc"], utc=True)
    # Scored events only
    ev = ev[ev["scored"]].copy()
    # Window as a timedelta
    window = pd.Timedelta(minutes=args.window_minutes)
    # Overall normal level across all scored slots
    print(f"All scored slots: {len(res)}, median {res['pct'].median():.3f} %, p99 {res['pct'].quantile(0.99):.3f} %, max {res['pct'].max():.3f} %\n")
    # Loop over episodes
    for episode, group in ev.groupby("episode"):
        # First and last change
        t0, t1 = group["t_utc"].min(), group["t_utc"].max()
        # Devices involved
        devices = ", ".join(sorted(set(group["device"])))
        # Before: the window ending at the first change
        b_med, b_max, b_n = level(res, t0 - window, t0)
        # During: from the first change to the last change plus the window
        d_med, d_max, d_n = level(res, t0, t1 + window)
        # After: the window starting after that
        a_med, a_max, a_n = level(res, t1 + window, t1 + 2 * window)
        # Header
        print(f"Episode {episode}: {t0} to {t1} UTC, {len(group)} changes: {devices}")
        # Levels
        print(f"  before  median {b_med:7.3f} %  max {b_max:7.3f} %  slots {b_n}")
        # Levels
        print(f"  during  median {d_med:7.3f} %  max {d_max:7.3f} %  slots {d_n}")
        # Levels
        print(f"  after   median {a_med:7.3f} %  max {a_max:7.3f} %  slots {a_n}")
        # Slot-by-slot detail around each change
        for _, row in group.iterrows():
            # Slots near the change
            near = around(res, row["t_utc"], args.detail_minutes)
            # Nothing scored nearby
            if near.empty:
                # Say so
                print(f"    {row['device']} {row['from_status']}->{row['to_status']} at {row['t_utc']}: no scored slots within {args.detail_minutes} min")
                # Next
                continue
            # One line of values
            values = " ".join(f"{v:.2f}" for v in near.to_numpy())
            # Print
            print(f"    {row['device']} {row['from_status']}->{row['to_status']} at {row['t_utc']}: {values}")
        # Blank line between episodes
        print()


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
