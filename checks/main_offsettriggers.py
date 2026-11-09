# =============================================================================
# File:        main_offsettriggers.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-09
#
# Purpose:     Say which meter caused each slot rejected by the physics
#              baseline's offset filter. The filter (main_physics.py
#              --offset-threshold) drops a whole ten-second slot when any
#              meter the plan uses captured its reading at or beyond the
#              threshold from the slot time. The results table records
#              only that the slot was rejected, not which meter did it.
#              This check reads the processed tables for the same meters
#              and window, applies the same rule (socal_physics.
#              offset_rejected, unchanged), and prints per fixed-PST day:
#              the slots rejected, the slots where two or more meters were
#              at or beyond the threshold, and for every meter the slots
#              where it was at or beyond the threshold ("far") and the
#              slots where it was the only meter that was ("only").
#              Several meters can reject the same slot, so the per-meter
#              "far" counts can add to more than the rejected total; the
#              "only" counts cannot. Days with no rejected slot are listed
#              in one line. Nothing is written.
#
# Usage (from the repo root):
#   python checks/main_offsettriggers.py --day 2025-07-29
#   python checks/main_offsettriggers.py --split test_normal --splits splits_v2.json
#   python checks/main_offsettriggers.py --config physics_config_b.json --start 2025-01-13T08:00:00 --end 2025-01-27T08:00:00
#
# Functions:
#   parse_args()                     command-line options with defaults
#   windows_from_args(args)          list of (label, start_utc, end_utc) to read
#   pst_day(index)                   calendar day per slot in fixed PST
#   trigger_table(table, meters, threshold)
#                                    per-day rejected, multi-trigger and per-meter far/only counts
#   print_triggers(rows, meters)     print the daily rows
#   main()                           build the plan, read each window, count, print
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON for the splits file
import json
# Standard library: paths
import os
# Standard library: repo root on the import path
import sys
# Third party: numerics
import numpy as np
# Third party: tables and timestamps
import pandas as pd

# This file lives in checks/; the modules are one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The physics baseline's helpers, used as they are
import socal_physics as sp


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Which meter triggered each slot rejected by the physics baseline's offset filter")
    # Same inputs as main_physics.py
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"), help="Folder with <YYYY-MM>/<meter>.parquet")
    # Topology
    parser.add_argument("--topology-dir", default="topology", help="Caltech topology folder copied into the repo")
    # Caltech code
    parser.add_argument("--caltech-dir", default="caltech_utils", help="Caltech utils folder copied into the repo")
    # Config, decides the planned meters
    parser.add_argument("--config", default="physics_config.json", help="Reference model and injection map; its planned meters are the ones checked")
    # Splits file
    parser.add_argument("--splits", default="splits.json", help="Split definition written by main_splits.py")
    # Named split
    parser.add_argument("--split", default=None, help="Read every block of this split")
    # Fixed-PST day
    parser.add_argument("--day", default=None, help="Fixed-PST calendar day, like 2025-07-29 (08:00 UTC to 08:00 UTC next day)")
    # UTC window
    parser.add_argument("--start", default=None, help="Window start, UTC, like 2025-07-29T08:00:00 (used when --split and --day are not given)")
    # End
    parser.add_argument("--end", default=None, help="Window end, UTC, exclusive")
    # Threshold, the same meaning as in main_physics.py
    parser.add_argument("--offset-threshold", type=float, default=0.5, help="Seconds; a meter triggers when its capture offset is at or beyond this")
    # Parse
    args = parser.parse_args()
    # Exactly one way to give the window
    if sum(x is not None for x in (args.split, args.day, args.start)) != 1 or (args.start is not None and args.end is None):
        # Refuse
        parser.error("give --split, or --day, or both --start and --end")
    # A threshold must be positive
    if args.offset_threshold <= 0:
        # Refuse
        parser.error("--offset-threshold must be a positive number of seconds")
    # Done
    return args


def windows_from_args(args):
    """Return the list of (label, start_utc, end_utc) windows to read, from --split, --day or --start/--end."""
    # Named split, every block
    if args.split:
        # Read splits
        with open(args.splits) as f:
            # Parse
            splits = json.load(f)
        # The split must exist
        if args.split not in splits:
            # Raise
            raise ValueError(f"Split {args.split} not in {args.splits}")
        # One window per block
        return [(f"{args.split}_{b['start']}_{b['end']}", b["start_utc"], b["end_utc_exclusive"]) for b in splits[args.split]]
    # Fixed-PST day: 08:00 UTC to 08:00 UTC the next day
    if args.day:
        # Start
        start = pd.Timestamp(args.day) + pd.Timedelta(hours=8)
        # End
        end = start + pd.Timedelta(days=1)
        # One window
        return [(f"day_{args.day}", start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S"))]
    # Explicit window
    return [(f"window_{args.start}_{args.end}".replace(":", ""), args.start, args.end)]


def pst_day(index):
    """Calendar day of every slot in fixed Pacific Standard Time, UTC minus 8 hours."""
    # Shift to fixed PST, drop the zone, and take the date as text
    return np.asarray((pd.DatetimeIndex(index) - pd.Timedelta(hours=8)).tz_localize(None).strftime("%Y-%m-%d"))


def trigger_table(table, meters, threshold):
    """
    One row per fixed-PST day with: slots, rejected (the filter's own
    verdict from socal_physics.offset_rejected), multi (slots with two or
    more meters at or beyond the threshold), and per meter far (slots
    where this meter was at or beyond the threshold) and only (slots
    where it was the only one).
    """
    # The filter's verdict per slot, the same function the baseline uses
    rejected = sp.offset_rejected(table, meters, threshold)
    # Per-meter trigger flags, one row per meter
    far = np.zeros((len(meters), len(table)), dtype=bool)
    # Loop over meters
    for k, meter in enumerate(meters):
        # Offset column
        col = f"{meter}{sp.SEP}{sp.TOFF}"
        # A meter with no column never triggers
        if col not in table.columns:
            # Next
            continue
        # Offsets, NaN where absent
        toff = table[col].to_numpy(dtype=np.float64)
        # Present and at or beyond the threshold
        far[k] = np.isfinite(toff) & (np.abs(toff) >= threshold)
    # The per-meter flags must reproduce the filter's verdict exactly
    if not np.array_equal(far.any(axis=0), rejected):
        # Refuse rather than print counts that do not add up
        raise ValueError("per-meter trigger flags do not reproduce offset_rejected; the two rules have diverged")
    # Number of triggering meters per slot
    n_trig = far.sum(axis=0)
    # Day key
    day = pst_day(table.index)
    # Rows
    rows = []
    # Each day
    for d in sorted(set(day)):
        # Slots of the day
        sel = day == d
        # Row with the totals
        row = {"day": d, "slots": int(sel.sum()), "rejected": int(rejected[sel].sum()), "multi": int((n_trig[sel] >= 2).sum())}
        # Per meter
        for k, meter in enumerate(meters):
            # Far count
            row[f"{meter}|far"] = int(far[k, sel].sum())
            # Only count: this meter far and no other
            row[f"{meter}|only"] = int((far[k, sel] & (n_trig[sel] == 1)).sum())
        # Store
        rows.append(row)
    # Table
    return pd.DataFrame(rows)


def print_triggers(rows, meters):
    """Print the daily rows; days without a rejected slot are listed together in one line."""
    # Days with nothing rejected
    clean = rows[rows["rejected"] == 0]
    # Days with rejections
    busy = rows[rows["rejected"] > 0]
    # Clean days in one line
    if len(clean):
        # Say
        print(f"\n{len(clean)} day(s) with no rejected slot: " + ", ".join(clean["day"]))
    # Nothing else: still state the total so the window's verdict is explicit
    if busy.empty:
        # Say
        print(f"0 rejected out of {int(rows['slots'].sum())} slots")
        # Done
        return
    # Header
    print(f"\n{'day':<12} {'slots':>6} {'rejected':>8} {'multi':>6}   per meter: far (only)")
    # Each busy day
    for _, r in busy.iterrows():
        # Meters that triggered at least once that day, largest first
        parts = sorted(((r[f"{m}|far"], r[f"{m}|only"], m) for m in meters if r[f"{m}|far"] > 0), reverse=True)
        # Text per meter
        text = ", ".join(f"{m} {f} ({o})" for f, o, m in parts)
        # Line
        print(f"{r['day']:<12} {r['slots']:>6} {r['rejected']:>8} {r['multi']:>6}   {text}")
    # Totals over the window
    print(f"\n{len(rows)} days; rejected slots {int(rows['rejected'].sum())} of {int(rows['slots'].sum())}; slots with two or more triggers {int(rows['multi'].sum())}")
    # Per-meter totals, largest first
    totals = sorted(((int(rows[f'{m}|far'].sum()), int(rows[f'{m}|only'].sum()), m) for m in meters), reverse=True)
    # Say
    print("per meter over the window, far (only): " + ", ".join(f"{m} {f} ({o})" for f, o, m in totals if f > 0))


def main():
    """Build the plan to learn the meters, read each window, count the triggers, print."""
    # Options
    args = parse_args()
    # Config
    config = sp.load_config(args.config)
    # Caltech modules
    ct = sp.setup_caltech(args.caltech_dir, args.topology_dir)
    # Reference model
    model = sp.build_reference_model(ct, args.topology_dir, config)
    # Measurement plan, for its meters
    plan = sp.measurement_plan(model, config)
    # Meters checked, the same set the filter checks
    meters = plan["meters"]
    # Say
    print(f"offset threshold {args.offset_threshold} s, whole slot; meters checked {meters}")
    # Each window
    for label, start_utc, end_utc in windows_from_args(args):
        # Say
        print(f"\n== {label}: {start_utc} to {end_utc} UTC ==")
        # Data
        table = sp.load_block(args.processed_dir, meters, start_utc, end_utc)
        # Empty
        if table.empty:
            # Say
            print("no processed data in this window")
            # Next
            continue
        # Count and print
        print_triggers(trigger_table(table, meters, args.offset_threshold), meters)


# Entry point
if __name__ == "__main__":
    # Run
    main()
