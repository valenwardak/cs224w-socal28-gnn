# =============================================================================
# File:        main_splits.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-18
#
# Purpose:     Write the fixed train / validation / test split to splits.json
#              and report what each split contains: days, coverage, events,
#              episodes. The split is by time blocks, decided before any
#              model is trained. Every later script reads splits.json; nobody
#              hard-codes dates anywhere else.
#
# Split design (approved 2026-10-18):
#              test_events   each event day with a 2-day margin on both sides;
#                            Nov 26 to Dec 7, 2023 treated as one block
#              test_normal   April 1 to August 31, 2025, quiet and well covered;
#                            where false alarms per day are measured
#              validation    two normal blocks, one per year, for early stopping
#                            and conformal calibration
#              train         everything else from 2023-08-15 to 2025-09-03
#              Train and validation must contain no scored event.
#
# Usage:       python main_splits.py
#              python main_splits.py --splits-file splits.json --out-dir data/processed
#              python main_splits.py --help
#
# Inputs:      data/processed/<YYYY-MM>/<meter>.parquet   (mask columns only)
#              data/processed/events.csv
# Outputs:     splits.json   in the repo root, committed
# =============================================================================

# Standard library: command-line argument parsing
import argparse
# Standard library: JSON output
import json
# Standard library: file and folder operations
import os
# Standard library: dates and times
from datetime import date, datetime, timedelta
# Third party: tables and Parquet reading
import pandas as pd

# -----------------------------------------------------------------------------
# Project defaults. Dates are inclusive, in fixed Pacific Standard Time.
# -----------------------------------------------------------------------------

# Processed data folder
DEFAULT_OUT_DIR = os.path.join("data", "processed")
# Where to write the split definition
DEFAULT_SPLITS_FILE = "splits.json"
# Hours added to standard time to get UTC
DEFAULT_UTC_OFFSET_HOURS = 8
# First usable day of the record
DEFAULT_RECORD_START = "2023-08-15"
# Last day of the record
DEFAULT_RECORD_END = "2025-09-03"
# Event test blocks: event day plus 2-day margin each side
DEFAULT_TEST_EVENT_BLOCKS = ["2023-09-15:2023-09-19", "2023-09-24:2023-09-28", "2023-11-24:2023-12-09", "2024-11-11:2024-11-15"]
# Normal test block for false alarms per day
DEFAULT_TEST_NORMAL_BLOCKS = ["2025-04-01:2025-08-31"]
# Validation blocks, one per year
DEFAULT_VALIDATION_BLOCKS = ["2024-01-08:2024-01-21", "2025-02-03:2025-02-16"]


def parse_block(text):
    """Turn 'YYYY-MM-DD:YYYY-MM-DD' into (start_date, end_date)."""
    # Split on the colon
    a, b = text.split(":")
    # Parse both
    return datetime.strptime(a, "%Y-%m-%d").date(), datetime.strptime(b, "%Y-%m-%d").date()


def block_to_utc(start, end, utc_offset_hours):
    """
    Return the UTC-aware [start, end) range covering the inclusive date
    block in standard time.
    """
    # Midnight at the start of the first day, standard time
    s = pd.Timestamp(datetime.combine(start, datetime.min.time())) + pd.Timedelta(hours=utc_offset_hours)
    # Midnight after the last day, standard time
    e = pd.Timestamp(datetime.combine(end + timedelta(days=1), datetime.min.time())) + pd.Timedelta(hours=utc_offset_hours)
    # Label as UTC
    return s.tz_localize("UTC"), e.tz_localize("UTC")


def blocks_overlap(blocks):
    """Return a list of (block_a, block_b) pairs that overlap, if any."""
    # Collect pairs
    bad = []
    # Compare every pair once
    for i in range(len(blocks)):
        # Inner loop
        for j in range(i + 1, len(blocks)):
            # Unpack
            (s1, e1), (s2, e2) = blocks[i], blocks[j]
            # Overlap if neither is entirely before the other
            if not (e1 < s2 or e2 < s1):
                # Record
                bad.append((blocks[i], blocks[j]))
    # Return
    return bad


def train_blocks(record_start, record_end, held_out):
    """
    Return the train blocks: the record with every held-out block removed.
    Each block is an inclusive (start_date, end_date) pair.
    """
    # Sort held-out blocks by start
    held = sorted(held_out)
    # Result
    result = []
    # Cursor begins at the record start
    cursor = record_start
    # Walk the held-out blocks
    for s, e in held:
        # Gap before this block becomes a train block
        if cursor <= s - timedelta(days=1):
            # Add
            result.append((cursor, s - timedelta(days=1)))
        # Move past the block
        cursor = max(cursor, e + timedelta(days=1))
    # Tail after the last block
    if cursor <= record_end:
        # Add
        result.append((cursor, record_end))
    # Return
    return result


def months_in_range(start, end):
    """Yield YYYY-MM labels for every month touching the inclusive date range."""
    # First of the start month
    cur = date(start.year, start.month, 1)
    # Loop
    while cur <= end:
        # Label
        yield cur.strftime("%Y-%m")
        # Next month
        cur = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)


def block_coverage(out_dir, start, end, utc_offset_hours):
    """
    Average coverage inside a block: the mean over all mask columns of all
    meters that have any reading in the block, over the block's slots.
    Returns (fraction, meters_with_data).
    """
    # UTC range
    s_utc, e_utc = block_to_utc(start, end, utc_offset_hours)
    # Per-meter fractions
    fractions = []
    # Meters seen
    meters = set()
    # Loop over months touching the block
    for month in months_in_range(start, end):
        # Month folder
        folder = os.path.join(out_dir, month)
        # Skip missing months
        if not os.path.isdir(folder):
            # Next
            continue
        # Loop over meter files
        for fn in sorted(os.listdir(folder)):
            # Only Parquet
            if not fn.endswith(".parquet"):
                # Next
                continue
            # Meter name
            meter = fn[:-len(".parquet")]
            # Read only the mask columns
            table = pd.read_parquet(os.path.join(folder, fn))
            # Mask columns
            mask_cols = [c for c in table.columns if c.startswith("mask|")]
            # Slice to the block
            part = table.loc[(table.index >= s_utc) & (table.index < e_utc), mask_cols]
            # Skip empty slices
            if len(part) == 0:
                # Next
                continue
            # Channels with any data in this slice
            live = [c for c in mask_cols if part[c].any()]
            # Skip meters with nothing
            if not live:
                # Next
                continue
            # Record the meter
            meters.add(meter)
            # Mean fraction over live channels
            fractions.append(part[live].mean().mean())
    # Overall
    frac = sum(fractions) / len(fractions) if fractions else 0.0
    # Return
    return frac, len(meters)


def events_in_block(events, start, end):
    """Return the scored events whose standard-time day falls inside the block."""
    # Parse the day column
    days = pd.to_datetime(events["day"]).dt.date
    # Inside and scored
    inside = (days >= start) & (days <= end) & events["scored"]
    # Return the subset
    return events[inside]


def parse_args():
    """Read the command-line options and return them as a namespace."""
    # Parser
    parser = argparse.ArgumentParser(description="Write splits.json and report what each split contains.")
    # Processed folder
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="processed data folder")
    # Splits file
    parser.add_argument("--splits-file", default=DEFAULT_SPLITS_FILE, help="where to write the split definition")
    # UTC offset
    parser.add_argument("--utc-offset-hours", type=int, default=DEFAULT_UTC_OFFSET_HOURS, help="hours added to standard time for UTC")
    # Record range
    parser.add_argument("--record-start", default=DEFAULT_RECORD_START, help="first usable day")
    # Record end
    parser.add_argument("--record-end", default=DEFAULT_RECORD_END, help="last day")
    # Blocks
    parser.add_argument("--test-event-blocks", nargs="+", default=DEFAULT_TEST_EVENT_BLOCKS, help="blocks as START:END")
    # Normal test
    parser.add_argument("--test-normal-blocks", nargs="+", default=DEFAULT_TEST_NORMAL_BLOCKS, help="blocks as START:END")
    # Validation
    parser.add_argument("--validation-blocks", nargs="+", default=DEFAULT_VALIDATION_BLOCKS, help="blocks as START:END")
    # Parse
    return parser.parse_args()


def main():
    """Build the split, check it, write splits.json, print the report."""
    # Options
    args = parse_args()
    # Parse blocks
    test_events = [parse_block(b) for b in args.test_event_blocks]
    # Normal test
    test_normal = [parse_block(b) for b in args.test_normal_blocks]
    # Validation
    validation = [parse_block(b) for b in args.validation_blocks]
    # Record range
    record_start = datetime.strptime(args.record_start, "%Y-%m-%d").date()
    # End
    record_end = datetime.strptime(args.record_end, "%Y-%m-%d").date()
    # Everything held out of training
    held_out = test_events + test_normal + validation
    # Overlap check
    bad = blocks_overlap(held_out)
    # Refuse to continue on overlap
    if bad:
        # Report each pair
        for a, b in bad:
            # Line
            print(f"ERROR: blocks overlap: {a} and {b}")
        # Stop
        raise SystemExit(1)
    # Train blocks
    train = train_blocks(record_start, record_end, held_out)

    # Events table
    events = pd.read_csv(os.path.join(args.out_dir, "events.csv"))

    # Build the JSON structure
    def block_dicts(blocks):
        # One dict per block with standard-time dates and UTC bounds
        out = []
        # Loop
        for s, e in blocks:
            # UTC bounds
            s_utc, e_utc = block_to_utc(s, e, args.utc_offset_hours)
            # Dict
            out.append({"start": s.isoformat(), "end": e.isoformat(), "start_utc": s_utc.isoformat(), "end_utc_exclusive": e_utc.isoformat()})
        # Return
        return out
    # The whole definition
    splits = {
        # How to read the dates
        "time_base": "Dates are inclusive calendar days in fixed Pacific Standard Time (UTC-8, no daylight saving). UTC bounds are given for each block; end_utc_exclusive is the first instant after the block.",
        # Offset used
        "utc_offset_hours": args.utc_offset_hours,
        # Record
        "record": {"start": record_start.isoformat(), "end": record_end.isoformat()},
        # Splits
        "train": block_dicts(train),
        "validation": block_dicts(validation),
        "test_events": block_dicts(test_events),
        "test_normal": block_dicts(test_normal),
        # Scoring rule
        "scoring": {"excluded_event_days": sorted(set(events.loc[~events["scored"], "day"])), "event_margin_days": 2},
    }
    # Write
    with open(args.splits_file, "w") as f:
        # Readable JSON
        json.dump(splits, f, indent=2)
    # Report header
    print(f"wrote {args.splits_file}\n")
    # Column header
    print(f"{'split':12s} {'block':25s} {'days':>5s} {'coverage':>9s} {'meters':>7s} {'events':>7s} {'episodes':>9s}")
    # Loop over splits
    for name, blocks in [("train", train), ("validation", validation), ("test_events", test_events), ("test_normal", test_normal)]:
        # Totals
        total_days = 0
        # Events
        total_events = 0
        # Episodes
        episode_ids = set()
        # Loop over blocks
        for s, e in blocks:
            # Days
            days = (e - s).days + 1
            # Coverage
            frac, n_meters = block_coverage(args.out_dir, s, e, args.utc_offset_hours)
            # Events
            ev = events_in_block(events, s, e)
            # Accumulate
            total_days += days
            # Events
            total_events += len(ev)
            # Episodes
            episode_ids |= set(ev["episode"])
            # Row
            print(f"{name:12s} {s.isoformat()} to {e.isoformat()} {days:5d} {frac:9.1%} {n_meters:7d} {len(ev):7d} {len(set(ev['episode'])):9d}")
        # Split total
        print(f"{name:12s} {'TOTAL':25s} {total_days:5d} {'':>9s} {'':>7s} {total_events:7d} {len(episode_ids):9d}\n")
        # Warn if train or validation holds any scored event
        if name in ("train", "validation") and total_events > 0:
            # Loud
            print(f"WARNING: {name} contains {total_events} scored event changes. Fix the blocks.\n")


# Run main() only when executed as a script
if __name__ == "__main__":
    main()