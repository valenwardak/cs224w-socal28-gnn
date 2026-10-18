# =============================================================================
# File:        main_cb146.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-18
#
# Purpose:     Decisive log-time check. egauge_7's current sensors sit on
#              breaker cb_146. The operator log says cb_146 opened on
#              2023-09-17 at 14:04:17 and closed on 2023-09-26 at 14:49:50.
#              This script prints egauge_7's per-minute average current on
#              both days around the logged minute and around the same
#              minute one hour earlier. The minute where the current drops
#              to near zero (Sept 17) or appears from near zero (Sept 26)
#              is the true event time in meter time. Read only.
#
# How to read the result:
#              Change at 14:04 and 14:49: log matches meter time.
#              Change at 13:04 and 13:49: log is one hour ahead of meters.
#
# Usage:       python main_cb146.py
#              python main_cb146.py --meter egauge_7 --channels S1 S2 S3
#              python main_cb146.py --help
#
# Data layout read:
#              <data_dir>/<meter>/<start>_<end>[_<channel>]/phasors/<meter>-<channel>.csv
#              Columns read: t and rms.
# =============================================================================

# Standard library: command-line argument parsing
import argparse
# Standard library: file and folder operations
import os
# Standard library: dates and times
from datetime import datetime, timedelta
# Third party: CSV reading and arithmetic
import pandas as pd

# -----------------------------------------------------------------------------
# Project defaults. Every one can be overridden on the command line.
# -----------------------------------------------------------------------------

# Folder holding the merged download
DEFAULT_DATA_DIR = "data"
# Meter whose current sensors are on cb_146
DEFAULT_METER = "egauge_7"
# Current channels
DEFAULT_CHANNELS = ["S1", "S2", "S3"]
# Logged event times for cb_146
DEFAULT_EVENTS = ["2023-09-17 14:04:17", "2023-09-26 14:49:50"]
# Minutes to print before and after each candidate minute
DEFAULT_SPAN_MINUTES = 8
# Alternative offset to test, in hours (negative means earlier)
DEFAULT_SHIFT_HOURS = -1


def parse_folder_label(label):
    """
    Split a request folder name into (start_date, end_date, channel).

    Returns None for names that do not fit, such as "_failed".
    """
    # Split on underscores
    parts = label.split("_")
    # Need at least two date parts
    if len(parts) < 2:
        # Not a request folder
        return None
    # Try to parse the two dates
    try:
        # Start date
        start = datetime.strptime(parts[0], "%Y-%m-%d").date()
        # End date
        end = datetime.strptime(parts[1], "%Y-%m-%d").date()
    # Anything that is not a date pair
    except ValueError:
        # Not a request folder
        return None
    # Optional channel suffix
    channel = parts[2] if len(parts) > 2 else None
    # Return the pieces
    return start, end, channel


def find_channel_files(data_dir, meter, day, channel):
    """
    Return the paths of every CSV for one meter and channel whose request
    folder covers the given day. Fallback folders are included.
    """
    # Meter folder
    meter_dir = os.path.join(data_dir, meter)
    # No folder, no files
    if not os.path.isdir(meter_dir):
        # Empty result
        return []
    # Collect matching files
    paths = []
    # Loop over request folders
    for label in os.listdir(meter_dir):
        # Parse the name
        parsed = parse_folder_label(label)
        # Skip non-request folders
        if parsed is None:
            # Next
            continue
        # Unpack
        start, end, folder_channel = parsed
        # Skip folders that do not cover the day
        if not (start <= day <= end):
            # Next
            continue
        # Skip single-channel folders for a different channel
        if folder_channel is not None and folder_channel != channel:
            # Next
            continue
        # Expected CSV path inside the folder
        path = os.path.join(meter_dir, label, "phasors", f"{meter}-{channel}.csv")
        # Keep it if it exists
        if os.path.exists(path):
            # Add
            paths.append(path)
    # Return all matches
    return paths


def load_rms(paths):
    """
    Read columns t and rms from every file and return one Series of rms
    indexed by timestamp, sorted, with duplicates removed.
    """
    # Nothing to read
    if not paths:
        # Empty series
        return pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    # Read only the two columns from each file
    frames = [pd.read_csv(p, usecols=["t", "rms"]) for p in paths]
    # Stack them
    df = pd.concat(frames, ignore_index=True)
    # Parse timestamps, bad rows become NaT
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    # Drop bad rows and duplicate timestamps, sort
    df = df.dropna(subset=["t"]).drop_duplicates(subset=["t"]).sort_values("t")
    # Series indexed by time
    return df.set_index("t")["rms"]


def minute_means(series, center, span_minutes):
    """
    Return per-minute average rms from span_minutes before to span_minutes
    after the center time, as a Series indexed by minute.
    """
    # Window start, on a whole minute
    lo = center.replace(second=0, microsecond=0) - timedelta(minutes=span_minutes)
    # Window end
    hi = center.replace(second=0, microsecond=0) + timedelta(minutes=span_minutes + 1)
    # Slice the window
    window = series.loc[lo:hi]
    # Average per minute
    return window.resample("1min").mean()


def parse_args():
    """Read the command-line options and return them as a namespace."""
    # Parser with a one-line description
    parser = argparse.ArgumentParser(description="Print egauge_7 current around the logged cb_146 switching times.")
    # Data folder
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="merged download folder")
    # Meter
    parser.add_argument("--meter", default=DEFAULT_METER, help="meter on cb_146")
    # Channels
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS, help="current channels")
    # Events
    parser.add_argument("--events", nargs="+", default=DEFAULT_EVENTS, help="logged times as 'YYYY-MM-DD HH:MM:SS'")
    # Span
    parser.add_argument("--span-minutes", type=int, default=DEFAULT_SPAN_MINUTES, help="minutes to print on each side")
    # Shift
    parser.add_argument("--shift-hours", type=int, default=DEFAULT_SHIFT_HOURS, help="alternative offset in hours")
    # Parse and return
    return parser.parse_args()


def main():
    """
    For each event and each candidate time, print a per-minute table of
    average current on every channel.
    """
    # Read options
    args = parse_args()
    # Loop over events
    for event_text in args.events:
        # Parse the logged time
        logged = datetime.strptime(event_text, "%Y-%m-%d %H:%M:%S")
        # The alternative time
        shifted = logged + timedelta(hours=args.shift_hours)
        # Load each channel once for this day
        series_by_channel = {}
        # Loop over channels
        for channel in args.channels:
            # Files covering the day
            paths = find_channel_files(args.data_dir, args.meter, logged.date(), channel)
            # Load
            series_by_channel[channel] = load_rms(paths)
        # Two candidate times to print
        for label, center in [("logged", logged), (f"shift {args.shift_hours:+d}h", shifted)]:
            # Section header
            print(f"\n=== {args.meter}  event logged {event_text}  showing {label} window around {center.strftime('%H:%M')} ===")
            # Column header
            print("minute   " + "  ".join(f"{c:>8s}" for c in args.channels))
            # Per-minute means per channel
            tables = {c: minute_means(s, center, args.span_minutes) for c, s in series_by_channel.items()}
            # All minutes across channels
            minutes = sorted(set().union(*[set(t.index) for t in tables.values()]))
            # Print each minute
            for m in minutes:
                # Mark the candidate minute
                mark = "<--" if m.strftime("%H:%M") == center.strftime("%H:%M") else "   "
                # Values per channel, blank when missing
                vals = []
                # Loop over channels
                for c in args.channels:
                    # Value or NaN
                    v = tables[c].get(m, float("nan"))
                    # Format
                    vals.append(f"{v:8.1f}" if pd.notna(v) else f"{'':>8s}")
                # Row
                print(f"{m.strftime('%H:%M')}  " + "  ".join(vals) + f"  {mark}")


# Run main() only when executed as a script
if __name__ == "__main__":
    main()