# =============================================================================
# File:        main_logtime.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-18
#
# Purpose:     Log-time check. The meters run on fixed standard time all
#              year (see main_dst.py). The operator log may use local clock
#              time, which in summer is one hour ahead. For each September
#              2023 logged event, this script measures how sharply the
#              readings jump at the logged minute and at the same minute
#              one hour earlier, and reports which one shows the change.
#              Read only; writes nothing.
#
# How to read the result:
#              "logged" wins: log and meters agree, no shift needed.
#              "minus 1h" wins: the log is in local daylight time, and the
#              loader must subtract one hour from summer log entries.
#
# Usage:       python main_logtime.py
#              python main_logtime.py --meters egauge_7 egauge_9 egauge_12
#              python main_logtime.py --channels S1 S2 S3 L1
#              python main_logtime.py --help
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
# Meters to inspect: every meter that reported in 2023 (3 and 4 were not installed)
DEFAULT_METERS = [f"egauge_{n}" for n in range(5, 23)]
# Channels to inspect: currents react most to switching, one voltage for reference
DEFAULT_CHANNELS = ["S1", "S2", "S3", "L1"]
# Logged event times from metadata/parameter_changes.csv, September 2023
DEFAULT_EVENTS = ["2023-09-17 11:02:10", "2023-09-17 14:04:17", "2023-09-26 14:49:50", "2023-09-26 15:12:10"]
# Minutes before and after a candidate time used to measure the jump
DEFAULT_WINDOW_MINUTES = 5
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
    Read columns t and rms from every file and return one DataFrame
    indexed by timestamp, sorted, with duplicates removed.
    """
    # Nothing to read
    if not paths:
        # Empty frame with the right shape
        return pd.DataFrame(columns=["rms"], index=pd.DatetimeIndex([]))
    # Read only the two columns from each file
    frames = [pd.read_csv(p, usecols=["t", "rms"]) for p in paths]
    # Stack them
    df = pd.concat(frames, ignore_index=True)
    # Parse timestamps, bad rows become NaT
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    # Drop bad rows and duplicate timestamps
    df = df.dropna(subset=["t"]).drop_duplicates(subset=["t"]).sort_values("t")
    # Index by time
    return df.set_index("t")[["rms"]]


def jump_score(df, when, window_minutes):
    """
    Measure how sharply rms changes at a given time.

    Takes the mean rms in the window before and after the time, and
    divides the absolute difference by the spread of the before window.
    A large number means a clear step at that moment. Returns None if
    either window is empty.
    """
    # Window length
    w = timedelta(minutes=window_minutes)
    # Readings before the time
    before = df.loc[when - w: when - timedelta(seconds=1), "rms"]
    # Readings after the time
    after = df.loc[when: when + w, "rms"]
    # Need data on both sides
    if len(before) < 3 or len(after) < 3:
        # Cannot judge
        return None
    # Spread of the before window, with a floor so we never divide by zero
    spread = max(before.std(), 1e-6)
    # Normalized step size
    return abs(after.mean() - before.mean()) / spread


def parse_args():
    """Read the command-line options and return them as a namespace."""
    # Parser with a one-line description
    parser = argparse.ArgumentParser(description="Check whether logged event times match meter time or local daylight time.")
    # Data folder
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="merged download folder")
    # Meters
    parser.add_argument("--meters", nargs="+", default=DEFAULT_METERS, help="meters to inspect")
    # Channels
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS, help="channels to inspect")
    # Events
    parser.add_argument("--events", nargs="+", default=DEFAULT_EVENTS, help="logged times as 'YYYY-MM-DD HH:MM:SS'")
    # Window
    parser.add_argument("--window-minutes", type=int, default=DEFAULT_WINDOW_MINUTES, help="minutes before and after to compare")
    # Shift
    parser.add_argument("--shift-hours", type=int, default=DEFAULT_SHIFT_HOURS, help="alternative offset in hours to test")
    # Parse and return
    return parser.parse_args()


def main():
    """
    For every event, meter, and channel, score the jump at the logged
    time and at the shifted time, print a table, and give a conclusion.
    """
    # Read options
    args = parse_args()
    # Header line
    print("event                meter      chan   logged    minus1h   winner")
    # Tally of winners
    tally = {"logged": 0, "minus 1h": 0, "unclear": 0}
    # Loop over events
    for event_text in args.events:
        # Parse the logged time
        logged = datetime.strptime(event_text, "%Y-%m-%d %H:%M:%S")
        # The alternative time
        shifted = logged + timedelta(hours=args.shift_hours)
        # Loop over meters
        for meter in args.meters:
            # Loop over channels
            for channel in args.channels:
                # Files covering the event day
                paths = find_channel_files(args.data_dir, meter, logged.date(), channel)
                # Skip meters with no file
                if not paths:
                    # Next channel
                    continue
                # Load rms with time index
                df = load_rms(paths)
                # Score at the logged time
                s_logged = jump_score(df, logged, args.window_minutes)
                # Score at the shifted time
                s_shift = jump_score(df, shifted, args.window_minutes)
                # Skip if either side had no data
                if s_logged is None or s_shift is None:
                    # Next channel
                    continue
                # Decide the winner: one must be clearly larger and clearly a step
                if s_logged > 2 * s_shift and s_logged > 3:
                    # Logged time shows the step
                    winner = "logged"
                elif s_shift > 2 * s_logged and s_shift > 3:
                    # Shifted time shows the step
                    winner = "minus 1h"
                else:
                    # No clear step at either
                    winner = "unclear"
                # Count it
                tally[winner] += 1
                # Print the row
                print(f"{event_text}  {meter:10s} {channel:5s}  {s_logged:7.2f}  {s_shift:7.2f}   {winner}")
    # Summary line
    print(f"\nWinners: logged={tally['logged']}  minus 1h={tally['minus 1h']}  unclear={tally['unclear']}")
    # Plain-language conclusion
    if tally["logged"] > tally["minus 1h"]:
        # Agreement
        print("Conclusion: log times match meter time. No shift needed for summer events.")
    elif tally["minus 1h"] > tally["logged"]:
        # Shift needed
        print("Conclusion: log is in local daylight time. Subtract one hour from summer log entries.")
    else:
        # Cannot tell
        print("Conclusion: unclear. Try a wider window or other channels.")


# Run main() only when executed as a script
if __name__ == "__main__":
    main()