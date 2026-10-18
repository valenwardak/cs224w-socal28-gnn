# =============================================================================
# File:        main_clocks.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-18
#
# Purpose:     Clock consistency check across meters. At a few known
#              moments the readings stop (a site-wide outage, or a meter
#              losing power when its breaker opened). For every meter this
#              script finds the gap in its readings nearest to each
#              reference moment and prints the gap start, the gap end, and
#              the offset in minutes between the gap start and the logged
#              time. Meters whose clocks agree show the same offset. A
#              meter one hour off shows an offset near plus or minus 60.
#              Read only; writes nothing.
#
# Reference moments (from metadata/parameter_changes.csv):
#              2024-11-13 17:52:20  site-wide: all meters stop
#              2023-11-26 09:27:10  egauge_9, 11, 13 lose power
#              2023-09-17 14:04:17  cb_146 opens; egauge_7 stops
#
# Usage:       python main_clocks.py
#              python main_clocks.py --meters egauge_15 egauge_16 egauge_19
#              python main_clocks.py --help
#
# Data layout read:
#              <data_dir>/<meter>/<start>_<end>[_<channel>]/phasors/<meter>-<channel>.csv
#              Only column t is read. Empty files are skipped.
# =============================================================================

# Standard library: command-line argument parsing
import argparse
# Standard library: file and folder operations
import os
# Standard library: dates and times
from datetime import datetime, timedelta
# Third party: CSV reading and timestamp parsing
import pandas as pd

# -----------------------------------------------------------------------------
# Project defaults. Every one can be overridden on the command line.
# -----------------------------------------------------------------------------

# Folder holding the merged download
DEFAULT_DATA_DIR = "data"
# All meters on the circuit
DEFAULT_METERS = [f"egauge_{n}" for n in range(3, 25)]
# Voltage channel to read; voltages are the most reliably reported
DEFAULT_CHANNEL = "L1"
# Reference moments as 'YYYY-MM-DD HH:MM:SS'
DEFAULT_REFERENCES = ["2024-11-13 17:52:20", "2023-11-26 09:27:10", "2023-09-17 14:04:17"]
# Search for gaps starting within this many minutes of the reference
DEFAULT_SEARCH_MINUTES = 90
# A pause between readings longer than this counts as a gap
DEFAULT_GAP_MINUTES = 5


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


def load_timestamps(paths):
    """
    Read column t from every file and return one sorted Series of unique
    timestamps, parsed as naive datetimes exactly as written.

    Empty files (a meter with no readings in that request) are skipped.
    """
    # Collect one frame per readable file
    frames = []
    # Loop over files
    for p in paths:
        # Skip files with no content
        if os.path.getsize(p) == 0:
            # Next file
            continue
        # Try to read the t column
        try:
            # Read
            frames.append(pd.read_csv(p, usecols=["t"]))
        # A file with a header but no rows, or no header at all
        except pd.errors.EmptyDataError:
            # Next file
            continue
    # Nothing readable
    if not frames:
        # Empty series
        return pd.Series([], dtype="datetime64[ns]")
    # Stack them
    t = pd.concat(frames, ignore_index=True)["t"]
    # Parse as datetimes, leaving bad rows as NaT
    t = pd.to_datetime(t, errors="coerce")
    # Drop unparseable rows and duplicates, then sort
    t = t.dropna().drop_duplicates().sort_values()
    # Return with a fresh index
    return t.reset_index(drop=True)


def nearest_gap(timestamps, reference, search_minutes, gap_minutes):
    """
    Find the gap in the readings whose start is closest to the reference
    time, among gaps starting within search_minutes of it.

    Returns (gap_start, gap_end) or None if there is no such gap.
    """
    # Need at least two readings
    if len(timestamps) < 2:
        # Nothing to measure
        return None
    # Search window
    lo = reference - timedelta(minutes=search_minutes)
    # Window end
    hi = reference + timedelta(minutes=search_minutes)
    # Time between consecutive readings
    diffs = timestamps.diff().shift(-1)
    # Candidate gaps: long pauses that begin inside the window
    is_gap = (diffs > timedelta(minutes=gap_minutes)) & (timestamps >= lo) & (timestamps <= hi)
    # Indices of candidates
    idx = timestamps.index[is_gap]
    # None found
    if len(idx) == 0:
        # No gap
        return None
    # Pick the one whose start is closest to the reference
    best = min(idx, key=lambda i: abs((timestamps[i] - reference).total_seconds()))
    # Gap start is the last reading before silence, gap end is the next reading
    return timestamps[best], timestamps[best + 1]


def parse_args():
    """Read the command-line options and return them as a namespace."""
    # Parser with a one-line description
    parser = argparse.ArgumentParser(description="Compare meter clocks using known moments when readings stop.")
    # Data folder
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="merged download folder")
    # Meters
    parser.add_argument("--meters", nargs="+", default=DEFAULT_METERS, help="meters to inspect")
    # Channel
    parser.add_argument("--channel", default=DEFAULT_CHANNEL, help="channel to read, for example L1")
    # References
    parser.add_argument("--references", nargs="+", default=DEFAULT_REFERENCES, help="reference moments as 'YYYY-MM-DD HH:MM:SS'")
    # Search window
    parser.add_argument("--search-minutes", type=int, default=DEFAULT_SEARCH_MINUTES, help="look for gaps within this many minutes")
    # Gap threshold
    parser.add_argument("--gap-minutes", type=int, default=DEFAULT_GAP_MINUTES, help="a pause longer than this is a gap")
    # Parse and return
    return parser.parse_args()


def main():
    """
    For each reference moment and meter, print the nearest gap and the
    offset of its start from the logged time.
    """
    # Read options
    args = parse_args()
    # Loop over reference moments
    for ref_text in args.references:
        # Parse the reference
        reference = datetime.strptime(ref_text, "%Y-%m-%d %H:%M:%S")
        # Section header
        print(f"\n=== reference {ref_text}  (offset = gap start minus logged time, in minutes) ===")
        # Column header
        print("meter       gap start            gap end              offset_min")
        # Loop over meters
        for meter in args.meters:
            # Files covering the day
            paths = find_channel_files(args.data_dir, meter, reference.date(), args.channel)
            # No data for this meter on this day
            if not paths:
                # Report and continue
                print(f"{meter:10s}  no files for this day")
                # Next meter
                continue
            # Timestamps
            t = load_timestamps(paths)
            # No readings at all
            if len(t) == 0:
                # Report and continue
                print(f"{meter:10s}  files are empty")
                # Next meter
                continue
            # Nearest gap
            gap = nearest_gap(t, reference, args.search_minutes, args.gap_minutes)
            # No gap near the reference
            if gap is None:
                # Report and continue
                print(f"{meter:10s}  no gap within {args.search_minutes} min")
                # Next meter
                continue
            # Unpack
            gap_start, gap_end = gap
            # Offset in minutes
            offset = (gap_start - reference).total_seconds() / 60.0
            # Row
            print(f"{meter:10s}  {gap_start.strftime('%Y-%m-%d %H:%M:%S')}  {gap_end.strftime('%Y-%m-%d %H:%M:%S')}  {offset:+8.1f}")


# Run main() only when executed as a script
if __name__ == "__main__":
    main()