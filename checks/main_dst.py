# =============================================================================
# File:        main_dst.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-18
#
# Purpose:     Daylight saving check. Counts phasor rows per clock hour in
#              the early morning of each clock-change night, and of a
#              reference night one week earlier, for a few meters. Prints a
#              table and a plain-language reading of whether timestamps
#              follow local clock time or fixed standard time. Read only;
#              writes nothing.
#
# How to read the result:
#              Local clock time: the fall-back night shows about double the
#              rows in the 1 a.m. hour, and the spring-forward nights show
#              zero rows in the 2 a.m. hour.
#              Fixed standard time: all hours look normal on every night.
#
# Usage:       python main_dst.py
#              python main_dst.py --meters egauge_7 egauge_12 egauge_22
#              python main_dst.py --data-dir data --channel L1
#              python main_dst.py --help
#
# Nights:      2023-11-05 fall back, 2024-03-10 spring forward,
#              2025-03-09 spring forward. 2024-11-03 falls in a site-wide
#              outage and is not testable.
#
# Data layout read:
#              <data_dir>/<meter>/<start>_<end>[_<channel>]/phasors/<meter>-<channel>.csv
#              Only column t is read.
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
# Meters to inspect; chosen for good coverage in both years
DEFAULT_METERS = ["egauge_7", "egauge_9", "egauge_12", "egauge_14", "egauge_22"]
# Voltage channel to count; voltages are the most reliably reported
DEFAULT_CHANNEL = "L1"
# Clock-change nights and their kind, as YYYY-MM-DD:kind
DEFAULT_NIGHTS = ["2023-11-05:fall", "2024-03-10:spring", "2025-03-09:spring"]
# First clock hour to count
DEFAULT_FIRST_HOUR = 0
# Last clock hour to count, inclusive
DEFAULT_LAST_HOUR = 5
# Reference night this many days earlier
DEFAULT_REFERENCE_DAYS = 7


def parse_folder_label(label):
    """
    Split a request folder name into (start_date, end_date, channel).

    Examples: "2024-09-01_2024-09-30"      -> (2024-09-01, 2024-09-30, None)
              "2025-07-01_2025-07-31_L2"   -> (2025-07-01, 2025-07-31, "L2")
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
    folder covers the given day.

    Fallback folders are included, so the same timestamps may appear in
    more than one file. The caller removes duplicates.
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

    Rows that fail to parse are dropped. Duplicates from overlapping
    fallback folders are dropped.
    """
    # Nothing to read
    if not paths:
        # Empty series
        return pd.Series([], dtype="datetime64[ns]")
    # Read only the t column from each file
    frames = [pd.read_csv(p, usecols=["t"]) for p in paths]
    # Stack them
    t = pd.concat(frames, ignore_index=True)["t"]
    # Parse as datetimes, leaving bad rows as NaT
    t = pd.to_datetime(t, errors="coerce")
    # Drop unparseable rows and duplicates, then sort
    t = t.dropna().drop_duplicates().sort_values()
    # Return the clean series with a fresh index
    return t.reset_index(drop=True)


def hourly_counts(timestamps, day, first_hour, last_hour):
    """
    Count timestamps in each clock hour of the given day from first_hour
    up to and including last_hour.

    Returns a list of counts, one per hour.
    """
    # Start of the day as a datetime
    day_start = datetime.combine(day, datetime.min.time())
    # One count per hour
    counts = []
    # Loop over the hours
    for hour in range(first_hour, last_hour + 1):
        # Hour window start
        lo = day_start + timedelta(hours=hour)
        # Hour window end
        hi = lo + timedelta(hours=1)
        # Count rows inside the window
        counts.append(int(((timestamps >= lo) & (timestamps < hi)).sum()))
    # Return the list
    return counts


def check_night(data_dir, meter, day, channel, first_hour, last_hour, reference_days):
    """
    Return hourly counts for one meter on the given night and on a
    reference night reference_days earlier.

    Returns (counts_night, counts_reference), each a list of counts.
    """
    # Files covering the night
    night_files = find_channel_files(data_dir, meter, day, channel)
    # Timestamps for the night
    night_t = load_timestamps(night_files)
    # Counts for the night
    night_counts = hourly_counts(night_t, day, first_hour, last_hour)
    # Reference day
    ref_day = day - timedelta(days=reference_days)
    # Files covering the reference day
    ref_files = find_channel_files(data_dir, meter, ref_day, channel)
    # Timestamps for the reference day
    ref_t = load_timestamps(ref_files)
    # Counts for the reference day
    ref_counts = hourly_counts(ref_t, ref_day, first_hour, last_hour)
    # Return both
    return night_counts, ref_counts


def verdict(night_counts, ref_counts, first_hour, kind):
    """
    Give a one-word reading of one night.

    kind is "fall" (local time gives a doubled 1 a.m. hour) or "spring"
    (local time gives an empty 2 a.m. hour).
    Returns "local", "standard", or "unclear".
    """
    # Index of the hour to inspect: 1 a.m. for fall, 2 a.m. for spring
    idx = (1 if kind == "fall" else 2) - first_hour
    # Out of range means we cannot judge
    if idx < 0 or idx >= len(night_counts):
        # Cannot judge
        return "unclear"
    # The hour's count on the night
    n = night_counts[idx]
    # The hour's count on the reference night
    r = ref_counts[idx]
    # No reference data means we cannot judge
    if r == 0:
        # Cannot judge
        return "unclear"
    # Fall back: local time gives roughly double
    if kind == "fall":
        # Clearly doubled
        if n > 1.6 * r:
            # Local clock time
            return "local"
        # Clearly normal
        if 0.6 * r < n < 1.4 * r:
            # Standard time
            return "standard"
        # In between
        return "unclear"
    # Spring forward: local time gives roughly zero
    if n < 0.1 * r:
        # Local clock time
        return "local"
    # Clearly normal
    if n > 0.6 * r:
        # Standard time
        return "standard"
    # In between
    return "unclear"


def parse_args():
    """
    Read the command-line options and return them as a namespace.

    Every option falls back to the project default above.
    """
    # Parser with a one-line description
    parser = argparse.ArgumentParser(description="Daylight saving check on SoCal 28-Bus phasor timestamps.")
    # Data folder
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="merged download folder")
    # Meters
    parser.add_argument("--meters", nargs="+", default=DEFAULT_METERS, help="meters to inspect")
    # Channel
    parser.add_argument("--channel", default=DEFAULT_CHANNEL, help="channel to count, for example L1")
    # Nights as DATE:kind
    parser.add_argument("--nights", nargs="+", default=DEFAULT_NIGHTS, help="nights as YYYY-MM-DD:fall or YYYY-MM-DD:spring")
    # First hour
    parser.add_argument("--first-hour", type=int, default=DEFAULT_FIRST_HOUR, help="first clock hour to count")
    # Last hour
    parser.add_argument("--last-hour", type=int, default=DEFAULT_LAST_HOUR, help="last clock hour to count, inclusive")
    # Reference offset
    parser.add_argument("--reference-days", type=int, default=DEFAULT_REFERENCE_DAYS, help="reference night this many days earlier")
    # Parse and return
    return parser.parse_args()


def main():
    """
    Run the check for every night and meter, print the table, and print
    a plain-language conclusion.
    """
    # Read options
    args = parse_args()
    # Hour labels for the table header
    hours = list(range(args.first_hour, args.last_hour + 1))
    # Header line
    print("meter      night       kind    which      " + " ".join(f"{h:02d}h" for h in hours) + "   reading")
    # Tally of readings across all meters and nights
    tally = {"local": 0, "standard": 0, "unclear": 0}
    # Loop over nights
    for item in args.nights:
        # Split DATE:kind
        day_text, kind = item.split(":")
        # Parse the date
        day = datetime.strptime(day_text, "%Y-%m-%d").date()
        # Loop over meters
        for meter in args.meters:
            # Counts for the night and the reference night
            night_counts, ref_counts = check_night(args.data_dir, meter, day, args.channel, args.first_hour, args.last_hour, args.reference_days)
            # Reading for this meter and night
            reading = verdict(night_counts, ref_counts, args.first_hour, kind)
            # Count it
            tally[reading] += 1
            # Print the night row
            print(f"{meter:10s} {day_text}  {kind:6s}  night      " + " ".join(f"{c:3d}" for c in night_counts) + f"   {reading}")
            # Print the reference row
            print(f"{'':10s} {'':10s}  {'':6s}  reference  " + " ".join(f"{c:3d}" for c in ref_counts))
    # Summary line
    print(f"\nReadings: local={tally['local']}  standard={tally['standard']}  unclear={tally['unclear']}")
    # Plain-language conclusion: local wins
    if tally["local"] > tally["standard"]:
        # Local clock time
        print("Conclusion: timestamps follow local clock time (America/Los_Angeles with daylight saving).")
    # Standard wins
    elif tally["standard"] > tally["local"]:
        # Fixed standard time
        print("Conclusion: timestamps follow fixed standard time (no daylight saving shift).")
    # Tie or nothing readable
    else:
        # Cannot tell
        print("Conclusion: unclear. Try other meters or nights.")


# Run main() only when executed as a script
if __name__ == "__main__":
    main()