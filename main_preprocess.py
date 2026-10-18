# =============================================================================
# File:        main_preprocess.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-18
#
# Purpose:     Main preprocessing script. Reads options from the command
#              line, with project defaults, and calls socal_preprocess.py
#              to turn the raw download into aligned, masked Parquet tables
#              plus meters.json, events.csv and coverage.csv. Rerunning
#              skips months already written unless --force is given.
#
# Usage:       python main_preprocess.py
#              python main_preprocess.py --start-date 2024-09-01 --end-date 2024-09-30
#              python main_preprocess.py --meters egauge_7 egauge_9
#              python main_preprocess.py --force
#              python main_preprocess.py --help
#
# Setup (once):
#              pip install pandas numpy pyarrow
#              cp -R <caltech repo>/sample_dataset/topology ./topology
#
# Inputs:      data/            the merged download (or a link to it)
#              topology/        Caltech's topology folder, copied into the repo
#              meters.txt       meters to process, one per line
# Outputs:     data/processed/<YYYY-MM>/<meter>.parquet
#              data/processed/meters.json
#              data/processed/events.csv
#              data/processed/coverage.csv
# =============================================================================

# Standard library: command-line argument parsing
import argparse
# Standard library: JSON for metadata
import json
# Standard library: file and folder operations
import os
# Our helper functions
import socal_preprocess as sp

# -----------------------------------------------------------------------------
# Project defaults. Every one can be overridden on the command line.
# -----------------------------------------------------------------------------

# Merged download folder (or a link to it)
DEFAULT_DATA_DIR = "data"
# Where processed files go
DEFAULT_OUT_DIR = os.path.join("data", "processed")
# Caltech topology folder copied into the repo
DEFAULT_TOPOLOGY_DIR = "topology"
# Meter list file
DEFAULT_METERS_FILE = "meters.txt"
# Meters to leave out: not in the network file
DEFAULT_EXCLUDE_METERS = ["egauge_1", "egauge_2"]
# Channels: three voltages, three currents
DEFAULT_CHANNELS = ["L1", "L2", "L3", "S1", "S2", "S3"]
# First day to process
DEFAULT_START_DATE = "2023-08-01"
# Last day to process
DEFAULT_END_DATE = "2025-09-03"
# Hours to add to standard time to get UTC
DEFAULT_UTC_OFFSET_HOURS = 8
# Changes closer than this share an episode
DEFAULT_EPISODE_GAP_MINUTES = 60
# Event days not scored
DEFAULT_EXCLUDE_DATES = ["2024-02-01"]


def parse_args():
    """Read the command-line options and return them as a namespace."""
    # Parser
    parser = argparse.ArgumentParser(description="Preprocess the SoCal 28-Bus download into aligned Parquet tables.")
    # Data folder
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="merged download folder")
    # Output folder
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="folder for processed files")
    # Topology folder
    parser.add_argument("--topology-dir", default=DEFAULT_TOPOLOGY_DIR, help="Caltech topology folder")
    # Meter list file
    parser.add_argument("--meters-file", default=DEFAULT_METERS_FILE, help="text file, one meter per line")
    # Explicit meter list overrides the file
    parser.add_argument("--meters", nargs="+", default=None, help="process only these meters")
    # Exclusions
    parser.add_argument("--exclude-meters", nargs="+", default=DEFAULT_EXCLUDE_METERS, help="meters to leave out")
    # Channels
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS, help="channel suffixes")
    # Dates
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="first day YYYY-MM-DD")
    # End date
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="last day YYYY-MM-DD")
    # UTC offset
    parser.add_argument("--utc-offset-hours", type=int, default=DEFAULT_UTC_OFFSET_HOURS, help="hours added to standard time for UTC")
    # Episode gap
    parser.add_argument("--episode-gap-minutes", type=int, default=DEFAULT_EPISODE_GAP_MINUTES, help="changes closer than this share an episode")
    # Excluded dates
    parser.add_argument("--exclude-dates", nargs="+", default=DEFAULT_EXCLUDE_DATES, help="event days not scored, YYYY-MM-DD")
    # Force rebuild
    parser.add_argument("--force", action="store_true", help="rebuild months that already exist")
    # Parse
    return parser.parse_args()


def main():
    """Build metadata, events, and every meter-month table."""
    # Options
    args = parse_args()
    # Make the output folder
    os.makedirs(args.out_dir, exist_ok=True)
    # Meter list: explicit or from file
    meters = args.meters if args.meters else sp.read_meters(args.meters_file)
    # Drop excluded meters
    meters = [m for m in meters if m not in set(args.exclude_meters)]
    # Dates
    start_date = sp.parse_date(args.start_date)
    # End
    end_date = sp.parse_date(args.end_date)

    # Network file and meter metadata
    network = sp.load_network(args.topology_dir)
    # Metadata dictionary
    meta = sp.meter_metadata(network)
    # Warn about meters not in the network file
    for m in meters:
        # Check
        if m not in meta:
            # Report
            print(f"warning: {m} is not in the network file; it will be processed but cannot be placed on the graph")
    # Write meters.json
    with open(os.path.join(args.out_dir, "meters.json"), "w") as f:
        # Readable JSON
        json.dump(meta, f, indent=2)
    # Report
    print(f"wrote meters.json ({len(meta)} meters in network file)")

    # Events
    events = sp.build_events(args.topology_dir, args.utc_offset_hours, args.episode_gap_minutes, args.exclude_dates)
    # Write events.csv
    events.to_csv(os.path.join(args.out_dir, "events.csv"), index=False)
    # Report
    print(f"wrote events.csv ({len(events)} changes, {events['episode'].nunique()} episodes, {int(events['scored'].sum())} scored changes)")

    # Fresh coverage file on a forced rebuild
    if args.force and os.path.exists(os.path.join(args.out_dir, "coverage.csv")):
        # Remove the old one
        os.remove(os.path.join(args.out_dir, "coverage.csv"))

    # Loop over months
    for month_start, month_end in sp.month_chunks(start_date, end_date):
        # Label such as 2024-09
        month_label = month_start.strftime("%Y-%m")
        # Month output folder
        month_dir = os.path.join(args.out_dir, month_label)
        # Make it
        os.makedirs(month_dir, exist_ok=True)
        # Loop over meters
        for meter in meters:
            # Output path
            out_path = os.path.join(month_dir, f"{meter}.parquet")
            # Skip existing unless forced
            if os.path.exists(out_path) and not args.force:
                # Say so
                print(f"skip {month_label} {meter} (exists)")
                # Next
                continue
            # Build the table
            table = sp.build_meter_month(args.data_dir, meter, args.channels, month_start, month_end, sp.VALUE_COLUMNS, args.utc_offset_hours)
            # Write Parquet
            table.to_parquet(out_path)
            # Coverage rows
            rows = sp.coverage_rows(table, meter, month_label)
            # Append to coverage.csv
            sp.append_coverage(args.out_dir, rows)
            # Report: mean fraction across channels
            mean_fraction = sum(r["fraction"] for r in rows) / len(rows)
            # Line
            print(f"{month_label} {meter}: {len(table)} slots, coverage {mean_fraction:.1%}")
    # Done
    print(f"Finished. Output in {args.out_dir}")


# Run main() only when executed as a script
if __name__ == "__main__":
    main()