# =============================================================================
# File:        main_socal.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-16
#
# Purpose:     Main script. Reads options from the command line, with the
#              project defaults below, and calls the helper functions in
#              socal_download.py. Running with no options downloads the full
#              project window for all meters in meters.txt.
#
# Usage:       python main_socal.py
#              python main_socal.py --start-date 2023-08-01 --end-date 2025-09-03
#              python main_socal.py --data-type magnitudes --channels L1 L2 L3 CT1 CT2 CT3
#              python main_socal.py --meters-file meters.txt --output-dir data
#              python main_socal.py --workers 4
#              python main_socal.py --help
#
# Inputs:      meters.txt   one meter name per line, for example egauge_1
# Outputs:     data/<meter>/<start>_<end>/...   downloaded files
#              data/missing_channels.json       channels that never came back
#              data/download_log.csv            one row per attempt
# =============================================================================

# Standard library: command-line argument parsing
import argparse
# Our helper functions
import socal_download as sd

# -----------------------------------------------------------------------------
# Project defaults. Every one can be overridden on the command line.
# -----------------------------------------------------------------------------

# First day to download, inclusive, as YYYY-MM-DD
DEFAULT_START_DATE = "2023-08-01"
# Last day to download, inclusive, as YYYY-MM-DD
DEFAULT_END_DATE = "2025-09-03"
# Folder for all downloaded data (kept out of git by .gitignore)
DEFAULT_OUTPUT_DIR = "data"
# Text file listing the meters, one per line
DEFAULT_METERS_FILE = "meters.txt"
# Data type to download: phasors, magnitudes, or waveforms
DEFAULT_DATA_TYPE = "phasors"
# Channels per meter: L1-L3 voltages, S1-S3 currents (phasor naming)
DEFAULT_CHANNELS = ["L1", "L2", "L3", "S1", "S2", "S3"]
# Where the Caltech client stores the GitHub login tokens
DEFAULT_CREDENTIALS_FILE = "dataset_api_credentials.json"
# Retries per request before treating it as failed
DEFAULT_RETRIES = 2
# Seconds to wait between retries
DEFAULT_RETRY_WAIT = 30
# Stop splitting a failed range once it is this many days or fewer
DEFAULT_MIN_DAYS = 1
# Meter-month downloads to overlap; threads wait on the network, not extra CPUs
DEFAULT_WORKERS = 4


def parse_args():
    """
    Read the command-line options and return them as a namespace.

    Every option falls back to the project default above.
    """
    # Parser with a one-line description shown by --help
    parser = argparse.ArgumentParser(description="Download SoCal 28-Bus data per meter and month.")
    # First day, inclusive
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="first day YYYY-MM-DD")
    # Last day, inclusive
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="last day YYYY-MM-DD")
    # Root folder for the data
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="folder for downloaded data")
    # File listing meter names
    parser.add_argument("--meters-file", default=DEFAULT_METERS_FILE, help="text file, one meter per line")
    # Which data type to fetch
    parser.add_argument("--data-type", default=DEFAULT_DATA_TYPE, choices=["phasors", "magnitudes", "waveforms"], help="data type to download")
    # Channel suffixes
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS, help="channel suffixes per meter")
    # Location of the credentials file
    parser.add_argument("--credentials-file", default=DEFAULT_CREDENTIALS_FILE, help="Caltech login token file")
    # Retry count
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="retries per request")
    # Retry wait in seconds
    parser.add_argument("--retry-wait", type=int, default=DEFAULT_RETRY_WAIT, help="seconds between retries")
    # Smallest range to split down to
    parser.add_argument("--min-days", type=int, default=DEFAULT_MIN_DAYS, help="stop splitting at this many days")
    # How many meter-month requests to overlap
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel meter-month downloads (threads, not processes); 1 = sequential")
    # Parse and return
    return parser.parse_args()


def main():
    """Run the download with the parsed options."""
    # Read options
    args = parse_args()
    # Load the meter list
    meters = sd.read_meters(args.meters_file)
    # Parse the dates
    start_date = sd.parse_date(args.start_date)
    # Parse the end date
    end_date = sd.parse_date(args.end_date)
    # Create the Caltech client (first run opens the GitHub sign-in)
    client = sd.make_client(args.credentials_file)
    # Download everything
    missing = sd.download_all(client, args.output_dir, meters, args.data_type, args.channels, start_date, end_date, args.retries, args.retry_wait, args.min_days, args.workers)
    # Summary of channels that never came back
    print(f"Finished. Meters with missing channels: {len(missing)}. See {args.output_dir}/{sd.MISSING_FILE}")


# Run main() only when executed as a script
if __name__ == "__main__":
    main()