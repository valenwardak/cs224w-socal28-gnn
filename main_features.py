# =============================================================================
# File:        main_features.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-21
#
# Purpose:     Export the processed Parquet tables as compact monthly
#              feature files for model training on Colab, one .npz per
#              month plus a manifest. All the work is in socal_features.py;
#              this file only parses options, loops over months, writes,
#              and reports sizes and spot-check errors.
#
# Usage:
#   Everything, fundamental only, float16 magnitudes:
#     python main_features.py
#   One month, to measure size before exporting all:
#     python main_features.py --start-date 2024-09-01 --end-date 2024-09-30
#   With harmonics 1 and 2:
#     python main_features.py --keep-harmonics
#   Sequential (in-process) months:
#     python main_features.py --workers 1
#   Options: python main_features.py --help
#
# Output under <out_dir> (default data/features):
#   <YYYY-MM>.npz    arrays for the month, see socal_features.py header
#   manifest.json    meter order, channel order, arrays, units, alignment
#   build_log.csv    one row per month: slots, meters with files, file size,
#                    spot-check errors, frequency disagreements, share of
#                    readings whose angle passed the checks
#
# Functions:
#   parse_args()        command-line options with defaults
#   append_log(path, row)
#                       append one row to build_log.csv, header on first write
#   run(args)           loop over months (optionally in parallel), log, manifest
#   main()              parse options and run
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: CSV writing for the log
import csv
# Standard library: process pool for independent months
from concurrent.futures import ProcessPoolExecutor
# Standard library: JSON for printing stats
import json
# Standard library: spawn context, safe with numpy/pandas
from multiprocessing import get_context
# Standard library: file and folder operations
import os
# Project: date helpers from preprocessing
import socal_preprocess as spp
# Project: feature helpers
import socal_features as sf


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser with the file's purpose as description
    parser = argparse.ArgumentParser(description="Export processed Parquet tables as compact monthly feature files")
    # Processed data folder
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"), help="Folder with <YYYY-MM>/<meter>.parquet")
    # Output folder
    parser.add_argument("--out-dir", default=os.path.join("data", "features"), help="Folder for <YYYY-MM>.npz and manifest.json")
    # Meter list
    parser.add_argument("--meters-file", default="meters.txt", help="Meter names, one per line, fixes the meter order")
    # Meters to leave out
    parser.add_argument("--exclude-meters", nargs="*", default=["egauge_1", "egauge_2"], help="Meters to skip, not in the network file")
    # First day
    parser.add_argument("--start-date", default="2023-08-01", help="First day, YYYY-MM-DD, inclusive")
    # Last day
    parser.add_argument("--end-date", default="2025-09-03", help="Last day, YYYY-MM-DD, inclusive")
    # Harmonics
    parser.add_argument("--keep-harmonics", action="store_true", help="Also export harmonics 1 and 2 (120 Hz and 180 Hz)")
    # Frequency tolerance
    parser.add_argument("--freq-tolerance", type=float, default=0.01, help="Hertz; a channel angle is accepted only when its frequency is within this of the meter's voltage frequency (0.01 means the same FFT bin)")
    # Spot-check size
    parser.add_argument("--verify-samples", type=int, default=200, help="Cells per month to compare against the Parquet, 0 to skip")
    # Random seed for the spot check
    parser.add_argument("--seed", type=int, default=0, help="Seed for the spot-check sample")
    # Rebuild
    parser.add_argument("--force", action="store_true", help="Rebuild months that already have a file")
    # Parallel months; 1 (or less) stays in-process
    parser.add_argument("--workers", type=int, default=4, help="Independent months to process at once, 1 = in-process (default 4)")
    # Parse
    return parser.parse_args()


def append_log(path, row):
    """Append one row to build_log.csv, writing the header if the file is new."""
    # Header needed only when new
    write_header = not os.path.exists(path)
    # Append
    with open(path, "a", newline="") as f:
        # Writer with the row's keys as columns
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        # Header on first write
        if write_header:
            # Write it
            writer.writeheader()
        # Row
        writer.writerow(row)


def _handle_result(result, log_path, months_done):
    """Apply one month's export result: skip, log, or count. Prints a line."""
    # Month label
    month = result["month"]
    # Already on disk
    if result["status"] == "exists":
        # Same message as the sequential skip
        print(f"{month}: exists, skipped")
        # Count it
        months_done.append(month)
        # Done
        return
    # No Parquet
    if result["status"] == "no_data":
        # Same message as the sequential skip
        print(f"{month}: no processed data, skipped")
        # Not counted in the manifest
        return
    # Written
    row = result["row"]
    # Append in caller-determined order
    append_log(log_path, row)
    # Count it
    months_done.append(month)
    # Print the row, including stage timings
    print(f"{month}: " + json.dumps(row))


def run(args):
    """Loop over months: build, save, verify, log. Then write the manifest."""
    # Meters in fixed order
    meters = sf.meter_order(args.meters_file, args.exclude_meters)
    # Dates
    start, end = spp.parse_date(args.start_date), spp.parse_date(args.end_date)
    # Output folder
    os.makedirs(args.out_dir, exist_ok=True)
    # Log path
    log_path = os.path.join(args.out_dir, "build_log.csv")
    # Months written or already present
    months_done = []
    # Workers: 1 or less stays in-process
    workers = 1 if args.workers < 2 else args.workers
    # Report the setup
    print(f"{len(meters)} meters, {args.start_date} to {args.end_date}, harmonics {'kept' if args.keep_harmonics else 'dropped'}, workers {workers}, out {args.out_dir}")
    # Months that still need work
    jobs = []
    # Loop over months for skip vs work
    for month_start, month_end in spp.month_chunks(start, end):
        # Month label
        month = month_start.strftime("%Y-%m")
        # Target path
        path = os.path.join(args.out_dir, f"{month}.npz")
        # Skip a readable complete file unless forced; rebuild truncated copies
        if not args.force and sf.npz_is_complete(path, args.keep_harmonics):
            # Say so
            print(f"{month}: exists, skipped")
            # Count it
            months_done.append(month)
            # Next
            continue
        # Queue
        jobs.append(month)
    # Shared kwargs for every job
    job_kwargs = dict(
        processed_dir=args.processed_dir,
        out_dir=args.out_dir,
        meters=meters,
        keep_harmonics=args.keep_harmonics,
        tolerance=args.freq_tolerance,
        verify_samples=args.verify_samples,
        seed=args.seed,
        force=args.force,
    )
    # In-process when one worker or one remaining month
    use_pool = workers > 1 and len(jobs) > 1
    # Months that raised; other months still finish
    errors = []
    if not use_pool:
        # Sequential, same order as before
        for month in jobs:
            try:
                # One month
                result = sf.export_month(month=month, **job_kwargs)
            except Exception as exc:
                # Keep going so later months are not abandoned
                print(f"{month}: failed: {exc}")
                errors.append((month, exc))
                # Next
                continue
            # Log and print
            _handle_result(result, log_path, months_done)
    # Parallel months, results consumed in calendar order so the log is deterministic
    else:
        # Bound concurrency to the number of remaining months
        pool_workers = min(workers, len(jobs))
        # Process pool; each worker holds one month
        with ProcessPoolExecutor(max_workers=pool_workers, mp_context=get_context("spawn")) as pool:
            # Submit all, then wait in month order
            futures = {month: pool.submit(sf.export_month, month=month, **job_kwargs) for month in jobs}
            # Calendar order
            for month in jobs:
                try:
                    # Block until this month is done
                    result = futures[month].result()
                except Exception as exc:
                    # Other workers keep running; this month can be retried
                    print(f"{month}: failed: {exc}")
                    errors.append((month, exc))
                    # Next
                    continue
                # Log and print
                _handle_result(result, log_path, months_done)
    # Manifest covering every month present
    manifest_path = sf.write_manifest(args.out_dir, meters, args.keep_harmonics, months_done)
    # Fail after writing whatever succeeded, so a retry can resume
    if errors:
        # Names only
        failed = ", ".join(month for month, _ in errors)
        # Chain the first exception
        raise RuntimeError(f"{len(errors)} month(s) failed: {failed}") from errors[0][1]
    # Done
    print(f"\nWrote {len(months_done)} months. Manifest: {manifest_path}. Log: {log_path}")
    # Return paths for tests
    return months_done, manifest_path, log_path


def main():
    """Parse options and run the export."""
    # Options
    run(parse_args())


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
