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
#   main()              loop over months, build, save, verify, log, manifest
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: CSV writing for the log
import csv
# Standard library: JSON for printing stats
import json
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


def main():
    """Loop over months: build, save, verify, log. Then write the manifest."""
    # Options
    args = parse_args()
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
    # Report the setup
    print(f"{len(meters)} meters, {args.start_date} to {args.end_date}, harmonics {'kept' if args.keep_harmonics else 'dropped'}, out {args.out_dir}")
    # Loop over months
    for month_start, month_end in spp.month_chunks(start, end):
        # Month label
        month = month_start.strftime("%Y-%m")
        # Target path
        path = os.path.join(args.out_dir, f"{month}.npz")
        # Skip existing unless forced
        if os.path.exists(path) and not args.force:
            # Say so
            print(f"{month}: exists, skipped")
            # Count it
            months_done.append(month)
            # Next
            continue
        # Build
        arrays, stats = sf.build_month(args.processed_dir, month, meters, args.keep_harmonics, args.freq_tolerance)
        # Nothing for this month
        if arrays is None:
            # Say so
            print(f"{month}: no processed data, skipped")
            # Next
            continue
        # Save
        path, size = sf.save_month(args.out_dir, month, arrays)
        # Spot check
        check = sf.verify_month(path, args.processed_dir, month, meters, args.verify_samples, args.seed) if args.verify_samples > 0 else {"checked": 0}
        # Log row
        row = {
            # Month
            "month": month,
            # Slots in the month
            "slots": int(len(arrays["t"])),
            # Meters that had a file
            "meters_with_files": stats["meters_with_files"],
            # File size in megabytes
            "size_mb": round(size / 1e6, 1),
            # Spot-check numbers
            "checked": check.get("checked", 0),
            # Worst magnitude error
            "rms_rel_err": check.get("rms_rel_err", ""),
            # Worst angle error
            "ang_err_deg": check.get("ang_err_deg", ""),
            # Worst frequency error
            "freq_err_hz": check.get("freq_err_hz", ""),
            # Frequency disagreements summed over meters
            "freq_disagreements": sum(stats.get("freq_disagreements", {}).values()),
            # Share of present readings whose angle passed, over all meters
            "angle_ok_fraction": round(float(arrays["angle_ok"].sum() / max(int(arrays["mask"].sum()), 1)), 4),
            # Sampled cells whose angle was rejected
            "angle_rejected_in_check": check.get("angle_rejected", 0),
        }
        # Append
        append_log(log_path, row)
        # Count it
        months_done.append(month)
        # Print the row
        print(f"{month}: " + json.dumps(row))
    # Manifest covering every month present
    manifest_path = sf.write_manifest(args.out_dir, meters, args.keep_harmonics, months_done)
    # Done
    print(f"\nWrote {len(months_done)} months. Manifest: {manifest_path}. Log: {log_path}")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
