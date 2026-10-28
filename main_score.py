# =============================================================================
# File:        main_score.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-27
#
# Purpose:     Score one trained run for event detection, and the
#              persistence baseline from the same tables, with one fixed
#              procedure for every model: conformal thresholds per meter
#              and channel from the calibration block, exceedances on the
#              test blocks, alarms by meter count, alarm episodes, one-to-
#              one matching against the logged switching episodes, and
#              false alarms per monitored day. The method is in
#              socal_score.py; this file parses options, runs the modes
#              asked for, prints a summary table, and writes one JSON per
#              run with every setting and result.
#
#              Every setting is an option with a default. The defaults are
#              the fixed scoring settings; they are recorded in the JSON so
#              a result can always be traced to its settings. These are
#              residual-only detections: silence is not scored here.
#
# Usage:
#   python main_score.py --run-dir data/processed/models/gru_full_aw001
#   python main_score.py --run-dir /content/drive/MyDrive/socal_runs/gcn_full_aw001 --events /content/features/events.csv
#   python main_score.py --run-dir ... --modes mag ang both --alpha 1e-4 --min-meters 2
#   Options: python main_score.py --help
#
# Output:
#   <run_dir>/score.json    settings, thresholds, per-block counts, per-episode
#                           results, false-alarm list, and scores, for the
#                           model and persistence in every mode requested
#   printed table           one line per source and mode
#
# Functions:
#   parse_args()        command-line options with defaults, checked for sense
#   clean(obj)          replace NaN and infinities with None so the JSON is standard
#   line(name, r)       one summary line for the table
#   main()              score the model and persistence in each mode, print, write
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON output
import json
# Standard library: paths
import os
# Standard library: finite checks for the JSON cleaning
import math
# Project: the scoring method
import socal_score as ss


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Event-detection scoring for one run, model and persistence")
    # Run folder
    parser.add_argument("--run-dir", required=True, help="Run folder with the residual tables from score_split")
    # Events
    parser.add_argument("--events", default=os.path.join("data", "processed", "events.csv"), help="Event table from preprocessing")
    # Calibration block
    parser.add_argument("--calib-label", default="validation_2025-02-03_2025-02-16", help="Validation block used for thresholds (block 1, never used for model selection)")
    # Test tables
    parser.add_argument("--test-prefixes", nargs="+", default=["test_events_", "test_normal_"], help="Prefixes of the test tables to score")
    # Modes
    parser.add_argument("--modes", nargs="+", choices=["mag", "ang", "both"], default=["mag", "ang", "both"], help="Which residuals fire a channel")
    # Alpha
    parser.add_argument("--alpha", type=float, default=1e-4, help="Per-channel, per-slot conformal level")
    # Minimum meters
    parser.add_argument("--min-meters", type=int, default=2, help="Distinct meters firing for a slot to be an alarm, and with a residual for a slot to be monitored")
    # Merge gap
    parser.add_argument("--merge-slots", type=int, default=6, help="Alarm slots at most this many slots apart join one alarm episode (6 = one minute)")
    # Window before
    parser.add_argument("--before-minutes", type=float, default=1.0, help="Minutes before an episode's first scored change where an alarm counts")
    # Window after
    parser.add_argument("--after-minutes", type=float, default=10.0, help="Minutes after an episode's last scored change where an alarm counts")
    # Bootstrap
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples over episodes")
    # Seed
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed")
    # Common slots
    parser.add_argument("--common", action=argparse.BooleanOptionalAction, default=True, help="Score model and persistence on the slots where both are valid (default); --no-common uses each source's own slots")
    # Output name
    parser.add_argument("--out", default="score.json", help="JSON file name written into the run folder")
    # Parse
    args = parser.parse_args()
    # Alpha must be a proper level
    if not 0.0 < args.alpha < 1.0:
        # Reject
        parser.error(f"--alpha must be between 0 and 1, got {args.alpha}")
    # At least one meter
    if args.min_meters < 1:
        # Reject
        parser.error(f"--min-meters must be at least 1, got {args.min_meters}")
    # At least one resample
    if args.n_boot < 1:
        # Reject
        parser.error(f"--n-boot must be at least 1, got {args.n_boot}")
    # Gaps and windows cannot be negative
    if args.merge_slots < 0 or args.before_minutes < 0 or args.after_minutes < 0:
        # Reject
        parser.error("--merge-slots, --before-minutes, and --after-minutes must not be negative")
    # Return
    return args


def clean(obj):
    """Walk a nested result and turn NaN and infinities into None, so json.dump writes null."""
    # Dictionaries
    if isinstance(obj, dict):
        # Each value
        return {k: clean(v) for k, v in obj.items()}
    # Lists and tuples
    if isinstance(obj, (list, tuple)):
        # Each item
        return [clean(v) for v in obj]
    # Floats that are not finite
    if isinstance(obj, float) and not math.isfinite(obj):
        # Null
        return None
    # Anything else unchanged
    return obj


def line(name, r):
    """One summary line: source, mode, episodes detected, false alarms per day, precision, recall, F1, delay."""
    # Formatter for optional numbers
    f = lambda v, d=3: "  n/a" if v is None else f"{v:.{d}f}"
    # Interval text
    b = r["bootstrap"]
    # Recall interval
    ci = "" if b["recall_lo"] is None else f" [{b['recall_lo']:.2f}, {b['recall_hi']:.2f}]"
    # Line
    return f"{name:<22} {r['episodes_detected']:>3}/{r['episodes_in_scope']:<3} {f(r['false_alarms_per_day'], 2):>8} {f(r['precision']):>7} {f(r['recall']):>7}{ci:<14} {f(r['f1']):>7} {f(r['delay_median_s'], 0):>8}"


def main():
    """Score the model and persistence in each mode, print the table, write the JSON."""
    # Options
    args = parse_args()
    # Settings shared by every call, recorded in the output
    settings = {
        # Thresholds
        "alpha": args.alpha, "calib_label": args.calib_label,
        # Alarms
        "min_meters": args.min_meters, "merge_slots": args.merge_slots,
        # Matching
        "before_minutes": args.before_minutes, "after_minutes": args.after_minutes,
        # Data
        "events": args.events, "test_prefixes": args.test_prefixes, "common": args.common,
        # Bootstrap
        "n_boot": args.n_boot, "seed": args.seed,
    }
    # Results per mode
    out = {"run_dir": os.path.abspath(args.run_dir), "settings": settings, "modes": {}}
    # Header
    print(f"{'source / mode':<22} {'det':>7} {'FA/day':>8} {'prec':>7} {'recall':>7}{'':<14} {'F1':>7} {'delay s':>8}")
    # Loop over modes
    for mode in args.modes:
        # Mode setting
        settings_mode = dict(settings, mode=mode)
        # Model
        model = ss.score_source(args.run_dir, "", settings_mode, other_prefix="pers_")
        # Persistence
        pers = ss.score_source(args.run_dir, "pers_", settings_mode, other_prefix="")
        # Keep
        out["modes"][mode] = {"model": model, "persistence": pers}
        # Print
        print(line(f"model / {mode}", model))
        # Print
        print(line(f"persistence / {mode}", pers))
        # Channels without thresholds, once per mode
        miss = model["calibration"]["channels_without_threshold"]
        # Report if any
        if miss["mag"] or miss["ang"]:
            # Say so
            print(f"  (model: no threshold for {len(miss['mag'])} mag and {len(miss['ang'])} ang channels)")
    # Path
    path = os.path.join(args.run_dir, args.out)
    # Write, with no NaN allowed so the file is standard JSON
    with open(path, "w") as f:
        # Pretty JSON, timestamps as strings
        json.dump(clean(out), f, indent=2, default=str, allow_nan=False)
    # Say so
    print(f"\nwrote {path}")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
