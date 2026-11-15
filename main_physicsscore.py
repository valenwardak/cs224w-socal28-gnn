# =============================================================================
# File:        main_physicsscore.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-15
#
# Purpose:     Score the physics baseline alone as an event detector, under
#              the alarm rules the learned models are scored with. The
#              proposal names a physics state estimator with no learning as
#              a baseline, so the comparison of detectors needs its row. All
#              the work is in socal_physicsscore.py. This file only reads the
#              options, runs the scoring and writes the score file.
#
# Usage (from the repo root), after main_physicsfeatures.py has written the
# <YYYY-MM>_physics.npz files:
#   Full period, the primary count:
#     python main_physicsscore.py --physics-dir data/features
#   Sensitivity analysis without a period (naive times are fixed PST), into its own file:
#     python main_physicsscore.py --physics-dir data/features --exclude 2025-07-29T00:00 2025-08-11T00:00 --out score_physics_sens.json
#   Options: python main_physicsscore.py --help
#
# Output under <out_dir>:
#   score_physics.json   settings, thresholds per side, valid slots per block,
#                        and for each of the three detectors (either side,
#                        side A, side B) the episodes caught, the coverage per
#                        episode, and the false alarms per monitored day
#
# Functions:
#   parse_args()   command-line options with defaults
#   main()         read the splits, score, print the summary, write the file
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: the score file
import json
# Standard library: paths
import os
# Third party: timestamps of the exclusion
import pandas as pd

# The model data module, for the splits file
import socal_modeldata as sd
# The scoring of the physics baseline
import socal_physicsscore as ps


def parse_args():
    """Read the command-line options. Every setting has a default, and the defaults are those of the scoring of the models."""
    # Parser
    parser = argparse.ArgumentParser(description="Score the physics baseline alone as an event detector")
    # Folder of the physics files
    parser.add_argument("--physics-dir", default=os.path.join("data", "features"), help="Folder with <YYYY-MM>_physics.npz (main_physicsfeatures.py)")
    # Splits
    parser.add_argument("--splits", default="splits_v2.json", help="Split definition; its blocks give the calibration block and the test blocks")
    # Events
    parser.add_argument("--events", default=os.path.join("data", "processed", "events.csv"), help="Event table from preprocessing")
    # Calibration block
    parser.add_argument("--calib-label", default="validation_2025-02-03_2025-02-16", help="Label of the block that sets the thresholds")
    # Test splits
    parser.add_argument("--test-prefixes", nargs="+", default=["test_events_", "test_normal_"], help="Starts of the labels of the blocks that are scored")
    # Level
    parser.add_argument("--alpha", type=float, default=1e-4, help="Conformal level of the threshold of each side")
    # Merging
    parser.add_argument("--merge-slots", type=int, default=6, help="Alarm slots at most this many slots apart join one alarm episode")
    # Window before
    parser.add_argument("--before-minutes", type=float, default=1.0, help="Minutes before an episode's first scored change that count as its window")
    # Window after
    parser.add_argument("--after-minutes", type=float, default=10.0, help="Minutes after an episode's last scored change that count as its window")
    # Bootstrap resamples
    parser.add_argument("--n-boot", type=int, default=1000, help="Resamples of the bootstrap interval of the recall")
    # Bootstrap seed
    parser.add_argument("--seed", type=int, default=0, help="Seed of the bootstrap")
    # Sensitivity exclusion
    parser.add_argument("--exclude", nargs=2, action="append", default=[], metavar=("START", "END"), help="Sensitivity analysis: remove the [START, END) interval from monitoring and alarms, thresholds unchanged; naive times are fixed PST (UTC-8); repeatable")
    # Output folder
    parser.add_argument("--out-dir", default=os.path.join("data", "processed", "physics_score"), help="Folder for the score file")
    # Output file
    parser.add_argument("--out", default="score_physics.json", help="Name of the score file")
    # Parse
    args = parser.parse_args()
    # Exclusion intervals as UTC strings
    exclusions = []
    # Each interval
    for start, end in args.exclude:
        # Both bounds as timestamps
        s0, e0 = pd.Timestamp(start), pd.Timestamp(end)
        # A naive time is fixed PST, which is UTC minus 8 hours all year
        s0 = (s0 + pd.Timedelta(hours=8)).tz_localize("UTC") if s0.tzinfo is None else s0.tz_convert("UTC")
        # The same for the end
        e0 = (e0 + pd.Timedelta(hours=8)).tz_localize("UTC") if e0.tzinfo is None else e0.tz_convert("UTC")
        # The start must come first
        if not s0 < e0:
            # Refuse
            parser.error(f"--exclude start must be before end: {start} {end}")
        # Keep
        exclusions.append([s0.isoformat(), e0.isoformat()])
    # Store
    args.exclude = exclusions
    # Done
    return args


def main():
    """Read the splits, score the physics baseline, print the summary and write the score file."""
    # Options
    args = parse_args()
    # The settings saved in the score file. The exclusion is saved only when one is given, as in the score files of the models.
    settings = {"physics_dir": os.path.abspath(args.physics_dir), "splits": os.path.abspath(args.splits), "events": os.path.abspath(args.events), "calib_label": args.calib_label, "test_prefixes": list(args.test_prefixes), "alpha": args.alpha, "merge_slots": args.merge_slots, "before_minutes": args.before_minutes, "after_minutes": args.after_minutes, "n_boot": args.n_boot, "seed": args.seed, **({"exclude": args.exclude} if args.exclude else {})}
    # Score
    body = ps.score(args.physics_dir, sd.load_splits(args.splits), settings)
    # Summary
    ps.print_summary(body)
    # Output folder
    os.makedirs(args.out_dir, exist_ok=True)
    # The score file
    path = os.path.join(args.out_dir, args.out)
    # Write it
    with open(path, "w") as f:
        # JSON
        json.dump(body, f, indent=1)
    # Say where it is
    print(f"\nwrote {path}")


# Entry point
if __name__ == "__main__":
    # Run
    main()
