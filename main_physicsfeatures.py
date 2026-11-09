# =============================================================================
# File:        main_physicsfeatures.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-09
#
# Purpose:     Export the physics baseline's residuals as model inputs and
#              print where they are available. We want to test whether
#              telling the forecasting model how badly the last minutes of
#              readings fit the known wiring changes its false alarms. For
#              that, every ten-second slot of every feature month needs the
#              A-side and B-side physics residual, or an explicit "not
#              available" flag, in a file next to the feature month. This
#              script reads two folders of physics results (one per side,
#              produced by main_physics.py month by month with the rules in
#              force), aligns them to the feature months, writes
#              <YYYY-MM>_physics.npz and physics_manifest.json, and prints
#              the availability table over every block of the splits file,
#              inside and outside the scored episode windows. The table is
#              read before any training; a side with no values in a block
#              cannot help there, and the report says so.
#
# Usage (from the repo root), after the two physics runs:
#   python main_physicsfeatures.py --physics-a data/processed/physics_feature_a --physics-b data/processed/physics_feature_b
#   python main_physicsfeatures.py --table-only      (reprint the availability table from the files already written)
#   Options: python main_physicsfeatures.py --help
#
# Functions:
#   parse_args()        command-line options with defaults
#   main()              read results, write the month files and the manifest, print availability
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: paths
import os
# Third party: numerics
import numpy as np
# Third party: timestamps
import pandas as pd

# The export helpers
import socal_physicsfeatures as pf
# The model data module, for the manifest and the splits
import socal_modeldata as sd
# The scoring module, for the scored episodes
import socal_score as ss


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Export physics residuals as model inputs and print their availability per block")
    # Feature folder
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with <YYYY-MM>.npz and manifest.json; the physics files are written here")
    # A-side results
    parser.add_argument("--physics-a", default=os.path.join("data", "processed", "physics_feature_a"), help="Folder of A-side physics results (main_physics.py, physics_config.json)")
    # B-side results
    parser.add_argument("--physics-b", default=os.path.join("data", "processed", "physics_feature_b"), help="Folder of B-side physics results (main_physics.py, physics_config_b_v2.json)")
    # Splits for the table
    parser.add_argument("--splits", default="splits_v2.json", help="Split definition for the availability table")
    # Events for the windows
    parser.add_argument("--events", default=os.path.join("data", "processed", "events.csv"), help="Event table from preprocessing")
    # Window, the scoring's defaults
    parser.add_argument("--before-minutes", type=float, default=1.0, help="Minutes before an episode's first scored change counted as inside the window (the scoring's default)")
    # After
    parser.add_argument("--after-minutes", type=float, default=10.0, help="Minutes after an episode's last scored change counted as inside the window (the scoring's default)")
    # Table only
    parser.add_argument("--table-only", action="store_true", help="Do not write the physics files; print and save the availability table from the existing ones")
    # Required safeguards of the result folders
    parser.add_argument("--require-threshold", type=float, default=0.5, help="The offset threshold the result folders must have been produced with")
    # Policy
    parser.add_argument("--require-policy", choices=["whole_slot", "per_meter"], default="whole_slot", help="The offset policy the result folders must have been produced with")
    # Parse
    return parser.parse_args()


def main():
    """Read both result folders, write one physics file per feature month, then print the availability table."""
    # Options
    args = parse_args()
    # Manifest, for the month list
    manifest = sd.load_manifest(args.features_dir)
    # Months
    months = list(manifest["months"])
    # Write unless table only
    if not args.table_only:
        # Results per side
        results = {"a": pf.load_results(args.physics_a), "b": pf.load_results(args.physics_b)}
        # Settings per side: present, identical within each folder, and equal to the required safeguards
        required = {"rank_check": True, "offset_threshold": args.require_threshold, "offset_policy": args.require_policy}
        # Check
        settings = {"a": pf.result_settings(args.physics_a, required), "b": pf.result_settings(args.physics_b, required)}
        # Say
        for side in pf.SIDES:
            # Line
            print(f"side {side}: {len(results[side])} physics slots from {args.physics_a if side == 'a' else args.physics_b}; settings {settings[side]}")
        # Each month
        written = []
        # Loop
        for month in months:
            # Feature month, for its slot times
            arrays = sd.load_month(args.features_dir, month, allow_missing=False, manifest=manifest)
            # Times
            t = arrays["t"]
            # Per side
            sides = {side: pf.month_arrays(results[side], t) for side in pf.SIDES}
            # Write
            path = pf.write_month(args.features_dir, month, t, sides)
            # Keep
            written.append(month)
            # Say
            print(f"wrote {path}: {len(t)} slots, valid A {int(sides['a'][1].sum())}, valid B {int(sides['b'][1].sum())}")
        # Manifest
        mpath = pf.write_manifest(args.features_dir, {"a": os.path.abspath(args.physics_a), "b": os.path.abspath(args.physics_b)}, written, settings)
        # Say
        print(f"wrote {mpath}")
    # Splits
    splits = sd.load_splits(args.splits)
    # Scored episodes
    episodes = ss.load_episodes(args.events)
    # Window lengths
    before, after = pd.Timedelta(minutes=args.before_minutes), pd.Timedelta(minutes=args.after_minutes)
    # Table
    rows = pf.availability(args.features_dir, splits, episodes, before, after)
    # Print
    pf.print_availability(rows)
    # Save
    spath = pf.save_availability(args.features_dir, rows, {"splits": os.path.abspath(args.splits), "events": os.path.abspath(args.events), "before_minutes": args.before_minutes, "after_minutes": args.after_minutes, "scope_rule": "socal_score.episodes_in_scope: only episodes whose whole window lies in the block"})
    # Note on the windows
    print(f"\nepisode windows: first scored change minus {args.before_minutes} min to last scored change plus {args.after_minutes} min, {len(episodes)} scored episodes from {args.events}, counted only where the whole window lies in the block; table saved to {spath}")


# Entry point
if __name__ == "__main__":
    # Run
    main()
