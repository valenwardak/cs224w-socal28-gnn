# =============================================================================
# File:        main_latefeatures.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-09
#
# Purpose:     Prepare experiment 2, the neighbor angle difference as a
#              model input. We want to test whether telling the model how
#              far each meter's voltage angle sits from its wired
#              neighbors' angles helps it forecast and detect. The model
#              computes that difference itself when it loads a month, but
#              it needs one thing the feature files do not carry: whether
#              each meter's capture was late at each slot, because late
#              captures gave poor angle agreement in Phase 1. This script
#              reads the capture offsets from the processed tables and
#              writes them per month as <YYYY-MM>_late.npz next to the
#              feature files, with late_manifest.json. Then it computes the
#              feature exactly as the model will, with the same helper, and
#              prints where it is available: per block, inside and outside
#              the scored episode windows, with the reasons for loss, and
#              per meter and voltage channel, so a meter that never
#              receives the feature is visible. Before the table it prints
#              the structural eligibility: which meters the wiring and the
#              phase map alone rule out in which edge sets, separately from
#              what the data then allows. The table is read before any
#              training.
#
# Usage (from the repo root):
#   python main_latefeatures.py
#   python main_latefeatures.py --table-only        (reprint the table from the late files already written)
#   Options: python main_latefeatures.py --help
#
# Functions:
#   parse_args()        command-line options with defaults
#   main()              write the late files and the manifest, validate the manifest, then print and save the availability
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: paths
import os
# Third party: timestamps
import pandas as pd

# The feature helpers
import socal_angdiff as sa
# The model data module, for the manifest, the splits and the months
import socal_modeldata as sd
# Standard library: the graph file
import json
# The scoring module, for the scored episodes
import socal_score as ss


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Write the per-month late-capture files and print where the neighbor angle difference is available")
    # Processed tables
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"), help="Folder with <YYYY-MM>/<meter>.parquet, the source of the capture offsets")
    # Feature folder
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with <YYYY-MM>.npz and manifest.json; the late files are written here")
    # Graph
    parser.add_argument("--graph", default="graph_meters_periods.json", help="Meter graph file, time-indexed or fixed: neighbors and the phase map")
    # Late threshold
    parser.add_argument("--late-threshold", type=float, default=sa.LATE_THRESHOLD, help="Capture offset in seconds at or beyond which a meter's angle is unusable; a missing offset is unusable too")
    # Resultant cutoff
    parser.add_argument("--min-resultant", type=float, default=sa.MIN_RESULTANT, help="Minimum mean resultant length of the neighbor mean for the direction to count")
    # Splits
    parser.add_argument("--splits", default="splits_v2.json", help="Split definition for the availability table")
    # Events
    parser.add_argument("--events", default=os.path.join("data", "processed", "events.csv"), help="Event table from preprocessing")
    # Window
    parser.add_argument("--before-minutes", type=float, default=1.0, help="Minutes before an episode's first scored change counted as inside the window (the scoring's default)")
    # After
    parser.add_argument("--after-minutes", type=float, default=10.0, help="Minutes after an episode's last scored change counted as inside the window (the scoring's default)")
    # Table only
    parser.add_argument("--table-only", action="store_true", help="Do not write the late files; print and save the availability from the existing ones")
    # Parse
    return parser.parse_args()


def main():
    """Write one late file per feature month and the manifest, then compute the feature and print its availability."""
    # Options
    args = parse_args()
    # Manifest
    manifest = sd.load_manifest(args.features_dir)
    # Months
    months = list(manifest["months"])
    # Graph file, fixed or time-indexed, read directly so this script needs no torch
    with open(args.graph) as f:
        # Parse
        body = json.load(f)
    # Periods and edge sets
    timeline = sa.timeline_from_graph(body)
    # Phase map identity, recorded with the table
    phase_id = sa.phase_map_identity(timeline["meters"], manifest["meters"])
    # Write unless table only
    if not args.table_only:
        # Each month
        written = []
        # Loop
        for month in months:
            # Feature month, for its slot times
            arrays = sd.load_month(args.features_dir, month, allow_missing=False, manifest=manifest)
            # Late flags from the processed tables
            late = sa.read_late_month(args.processed_dir, month, manifest["meters"], arrays["t"], args.late_threshold)
            # Write, with the meter order the columns follow
            path = sa.write_late(args.features_dir, month, arrays["t"], late, manifest["meters"])
            # Keep
            written.append(month)
            # Say
            print(f"wrote {path}: {late.shape[0]} slots, late meter-slots {int(late.sum())} of {late.size} ({late.mean():.3f})")
        # Manifest
        mpath = sa.write_late_manifest(args.features_dir, written, args.late_threshold, args.processed_dir, manifest["meters"])
        # Say
        print(f"wrote {mpath}")
    # The late manifest must exist and carry a finite, positive threshold equal to the requested one (also when only reprinting)
    late_manifest, _ = sa.read_late_manifest(args.features_dir, args.late_threshold)
    # Its meter list must be the manifest's
    if list(late_manifest["meters"]) != list(manifest["meters"]):
        # Raise
        raise ValueError(f"late_manifest.json lists meters {late_manifest['meters']} but the feature manifest lists {list(manifest['meters'])}")
    # Settings of the feature
    settings = {"graph": os.path.abspath(args.graph), "phase_map": phase_id, "late_threshold": args.late_threshold, "min_resultant": args.min_resultant, "splits": os.path.abspath(args.splits), "events": os.path.abspath(args.events), "before_minutes": args.before_minutes, "after_minutes": args.after_minutes, "scope_rule": "socal_score.episodes_in_scope: only episodes whose whole window lies in the block", "opportunity": "one slot of one meter's voltage channel that maps to a phase-to-neutral phase"}
    # Splits
    splits = sd.load_splits(args.splits)
    # Episodes
    episodes = ss.load_episodes(args.events)
    # Window lengths
    before, after = pd.Timedelta(minutes=args.before_minutes), pd.Timedelta(minutes=args.after_minutes)
    # Structural eligibility first: what the wiring and the phase map allow before any data
    elig = sa.structural_eligibility(timeline, timeline["meters"], manifest["meters"], manifest["channels"])
    # Print
    sa.print_eligibility(elig, manifest["meters"])
    # Rows
    rows = sa.availability(args.features_dir, splits, episodes, before, after, timeline, manifest, settings)
    # Print
    sa.print_availability(rows, manifest)
    # Save
    spath = sa.save_availability(args.features_dir, rows, settings)
    # Nothing valid anywhere is a fault in the inputs, not a result: stop with a clear error
    if sum(r.get("valid", 0) for r in rows) == 0:
        # Raise
        raise ValueError(f"the neighbor angle feature is valid in no block; the table is saved to {spath} for inspection, but training must not start on all-zero columns")
    # Note
    print(f"\nneighbor angle difference: equal-weight neighbor mean from {args.graph} (phase map {phase_id}), late threshold {args.late_threshold} s, minimum resultant {args.min_resultant}; episode windows as the scoring's; table saved to {spath}")


# Entry point
if __name__ == "__main__":
    # Run
    main()
