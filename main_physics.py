# =============================================================================
# File:        main_physics.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-20
#
# Purpose:     Run the physics baseline (least-squares state estimator, no
#              learning) over a time window or a named split and write the
#              residual per ten-second slot. All the work is in
#              socal_physics.py; this file only parses options and loops
#              over blocks.
#
#              The reference wiring and the injection map come from
#              physics_config.json. Caltech's utils are read from
#              caltech_utils/ with their paths redirected at startup (see
#              the header of socal_physics.py), so their files stay unchanged.
#
# Usage:
#   First check, Caltech's normal hour (paper Section VI-A, 0.851 percent):
#     python main_physics.py --start 2024-11-14T15:00:00 --end 2024-11-14T16:00:00
#   One named split from splits.json, numpy solver for speed:
#     python main_physics.py --split test_events --solver numpy
#   B-side (component 0) with the rank safeguard, into its own folder:
#     python main_physics.py --config physics_config_b.json --rank-check --split test_normal --splits splits_v2.json --solver numpy --out-dir data/processed/physics_b
#   With the Caltech-inspired offset filter (whole slot dropped when any planned meter captured 0.5 s or more from the slot time), into its own folder:
#     python main_physics.py --rank-check --offset-threshold 0.5 --solver numpy --split test_normal --splits splits_v2.json --out-dir data/processed/physics_offset05
#   Per-meter variant (only the late meter's rows dropped, the rest solved if it still determines the state):
#     python main_physics.py --rank-check --offset-threshold 0.5 --offset-policy per_meter --solver numpy --split test_events --splits splits_v2.json --out-dir data/processed/physics_permeter05
#   Options: python main_physics.py --help
#
# Output under <out_dir>:
#   <label>.parquet   residual per slot: residual, residual_mean, n_voltage,
#                     n_current, v_dropped, i_dropped, status
#   <label>.json      summary numbers and the options used, including the
#                     offset rule's threshold, policy and meters when it is on
#
# Functions:
#   parse_args()                 command-line options with defaults
#   blocks_from_args(args)       list of (label, start_utc, end_utc) to run
#   main()                       build the model once, run each block, write
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON for splits and summaries
import json
# Standard library: file and folder operations
import os
# Third party: timestamps
import pandas as pd
# Project helpers
import socal_physics as sp


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser with the file's purpose as description
    parser = argparse.ArgumentParser(description="Physics baseline: least-squares state estimator residual per slot")
    # Processed data folder
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"), help="Folder with <YYYY-MM>/<meter>.parquet")
    # Topology folder
    parser.add_argument("--topology-dir", default="topology", help="Caltech topology folder copied into the repo")
    # Caltech utils folder
    parser.add_argument("--caltech-dir", default="caltech_utils", help="Caltech utils folder copied into the repo")
    # Config file
    parser.add_argument("--config", default="physics_config.json", help="Reference model and injection map")
    # Splits file
    parser.add_argument("--splits", default="splits.json", help="Split definition written by main_splits.py")
    # Named split
    parser.add_argument("--split", default=None, help="Run every block of this split: train, validation, test_events, test_normal")
    # Explicit window start
    parser.add_argument("--start", default=None, help="Window start, UTC, like 2024-11-14T15:00:00 (used when --split is not given)")
    # Explicit window end
    parser.add_argument("--end", default=None, help="Window end, UTC, exclusive")
    # Output folder
    parser.add_argument("--out-dir", default=os.path.join("data", "processed", "physics"), help="Folder for results")
    # Solver
    parser.add_argument("--solver", default="cvxpy", choices=["cvxpy", "numpy"], help="cvxpy matches Caltech's objective, numpy is the fast least-squares version")
    # Current weight
    parser.add_argument("--current-weight", type=float, default=None, help="Weight on current residuals, default from config (Caltech uses 0.1)")
    # Slot stride
    parser.add_argument("--step", type=int, default=1, help="Solve every step-th slot, 1 means all")
    # Frequency tolerance
    parser.add_argument("--freq-tolerance", type=float, default=0.01, help="Hertz; a reading is dropped when its own frequency differs from the meter's voltage frequency by more (0.01 means the same FFT bin)")
    # No progress bar
    parser.add_argument("--quiet", action="store_true", help="Do not show a progress bar")
    # Rank safeguard, off by default so earlier runs and results stay reproducible
    parser.add_argument("--rank-check", action="store_true", help="Mark a slot 'undetermined' (no residual) when its available measurements do not determine the state; off by default")
    # Its tolerance
    parser.add_argument("--rank-rel-tol", type=float, default=1e-9, help="Singular values below this fraction of the largest count as zero in --rank-check")
    # Offset filter, off by default; Caltech's produce_phasors_matrix uses 0.5 s measured from the synchronized request time, ours is measured from the ten-second slot
    parser.add_argument("--offset-threshold", type=float, default=None, help="Seconds; mark a slot 'offset_rejected' (no residual) when any meter the plan uses captured this far or farther from the slot time, whole slot, as Caltech's 0.5 s rule does per timestamp; off by default")
    # Policy of the offset filter
    parser.add_argument("--offset-policy", choices=["whole_slot", "per_meter"], default="whole_slot", help="whole_slot: any late planned meter rejects the slot (Caltech's rule); per_meter: only the late meter's voltage and current rows are dropped and the rest is solved if it still determines the state (needs --rank-check)")
    # Parse
    args = parser.parse_args()
    # A threshold must be positive
    if args.offset_threshold is not None and args.offset_threshold <= 0:
        # Refuse
        parser.error("--offset-threshold must be a positive number of seconds")
    # The per-meter policy needs the rank check
    if args.offset_threshold is not None and args.offset_policy == "per_meter" and not args.rank_check:
        # Refuse
        parser.error("--offset-policy per_meter requires --rank-check")
    # Done
    return args


def blocks_from_args(args):
    """
    Return the list of (label, start_utc, end_utc) windows to run.
    With --split, every block of that split from splits.json.
    Otherwise the single --start / --end window.
    """
    # Named split
    if args.split:
        # Read splits
        with open(args.splits) as f:
            # Parse
            splits = json.load(f)
        # The split must exist
        if args.split not in splits:
            # Raise
            raise ValueError(f"Split {args.split} not in {args.splits}")
        # One window per block
        return [(f"{args.split}_{b['start']}_{b['end']}", b["start_utc"], b["end_utc_exclusive"]) for b in splits[args.split]]
    # Explicit window needs both ends
    if not (args.start and args.end):
        # Raise
        raise ValueError("Give --split, or both --start and --end")
    # Label from the window
    label = f"window_{args.start}_{args.end}".replace(":", "")
    # One window
    return [(label, args.start, args.end)]


def main():
    """Build the reference model once, then run and write each block."""
    # Options
    args = parse_args()
    # Config
    config = sp.load_config(args.config)
    # Current weight: option overrides config
    weight = args.current_weight if args.current_weight is not None else float(config.get("current_weight", 0.1))
    # Caltech modules with paths redirected
    ct = sp.setup_caltech(args.caltech_dir, args.topology_dir)
    # Reference model
    model = sp.build_reference_model(ct, args.topology_dir, config)
    # Measurement plan
    plan = sp.measurement_plan(model, config)
    # Report the model
    print(f"Reference time {config['reference_time']}: {model['n_nodes']} nodes, {len(model['edges'])} edges, Y {model['Y'].shape}")
    # Report the plan
    print(f"Plan: {len(plan['voltages'])} voltage terms, {len(plan['injections'])} measured injection nodes, {len(plan['zeros'])} zero-injection nodes, meters {plan['meters']}")
    # Output folder
    os.makedirs(args.out_dir, exist_ok=True)
    # Loop over blocks
    for label, start_utc, end_utc in blocks_from_args(args):
        # Announce
        print(f"\nBlock {label}: {start_utc} to {end_utc}")
        # Data
        table = sp.load_block(args.processed_dir, plan["meters"], start_utc, end_utc)
        # Nothing to do
        if table.empty:
            # Say so
            print("  no processed data in this window")
            # Next
            continue
        # Solve
        results = sp.run_block(model, plan, table, solver=args.solver, weight=weight, step=args.step, progress=not args.quiet, freq_tolerance=args.freq_tolerance, rank_check=args.rank_check, rank_rel_tol=args.rank_rel_tol, offset_threshold=args.offset_threshold, offset_policy=args.offset_policy)
        # Summary
        summary = sp.summarize(results)
        # Add the options used, the offset rule's settings copied from the results so the file records what was applied
        summary.update({"label": label, "start_utc": start_utc, "end_utc": end_utc, "solver": args.solver, "current_weight": weight, "step": args.step, "freq_tolerance": args.freq_tolerance, "reference_time": config["reference_time"],
                        "config": args.config, "rank_check": args.rank_check, "rank_rel_tol": args.rank_rel_tol if args.rank_check else None,
                        "offset_threshold": results.attrs["offset_threshold"], "offset_policy": results.attrs["offset_policy"], "offset_meters": results.attrs["offset_meters"]})
        # Write results
        results.to_parquet(os.path.join(args.out_dir, f"{label}.parquet"))
        # Write summary
        with open(os.path.join(args.out_dir, f"{label}.json"), "w") as f:
            # Pretty JSON
            json.dump(summary, f, indent=2, default=str)
        # Print summary
        print("  " + json.dumps(summary, default=str))


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
