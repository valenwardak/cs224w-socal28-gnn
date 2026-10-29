# =============================================================================
# File:        main_scoreview.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-29
#
# Purpose:     Read what the scoring already stored and print it side by
#              side: the channel statistic's score.json and the grid
#              statistic's score_grid.json for one run. No scoring logic.
#              Prints:
#                1. the settings of both files, with any difference beyond
#                   the statistic and the grid reference settings flagged
#                2. exclusions, per mode and source and per residual kind:
#                   channels without a threshold (channel statistic),
#                   channels without a reference (grid), split into both,
#                   grid only, channel only; channels at the floor listed
#                   separately since they stay included
#                3. paired episode outcomes by episode id: detected under
#                   channel, under grid, both, lost, gained, with delays
#                4. exposure per split for both statistics: monitored days,
#                   false alarms per day, alarm-slot fraction, merged-span
#                   median and maximum
#                5. the first N unmatched alarms of each statistic
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root.
#
# Usage:
#   python checks/main_scoreview.py --run-dir /content/drive/MyDrive/socal_runs/gcn_full_aw001
#   python checks/main_scoreview.py --run-dir ... --modes ang --top 10
#
# Functions:
#   parse_args()                   command-line options with defaults
#   load(run_dir, name)            read one score file
#   section(title)                 print a section header
#   settings_diff(a, b, ignore)    settings that differ beyond the expected keys
#   exclusions(ch, gr, kind)       both / grid only / channel only sets for one kind
#   episodes_by_id(r)              episode results keyed by episode id
#   main()                         print the five parts
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON reading
import json
# Standard library: paths
import os


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Read a run's channel and grid score files side by side")
    # Run folder
    parser.add_argument("--run-dir", required=True, help="Run folder holding the score files")
    # Channel statistic file
    parser.add_argument("--channel-score", default="score.json", help="Score file of the channel statistic")
    # Grid statistic file
    parser.add_argument("--grid-score", default="score_grid.json", help="Score file of the grid statistic")
    # Modes
    parser.add_argument("--modes", nargs="+", default=["mag", "ang", "both"], help="Modes to print")
    # Unmatched alarms to list
    parser.add_argument("--top", type=int, default=5, help="Unmatched alarms listed per statistic, mode, and source")
    # Parse
    return parser.parse_args()


def load(run_dir, name):
    """Read one score file, or raise with the path."""
    # Path
    path = os.path.join(run_dir, name)
    # Must exist
    if not os.path.exists(path):
        # Raise
        raise FileNotFoundError(f"no score file at {path}")
    # Read
    with open(path) as f:
        # Parse
        return json.load(f)


def section(title):
    """Print a section header."""
    # Blank line and title
    print(f"\n=== {title} ===")


def settings_diff(a, b, ignore):
    """Keys whose values differ between two settings dicts, beyond the ignored keys."""
    # Union of keys
    keys = sorted(set(a) | set(b))
    # Differences
    return {k: (a.get(k), b.get(k)) for k in keys if k not in ignore and a.get(k) != b.get(k)}


def exclusions(ch, gr, kind):
    """
    Exclusion sets for one residual kind: channels without a threshold
    (channel statistic), channels without a reference (grid), and their
    overlap. Returns (both, grid_only, channel_only, floored).
    """
    # Channel statistic exclusions
    c = set(ch["calibration"]["channels_without_threshold"][kind])
    # Grid exclusions
    g = set(gr["grid"]["channels_without_reference"][kind]) if gr.get("grid") else set()
    # Floored, still included
    f = set(gr["grid"]["channels_at_floor"][kind]) if gr.get("grid") else set()
    # Return
    return sorted(c & g), sorted(g - c), sorted(c - g), sorted(f)


def episodes_by_id(r):
    """Episode results keyed by episode id."""
    # Dictionary
    return {e["episode"]: e for e in r["episode_results"]}


def main():
    """Print the five parts."""
    # Options
    args = parse_args()
    # Files
    ch_all, gr_all = load(args.run_dir, args.channel_score), load(args.run_dir, args.grid_score)
    # Header
    print(f"run {args.run_dir}: {args.channel_score} (statistic {ch_all['settings'].get('statistic', 'channel')}) vs {args.grid_score} (statistic {gr_all['settings'].get('statistic', 'channel')})")

    # 1. Settings
    section("1. settings")
    # Keys expected to differ
    ignore = {"statistic", "design_label", "ref_percentile", "ref_min_count", "ref_floor_mag", "ref_floor_ang"}
    # Shared settings
    for k, v in sorted(ch_all["settings"].items()):
        # Print
        print(f"  {k}: {v}" + (f"  |  grid file: {gr_all['settings'].get(k)}" if k in ignore else ""))
    # Differences beyond the expected
    diffs = settings_diff(ch_all["settings"], gr_all["settings"], ignore)
    # Flag
    print("  settings differ beyond the statistic: " + (str(diffs) if diffs else "none"))

    # Loop over modes
    for mode in args.modes:
        # Skip modes absent from either file
        if mode not in ch_all["modes"] or mode not in gr_all["modes"]:
            # Say so
            print(f"\nmode {mode} missing from one of the files, skipped")
            # Next
            continue
        # Loop over sources
        for source in ("model", "persistence"):
            # Results
            ch, gr = ch_all["modes"][mode][source], gr_all["modes"][mode][source]
            # 2. Exclusions
            section(f"2. exclusions, mode {mode}, {source}")
            # Per kind
            for kind in ("mag", "ang"):
                # Sets
                both, g_only, c_only, floored = exclusions(ch, gr, kind)
                # Print
                print(f"  {kind}: excluded by both ({len(both)}): {both}")
                # Grid only
                print(f"       grid only, no reference ({len(g_only)}): {g_only}")
                # Channel only
                print(f"       channel only, no threshold ({len(c_only)}): {c_only}")
                # Floored
                print(f"       at the floor, still included ({len(floored)}): {floored}")
            # Grid threshold
            if gr.get("grid"):
                # Print
                print(f"  grid threshold {gr['grid']['threshold']:.3f}, calibration coverage {gr['grid']['calibration_coverage']:.5f}, meters needed {gr['grid']['meters_needed']}")

            # 3. Paired episodes
            section(f"3. episodes by id, mode {mode}, {source}: channel vs grid")
            # Keyed
            ce, ge = episodes_by_id(ch), episodes_by_id(gr)
            # Union of ids
            ids = sorted(set(ce) | set(ge))
            # Header
            print(f"  {'episode':>7} {'first change':<26} {'channel':<22} {'channel alarm start':<26} {'grid':<22} {'grid alarm start':<26} outcome")
            # Tallies, scope differences counted apart from detection outcomes
            tally = {"both": 0, "lost": 0, "gained": 0, "neither": 0, "scope": 0}
            # Text for one record
            text = lambda r: "not in scope" if r is None else (f"detected, delay {r['delay_s']:.0f} s" if r["detected"] else "missed")
            # Alarm start for one record
            start = lambda r: "" if r is None or r.get("alarm_start") is None else r["alarm_start"]
            # Loop
            for i in ids:
                # Records
                c, g = ce.get(i), ge.get(i)
                # Scope differs: only one file has the episode
                if c is None or g is None:
                    # Label
                    outcome = "scope differs"
                    # Tally
                    tally["scope"] += 1
                # Both in scope: compare detections
                else:
                    # Flags
                    cd, gd = bool(c["detected"]), bool(g["detected"])
                    # Label
                    outcome = "both" if cd and gd else ("lost under grid" if cd else ("gained under grid" if gd else "neither"))
                    # Tally
                    tally[{"both": "both", "lost under grid": "lost", "gained under grid": "gained", "neither": "neither"}[outcome]] += 1
                # First change
                first = (c or g)["first"]
                # Print
                print(f"  {i:>7} {first:<26} {text(c):<22} {start(c):<26} {text(g):<22} {start(g):<26} {outcome}")
            # Summary, detection outcomes over episodes in both files only
            print(f"  episodes in both files: detected under both {tally['both']}, lost under grid {tally['lost']}, gained under grid {tally['gained']}, neither {tally['neither']}; scope differs {tally['scope']}")

            # 4. Exposure per split
            section(f"4. exposure per split, mode {mode}, {source}")
            # Header
            print(f"  {'statistic':<9} {'split':<13} {'days':>7} {'FA/day':>8} {'alarm-slot frac':>16} {'span med s':>11} {'span max s':>11}")
            # Both statistics
            for name, r in (("channel", ch), ("grid", gr)):
                # Splits
                for p, v in r["per_split"].items():
                    # Format
                    f = lambda x, w, d: (" " * (w - 3) + "n/a") if x is None else f"{x:>{w}.{d}f}"
                    # Print
                    print(f"  {name:<9} {p.rstrip('_'):<13} {f(v['monitored_days'], 7, 1)} {f(v['false_alarms_per_day'], 8, 2)} {f(v.get('alarm_slot_fraction'), 16, 6)} {f(v.get('merged_span_median_s'), 11, 0)} {f(v.get('merged_span_max_s'), 11, 0)}")

            # 5. Unmatched alarms
            section(f"5. first {args.top} unmatched alarms, mode {mode}, {source}")
            # Both statistics
            for name, r in (("channel", ch), ("grid", gr)):
                # List
                fl = r["false_alarm_list"]
                # Print
                print(f"  {name} ({len(fl)} total): " + "; ".join(f"{s} to {e}" for s, e in fl[:args.top]))


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
