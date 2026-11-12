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
#              Scoring versions. Version 1 (fixed 2026-11-05) calibrates
#              every test block on --calib-label. Version 2 (defined
#              2026-11-06) adds --era-calib: rules "prefix=calib_label"
#              that give test blocks starting with a prefix their own
#              calibration block, so each era is calibrated on a held-out
#              quiet block of its own. The quiet block keeps the default,
#              so its version-2 result is the version-1 result reused, not
#              another evaluation. Version 2 is retrospective: the 2023
#              calibration block follows the September 2023 events, and
#              February 2025 follows the November 2024 events. Each
#              calibration block is checked against the run's training
#              blocks and an overlap raises. The version and freeze date
#              are recorded in the JSON; write version 2 to its own file.
#
#              Sensitivity exclusion (--exclude START END, added
#              2026-11-11, repeatable). A labeled sensitivity analysis that
#              removes the given [START, END) interval from monitoring and
#              alarms on the test blocks, keeps every threshold and
#              reference of the full-period scoring, recomputes monitored
#              exposure and merges alarms only within kept stretches, so no
#              alarm bridges the exclusion. Naive timestamps are read in
#              fixed Pacific Standard Time (UTC minus 8 hours), the
#              project's block convention; timestamps with an offset are
#              used as given. The intervals are recorded in the JSON in
#              UTC. The output must go to a file of its own: any --out
#              that resolves to one of the four primary files in the run
#              folder (score.json, score_grid.json, score_era.json,
#              score_grid_era.json, by resolved path, so ./score.json and
#              an absolute path count) is refused with --exclude, so the
#              primary results cannot be overwritten. Before a sensitivity
#              result is written it is checked against the saved primary
#              file of the same statistic and version (--primary, by
#              default that file in the run folder): the thresholds,
#              missing-threshold lists, references, grid thresholds,
#              calibration assignments and every setting other than the
#              exclusion record must be present on both sides and
#              identical, for every requested mode (magnitude and angle at
#              least) and both sources, or nothing is written and the
#              differences are printed. Without --exclude nothing changes.
#
# Usage:
#   python main_score.py --run-dir data/processed/models/gru_full_aw001
#   python main_score.py --run-dir /content/drive/MyDrive/socal_runs/gcn_full_aw001 --events /content/features/events.csv
#   python main_score.py --run-dir ... --modes mag ang both --alpha 1e-4 --min-meters 2
#   Version 2, per-era calibration, on a run trained with splits_v2.json:
#     python main_score.py --run-dir ... --era-calib test_events_2023=validation_2023-11-01_2023-11-14 --out score_era.json
#   Sensitivity analysis without the July 29 to August 10, 2025 period (fixed PST), to its own file, checked against score.json:
#     python main_score.py --run-dir ... --exclude 2025-07-29T00:00 2025-08-11T00:00 --out score_sens.json
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
#   protected_name(run_dir, out)
#                       the primary file name that --out resolves to, or None
#   primary_name(statistic, era_calib)
#                       the primary file name of a statistic and version
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
# Third party: timestamps for the exclusion option
import pandas as pd
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
    # Which statistic raises alarms
    parser.add_argument("--statistic", choices=["channel", "grid"], default="channel", help="channel = per-channel thresholds and a meter count (default, primary); grid = one threshold on the second-largest meter score (exploratory)")
    # Design block for the grid references
    parser.add_argument("--design-label", default="validation_2024-01-08_2024-01-21", help="Validation block 0, used for the grid reference scales, never for the final threshold")
    # Reference percentile
    parser.add_argument("--ref-percentile", type=float, default=99.0, help="Percentile of a channel's design-block residuals used as its reference scale")
    # Reference minimum count
    parser.add_argument("--ref-min-count", type=int, default=1000, help="Channels with fewer design-block residuals get no reference")
    # Reference floors
    parser.add_argument("--ref-floor-mag", type=float, default=1e-3, help="Floor for magnitude references, scaled units")
    # Angle floor
    parser.add_argument("--ref-floor-ang", type=float, default=1.0, help="Floor for angle references, degrees")
    # Common-channel restriction
    parser.add_argument("--restrict-common", action="store_true", help="Sensitivity analysis: limit both statistics to channels with both a threshold and a grid reference (needs the design block)")
    # Output name
    parser.add_argument("--out", default="score.json", help="JSON file name written into the run folder")
    # Scoring version 2: per-era calibration rules
    parser.add_argument("--era-calib", nargs="*", default=[], metavar="PREFIX=CALIB_LABEL", help="Version 2: test blocks whose label starts with PREFIX are calibrated on CALIB_LABEL instead of --calib-label; e.g. test_events_2023=validation_2023-11-01_2023-11-14")
    # Sensitivity exclusion intervals, repeatable, [START, END), naive times in fixed PST
    parser.add_argument("--exclude", nargs=2, action="append", default=[], metavar=("START", "END"), help="Sensitivity analysis: remove the [START, END) interval from monitoring and alarms, calibration unchanged; naive times are fixed PST (UTC-8); repeatable; needs --out other than the four primary files")
    # The saved primary file a sensitivity result is checked against
    parser.add_argument("--primary", default=None, help="With --exclude: the saved primary file whose protected values the sensitivity result must match; default is the primary file of the same statistic and version in the run folder")
    # Parse
    args = parser.parse_args()
    # Exclusion intervals to UTC strings
    exclusions = []
    # Loop
    for start, end in args.exclude:
        # Parse both
        try:
            # Timestamps
            s0, e0 = pd.Timestamp(start), pd.Timestamp(end)
        # Unreadable
        except (ValueError, TypeError):
            # Reject
            parser.error(f"--exclude needs two timestamps, got {start} {end}")
        # Naive means fixed PST, which is UTC minus 8 hours all year
        s0 = (s0 + pd.Timedelta(hours=8)).tz_localize("UTC") if s0.tzinfo is None else s0.tz_convert("UTC")
        # End
        e0 = (e0 + pd.Timedelta(hours=8)).tz_localize("UTC") if e0.tzinfo is None else e0.tz_convert("UTC")
        # Ordered
        if not s0 < e0:
            # Reject
            parser.error(f"--exclude start must be before end: {start} {end}")
        # Keep as ISO strings in UTC
        exclusions.append([s0.isoformat(), e0.isoformat()])
    # Store
    args.exclude = exclusions
    # The primary files are never written by a sensitivity run, whatever path spelling names them
    if args.exclude and protected_name(args.run_dir, args.out) is not None:
        # Reject
        parser.error(f"--exclude writes a sensitivity file; --out resolves to the primary file {protected_name(args.run_dir, args.out)}, got {args.out}")
    # The primary check needs --exclude
    if args.primary is not None and not args.exclude:
        # Reject
        parser.error("--primary is only used with --exclude")
    # A sensitivity run scores magnitude and angle, both checked against the primary file
    if args.exclude and not {"mag", "ang"} <= set(args.modes):
        # Reject
        parser.error(f"--exclude needs --modes to include mag and ang, got {args.modes}")
    # Era rules: "prefix=label" pairs, non-empty, no duplicate prefix
    rules = []
    # Loop
    for item in args.era_calib:
        # Split once
        if "=" not in item:
            # Reject
            parser.error(f"--era-calib rule must be PREFIX=CALIB_LABEL, got {item}")
        # Parts
        prefix, label = item.split("=", 1)
        # Both parts
        if not prefix or not label:
            # Reject
            parser.error(f"--era-calib rule has an empty part: {item}")
        # Duplicate prefix
        if any(prefix == r[0] for r in rules):
            # Reject
            parser.error(f"--era-calib prefix given twice: {prefix}")
        # Keep
        rules.append([prefix, label])
    # Store
    args.era_calib = rules
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
    # The grid statistic and the restriction both compute references, so both need the reference settings checked,
    # two meters (the second-largest meter score, and the same monitoring rule for both statistics), and separate blocks
    if args.statistic == "grid" or args.restrict_common:
        # Two meters
        if args.min_meters < 2:
            # Reject
            parser.error("--statistic grid and --restrict-common need --min-meters of at least 2")
        # Floors, positive and finite
        if not (0 < args.ref_floor_mag < float("inf")) or not (0 < args.ref_floor_ang < float("inf")):
            # Reject
            parser.error("--ref-floor-mag and --ref-floor-ang must be positive and finite")
        # Minimum count
        if args.ref_min_count < 1:
            # Reject
            parser.error("--ref-min-count must be at least 1")
        # Percentile
        if not 0.0 < args.ref_percentile < 100.0:
            # Reject
            parser.error("--ref-percentile must be between 0 and 100")
        # Separate blocks
        if args.design_label == args.calib_label:
            # Reject
            parser.error("--design-label and --calib-label must be different blocks")
    # The design block chose models, so it is never a calibration block under either statistic
    for prefix, label in args.era_calib:
        # Reject
        if label == args.design_label:
            # Say which
            parser.error(f"--era-calib {prefix}={label} uses the design block (model selection) as a calibration block")
    # Return
    return args


def protected_name(run_dir, out):
    """The primary file name (score.json, score_grid.json, score_era.json or score_grid_era.json) that --out resolves to inside the run folder, or None."""
    # Resolved destination, relative names inside the run folder as main writes them
    dest = os.path.realpath(os.path.join(run_dir, out))
    # Each primary file
    for name in ("score.json", "score_grid.json", "score_era.json", "score_grid_era.json"):
        # Same resolved file
        if dest == os.path.realpath(os.path.join(run_dir, name)):
            # Name
            return name
    # None
    return None


def primary_name(statistic, era_calib):
    """The primary file name of a statistic and scoring version: score.json, score_grid.json (grid), score_era.json (version 2), score_grid_era.json (both)."""
    # Name
    return "score" + ("_grid" if statistic == "grid" else "") + ("_era" if era_calib else "") + ".json"


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
        # Statistic and the grid settings
        "statistic": args.statistic, "design_label": args.design_label, "ref_percentile": args.ref_percentile, "ref_min_count": args.ref_min_count, "ref_floor_mag": args.ref_floor_mag, "ref_floor_ang": args.ref_floor_ang,
        # Restriction
        "restrict_common": args.restrict_common,
        # Scoring version: 2 with era rules, else 1, with the freeze dates
        "era_calib": args.era_calib,
        "scoring_version": 2 if args.era_calib else 1,
        "scoring_version_date": "2026-11-06" if args.era_calib else "2026-11-05",
        "scoring_note": ("version 2, per-era calibration: retrospective, the calibration blocks follow the events of their era in time; the quiet block's result is the version-1 result reused" if args.era_calib else "version 1, one calibration block for every test block"),
        # Sensitivity exclusion, recorded only when given so the frozen settings layout is unchanged otherwise
        **({"exclude": args.exclude, "exclude_note": "sensitivity analysis: the listed [start, end) UTC intervals are removed from monitoring and alarms on the test blocks; calibration unchanged; no alarm episode bridges an exclusion; the full-period results are primary"} if args.exclude else {}),
    }
    # Results per mode
    out = {"run_dir": os.path.abspath(args.run_dir), "settings": settings, "modes": {}}
    # Version line
    print(f"scoring version {settings['scoring_version']} ({settings['scoring_version_date']}): {settings['scoring_note']}")
    # Statistic line
    print(f"statistic: {args.statistic}" + (" (exploratory; references from the design block, one threshold from the calibration block)" if args.statistic == "grid" else " (primary)") + ("; restricted to channels with both a threshold and a reference (sensitivity analysis)" if args.restrict_common else ""))
    # Exclusion line
    if args.exclude:
        # Say
        print("sensitivity analysis: excluding " + "; ".join(f"[{a}, {b})" for a, b in args.exclude) + " from monitoring and alarms; calibration unchanged; full-period results remain primary")
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
        # Version 2: which calibration block each test block used, and rules that matched nothing, once
        if settings["era_calib"] and mode == args.modes[0]:
            # Resolution
            print("  calibration per test block: " + "; ".join(f"{b['label']} <- {b['calib_label']}" for b in model["blocks"]))
            # Unused rules
            if model["era_rules_unused"]:
                # Say so
                print(f"  WARNING: era rules that matched no test block: {model['era_rules_unused']}")
        # Exclusion record, once per mode
        if model.get("exclusions") is not None:
            # Print
            print(f"  excluded slots over the test blocks: {model['exclusions']['excluded_slots']}; episodes dropped from scope: {[d['episode'] for d in model['exclusions']['episodes_dropped']] or 'none'}")
        # Print
        print(line(f"model / {mode}", model))
        # Print
        print(line(f"persistence / {mode}", pers))
        # Per split, model and persistence: false alarms per day, alarm-slot fraction, merged episode spans
        for name, r in (("model", model), ("persistence", pers)):
            # Pieces
            parts = []
            # Loop over splits
            for p, v in r["per_split"].items():
                # Skip empty
                if v["false_alarms_per_day"] is None:
                    # Next
                    continue
                # Span text, merged spans are not time in alarm
                span = f"merged span median {v['merged_span_median_s']:.0f} s, max {v['merged_span_max_s']:.0f} s" if v["merged_span_median_s"] is not None else "no episodes"
                # Piece
                parts.append(f"{p.rstrip('_')}: {v['false_alarms_per_day']:.2f} FA/day over {v['monitored_days']:.1f} days, alarm-slot fraction {v['alarm_slot_fraction']:.2e}, {span}")
            # Print
            print(f"  {name} by split: " + " | ".join(parts))
        # Grid details, once per mode
        if model["grid"] is not None:
            # Both sources
            for name, r in (("model", model), ("persistence", pers)):
                # Line
                g = r["grid"]
                # Print
                print(f"  {name} grid: threshold {g['threshold']:.3f} on {g['n_calibration']} calibration slots, calibration coverage {g['calibration_coverage']:.5f}; {len(g['channels_without_reference']['mag'])} mag / {len(g['channels_without_reference']['ang'])} ang channels without reference, {len(g['channels_at_floor']['mag'])} mag / {len(g['channels_at_floor']['ang'])} ang at the floor; alarm-slot rate by available meters: " + ", ".join(f"{k}: {v['rate']:.2e}" for k, v in g['alarm_slots_by_available_meters'].items() if v['rate'] is not None))
        # Restriction, once per mode
        if model["restriction"] is not None:
            # Print
            print("  restriction: " + "; ".join(f"{k}: {v['channels_kept']} channels kept, dropped {v['channels_dropped']}" for k, v in model["restriction"].items() if k in ("mag", "ang")))
        # Channels without thresholds, per calibration block in use
        for c, rec in model["calibrations"].items():
            # Missing
            miss = rec["channels_without_threshold"]
            # Formatter for optional numbers
            g = lambda v, d=5: "n/a" if v is None else f"{v:.{d}f}"
            # Report
            print(f"  (model, calibration {c}: no threshold for {len(miss['mag'])} mag and {len(miss['ang'])} ang channels; coverage mag {g(rec['coverage_mag_mean'])} ang {g(rec['coverage_ang_mean'])}" + (f"; grid threshold {g(rec['grid_threshold'], 3)}, coverage {g(rec['grid_calibration_coverage'])}" if rec["grid_threshold"] is not None else "") + ")")
    # Path
    path = os.path.join(args.run_dir, args.out)
    # The result as it will be written: NaN to null, timestamps to strings, as the primary file was written
    final = json.loads(json.dumps(clean(out), default=str, allow_nan=False))
    # A sensitivity result is checked against its saved primary file before anything is written
    if args.exclude:
        # Primary file: given, or the one of this statistic and version in the run folder
        primary_path = args.primary if args.primary is not None else os.path.join(args.run_dir, primary_name(args.statistic, args.era_calib))
        # Say
        print(f"\nchecking protected values against the primary file {primary_path}")
        # Must exist, nothing is substituted
        if not os.path.isfile(primary_path):
            # Refuse
            raise SystemExit(f"FAIL: primary file {primary_path} not found; the sensitivity result was not written")
        # Load
        with open(primary_path) as f:
            # Saved result
            primary = json.load(f)
        # Differences
        findings = ss.protected_mismatches(final, primary, args.modes)
        # Any difference refuses the write
        if findings:
            # Each
            for item in findings:
                # Print
                print(f"FAIL  {primary_path}: {item}")
            # Refuse
            raise SystemExit(f"FAIL: {len(findings)} protected value(s) differ from {primary_path}; the sensitivity result was not written")
        # Say
        print(f"PASS  protected values identical to {primary_path} for modes {list(out['modes'])}, model and persistence")
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
