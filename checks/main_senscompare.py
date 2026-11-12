# =============================================================================
# File:        main_senscompare.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-11
#
# Purpose:     Read-only comparison of a run's sensitivity score files
#              (written with main_score.py --exclude) against its primary
#              score files. The July 29 to August 10, 2025 exclusion lies
#              inside the quiet block, so the quiet-block exposure, alarms
#              and false-alarm figures are expected to change, and the
#              pooled whole-test figures with them. Everything that depends
#              only on the event blocks must not change, and this check
#              says so field by field, for magnitude and angle, model and
#              persistence:
#                - event-block records (every block whose label starts
#                  with an event prefix): calibration assignment, monitored
#                  slots, alarm slots, alarm fraction, merged-alarm count
#                  and span summaries equal; an added excluded_slots field
#                  must be zero and is the only field ignored;
#                - episode scope: the same episode identities and count,
#                  the same partly covered list, and no episode dropped by
#                  the exclusion;
#                - detections: each episode's detected flag, matched alarm
#                  start and delay equal;
#                - event-block false alarms: the unmatched alarms that
#                  start inside an event block, and the event-split
#                  false-alarm summary, equal.
#              Before any value is compared, each file must hold the
#              required modes (magnitude and angle by default) with a model
#              and a persistence result each, and each result must hold the
#              required fields with the right shape; a field missing from
#              both sides is a failure, not a match. Legitimately empty
#              lists and the nullable values the scoring writes (no alarm
#              start or delay for a missed episode, no span summary for a
#              block without episodes, no alarm fraction for a block with
#              no monitored slot) are allowed. Every file pair selected with
#              --pairs must exist on both sides, so a run with only the
#              version-1 sensitivity files is checked by selecting those
#              two pairs. The check prints how many file pairs and how many
#              mode/source comparisons it completed and fails when none
#              were. It compares the fields that are in the saved files and
#              adds nothing to them. A FAIL is a finding to investigate,
#              never a reason to change settings or touch a primary file.
#
# Usage (from the repo root):
#   python checks/main_senscompare.py --run-dirs RUN1 RUN2 --pairs score_sens.json=score.json score_grid_sens.json=score_grid.json
#   python checks/main_senscompare.py --run-dirs /content/drive/MyDrive/socal_runs/gcn_dyn_aw001
#
# Functions:
#   parse_args()                      command-line options with defaults
#   load(path)                        read one JSON file
#   in_event_block(ts, labels, prefixes)
#                                     whether a timestamp lies in one of the event blocks
#   schema_findings(result, side, prefixes, is_sens)
#                                     missing or malformed required fields of one result
#   compare_source(sens, primary, prefixes)
#                                     findings for one mode and source
#   compare_files(sens, primary, prefixes, modes)
#                                     findings per required mode and source for one file pair
#   main()                            compare every pair of every run, print PASS/FAIL and counts, exit 1 on any FAIL
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON
import json
# Standard library: paths
import os
# Standard library: exit code
import sys
# Third party: timestamps
import pandas as pd

# The repo root, one level above checks/, so the project module imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Import from the root
sys.path.insert(0, ROOT)

# Project: block bounds from labels
import socal_score as ss

# The four file pairs by default, sensitivity file to primary file
DEFAULT_PAIRS = ["score_sens.json=score.json", "score_grid_sens.json=score_grid.json", "score_era_sens.json=score_era.json", "score_grid_era_sens.json=score_grid_era.json"]
# The event-block record fields compared one by one
BLOCK_FIELDS = ["calib_label", "slots", "monitored_slots", "alarm_slots", "alarm_slot_fraction", "alarm_episodes", "merged_span_median_s", "merged_span_max_s"]
# Block fields that the scoring may write as null (no monitored slot, no episode)
BLOCK_NULLABLE = {"alarm_slot_fraction", "merged_span_median_s", "merged_span_max_s"}
# Required fields of a result and their allowed types (None where the scoring writes null)
REQUIRED = {"blocks": list, "episodes_in_scope": int, "episode_results": list, "episodes_partial_excluded": list, "false_alarm_list": list, "per_split": dict}
# Required fields of an episode record; alarm_start and delay_s are null for a missed episode
EPISODE_FIELDS = {"episode": int, "detected": bool, "alarm_start": (str, type(None)), "delay_s": (int, float, type(None))}
# The modes required by default
DEFAULT_MODES = ["mag", "ang"]
# The two sources
SOURCES = ("model", "persistence")


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Read-only comparison of sensitivity score files against primary score files: event-block results must be unchanged")
    # Runs
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run folders holding the score files")
    # Pairs
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, metavar="SENS=PRIMARY", help="Sensitivity file to primary file, by name inside each run folder")
    # Event prefixes
    parser.add_argument("--event-prefixes", nargs="+", default=["test_events_"], help="Block-label prefixes of the event blocks")
    # Required modes
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="Modes that must be present, with model and persistence results, in both files")
    # Parse
    args = parser.parse_args()
    # Pairs as tuples
    pairs = []
    # Loop
    for item in args.pairs:
        # Split once
        if "=" not in item:
            # Reject
            parser.error(f"--pairs entries are SENS=PRIMARY, got {item}")
        # Keep
        pairs.append(tuple(item.split("=", 1)))
    # Store
    args.pairs = pairs
    # Return
    return args


def load(path):
    """Read one JSON file."""
    # Open
    with open(path) as f:
        # Parse
        return json.load(f)


def in_event_block(ts, labels, prefixes):
    """Whether the ISO timestamp lies inside one of the blocks whose label starts with an event prefix."""
    # Time
    t = pd.Timestamp(ts)
    # UTC
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    # Each event block
    for label in labels:
        # Event block
        if any(label.startswith(p) for p in prefixes):
            # Bounds
            b0, b1 = ss.label_bounds(label)
            # Inside
            if b0 <= t < b1:
                # Yes
                return True
    # No
    return False


def schema_findings(result, side, prefixes, is_sens):
    """Missing or malformed required fields of one mode/source result; side names the file for the message. Empty lists and the nullable values the scoring writes are allowed."""
    # Findings
    out = []
    # Not a record at all
    if not isinstance(result, dict) or not result:
        # Finding
        return [f"{side}: result record missing or empty"]
    # Required fields and types
    for f, typ in REQUIRED.items():
        # Missing
        if f not in result:
            # Finding
            out.append(f"{side}: required field {f} missing")
        # Wrong shape
        elif not isinstance(result[f], typ):
            # Finding
            out.append(f"{side}: field {f} has type {type(result[f]).__name__}, expected {typ.__name__}")
    # Blocks: label and every compared field present, nullable ones may be null
    for b in (result.get("blocks") if isinstance(result.get("blocks"), list) else []):
        # Shape
        if not isinstance(b, dict) or "label" not in b:
            # Finding
            out.append(f"{side}: a block record has no label")
            # Next
            continue
        # Fields
        for f in BLOCK_FIELDS:
            # Missing
            if f not in b:
                # Finding
                out.append(f"{side}: block {b['label']} lacks {f}")
            # Null where not allowed
            elif b[f] is None and f not in BLOCK_NULLABLE:
                # Finding
                out.append(f"{side}: block {b['label']} has null {f}")
    # At least one event block
    labels = [b.get("label", "") for b in (result.get("blocks") if isinstance(result.get("blocks"), list) else [])]
    # None
    if isinstance(result.get("blocks"), list) and not any(l.startswith(p) for l in labels for p in prefixes):
        # Finding
        out.append(f"{side}: no event block among {labels}")
    # Episode records
    for r in (result.get("episode_results") if isinstance(result.get("episode_results"), list) else []):
        # Shape
        if not isinstance(r, dict):
            # Finding
            out.append(f"{side}: an episode record is not a record")
            # Next
            continue
        # Fields
        for f, typ in EPISODE_FIELDS.items():
            # Missing
            if f not in r:
                # Finding
                out.append(f"{side}: episode record {r.get('episode', '?')} lacks {f}")
            # Wrong type
            elif not isinstance(r[f], typ):
                # Finding
                out.append(f"{side}: episode record {r.get('episode', '?')} has {f} of type {type(r[f]).__name__}")
    # Event-split summaries
    if isinstance(result.get("per_split"), dict):
        # Each prefix
        for p in prefixes:
            # Missing
            if p not in result["per_split"]:
                # Finding
                out.append(f"{side}: per_split lacks {p}")
    # Sensitivity side must carry the exclusion record with its dropped list
    if is_sens and not (isinstance(result.get("exclusions"), dict) and isinstance(result["exclusions"].get("episodes_dropped"), list)):
        # Finding
        out.append(f"{side}: exclusions.episodes_dropped missing")
    # Return
    return out


def compare_source(sens, primary, prefixes):
    """Findings for one mode and source: the required fields must be present on both sides, then everything that depends only on the event blocks must be equal."""
    # Schema first, both sides, so a field missing from both sides fails instead of matching
    out = schema_findings(sens, "sensitivity", prefixes, True) + schema_findings(primary, "primary", prefixes, False)
    # No value comparison on a malformed record
    if out:
        # Findings
        return out
    # Event-block records by label
    sb = {b["label"]: b for b in sens.get("blocks", []) if any(b["label"].startswith(p) for p in prefixes)}
    # Primary
    pb = {b["label"]: b for b in primary.get("blocks", []) if any(b["label"].startswith(p) for p in prefixes)}
    # Same labels
    if set(sb) != set(pb):
        # Finding
        out.append(f"event blocks {sorted(sb)} against primary {sorted(pb)}")
    # Field by field
    for label in sorted(set(sb) & set(pb)):
        # Each field
        for f in BLOCK_FIELDS:
            # Differ
            if sb[label].get(f, "<missing>") != pb[label].get(f, "<missing>"):
                # Finding
                out.append(f"block {label}: {f} {sb[label].get(f, '<missing>')} against primary {pb[label].get(f, '<missing>')}")
        # The added field must be zero on an event block
        if sb[label].get("excluded_slots", 0) != 0:
            # Finding
            out.append(f"block {label}: excluded_slots is {sb[label]['excluded_slots']}, expected 0")
        # No other added or removed fields
        extra = (set(sb[label]) - {"excluded_slots"}) ^ set(pb[label])
        # Differ
        if extra:
            # Finding
            out.append(f"block {label}: field set differs: {sorted(extra)}")
    # Episode scope
    if sens.get("episodes_in_scope") != primary.get("episodes_in_scope"):
        # Finding
        out.append(f"episodes_in_scope {sens.get('episodes_in_scope')} against primary {primary.get('episodes_in_scope')}")
    # Identities
    s_ids = [r["episode"] for r in sens.get("episode_results", [])]
    # Primary
    p_ids = [r["episode"] for r in primary.get("episode_results", [])]
    # Same
    if s_ids != p_ids:
        # Finding
        out.append(f"episode identities {s_ids} against primary {p_ids}")
    # Partly covered list
    if sens.get("episodes_partial_excluded") != primary.get("episodes_partial_excluded"):
        # Finding
        out.append("episodes_partial_excluded differs from the primary file")
    # Nothing dropped by the exclusion
    dropped = sens.get("exclusions", {}).get("episodes_dropped", [])
    # Must be empty
    if dropped:
        # Finding
        out.append(f"exclusions.episodes_dropped is not empty: {[d['episode'] for d in dropped]}")
    # Detections, per episode
    p_by = {r["episode"]: r for r in primary.get("episode_results", [])}
    # Each sensitivity episode
    for r in sens.get("episode_results", []):
        # Primary record
        q = p_by.get(r["episode"])
        # Missing
        if q is None:
            # Already reported by the identity check
            continue
        # Each field
        for f in ("detected", "alarm_start", "delay_s"):
            # Differ
            if r.get(f, "<missing>") != q.get(f, "<missing>"):
                # Finding
                out.append(f"episode {r['episode']}: {f} {r.get(f, '<missing>')} against primary {q.get(f, '<missing>')}")
    # Unmatched alarms starting inside an event block
    labels = [b["label"] for b in primary.get("blocks", [])]
    # Sensitivity
    s_fa = [tuple(x) for x in sens.get("false_alarm_list", []) if in_event_block(x[0], labels, prefixes)]
    # Primary
    p_fa = [tuple(x) for x in primary.get("false_alarm_list", []) if in_event_block(x[0], labels, prefixes)]
    # Same
    if s_fa != p_fa:
        # Finding
        out.append(f"event-block false alarms: {len(s_fa)} against primary {len(p_fa)}" + ("" if len(s_fa) != len(p_fa) else " (same count, different records)"))
    # Event-split summary
    for p in prefixes:
        # Records
        if sens.get("per_split", {}).get(p) != primary.get("per_split", {}).get(p):
            # Finding
            out.append(f"per_split[{p}] differs from the primary file")
    # Return
    return out


def compare_files(sens, primary, prefixes, modes=DEFAULT_MODES):
    """Findings per (mode, source) for one file pair: every required mode with both sources must be present and non-empty in both files; a mode present in both files beyond the required ones is compared too."""
    # Results
    out = {}
    # Mode tables, must be records
    s_modes = sens.get("modes") if isinstance(sens.get("modes"), dict) else {}
    # Primary
    p_modes = primary.get("modes") if isinstance(primary.get("modes"), dict) else {}
    # Required modes first, then extra modes present on both sides
    for mode in list(modes) + sorted((set(s_modes) & set(p_modes)) - set(modes)):
        # Sources
        for src in SOURCES:
            # Records
            r = s_modes.get(mode, {}).get(src) if isinstance(s_modes.get(mode), dict) else None
            # Primary
            p = p_modes.get(mode, {}).get(src) if isinstance(p_modes.get(mode), dict) else None
            # Missing or empty on either side
            missing = [name for name, rec in (("sensitivity", r), ("primary", p)) if not isinstance(rec, dict) or not rec]
            # Finding
            if missing:
                # Record
                out[(mode, src)] = [f"required mode/source {mode}/{src} missing or empty in the {' and '.join(missing)} file"]
                # Next
                continue
            # Compare
            out[(mode, src)] = compare_source(r, p, prefixes)
    # Return
    return out


def main():
    """Compare every pair of every run, print PASS and FAIL lines, exit 1 on any FAIL."""
    # Options
    args = parse_args()
    # Any failure
    failed = False
    # Counts of completed file pairs and mode/source comparisons
    n_pairs, n_cmp = 0, 0
    # Each run
    for run_dir in args.run_dirs:
        # Name
        print(f"\nrun {os.path.basename(os.path.normpath(run_dir))}")
        # Each pair
        for s_name, p_name in args.pairs:
            # Paths
            s_path, p_path = os.path.join(run_dir, s_name), os.path.join(run_dir, p_name)
            # Every selected file must exist on both sides
            absent = [n for n, path in ((s_name, s_path), (p_name, p_path)) if not os.path.isfile(path)]
            # A missing file is a failure
            if absent:
                # Say
                print(f"  FAIL  {s_name} against {p_name}: file(s) not present: {', '.join(absent)}")
                # Mark
                failed = True
                # Next
                continue
            # Load both
            sens, primary = load(s_path), load(p_path)
            # The sensitivity file must say it is one
            if not sens.get("settings", {}).get("exclude"):
                # Say
                print(f"  FAIL  {s_name}: no exclusion recorded in its settings, so it is not a sensitivity file")
                # Mark
                failed = True
                # Next
                continue
            # Compare
            findings = compare_files(sens, primary, args.event_prefixes, args.modes)
            # Count
            n_pairs += 1
            # Each mode and source
            for (mode, src), items in findings.items():
                # Count
                n_cmp += 1
                # Verdict
                if items:
                    # Mark
                    failed = True
                    # Header
                    print(f"  FAIL  {s_name} against {p_name}, {mode}/{src}: {len(items)} difference(s)")
                    # Each
                    for item in items:
                        # Print
                        print(f"        {item}")
                # Pass
                else:
                    # Line
                    print(f"  PASS  {s_name} against {p_name}, {mode}/{src}: event-block records, episode scope, detections and event-block false alarms identical")
    # Nothing compared is a failure
    if n_cmp == 0:
        # Say
        print("\nFAIL  no mode/source comparison was completed")
        # Mark
        failed = True
    # Counts
    print(f"\nfile pairs compared: {n_pairs}; mode/source comparisons completed: {n_cmp}")
    # Verdict
    print("all comparisons passed" if not failed else "some comparisons FAILED")
    # Exit
    sys.exit(1 if failed else 0)


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
