# =============================================================================
# File:        main_runcompare.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-10
#
# Purpose:     Read-only comparison of saved runs before some of them are
#              reused in a new matched comparison. We want to reuse the
#              existing original-input, true-edge runs at seeds 1 and 2
#              instead of training them again, but only if they were
#              trained and scored exactly like the seed-0 reference. A
#              folder name does not prove that, so this check reads the
#              evidence from each run's own files, prints it side by side,
#              and lists every difference. Nothing is inferred from names,
#              written, retrained or rescored.
#
#              Evidence read per run:
#                - the resolved folder path and which required files exist
#                  (train_log.txt, scale.json, checkpoint_best.pt) and which
#                  score files exist (score.json, score_grid.json,
#                  score_era.json, score_grid_era.json);
#                - the options line of train_log.txt: the experiment as the
#                  training script recorded it, printed in full, with the
#                  graph identity hash and every training block listed;
#                - the model parameter-count line of train_log.txt;
#                - the settings dictionary of every score file.
#
#              Required training keys (REQUIRED_KEYS) are the 29 keys the
#              training script logged when the original runs were made
#              (main_gcn.py at commit 8a1637b, 2026-11-07, and main_gnn.py
#              at 89a6711 and 7e19e44, 2026-11-07 and 11-08; the key set is
#              identical in all three): seed, model, window, optimizer and
#              stopping settings, graph identity and selection, edge
#              settings, masked-objective settings, splits file and training
#              blocks. A required key or file missing on any side, even on
#              both, blocks a pass. Values are validated, not only present:
#              train_blocks must be a list of exactly nine distinct strings
#              "start/end" with ISO 8601 bounds and start before end (null,
#              a string, an empty entry, a repeated block or an unparsable
#              bound all fail, on any side), and graph_id must equal the
#              expected identity (--expected-graph, by default
#              graph_meters_periods.json:39977b2b42a72afd) on every run,
#              which also catches three runs that agree with one another
#              on a different graph. Block lists are compared in order only
#              when both sides are valid. Later metadata keys (LATER_KEYS: physics,
#              n_features, features, physics_scale, added at ff633d1 on
#              2026-11-09; angdiff, angdiff_settings, added at dbb5450) are
#              reported separately: their absence from all runs is not a
#              mismatch, and their presence on one side only is flagged for
#              review. Absence does not establish five features: the input
#              configuration is labeled "consistent with the historical
#              script by key set" only when a run's key set equals the
#              29-key set exactly (those script revisions define five
#              features per channel and no optional inputs), otherwise
#              "not independently verified".
#
#              Sender gating (sender_gate, added 2026-11-12) is a later key
#              with a stated meaning when absent: a run saved before the
#              option existed did not use sender gating, so its absence is
#              read as False (ABSENT_MEANS), the same reading the training
#              script applies to old checkpoints. Runs are compared on that
#              effective value. A difference is a FAIL, since a gated and
#              an ungated run pass different messages and must not be
#              pooled as seeds of one model. The key present on one side
#              only with the same effective value is INFO, not REVIEW. This
#              check is about seeds of one configuration. The planned
#              controlled comparison between a gated and an ungated
#              configuration is a different thing and is not refused here.
#
#              The mask policy (mask_policy, added 2026-11-12) is handled
#              the same way. A run saved before the option existed used
#              the first masked trial's rule, so its absence is read as
#              "legacy". An ordinary run records "legacy" too, so the key
#              on one side only is INFO. Runs whose effective policies
#              differ hide and score different targets and are a FAIL as
#              seeds of one configuration.
#
#              Scoring evidence: main_score.py last changed in commit
#              81c4857 (authored 2026-11-05 21:55 -0400, which is 2026-11-06
#              01:55 UTC) and unchanged since; it writes a JSON
#              with top-level keys run_dir, settings and modes; settings has
#              the 22 keys in SCORE_REQUIRED. Each file type must carry its
#              statistic and scoring version (score.json channel 1,
#              score_grid.json grid 1, score_era.json channel 2,
#              score_grid_era.json grid 2). Settings are compared with the
#              reference run and with the Step 60 "Scoring settings per
#              file" values (STEP60). Date field: settings has no execution
#              date; scoring_version_date is a constant written by hand in
#              main_score.py ("2026-11-05" for version 1, "2026-11-06" for
#              version 2), introduced by that same commit, whose header
#              also states "Version 1 (fixed 2026-11-05)" and "Version 2
#              (2026-11-06)". The two dates are specification labels set
#              in the code, not clock readings: version 1's is the freeze
#              date the README records, version 2's is the label the
#              author gave the era-calibration revision in the commit that
#              added it. A difference in this field is therefore a
#              specification difference and is flagged for review. The
#              bootstrap settings n_boot and seed change only the confidence
#              intervals and are flagged for review when they differ. The
#              events path and the run_dir are provenance and are printed
#              as information only.
#
#              Findings are FAIL (blocks the pass), REVIEW (flagged, blocks
#              the pass until reviewed) or INFO (printed only). Verdicts are
#              given separately for training compatibility, input
#              provenance and scoring readiness. Exit code 0 only when there
#              is no FAIL and no REVIEW finding, else 1. Exit 1 says what
#              differs or is missing; a missing score file affects
#              comparison readiness and does not by itself call for
#              retraining.
#
#              --self-test builds throwaway fixtures in a temporary folder
#              and runs the regression cases listed in SELF_TEST_CASES
#              through the same code, printing one line per case.
#
# Usage (from the repo root, on Colab with Drive mounted):
#   python checks/main_runcompare.py
#   python checks/main_runcompare.py --candidates gcn_dyn_aw001_s1 --expected-seeds 0 1
#   python checks/main_runcompare.py --expected-graph graph_meters_periods.json:39977b2b42a72afd
#   python checks/main_runcompare.py --self-test        (anywhere, needs no data)
#
# Functions:
#   parse_args()                 command-line options with defaults
#   read_log(run_dir)            options dictionary, parameter count and reasons from train_log.txt
#   read_score(run_dir, name)    settings dictionary of one score file, with a reason when unusable
#   validate_blocks(blocks, n)   whether a train_blocks value is a valid list of n distinct intervals
#   gather(name, run_dir)        one run's evidence as a dictionary
#   show_run(run)                print one run's evidence
#   check_runs(runs, seeds, ...) all findings over the reference and the candidates
#   verdicts(runs, findings)     the three verdicts and the exit code
#   evaluate(...)                gather, show, check, print verdicts, return the exit code
#   make_fixture(...)            write one fake run folder for the self-test
#   self_test()                  run the regression cases, return the exit code
#   main()                       evaluate or self-test
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: parse the options line, which is a Python dictionary printed as text
import ast
# Standard library: score files
import json
# Standard library: the block interval bounds
from datetime import datetime
# Standard library: paths
import os
# Standard library: the parameter-count line
import re
# Standard library: exit code
import sys
# Standard library: the self-test folder
import tempfile

# The keys the training script logged when the original runs were made (see the header for the commits)
REQUIRED_KEYS = ["kind", "history", "horizon", "max_gap", "hidden", "embed", "layers", "dropout", "gnn_layers", "heads", "predict_delta", "edges", "edge_weights", "min_meters", "graph_id", "graph_selection", "masked", "maskable_meters", "mask_chunk", "lr", "weight_decay", "batch_size", "angle_weight", "epochs", "patience", "min_delta", "seed", "train_blocks", "splits_file"]
# Keys later script revisions added; reported separately, never required
LATER_KEYS = ["physics", "n_features", "features", "physics_scale", "angdiff", "angdiff_settings", "sender_gate", "mask_policy"]
# Later keys whose absence has a stated meaning: a run saved before sender gating existed did not use it,
# and a run saved before the mask policies existed used the first masked trial's rule, "legacy"
ABSENT_MEANS = {"sender_gate": False, "mask_policy": "legacy"}
# Files every complete run must have
REQUIRED_FILES = ["train_log.txt", "scale.json", "checkpoint_best.pt"]
# Score files with the statistic and scoring version each must carry
SCORE_FILES = {"score.json": ("channel", 1), "score_grid.json": ("grid", 1), "score_era.json": ("channel", 2), "score_grid_era.json": ("grid", 2)}
# The settings keys main_score.py writes
SCORE_REQUIRED = ["alpha", "calib_label", "min_meters", "merge_slots", "before_minutes", "after_minutes", "events", "test_prefixes", "common", "n_boot", "seed", "statistic", "design_label", "ref_percentile", "ref_min_count", "ref_floor_mag", "ref_floor_ang", "restrict_common", "era_calib", "scoring_version", "scoring_version_date", "scoring_note"]
# Settings that define the scoring specification; a difference is a FAIL, except the date, which is REVIEW
SCORE_SPEC = ["alpha", "calib_label", "min_meters", "merge_slots", "before_minutes", "after_minutes", "test_prefixes", "common", "statistic", "design_label", "ref_percentile", "ref_min_count", "ref_floor_mag", "ref_floor_ang", "restrict_common", "era_calib", "scoring_version", "scoring_note"]
# Settings flagged for review when they differ: the specification date and the bootstrap settings
SCORE_REVIEW = ["scoring_version_date", "n_boot", "seed"]
# Settings printed as information only
SCORE_INFO = ["events"]
# Step 60, "Scoring settings per file": the values every file must show, shared by all four
STEP60_COMMON = {"alpha": 0.0001, "min_meters": 2, "merge_slots": 6, "before_minutes": 1.0, "after_minutes": 10.0, "common": True, "restrict_common": False, "calib_label": "validation_2025-02-03_2025-02-16", "design_label": "validation_2024-01-08_2024-01-21", "ref_percentile": 99.0, "ref_min_count": 1000, "ref_floor_mag": 0.001, "ref_floor_ang": 1.0}
# Step 60: the era rule of the version 2 files, empty for version 1
STEP60_ERA = [["test_events_2023", "validation_2023-11-01_2023-11-14"]]
# The graph identity every run must carry (file name and content hash, as the training script records it)
EXPECTED_GRAPH = "graph_meters_periods.json:39977b2b42a72afd"
# The parameter-count line the training script logs
PARAM_LINE = re.compile(r"model \S+ with (\d+) parameters")
# Training option keys printed first, in this order; the rest follow alphabetically
SHOW_FIRST = ["seed", "angle_weight", "kind", "history", "horizon", "max_gap", "hidden", "embed", "layers", "dropout", "gnn_layers", "heads", "predict_delta", "edges", "edge_weights", "min_meters", "graph_id", "graph_selection", "masked", "maskable_meters", "mask_chunk", "lr", "weight_decay", "batch_size", "epochs", "patience", "min_delta", "splits_file"]


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Read-only comparison of saved runs against a seed-0 reference; exit 0 only when nothing fails and nothing is flagged for review")
    # Where the runs live
    parser.add_argument("--runs-dir", default="/content/drive/MyDrive/socal_runs", help="Folder that holds the run folders")
    # The reference run
    parser.add_argument("--reference", default="gcn_dyn_aw001", help="Run folder of the seed-0 reference")
    # The candidate runs
    parser.add_argument("--candidates", nargs="+", default=["gcn_dyn_aw001_s1", "gcn_dyn_aw001_s2"], help="Run folders to compare with the reference")
    # The seeds the runs must carry, reference first
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0, 1, 2], help="Saved seed required of the reference and then of each candidate, in order")
    # The graph identity
    parser.add_argument("--expected-graph", default=EXPECTED_GRAPH, help="graph_id every run must carry")
    # The number of training blocks
    parser.add_argument("--expected-blocks", type=int, default=9, help="Number of training blocks every run must list")
    # Self-test
    parser.add_argument("--self-test", action="store_true", help="Run the regression cases on throwaway fixtures instead of real runs")
    # Parse
    return parser.parse_args()


def read_log(run_dir):
    """The options dictionary and the parameter count from train_log.txt, each None with a reason when unreadable."""
    # Log path
    path = os.path.join(run_dir, "train_log.txt")
    # Missing log
    if not os.path.exists(path):
        # Both unavailable
        return None, "train_log.txt missing", None, "train_log.txt missing"
    # Read every line
    with open(path) as f:
        # Lines
        lines = f.read().splitlines()
    # The first line that carries the options
    hits = [ln for ln in lines if ", options {" in ln]
    # Default
    options, reason = None, "no options line in train_log.txt"
    # Found
    if hits:
        # The dictionary text starts at the brace after "options "
        text = hits[0][hits[0].index(", options {") + len(", options "):]
        # Parse the printed dictionary
        try:
            # Literal Python
            options, reason = ast.literal_eval(text), None
        # Unreadable
        except (ValueError, SyntaxError) as e:
            # Reason
            reason = f"options line could not be parsed: {e}"
    # The parameter-count line
    counts = [PARAM_LINE.search(ln) for ln in lines]
    # First match
    counts = [m for m in counts if m]
    # Count or reason
    params, preason = (int(counts[0].group(1)), None) if counts else (None, "no parameter-count line in train_log.txt")
    # Return
    return options, reason, params, preason


def read_score(run_dir, name):
    """The settings dictionary of one score file, or None with a reason when the file is missing or not in the scoring code's shape."""
    # Path
    path = os.path.join(run_dir, name)
    # Missing
    if not os.path.exists(path):
        # Reason
        return None, "file missing"
    # Load
    try:
        # JSON
        with open(path) as f:
            # Body
            body = json.load(f)
    # Unreadable
    except (ValueError, OSError) as e:
        # Reason
        return None, f"not readable as JSON: {e}"
    # Shape: a dictionary with run_dir, settings and modes
    if not isinstance(body, dict) or not {"run_dir", "settings", "modes"} <= set(body):
        # Reason
        return None, "not the scoring code's shape (run_dir, settings, modes)"
    # Settings must be a dictionary with content
    if not isinstance(body["settings"], dict) or not body["settings"]:
        # Reason
        return None, "settings missing or empty"
    # Settings
    return body["settings"], None


def validate_blocks(blocks, n_blocks):
    """Check a train_blocks value: a list of exactly n_blocks distinct strings "start/end" with ISO 8601 bounds and start before end. Return (ok, reason)."""
    # Must be a list
    if not isinstance(blocks, list):
        # Reason
        return False, f"train_blocks is {type(blocks).__name__}, not a list"
    # Must have the expected count
    if len(blocks) != n_blocks:
        # Reason
        return False, f"{len(blocks)} training blocks, expected {n_blocks}"
    # Each entry
    for i, b in enumerate(blocks):
        # A non-empty string
        if not isinstance(b, str) or not b.strip():
            # Reason
            return False, f"block {i + 1} is not a non-empty string: {b!r}"
        # Exactly one separator
        parts = b.split("/")
        # Two sides
        if len(parts) != 2:
            # Reason
            return False, f"block {i + 1} is not start/end: {b!r}"
        # Parse both bounds
        try:
            # ISO 8601
            start, end = datetime.fromisoformat(parts[0]), datetime.fromisoformat(parts[1])
        # Unparsable
        except ValueError:
            # Reason
            return False, f"block {i + 1} has a bound that is not a timestamp: {b!r}"
        # Both must carry a time zone or neither, else they cannot be compared
        if (start.tzinfo is None) != (end.tzinfo is None):
            # Reason
            return False, f"block {i + 1} mixes aware and naive bounds: {b!r}"
        # Start before end
        if not start < end:
            # Reason
            return False, f"block {i + 1} has start not before end: {b!r}"
    # Distinct
    if len(set(blocks)) != len(blocks):
        # Reason
        return False, f"repeated training block: {sorted(b for b in set(blocks) if blocks.count(b) > 1)}"
    # Valid
    return True, None


def gather(name, run_dir):
    """One run's evidence: files present, options, parameter count, score settings, with reasons for anything missing."""
    # Files
    present = {f: os.path.exists(os.path.join(run_dir, f)) for f in REQUIRED_FILES + list(SCORE_FILES)}
    # Log
    options, reason, params, preason = read_log(run_dir)
    # Scores
    scores = {f: read_score(run_dir, f) for f in SCORE_FILES}
    # Evidence
    return {"name": name, "dir": os.path.abspath(run_dir), "exists": os.path.isdir(run_dir), "present": present, "options": options, "reason": reason, "params": params, "preason": preason, "scores": scores}


def show_run(run):
    """Print one run's evidence in full."""
    # Header
    print("=" * 30, run["name"], "=" * 30)
    # Folder
    print(f"folder: {run['dir']}" + ("" if run["exists"] else "   (MISSING)"))
    # Files
    for f, ok in run["present"].items():
        # Line
        print(f"  {f:<22} {'present' if ok else 'MISSING'}")
    # Options
    o = run["options"]
    # Unreadable
    if o is None:
        # Say so
        print(f"training options: {run['reason']}")
    # Readable
    else:
        # Title
        print("training options (from train_log.txt):")
        # Required keys first in the display order, then any other key alphabetically, blocks last
        keys = [k for k in SHOW_FIRST if k in o] + sorted(k for k in o if k not in SHOW_FIRST and k not in LATER_KEYS and k != "train_blocks")
        # Print each
        for k in keys:
            # Line
            print(f"  {k:<18} {o[k]}")
        # Required keys absent
        for k in [k for k in REQUIRED_KEYS if k not in o]:
            # Line
            print(f"  {k:<18} <absent, required>")
        # Later metadata
        print("later metadata (not required):")
        # Each
        for k in LATER_KEYS:
            # Line, with the stated meaning of an absent key where there is one
            print(f"  {k:<18} {o.get(k, '<absent, read as ' + str(ABSENT_MEANS[k]) + '>' if k in ABSENT_MEANS else '<absent>')}")
        # Blocks
        blocks = o.get("train_blocks")
        # Title
        print(f"  train_blocks       {len(blocks) if isinstance(blocks, list) else '<absent, required>'}")
        # Each block in order
        for i, b in enumerate(blocks if isinstance(blocks, list) else []):
            # Line
            print(f"    {i + 1}. {b}")
    # Parameters
    print(f"model parameters: {run['params'] if run['params'] is not None else run['preason']}")
    # Scores
    for f, (s, reason) in run["scores"].items():
        # Unusable
        if s is None:
            # Say so
            print(f"scoring settings {f}: {reason}")
        # Usable
        else:
            # Specification keys
            print(f"scoring settings {f}: " + ", ".join(f"{k}={s.get(k, '<absent>')}" for k in SCORE_SPEC + SCORE_REVIEW))
            # Information keys
            print("  info: " + ", ".join(f"{k}={s.get(k, '<absent>')}" for k in SCORE_INFO))
    # Blank line
    print()


def check_runs(runs, seeds, n_blocks, graph):
    """Every finding over the reference (first) and the candidates, as (level, text) with level FAIL, REVIEW or INFO."""
    # Findings
    out = []
    # Shorthand
    ref = runs[0]
    # Per run: files, seeds, required keys, blocks, provenance, score files
    for run, seed in zip(runs, seeds):
        # Folder
        if not run["exists"]:
            # Fail
            out.append(("FAIL", f"{run['name']}: folder missing"))
        # Required files
        for f in REQUIRED_FILES:
            # Missing
            if not run["present"][f]:
                # Fail
                out.append(("FAIL", f"{run['name']}: required file {f} missing"))
        # Options
        o = run["options"]
        # Unreadable
        if o is None:
            # Fail
            out.append(("FAIL", f"{run['name']}: training options unavailable ({run['reason']})"))
            # Provenance
            out.append(("FAIL", f"{run['name']}: input configuration not independently verified (no options)"))
        # Readable
        else:
            # Required keys
            missing = [k for k in REQUIRED_KEYS if k not in o]
            # Any
            if missing:
                # Fail
                out.append(("FAIL", f"{run['name']}: required training keys missing: {missing}"))
            # Seed
            if "seed" in o and o["seed"] != seed:
                # Fail
                out.append(("FAIL", f"{run['name']}: saved seed {o['seed']}, expected {seed}"))
            # Blocks, validated whenever the key exists (a missing key is already a FAIL above)
            if "train_blocks" in o:
                # Validate
                ok, why = validate_blocks(o["train_blocks"], n_blocks)
                # Malformed
                if not ok:
                    # Fail
                    out.append(("FAIL", f"{run['name']}: train_blocks invalid: {why}"))
            # Graph identity against the expected one
            if "graph_id" in o and o["graph_id"] != graph:
                # Fail
                out.append(("FAIL", f"{run['name']}: graph_id {o['graph_id']!r}, expected {graph!r}"))
            # Provenance by key set
            extra = sorted(set(o) - set(REQUIRED_KEYS) - set(LATER_KEYS))
            # Exactly the historical set
            if not missing and not extra and not any(k in o for k in LATER_KEYS):
                # Consistent
                out.append(("INFO", f"{run['name']}: input configuration consistent with the historical script by key set (29 keys, five features per channel, no optional inputs); not a saved feature list"))
            # Later metadata present
            elif not missing and not extra:
                # Recorded
                out.append(("INFO", f"{run['name']}: input configuration recorded by a later script: " + ", ".join(f"{k}={o[k]}" for k in LATER_KEYS if k in o)))
            # Unknown keys
            else:
                # Review
                out.append(("REVIEW", f"{run['name']}: input configuration not independently verified (unexpected keys {extra} or required keys missing)"))
        # Parameter count
        if run["params"] is None:
            # Info
            out.append(("INFO", f"{run['name']}: {run['preason']}"))
        # Score files
        for f, (st, ver) in SCORE_FILES.items():
            # Settings
            s, reason = run["scores"][f]
            # Unusable
            if s is None:
                # Fail
                out.append(("FAIL", f"{run['name']}: {f} {reason}"))
                # Next
                continue
            # Required settings keys
            miss = [k for k in SCORE_REQUIRED if k not in s]
            # Any
            if miss:
                # Fail
                out.append(("FAIL", f"{run['name']}: {f} settings keys missing: {miss}"))
            # Statistic and version of the file type
            if s.get("statistic") != st or s.get("scoring_version") != ver:
                # Fail
                out.append(("FAIL", f"{run['name']}: {f} is statistic {s.get('statistic')} version {s.get('scoring_version')}, expected {st} version {ver}"))
            # Step 60 values
            for k, v in STEP60_COMMON.items():
                # Different
                if k in s and s[k] != v:
                    # Fail
                    out.append(("FAIL", f"{run['name']}: {f} {k}={s[k]}, Step 60 reference {v}"))
            # Era rule of the file type, compared as lists
            era = json.loads(json.dumps(s.get("era_calib", [])))
            # Expected
            want = STEP60_ERA if ver == 2 else []
            # Different
            if era != want:
                # Fail
                out.append(("FAIL", f"{run['name']}: {f} era_calib={era}, Step 60 reference {want}"))
    # Seeds identical across runs
    saved = [r["options"].get("seed") for r in runs if r["options"] is not None and "seed" in r["options"]]
    # Duplicates
    if len(saved) != len(set(saved)):
        # Fail
        out.append(("FAIL", f"saved seeds are not distinct: {saved}"))
    # Candidates against the reference
    for cand in runs[1:]:
        # Both readable
        if ref["options"] is not None and cand["options"] is not None:
            # Shorthand
            ro, co = ref["options"], cand["options"]
            # Required keys, seed and blocks handled separately
            for k in [k for k in REQUIRED_KEYS if k not in ("seed", "train_blocks")]:
                # Values
                rv, cv = ro.get(k, "<absent>"), co.get(k, "<absent>")
                # Different
                if rv != cv:
                    # Fail
                    out.append(("FAIL", f"{cand['name']}: option {k}: reference {rv}, candidate {cv}"))
            # Blocks in order, compared only when both sides are valid (an invalid side is already a FAIL)
            rb, cb = ro.get("train_blocks"), co.get("train_blocks")
            # Both valid
            if validate_blocks(rb, n_blocks)[0] and validate_blocks(cb, n_blocks)[0]:
                # Same
                if rb == cb:
                    # Info
                    out.append(("INFO", f"{cand['name']}: train_blocks identical to the reference, {len(rb)} blocks in the same order"))
                # Different
                else:
                    # Fail
                    out.append(("FAIL", f"{cand['name']}: train_blocks differ from the reference; only on one side: {sorted(set(rb) ^ set(cb))}; order differs: {sorted(rb) == sorted(cb)}"))
            # Later keys whose absence has a stated meaning are compared on their effective value
            for k, default in ABSENT_MEANS.items():
                # Effective values, the default standing in for an absent key
                rv, cv = ro.get(k, default), co.get(k, default)
                # Different: not the same message passing, so not seeds of one model
                if rv != cv:
                    # Fail
                    out.append(("FAIL", f"{cand['name']}: option {k}: reference {rv}, candidate {cv} (an absent key is read as {default})"))
                # Same value, key on one side only
                elif (k in ro) != (k in co):
                    # Info
                    out.append(("INFO", f"{cand['name']}: {k} recorded on {'reference' if k in ro else 'candidate'} only; absent is read as {default}, so both sides are {rv}"))
            # Later metadata on one side only
            for k in [k for k in LATER_KEYS if k not in ABSENT_MEANS]:
                # One side
                if (k in ro) != (k in co):
                    # Review
                    out.append(("REVIEW", f"{cand['name']}: later metadata {k} present on {'reference' if k in ro else 'candidate'} only ({ro.get(k, '<absent>')} / {co.get(k, '<absent>')})"))
                # Both sides, different
                elif k in ro and ro[k] != co[k]:
                    # Fail
                    out.append(("FAIL", f"{cand['name']}: later metadata {k}: reference {ro[k]}, candidate {co[k]}"))
            # Unexpected keys on one side only
            for k in sorted((set(ro) ^ set(co)) - set(REQUIRED_KEYS) - set(LATER_KEYS)):
                # Review
                out.append(("REVIEW", f"{cand['name']}: key {k} present on one side only"))
        # Parameter counts
        if ref["params"] is not None and cand["params"] is not None:
            # Same
            if ref["params"] == cand["params"]:
                # Info
                out.append(("INFO", f"{cand['name']}: model parameters {cand['params']}, same as the reference (supports the same model size; does not prove the same input width or feature meaning)"))
            # Different
            else:
                # Fail
                out.append(("FAIL", f"{cand['name']}: model parameters {cand['params']}, reference {ref['params']}"))
        # Score settings against the reference
        for f in SCORE_FILES:
            # Both usable
            rs, cs = ref["scores"][f][0], cand["scores"][f][0]
            # Skip when either is unusable (already a FAIL)
            if rs is None or cs is None:
                # Next
                continue
            # Specification keys
            for k in SCORE_SPEC:
                # Values as JSON so tuples and lists compare alike
                rv, cv = json.dumps(rs.get(k, "<absent>"), sort_keys=True), json.dumps(cs.get(k, "<absent>"), sort_keys=True)
                # Different
                if rv != cv:
                    # Fail
                    out.append(("FAIL", f"{cand['name']}: {f} {k}: reference {rv}, candidate {cv}"))
            # Review keys
            for k in SCORE_REVIEW:
                # Different
                if rs.get(k, "<absent>") != cs.get(k, "<absent>"):
                    # Review
                    out.append(("REVIEW", f"{cand['name']}: {f} {k}: reference {rs.get(k, '<absent>')}, candidate {cs.get(k, '<absent>')}" + (" (scoring specification date, see header)" if k == "scoring_version_date" else " (bootstrap setting)")))
            # Information keys
            for k in SCORE_INFO:
                # Different
                if rs.get(k) != cs.get(k):
                    # Info
                    out.append(("INFO", f"{cand['name']}: {f} {k}: reference {rs.get(k)}, candidate {cs.get(k)} (provenance only)"))
    # Findings
    return out


def verdicts(findings):
    """The three verdicts and the exit code from the findings."""
    # Texts by level
    fails = [t for lvl, t in findings if lvl == "FAIL"]
    # Reviews
    reviews = [t for lvl, t in findings if lvl == "REVIEW"]
    # Training: any FAIL that is not about a score file
    train_fail = [t for t in fails if not any(f in t for f in SCORE_FILES)]
    # Scoring: any FAIL about a score file
    score_fail = [t for t in fails if any(f in t for f in SCORE_FILES)]
    # Provenance: any finding about input configuration not independently verified
    prov = [t for lvl, t in findings if "not independently verified" in t]
    # Review items by side
    train_review = [t for t in reviews if not any(f in t for f in SCORE_FILES)]
    # Scoring review
    score_review = [t for t in reviews if any(f in t for f in SCORE_FILES)]
    # Training verdict
    training = "fail" if train_fail else ("review" if train_review else "pass")
    # Provenance verdict
    provenance = "not independently verified" if prov else "consistent with the historical script by key set, or recorded by a later script"
    # Scoring verdict
    scoring = "not ready" if score_fail else ("review" if score_review else "ready")
    # Exit
    code = 0 if not fails and not reviews else 1
    # Return
    return training, provenance, scoring, code


def evaluate(runs_dir, reference, candidates, seeds, n_blocks, graph, quiet=False):
    """Gather, show, check and judge the runs; return the exit code and the findings."""
    # Names
    names = [reference] + list(candidates)
    # Seeds must line up with the runs
    if len(seeds) != len(names):
        # Stop
        raise ValueError(f"--expected-seeds has {len(seeds)} values for {len(names)} runs")
    # Evidence
    runs = [gather(n, os.path.join(runs_dir, n)) for n in names]
    # Show
    if not quiet:
        # Each
        for r in runs:
            # Print
            show_run(r)
    # Findings
    findings = check_runs(runs, seeds, n_blocks, graph)
    # Verdicts
    training, provenance, scoring, code = verdicts(findings)
    # Print
    if not quiet:
        # Title
        print("-" * 20, "findings", "-" * 20)
        # Each, FAIL first
        for lvl in ("FAIL", "REVIEW", "INFO"):
            # Lines
            for l, t in findings:
                # Match
                if l == lvl:
                    # Print
                    print(f"{lvl:<7} {t}")
        # Verdicts
        print(f"\ntraining compatibility: {training}")
        # Provenance
        print(f"input provenance: {provenance}")
        # Scoring
        print(f"scoring readiness: {scoring}")
        # Exit
        print(f"exit code: {code}")
    # Return
    return code, findings


def make_fixture(folder, seed, options=None, params=36500, files=None, scores=None, setting_edits=None):
    """Write one fake run folder: a log with an options line and a parameter line, the required files, and the four score files."""
    # Folder
    os.makedirs(folder, exist_ok=True)
    # Default options: the 29 historical keys
    o = {"kind": "gcn", "history": 60, "horizon": 1, "max_gap": 12, "hidden": 64, "embed": 8, "layers": 1, "dropout": 0.0, "gnn_layers": 1, "heads": 0, "predict_delta": True, "edges": "graph", "edge_weights": "admittance", "min_meters": 1, "graph_id": "graph_meters_periods.json:39977b2b42a72afd", "graph_selection": "last_history_slot", "masked": False, "maskable_meters": None, "mask_chunk": 7, "lr": 0.001, "weight_decay": 0.0, "batch_size": 64, "angle_weight": 0.01, "epochs": 20, "patience": 3, "min_delta": 1e-05, "seed": seed, "train_blocks": list(FIXTURE_BLOCKS), "splits_file": "/content/repo/splits_v2.json"}
    # Edits
    if options is not None:
        # Apply, None removes
        for k, v in options.items():
            # Remove
            if v is None and k in o:
                # Drop
                del o[k]
            # The word "null" stands for a None value, since None itself means remove
            elif v == "null":
                # Null
                o[k] = None
            # Set
            elif v is not None:
                # Keep
                o[k] = v
    # Log text
    text = f"device cuda, run {folder}, options {o}\n" + (f"model gcn with {params} parameters\n" if params is not None else "") + "epoch 1: train 0.1 val 0.1\n"
    # Write
    with open(os.path.join(folder, "train_log.txt"), "w") as f:
        # Text
        f.write(text)
    # Other required files
    for name in (files if files is not None else ["scale.json", "checkpoint_best.pt"]):
        # Touch
        with open(os.path.join(folder, name), "w") as f:
            # Placeholder
            f.write("x")
    # Score files
    for name, (st, ver) in SCORE_FILES.items():
        # Skip when excluded
        if scores is not None and name not in scores:
            # Next
            continue
        # Settings as main_score.py writes them
        s = dict(STEP60_COMMON, events="/content/features/events.csv", test_prefixes=["test_events_", "test_normal_"], n_boot=200, seed=0, statistic=st, era_calib=(STEP60_ERA if ver == 2 else []), scoring_version=ver, scoring_version_date=("2026-11-06" if ver == 2 else "2026-11-05"), scoring_note="note")
        # Edits for this file; the word "empty" instead of a dictionary asks for empty settings
        edits = (setting_edits or {}).get(name, {})
        # Apply dictionary edits
        for k, v in (edits if isinstance(edits, dict) else {}).items():
            # Remove
            if v is None and k in s:
                # Drop
                del s[k]
            # Set
            elif v is not None:
                # Keep
                s[k] = v
        # Empty settings when asked
        body = {"run_dir": folder, "settings": ({} if edits == "empty" else s), "modes": {}}
        # Write
        with open(os.path.join(folder, name), "w") as f:
            # JSON
            json.dump(body, f)


# Nine valid, distinct training blocks for the fixtures, the real ones of splits_v2.json
FIXTURE_BLOCKS = ("2023-08-15T08:00:00+00:00/2023-09-15T08:00:00+00:00", "2023-09-20T08:00:00+00:00/2023-09-24T08:00:00+00:00", "2023-09-29T08:00:00+00:00/2023-11-01T08:00:00+00:00", "2023-11-15T08:00:00+00:00/2023-11-24T08:00:00+00:00", "2023-12-10T08:00:00+00:00/2024-01-08T08:00:00+00:00", "2024-01-22T08:00:00+00:00/2024-11-11T08:00:00+00:00", "2024-11-16T08:00:00+00:00/2025-02-03T08:00:00+00:00", "2025-02-17T08:00:00+00:00/2025-04-01T08:00:00+00:00", "2025-09-01T08:00:00+00:00/2025-09-04T08:00:00+00:00")
# Regression cases: name, expected exit code, text every case must produce, and the edits to the three fixtures (reference, s1, s2)
SELF_TEST_CASES = [
    ("complete match at seeds 0, 1, 2", 0, "training compatibility: pass", [{}, {}, {}]),
    ("wrong seed on candidate 2", 1, "saved seed 1, expected 2", [{}, {}, {"seed": 1}]),
    ("identical seeds", 1, "saved seeds are not distinct", [{}, {"seed": 0}, {}]),
    ("seed missing on both candidates", 1, "required training keys missing: ['seed']", [{}, {"options": {"seed": None}}, {"options": {"seed": None}}]),
    ("required key lr missing on all three", 1, "required training keys missing: ['lr']", [{"options": {"lr": None}}, {"options": {"lr": None}}, {"options": {"lr": None}}]),
    ("train_blocks missing on all three", 1, "required training keys missing: ['train_blocks']", [{"options": {"train_blocks": None}}] * 3),
    ("eight blocks on all three", 1, "train_blocks invalid: 8 training blocks, expected 9", [{"options": {"train_blocks": list(FIXTURE_BLOCKS[:8])}}] * 3),
    ("train_blocks null on the reference", 1, "gcn_dyn_aw001: train_blocks invalid: train_blocks is NoneType, not a list", [{"options": {"train_blocks": "null"}}, {}, {}]),
    ("train_blocks a string on candidate 1", 1, "gcn_dyn_aw001_s1: train_blocks invalid: train_blocks is str, not a list", [{}, {"options": {"train_blocks": "2023-08-15T08:00:00+00:00/2023-09-15T08:00:00+00:00"}}, {}]),
    ("train_blocks a string on all three", 1, "gcn_dyn_aw001_s2: train_blocks invalid: train_blocks is str, not a list", [{"options": {"train_blocks": "x"}}] * 3),
    ("nine empty entries on candidate 2", 1, "train_blocks invalid: block 1 is not a non-empty string", [{}, {}, {"options": {"train_blocks": [""] * 9}}]),
    ("a repeated block on all three", 1, "train_blocks invalid: repeated training block", [{"options": {"train_blocks": list(FIXTURE_BLOCKS[:8]) + [FIXTURE_BLOCKS[0]]}}] * 3),
    ("a block with start after end on candidate 1", 1, "train_blocks invalid: block 9 has start not before end", [{}, {"options": {"train_blocks": list(FIXTURE_BLOCKS[:8]) + ["2025-09-04T08:00:00+00:00/2025-09-01T08:00:00+00:00"]}}, {}]),
    ("a block with a bad bound on candidate 2", 1, "train_blocks invalid: block 3 has a bound that is not a timestamp", [{}, {}, {"options": {"train_blocks": list(FIXTURE_BLOCKS[:2]) + ["2024-01-22T08:00:00+00:00/soon"] + list(FIXTURE_BLOCKS[3:])}}]),
    ("empty graph identity on all three", 1, "graph_id '', expected 'graph_meters_periods.json:39977b2b42a72afd'", [{"options": {"graph_id": ""}}] * 3),
    ("same wrong graph hash on all three", 1, "graph_id 'graph_meters_periods.json:0000000000000000', expected", [{"options": {"graph_id": "graph_meters_periods.json:0000000000000000"}}] * 3),
    ("different graph hash on candidate 2 only", 1, "gcn_dyn_aw001_s2: graph_id 'graph_meters.json:abc', expected", [{}, {}, {"options": {"graph_id": "graph_meters.json:abc"}}]),
    ("scale.json missing on all three", 1, "required file scale.json missing", [{"files": ["checkpoint_best.pt"]}] * 3),
    ("score_grid_era.json missing on candidate 1", 1, "score_grid_era.json file missing", [{}, {"scores": ["score.json", "score_grid.json", "score_era.json"]}, {}]),
    ("empty settings in score.json on candidate 2", 1, "score.json settings missing or empty", [{}, {}, {"setting_edits": {"score.json": "empty"}}]),
    ("alpha missing from score_era.json on all three", 1, "score_era.json settings keys missing: ['alpha']", [{"setting_edits": {"score_era.json": {"alpha": None}}}] * 3),
    ("changed training setting history on candidate 1", 1, "option history: reference 60, candidate 120", [{}, {"options": {"history": 120}}, {}]),
    ("changed scoring alpha on candidate 2", 1, "score.json alpha=0.001, Step 60 reference 0.0001", [{}, {}, {"setting_edits": {"score.json": {"alpha": 0.001}}}]),
    ("scoring specification date differs on candidate 1", 1, "scoring_version_date: reference 2026-11-05, candidate 2026-11-04 (scoring specification date", [{}, {"setting_edits": {"score.json": {"scoring_version_date": "2026-11-04"}}}, {}]),
    ("events path differs on candidate 1", 0, "events: reference /content/features/events.csv, candidate /other/events.csv (provenance only)", [{}, {"setting_edits": {"score.json": {"events": "/other/events.csv"}}}, {}]),
    ("bootstrap n_boot differs on candidate 2", 1, "n_boot: reference 200, candidate 500 (bootstrap setting)", [{}, {}, {"setting_edits": {"score_grid.json": {"n_boot": 500}}}]),
    ("later metadata on reference only", 1, "later metadata physics present on reference only", [{"options": {"physics": "none", "n_features": 5, "features": ["mag", "cos", "sin", "mag_valid", "ang_valid"], "physics_scale": None}}, {}, {}]),
    ("later metadata absent on all three", 0, "consistent with the historical script by key set", [{}, {}, {}]),
    ("parameter count differs on candidate 2", 1, "model parameters 36600, reference 36500", [{}, {}, {"params": 36600}]),
    ("parameter line missing on candidate 1", 0, "no parameter-count line in train_log.txt", [{}, {"params": None}, {}]),
    ("unexpected key on candidate 1", 1, "key extra_key present on one side only", [{}, {"options": {"extra_key": 1}}, {}]),
    ("sender_gate False recorded on candidate 1 only", 0, "sender_gate recorded on candidate only; absent is read as False, so both sides are False", [{}, {"options": {"sender_gate": False}}, {}]),
    ("sender_gate True on candidate 2, absent on the reference", 1, "option sender_gate: reference False, candidate True", [{}, {}, {"options": {"sender_gate": True}}]),
    ("sender_gate True on the reference, absent on candidate 1", 1, "gcn_dyn_aw001_s1: option sender_gate: reference True, candidate False", [{"options": {"sender_gate": True}}, {}, {"options": {"sender_gate": True}}]),
    ("sender_gate True on all three", 0, "training compatibility: pass", [{"options": {"sender_gate": True}}] * 3),
    ("sender_gate False on all three", 0, "training compatibility: pass", [{"options": {"sender_gate": False}}] * 3),
    ("mask_policy legacy recorded on candidate 1 only", 0, "mask_policy recorded on candidate only; absent is read as legacy, so both sides are legacy", [{}, {"options": {"mask_policy": "legacy"}}, {}]),
    ("mask_policy legacy on all three", 0, "training compatibility: pass", [{"options": {"mask_policy": "legacy"}}] * 3),
    ("mask_policy available_neighbor on all three", 0, "training compatibility: pass", [{"options": {"mask_policy": "available_neighbor"}}] * 3),
    ("mask_policy available_neighbor on candidate 2, absent on the reference", 1, "option mask_policy: reference legacy, candidate available_neighbor", [{}, {}, {"options": {"mask_policy": "available_neighbor"}}]),
    ("mask_policy available_neighbor on the reference, legacy on candidate 1", 1, "gcn_dyn_aw001_s1: option mask_policy: reference available_neighbor, candidate legacy", [{"options": {"mask_policy": "available_neighbor"}}, {"options": {"mask_policy": "legacy"}}, {"options": {"mask_policy": "available_neighbor"}}]),
]


def self_test():
    """Run every regression case on throwaway fixtures and return 0 when all behave as expected."""
    # Failures
    bad = 0
    # Temporary folder
    with tempfile.TemporaryDirectory() as tmp:
        # Each case
        for i, (name, want_code, want_text, edits) in enumerate(SELF_TEST_CASES):
            # Case folder
            base = os.path.join(tmp, f"case{i}")
            # Three fixtures at seeds 0, 1, 2 with the case's edits
            for run, seed, e in zip(["gcn_dyn_aw001", "gcn_dyn_aw001_s1", "gcn_dyn_aw001_s2"], [0, 1, 2], edits):
                # Write
                make_fixture(os.path.join(base, run), e.get("seed", seed), e.get("options"), e.get("params", 36500), e.get("files"), e.get("scores"), e.get("setting_edits"))
            # Evaluate quietly
            code, findings = evaluate(base, "gcn_dyn_aw001", ["gcn_dyn_aw001_s1", "gcn_dyn_aw001_s2"], [0, 1, 2], 9, EXPECTED_GRAPH, quiet=True)
            # The verdict lines are not findings, so check them too
            texts = [t for _, t in findings] + [f"training compatibility: {verdicts(findings)[0]}"]
            # Expected text present
            seen = any(want_text in t for t in texts)
            # Outcome
            ok = code == want_code and seen
            # Count
            bad += 0 if ok else 1
            # Print
            print(f"{'PASS' if ok else 'FAIL'}  exit {code} (expected {want_code})  {name}" + ("" if seen else f"   [expected text not found: {want_text}]"))
    # Summary
    print(f"\nself-test: {len(SELF_TEST_CASES) - bad} of {len(SELF_TEST_CASES)} cases behaved as expected")
    # Code
    return 0 if bad == 0 else 1


def main():
    """Evaluate the real runs, or run the self-test, and exit with the code."""
    # Options
    args = parse_args()
    # Self-test
    if args.self_test:
        # Exit
        sys.exit(self_test())
    # Real runs
    code, _ = evaluate(args.runs_dir, args.reference, args.candidates, args.expected_seeds, args.expected_blocks, args.expected_graph)
    # Exit
    sys.exit(code)


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
