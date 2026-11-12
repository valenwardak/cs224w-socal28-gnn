# =============================================================================
# File:        main_exclusioncheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-11
#
# Purpose:     Regression check of the sensitivity exclusion added to the
#              scoring (socal_score.py settings["exclude"], main_score.py
#              --exclude and --primary) and of the read-only comparison
#              checks/main_senscompare.py. The scoring procedure is frozen,
#              so the first thing to show is that without an exclusion the
#              new code gives exactly the result the reference snapshot of
#              the code gave, for magnitude and angle, model and
#              persistence, both statistics and both scoring versions. The
#              rest shows that with an exclusion the rules hold on the
#              tested cases: thresholds, missing-threshold lists, references
#              and grid thresholds unchanged, under version 2 with a second,
#              different calibration block as well; excluded slots neither
#              monitored nor alarms, with exposure falling by the excluded
#              monitored slots and not by every excluded row; no alarm
#              episode bridging an exclusion, whether or not a recorded slot
#              lies inside it; an alarm in progress at the boundary ending
#              there; an event episode whose window touches the exclusion
#              leaving the scope and being listed; an exclusion touching a
#              calibration or design block refused; the main script refusing
#              every path that resolves to a primary file; the main script
#              refusing to write when the saved primary file is missing or
#              differs in a protected value; and the event-block and episode
#              preservation comparison passing where it should and failing
#              where it should, and that comparison refusing files with a
#              missing or empty required mode or source, a required field
#              missing from both sides, a selected file that does not exist,
#              and a run with no completed comparison; and the pre-write
#              check of main_score.py refusing, before any output is
#              created, an omitted mode, an empty source record and
#              calibration evidence missing from one or both sides, while
#              accepting the nulls and empty lists the scoring writes.
#              These are regression results for the tested cases, not a
#              proof for every input.
#
#              Everything runs on a synthetic run folder built here in a
#              temporary directory: three meters, one channel, two
#              calibration blocks with different noise levels, a design
#              block, two event blocks with one logged episode each, and a
#              quiet block whose slots lie around the two boundaries of the
#              exclusion plus a sparse stretch of two slots 30 seconds
#              apart, with planted alarms: one straddling the first
#              boundary, one entirely inside the exclusion, one just before
#              and one just after the second boundary within the merge
#              window of each other, one well outside, one on each sparse
#              slot, and a stretch of unmonitored rows inside the
#              exclusion. Residuals are the same for the model and
#              persistence columns so the common-slot masking is exercised
#              without changing the plant.
#
#              The reference snapshot is socal_score.py as committed at
#              --frozen-commit, read with git show and imported under
#              another name, so the equivalence test compares against the
#              code as committed. The check prints which commit last
#              changed that file up to the snapshot, as the evidence of
#              which scoring revision the snapshot contains.
#
# Usage (from the repo root):
#   python checks/main_exclusioncheck.py
#   python checks/main_exclusioncheck.py --frozen-commit 6f51419 --alpha 0.01
#
# Functions:
#   parse_args()                 command-line options with defaults
#   report(name, ok, note)       print one PASS or FAIL line, return ok
#   frozen_module(commit, root)  import socal_score.py as of a commit under another name
#   slots(start, n)              n ten-second UTC slots from start
#   write_table(run_dir, label, t, mag, ang, invalid)
#                                write one residual table and its summary file
#   build_fixture(root)          the synthetic run folder, events file, and the planted facts
#   settings_for(events, statistic, era, alpha, mode, exclude)
#                                a settings dict for score_source
#   same(a, b)                   deep equality with NaN equal to NaN
#   quiet_block(result)          the quiet block's record from a result
#   run_main(argv)               run main_score.main() quietly, return (exit code, printed text)
#   run_compare(argv)            run main_senscompare.main() quietly, return (exit code, printed text)
#   main()                       run every case, exit 0 when all pass
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: silence printed output where only the outcome matters
import contextlib
# Standard library: a sink for that output
import io
# Standard library: load the reference module from text
import importlib.util
# Standard library: JSON
import json
# Standard library: paths
import os
# Standard library: copies of files
import shutil
# Standard library: git show
import subprocess
# Standard library: exit code
import sys
# Standard library: the fixture folder
import tempfile
# Third party: numerics
import numpy as np
# Third party: tables and timestamps
import pandas as pd

# The repo root, one level above checks/, so the project modules import
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Import from the root
sys.path.insert(0, ROOT)
# And from checks/, for the comparison check
sys.path.insert(0, os.path.join(ROOT, "checks"))

# Project: the scoring module under test
import socal_score as ss
# Project: the main script, for its argument handling and its write path
import main_score as ms
# Project: the read-only comparison of sensitivity files against primary files
import main_senscompare as sc

# Meters and the one channel of the fixture
METERS = ["m1", "m2", "m3"]
# Channel
CHANNEL = "L1"
# Block labels: default calibration block, second calibration block, design block
CALIB = "validation_2025-02-03_2025-02-16"
# Second calibration block, for version 2
CALIB2 = "validation_2023-11-01_2023-11-14"
# Design
DESIGN = "validation_2024-01-08_2024-01-21"
# Event blocks
EVENTS = "test_events_2024-11-11_2024-11-15"
# The 2023 event block, calibrated on CALIB2 under version 2
EVENTS23 = "test_events_2023-09-18_2023-09-22"
# Quiet block
QUIET = "test_normal_2025-04-01_2025-08-31"
# The exclusion under test, [start, end) UTC: fixed PST July 29 00:00 to August 11 00:00, 2025
EXCL = ["2025-07-29T08:00:00+00:00", "2025-08-11T08:00:00+00:00"]
# A ten-second exclusion between two recorded slots 30 seconds apart, no recorded slot inside it
SPARSE = ["2025-08-20T10:00:10+00:00", "2025-08-20T10:00:20+00:00"]
# A short exclusion inside the quiet block that touches no episode, for the preservation comparison
SHORT = ["2025-07-29T08:00:00+00:00", "2025-07-29T09:00:00+00:00"]
# Quiet and alarm levels, magnitude (scaled) and angle (degrees)
MAG_QUIET, MAG_ALARM, ANG_QUIET, ANG_ALARM = 0.001, 1.0, 1.0, 90.0
# Rows inside the exclusion made unmonitored (every meter invalid): count
N_UNMONITORED = 100
# The four primary file names
PRIMARY = ("score.json", "score_grid.json", "score_era.json", "score_grid_era.json")


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Regression check of the scoring's sensitivity exclusion against the reference snapshot of the scoring")
    # The reference snapshot
    parser.add_argument("--frozen-commit", default="6f51419", help="Commit whose socal_score.py is the reference snapshot")
    # Alpha for the fixture, loose so a short calibration table has enough residuals
    parser.add_argument("--alpha", type=float, default=0.01, help="Conformal level for the fixture")
    # Parse
    return parser.parse_args()


def report(name, ok, note=""):
    """Print one PASS or FAIL line and return ok."""
    # Line
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + note if note else ''}")
    # Pass through
    return bool(ok)


def frozen_module(commit, root):
    """Import socal_score.py as it was at the commit, under the name socal_score_frozen; print the provenance evidence."""
    # Text from git
    text = subprocess.run(["git", "show", f"{commit}:socal_score.py"], cwd=root, capture_output=True, text=True, check=True).stdout
    # The last commit that changed the file up to the snapshot
    last = subprocess.run(["git", "log", "-n", "1", "--format=%h %ad %s", "--date=short", commit, "--", "socal_score.py"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    # Say
    print(f"reference snapshot: socal_score.py at commit {commit}; last commit changing that file up to the snapshot: {last}")
    # Temporary file
    path = os.path.join(tempfile.mkdtemp(), "socal_score_frozen.py")
    # Write
    with open(path, "w") as f:
        # Text
        f.write(text)
    # Import
    spec = importlib.util.spec_from_file_location("socal_score_frozen", path)
    # Module
    mod = importlib.util.module_from_spec(spec)
    # Load
    spec.loader.exec_module(mod)
    # Done
    return mod


def slots(start, n):
    """n ten-second UTC slots starting at start."""
    # Range
    return pd.date_range(pd.Timestamp(start), periods=n, freq=f"{ss.SLOT_SECONDS}s", tz="UTC")


def write_table(run_dir, label, t, mag, ang, invalid=None):
    """Write one residual table (model and persistence columns alike) and the summary file the held-out check reads; invalid marks rows where every meter has no valid residual."""
    # Columns
    cols = {}
    # Validity per row
    valid = np.ones(len(t), dtype=bool) if invalid is None else ~np.asarray(invalid, dtype=bool)
    # Loop over meters
    for mi, m in enumerate(METERS):
        # Both sources
        for prefix in ("", "pers_"):
            # Residuals
            cols[f"{m}|{CHANNEL}|{prefix}mag_res"] = mag[:, mi].astype(np.float32)
            # Angle residuals
            cols[f"{m}|{CHANNEL}|{prefix}ang_res"] = ang[:, mi].astype(np.float32)
            # Validity
            cols[f"{m}|{CHANNEL}|{prefix}mag_valid"] = valid.copy()
            # Angle validity
            cols[f"{m}|{CHANNEL}|{prefix}ang_valid"] = valid.copy()
    # Table
    pd.DataFrame(cols, index=t).to_parquet(os.path.join(run_dir, f"{label}.parquet"))
    # Summary with training blocks far from every block here
    with open(os.path.join(run_dir, f"{label}.json"), "w") as f:
        # Minimal options
        json.dump({"options": {"train_blocks": ["2023-01-15T08:00:00+00:00/2023-02-15T08:00:00+00:00"], "splits_file": "fixture"}}, f)


def build_fixture(root):
    """Build the synthetic run folder and events file; return their paths and the planted facts."""
    # Run folder
    run_dir = os.path.join(root, "run")
    # Make
    os.makedirs(run_dir)
    # Random quiet residuals
    rng = np.random.default_rng(0)
    # Quiet noise of a block: magnitude and angle, with a level factor
    noise = lambda n, k=1.0: (rng.uniform(0, MAG_QUIET * k, (n, 3)), rng.uniform(0, ANG_QUIET * k, (n, 3)))
    # Default calibration block: 2,000 slots of quiet noise
    n_cal = 2000
    # Write
    write_table(run_dir, CALIB, slots("2025-02-03T08:00:00Z", n_cal), *noise(n_cal))
    # Second calibration block: noise twice as large, so its thresholds differ
    write_table(run_dir, CALIB2, slots("2023-11-01T08:00:00Z", n_cal), *noise(n_cal, 2.0))
    # Design block
    write_table(run_dir, DESIGN, slots("2024-01-08T08:00:00Z", n_cal), *noise(n_cal))
    # Event block 2024: 720 slots with one planted detection at slot 300 on two meters
    t_ev = slots("2024-11-13T08:00:00Z", 720)
    # Residuals
    m_ev, a_ev = noise(720)
    # Plant
    m_ev[300:303, :2] = MAG_ALARM; a_ev[300:303, :2] = ANG_ALARM
    # Write
    write_table(run_dir, EVENTS, t_ev, m_ev, a_ev)
    # Event block 2023: the same with the plant at slot 400
    t_ev23 = slots("2023-09-20T08:00:00Z", 720)
    # Residuals
    m_23, a_23 = noise(720)
    # Plant
    m_23[400:403, :2] = MAG_ALARM; a_23[400:403, :2] = ANG_ALARM
    # Write
    write_table(run_dir, EVENTS23, t_ev23, m_23, a_23)
    # Quiet block: two stretches of 720 slots around each exclusion boundary (07:00 to 09:00 UTC), then two sparse slots 30 s apart
    t_q = slots("2025-07-29T07:00:00Z", 720).append(slots("2025-08-11T07:00:00Z", 720)).append(pd.DatetimeIndex([pd.Timestamp(SPARSE[0]) - pd.Timedelta(seconds=10), pd.Timestamp(SPARSE[1]) + pd.Timedelta(seconds=10)]))
    # Residuals
    m_q, a_q = noise(len(t_q))
    # Index of a time
    at = lambda s: int(np.flatnonzero(t_q == pd.Timestamp(s))[0])
    # Plant on two meters over an index range, both residual kinds
    def plant(i0, i1):
        # Magnitude
        m_q[i0:i1 + 1, :2] = MAG_ALARM
        # Angle
        a_q[i0:i1 + 1, :2] = ANG_ALARM
    # Plant A: an alarm straddling the first boundary, 07:59:00 to 08:01:00
    plant(at("2025-07-29T07:59:00Z"), at("2025-07-29T08:01:00Z"))
    # Plant B: an alarm entirely inside the exclusion, 08:30:00 to 08:31:00
    plant(at("2025-07-29T08:30:00Z"), at("2025-07-29T08:31:00Z"))
    # Plant C: alarms just before and just after the second boundary, 07:59:50 and 08:00:10, two slots apart
    c0, c1 = at("2025-08-11T07:59:50Z"), at("2025-08-11T08:00:10Z")
    # Set both
    plant(c0, c0); plant(c1, c1)
    # Plant D: an alarm well outside, 08:40:00 to 08:41:00 on August 11
    d0 = at("2025-08-11T08:40:00Z")
    # Set
    plant(d0, at("2025-08-11T08:41:00Z"))
    # Plant E: the two sparse slots
    e0, e1 = len(t_q) - 2, len(t_q) - 1
    # Set both
    plant(e0, e0); plant(e1, e1)
    # Unmonitored rows inside the exclusion: 08:10:00 onward on July 29, every meter invalid
    invalid = np.zeros(len(t_q), dtype=bool)
    # Start
    u0 = at("2025-07-29T08:10:00Z")
    # Mark
    invalid[u0:u0 + N_UNMONITORED] = True
    # Write
    write_table(run_dir, QUIET, t_q, m_q, a_q, invalid)
    # Events: episode 1 in the 2024 event block 20 s before its plant, episode 2 inside the exclusion, episode 3 in the 2023 block 20 s before its plant
    ev = pd.DataFrame({"t_utc": [t_ev[298].isoformat(), "2025-08-01T12:00:00+00:00", t_ev23[398].isoformat()], "scored": [True, True, True], "episode": [1, 2, 3], "device": ["sw_a", "sw_b", "sw_c"]})
    # Path
    events = os.path.join(root, "events.csv")
    # Write
    ev.to_csv(events, index=False)
    # Facts for the assertions
    facts = {"excluded_rows_quiet": int(((t_q >= pd.Timestamp(EXCL[0])) & (t_q < pd.Timestamp(EXCL[1]))).sum()), "t_q": t_q,
             "a_end_kept": t_q[at("2025-07-29T07:59:50Z")], "c_before": t_q[c0], "c_after": t_q[c1], "d_start": t_q[d0], "e_first": t_q[e0], "e_second": t_q[e1]}
    # Done
    return run_dir, events, facts


def settings_for(events, statistic, era, alpha, mode="mag", exclude=None):
    """A settings dict as main_score.py builds it, for one statistic, version and mode, with the given exclusion intervals if any."""
    # Base
    s = {"alpha": alpha, "calib_label": CALIB, "min_meters": 2, "merge_slots": 6, "before_minutes": 1.0, "after_minutes": 10.0,
         "events": events, "test_prefixes": ["test_events_", "test_normal_"], "common": True, "n_boot": 50, "seed": 0,
         "statistic": statistic, "design_label": DESIGN, "ref_percentile": 99.0, "ref_min_count": 100, "ref_floor_mag": 0.001, "ref_floor_ang": 1.0,
         "restrict_common": False, "era_calib": ([["test_events_2023", CALIB2]] if era else []), "mode": mode}
    # Exclusion
    if exclude:
        # Add
        s["exclude"] = [list(x) for x in exclude]
    # Done
    return s


def same(a, b):
    """Deep equality of two results, with NaN equal to NaN and timestamps compared as strings."""
    # Dictionaries
    if isinstance(a, dict) and isinstance(b, dict):
        # Keys and values
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    # Sequences
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        # Length and items
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    # Floats
    if isinstance(a, float) and isinstance(b, float):
        # NaN equal to NaN
        return (np.isnan(a) and np.isnan(b)) or a == b
    # Timestamps and the rest
    return str(a) == str(b)


def quiet_block(result):
    """The quiet block's record from a result."""
    # Find
    return [b for b in result["blocks"] if b["label"] == QUIET][0]


def run_main(argv):
    """Run main_score.main() with the given arguments, output captured; return (exit code, printed text)."""
    # Keep
    saved = sys.argv
    # Set
    sys.argv = ["main_score.py"] + argv
    # Sink
    sink = io.StringIO()
    # Run
    try:
        # Quietly
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            # Go
            ms.main()
        # Clean exit
        code = 0
    # Exits
    except SystemExit as e:
        # argparse gives 2, the primary check gives a message (exit 1)
        code = e.code if isinstance(e.code, int) else 1
        # The message is part of the printed text
        if not isinstance(e.code, int):
            # Add
            sink.write(str(e.code))
    # Restore
    finally:
        # Back
        sys.argv = saved
    # Done
    return code, sink.getvalue()


def run_compare(argv):
    """Run main_senscompare.main() with the given arguments, output captured; return (exit code, printed text)."""
    # Keep
    saved = sys.argv
    # Set
    sys.argv = ["main_senscompare.py"] + argv
    # Sink
    sink = io.StringIO()
    # Run
    try:
        # Quietly
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            # Go
            sc.main()
        # Clean
        code = 0
    # Exit
    except SystemExit as e:
        # Code
        code = e.code if isinstance(e.code, int) else 1
    # Restore
    finally:
        # Back
        sys.argv = saved
    # Done
    return code, sink.getvalue()


def main():
    """Run every case and exit 0 when all pass."""
    # Options
    args = parse_args()
    # Reference snapshot
    frozen = frozen_module(args.frozen_commit, ROOT)
    # Fixture
    root = tempfile.mkdtemp()
    # Build
    run_dir, events, facts = build_fixture(root)
    # All pass so far
    ok = True
    # Fixed arguments
    alpha = args.alpha
    # Case 1: equivalence without an exclusion, both statistics, both versions, both modes, both sources
    for statistic in ("channel", "grid"):
        # Versions
        for era in (False, True):
            # Modes
            for mode in ("mag", "ang"):
                # Settings without exclusion
                s = settings_for(events, statistic, era, alpha, mode)
                # Sources
                for prefix, other in (("", "pers_"), ("pers_", "")):
                    # New
                    new = ss.score_source(run_dir, prefix, s, other_prefix=other)
                    # Reference
                    old = frozen.score_source(run_dir, prefix, dict(s), other_prefix=other)
                    # Compare
                    ok &= report(f"no exclusion, {statistic}, version {2 if era else 1}, {mode}, {prefix or 'model'}: result identical to the reference snapshot", same(new, old))
    # Case 2: the fixed exclusion, channel statistic, version 1, model, both modes
    for mode in ("mag", "ang"):
        # Full period
        base = ss.score_source(run_dir, "", settings_for(events, "channel", False, alpha, mode), other_prefix="pers_")
        # With exclusion
        sens = ss.score_source(run_dir, "", settings_for(events, "channel", False, alpha, mode, [EXCL]), other_prefix="pers_")
        # Thresholds and missing lists unchanged
        ok &= report(f"exclusion, {mode}: per-channel thresholds, missing-threshold lists and calibration records identical to the full-period scoring", same(sens["thresholds"], base["thresholds"]) and same(sens["calibrations"], base["calibrations"]) and same(sens["calibration"], base["calibration"]))
        # Quiet block records
        bq, sq = quiet_block(base), quiet_block(sens)
        # Excluded rows recorded
        ok &= report(f"exclusion, {mode}: excluded slot count on the quiet block equals the recorded rows inside the interval", sq.get("excluded_slots") == facts["excluded_rows_quiet"], f"{sq.get('excluded_slots')} of {facts['excluded_rows_quiet']}")
        # Monitored slots fall by the excluded monitored rows only: the unmonitored rows inside the exclusion were never monitored
        ok &= report(f"exclusion, {mode}: monitored slots fall by the excluded monitored slots ({facts['excluded_rows_quiet'] - N_UNMONITORED}), not by every excluded row ({facts['excluded_rows_quiet']})", bq["monitored_slots"] - sq["monitored_slots"] == facts["excluded_rows_quiet"] - N_UNMONITORED, f"{bq['monitored_slots']} to {sq['monitored_slots']}")
        # Event blocks untouched
        for label in (EVENTS, EVENTS23):
            # Records
            be, se = [b for b in base["blocks"] if b["label"] == label][0], [b for b in sens["blocks"] if b["label"] == label][0]
            # Same apart from the added zero
            ok &= report(f"exclusion, {mode}: event block {label} unchanged", all(be[k] == se[k] for k in be) and se.get("excluded_slots") == 0)
        # Alarm list of the sensitivity run, as timestamps
        fa = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in sens["false_alarm_list"]]
        # No alarm starts or ends inside the exclusion
        inside = [x for x in fa if (pd.Timestamp(EXCL[0]) <= x[0] < pd.Timestamp(EXCL[1])) or (pd.Timestamp(EXCL[0]) <= x[1] < pd.Timestamp(EXCL[1]))]
        # Check
        ok &= report(f"exclusion, {mode}: no alarm episode starts or ends inside the excluded interval", not inside, f"{len(inside)} did")
        # No alarm bridges the exclusion
        ok &= report(f"exclusion, {mode}: no alarm episode bridges the exclusion", not [x for x in fa if x[0] < pd.Timestamp(EXCL[0]) and x[1] >= pd.Timestamp(EXCL[1])])
        # Plant A ends at the last kept slot before the boundary
        a_eps = [x for x in fa if x[0] == pd.Timestamp("2025-07-29T07:59:00Z")]
        # Check
        ok &= report(f"exclusion, {mode}: the alarm straddling the first boundary ends at the last kept slot before it", len(a_eps) == 1 and a_eps[0][1] == facts["a_end_kept"], f"{a_eps}")
        # Plant C: the slot after the second boundary starts its own episode and the one before is excluded
        ok &= report(f"exclusion, {mode}: the alarm just after the second boundary starts fresh and the one just before it is excluded", len([x for x in fa if x[0] == facts["c_after"]]) == 1 and not any(x[0] == facts["c_before"] for x in fa))
        # In the full-period scoring the same two slots merge into one episode
        fa_base = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in base["false_alarm_list"]]
        # Check
        ok &= report(f"full period, {mode}: the two slots around the second boundary merge into one episode", any(x[0] == facts["c_before"] and x[1] == facts["c_after"] for x in fa_base))
        # Plant B gone, plant D kept
        ok &= report(f"exclusion, {mode}: the alarm inside the interval is gone and the one outside is kept", not any(x[0] == pd.Timestamp("2025-07-29T08:30:00Z") for x in fa) and any(x[0] == facts["d_start"] for x in fa))
        # Episode 2 dropped from scope and listed; episodes 1 and 3 still scored
        ok &= report(f"exclusion, {mode}: the episode inside the interval leaves the scope and is listed", sens["episodes_in_scope"] == 2 and [d["episode"] for d in sens["exclusions"]["episodes_dropped"]] == [2] and base["episodes_in_scope"] == 3)
        # Detections unchanged
        ok &= report(f"exclusion, {mode}: the event-block detections are unchanged", sens["episodes_detected"] == base["episodes_detected"] == 2 and [(r["detected"], str(r["alarm_start"]), r["delay_s"]) for r in sens["episode_results"]] == [(r["detected"], str(r["alarm_start"]), r["delay_s"]) for r in base["episode_results"] if r["episode"] != 2])
        # Record present and the per-split exposure recomputed
        ok &= report(f"exclusion, {mode}: the record carries the interval and the quiet-block monitored days shrank", sens["exclusions"]["intervals"] == [list(EXCL)] and sens["per_split"]["test_normal_"]["monitored_days"] < base["per_split"]["test_normal_"]["monitored_days"])
    # Case 3: grid statistic under the exclusion: references and grid threshold unchanged, exposure by monitored rows
    bg = ss.score_source(run_dir, "", settings_for(events, "grid", False, alpha), other_prefix="pers_")
    # With exclusion
    sg = ss.score_source(run_dir, "", settings_for(events, "grid", False, alpha, exclude=[EXCL]), other_prefix="pers_")
    # Check
    ok &= report("exclusion, grid: references and grid threshold identical to the full-period scoring", same(sg["grid"]["ref_mag"], bg["grid"]["ref_mag"]) and same(sg["grid"]["ref_ang"], bg["grid"]["ref_ang"]) and sg["grid"]["threshold"] == bg["grid"]["threshold"] and same(sg["thresholds"], bg["thresholds"]))
    # Monitored slots
    ok &= report("exclusion, grid: monitored slots fall by the excluded monitored slots", quiet_block(bg)["monitored_slots"] - quiet_block(sg)["monitored_slots"] == facts["excluded_rows_quiet"] - N_UNMONITORED)
    # Case 4: version 2 with a second, different calibration block
    b2 = ss.score_source(run_dir, "", settings_for(events, "channel", True, alpha), other_prefix="pers_")
    # With exclusion
    s2 = ss.score_source(run_dir, "", settings_for(events, "channel", True, alpha, exclude=[EXCL]), other_prefix="pers_")
    # Two blocks in use, with different thresholds
    ok &= report("version 2 fixture: two calibration blocks in use with different thresholds", set(b2["calibrations"]) == {CALIB, CALIB2} and b2["calibrations"][CALIB]["thresholds"] != b2["calibrations"][CALIB2]["thresholds"] and b2["block_calibration"][EVENTS23] == CALIB2)
    # Preserved under the exclusion
    ok &= report("exclusion, version 2: both calibration records and the block assignments identical to the full-period scoring", same(s2["calibrations"], b2["calibrations"]) and s2["block_calibration"] == b2["block_calibration"] and [b["calib_label"] for b in s2["blocks"]] == [b["calib_label"] for b in b2["blocks"]])
    # Case 5: an exclusion containing no recorded slot still separates the slots on either side
    segs = ss.kept_segments(facts["t_q"].to_numpy(), [SPARSE])
    # The two sparse slots are the last two rows and must be in different stretches
    ok &= report("sparse exclusion: kept stretches split at an exclusion boundary with no recorded slot inside it", len(segs) == 2 and segs[0][1] == len(facts["t_q"]) - 1 and segs[1] == (len(facts["t_q"]) - 1, len(facts["t_q"])), f"{segs[-2:]}")
    # Through the scoring
    b5 = ss.score_source(run_dir, "", settings_for(events, "channel", False, alpha), other_prefix="pers_")
    # With the sparse exclusion only
    s5 = ss.score_source(run_dir, "", settings_for(events, "channel", False, alpha, exclude=[SPARSE]), other_prefix="pers_")
    # Alarm lists
    fb = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in b5["false_alarm_list"]]
    # Sensitivity
    fs = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in s5["false_alarm_list"]]
    # Full period merges the two sparse alarms; the exclusion keeps them apart with no slot excluded
    ok &= report("sparse exclusion: the two alarms 30 s apart merge in the full period and stay separate across the exclusion", any(x == (facts["e_first"], facts["e_second"]) for x in fb) and any(x == (facts["e_first"], facts["e_first"]) for x in fs) and any(x == (facts["e_second"], facts["e_second"]) for x in fs) and s5["exclusions"]["excluded_slots"] == 0)
    # Case 6: refusals in the module
    for name, bad, what in (("an exclusion overlapping the calibration block", [["2025-02-05T00:00:00+00:00", "2025-02-06T00:00:00+00:00"]], "channel"),
                            ("an exclusion overlapping the design block under the grid statistic", [["2024-01-10T00:00:00+00:00", "2024-01-11T00:00:00+00:00"]], "grid")):
        # Run
        try:
            # Score
            ss.score_source(run_dir, "", settings_for(events, what, False, alpha, exclude=bad), other_prefix="pers_")
            # Should not get here
            ok &= report(f"refusal: {name} is refused", False)
        # Refused
        except ValueError as e:
            # Good
            ok &= report(f"refusal: {name} is refused", "overlaps calibration or design block" in str(e))
    # Reversed interval
    try:
        # Reversed
        ss.exclusion_mask(facts["t_q"].to_numpy(), [[EXCL[1], EXCL[0]]])
        # Should not get here
        ok &= report("refusal: a reversed exclusion interval is refused", False)
    # Refused
    except ValueError:
        # Good
        ok &= report("refusal: a reversed exclusion interval is refused", True)
    # Case 7: the main script's output-path protection by resolved path
    base_argv = ["--run-dir", run_dir, "--events", events, "--exclude", "2025-07-29T00:00", "2025-08-11T00:00"]
    # Refused spellings: plain names, relative spellings, absolute paths
    refused = list(PRIMARY) + ["./score.json", os.path.join(".", "sub", "..", "score_grid.json"), os.path.join(run_dir, "score.json"), os.path.abspath(os.path.join(run_dir, "..", "run", "score_era.json"))]
    # Each
    for out in refused:
        # Expect a parser error (exit 2)
        code, _ = run_main(base_argv + ["--out", out])
        # Check
        ok &= report(f"main script: --exclude with --out {out} is refused", code == 2)
    # Allowed spellings are not refused by the path rule (they fail later only because no primary file exists yet)
    for out in ("score_sens.json", os.path.join(root, "elsewhere_score.json")):
        # Run
        code, text = run_main(base_argv + ["--out", out])
        # Check: not an argument error, and the refusal names the missing primary file
        ok &= report(f"main script: --exclude with --out {out} passes the path rule", code != 2 and "primary file" in text and "not found" in text)
    # Naive times to fixed PST
    saved = sys.argv
    # Arguments
    sys.argv = ["main_score.py"] + base_argv + ["--out", "score_sens.json"]
    # Parse
    a = ms.parse_args()
    # Restore
    sys.argv = saved
    # Check
    ok &= report("main script: naive --exclude times are read as fixed PST and recorded in UTC", a.exclude == [list(EXCL)], f"{a.exclude}")
    # --primary without --exclude
    code, _ = run_main(["--run-dir", run_dir, "--events", events, "--primary", "x.json"])
    # Check
    ok &= report("main script: --primary without --exclude is refused", code == 2)
    # Case 8: the main script's primary check, end to end on the fixture
    common = ["--run-dir", run_dir, "--events", events, "--alpha", str(alpha), "--ref-min-count", "100", "--n-boot", "20", "--modes", "mag", "ang"]
    # Nothing written yet
    code, text = run_main(common + ["--exclude", "2025-07-29T00:00", "2025-08-11T00:00", "--out", "score_sens.json"])
    # Refused for a missing primary, nothing written
    ok &= report("main script: with no saved primary file the sensitivity result is refused and not written", code != 0 and "not found" in text and not os.path.exists(os.path.join(run_dir, "score_sens.json")))
    # Write the primary files, version 1, channel and grid
    code1, _ = run_main(common)
    # Grid
    code2, _ = run_main(common + ["--statistic", "grid", "--out", "score_grid.json"])
    # Check
    ok &= report("main script: primary files score.json and score_grid.json written on the fixture", code1 == 0 and code2 == 0 and all(os.path.isfile(os.path.join(run_dir, n)) for n in ("score.json", "score_grid.json")))
    # Sensitivity files against them
    code1, text1 = run_main(common + ["--exclude", "2025-07-29T00:00", "2025-08-11T00:00", "--out", "score_sens.json"])
    # Grid
    code2, text2 = run_main(common + ["--statistic", "grid", "--exclude", "2025-07-29T00:00", "2025-08-11T00:00", "--out", "score_grid_sens.json"])
    # Check
    ok &= report("main script: sensitivity results pass the protected-value check against their primary files and are written", code1 == 0 and code2 == 0 and "PASS  protected values identical" in text1 and "PASS  protected values identical" in text2 and all(os.path.isfile(os.path.join(run_dir, n)) for n in ("score_sens.json", "score_grid_sens.json")))
    # The sensitivity file records the exclusion and the primary file does not
    with open(os.path.join(run_dir, "score_sens.json")) as f:
        # Load
        sens_file = json.load(f)
    # Primary
    with open(os.path.join(run_dir, "score.json")) as f:
        # Load
        prim_file = json.load(f)
    # Check
    ok &= report("main script: the sensitivity file records the exclusion and the primary file is unchanged in layout", sens_file["settings"]["exclude"] == [list(EXCL)] and "exclude" not in prim_file["settings"] and "exclusions" not in prim_file["modes"]["mag"]["model"])
    # Tampered primary files: one protected value changed each, the write must be refused and name the field
    tampers = (("a per-channel threshold", lambda d: d["modes"]["mag"]["model"]["thresholds"]["mag"][0].__setitem__(0, 123.0), "thresholds"),
               ("a missing-threshold list", lambda d: d["modes"]["ang"]["persistence"]["calibration"]["channels_without_threshold"]["ang"].append("m9|L1"), "channels_without_threshold"),
               ("a calibration-record threshold", lambda d: d["modes"]["mag"]["model"]["calibrations"][CALIB]["thresholds"]["mag"][1].__setitem__(0, 5.0), f"calibrations[{CALIB}].thresholds"),
               ("a block's calibration assignment", lambda d: d["modes"]["mag"]["model"]["blocks"][0].__setitem__("calib_label", "other"), "per-block calibration labels"),
               ("a scoring setting", lambda d: d["settings"].__setitem__("merge_slots", 7), "settings.merge_slots"),
               ("a missing mode", lambda d: d["modes"].pop("ang"), "ang/model: required mode/source missing or empty in the primary result"),
               ("an empty persistence record", lambda d: d["modes"]["mag"].__setitem__("persistence", {}), "mag/persistence: required mode/source missing or empty in the primary result"),
               ("a calibration record without its thresholds", lambda d: d["modes"]["ang"]["model"]["calibrations"][CALIB].pop("thresholds"), f"ang/model primary: calibrations[{CALIB}].thresholds missing"),
               ("no block_calibration map", lambda d: d["modes"]["mag"]["persistence"].pop("block_calibration"), "mag/persistence primary: block_calibration missing"))
    # Each
    for name, change, field in tampers:
        # Copy
        tampered = json.loads(json.dumps(prim_file))
        # Change
        change(tampered)
        # Save
        path = os.path.join(run_dir, "tampered.json")
        # Write
        with open(path, "w") as f:
            # JSON
            json.dump(tampered, f)
        # Run against it
        code, text = run_main(common + ["--exclude", "2025-07-29T00:00", "2025-08-11T00:00", "--out", "score_refused.json", "--primary", path])
        # Refused, field named, nothing written
        ok &= report(f"main script: a primary file with {name} changed refuses the write and names the field", code != 0 and field in text and not os.path.exists(os.path.join(run_dir, "score_refused.json")), f"exit {code}")
    # Grid: a changed reference
    with open(os.path.join(run_dir, "score_grid.json")) as f:
        # Load
        grid_file = json.load(f)
    # Change
    grid_file["modes"]["mag"]["model"]["grid"]["ref_mag"][0][0] = 9.0
    # Save
    with open(os.path.join(run_dir, "tampered.json"), "w") as f:
        # JSON
        json.dump(grid_file, f)
    # Run
    code, text = run_main(common + ["--statistic", "grid", "--exclude", "2025-07-29T00:00", "2025-08-11T00:00", "--out", "score_refused.json", "--primary", os.path.join(run_dir, "tampered.json")])
    # Check
    ok &= report("main script: a primary file with a grid reference changed refuses the write and names the field", code != 0 and "grid.ref_mag" in text and not os.path.exists(os.path.join(run_dir, "score_refused.json")))
    # Grid primary without its grid record
    grid_file["modes"]["mag"]["model"]["grid"] = None
    # Save
    with open(os.path.join(run_dir, "tampered.json"), "w") as f:
        # JSON
        json.dump(grid_file, f)
    # Run
    code, text = run_main(common + ["--statistic", "grid", "--exclude", "2025-07-29T00:00", "2025-08-11T00:00", "--out", "score_refused.json", "--primary", os.path.join(run_dir, "tampered.json")])
    # Check
    ok &= report("main script: a grid primary file without its grid record refuses the write before output creation", code != 0 and "grid record missing under the grid statistic" in text and not os.path.exists(os.path.join(run_dir, "score_refused.json")))
    # A sensitivity run must score magnitude and angle
    code, _ = run_main(common[:-3] + ["--modes", "mag", "--exclude", "2025-07-29T00:00", "2025-08-11T00:00", "--out", "score_refused.json"])
    # Check
    ok &= report("main script: --exclude with --modes mag alone is refused", code == 2 and not os.path.exists(os.path.join(run_dir, "score_refused.json")))
    # Direct cases on the pre-write check: the untouched channel and grid pairs pass
    with open(os.path.join(run_dir, "score_grid_sens.json")) as f:
        # Load
        grid_sens = json.load(f)
    # Primary grid
    with open(os.path.join(run_dir, "score_grid.json")) as f:
        # Load
        grid_prim = json.load(f)
    # Check
    ok &= report("pre-write check: the untouched channel and grid pairs have no finding for mag and ang", ss.protected_mismatches(sens_file, prim_file, ["mag", "ang"]) == [] and ss.protected_mismatches(grid_sens, grid_prim, ["mag", "ang"]) == [])
    # Structural cases on the sensitivity side
    pre = (("the angle mode omitted from the sensitivity result", lambda d: d["modes"].pop("ang"), ["ang/model: required mode/source missing or empty in the sensitivity result", "ang/persistence: required mode/source missing or empty in the sensitivity result"]),
           ("an empty modes table in the sensitivity result", lambda d: d.__setitem__("modes", {}), [f"{m}/{x}: required mode/source missing or empty in the sensitivity result" for m in ("mag", "ang") for x in ("model", "persistence")]),
           ("an empty model record in the sensitivity result", lambda d: d["modes"]["ang"].__setitem__("model", {}), ["ang/model: required mode/source missing or empty in the sensitivity result"]))
    # Each
    for name, change, expected in pre:
        # Copy
        tampered = json.loads(json.dumps(sens_file))
        # Change
        change(tampered)
        # Findings
        found = ss.protected_mismatches(tampered, prim_file, ["mag", "ang"])
        # Exactly the expected findings
        ok &= report(f"pre-write check: {name} fails for those pairs only", sorted(found) == sorted(expected), str(found)[:200])
    # Required calibration evidence missing from both sides fails and names both sides
    both = (("thresholds", lambda d: d["modes"]["mag"]["model"].pop("thresholds"), ["mag/model sensitivity: thresholds record missing or lacking non-empty meters, channels, mag and ang lists", "mag/model primary: thresholds record missing or lacking non-empty meters, channels, mag and ang lists"]),
            ("calibration.channels_without_threshold", lambda d: d["modes"]["ang"]["persistence"]["calibration"].pop("channels_without_threshold"), ["ang/persistence sensitivity: calibration.channels_without_threshold missing or lacking mag and ang lists", "ang/persistence primary: calibration.channels_without_threshold missing or lacking mag and ang lists"]),
            (f"calibrations[{CALIB}].grid_threshold", lambda d: d["modes"]["mag"]["persistence"]["calibrations"][CALIB].pop("grid_threshold"), [f"mag/persistence sensitivity: calibrations[{CALIB}].grid_threshold missing", f"mag/persistence primary: calibrations[{CALIB}].grid_threshold missing"]),
            ("the grid key", lambda d: d["modes"]["ang"]["model"].pop("grid"), ["ang/model sensitivity: grid key missing", "ang/model primary: grid key missing"]),
            ("the whole calibrations table", lambda d: d["modes"]["mag"]["model"].__setitem__("calibrations", {}), ["mag/model sensitivity: calibrations missing or empty", "mag/model primary: calibrations missing or empty"]))
    # Each
    for name, change, expected in both:
        # Copies
        t_s, t_p = json.loads(json.dumps(sens_file)), json.loads(json.dumps(prim_file))
        # Remove on both sides
        change(t_s); change(t_p)
        # Findings
        found = ss.protected_mismatches(t_s, t_p, ["mag", "ang"])
        # Exactly those, nothing else
        ok &= report(f"pre-write check: {name} missing from both sides fails and names both sides", sorted(found) == sorted(expected), str(found)[:200])
    # Grid evidence missing from both grid files
    t_s, t_p = json.loads(json.dumps(grid_sens)), json.loads(json.dumps(grid_prim))
    # Remove the references on both sides
    t_s["modes"]["mag"]["model"]["grid"].pop("ref_mag"); t_p["modes"]["mag"]["model"]["grid"].pop("ref_mag")
    # Findings
    found = ss.protected_mismatches(t_s, t_p, ["mag", "ang"])
    # Check
    ok &= report("pre-write check: grid references missing from both grid files fails and names both sides", sorted(found) == ["mag/model primary: grid.ref_mag and grid.ref_ang must be non-empty lists", "mag/model sensitivity: grid.ref_mag and grid.ref_ang must be non-empty lists"], str(found)[:200])
    # Legitimate nulls: a null per-channel threshold on both sides is accepted
    t_s, t_p = json.loads(json.dumps(sens_file)), json.loads(json.dumps(prim_file))
    # Null a threshold on both sides, in both places it is recorded
    for d in (t_s, t_p):
        # Default record
        d["modes"]["mag"]["model"]["thresholds"]["mag"][0][0] = None
        # Calibration record
        d["modes"]["mag"]["model"]["calibrations"][CALIB]["thresholds"]["mag"][0][0] = None
    # Findings
    ok &= report("pre-write check: a null per-channel threshold recorded on both sides is accepted (channels without a threshold are written as null)", ss.protected_mismatches(t_s, t_p, ["mag", "ang"]) == [])
    # Channel statistic with a grid record present on both sides is refused as inconsistent evidence
    t_s, t_p = json.loads(json.dumps(sens_file)), json.loads(json.dumps(prim_file))
    # Put a grid record where the channel statistic writes null
    for d in (t_s, t_p):
        # Set
        d["modes"]["ang"]["persistence"]["grid"] = {"threshold": 1.0}
    # Findings
    found = ss.protected_mismatches(t_s, t_p, ["mag", "ang"])
    # Check
    ok &= report("pre-write check: a grid record under the channel statistic on both sides is a finding on both sides", sorted(found) == ["ang/persistence primary: grid record present under the channel statistic", "ang/persistence sensitivity: grid record present under the channel statistic"], str(found)[:200])
    # Case 9: the event-block and episode preservation comparison
    with open(os.path.join(run_dir, "score_grid_sens.json")) as f:
        # Load
        grid_sens = json.load(f)
    # With the fixed exclusion, episode 2 is dropped, which the comparison must report for every mode and source
    found = sc.compare_files(sens_file, prim_file, ["test_events_"])
    # Check
    ok &= report("preservation comparison: with episode 2 inside the exclusion every mode and source reports the dropped episode, the scope count, the identities and the partial list, and no event-block or detection difference", set(found) == {("mag", "model"), ("mag", "persistence"), ("ang", "model"), ("ang", "persistence")} and all(len(v) == 4 and all(any(w in item for item in v) for w in ("episodes_dropped", "episodes_in_scope", "episode identities", "episodes_partial_excluded")) for v in found.values()))
    # A short exclusion that touches no episode: the comparison passes for every mode and source
    code, text = run_main(common + ["--exclude", "2025-07-29T08:00:00+00:00", "2025-07-29T09:00:00+00:00", "--out", "score_short.json"])
    # Load
    with open(os.path.join(run_dir, "score_short.json")) as f:
        # File
        short_file = json.load(f)
    # Compare
    found = sc.compare_files(short_file, prim_file, ["test_events_"])
    # Check
    ok &= report("preservation comparison: with an exclusion touching no episode, event-block records, scope, detections and event-block false alarms are identical for mag and ang, model and persistence", code == 0 and short_file["settings"]["exclude"] == [SHORT] and len(found) == 4 and all(not v for v in found.values()), str({k: v for k, v in found.items() if v}))
    # The quiet block did change in that file, so the comparison is not vacuous
    ok &= report("preservation comparison: the quiet block of the short-exclusion file differs from the primary file", quiet_block(short_file["modes"]["mag"]["model"])["monitored_slots"] < quiet_block(prim_file["modes"]["mag"]["model"])["monitored_slots"])
    # Tampered event-block fields are caught: monitored slots, a detection, an event-block false alarm, the event split summary
    tampers = (("an event-block monitored count", lambda d: [b for b in d["modes"]["mag"]["model"]["blocks"] if b["label"] == EVENTS][0].__setitem__("monitored_slots", 1), "monitored_slots"),
               ("a detection's alarm start", lambda d: d["modes"]["ang"]["model"]["episode_results"][0].__setitem__("alarm_start", "2024-11-13T09:00:00+00:00"), "alarm_start"),
               ("the event-split summary", lambda d: d["modes"]["mag"]["persistence"]["per_split"]["test_events_"].__setitem__("false_alarm_episodes", 99), "per_split[test_events_]"),
               ("a nonzero excluded_slots on an event block", lambda d: [b for b in d["modes"]["mag"]["model"]["blocks"] if b["label"] == EVENTS23][0].__setitem__("excluded_slots", 5), "excluded_slots"),
               ("an event-block false alarm", lambda d: d["modes"]["mag"]["model"]["false_alarm_list"].insert(0, ["2024-11-13T08:30:00+00:00", "2024-11-13T08:30:10+00:00"]), "event-block false alarms"))
    # Each
    for name, change, field in tampers:
        # Copy
        tampered = json.loads(json.dumps(short_file))
        # Change
        change(tampered)
        # Compare
        found = sc.compare_files(tampered, prim_file, ["test_events_"])
        # Check
        ok &= report(f"preservation comparison: {name} changed is reported", any(any(field in item for item in v) for v in found.values()))
    # Required modes and sources, and required fields, must be present on both sides
    structural = (("the angle mode omitted from the sensitivity file", lambda d: d["modes"].pop("ang"), [("ang", "model"), ("ang", "persistence")], "missing or empty in the sensitivity file"),
                  ("an empty modes table in the sensitivity file", lambda d: d.__setitem__("modes", {}), [("mag", "model"), ("mag", "persistence"), ("ang", "model"), ("ang", "persistence")], "missing or empty in the sensitivity file"),
                  ("an empty persistence record in the sensitivity file", lambda d: d["modes"]["mag"].__setitem__("persistence", {}), [("mag", "persistence")], "missing or empty in the sensitivity file"))
    # Each
    for name, change, keys, phrase in structural:
        # Copy
        tampered = json.loads(json.dumps(short_file))
        # Change
        change(tampered)
        # Compare
        found = sc.compare_files(tampered, prim_file, ["test_events_"])
        # Check: exactly those keys fail with the phrase, the others pass
        ok &= report(f"preservation comparison: {name} fails for the affected mode/source pairs only", all(len(found[k]) == 1 and phrase in found[k][0] for k in keys) and all(not v for k, v in found.items() if k not in keys) and len(found) == 4)
    # A required field missing from both sides fails rather than matching
    for field in ("episode_results", "per_split", "blocks"):
        # Copies
        t_s, t_p = json.loads(json.dumps(short_file)), json.loads(json.dumps(prim_file))
        # Remove on both sides
        t_s["modes"]["mag"]["model"].pop(field); t_p["modes"]["mag"]["model"].pop(field)
        # Compare
        found = sc.compare_files(t_s, t_p, ["test_events_"])
        # Check: both sides named, only that pair fails
        ok &= report(f"preservation comparison: {field} missing from both sides fails and names both files", any(f"sensitivity: required field {field} missing" in x for x in found[("mag", "model")]) and any(f"primary: required field {field} missing" in x for x in found[("mag", "model")]) and all(not v for k, v in found.items() if k != ("mag", "model")))
    # A legitimately empty list and the nullable values pass: the primary file's missed episodes carry null alarm_start and delay_s, and the fixture's event blocks have empty event-block false alarms
    ok &= report("preservation comparison: null alarm_start and delay_s of missed episodes and empty lists are accepted", any(r["alarm_start"] is None for m in prim_file["modes"].values() for r in m["model"]["episode_results"]) and not sc.schema_findings(prim_file["modes"]["mag"]["model"], "primary", ["test_events_"], False))
    # The comparison script end to end: the two selected version-1 pairs (score_short.json stands in for the sensitivity file of score.json)
    code, text = run_compare(["--run-dirs", run_dir, "--pairs", "score_short.json=score.json", "score_grid_sens.json=score_grid.json"])
    # The grid pair has the fixed exclusion (episode 2 dropped) so it fails; the short pair passes; counts printed
    ok &= report("comparison script: with two selected pairs present it compares both, prints the counts, and exits 1 on the dropped-episode pair", code == 1 and "file pairs compared: 2; mode/source comparisons completed: 8" in text and text.count("PASS  score_short.json against score.json") == 4 and "FAIL  score_grid_sens.json against score_grid.json" in text)
    # One selected pair that passes, exit 0
    code, text = run_compare(["--run-dirs", run_dir, "--pairs", "score_short.json=score.json"])
    # Check
    ok &= report("comparison script: one selected pair that passes exits 0 with its counts", code == 0 and "file pairs compared: 1; mode/source comparisons completed: 4" in text and "all comparisons passed" in text)
    # A selected file that does not exist is a failure, not a skip
    code, text = run_compare(["--run-dirs", run_dir, "--pairs", "score_short.json=score.json", "score_era_sens.json=score_era.json"])
    # Check
    ok &= report("comparison script: a selected pair with a missing file fails and names the file(s)", code == 1 and "FAIL  score_era_sens.json against score_era.json: file(s) not present: score_era_sens.json, score_era.json" in text and "file pairs compared: 1" in text)
    # The default four pairs on this fixture: the version-2 files are absent, so it fails
    code, text = run_compare(["--run-dirs", run_dir])
    # Check
    ok &= report("comparison script: the default four pairs fail on a folder without the version-2 files", code == 1 and "file(s) not present: score_era_sens.json, score_era.json" in text)
    # Zero comparisons is a failure of its own
    code, text = run_compare(["--run-dirs", run_dir, "--pairs", "nothing_sens.json=nothing.json"])
    # Check
    ok &= report("comparison script: zero completed comparisons fails with its own line", code == 1 and "FAIL  no mode/source comparison was completed" in text and "mode/source comparisons completed: 0" in text)
    # A required mode missing from a file on disk is reported by the script
    with open(os.path.join(run_dir, "score_noang.json"), "w") as f:
        # Copy without the angle mode
        json.dump({k: (v if k != "modes" else {m: v[m] for m in v if m != "ang"}) for k, v in short_file.items()}, f)
    # Run
    code, text = run_compare(["--run-dirs", run_dir, "--pairs", "score_noang.json=score.json"])
    # Check
    ok &= report("comparison script: a sensitivity file without the angle mode fails for ang/model and ang/persistence", code == 1 and "ang/model: 1 difference(s)" in text and "ang/persistence: 1 difference(s)" in text and "mode/source comparisons completed: 4" in text)
    # Verdict
    print("\nall checks passed" if ok else "\nsome checks FAILED")
    # Exit
    sys.exit(0 if ok else 1)


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
