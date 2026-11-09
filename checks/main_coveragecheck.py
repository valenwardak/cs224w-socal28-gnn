# =============================================================================
# File:        main_coveragecheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-08
#
# Purpose:     Conformal coverage check for scored runs. The scoring
#              (socal_score.py, main_score.py) sets each channel's
#              threshold on one calibration block at level alpha and
#              reports coverage on that block only, where it is fixed by
#              the conformal rank. This check measures the empirical
#              non-exceedance rate (coverage) against the nominal 1 - alpha
#              on every block the run has residuals for: the validation
#              blocks, the event blocks and the quiet block.
#
#              Nothing is rescored and no threshold changes. The thresholds,
#              references, calibration mapping and settings are read from
#              the score files the run already has (score.json,
#              score_grid.json, score_era.json, score_grid_era.json). The
#              validity rules are the scorer's own, reused by importing
#              socal_score: stored validity flags, the common model and
#              persistence mask, the common-channel restriction when it was
#              used, channels without a threshold (excluded, counted
#              separately), and the exclusion of slots inside the windows
#              of event episodes only partly covered by the scored blocks.
#
#              Before counting, the check recomputes every calibration set
#              with the scorer's functions from the saved settings and
#              compares it with what the score file stores: per-channel
#              thresholds (NaN locations must match exactly; finite values
#              within a tolerance), grid references and the grid threshold.
#              A mismatch is reported and the check stops, since the counts
#              would not describe the thresholds the scoring used. After
#              the comparison passes, the counts use the stored thresholds,
#              grid thresholds and the main calibration set's stored
#              reference arrays, not the recomputed ones, so a difference
#              inside the tolerance cannot move an exceedance; other
#              calibration sets have no stored reference array, so theirs
#              are recomputed from the design block with their own masks
#              and verified against the saved exclusion and floor lists. The test blocks and their calibration
#              labels come from the score file's own block_calibration map;
#              the saved rules must reproduce it, every mapped table must
#              exist, and tables in the folder that the file never scored
#              are left out and named.
#
#              Three roles are kept apart in the output and never pooled:
#                calibration, in-sample     the block that set the threshold
#                design/selection           validation block 0 (January
#                                           2024): early stopping and the
#                                           grid reference scales
#                held out                   every other block, with the
#                                           calibration set that was applied
#                                           named on the row; under version 2
#                                           November 2023 calibrates the 2023
#                                           event blocks and is held out for
#                                           the rest, and the role says so
#              Test blocks are measured under the calibration set the score
#              file assigned to them. Validation blocks are extra
#              diagnostics: each calibration set in the file is applied to
#              them in turn and the row names which one.
#
#              Quiet-block and event-block tables are separate. Event-block
#              exceedances include detection behavior and possible
#              distribution shift, not only miscalibration, so every event
#              row also gives exceedances and opportunities inside and
#              outside the scored episode windows (the scorer's scope and
#              window definitions). Coverage is pooled as total exceedances
#              over total opportunities, never as an average of channel or
#              monthly rates, and never over calibration or design/selection
#              rows; per-channel summaries stand beside it, and the worst
#              channels of a block are followed month by month (the same
#              channel list every month; months in the project's fixed
#              Pacific Standard Time, as the block labels). The channel
#              statistic's mode "both" is a union of two calibrated kinds
#              and is labeled without a nominal level. Slots ten seconds
#              apart are dependent, so counts and rates only, no intervals.
#
#              What the channel coverage can and cannot say: it locates
#              threshold-transfer failures per channel. The alarm-slot
#              occupancy the scoring reports additionally depends on two
#              meters exceeding in the same slot, so the two rates are
#              related, not interchangeable.
#
# Usage (from the repo root; the four default files are read if present):
#   python checks/main_coveragecheck.py --run-dir runs/gcn_dyn_aw001 runs/gcn_dyn_none
#   python checks/main_coveragecheck.py --run-dir runs/gcn_dyn_aw001 --files score.json --modes mag --no-months
#   python checks/main_coveragecheck.py --run-dir runs/gcn_dyn_aw001 --events /content/features/events.csv
#
# Functions:
#   parse_args()                          command-line options with defaults
#   load_score(run_dir, name)             read one score file, or None if absent
#   describe_file(score)                  "scoring version N, channel/grid statistic"
#   nominal(settings)                     the nominal coverage 1 - alpha
#   recompute(run_dir, score, r, mode, source, events)
#                                         redo the calibration sets with the scorer's functions
#   same_pattern(stored, fresh, tol)      NaN locations equal and finite values within tol
#   check_calibration(r, rec, tol)        compare stored thresholds, references, grid threshold
#   block_role(label, applied, settings, cal_labels)
#                                         the role string of a block under one calibration set
#   month_keys(t)                         "YYYY-MM" per slot
#   channel_counts(arr, thr, mode, keep)  per-channel and per-slot opportunity, exceedance, excluded counts
#   grid_counts(arr, grid, mode, keep)    per-slot monitored and exceedance flags and available meters
#   rate(exc, opp)                        coverage text from counts
#   names_of(mask, meters, channels)      "meter|channel" names of the True entries of a mask
#   worst_channels(opp, exc, n)           indices of the channels with the lowest coverage
#   block_rows(run_dir, rec, mode, settings, events, months, n_worst)
#                                         the coverage rows for one source and mode
#   print_rows(rows, settings, grid, mode)
#                                         print the tables for one source and mode
#   main()                                loop over runs, files, modes and sources
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: score files
import json
# Standard library: paths
import os
# Standard library: repo root on the import path, exit on failure
import sys
# Third party: numerics
import numpy as np
# Third party: timestamps and month keys
import pandas as pd

# This file lives in checks/; the modules are one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The scorer: its validity rules and calibration functions are reused, not copied
import socal_score as ss

# The four score files main_score.py writes, in the order the summary reader uses
DEFAULT_FILES = ["score.json", "score_grid.json", "score_era.json", "score_grid_era.json"]


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Conformal coverage check on every scored block of one or more runs")
    # Runs
    parser.add_argument("--run-dir", nargs="+", required=True, help="Run folders with residual tables and score files")
    # Score files
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES, help="Score files to read in each run folder; missing ones are skipped")
    # Modes
    parser.add_argument("--modes", nargs="+", choices=["mag", "ang", "both"], default=["mag", "ang"], help="Residual kinds to check")
    # Sources
    parser.add_argument("--sources", nargs="+", choices=["model", "persistence"], default=["model", "persistence"], help="Sources to check")
    # Monthly breakdown
    parser.add_argument("--months", action=argparse.BooleanOptionalAction, default=True, help="Monthly breakdown inside every block longer than one month (default on)")
    # Tolerance
    parser.add_argument("--tol", type=float, default=1e-6, help="Relative and absolute tolerance for finite threshold and reference values")
    # Events file override
    parser.add_argument("--events", default=None, help="Event table; default is the path recorded in each score file")
    # Worst channels to list
    parser.add_argument("--worst", type=int, default=5, help="Channels with the lowest coverage listed per block")
    # Parse
    args = parser.parse_args()
    # Checks
    if args.tol <= 0:
        # Refuse
        parser.error(f"--tol must be positive, got {args.tol}")
    # Checks
    if args.worst < 0:
        # Refuse
        parser.error(f"--worst must not be negative, got {args.worst}")
    # Done
    return args


def load_score(run_dir, name):
    """Read one score file from a run folder, or return None when it is absent."""
    # Path
    path = os.path.join(run_dir, name)
    # Absent
    if not os.path.exists(path):
        # Skip
        return None
    # Read
    with open(path) as f:
        # Parse
        return json.load(f)


def describe_file(score):
    """Name the scoring version and statistic of a score file."""
    # Settings
    s = score["settings"]
    # Text
    return f"scoring version {s.get('scoring_version', 1)}, {s.get('statistic', 'channel')} statistic"


def nominal(settings):
    """The nominal coverage, one minus alpha."""
    # Value
    return 1.0 - float(settings["alpha"])


def recompute(run_dir, score, r, mode, source, events):
    """
    Redo the calibration of one source and mode with the scorer's own
    functions from the saved settings. The test blocks and their
    calibration labels are read from the stored result's block_calibration
    map, which is authoritative: resolving the saved era rules must
    reproduce it, every mapped table must exist, and tables present in the
    folder but absent from the map are not scored (they would widen the
    original scope). Builds the design-block references when the grid
    statistic or the restriction is in use, and one calibration set per
    calibration label. Returns a dict with meters, channels, labels,
    calib_of, cal_sets, cal_labels, settings (with mode and events),
    prefix, other.
    """
    # Settings with the mode and the events path filled in
    settings = dict(score["settings"], mode=mode, events=events)
    # Source prefixes as the scorer uses them
    prefix, other = ("", "pers_") if source == "model" else ("pers_", "")
    # Meters and channels from the calibration table
    meters, channels = ss.table_axes(ss.load_table(run_dir, settings["calib_label"]))
    # Saved map: test label -> calibration label
    calib_of = dict(r["block_calibration"])
    # Test labels in the saved order
    labels = list(calib_of)
    # Every mapped table must exist (load_table raises otherwise)
    for lab in labels:
        # Check
        ss.load_table(run_dir, lab)
    # Resolve the saved rules on the saved labels
    resolved, _ = ss.resolve_calibration(labels, settings)
    # Must reproduce the saved map
    if resolved != calib_of:
        # Refuse
        raise ValueError(f"the saved era rules resolve to {resolved} but the score file maps {calib_of}")
    # Tables in the folder that the score file never scored
    extra = [lab for p in settings["test_prefixes"] for lab in ss.list_tables(run_dir, p) if lab not in calib_of]
    # Say so
    if extra:
        # Note, not an error: they are left out
        print(f"  note: test tables in the folder but not in this score file, left out: {extra}")
    # Calibration labels, default first, as the scorer orders them
    cal_labels = sorted(set(calib_of.values()), key=lambda c: (c != settings["calib_label"], c))
    # Grid or restriction needs the design-block references
    need_refs = settings.get("statistic", "channel") == "grid" or bool(settings.get("restrict_common", False))
    # One calibration set per label
    cal_sets = {}
    # Loop
    for c in cal_labels:
        # Held-out check, as the scorer does
        ss.check_held_out(run_dir, c)
        # References start empty
        refs = None
        # Needed
        if need_refs:
            # Design table
            des = ss.load_table(run_dir, settings["design_label"])
            # Arrays of this source
            darr = ss.residual_arrays(des, meters, channels, prefix)
            # Common mask with the other source
            if settings["common"]:
                # Mask
                ss.common_mask(darr, ss.residual_arrays(des, meters, channels, other))
            # References
            refs = ss.references(darr, settings["ref_percentile"], settings["ref_min_count"], settings["ref_floor_mag"], settings["ref_floor_ang"])
        # Calibration set
        cal_sets[c] = ss.calibrate_source(run_dir, c, prefix, other if settings["common"] else None, settings, meters, channels, refs)
    # Done
    return {"meters": meters, "channels": channels, "labels": labels, "calib_of": calib_of, "cal_sets": cal_sets, "cal_labels": cal_labels, "settings": settings, "prefix": prefix, "other": other}


def same_pattern(stored, fresh, tol):
    """
    Compare a stored array with a recomputed one: NaN locations must match
    exactly, finite values within tol (relative and absolute). Returns
    (ok, text) where text gives the counts and the largest difference.
    """
    # Arrays as float
    a = np.asarray(stored, dtype=np.float64)
    # Fresh
    b = np.asarray(fresh, dtype=np.float64)
    # Shape
    if a.shape != b.shape:
        # Fail
        return False, f"shape {a.shape} stored vs {b.shape} recomputed"
    # NaN patterns
    na, nb = np.isnan(a), np.isnan(b)
    # Mismatch count
    n_pat = int((na != nb).sum())
    # Finite in both
    both = ~na & ~nb
    # Largest difference
    diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
    # Within tolerance
    close = bool(np.allclose(a[both], b[both], rtol=tol, atol=tol)) if both.any() else True
    # Verdict
    ok = n_pat == 0 and close
    # Text
    return ok, f"{int(both.sum())} finite, {int(na.sum())} NaN stored, NaN pattern mismatches {n_pat}, max abs diff {diff:.2e}"


def names_of(mask, meters, channels):
    """Sorted "meter|channel" names of the True entries of an (M, C) mask."""
    # Names
    return sorted(f"{meters[mi]}|{channels[ci]}" for mi, ci in zip(*np.nonzero(mask)))


def check_calibration(r, rec, tol):
    """
    Compare the stored result of one source and mode with the recomputed
    calibration sets: per-channel thresholds of every calibration label,
    the grid threshold of every label, the grid references (the main
    label's stored arrays with an exact NaN pattern; for every label the
    recorded channels without a reference and at the floor), and the
    restriction's dropped channels when it was used. Prints one line per
    comparison. When everything agrees, the stored thresholds, grid
    thresholds and the main label's reference arrays replace the
    recomputed ones in rec, so the counts use the saved values; the other
    labels' references have no stored array and stay recomputed from the
    design block with their own verified masks (the output says so).
    Returns True when all agree.
    """
    # Overall
    ok = True
    # Axes
    meters, channels = rec["meters"], rec["channels"]
    # Main label
    main = rec["cal_labels"][0]
    # Stored main references, when the grid is in use
    stored_refs = None if r.get("grid") is None else {k: np.asarray(r["grid"][f"ref_{k}"], dtype=np.float32) for k in ("mag", "ang")}
    # Each calibration label
    for c, cset in rec["cal_sets"].items():
        # Stored record
        stored = r["calibrations"].get(c)
        # Absent
        if stored is None:
            # Fail
            print(f"  FAIL calibration {c}: not in the score file")
            # Mark
            ok = False
            # Next
            continue
        # Each kind
        for kind in ("mag", "ang"):
            # Compare thresholds
            good, text = same_pattern(stored["thresholds"][kind], cset["thr"][f"thr_{kind}"], tol)
            # Line
            print(f"  {'PASS' if good else 'FAIL'} thresholds {kind}, calibration {c}: {text}")
            # Fold
            ok &= good
            # Use the stored values from here on
            cset["thr"][f"thr_{kind}"] = np.asarray(stored["thresholds"][kind], dtype=np.float32)
        # Restriction
        if cset["restriction"] is not None or stored.get("restriction") is not None:
            # Both present
            if cset["restriction"] is None or stored.get("restriction") is None:
                # Fail
                print(f"  FAIL restriction, calibration {c}: present in one of stored and recomputed only")
                # Mark
                ok = False
            else:
                # Each kind
                for kind in ("mag", "ang"):
                    # Dropped lists
                    good = sorted(stored["restriction"][kind]["channels_dropped"]) == sorted(cset["restriction"][kind]["channels_dropped"])
                    # Line
                    print(f"  {'PASS' if good else 'FAIL'} restriction dropped {kind}, calibration {c}: stored {len(stored['restriction'][kind]['channels_dropped'])}, recomputed {len(cset['restriction'][kind]['channels_dropped'])}")
                    # Fold
                    ok &= good
        # Grid
        if cset["grid"] is not None:
            # Stored threshold
            g_st = stored.get("grid_threshold")
            # Fresh
            g_fr = cset["grid"]["threshold"]
            # Compare
            good = g_st is not None and abs(float(g_st) - float(g_fr)) <= tol * max(1.0, abs(float(g_fr)))
            # Line
            print(f"  {'PASS' if good else 'FAIL'} grid threshold, calibration {c}: stored {g_st} recomputed {g_fr:.6f}")
            # Fold
            ok &= good
            # Use the stored value
            if g_st is not None:
                # Replace
                cset["grid"]["threshold"] = float(g_st)
            # Each kind: exclusions, floor flags, values
            for kind in ("mag", "ang"):
                # Recomputed references of this set, rebuilt from the design block with this set's masks
                ref = cset["grid"]["refs"][f"ref_{kind}"]
                # Channels without a reference
                good = names_of(~np.isfinite(ref), meters, channels) == sorted(stored["grid_channels_without_reference"][kind])
                # Line
                print(f"  {'PASS' if good else 'FAIL'} grid channels without reference {kind}, calibration {c}: stored {len(stored['grid_channels_without_reference'][kind])}, recomputed {int((~np.isfinite(ref)).sum())}")
                # Fold
                ok &= good
                # Channels at the floor
                good = names_of(cset["grid"]["refs"][f"floored_{kind}"], meters, channels) == sorted(stored["grid_channels_at_floor"][kind])
                # Line
                print(f"  {'PASS' if good else 'FAIL'} grid channels at floor {kind}, calibration {c}: stored {len(stored['grid_channels_at_floor'][kind])}, recomputed {int(cset['grid']['refs'][f'floored_{kind}'].sum())}")
                # Fold
                ok &= good
                # The main set has its reference array stored: exact NaN pattern, values within tolerance, then the stored array is used
                if c == main and stored_refs is not None:
                    # Compare
                    good, text = same_pattern(stored_refs[kind], ref, tol)
                    # Line
                    print(f"  {'PASS' if good else 'FAIL'} grid reference values {kind}, calibration {c} (stored array): {text}")
                    # Fold
                    ok &= good
                    # Use the stored array
                    cset["grid"]["refs"][f"ref_{kind}"] = stored_refs[kind].copy()
                # Other sets have no stored array: their references stay recomputed, verified by the lists above; where both are finite the values must agree with the stored main array
                elif stored_refs is not None:
                    # Finite in both
                    both = np.isfinite(ref) & np.isfinite(stored_refs[kind])
                    # Compare
                    good = (not both.any()) or bool(np.allclose(ref[both], stored_refs[kind][both], rtol=tol, atol=tol))
                    # Line
                    print(f"  {'PASS' if good else 'FAIL'} grid reference values {kind}, calibration {c} (recomputed and verified; no stored array for this set): {int(np.isfinite(ref).sum())} finite, {int(both.sum())} shared with the stored {main} array, max abs diff {float(np.max(np.abs(ref[both] - stored_refs[kind][both]))) if both.any() else 0.0:.2e}")
                    # Fold
                    ok &= good
    # Done
    return ok


def block_role(label, applied, settings, cal_labels):
    """
    The role of a block under the calibration set `applied`: in-sample
    calibration, design/selection, held-out validation (noting when it is
    a calibration block of another set in this file), event block, or
    quiet block.
    """
    # The block set this threshold
    if label == applied:
        # In-sample
        return "calibration, in-sample"
    # Design block
    if label == settings["design_label"]:
        # Selection role
        return "design/selection, held out of this threshold"
    # Other validation block
    if label.startswith("validation_"):
        # Calibrates another set in this file
        if label in cal_labels:
            # Say so
            return "validation, held out of this threshold; calibrates other blocks in this file"
        # Plain
        return "validation, held out of this threshold"
    # Event blocks
    if label.startswith("test_events_"):
        # Role
        return "event block, held out"
    # Quiet block
    if label.startswith("test_normal_"):
        # Role
        return "quiet block, held out"
    # Anything else
    return "held out"


def month_keys(t):
    """Calendar month "YYYY-MM" of every slot in the project's fixed Pacific Standard Time (UTC minus 8 hours, no daylight saving), as the block labels use."""
    # Timestamps shifted to fixed PST
    times = pd.to_datetime(t, utc=True) - pd.Timedelta(hours=8)
    # Keys
    return np.asarray(times.strftime("%Y-%m"))


def channel_counts(arr, thr, mode, keep, mk=None):
    """
    Channel-statistic counts on one block. keep is the (T,) bool of slots
    kept (outside partial-episode windows); mk the (T,) month key per slot
    or None. Returns per-channel (M, C) opportunities, exceedances and
    excluded (valid residual on a channel without a threshold), per-slot
    (T,) opportunities and exceedances summed over channels, and with mk
    the per-month per-channel counts.
    """
    # Kinds in the mode
    kinds = ["mag", "ang"] if mode == "both" else [mode]
    # Watched: valid residual on a thresholded channel
    watched = np.zeros(arr["mag"].shape, dtype=bool)
    # Excluded: valid residual on a channel without a threshold
    excluded = np.zeros(arr["mag"].shape, dtype=bool)
    # Each kind
    for kind in kinds:
        # Threshold present
        has = np.isfinite(thr[f"thr_{kind}"])[None, :, :]
        # Watched
        watched |= arr[f"{kind}_valid"] & has
        # Excluded
        excluded |= arr[f"{kind}_valid"] & ~has
    # Firing channels, the scorer's rule
    fires = ss.exceedances(arr, thr, mode)
    # Only kept slots
    watched &= keep[:, None, None]
    # Only kept slots
    excluded &= keep[:, None, None]
    # Fires only count where watched
    fires &= watched
    # Result
    out = {"opp": watched.sum(axis=0), "exc": fires.sum(axis=0), "excl": excluded.sum(axis=0),
           "slot_opp": watched.sum(axis=(1, 2)), "slot_exc": fires.sum(axis=(1, 2))}
    # Per month and channel
    if mk is not None:
        # Months in order
        months = sorted(set(mk))
        # Counts per month: (month, opp (M, C), exc (M, C))
        out["month_channel"] = [(m, watched[mk == m].sum(axis=0), fires[mk == m].sum(axis=0)) for m in months]
    # Done
    return out


def grid_counts(arr, grid, mode, keep):
    """
    Grid-statistic flags on one block: monitored slots (finite grid score,
    outside partial windows), exceedances (score above the grid threshold),
    the available-meter count per slot, and the number of valid channel
    residuals on kept slots whose channel has no grid reference.
    """
    # Scores and availability, the scorer's rule
    gscore, gavail = ss.grid_scores(arr, grid["refs"], mode, grid["need"])
    # Monitored
    mon = np.isfinite(gscore) & keep
    # Exceedance
    exc = mon & (np.nan_to_num(gscore, nan=-np.inf) > grid["threshold"])
    # Valid channel residuals without a reference
    no_ref = 0
    # Each kind in the mode
    for kind in (["mag", "ang"] if mode == "both" else [mode]):
        # Count
        no_ref += int((arr[f"{kind}_valid"] & ~np.isfinite(grid["refs"][f"ref_{kind}"])[None, :, :] & keep[:, None, None]).sum())
    # Done
    return {"mon": mon, "exc": exc, "avail": gavail, "no_ref": no_ref}


def rate(exc, opp):
    """Coverage text from exceedance and opportunity counts, or n/a without opportunities."""
    # None
    if opp == 0:
        # Text
        return "   n/a  "
    # Value
    return f"{1.0 - exc / opp:.5f}"


def worst_channels(opp, exc, n):
    """The (meter index, channel index) pairs of the n channels with the lowest coverage among those with opportunities."""
    # Nothing asked
    if n == 0:
        # Empty
        return []
    # Candidates
    rows = []
    # Each channel
    for mi in range(opp.shape[0]):
        # Each
        for ci in range(opp.shape[1]):
            # With opportunities
            if opp[mi, ci] > 0:
                # Keep
                rows.append((1.0 - exc[mi, ci] / opp[mi, ci], mi, ci))
    # Sort by coverage
    rows.sort(key=lambda x: x[0])
    # Indices
    return [(mi, ci) for _, mi, ci in rows[:n]]


def block_rows(run_dir, rec, mode, settings, events, months, n_worst):
    """
    Coverage rows for one source and mode: every block in the run folder
    under the calibration set(s) that apply to it. Test blocks use the
    score file's mapping; validation blocks are measured under each
    calibration set in the file. Returns a list of row dicts.
    """
    # Grid in use
    use_grid = settings.get("statistic", "channel") == "grid"
    # Event windows
    before, after = pd.Timedelta(minutes=settings["before_minutes"]), pd.Timedelta(minutes=settings["after_minutes"])
    # Episodes
    episodes = ss.load_episodes(events)
    # All blocks with residual tables
    all_labels = sorted(set(ss.list_tables(run_dir, "validation_") + rec["labels"]))
    # Test block bounds, the scorer's scope
    test_bounds = {}
    # Each test label
    for lab in rec["labels"]:
        # Table times only
        t = ss.load_table(run_dir, lab).index
        # Bounds
        test_bounds[lab] = (pd.to_datetime(t[0], utc=True), pd.to_datetime(t[-1], utc=True))
    # Scope over the test blocks together, as the scorer computes it
    scoped_t, partial_t = ss.episodes_in_scope(episodes, list(test_bounds.values()), before, after)
    # Rows
    rows = []
    # Each block
    for lab in all_labels:
        # Calibration sets to apply
        applied = [rec["calib_of"][lab]] if lab in rec["calib_of"] else rec["cal_labels"]
        # Table
        table = ss.load_table(run_dir, lab)
        # Arrays of this source
        arr = ss.residual_arrays(table, rec["meters"], rec["channels"], rec["prefix"])
        # Common mask
        if settings["common"]:
            # Mask
            ss.common_mask(arr, ss.residual_arrays(table, rec["meters"], rec["channels"], rec["other"]))
        # Scope for this block
        if lab in rec["calib_of"]:
            # Test block: the scorer's scope
            scoped, partial = scoped_t, partial_t
        else:
            # Validation block: its own bounds
            b = (pd.to_datetime(arr["t"][0], utc=True), pd.to_datetime(arr["t"][-1], utc=True))
            # Scope
            scoped, partial = ss.episodes_in_scope(episodes, [b], before, after)
        # Slots kept: outside partial windows
        keep = ~ss.window_mask(arr["t"], partial, before, after)
        # Slots inside scoped episode windows
        inwin = ss.window_mask(arr["t"], scoped, before, after) & keep
        # Months
        mk = month_keys(arr["t"]) if months else None
        # Each calibration set
        for c in applied:
            # Set
            cset = rec["cal_sets"][c]
            # Row skeleton
            row = {"label": lab, "applied": c, "role": block_role(lab, c, settings, rec["cal_labels"]), "slots": int(len(arr["t"])), "partial_excluded_slots": int((~keep).sum())}
            # Channel statistic
            if not use_grid:
                # Counts
                cc = channel_counts(arr, cset["thr"], mode, keep, mk)
                # Totals
                row.update(opp=int(cc["opp"].sum()), exc=int(cc["exc"].sum()), excl=int(cc["excl"].sum()))
                # In window
                row.update(in_opp=int(cc["slot_opp"][inwin].sum()), in_exc=int(cc["slot_exc"][inwin].sum()))
                # Out of window
                row.update(out_opp=int(cc["slot_opp"][~inwin].sum()), out_exc=int(cc["slot_exc"][~inwin].sum()))
                # Channels with opportunities
                has = cc["opp"] > 0
                # Per-channel coverage
                cov = 1.0 - cc["exc"][has] / cc["opp"][has]
                # Worst channels over the whole block
                worst = worst_channels(cc["opp"], cc["exc"], n_worst)
                # Summary
                row.update(n_channels=int(has.sum()), cov_median=float(np.median(cov)) if has.any() else None, cov_min=float(cov.min()) if has.any() else None,
                           worst=[(f"{rec['meters'][mi]}|{rec['channels'][ci]}", int(cc["exc"][mi, ci]), int(cc["opp"][mi, ci])) for mi, ci in worst])
                # Monthly
                if months:
                    # Per month, all channels
                    row["months"] = [(m, int(e.sum()), int(o.sum())) for m, o, e in cc["month_channel"]]
                    # Per month for the same worst channels
                    row["worst_months"] = [(f"{rec['meters'][mi]}|{rec['channels'][ci]}", [(m, int(e[mi, ci]), int(o[mi, ci])) for m, o, e in cc["month_channel"]]) for mi, ci in worst]
            # Grid statistic
            else:
                # Flags
                gc = grid_counts(arr, cset["grid"], mode, keep)
                # Totals
                row.update(opp=int(gc["mon"].sum()), exc=int(gc["exc"].sum()), excl=int((keep & ~gc["mon"]).sum()), no_ref=gc["no_ref"])
                # In window
                row.update(in_opp=int((gc["mon"] & inwin).sum()), in_exc=int((gc["exc"] & inwin).sum()))
                # Out of window
                row.update(out_opp=int((gc["mon"] & ~inwin).sum()), out_exc=int((gc["exc"] & ~inwin).sum()))
                # By available meters
                row["by_avail"] = [(int(a), int((gc["exc"] & (gc["avail"] == a)).sum()), int((gc["mon"] & (gc["avail"] == a)).sum())) for a in np.unique(gc["avail"][gc["mon"]])]
                # Monthly
                if months:
                    # Per month
                    row["months"] = [(m, int((gc["exc"] & (mk == m)).sum()), int((gc["mon"] & (mk == m)).sum())) for m in sorted(set(mk))]
            # Keep
            rows.append(row)
    # Done
    return rows


def print_rows(rows, settings, grid, mode):
    """Print the validation, event and quiet tables for one source and mode, then the detail lines: per-channel summary, worst channels by month, grid by available meters, months."""
    # Nominal text: the channel statistic's "both" is a union of two calibrated kinds and has no single nominal level
    if grid or mode != "both":
        # Level
        nom_text = f"nominal coverage {nominal(settings):.5f}"
    else:
        # Union
        nom_text = "union of magnitude and angle exceedances, no single nominal level (each kind alone is calibrated at " + f"{nominal(settings):.5f})"
    # Header
    head = f"  {'block':<36} {'threshold from':<34} {'role':<70} {'opp':>10} {'exc':>7} {'coverage':>9} {'excl':>9} | {'in-window exc/opp':>20} {'out-window exc/opp':>20}"
    # Groups
    groups = [("validation blocks (diagnostic; each calibration set applied in turn)", lambda l: l.startswith("validation_")),
              ("event blocks (exceedances include detection behavior and possible distribution shift)", lambda l: l.startswith("test_events_")),
              ("quiet block", lambda l: l.startswith("test_normal_"))]
    # Each group
    for title, pick in groups:
        # Rows of the group
        sel = [r for r in rows if pick(r["label"])]
        # Empty
        if not sel:
            # Next
            continue
        # Title
        print(f"\n  {title}; {nom_text}")
        # Column meanings
        print("  opp = opportunities (" + ("valid residuals on thresholded channels" if not grid else "monitored slots") + ", outside partial-episode windows); exc = exceedances; excl = " + ("valid residuals on channels without a threshold" if not grid else "kept slots not monitored, fewer meters available than needed") + "; in/out-window = inside/outside the scored episode windows")
        # Header
        print(head)
        # Each row
        for r in sel:
            # Line
            print(f"  {r['label']:<36} {r['applied']:<34} {r['role']:<70} {r['opp']:>10} {r['exc']:>7} {rate(r['exc'], r['opp']):>9} {r['excl']:>9} | {r['in_exc']:>8}/{r['in_opp']:<11} {r['out_exc']:>8}/{r['out_opp']:<11}")
        # Pooled over the group: held-out rows only, never calibration or design/selection, per calibration set so a block is never counted twice
        for c in sorted(set(r["applied"] for r in sel)):
            # Held-out rows under this set
            held = [r for r in sel if r["applied"] == c and not r["role"].startswith(("calibration", "design"))]
            # Totals
            if held:
                # Sum
                o, e = sum(r["opp"] for r in held), sum(r["exc"] for r in held)
                # Line
                print(f"  pooled over the held-out rows above with threshold from {c} (total exceedances over total opportunities; calibration and design/selection rows left out): {e}/{o} coverage {rate(e, o)}")
        # Detail per row
        for r in sel:
            # Channel detail
            if not grid and r.get("n_channels") is not None:
                # Summary
                med = "n/a" if r["cov_median"] is None else f"{r['cov_median']:.5f}"
                # Min
                mn = "n/a" if r["cov_min"] is None else f"{r['cov_min']:.5f}"
                # Worst text
                worst = "; ".join(f"{name} {rate(e, o).strip()} ({e}/{o})" for name, e, o in r["worst"]) if r["worst"] else "none"
                # Line
                print(f"    {r['label']} <- {r['applied']}: {r['n_channels']} channels with opportunities, per-channel coverage median {med}, min {mn}; lowest over the block: {worst}")
                # Worst channels by month, same channel list every month
                if r.get("worst_months") and r.get("months") and len(r["months"]) > 1:
                    # Each channel
                    for name, per in r["worst_months"]:
                        # Line
                        print(f"      {name} by month (exc/opp, coverage): " + ", ".join(f"{m}: {e}/{o} {rate(e, o).strip()}" for m, e, o in per))
            # Grid detail
            if grid:
                # Reference exclusions
                print(f"    {r['label']} <- {r['applied']}: valid channel residuals on channels without a grid reference (channel-slots, not slots): {r.get('no_ref', 0)}")
                # By available meters
                if r.get("by_avail"):
                    # Line
                    print(f"    {r['label']} <- {r['applied']}: by available meters (exc/opp, coverage): " + ", ".join(f"{a}: {e}/{o} {rate(e, o).strip()}" for a, e, o in r["by_avail"]))
            # Monthly, all channels or all slots
            if r.get("months") and len(r["months"]) > 1:
                # Line
                print(f"    {r['label']} <- {r['applied']}: by fixed-PST month (exc/opp, coverage): " + ", ".join(f"{m}: {e}/{o} {rate(e, o).strip()}" for m, e, o in r["months"]))


def main():
    """Loop over runs, score files, modes and sources; check the calibration, then print the coverage tables."""
    # Options
    args = parse_args()
    # Summary lines for the end
    summary = []
    # Each run
    for run_dir in args.run_dir:
        # Name
        name = os.path.basename(os.path.normpath(run_dir))
        # Each score file
        for fname in args.files:
            # Read
            score = load_score(run_dir, fname)
            # Missing
            if score is None:
                # Note
                print(f"\n{name}: {fname} not found, skipped")
                # Next
                continue
            # Settings
            settings = score["settings"]
            # Events path
            events = args.events if args.events is not None else settings["events"]
            # Grid
            use_grid = settings.get("statistic", "channel") == "grid"
            # Each mode
            for mode in args.modes:
                # Result of the mode
                rm = score["modes"].get(mode)
                # Not scored in this mode
                if rm is None:
                    # Note
                    print(f"\n{name}, {fname}: mode {mode} not in the file, skipped")
                    # Next
                    continue
                # Each source
                for source in args.sources:
                    # Stored result
                    r = rm[source]
                    # Banner
                    print(f"\n== {name}, {fname} ({describe_file(score)}), mode {mode}, source {source} ==")
                    # Settings line
                    print(f"  alpha {settings['alpha']}, common {settings['common']}, restrict_common {settings.get('restrict_common', False)}, min_meters {settings['min_meters']}, windows -{settings['before_minutes']} / +{settings['after_minutes']} min, calibration default {settings['calib_label']}, era rules {settings.get('era_calib') or []}, design {settings['design_label']}")
                    # Recompute the calibration
                    rec = recompute(run_dir, score, r, mode, source, events)
                    # Compare
                    if not check_calibration(r, rec, args.tol):
                        # Stop
                        print("  calibration mismatch: the counts would not describe the thresholds the scoring used; stopping")
                        # Exit
                        sys.exit(1)
                    # Rows
                    rows = block_rows(run_dir, rec, mode, rec["settings"], events, args.months, args.worst)
                    # Print
                    print_rows(rows, settings, use_grid, mode)
                    # Quiet-block summary
                    quiet = [row for row in rows if row["label"].startswith("test_normal_")]
                    # Totals
                    o, e = sum(row["opp"] for row in quiet), sum(row["exc"] for row in quiet)
                    # Keep
                    summary.append((name, fname, mode, source, e, o))
    # Role note
    print("\nRoles: calibration, in-sample = the block that set the threshold; design/selection = validation block 0, used for early stopping and for the grid reference scales; everything else is held out of the threshold applied on its row")
    # Summary
    print("\nSummary: held-out quiet-block coverage (exceedances/opportunities) per run, file, mode, source")
    # Each line
    for name, fname, mode, source, e, o in summary:
        # Line
        print(f"  {name:<22} {fname:<20} {mode:<4} {source:<12} {e}/{o} coverage {rate(e, o).strip()}")


# Entry point
if __name__ == "__main__":
    # Run
    main()
