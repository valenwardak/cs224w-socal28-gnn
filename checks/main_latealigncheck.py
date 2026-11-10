# =============================================================================
# File:        main_latealigncheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-10
#
# Purpose:     Read-only check that the late-capture files written by
#              main_latefeatures.py line up with the processed tables they
#              came from. We need to know that a late flag at a feature
#              slot refers to the same instant as the processed reading
#              and follows the documented rule, before the flags decide
#              which angles the neighbor feature may use.
#
#              For a few meters of one month, per meter: the processed
#              index (dtype, time zone, first values); every row matched
#              to the feature slot of the same instant; the stored capture
#              offsets (count, smallest, largest, negative, at or above the
#              cutoff); every saved flag compared with the rule (late when
#              the offset is missing, not finite, or at or beyond the
#              threshold; a slot with no row, or a meter with no table, is
#              late); rows that do not match a slot, split into rows inside
#              the feature month's time range (an alignment failure) and
#              rows outside it (counted, their time range printed, and
#              compared with the documented month bounds); and example
#              rows on both sides of the cutoff.
#
#              Three verdicts, per meter and overall:
#                passed        matched rows were checked, every saved flag
#                              equals the rule, the index carries a time
#                              zone, no row is displaced, and the feature
#                              month covers exactly the expected bounds
#                failed        a flag mismatch, a negative offset, a row
#                              inside the feature range with no slot, or a
#                              row outside the documented processed month
#                not verified  the evidence is not enough: no matched rows,
#                              a time-zone-naive index (reading it as UTC
#                              is an assumption, not evidence), or a feature
#                              month whose coverage differs from the
#                              expected bounds without independent
#                              documentation. The expected bounds are the
#                              documented month (socal_modeldata.month_bounds)
#                              unless --coverage-bounds gives independently
#                              documented partial coverage; a shorter
#                              feature range is never taken as intentional
#                              on its own.
#              The overall verdict is the worst per-meter one. Exit code 0
#              only for passed, 1 for failed, 2 for not verified.
#
#              Time base, from the preprocessing code: socal_preprocess
#              indexes each table by the ten-second UTC grid with the UTC
#              label kept, and toff is the reading's UTC time minus its
#              slot's floor, so toff is never negative by construction.
#              The feature month covers the same documented bounds
#              (socal_modeldata.month_bounds: the fixed-PST calendar month
#              shifted to UTC). Both are verified on the files here, not
#              assumed. Nothing is written or changed.
#
# Usage (from the repo root, on the Mac where the processed tables are):
#   python checks/main_latealigncheck.py
#   python checks/main_latealigncheck.py --month 2024-11 --meters egauge_7 egauge_9 egauge_13
#   python checks/main_latealigncheck.py --month 2025-09 --coverage-bounds 1756713600 1756972800   (documented partial month)
#
# Functions:
#   parse_args()                command-line options with defaults
#   epoch_seconds(idx)          a DatetimeIndex as epoch seconds, with its time zone described
#   inspect_meter(...)          one meter's evidence and verdict
#   overall(verdicts)           the worst verdict and its exit code
#   main()                      inspect, print, exit with the code
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: paths
import os
# Standard library: import path for the repo root, and the exit code
import sys
# Third party: numerics
import numpy as np
# Third party: tables
import pandas as pd

# Repo root on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The late-file helpers, for the manifest validation and the column names
import socal_angdiff as sa
# The feature months and the documented month bounds
import socal_modeldata as sd

# Verdict order, worst last, and exit codes
VERDICTS = {"passed": 0, "failed": 1, "not verified": 2}


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Read-only check that the late files line up with the processed tables' capture offsets; exit 0 only when passed")
    # Processed tables
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"), help="Folder with <YYYY-MM>/<meter>.parquet")
    # Features
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with the feature months and the late files")
    # Month
    parser.add_argument("--month", default="2025-01", help="Month to inspect")
    # Meters
    parser.add_argument("--meters", nargs="+", default=["egauge_3", "egauge_9", "egauge_13"], help="Meters to inspect")
    # Rows to print per group
    parser.add_argument("--rows", type=int, default=3, help="Example rows printed per group (first rows, below the cutoff, at or above, missing offset)")
    # Threshold
    parser.add_argument("--late-threshold", type=float, default=sa.LATE_THRESHOLD, help="Threshold the late files must have been written with")
    # Independently documented coverage
    parser.add_argument("--coverage-bounds", nargs=2, type=int, default=None, metavar=("START_S", "END_S"), help="Independently documented feature coverage of the month as epoch seconds [start, end exclusive); default the documented month bounds. The feature month must match these exactly to pass")
    # Parse
    return parser.parse_args()


def epoch_seconds(idx):
    """A DatetimeIndex as epoch seconds, whether the index carries a time zone (converted to UTC) or not (read as if UTC), and a text describing dtype and time zone."""
    # Description
    text = f"dtype {idx.dtype}, tz {idx.tz}"
    # Aware: convert
    if idx.tz is not None:
        # UTC
        naive = idx.tz_convert("UTC").tz_localize(None)
    # Naive: the numbers are read as if UTC, which the caller treats as unverified
    else:
        # As is
        naive = idx
    # Seconds
    return np.asarray(naive, dtype="datetime64[s]").astype(np.int64), text


def inspect_meter(meter, mi, path, col, t, late, threshold, month_start, month_end, rows, expected=None):
    """
    Gather one meter's evidence, print it, and return its verdict. t is
    the feature month's slot times, late its late flags (T, M), mi the
    meter's column, month_start and month_end the documented bounds of the
    processed month in epoch seconds, expected the independently documented
    feature coverage [start, end) or None for the documented month. The
    feature range must equal the expected bounds exactly; otherwise the
    coverage is a discrepancy the files cannot explain and the verdict is
    at best not verified.
    """
    # Expected coverage
    exp_start, exp_end = (month_start, month_end) if expected is None else (int(expected[0]), int(expected[1]))
    # Does the feature month cover exactly the expected bounds
    covers = bool(t[0] == exp_start and int(t[-1]) + 10 == exp_end)
    # Header
    print(f"\n== {meter} ==")
    # Problems that make the verdict failed, and gaps that make it not verified
    failures, gaps = [], []
    # No table: the rule says late throughout
    if not os.path.exists(path):
        # All late
        all_late = bool(late[:, mi].all())
        # Say
        print(f"  no table at {path}: the rule says late throughout; saved flags late in {int(late[:, mi].sum())} of {len(t)} slots: {'as the rule says' if all_late else 'MISMATCH'}")
        # A missing table cannot verify alignment, only the flags
        if not all_late:
            # Failed
            failures.append("flags not all late for a meter without a table")
        # Verdict
        verdict = "failed" if failures else "not verified"
        # Reason
        print(f"  verdict: {verdict}" + ("" if failures else " (no readings to align; flags follow the rule)"))
        # Return
        return verdict
    # Read the offset column only
    table = pd.read_parquet(path, columns=[col])
    # Index as seconds
    ts, text = epoch_seconds(table.index)
    # Time zone
    aware = table.index.tz is not None
    # Say
    print(f"  processed index: {text}; {len(table)} rows; first values {list(table.index[:2])}; assumption used: {'none, the index carries a time zone and is converted to UTC' if aware else 'NAIVE INDEX READ AS UTC, not independently verified'}")
    # Naive is not verified
    if not aware:
        # Gap
        gaps.append("time-zone-naive processed index; its convention is not established by this check")
    # Offsets
    toff = table[col].to_numpy(dtype=np.float64)
    # Finite
    finite = np.isfinite(toff)
    # Range
    if finite.any():
        # Say
        print(f"  offsets: finite {int(finite.sum())} of {len(toff)}, min {toff[finite].min():.4f} s, max {toff[finite].max():.4f} s, negative {int((toff[finite] < 0).sum())}, at or above {threshold} s: {int((toff[finite] >= threshold).sum())}, missing or non-finite {int((~finite).sum())}")
        # Negative offsets contradict the construction
        if (toff[finite] < 0).any():
            # Failed
            failures.append(f"{int((toff[finite] < 0).sum())} negative offsets, impossible under toff = reading time minus slot floor")
    else:
        # Say
        print(f"  offsets: none finite ({len(toff)} rows)")
    # Match rows to slots
    pos = np.searchsorted(t, ts)
    # On the grid
    hit = (pos < len(t)) & (t[np.minimum(pos, len(t) - 1)] == ts)
    # Unmatched rows split by the feature month's time range
    in_range = (ts >= t[0]) & (ts <= t[-1])
    # Inside and unmatched: displaced rows
    displaced = (~hit) & in_range
    # Outside the range
    outside = ~in_range
    # Say
    print(f"  rows matched to a feature slot: {int(hit.sum())} of {len(ts)}; unmatched inside the feature range [{int(t[0])}, {int(t[-1])}] s: {int(displaced.sum())}; outside it: {int(outside.sum())}")
    # Displaced rows fail
    if displaced.any():
        # Examples
        ex = ts[displaced][:3]
        # Failed
        failures.append(f"{int(displaced.sum())} rows inside the feature range match no slot (first {ex.tolist()} s)")
    # Feature range against the expected coverage, said once
    print(f"  feature month range [{int(t[0])}, {int(t[-1]) + 10}) s against expected [{exp_start}, {exp_end}) s ({'documented month' if expected is None else 'given --coverage-bounds'}): {'equal' if covers else 'DIFFERENT'}; documented processed month [{month_start}, {month_end}) s")
    # A coverage difference the files cannot explain: not verified, whatever the overlapping rows say
    if not covers:
        # Gap
        gaps.append(f"feature coverage [{int(t[0])}, {int(t[-1]) + 10}) differs from the expected bounds [{exp_start}, {exp_end}); a shorter or shifted feature month is not taken as intentional without independent documentation (--coverage-bounds)")
    # Rows outside the feature range
    if outside.any():
        # Range
        lo, hi = int(ts[outside].min()), int(ts[outside].max())
        # Inside the documented processed month
        in_doc = (ts[outside] >= month_start) & (ts[outside] < month_end)
        # Say
        print(f"  rows outside the feature range: {int(outside.sum())}, from {lo} s ({pd.Timestamp(lo, unit='s', tz='UTC')}) to {hi} s ({pd.Timestamp(hi, unit='s', tz='UTC')}); of these inside the documented processed month: {int(in_doc.sum())}, outside it: {int((~in_doc).sum())}")
        # Rows outside the documented processed month: unexplained displacement
        if (~in_doc).any():
            # Failed
            failures.append(f"{int((~in_doc).sum())} rows lie outside the documented processed month [{month_start}, {month_end})")
        # Rows inside the documented processed month but outside the feature range
        if in_doc.any():
            # Explained only by documented partial coverage that the files match
            if expected is not None and covers:
                # Say
                print(f"  explained by the given coverage bounds: {int(in_doc.sum())} processed rows fall outside the documented feature coverage")
            # Otherwise part of the coverage discrepancy (already a gap when not covers; a contradiction when covers)
            elif covers:
                # The feature month covers the documented month, so these rows should have matched
                failures.append(f"{int(in_doc.sum())} rows lie inside the documented month but outside a feature range that covers it")
    # Rule per row
    rule_late = ~(finite & (toff < threshold))
    # Compare on matched rows
    saved = late[pos[hit], mi]
    # Agreement
    agree = saved == rule_late[hit]
    # Slots with no row must be late
    covered = np.zeros(len(t), dtype=bool)
    # Mark
    covered[pos[hit]] = True
    # Uncovered
    uncovered_late = bool(late[~covered, mi].all()) if (~covered).any() else True
    # Say
    print(f"  saved flag equals the rule on {int(agree.sum())} of {int(hit.sum())} matched rows; slots with no row: {int((~covered).sum())}, all flagged late: {uncovered_late}")
    # Mismatches fail
    if (~agree).any():
        # Failed
        failures.append(f"{int((~agree).sum())} matched rows where the saved flag differs from the rule")
    # Uncovered not late fails
    if not uncovered_late:
        # Failed
        failures.append("slots with no processed row that are not flagged late")
    # No matched rows: nothing verified
    if not hit.any():
        # Gap
        gaps.append("no processed row matched a feature slot")
    # Example rows
    groups = {"first rows": np.arange(min(rows, len(ts))),
              "below the cutoff": np.nonzero(finite & (toff < threshold))[0][:rows],
              "at or above the cutoff": np.nonzero(finite & (toff >= threshold))[0][:rows],
              "missing or non-finite offset": np.nonzero(~finite)[0][:rows]}
    # Each group
    for name, idx in groups.items():
        # None
        if len(idx) == 0:
            # Say
            print(f"  {name}: none in this month")
            # Next
            continue
        # Header
        print(f"  {name}:")
        # Rows
        for r in idx:
            # Slot
            slot = int(pos[r]) if hit[r] else None
            # Print
            print(f"    processed {table.index[r]} -> {int(ts[r])} s | feature slot {int(t[slot]) if slot is not None else 'none'} s {pd.Timestamp(int(t[slot]), unit='s', tz='UTC') if slot is not None else ''} same instant {slot is not None and int(t[slot]) == int(ts[r])} | toff {float(toff[r])} | saved late {bool(late[slot, mi]) if slot is not None else 'n/a'} | rule late {bool(rule_late[r])}")
    # Verdict: failures first, then gaps
    if failures:
        # Failed
        verdict = "failed"
    elif gaps:
        # Not verified
        verdict = "not verified"
    else:
        # Passed
        verdict = "passed"
    # Say
    print(f"  verdict: {verdict}" + (f"; failures: {failures}" if failures else "") + (f"; not verified because: {gaps}" if gaps and not failures else ""))
    # Return
    return verdict


def overall(verdicts):
    """The worst verdict over the meters (failed beats not verified beats passed) and its exit code."""
    # Worst
    if "failed" in verdicts.values():
        # Failed
        return "failed", 1
    # Any unverified
    if "not verified" in verdicts.values():
        # Not verified
        return "not verified", 2
    # All passed, and at least one meter
    if verdicts and all(v == "passed" for v in verdicts.values()):
        # Passed
        return "passed", 0
    # Nothing inspected
    return "not verified", 2


def main():
    """Inspect the meters, print per-meter and overall verdicts, exit with the code."""
    # Options
    args = parse_args()
    # The late manifest, validated as the training script validates it
    body, _ = sa.read_late_manifest(args.features_dir, args.late_threshold)
    # Say
    print(f"late manifest: threshold {body['threshold_seconds']} s, rule: {body['rule']}")
    # Feature month
    manifest = sd.load_manifest(args.features_dir)
    # Load
    arrays = sd.load_month(args.features_dir, args.month, manifest=manifest)
    # Slot times
    t = arrays["t"]
    # Late file, with its own checks
    late = sa.load_late(args.features_dir, args.month, t, manifest["meters"])
    # Documented bounds of the month
    month_start, month_end = sd.month_bounds(args.month)
    # Say
    print(f"feature month {args.month}: {len(t)} slots, first {int(t[0])} s = {pd.Timestamp(int(t[0]), unit='s', tz='UTC')}, last {int(t[-1])} s = {pd.Timestamp(int(t[-1]), unit='s', tz='UTC')}; documented month bounds [{month_start}, {month_end}) s; expected feature coverage {'the documented month' if args.coverage_bounds is None else args.coverage_bounds}; late file shape {late.shape}, meters as the manifest")
    # Verdicts
    verdicts = {}
    # Each meter
    for meter in args.meters:
        # Column
        mi = manifest["meters"].index(meter)
        # Table and column
        path = os.path.join(args.processed_dir, args.month, f"{meter}.parquet")
        # Offset column
        col = f"{meter}{sa.SEP}{sa.TOFF}"
        # Inspect
        verdicts[meter] = inspect_meter(meter, mi, path, col, t, late, args.late_threshold, month_start, month_end, args.rows, args.coverage_bounds)
    # Overall
    verdict, code = overall(verdicts)
    # Per meter
    print("\nper meter: " + ", ".join(f"{m} {v}" for m, v in verdicts.items()))
    # Overall
    print(f"overall verdict: {verdict} (exit code {code}; only passed exits 0)")
    # Exit
    sys.exit(code)


# Entry point
if __name__ == "__main__":
    # Run
    main()
