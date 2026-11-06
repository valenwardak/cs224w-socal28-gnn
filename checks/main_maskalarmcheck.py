# =============================================================================
# File:        main_maskalarmcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-05
#
# Purpose:     One-off check of where a run's quiet-block exceedances come
#              from, written for the masked model, whose quiet block was in
#              alarm 48 percent of the time in magnitude mode under thresholds
#              set on February 2025. The thresholds are read from the run's
#              score.json and never recomputed, so this looks at the frozen
#              procedure's own exceedances. For each mode (mag, ang) it prints:
#                1. the calibration block's exceedance rate per meter, which
#                   should sit near alpha by construction, as a sanity check
#                2. the test block by month: valid opportunities (a valid
#                   residual on a channel that has a threshold), exceedances,
#                   the rate, and the rate as a multiple of alpha
#                3. the test block by meter: the same, with the worst month
#                4. the test block by meter and by how many of that meter's
#                   graph neighbors were reporting in the slot, with the
#                   opportunity count beside every rate, so a rate from a
#                   handful of slots is not read as a finding
#                5. the ten days with the most exceedances
#              A neighbor is counted in a slot when it has a valid persistence
#              magnitude residual there (a valid target, a usable history, and
#              a previous valid value), read before any common mask, so a
#              meter whose own masked residual is invalid (egauge_24) still
#              counts as a neighbor; this is a proxy for the neighbor input
#              the encoder saw, not a measure of it. Neighbors come from the
#              graph file's edges and are listed by name once. Opportunities
#              and exceedances are counted under the scorer's common mask
#              when the score was made with --common, so they match what was
#              scored. Rates are exceedance rates per channel-slot, not alarm
#              rates: the alarm needs two meters in the same slot, which this
#              does not reproduce. The calibration rate may sit below alpha
#              because of ties and finite-sample ranks. Nothing here changes
#              any threshold.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root.
#
# Usage:
#   python checks/main_maskalarmcheck.py --run-dir /content/drive/MyDrive/socal_runs/gcn_masked_aw001 --graph graph_meters.json
#   python checks/main_maskalarmcheck.py --run-dir ... --modes mag --csv mask_alarms.csv
#
# Functions:
#   parse_args()                 command-line options with defaults
#   load_thresholds(path)        thresholds, axes, and settings from score.json
#   exceed_arrays(arrays, thr)   opportunity and exceedance arrays (T, M, C) for one mode
#   neighbor_counts(graph, meters, present)
#                                reporting neighbors per slot and meter, (T, M)
#   month_labels(t)              YYYY-MM per slot in the files' fixed-PST convention
#   rate_line(name, opp, exc, alpha)
#                                one formatted line with the count beside the rate
#   main()                       run the report
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON reading
import json
# Standard library: repo root on the import path
import os
import sys
# Third party: numerics
import numpy as np
# Third party: timestamps and the CSV
import pandas as pd

# The repo root, one level above checks/, so the project modules import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project: the scoring module's loaders, so the arrays match what was scored
import socal_score as ss

# Hours added to standard time to get UTC, the files' convention
UTC_OFFSET_HOURS = 8


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Where a run's quiet-block exceedances come from, under its frozen thresholds")
    # Run folder
    parser.add_argument("--run-dir", required=True, help="Run folder holding the residual tables and score.json")
    # Score file with the thresholds
    parser.add_argument("--score", default="score.json", help="Score file in the run folder, for the thresholds")
    # Calibration block
    parser.add_argument("--calib-label", default="validation_2025-02-03_2025-02-16", help="Block the thresholds were set on")
    # Test block
    parser.add_argument("--test-label", default="test_normal_2025-04-01_2025-08-31", help="Block to examine")
    # Graph for the neighbors
    parser.add_argument("--graph", default="graph_meters.json", help="Graph file whose edges define the neighbors")
    # Modes
    parser.add_argument("--modes", nargs="+", default=["mag", "ang"], help="Residual kinds to examine")
    # Days to list
    parser.add_argument("--top-days", type=int, default=10, help="Days with the most exceedances to list")
    # CSV
    parser.add_argument("--csv", default=None, help="Write per channel and month rows to this CSV as well")
    # Parse
    return parser.parse_args()


def load_thresholds(path):
    """
    Thresholds from score.json: meters, channels, and (M, C) arrays for mag
    and ang, plus alpha. The thresholds are per channel and do not depend
    on the alarm mode, so they are read from the first mode's model entry.
    """
    # Read
    with open(path) as f:
        # Parse
        score = json.load(f)
    # First mode's model result holds the thresholds
    first = next(iter(score["modes"].values()))
    # Threshold block
    thr = first["model"]["thresholds"]
    # Arrays, NaN where no threshold
    out = {k: np.array(thr[k], dtype=np.float64) for k in ("mag", "ang")}
    # Settings the score was made with
    settings = score["settings"]
    # Done
    return thr["meters"], thr["channels"], out, settings


def exceed_arrays(arrays, thr, kind):
    """Opportunity (valid residual with a threshold) and exceedance (residual above it) arrays, (T, M, C) bool."""
    # Residuals
    r = arrays[kind]
    # Valid with a threshold
    opp = arrays[f"{kind}_valid"] & np.isfinite(thr)[None, :, :]
    # Above the threshold, only where it is an opportunity
    exc = opp & (np.nan_to_num(r, nan=-np.inf) > thr[None, :, :])
    # Done
    return opp, exc


def neighbor_counts(graph, meters, present):
    """
    Reporting neighbors per slot and meter, (T, M) int. A neighbor is a
    meter on a node joined to this meter's node by an edge in the graph
    file; it is reporting in a slot when present[t, neighbor] is True.
    """
    # Node per meter name
    node_of = {m: int(rec["node"]) for m, rec in graph["meters"].items()}
    # Nodes joined to each node
    joined = {}
    # Loop over edges
    for e in graph["edges"]:
        # Both ways
        joined.setdefault(e["source"], set()).add(e["target"])
        joined.setdefault(e["target"], set()).add(e["source"])
    # Neighbor indices per meter
    nbrs = []
    # Loop over meters in table order
    for m in meters:
        # Nodes reached
        reach = joined.get(node_of.get(m), set())
        # Meters on them
        nbrs.append([j for j, other in enumerate(meters) if other != m and node_of.get(other) in reach])
    # Counts
    counts = np.zeros(present.shape, dtype=np.int64)
    # Loop
    for mi, js in enumerate(nbrs):
        # Sum of reporting neighbors
        if js:
            # Add
            counts[:, mi] = present[:, js].sum(axis=1)
    # Done
    return counts, nbrs


def month_labels(t):
    """YYYY-MM per slot, in the fixed-PST convention of the feature files."""
    # Shift to standard time and take the month
    return (pd.DatetimeIndex(t) - pd.Timedelta(hours=UTC_OFFSET_HOURS)).strftime("%Y-%m").to_numpy()


def rate_line(name, opp, exc, alpha, width=28):
    """One formatted line: name, opportunities, exceedances, rate, and the rate as a multiple of alpha."""
    # Rate
    rate = exc / opp if opp else float("nan")
    # Multiple
    mult = rate / alpha if opp else float("nan")
    # Line
    return f"  {name:{width}s} {opp:12d} {exc:10d} {rate:10.2e} {mult:10.1f}x"


def main():
    """Run the report."""
    # Options
    args = parse_args()
    # Thresholds, axes, and the settings behind them
    meters, channels, thr, settings = load_thresholds(os.path.join(args.run_dir, args.score))
    # Alpha
    alpha = float(settings["alpha"])
    # The score must be the channel statistic, whose thresholds these are
    if settings.get("statistic", "channel") != "channel":
        # Raise
        raise ValueError(f"{args.score} was made with statistic {settings['statistic']}, not channel; its thresholds are not per channel")
    # The calibration block must be the one the thresholds came from
    if settings["calib_label"] != args.calib_label:
        # Raise
        raise ValueError(f"--calib-label {args.calib_label} differs from the score's calibration block {settings['calib_label']}")
    # Graph
    with open(args.graph) as f:
        # Parse
        graph = json.load(f)
    # Tables
    calib = ss.load_table(args.run_dir, args.calib_label)
    # Test
    test = ss.load_table(args.run_dir, args.test_label)
    # Axes must match the score file
    for table, name in ((calib, args.calib_label), (test, args.test_label)):
        # Compare
        tm, tc = ss.table_axes(table)
        # Mismatch
        if tm != meters or tc != channels:
            # Raise
            raise ValueError(f"{name}: table axes differ from {args.score}")
    # Arrays, model
    a_cal = ss.residual_arrays(calib, meters, channels)
    # Test
    a_test = ss.residual_arrays(test, meters, channels)
    # Persistence arrays, for the neighbor proxy and the common mask
    p_cal = ss.residual_arrays(calib, meters, channels, prefix="pers_")
    # Test
    p_test = ss.residual_arrays(test, meters, channels, prefix="pers_")
    # Neighbor proxy, read before any masking: persistence magnitude residual valid on any channel
    present = p_test["mag_valid"].any(axis=2)
    # The scorer's common mask, when the score used it, so opportunities match what was scored
    if settings.get("common", True):
        # Calibration
        ss.common_mask(a_cal, p_cal)
        # Test
        ss.common_mask(a_test, p_test)
    # Reporting neighbors
    counts, nbrs = neighbor_counts(graph, meters, present)
    # Months
    months = month_labels(a_test["t"])
    # Days, for the top list
    days = (pd.DatetimeIndex(a_test["t"]) - pd.Timedelta(hours=UTC_OFFSET_HOURS)).strftime("%Y-%m-%d").to_numpy()
    # CSV rows
    rows = []
    # Header
    print(f"run {args.run_dir}: thresholds from {args.score} (alpha {alpha:g}, calibration {args.calib_label}), examining {args.test_label}")
    # Common mask
    print(f"opportunities counted under the scorer's common mask: {bool(settings.get('common', True))}")
    # Neighbor table once, names and degree
    print(f"neighbors from {args.graph} (a neighbor counts in a slot when its persistence magnitude residual is valid there):")
    # Loop
    for m, js in zip(meters, nbrs):
        # Line
        print(f"  {m:12s} [{len(js)}] {', '.join(meters[j] for j in js) or '-'}")
    # Loop over modes
    for kind in args.modes:
        # Arrays
        opp_c, exc_c = exceed_arrays(a_cal, thr[kind], kind)
        # Test
        opp_t, exc_t = exceed_arrays(a_test, thr[kind], kind)
        # Column header
        head = f"  {'':28s} {'opportunities':>12s} {'exceed':>10s} {'rate':>10s} {'x alpha':>11s}"

        # 1. calibration sanity
        print(f"\n=== mode {kind}: 1. calibration block {args.calib_label}, exceedance rate per meter (near alpha, possibly below it from ties and finite-sample ranks) ===")
        # Header
        print(head)
        # Loop over meters
        for mi, m in enumerate(meters):
            # Counts
            o, e = int(opp_c[:, mi].sum()), int(exc_c[:, mi].sum())
            # Only meters with opportunities
            if o:
                # Line
                print(rate_line(m, o, e, alpha))
        # Pooled
        print(rate_line("all channels", int(opp_c.sum()), int(exc_c.sum()), alpha))

        # 2. test block by month
        print(f"\n=== mode {kind}: 2. {args.test_label} by month ===")
        # Header
        print(head)
        # Loop over months
        for month in sorted(set(months)):
            # Rows of the month
            sel = months == month
            # Counts
            o, e = int(opp_t[sel].sum()), int(exc_t[sel].sum())
            # Line
            print(rate_line(month, o, e, alpha))
            # CSV rows per channel
            for mi, m in enumerate(meters):
                # Channels
                for ci, ch in enumerate(channels):
                    # Counts
                    oc, ec = int(opp_t[sel, mi, ci].sum()), int(exc_t[sel, mi, ci].sum())
                    # Row
                    rows.append({"mode": kind, "month": month, "meter": m, "channel": ch, "opportunities": oc, "exceedances": ec, "rate": ec / oc if oc else np.nan})
        # Pooled
        print(rate_line("whole block", int(opp_t.sum()), int(exc_t.sum()), alpha))

        # 3. test block by meter, with the worst month
        print(f"\n=== mode {kind}: 3. {args.test_label} by meter ===")
        # Header
        print(head + "   worst month (rate, opportunities)")
        # Loop over meters
        for mi, m in enumerate(meters):
            # Counts
            o, e = int(opp_t[:, mi].sum()), int(exc_t[:, mi].sum())
            # Skip meters with nothing
            if not o:
                # Next
                continue
            # Worst month by rate among months with opportunities
            worst, worst_rate, worst_opp = None, -1.0, 0
            # Loop
            for month in sorted(set(months)):
                # Rows
                sel = months == month
                # Counts
                om, em = int(opp_t[sel, mi].sum()), int(exc_t[sel, mi].sum())
                # Better
                if om and em / om > worst_rate:
                    # Keep
                    worst, worst_rate, worst_opp = month, em / om, om
            # Line
            print(rate_line(m, o, e, alpha) + f"   {worst} ({worst_rate:.2e}, {worst_opp})")

        # 4. by meter and reporting neighbors
        print(f"\n=== mode {kind}: 4. {args.test_label} by meter and number of neighbors with a valid persistence magnitude residual in the slot ===")
        # Header
        print(f"  {'meter':12s} " + " ".join(f"{'n=' + str(k):>22s}" for k in range(0, int(counts.max()) + 1)))
        # Loop over meters
        for mi, m in enumerate(meters):
            # Skip meters with no neighbors or no opportunities
            if not nbrs[mi] or not opp_t[:, mi].any():
                # Next
                continue
            # Cells
            cells = []
            # Loop over neighbor counts
            for k in range(0, int(counts.max()) + 1):
                # Slots with k reporting neighbors
                sel = counts[:, mi] == k
                # Counts
                o, e = int(opp_t[sel, mi].sum()), int(exc_t[sel, mi].sum())
                # Cell: rate and opportunities, or blank
                cells.append(f"{e / o:9.2e} ({o:9d})" if o else f"{'':22s}")
            # Line
            print(f"  {m:12s} " + " ".join(cells))
        # Pooled over meters
        cells = []
        # Loop
        for k in range(0, int(counts.max()) + 1):
            # Slots
            sel = counts == k
            # Counts over channels
            o, e = int((opp_t & sel[:, :, None]).sum()), int((exc_t & sel[:, :, None]).sum())
            # Cell
            cells.append(f"{e / o:9.2e} ({o:9d})" if o else f"{'':22s}")
        # Line
        print(f"  {'all meters':12s} " + " ".join(cells))

        # 5. top days
        print(f"\n=== mode {kind}: 5. {args.top_days} days with the most exceedances ===")
        # Exceedances per slot
        per_slot = exc_t.sum(axis=(1, 2))
        # Per day
        per_day = pd.Series(per_slot).groupby(days).sum().sort_values(ascending=False)
        # Opportunities per day
        opp_day = pd.Series(opp_t.sum(axis=(1, 2))).groupby(days).sum()
        # Header
        print(head)
        # Loop
        for day, e in per_day.head(args.top_days).items():
            # Line
            print(rate_line(day, int(opp_day[day]), int(e), alpha))
    # CSV
    if args.csv:
        # Write
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        # Say so
        print(f"\nwrote {args.csv}: {len(rows)} rows")


# Run when called as a script
if __name__ == "__main__":
    # Go
    main()
