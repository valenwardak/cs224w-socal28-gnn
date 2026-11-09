# =============================================================================
# File:        main_physicsdaily.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-08
#
# Purpose:     Read the physics baseline's results (the per-slot residual
#              tables main_physics.py writes) and print one line per day:
#              the median and 99th percentile residual in percent of
#              nominal, the largest residual, the number of slots solved,
#              the number of slots with no answer by status (no data, too
#              few readings, solver failed, undetermined under the rank
#              safeguard, rejected by the offset rule), and the median number of
#              voltage and current terms in the equations that day. The
#              counts stand beside the residual so that a day with fewer
#              readings is not read as a day with an electrical change,
#              and a day with many rejected slots is read as lost coverage,
#              not as a quiet day.
#              With --compare, two result folders holding the same block
#              (for example the cvxpy and numpy solvers) are read side by
#              side and the per-slot differences are summarized. Reading
#              refuses duplicate timestamps across the files (overlapping
#              saved windows would double-count) and slots marked ok
#              without a finite residual. Nothing is recomputed and
#              nothing is written.
#
#              Days are grouped in the project's fixed Pacific Standard
#              Time (UTC minus 8 hours, no daylight saving), as the block
#              labels and the meter clocks use.
#
# Usage (from the repo root):
#   python checks/main_physicsdaily.py --physics-dir data/processed/physics_numpy --prefix test_normal
#   python checks/main_physicsdaily.py --physics-dir data/processed/physics_numpy --prefix test_events --mark-events
#   python checks/main_physicsdaily.py --physics-dir data/processed/physics_compare_cvxpy --compare data/processed/physics_compare --prefix window
#
# Functions:
#   parse_args()                       command-line options with defaults
#   load_results(physics_dir, prefix)  every matching result table as one table sorted by time
#   pst_day(index)                     calendar day per slot in fixed PST
#   daily_table(res)                   one row per day with residual and count summaries
#   print_daily(table, events)         print the daily rows, marking days with logged events
#   compare(a, b)                      per-slot comparison of two result tables on their shared slots
#   main()                             read, summarize, print
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: file matching
import glob
# Standard library: paths
import os
# Third party: numerics
import numpy as np
# Third party: tables and timestamps
import pandas as pd


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Daily summary of the physics baseline's residuals, with measurement counts")
    # Result folder
    parser.add_argument("--physics-dir", default=os.path.join("data", "processed", "physics"), help="Folder with <label>.parquet residual tables")
    # Which files
    parser.add_argument("--prefix", default="test_normal", help="Only result files whose name starts with this")
    # Events
    parser.add_argument("--events", default=os.path.join("data", "processed", "events.csv"), help="Event table, used to mark days with logged changes")
    # Mark events
    parser.add_argument("--mark-events", action="store_true", help="Mark days that hold a scored logged change")
    # Second folder for a comparison
    parser.add_argument("--compare", default=None, help="A second result folder; the same prefix is read there and the two are compared slot by slot")
    # Parse
    return parser.parse_args()


def load_results(physics_dir, prefix):
    """Read every result table in the folder whose name starts with prefix, as one table sorted by time."""
    # Matching files
    paths = sorted(glob.glob(os.path.join(physics_dir, f"{prefix}*.parquet")))
    # None
    if not paths:
        # Refuse
        raise FileNotFoundError(f"no {prefix}*.parquet in {physics_dir}")
    # Read and join
    res = pd.concat([pd.read_parquet(p) for p in paths]).sort_index()
    # Say what was read
    print(f"read {len(paths)} file(s) from {physics_dir}: " + ", ".join(os.path.basename(p) for p in paths))
    # Index as UTC timestamps
    res.index = pd.to_datetime(res.index, utc=True)
    # Duplicate timestamps would double-count readings (overlapping saved windows)
    dup = int(res.index.duplicated().sum())
    # Refuse
    if dup:
        # Stop
        raise ValueError(f"{dup} duplicate timestamps across the files read from {physics_dir}; saved windows overlap")
    # A slot marked ok must carry a finite residual
    bad = int(((res["status"] == "ok") & ~np.isfinite(res["residual_mean"])).sum())
    # Refuse
    if bad:
        # Stop
        raise ValueError(f"{bad} slots marked ok have a non-finite residual in {physics_dir}; the results are inconsistent")
    # Done
    return res


def pst_day(index):
    """Calendar day of every slot in fixed Pacific Standard Time, UTC minus 8 hours."""
    # Shift to fixed PST, drop the zone, and take the date as text
    return np.asarray((pd.DatetimeIndex(index) - pd.Timedelta(hours=8)).tz_localize(None).strftime("%Y-%m-%d"))


def daily_table(res):
    """
    One row per day: slots, solved, slots by failure status, readings
    dropped by the per-meter offset policy (0 for other runs), median and
    99th percentile and max residual in percent (solved slots only), and
    the median voltage and current term counts over solved slots.
    """
    # Any status this reader does not list would vanish from the table; refuse rather than hide it
    unknown = set(res["status"].unique()) - {"ok", "no_data", "too_few", "solver_failed", "undetermined", "offset_rejected"}
    # Refuse
    if unknown:
        # Stop
        raise ValueError(f"unknown status values in the results: {sorted(unknown)}")
    # Day key
    day = pst_day(res.index)
    # Solved slots
    ok = res["status"] == "ok"
    # Residual in percent
    pct = 100.0 * res["residual_mean"]
    # Rows
    rows = []
    # Each day
    for d in sorted(set(day)):
        # Slots of the day
        sel = day == d
        # Solved that day
        s = sel & ok
        # Status counts
        st = res.loc[sel, "status"].value_counts()
        # Row
        rows.append({"day": d, "slots": int(sel.sum()), "ok": int(s.sum()),
                     "no_data": int(st.get("no_data", 0)), "too_few": int(st.get("too_few", 0)), "failed": int(st.get("solver_failed", 0)), "undetermined": int(st.get("undetermined", 0)), "offset": int(st.get("offset_rejected", 0)),
                     "dropped": int(res.loc[sel, "v_dropped"].sum() + res.loc[sel, "i_dropped"].sum()) if "v_dropped" in res.columns else 0,
                     "median_pct": float(pct[s].median()) if s.any() else float("nan"),
                     "p99_pct": float(pct[s].quantile(0.99)) if s.any() else float("nan"),
                     "max_pct": float(pct[s].max()) if s.any() else float("nan"),
                     "n_voltage": float(res.loc[s, "n_voltage"].median()) if s.any() else float("nan"),
                     "n_current": float(res.loc[s, "n_current"].median()) if s.any() else float("nan")})
    # Table
    return pd.DataFrame(rows)


def print_daily(table, event_days):
    """Print the daily rows; a day holding a scored logged change is marked with an asterisk."""
    # Header
    print(f"\n{'day':<12} {'slots':>6} {'ok':>6} {'no_data':>7} {'too_few':>7} {'failed':>6} {'undet.':>6} {'offset':>6} {'dropped':>7} {'median %':>9} {'p99 %':>8} {'max %':>8} {'nV':>5} {'nI':>5}  event")
    # Each row
    for _, r in table.iterrows():
        # Formatter for a possibly missing number
        f = lambda v, d=3: "   n/a" if not np.isfinite(v) else f"{v:.{d}f}"
        # Mark
        mark = "*" if r["day"] in event_days else ""
        # Line
        print(f"{r['day']:<12} {r['slots']:>6} {r['ok']:>6} {r['no_data']:>7} {r['too_few']:>7} {r['failed']:>6} {r['undetermined']:>6} {r['offset']:>6} {r['dropped']:>7} {f(r['median_pct']):>9} {f(r['p99_pct']):>8} {f(r['max_pct']):>8} {f(r['n_voltage'], 0):>5} {f(r['n_current'], 0):>5}  {mark}")
    # Whole-period summary over the days, with the slots lost to the offset rule
    print(f"\n{len(table)} days; median of daily medians {table['median_pct'].median():.3f} %, lowest daily median {table['median_pct'].min():.3f} %, highest daily median {table['median_pct'].max():.3f} %; solved slots {int(table['ok'].sum())} of {int(table['slots'].sum())}; offset-rejected slots {int(table['offset'].sum())}; readings dropped by the per-meter policy {int(table['dropped'].sum())}")


def compare(a, b):
    """Compare two result tables on their shared slots: status agreement and the per-slot residual difference in percent."""
    # Shared slots
    shared = a.index.intersection(b.index)
    # Say
    print(f"\ncomparison: {len(a)} slots in A, {len(b)} in B, {len(shared)} shared")
    # Both solved
    both = shared[(a.loc[shared, "status"] == "ok") & (b.loc[shared, "status"] == "ok")]
    # Status disagreement
    diff_status = int((a.loc[shared, "status"] != b.loc[shared, "status"]).sum())
    # Say
    print(f"  status differs on {diff_status} shared slots; both solved on {len(both)}")
    # Nothing to compare
    if len(both) == 0:
        # Stop
        return
    # Residuals in percent
    pa, pb = 100.0 * a.loc[both, "residual_mean"], 100.0 * b.loc[both, "residual_mean"]
    # Difference B minus A
    d = pb - pa
    # Summary
    print(f"  A median {pa.median():.3f} %, B median {pb.median():.3f} %; B minus A: median {d.median():+.4f}, 5th to 95th percentile {d.quantile(0.05):+.4f} to {d.quantile(0.95):+.4f}, max abs {d.abs().max():.4f} points")
    # Correlation
    print(f"  correlation of per-slot residuals {np.corrcoef(pa, pb)[0, 1]:.4f}")


def main():
    """Read the results, print the daily table, and compare two folders when asked."""
    # Options
    args = parse_args()
    # Results
    res = load_results(args.physics_dir, args.prefix)
    # Event days
    event_days = set()
    # Marking asked
    if args.mark_events:
        # Events
        ev = pd.read_csv(args.events)
        # Scored only
        ev = ev[ev["scored"].astype(bool)]
        # Days in fixed PST
        event_days = set(pst_day(pd.to_datetime(ev["t_utc"], utc=True, format="ISO8601")))
    # Daily table
    print_daily(daily_table(res), event_days)
    # Comparison
    if args.compare:
        # Second table
        other = load_results(args.compare, args.prefix)
        # Compare
        compare(res, other)


# Entry point
if __name__ == "__main__":
    # Run
    main()
