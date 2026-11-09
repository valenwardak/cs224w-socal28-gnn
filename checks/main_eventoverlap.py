# =============================================================================
# File:        main_eventoverlap.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-09
#
# Purpose:     Does the physics baseline's capture-time filter remove the
#              very slots that showed a logged event? On the November 13,
#              2024 A-side episode the unfiltered run showed a large
#              residual for the hours a switch was open, and the filtered
#              run rejects a fifth of that day. This check lines the two
#              runs up against the logged operations and counts, inside
#              and outside the open-switch interval of one named device
#              (its opening change on the chosen day up to, but excluding,
#              its next change, the closing; the log puts those in
#              separate episodes, so an episode is not the interval):
#              - slots rejected for timing, with the meter(s) that
#                triggered each rejection, taken from the processed
#                tables by the filter's own rule;
#              - slots unavailable for other reasons (undetermined, no
#                data, too few, solver failed);
#              - slots still solved, with the filtered and the unfiltered
#                residual compared at matching timestamps;
#              - the unfiltered residual at the rejected timestamps, which
#                says whether the removed slots were the ones carrying the
#                physics response.
#              It also prints the day in half-hour bins. Nothing is
#              written, nothing is rescored, no rule is changed.
#
# Usage (from the repo root):
#   python checks/main_eventoverlap.py --day 2024-11-13 --device swmp_17
#   python checks/main_eventoverlap.py --day 2024-11-13 --device cb_121 --filtered-dir data/processed/physics_offset05 --unfiltered-dir data/processed/physics_numpy --prefix test_events
#
# Functions:
#   parse_args()                       command-line options with defaults
#   pick_interval(ev, args)            the device's opening and closing changes on the day and the interval between
#   align_runs(filt, unf, days)        the two result tables restricted to the episode's days, same timestamps
#   trigger_flags(table, meters, threshold)
#                                      per-meter far flags per slot, the filter's rule
#   count_table(filt, unf, inside, far, meters)
#                                      the counts inside and outside the interval
#   print_bins(filt, unf, inside, minutes)
#                                      half-hour picture of the day(s)
#   main()                             read, align, count, print
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: paths
import os
# Standard library: import path for the repo root and the checks folder
import sys
# Third party: numerics
import numpy as np
# Third party: tables and timestamps
import pandas as pd

# Repo root on the path for the physics helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Checks folder on the path for the daily reader's loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Physics helpers
import socal_physics as sp
# The daily reader, for its result loader and fixed-PST day function
import main_physicsdaily as daily


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Filtered against unfiltered physics results around a logged episode")
    # Filtered run folder
    parser.add_argument("--filtered-dir", default=os.path.join("data", "processed", "physics_offset05"), help="Results folder of the run with the offset filter")
    # Unfiltered run folder
    parser.add_argument("--unfiltered-dir", default=os.path.join("data", "processed", "physics_numpy"), help="Results folder of the run without the filter")
    # File prefix
    parser.add_argument("--prefix", default="test_events", help="Result file prefix in both folders")
    # Events
    parser.add_argument("--events", default=os.path.join("data", "processed", "events.csv"), help="Event table from preprocessing")
    # Day
    parser.add_argument("--day", default="2024-11-13", help="Fixed-PST day of the device's opening change, like 2024-11-13")
    # Device
    parser.add_argument("--device", default="swmp_17", help="Device whose opening and closing changes define the interval")
    # Processed tables, for the trigger flags
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"), help="Folder with <YYYY-MM>/<meter>.parquet")
    # Topology and Caltech code, for the plan
    parser.add_argument("--topology-dir", default="topology", help="Caltech topology folder")
    # Caltech
    parser.add_argument("--caltech-dir", default="caltech_utils", help="Caltech utils folder")
    # Config, decides the planned meters
    parser.add_argument("--config", default="physics_config.json", help="Configuration of the filtered run; its planned meters are the filter's meters")
    # Threshold used by the filtered run
    parser.add_argument("--offset-threshold", type=float, default=0.5, help="Seconds; the threshold the filtered run used")
    # Bin width
    parser.add_argument("--bin-minutes", type=int, default=30, help="Width of the time bins in the day picture")
    # Parse
    return parser.parse_args()


def pick_interval(ev, args):
    """
    Find the device's first logged change on the chosen fixed-PST day (the
    opening) and its next logged change after that (the closing, whenever
    it is), and return both rows, the interval [opening, closing), and
    every other device's change inside that interval for information.
    The two boundary rows are printed by the caller so the interval can
    be verified against the log; other devices may close at other times
    and are not treated as the same.
    """
    # Timestamps
    ev["t_utc"] = pd.to_datetime(ev["t_utc"], utc=True, format="ISO8601")
    # Sorted
    ev = ev.sort_values("t_utc").reset_index(drop=True)
    # Day of every change
    d = daily.pst_day(ev["t_utc"])
    # The device's changes on the day
    on_day = ev[(ev["device"] == args.device) & (d == args.day)]
    # None
    if on_day.empty:
        # Refuse
        raise ValueError(f"{args.device} has no logged change on {args.day}")
    # Opening: its first change that day
    opening = on_day.iloc[0]
    # Closing: the device's next change after the opening, any day
    later = ev[(ev["device"] == args.device) & (ev["t_utc"] > opening["t_utc"])]
    # None
    if later.empty:
        # Refuse
        raise ValueError(f"{args.device} has no logged change after {opening['t_utc']}; no closing to bound the interval")
    # Closing
    closing = later.iloc[0]
    # Other devices' changes inside the interval
    others = ev[(ev["device"] != args.device) & (ev["t_utc"] >= opening["t_utc"]) & (ev["t_utc"] < closing["t_utc"])]
    # Return
    return opening, closing, others


def align_runs(filt, unf, days):
    """Restrict both result tables to the given fixed-PST days and require identical timestamps."""
    # Day of every slot
    fd, ud = daily.pst_day(filt.index), daily.pst_day(unf.index)
    # Restrict
    f = filt[np.isin(fd, days)]
    # Restrict
    u = unf[np.isin(ud, days)]
    # Same slots or stop
    if len(f) != len(u) or not f.index.equals(u.index):
        # Refuse
        raise ValueError(f"the two runs do not cover the same slots on {days}: {len(f)} against {len(u)}")
    # Return
    return f, u


def trigger_flags(table, meters, threshold):
    """Per-meter flags, one row per meter: True where that meter's offset is at or beyond the threshold, the filter's own rule."""
    # Flags
    far = np.zeros((len(meters), len(table)), dtype=bool)
    # Each meter
    for k, meter in enumerate(meters):
        # Column
        col = f"{meter}{sp.SEP}{sp.TOFF}"
        # Absent meter never triggers
        if col not in table.columns:
            # Next
            continue
        # Offsets
        toff = table[col].to_numpy(dtype=np.float64)
        # Far
        far[k] = np.isfinite(toff) & (np.abs(toff) >= threshold)
    # Return
    return far


def count_table(filt, unf, inside, far, meters):
    """
    Print the counts inside and outside the interval: filtered statuses,
    triggering meters for the rejected slots, the unfiltered residual at
    the rejected timestamps, and the two residuals on the slots both runs
    solved.
    """
    # Statuses present
    statuses = ["ok", "offset_rejected", "undetermined", "no_data", "too_few", "solver_failed"]
    # Header
    print(f"\n  filtered status                inside  outside")
    # Each status
    for st in statuses:
        # Mask
        m = (filt["status"] == st).to_numpy()
        # Line
        print(f"  {st:<28} {int((m & inside).sum()):>7}  {int((m & ~inside).sum()):>7}")
    # Rejected slots
    rej = (filt["status"] == "offset_rejected").to_numpy()
    # Triggering meters
    print(f"\n  meters at or beyond the threshold on rejected slots, far (only)   inside  outside")
    # Number of triggers per slot
    n_trig = far.sum(axis=0)
    # Each meter with any trigger
    for k, meter in enumerate(meters):
        # Far on rejected slots
        f_in, f_out = int((far[k] & rej & inside).sum()), int((far[k] & rej & ~inside).sum())
        # Only
        o_in, o_out = int((far[k] & rej & inside & (n_trig == 1)).sum()), int((far[k] & rej & ~inside & (n_trig == 1)).sum())
        # Skip meters that never triggered
        if f_in + f_out == 0:
            # Next
            continue
        # Line
        print(f"  {meter:<12} {f_in:>6} ({o_in:<5})  {f_out:>6} ({o_out:<5})")
    # Unfiltered residual at the rejected timestamps
    print(f"\n  unfiltered residual (percent per term) at the timestamps the filter rejected")
    # Inside and outside
    for name, sel in (("inside", rej & inside), ("outside", rej & ~inside)):
        # Values where the unfiltered run solved
        v = unf.loc[sel & (unf["status"] == "ok").to_numpy(), "residual_mean"].to_numpy() * 100.0
        # Say
        if v.size:
            # Line
            print(f"  {name:<8} {v.size:>6} solved by the unfiltered run: median {np.median(v):8.3f}  p90 {np.percentile(v, 90):8.3f}  max {v.max():8.3f}")
        # Nothing
        else:
            # Line
            print(f"  {name:<8}      0 solved by the unfiltered run")
    # Slots solved by both
    both = ((filt["status"] == "ok") & (unf["status"] == "ok")).to_numpy()
    # Compare
    print(f"\n  slots solved by both runs: filtered against unfiltered residual (percent per term)")
    # Inside and outside
    for name, sel in (("inside", both & inside), ("outside", both & ~inside)):
        # Values
        fv = filt.loc[sel, "residual_mean"].to_numpy() * 100.0
        # Values
        uv = unf.loc[sel, "residual_mean"].to_numpy() * 100.0
        # Say
        if fv.size:
            # Line
            print(f"  {name:<8} {fv.size:>6} slots: filtered median {np.median(fv):8.3f} p90 {np.percentile(fv, 90):8.3f}; unfiltered median {np.median(uv):8.3f} p90 {np.percentile(uv, 90):8.3f}; max |difference| {np.max(np.abs(fv - uv)):.3e}")
        # Nothing
        else:
            # Line
            print(f"  {name:<8}      0 slots")
    # Slots the unfiltered run solved that the filtered run did not, by filtered status
    lost = ((unf["status"] == "ok") & (filt["status"] != "ok")).to_numpy()
    # Say
    print(f"\n  slots solved by the unfiltered run but not by the filtered run: inside {int((lost & inside).sum())}, outside {int((lost & ~inside).sum())}")


def print_bins(filt, unf, inside, minutes):
    """Half-hour picture: per bin, slots, rejected, other unavailable, solved, unfiltered median, filtered median, and whether the bin lies in the interval."""
    # Bin start of every slot
    start = filt.index.floor(f"{minutes}min")
    # Header
    print(f"\n  {'bin start UTC':<22} {'slots':>5} {'rejected':>8} {'other':>5} {'solved':>6} {'unf. median %':>14} {'filt. median %':>15}  interval")
    # Each bin
    for b in sorted(set(start)):
        # Mask over all slots
        m = np.asarray(start == b)
        # Statuses over all slots
        st = filt["status"].to_numpy()
        # Counts
        n_rej, n_ok = int((m & (st == "offset_rejected")).sum()), int((m & (st == "ok")).sum())
        # Other unavailable
        n_oth = int(m.sum()) - n_rej - n_ok
        # Unfiltered median over its solved slots in the bin
        uv = unf.loc[m & (unf["status"] == "ok").to_numpy(), "residual_mean"]
        # Filtered median
        fv = filt.loc[m & (st == "ok"), "residual_mean"]
        # Text
        u_txt = f"{100 * uv.median():14.3f}" if len(uv) else f"{'n/a':>14}"
        # Text
        f_txt = f"{100 * fv.median():15.3f}" if len(fv) else f"{'n/a':>15}"
        # Interval share
        share = inside[m].mean()
        # Mark
        mark = "yes" if share == 1 else ("part" if share > 0 else "")
        # Line
        print(f"  {str(b)[:19]:<22} {int(m.sum()):>5} {n_rej:>8} {n_oth:>5} {n_ok:>6} {u_txt} {f_txt}  {mark}")


def main():
    """Read both runs and the events, line them up on the episode's days, count, print."""
    # Options
    args = parse_args()
    # Events
    ev = pd.read_csv(args.events)
    # Interval from the device's two changes
    opening, closing, others = pick_interval(ev, args)
    # Bounds
    t0, t1 = opening["t_utc"], closing["t_utc"]
    # Say
    print(f"{args.device}: opening {t0} UTC (episode {int(opening['episode'])}, {opening['from_status']} -> {opening['to_status']}, fixed PST {(t0 - pd.Timedelta(hours=8)).tz_localize(None)}); closing {t1} UTC (episode {int(closing['episode'])}, {closing['from_status']} -> {closing['to_status']}, fixed PST {(t1 - pd.Timedelta(hours=8)).tz_localize(None)}); interval [opening, closing) = {(t1 - t0).total_seconds() / 3600:.2f} h")
    # Other devices inside, for information
    if len(others):
        # Say
        print(f"  other devices logged inside the interval (not treated as the same interval): " + "; ".join(f"{r['t_utc']} {r['device']} {r['from_status']} -> {r['to_status']}" for _, r in others.iterrows()))
    # Days the interval touches
    days = sorted(set(daily.pst_day(pd.DatetimeIndex([t0, t1 - pd.Timedelta(seconds=1)]))))
    # Results
    filt = daily.load_results(args.filtered_dir, args.prefix)
    # Unfiltered
    unf = daily.load_results(args.unfiltered_dir, args.prefix)
    # Restrict and align
    filt, unf = align_runs(filt, unf, days)
    # Inside the interval: from the opening up to, but excluding, the closing
    inside = np.asarray((filt.index >= t0) & (filt.index < t1))
    # Say
    print(f"\ndays {days} (fixed PST): {len(filt)} slots, {int(inside.sum())} inside the interval, {int((~inside).sum())} outside")
    # Planned meters, for the triggers
    config = sp.load_config(args.config)
    # Caltech
    ct = sp.setup_caltech(args.caltech_dir, args.topology_dir)
    # Model
    model = sp.build_reference_model(ct, args.topology_dir, config)
    # Plan
    plan = sp.measurement_plan(model, config)
    # Meters
    meters = plan["meters"]
    # Processed tables over the same slots
    table = sp.load_block(args.processed_dir, meters, str(filt.index[0]), str(filt.index[-1] + pd.Timedelta(seconds=10)))
    # Same slots required
    if len(table) != len(filt) or not table.index.equals(filt.index):
        # Refuse
        raise ValueError("the processed tables do not cover the same slots as the results")
    # Flags
    far = trigger_flags(table, meters, args.offset_threshold)
    # The flags must reproduce the filtered run's rejections
    rej = (filt["status"] == "offset_rejected").to_numpy()
    # Compare
    if not np.array_equal(far.any(axis=0), rej):
        # Refuse
        raise ValueError(f"the trigger flags do not reproduce the filtered run's rejections ({int(far.any(axis=0).sum())} against {int(rej.sum())}); check --config and --offset-threshold")
    # Counts
    count_table(filt, unf, inside, far, meters)
    # Picture
    print_bins(filt, unf, inside, args.bin_minutes)


# Entry point
if __name__ == "__main__":
    # Run
    main()
