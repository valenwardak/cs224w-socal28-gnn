# =============================================================================
# File:        main_alarmcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-28
#
# Purpose:     One-off diagnostic of where a run's alarms come from on one
#              test block, usually the quiet block, with the same settings
#              the scoring uses. The first scoring run produced tens of
#              unmatched alarms per day; this check asks why. It prints:
#                1. calibration: residual counts and thresholds per channel,
#                   channels without a threshold
#                2. exceedance rates per channel on the block, as exceedances
#                   divided by thresholded valid opportunities, against the
#                   alpha they were calibrated to; overall, by channel (worst
#                   first), and by month
#                3. active meters: how many meters hold a thresholded valid
#                   residual per slot, by month (mean and minimum)
#                4. alarm structure: how many meters fire together in alarm
#                   slots, alarm episode durations, alarm-slot fraction
#                5. alarms per day (busiest days) and by local hour of day
#                   (America/Los_Angeles)
#                6. a few alarm traces: the first alarm episodes printed
#                   slot by slot with the firing meters and channels and
#                   their residual-to-threshold ratios
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root. It reuses the
#              scoring functions from socal_score.py, and it reads the
#              settings saved in the run's score.json and refuses to run
#              when its own options differ from them, so the thresholds and
#              alarms are exactly the ones the scoring computed. The block
#              inspected is the quiet block: the scorer's partial-event-
#              window exclusions are not reproduced here, so an event block
#              is only accepted with --allow-events, and then the alarm
#              counts can differ from the scoring's.
#
# Usage:
#   python checks/main_alarmcheck.py --run-dir data/processed/models/gru_full_aw001
#   python checks/main_alarmcheck.py --run-dir /content/drive/MyDrive/socal_runs/gcn_full_aw001 --mode ang --traces 5
#   python checks/main_alarmcheck.py --run-dir ... --label test_events_2024-11-11_2024-11-15 --allow-events
#
# Functions:
#   parse_args()                 command-line options with defaults
#   section(title)               print a section header
#   check_settings(args)         compare the options with the run's score.json settings
#   rate_table(arr, thr, kind)   exceedance rates per channel for one kind
#   main()                       run the six parts
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON for the saved settings
import json
# Standard library: paths
import os
# Standard library: so the project module is found when run from the repo root
import sys
# Third party: numerics
import numpy as np
# Third party: tables and timestamps
import pandas as pd

# Repo root on the path, this file lives in checks/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Project: the scoring method
import socal_score as ss


def parse_args():
    """Read the command-line options. Every setting has a default matching the scoring."""
    # Parser
    parser = argparse.ArgumentParser(description="Where do a run's alarms come from on one test block")
    # Run folder
    parser.add_argument("--run-dir", required=True, help="Run folder with the residual tables")
    # Block to inspect
    parser.add_argument("--label", default="test_normal_2025-04-01_2025-08-31", help="Block label to inspect")
    # Calibration block
    parser.add_argument("--calib-label", default="validation_2025-02-03_2025-02-16", help="Calibration block, as in the scoring")
    # Source
    parser.add_argument("--source", choices=["model", "persistence"], default="model", help="Whose residuals to inspect")
    # Mode
    parser.add_argument("--mode", choices=["mag", "ang", "both"], default="both", help="Which residuals fire a channel")
    # Alpha
    parser.add_argument("--alpha", type=float, default=1e-4, help="Conformal level, as in the scoring")
    # Minimum meters
    parser.add_argument("--min-meters", type=int, default=2, help="Meters firing for an alarm slot, as in the scoring")
    # Merge gap
    parser.add_argument("--merge-slots", type=int, default=6, help="Merge gap in slots, as in the scoring")
    # Common slots
    parser.add_argument("--common", action=argparse.BooleanOptionalAction, default=True, help="Mask to slots valid for both model and persistence, as in the scoring")
    # Traces
    parser.add_argument("--traces", type=int, default=3, help="How many alarm episodes to print slot by slot")
    # Context slots around a trace
    parser.add_argument("--context", type=int, default=2, help="Slots before and after an episode to include in its trace")
    # Worst channels to list
    parser.add_argument("--top", type=int, default=15, help="How many channels to list in the worst-first tables")
    # Local time zone for the hour-of-day table
    parser.add_argument("--tz", default="America/Los_Angeles", help="Local time zone for the hour-of-day counts")
    # Saved settings
    parser.add_argument("--score", default="score.json", help="The scoring output in the run folder whose settings this check must match")
    # Event blocks
    parser.add_argument("--allow-events", action="store_true", help="Allow an event block, whose partial-window exclusions this check does not reproduce")
    # Parse
    return parser.parse_args()


def check_settings(args):
    """
    Read the run's score.json and compare its settings with this check's
    options. Raises when they differ, so the diagnosis matches the result.
    Returns the saved settings.
    """
    # Path
    path = os.path.join(args.run_dir, args.score)
    # Must exist
    if not os.path.exists(path):
        # Raise
        raise FileNotFoundError(f"no {args.score} in {args.run_dir}; run main_score.py first")
    # Read
    with open(path) as f:
        # Parse
        saved = json.load(f)["settings"]
    # Pairs to compare: option name, saved key
    pairs = (("alpha", "alpha"), ("min_meters", "min_meters"), ("merge_slots", "merge_slots"), ("calib_label", "calib_label"), ("common", "common"))
    # Differences
    diffs = {k: (saved.get(k), getattr(args, a)) for a, k in pairs if saved.get(k) != getattr(args, a)}
    # Any difference is a different experiment
    if diffs:
        # Raise
        raise ValueError(f"options differ from {path} (saved, given): {diffs}")
    # Return
    return saved


def section(title):
    """Print a section header."""
    # Blank line, title, underline
    print(f"\n=== {title} ===")


def rate_table(arr, thr, kind):
    """
    Exceedance rate per channel for one kind: exceedances divided by
    valid slots on channels that have a threshold. Returns a DataFrame
    with meter, channel, n (opportunities), exceed, rate, threshold.
    """
    # Opportunities: valid residual and a threshold
    opp = arr[f"{kind}_valid"] & np.isfinite(thr[f"thr_{kind}"])[None, :, :]
    # Exceedances
    exc = opp & (np.nan_to_num(arr[kind], nan=-np.inf) > np.nan_to_num(thr[f"thr_{kind}"], nan=np.inf)[None, :, :])
    # Counts per channel
    n, e = opp.sum(axis=0), exc.sum(axis=0)
    # Rows
    rows = []
    # Loop over channels
    for mi, meter in enumerate(arr["meters"]):
        # Loop
        for ci, ch in enumerate(arr["channels"]):
            # Row
            rows.append({"meter": meter, "channel": ch, "n": int(n[mi, ci]), "exceed": int(e[mi, ci]), "rate": (e[mi, ci] / n[mi, ci]) if n[mi, ci] else np.nan, "threshold": float(thr[f"thr_{kind}"][mi, ci])})
    # Table
    return pd.DataFrame(rows)


def main():
    """Run the six parts."""
    # Options
    args = parse_args()
    # Settings must match the scoring being diagnosed
    check_settings(args)
    # Event blocks need the flag
    if not args.label.startswith("test_normal") and not args.allow_events:
        # Raise
        raise ValueError(f"{args.label} is not a quiet block; the scorer's partial-event-window exclusions are not reproduced here. Use --allow-events to proceed anyway.")
    # Source prefix and the other one
    prefix, other = ("", "pers_") if args.source == "model" else ("pers_", "")
    # Calibration table
    cal = ss.load_table(args.run_dir, args.calib_label)
    # Axes
    meters, channels = ss.table_axes(cal)
    # Calibration residuals
    carr = ss.residual_arrays(cal, meters, channels, prefix)
    # Common slots
    if args.common:
        # Mask
        ss.common_mask(carr, ss.residual_arrays(cal, meters, channels, other))
    # Thresholds
    thr = ss.calibrate(carr, args.alpha)
    # Block table
    table = ss.load_table(args.run_dir, args.label)
    # Residuals
    arr = ss.residual_arrays(table, meters, channels, prefix)
    # Common slots
    if args.common:
        # Mask
        ss.common_mask(arr, ss.residual_arrays(table, meters, channels, other))
    # Axes kept with the arrays for the tables
    arr["meters"], arr["channels"] = meters, channels
    # Timestamps
    t = pd.to_datetime(arr["t"], utc=True)
    # Header
    print(f"{args.source} residuals, block {args.label}: {len(t)} slots from {t[0]} to {t[-1]}; mode {args.mode}, alpha {args.alpha}, min_meters {args.min_meters}")

    # 1. Calibration
    section("1. calibration: counts and thresholds per channel")
    # Header
    print(f"{'meter':<10} {'ch':<3} {'n_mag':>8} {'thr_mag':>9} {'n_ang':>8} {'thr_ang':>9}")
    # Rows
    for mi, meter in enumerate(meters):
        # Loop
        for ci, ch in enumerate(channels):
            # Print
            print(f"{meter:<10} {ch:<3} {thr['n_mag'][mi, ci]:>8} {thr['thr_mag'][mi, ci]:>9.5f} {thr['n_ang'][mi, ci]:>8} {thr['thr_ang'][mi, ci]:>9.3f}")
    # Needed count for this alpha
    print(f"\nminimum calibration count for alpha {args.alpha}: {int(np.ceil(1.0 / args.alpha)) - 1} (k = ceil((n+1)(1-alpha)) must be <= n)")
    # Missing
    for kind in ("mag", "ang"):
        # Channels without a threshold
        miss = [f"{meters[mi]}|{channels[ci]}" for mi, ci in zip(*np.nonzero(~np.isfinite(thr[f"thr_{kind}"])))]
        # Print
        print(f"no {kind} threshold ({len(miss)}): {miss}")

    # 2. Exceedance rates
    kinds = ["mag", "ang"] if args.mode == "both" else [args.mode]
    # Loop over kinds
    for kind in kinds:
        # Table
        df = rate_table(arr, thr, kind)
        # Overall
        tot_n, tot_e = df["n"].sum(), df["exceed"].sum()
        # Section
        section(f"2. {kind} exceedance rate on the block, calibrated to {args.alpha}")
        # Overall line
        print(f"overall: {tot_e} exceedances over {tot_n} opportunities = {tot_e / max(tot_n, 1):.2e} ({(tot_e / max(tot_n, 1)) / args.alpha:.1f} x alpha); expected at alpha: {tot_n * args.alpha:.0f}")
        # Worst channels
        print(f"\nworst {args.top} channels by rate:")
        # Header
        print(f"{'meter':<10} {'ch':<3} {'n':>9} {'exceed':>7} {'rate':>10} {'x alpha':>8} {'threshold':>10}")
        # Rows
        for _, r in df.sort_values("rate", ascending=False).head(args.top).iterrows():
            # Skip empty
            if r["n"] == 0:
                # Next
                continue
            # Print
            print(f"{r['meter']:<10} {r['channel']:<3} {r['n']:>9} {r['exceed']:>7} {r['rate']:>10.2e} {r['rate'] / args.alpha:>8.1f} {r['threshold']:>10.5f}")
        # By month
        print(f"\nby month:")
        # Opportunities and exceedances per slot
        opp = arr[f"{kind}_valid"] & np.isfinite(thr[f"thr_{kind}"])[None, :, :]
        # Exceedances
        exc = opp & (np.nan_to_num(arr[kind], nan=-np.inf) > np.nan_to_num(thr[f"thr_{kind}"], nan=np.inf)[None, :, :])
        # Month labels, from naive UTC so pandas does not warn about the zone
        months = t.tz_convert(None).to_period("M")
        # Loop over months
        for m in months.unique():
            # Slots of this month
            sel = (months == m)
            # Counts
            n_m, e_m = int(opp[sel].sum()), int(exc[sel].sum())
            # Print
            print(f"  {m}: {e_m} / {n_m} = {e_m / max(n_m, 1):.2e} ({(e_m / max(n_m, 1)) / args.alpha:.1f} x alpha)")

    # 3. Active meters
    section("3. active meters per slot (thresholded valid residual on any channel), by month")
    # Watched channels by mode
    if args.mode == "mag":
        # Magnitude
        watched = arr["mag_valid"] & np.isfinite(thr["thr_mag"])[None, :, :]
    elif args.mode == "ang":
        # Angle
        watched = arr["ang_valid"] & np.isfinite(thr["thr_ang"])[None, :, :]
    else:
        # Either
        watched = (arr["mag_valid"] & np.isfinite(thr["thr_mag"])[None, :, :]) | (arr["ang_valid"] & np.isfinite(thr["thr_ang"])[None, :, :])
    # Meters active per slot
    active = watched.any(axis=2).sum(axis=1)
    # Month labels, from naive UTC
    months = t.tz_convert(None).to_period("M")
    # Loop over months
    for m in months.unique():
        # Slots
        sel = (months == m)
        # Print
        print(f"  {m}: mean {active[sel].mean():.1f}, min {active[sel].min()}, max {active[sel].max()} of {len(meters)} meters; slots with < {args.min_meters}: {int((active[sel] < args.min_meters).sum())}")

    # 4. Alarm structure
    section("4. alarm structure")
    # Firing channels
    fires = ss.exceedances(arr, thr, args.mode)
    # Meters firing per slot
    score = ss.slot_scores(fires)
    # Monitored
    mon = ss.monitored_slots(arr, thr, args.mode, args.min_meters)
    # Alarm slots
    alarm = (score >= args.min_meters) & mon
    # Episodes
    eps = ss.alarm_episodes(arr["t"], alarm, args.merge_slots)
    # Fractions
    print(f"monitored slots {int(mon.sum())} of {len(t)}; alarm slots {int(alarm.sum())} ({alarm.sum() / max(mon.sum(), 1):.2e} of monitored); alarm episodes {len(eps)}")
    # Slots with exactly one meter firing, which the rule suppresses, printed whether or not there are alarms
    print(f"slots with exactly 1 meter firing (suppressed): {int(((score == 1) & mon).sum())}; slots with any meter firing: {int(((score >= 1) & mon).sum())}")
    # Meters firing together in alarm slots
    if alarm.any():
        # Distribution
        vals, cnts = np.unique(score[alarm], return_counts=True)
        # Print
        print("meters firing together in alarm slots: " + ", ".join(f"{v}: {c}" for v, c in zip(vals, cnts)))
    # Durations
    if eps:
        # Seconds
        dur = np.array([(e - s).total_seconds() + ss.SLOT_SECONDS for s, e in eps])
        # Print
        print(f"episode duration seconds: median {np.median(dur):.0f}, p90 {np.percentile(dur, 90):.0f}, max {dur.max():.0f}; single-slot episodes {int((dur <= ss.SLOT_SECONDS).sum())}")

    # 5. Alarms per day and by local hour
    section("5. alarm episodes per day (busiest 10) and by local hour of day")
    # Monitored days from monitored slots
    days = mon.sum() * ss.SLOT_SECONDS / 86400.0
    # Local dates of every slot
    local_dates = pd.Series(t.tz_convert(args.tz).date)
    # Monitored exposure per local date, in hours, only dates with monitored slots
    exposure = local_dates[mon].value_counts().sort_index() * ss.SLOT_SECONDS / 3600.0
    # Rate line, printed whether or not there are alarms
    print(f"{len(eps)} episodes over {days:.1f} monitored days = {len(eps) / max(days, 1e-9):.2f} per day; {len(exposure)} local dates with monitored slots, exposure hours median {exposure.median():.1f}, min {exposure.min():.1f}" if len(exposure) else "no monitored slots")
    # Start times
    starts = pd.DatetimeIndex([s for s, _ in eps])
    # Per date, zero on monitored dates without an episode
    per_day = pd.Series(1, index=starts).groupby(starts.tz_convert(args.tz).date).sum() if len(eps) else pd.Series(dtype=int)
    # Reindex to the monitored dates
    per_day = per_day.reindex(exposure.index, fill_value=0)
    # Busiest, with exposure
    print("busiest days (episodes, monitored hours): " + ", ".join(f"{d} ({int(n)}, {exposure[d]:.1f}h)" for d, n in per_day.sort_values(ascending=False).head(10).items()))
    # Days with none, among monitored dates
    print(f"monitored dates with no episode: {int((per_day == 0).sum())} of {len(per_day)}")
    # By hour
    by_hour = pd.Series(1, index=starts).groupby(starts.tz_convert(args.tz).hour).sum() if len(eps) else pd.Series(dtype=int)
    # Print
    print("by local hour: " + ", ".join(f"{h:02d}h {int(by_hour.get(h, 0))}" for h in range(24)))

    # 6. Traces
    section(f"6. first {args.traces} alarm episodes, slot by slot")
    # Loop over the first episodes
    for s, e in eps[:args.traces]:
        # Slot index range with context
        i0 = max(int(np.searchsorted(t, s)) - args.context, 0)
        # End
        i1 = min(int(np.searchsorted(t, e)) + args.context + 1, len(t))
        # Header
        print(f"\nepisode {s} to {e} ({t[i0:i1].tz_convert(args.tz)[0].strftime('%Y-%m-%d %H:%M')} local)")
        # Loop over slots
        for i in range(i0, i1):
            # Firing channels in this slot with their ratios
            parts = []
            # Loop over meters and channels
            for mi, meter in enumerate(meters):
                # Loop
                for ci, ch in enumerate(channels):
                    # Fired
                    if fires[i, mi, ci]:
                        # Ratios for the kinds in the mode
                        r = []
                        # Magnitude
                        if args.mode in ("mag", "both") and arr["mag_valid"][i, mi, ci] and np.isfinite(thr["thr_mag"][mi, ci]) and arr["mag"][i, mi, ci] > thr["thr_mag"][mi, ci]:
                            # Ratio
                            r.append(f"mag {arr['mag'][i, mi, ci] / thr['thr_mag'][mi, ci]:.1f}x")
                        # Angle
                        if args.mode in ("ang", "both") and arr["ang_valid"][i, mi, ci] and np.isfinite(thr["thr_ang"][mi, ci]) and arr["ang"][i, mi, ci] > thr["thr_ang"][mi, ci]:
                            # Ratio
                            r.append(f"ang {arr['ang'][i, mi, ci] / thr['thr_ang'][mi, ci]:.1f}x")
                        # Keep
                        parts.append(f"{meter.replace('egauge_', 'e')}|{ch} ({', '.join(r)})")
            # Marker
            mark = "ALARM" if alarm[i] else ("fire " if score[i] else "     ")
            # Print
            print(f"  {t[i].strftime('%H:%M:%S')} {mark} meters {score[i]:>2}: " + "; ".join(parts))


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
