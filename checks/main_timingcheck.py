# =============================================================================
# File:        main_timingcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-09
#
# Purpose:     Follow up the angle check (main_anglecheck.py), which found
#              that on the high-residual days one or more meters' aligned
#              voltage angles are scattered relative to the other meters
#              while their own three phases stay 120 degrees apart, and
#              that those meters had unusually large saved capture offsets
#              (toff) on those days. This check asks two things, without
#              changing any data or result:
#
#              Part 1 (processed tables only). For a meter and a reference
#              meter on chosen days, the per-slot deviation of the aligned
#              cross-meter angle from its ordinary value, binned by the
#              meter's saved capture offset. The ordinary value is the
#              circular mean of the same difference on the ordinary day
#              given first, so a meter behind transformers is compared
#              with its own normal shift, not with zero. The same meter is
#              shown on the ordinary day and on the high days, and the
#              same-node pair egauge_19 and egauge_21 can be given as meter
#              and reference. A relationship between offset and deviation
#              would support the timing hypothesis; it does not by itself
#              establish the cause.
#
#              Part 2 (raw files). For a handful of slots on a day, trace
#              both meters' readings from the raw phasor CSVs through the
#              steps preprocessing and the baseline take. For every
#              channel: the raw timestamp t as written, the offset
#              recomputed as its UTC time minus the slot time (floor to
#              ten seconds, as grid_channel does), the frequency and angle
#              as written (so their precision is visible), the
#              zero-crossing time minus t, how many rows landed in the
#              slot and that the earliest was retained. For the meter:
#              the saved offset and which channel supplied it
#              (preprocessing takes the first channel in order that has a
#              raw row in the slot, found here from the raw rows), the
#              meter frequency and which voltage channel supplied it (the
#              baseline takes the first voltage channel), and whether the
#              channels' timestamps agree. For the phase-a channel, two
#              alignments side by side: the channel's own frequency and
#              offset, and the baseline's rule (meter frequency, saved
#              offset), with their difference and the processed table's
#              aligned angle. The slots are the ones with the smallest and
#              largest saved offsets in the day, or given with --times.
#              The code establishes that toff is the offset of raw t
#              within its slot; whether t is the exact capture time is
#              part of what this trace investigates.
#
#              Angles are summarized circularly (mean direction, R, 5th and
#              95th percentile of the wrapped deviation, count). Nothing is
#              written; the physics weights and saved results are untouched.
#
# Usage (from the repo root):
#   python checks/main_timingcheck.py --meter egauge_9 --days 2025-06-12 2025-04-06
#   python checks/main_timingcheck.py --meter egauge_13 --days 2025-06-12 2025-07-29 2025-08-10 2025-08-17
#   python checks/main_timingcheck.py --meter egauge_19 --reference egauge_21 --days 2025-06-12 2025-08-10
#   python checks/main_timingcheck.py --meter egauge_9 --days 2025-06-12 2025-04-06 --trace --data-dir data
#
# Functions:
#   parse_args()                       command-line options with defaults
#   wrap(deg), circ(deg), ctext(stats) circular helpers, as in main_anglecheck.py
#   day_bounds(day)                    fixed-PST day to UTC start and end
#   load_day(processed_dir, meters, day)
#                                      processed table for the meters over the day
#   phase_a_phasor(table, meter, plan_map, freq_tolerance, mag_floor)
#                                      aligned phasor of the meter's phase-a voltage channel
#   deviation_by_offset(table, meter, reference, mu, plan_map, freq_tolerance, bins, mag_floor)
#                                      per-slot deviation binned by the meter's saved offset
#   print_bins(day, meter, reference, rows)
#                                      print Part 1 for one day
#   raw_rows(data_dir, meter, channel, day, utc_offset)
#                                      raw CSV rows of one channel for the day, as written, with rows per slot
#   meter_sources(table, meter, slot, channels, raws)
#                                      which channel supplied the saved offset (from the raw rows) and the meter frequency
#   trace_slots(table, raws, meter, reference, plan_map, slots, channels, freq_tolerance, mag_floor)
#                                      print Part 2 for the chosen slots
#   main()                             run Part 1 per day and Part 2 when asked
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: paths
import os
# Standard library: repo root on the import path
import sys
# Third party: numerics
import numpy as np
# Third party: tables and timestamps
import pandas as pd

# This file lives in checks/; the modules are one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The physics baseline's helpers: alignment, data loading, the measurement plan with the phase map
import socal_physics as sp
# Preprocessing helpers: raw file discovery, raw loading, the UTC and slot rules
import socal_preprocess as pre


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Aligned-angle deviation against the saved capture offset, and a raw trace of the alignment steps")
    # Processed data
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"), help="Folder with <YYYY-MM>/<meter>.parquet")
    # Raw data for the trace
    parser.add_argument("--data-dir", default="data", help="Raw download folder, for --trace")
    # Topology and config, for the phase map
    parser.add_argument("--topology-dir", default="topology", help="Caltech topology folder")
    # Caltech code
    parser.add_argument("--caltech-dir", default="caltech_utils", help="Caltech utils folder")
    # Config
    parser.add_argument("--config", default="physics_config.json", help="Physics config naming the component (the phase map comes from the network file through it)")
    # Meter
    parser.add_argument("--meter", required=True, help="Meter whose angle is examined")
    # Reference
    parser.add_argument("--reference", default="egauge_21", help="Reference meter")
    # Days
    parser.add_argument("--days", nargs="+", required=True, help="Fixed-PST days; the first is the ordinary day that sets the normal angle difference")
    # Offset bins
    parser.add_argument("--bins", nargs="+", type=float, default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], help="Edges of the saved-offset bins in seconds")
    # Frequency tolerance
    parser.add_argument("--freq-tolerance", type=float, default=0.01, help="Hertz; as in the baseline")
    # Trace
    parser.add_argument("--trace", action="store_true", help="Also trace raw readings for a few slots (needs the raw download)")
    # How many slots per end
    parser.add_argument("--n-trace", type=int, default=3, help="Slots traced from each end of the day's saved-offset range")
    # Explicit times
    parser.add_argument("--times", nargs="*", default=None, help="UTC slot times to trace instead, like 2025-04-06T15:00:00")
    # UTC offset of the raw timestamps
    parser.add_argument("--utc-offset-hours", type=int, default=8, help="Hours added to the raw fixed-PST timestamps to get UTC, as preprocessing does")
    # Magnitude floor
    parser.add_argument("--mag-floor", type=float, default=1.0, help="Volts; phasors below this have no usable angle and are left out")
    # Channel order preprocessing uses when it picks the offset
    parser.add_argument("--channels", nargs="+", default=["L1", "L2", "L3", "S1", "S2", "S3"], help="Channel order as in preprocessing; the saved offset comes from the first channel with a reading")
    # Parse
    args = parser.parse_args()
    # Bins must increase
    if any(b <= a for a, b in zip(args.bins, args.bins[1:])):
        # Refuse
        parser.error("--bins must increase")
    # Done
    return args


def wrap(deg):
    """Wrap angles in degrees to (-180, 180]."""
    # Shift, modulo, shift back
    return -((-np.asarray(deg, dtype=float) + 180.0) % 360.0) + 180.0


def circ(deg):
    """Circular mean, concentration R, 5th and 95th percentile of the wrapped deviation, and count; None when empty."""
    # Finite
    d = np.asarray(deg, dtype=float)
    # Keep
    d = d[np.isfinite(d)]
    # Nothing
    if len(d) == 0:
        # None
        return None
    # Mean vector
    m = np.exp(1j * np.deg2rad(d)).mean()
    # Direction
    mean = float(np.rad2deg(np.angle(m)))
    # Deviations
    dev = wrap(d - mean)
    # Done
    return {"mean": mean, "R": float(np.abs(m)), "p5": float(np.percentile(dev, 5)), "p95": float(np.percentile(dev, 95)), "n": int(len(d))}


def ctext(stats):
    """One text cell for a circ() result."""
    # Nothing
    if stats is None:
        # Cell
        return f"{'n/a':>31}"
    # Cell
    return f"{stats['mean']:+7.1f} R{stats['R']:.3f} [{stats['p5']:+5.1f},{stats['p95']:+5.1f}] n{stats['n']:>5}"


def day_bounds(day):
    """Fixed-PST calendar day to UTC start and end texts."""
    # Start
    start = pd.Timestamp(day) + pd.Timedelta(hours=8)
    # End
    end = start + pd.Timedelta(days=1)
    # Texts
    return start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S")


def load_day(processed_dir, meters, day):
    """Processed table for the meters over the fixed-PST day."""
    # Bounds
    start, end = day_bounds(day)
    # Load
    return sp.load_block(processed_dir, meters, start, end)


def phase_a_phasor(table, meter, plan_map, freq_tolerance, mag_floor):
    """
    Aligned phasor of the meter's phase-a voltage channel, using the
    channel the network file assigns to phase a (plan_map[meter]["a"]),
    with phasors below mag_floor set to NaN (no usable angle).
    Returns (phasor array, channel name).
    """
    # Channel for phase a
    channel = plan_map[meter]["a"]
    # Aligned phasor, the baseline's own function
    ph = sp.aligned_phasor(table, meter, channel, freq_tolerance)
    # Floor
    ph = np.where(np.abs(ph) < mag_floor, np.nan + 0j, ph)
    # Done
    return ph, channel


def deviation_by_offset(table, meter, reference, mu, plan_map, freq_tolerance, bins, mag_floor):
    """
    Per slot: the aligned phase-a angle of the meter minus the reference,
    minus mu (the ordinary difference), wrapped; the meter's saved offset;
    the reference's saved offset. Returns (rows per bin, deviation array,
    offset array, the day's overall circ of the deviation).
    """
    # Phasors
    pm, _ = phase_a_phasor(table, meter, plan_map, freq_tolerance, mag_floor)
    # Reference
    pr, _ = phase_a_phasor(table, reference, plan_map, freq_tolerance, mag_floor)
    # Deviation where both usable
    dev = wrap(np.rad2deg(np.angle(pm)) - np.rad2deg(np.angle(pr)) - mu)
    # Mask
    both = np.isfinite(pm) & np.isfinite(pr)
    # NaN elsewhere
    dev = np.where(both, dev, np.nan)
    # Saved offsets
    toff_col = f"{meter}{sp.SEP}{sp.TOFF}"
    # Meter offset
    toff = table[toff_col].to_numpy(dtype=float) if toff_col in table.columns else np.full(len(table), np.nan)
    # Reference offset
    rcol = f"{reference}{sp.SEP}{sp.TOFF}"
    # Values
    toff_r = table[rcol].to_numpy(dtype=float) if rcol in table.columns else np.full(len(table), np.nan)
    # Rows per bin
    rows = []
    # Each bin
    for lo, hi in zip(bins, bins[1:]):
        # Slots in the bin with a usable deviation
        sel = (toff >= lo) & (toff < hi) & np.isfinite(dev)
        # Row
        rows.append({"lo": lo, "hi": hi, "n_slots": int(((toff >= lo) & (toff < hi)).sum()), "stats": circ(dev[sel]), "median_abs": float(np.median(np.abs(dev[sel]))) if sel.any() else float("nan"),
                     "toff_r_median": float(np.nanmedian(toff_r[sel])) if sel.any() else float("nan")})
    # Done
    return rows, dev, toff, toff_r


def print_bins(day, meter, reference, rows, overall, toff, toff_r):
    """Print Part 1 for one day."""
    # Banner
    print(f"\n  {day}: deviation of aligned phase-a angle ({meter} minus {reference}) from its ordinary-day value, by {meter}'s saved capture offset")
    # Offsets summary
    print(f"  saved offset medians this day: {meter} {np.nanmedian(toff):.4f} s, {reference} {np.nanmedian(toff_r):.4f} s; whole day: {ctext(overall)}")
    # Header
    print(f"  {'offset bin (s)':<16} {'slots':>6} {'deviation: mean  R  [p5,p95]  n':>36} {'median |dev|':>13} {reference + ' offset':>18}")
    # Each bin
    for r in rows:
        # Formatter
        f = lambda v, d=1: "n/a" if not np.isfinite(v) else f"{v:.{d}f}"
        # Line
        print(f"  [{r['lo']:>4.1f}, {r['hi']:>4.1f})    {r['n_slots']:>6} {ctext(r['stats']):>36} {f(r['median_abs']):>13} {f(r['toff_r_median'], 4):>18}")


def raw_rows(data_dir, meter, channel, day, utc_offset):
    """
    Raw CSV rows of one channel over the fixed-PST day, read with
    preprocessing's own file discovery and loader (t parsed, duplicate
    timestamps dropped, sorted), plus: the values as written (text), t in
    UTC, the assigned slot (floor to ten seconds, as grid_channel does),
    the recomputed offset, the zero-crossing time minus t, and per slot
    how many rows landed in it (preprocessing keeps the first, which
    after sorting is the earliest t). Indexed by slot; the retained row
    is flagged.
    """
    # Day as a date
    d = pd.Timestamp(day).date()
    # Files overlapping the day
    paths = pre.find_channel_files(data_dir, meter, d, d, channel)
    # Columns to read, including the zero crossing, which preprocessing does not keep
    cols = ["t_zero_crossing", "frequency", "phase_angle_harmonic_0", "rms"]
    # Load as preprocessing does
    raw = pre.load_channel(paths, cols)
    # Nothing
    if raw.empty:
        # Empty
        return raw
    # Values exactly as written
    texts = []
    # Each file
    for p in paths:
        # Skip empty
        if os.path.getsize(p) == 0:
            # Next
            continue
        # Columns present
        header = pd.read_csv(p, nrows=0).columns.tolist()
        # Read as text
        texts.append(pd.read_csv(p, usecols=[c for c in ("t", "frequency", "phase_angle_harmonic_0") if c in header], dtype=str))
    # Join
    text = pd.concat(texts, ignore_index=True) if texts else pd.DataFrame(columns=["t", "frequency", "phase_angle_harmonic_0"])
    # Parsed t as the key, first occurrence kept as load_channel keeps it
    text["t_key"] = pd.to_datetime(text["t"], errors="coerce")
    # Drop duplicates
    text = text.drop_duplicates(subset=["t_key"]).set_index("t_key")
    # Attach
    raw["t_text"] = raw["t"].map(text["t"])
    # Frequency as written
    raw["freq_text"] = raw["t"].map(text["frequency"]) if "frequency" in text.columns else ""
    # Angle as written
    raw["ang_text"] = raw["t"].map(text["phase_angle_harmonic_0"]) if "phase_angle_harmonic_0" in text.columns else ""
    # UTC
    raw["t_utc"] = pre.to_utc(raw["t"], utc_offset)
    # Slot, as grid_channel does
    raw["slot"] = raw["t_utc"].dt.floor(f"{pre.GRID_SECONDS}s")
    # Recomputed offset
    raw["toff_recomputed"] = (raw["t_utc"] - raw["slot"]).dt.total_seconds()
    # Zero crossing minus t, seconds, when present
    zc = pd.to_datetime(raw["t_zero_crossing"], errors="coerce")
    # Difference
    raw["zc_minus_t_s"] = (zc - raw["t"]).dt.total_seconds()
    # Rows per slot
    raw["rows_in_slot"] = raw.groupby("slot")["t"].transform("size")
    # Retained row: the first per slot in t order, as grid_channel keeps it
    raw["retained"] = ~raw.duplicated(subset=["slot"], keep="first")
    # Keep the day's slots only
    start, end = day_bounds(day)
    # Clip
    raw = raw[(raw["slot"] >= pd.Timestamp(start, tz="UTC")) & (raw["slot"] < pd.Timestamp(end, tz="UTC"))]
    # Done
    return raw.set_index("slot")


def meter_sources(table, meter, slot, channels, raws):
    """
    Which channel supplied the saved offset and which supplied the meter
    frequency. Preprocessing takes the offset from the first channel in
    order that has a raw row in the slot (a timestamp), whether or not its
    values are present, so the source is found from the raw rows, not from
    the processed values; "source unknown" when no raw rows are available.
    The meter frequency is the baseline's rule: the first voltage channel
    with a finite frequency. Returns a dict with toff_from, freq_from, and
    the frequency per voltage channel.
    """
    # Row
    row = table.loc[slot] if slot in table.index else None
    # Nothing
    if row is None:
        # Empty
        return {"toff_from": "source unknown", "freq_from": None, "freqs": {}}
    # Offset source from the raw rows, in preprocessing's channel order
    toff_from = "source unknown"
    # Any raw table at all for this meter
    if any(raws.get((meter, c)) is not None and not raws[(meter, c)].empty for c in channels):
        # First channel with a raw row in the slot
        toff_from = next((c for c in channels if raws.get((meter, c)) is not None and not raws[(meter, c)].empty and slot in raws[(meter, c)].index), "no raw row in any channel")
    # Frequencies per voltage channel
    freqs = {c: float(row[f"{meter}{sp.SEP}{c}{sp.SEP}freq"]) for c in sp.VOLTAGE_CHANNELS if f"{meter}{sp.SEP}{c}{sp.SEP}freq" in table.columns}
    # Frequency source: first voltage channel with a finite frequency, as meter_frequency() does
    freq_from = next((c for c in sp.VOLTAGE_CHANNELS if c in freqs and np.isfinite(freqs[c])), None)
    # Done
    return {"toff_from": toff_from, "freq_from": freq_from, "freqs": freqs}


def trace_slots(table, raws, meter, reference, plan_map, slots, channels, freq_tolerance, mag_floor):
    """
    Print Part 2. raws maps (meter name, channel) to the raw table from
    raw_rows. For each slot and each of the two meters: the saved offset
    and which channel supplied it, the meter frequency and which channel
    supplied it, every channel's raw timestamp, recomputed offset,
    frequency and angle as written, rows that landed in the slot and the
    one retained; then for the phase-a channel two alignments side by
    side: the channel's own (its frequency, its offset) and the baseline's
    (meter frequency, saved offset), their difference, and the processed
    aligned angle from socal_physics.aligned_phasor with its difference
    from each. Finally the aligned difference between the two meters.
    """
    # Banner
    print(f"\n  raw trace, {meter} beside {reference}; slot = floor of the UTC time of raw t to 10 s, as preprocessing assigns it; offset = that time minus the slot time; alignment = angle - 360 x frequency x offset")
    # Each slot
    for s in slots:
        # Slot line
        print(f"\n  slot {s.isoformat()}")
        # Each meter
        for name in (meter, reference):
            # Sources
            src = meter_sources(table, name, s, channels, raws)
            # Saved offset
            toff_saved = float(table[f"{name}{sp.SEP}{sp.TOFF}"].get(s, np.nan)) if f"{name}{sp.SEP}{sp.TOFF}" in table.columns and s in table.index else float("nan")
            # Meter frequency, the baseline's rule
            f_meter = sp.meter_frequency(table.loc[[s]], name)[0] if s in table.index else float("nan")
            # Header for the meter
            print(f"    {name}: saved offset {toff_saved:.6f} s from channel {src['toff_from']}; meter frequency {f_meter:.6f} Hz from channel {src['freq_from']}")
            # Each channel's raw row
            offsets = []
            # Loop
            for c in channels:
                # Raw table
                raw = raws.get((name, c))
                # Row present
                if raw is None or raw.empty or s not in raw.index:
                    # Say
                    print(f"      {c}: no raw row in this slot")
                    # Next
                    continue
                # Rows that landed here
                rows = raw.loc[[s]]
                # The retained one
                r = rows[rows["retained"]].iloc[0] if rows["retained"].any() else rows.iloc[0]
                # Keep the offset for the agreement check
                offsets.append(float(r["toff_recomputed"]))
                # Line
                print(f"      {c}: raw t '{r['t_text']}'  offset {r['toff_recomputed']:.6f} s  freq '{r['freq_text']}'  angle '{r['ang_text']}'  rms {float(r['rms']):.2f}  zero-crossing minus t {r['zc_minus_t_s']:+.6f} s  rows in slot {int(r['rows_in_slot'])}" + (f", retained the earliest t" if int(r['rows_in_slot']) > 1 else ""))
            # Timestamp agreement across channels
            if len(offsets) > 1:
                # Spread
                print(f"      channel offsets agree within {max(offsets) - min(offsets):.6f} s")
            # Phase-a channel
            ca = plan_map[name]["a"]
            # Raw row for it
            raw = raws.get((name, ca))
            # Processed aligned phasor
            ph = sp.aligned_phasor(table.loc[[s]], name, ca, freq_tolerance)[0] if s in table.index else np.nan + 0j
            # Floor
            if np.isfinite(ph) and abs(ph) < mag_floor:
                # Unusable
                ph = np.nan + 0j
            # Processed angle
            a_proc = float(np.rad2deg(np.angle(ph))) if np.isfinite(ph) else float("nan")
            # Raw-based alignments
            if raw is not None and not raw.empty and s in raw.index:
                # Retained row
                rows = raw.loc[[s]]
                # Pick
                r = rows[rows["retained"]].iloc[0] if rows["retained"].any() else rows.iloc[0]
                # The channel's own alignment
                a_own = float(wrap(float(r["phase_angle_harmonic_0"]) - 360.0 * float(r["frequency"]) * float(r["toff_recomputed"])))
                # The baseline's alignment: meter frequency and saved offset
                a_base = float(wrap(float(r["phase_angle_harmonic_0"]) - 360.0 * f_meter * toff_saved)) if np.isfinite(f_meter) and np.isfinite(toff_saved) else float("nan")
                # Line
                print(f"      phase a ({ca}): aligned with the channel's own frequency and offset {a_own:+.2f} deg; with the meter frequency and saved offset (the baseline's rule) {a_base:+.2f} deg; difference {wrap(a_base - a_own)[()]:+.2f} deg; processed table gives {a_proc:+.2f} deg (differs from baseline-rule {wrap(a_proc - a_base)[()]:+.2f}, from own {wrap(a_proc - a_own)[()]:+.2f})")
            else:
                # No raw
                print(f"      phase a ({ca}): no raw row; processed table gives {a_proc:+.2f} deg")
        # Aligned difference between the meters from the processed table
        pm = sp.aligned_phasor(table.loc[[s]], meter, plan_map[meter]["a"], freq_tolerance)[0] if s in table.index else np.nan + 0j
        # Reference
        pr = sp.aligned_phasor(table.loc[[s]], reference, plan_map[reference]["a"], freq_tolerance)[0] if s in table.index else np.nan + 0j
        # Both usable above the floor
        if np.isfinite(pm) and np.isfinite(pr) and abs(pm) >= mag_floor and abs(pr) >= mag_floor:
            # Print
            print(f"    aligned difference from the processed table ({meter} minus {reference}, phase a): {wrap(np.rad2deg(np.angle(pm)) - np.rad2deg(np.angle(pr)))[()]:+.2f} deg")
        else:
            # Not available
            print("    aligned difference: not available (one side unusable after alignment or below the floor)")


def main():
    """Part 1 per day, then Part 2 when asked."""
    # Options
    args = parse_args()
    # Config and plan, for the phase map and the meter list
    config = sp.load_config(args.config)
    # Caltech
    ct = sp.setup_caltech(args.caltech_dir, args.topology_dir)
    # Model
    model = sp.build_reference_model(ct, args.topology_dir, config)
    # Plan
    plan = sp.measurement_plan(model, config)
    # Phase map per meter: phase letter to channel
    plan_map = {}
    # Each voltage term
    for v in plan["voltages"]:
        # Store
        plan_map.setdefault(v["meter"], {})[sp.PHASES[v["phase_idx"]]] = v["channel"]
    # Both meters must have voltage terms
    for m in (args.meter, args.reference):
        # Check
        if m not in plan_map or "a" not in plan_map[m]:
            # Stop
            raise SystemExit(f"{m} has no phase-a voltage term in the plan of {args.config}")
    # Say
    print(f"meter {args.meter} (phase a on channel {plan_map[args.meter]['a']}), reference {args.reference} (phase a on channel {plan_map[args.reference]['a']}); ordinary day {args.days[0]} sets the normal difference; alignment and the off-bin rule as in the baseline (tolerance {args.freq_tolerance} Hz)")
    # Ordinary difference
    mu = None
    # Each day
    for i, day in enumerate(args.days):
        # Table
        table = load_day(args.processed_dir, [args.meter, args.reference], day)
        # Empty
        if table.empty:
            # The ordinary day must exist, since it sets the reference difference
            if i == 0:
                # Stop
                raise SystemExit(f"the ordinary day {day} has no processed data; it sets the normal difference, so the check stops")
            # Say
            print(f"\n  {day}: no processed data")
            # Next
            continue
        # First day: the normal difference
        if mu is None:
            # Phasors
            pm, _ = phase_a_phasor(table, args.meter, plan_map, args.freq_tolerance, args.mag_floor)
            # Reference
            pr, _ = phase_a_phasor(table, args.reference, plan_map, args.freq_tolerance, args.mag_floor)
            # Difference
            st = circ(wrap(np.rad2deg(np.angle(pm)) - np.rad2deg(np.angle(pr))))
            # None usable
            if st is None:
                # Stop
                raise SystemExit(f"no usable slots on the ordinary day {day} to set the normal difference")
            # Store
            mu = st["mean"]
            # Say
            print(f"  ordinary difference {args.meter} minus {args.reference} on {day}: {mu:+.2f} deg, R {st['R']:.3f}, n {st['n']}")
        # Part 1
        rows, dev, toff, toff_r = deviation_by_offset(table, args.meter, args.reference, mu, plan_map, args.freq_tolerance, args.bins, args.mag_floor)
        # Print
        print_bins(day, args.meter, args.reference, rows, circ(dev), toff, toff_r)
        # Part 2
        if args.trace:
            # Slots to trace
            if args.times:
                # Given
                slots = [pd.Timestamp(t, tz="UTC") for t in args.times]
            else:
                # Smallest and largest saved offsets with a usable deviation
                ok = np.isfinite(dev) & np.isfinite(toff)
                # Order
                order = np.argsort(np.where(ok, toff, np.nan))
                # Finite only
                order = [k for k in order if ok[k]]
                # Pick
                picks = order[:args.n_trace] + order[-args.n_trace:] if len(order) > 2 * args.n_trace else order
                # Times
                slots = [pd.Timestamp(table.index[k]) for k in picks]
            # Raw rows for every channel of both meters
            raws = {(m, c): raw_rows(args.data_dir, m, c, day, args.utc_offset_hours) for m in (args.meter, args.reference) for c in args.channels}
            # Trace
            trace_slots(table, raws, args.meter, args.reference, plan_map, slots, args.channels, args.freq_tolerance, args.mag_floor)


# Entry point
if __name__ == "__main__":
    # Run
    main()
