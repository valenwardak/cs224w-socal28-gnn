# =============================================================================
# File:        socal_score.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-27
#
# Purpose:     Turn a run's residual tables into event-detection scores, the
#              same way for every model. One script scores the GRU, the
#              GNNs, Landon's models, and the persistence baseline, from
#              the Parquet tables that score_split (socal_models.py) writes.
#              Imported by the thin main script main_score.py.
#
# The method, in order:
#   1. Calibrate. On one validation block (block 1, February 2025, never
#      used to choose a model), take every valid residual of each meter and
#      channel and set that channel's threshold at the conformal rank
#      k = ceil((n + 1)(1 - alpha)) among its n sorted residuals. No
#      distribution is assumed. If k > n (too few residuals for the
#      requested alpha) or the channel has no residual, it gets no
#      threshold and never fires; those channels are listed. The empirical
#      coverage on the calibration block is reported per channel. Slots
#      are not independent in time, so alpha is a per-slot, per-channel
#      design level, not a guaranteed grid-level false-alarm rate.
#   2. Exceed. On a test block, a channel exceeds when its residual is above
#      its threshold. Magnitude and angle each have their own threshold.
#      Mode "mag" uses magnitude only, "ang" angle only, "both" fires a
#      channel if either exceeds; a channel fires once, not twice.
#   3. Aggregate. A slot's score is the number of distinct meters with at
#      least one firing channel, so one meter's three phases count once.
#      The slot is an alarm slot when the score is at least min_meters.
#   4. Episodes. Alarm slots closer together than merge_slots (gaps of that
#      many slots or fewer) merge into one alarm episode with a start and
#      an end, so a sustained alarm counts once.
#   5. Match. The logged switching events are grouped into episodes (the
#      "episode" column of events.csv). An event episode's window runs from
#      its first scored change minus before_minutes to its last scored
#      change plus after_minutes. It is detected when an alarm episode
#      starts inside that window. Matching is one to one: each alarm
#      episode matches at most one event episode, the earliest unmatched
#      one. Delay is from the first scored change to the alarm start.
#      Only event episodes whose window lies inside a scored test block
#      are in scope.
#   6. Count. Unmatched alarm episodes are false alarms. The false-alarm
#      rate is per monitored day, pooled over every test table and also
#      per split (the event blocks and the quiet block separately, since
#      the quiet block is the clean false-alarm measure). Monitored slots are the scored
#      slots with at least min_meters meters holding a valid residual on a
#      channel that has a threshold (a channel without one cannot fire, so
#      it does not count as watched). Slots inside the window of an event
#      episode that is only partly covered by the scored blocks are
#      excluded from both alarms and monitored time, and those episodes
#      are reported, so a half-seen event is neither a hit nor a false
#      alarm. Precision, recall, and F1 are over episodes. The bootstrap
#      resamples episodes only: the recall interval is a plain bootstrap;
#      the precision and F1 intervals hold the false-alarm count fixed and
#      are conditional on it, which the output says.
#   7. Fairness. Persistence is calibrated and scored from its own columns
#      by the same steps. With common=True, every residual is first masked
#      to the slots where both the model and persistence are valid for that
#      channel, in calibration and in test, so the two see the same slots.
#
# Grid statistic (statistic="grid", exploratory; "channel" above is the
# default and the primary result). Per-channel thresholds calibrate each
# channel, not the alarm: with 132 channels at alpha 1e-4, chance pairs of
# exceeding meters appear many times a day. The grid statistic calibrates
# the one number that raises the alarm:
#   a. Reference scales. On the design block (validation block 0, January
#      2024, never used for the final threshold), each channel's reference
#      is the ref_percentile (99th) of its valid residuals, raised to a
#      positive floor (ref_floor_mag in scaled units, ref_floor_ang in
#      degrees) so a quantized channel whose residuals are mostly zero
#      cannot divide by zero. A channel with fewer than ref_min_count
#      residuals on the design block has no reference and is left out.
#      Channels at the floor and channels left out are reported.
#   b. Channel score = residual / reference, magnitude and angle each
#      against their own reference; in mode "both" a channel's score is the
#      larger of the two.
#   c. Meter score = the maximum over the meter's available channels, so
#      the three phases count once. A meter is available in a slot when at
#      least one channel of the mode has a valid residual and a reference.
#   d. Grid score = the second-largest meter score among available meters.
#      Two meters must be elevated for it to be high. With fixed references
#      removing meters cannot raise it for that slot; different meter sets
#      between calibration and test can still shift its distribution, so
#      exceedance rates by available-meter count are reported.
#   e. Threshold. One conformal threshold on the grid score from the
#      calibration block (block 1) at alpha, the same rank rule as above. A
#      slot is monitored when at least max(2, min_meters) meters are
#      available, and is an alarm slot when its grid score exceeds the
#      threshold. Episodes, matching, and reporting are unchanged.
#   The coverage reported on block 1 is empirical calibration coverage and
#   is largely fixed by the fitted rank; the alarm-slot fraction and
#   episode rates on the test blocks are the generalization checks.
#
# These are residual-only detections: a meter that goes silent produces no
# residual and no alarm here. A silence detector from the feature mask is a
# separate component.
#
# Functions:
#   Tables
#     list_tables(run_dir, prefix)        block labels whose Parquet file starts with prefix
#     load_table(run_dir, label)          read one residual table
#     table_axes(table)                   (meters, channels) from the column names
#     residual_arrays(table, meters, channels, prefix)
#                                         dict of (T, M, C) arrays: mag, ang, mag_valid, ang_valid
#     common_mask(model, pers)            mask both sources to their shared valid slots
#   Calibration
#     conformal_threshold(values, alpha)  one threshold from a 1-D array of residuals, or NaN
#     calibrate(arrays, alpha)            (M, C) thresholds and coverage for mag and ang
#   Grid statistic
#     references(arrays, percentile, min_count, floor_mag, floor_ang)
#                                         (M, C) reference scales for mag and ang from the design block
#     grid_scores(arrays, refs, mode, need)
#                                         (T,) second-largest meter score and (T,) available-meter count
#   Detection
#     exceedances(arrays, thresholds, mode)
#                                         (T, M, C) bool, a channel fires or not
#     slot_scores(fires)                  (T,) number of distinct meters firing
#     monitored_slots(arrays, thresholds, mode, min_meters)
#                                         (T,) bool, enough meters had a residual on a thresholded channel
#     window_mask(t, episodes, before, after)
#                                         (T,) bool, slots inside the given episodes' windows
#     alarm_episodes(t, alarm, merge_slots)
#                                         list of (start, end) timestamps
#   Events
#     load_episodes(path)                 scored events grouped into episodes
#     episodes_in_scope(episodes, blocks, before, after)
#                                         (in scope, partial): windows inside a block, windows only partly covered
#     match(alarms, episodes, before, after)
#                                         per-episode detection and delay, unmatched alarms
#   Scoring
#     bootstrap(detected, false_count, n_boot, seed)
#                                         recall interval; conditional precision and F1 intervals
#     score_source(run_dir, prefix, settings, other_prefix)
#                                         the whole pipeline for one source, returns a dict
# =============================================================================

# Standard library: ceilings for the conformal rank
import math
# Standard library: to silence numpy's all-NaN warnings where NaN means "no score"
import warnings
# Standard library: paths
import os
# Third party: numerics
import numpy as np
# Third party: tables and timestamps
import pandas as pd

# Seconds per slot, the feature grid
SLOT_SECONDS = 10


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------

def list_tables(run_dir, prefix):
    """Block labels in a run folder whose Parquet file name starts with prefix, sorted."""
    # Matching files
    names = [f[:-8] for f in os.listdir(run_dir) if f.startswith(prefix) and f.endswith(".parquet")]
    # A requested split with no table is an error, not a silent gap in the results
    if not names:
        # Raise
        raise FileNotFoundError(f"no table starting with '{prefix}' in {run_dir}; score that split first")
    # Sorted by label, which sorts by date
    return sorted(names)


def load_table(run_dir, label):
    """Read one residual table written by score_split."""
    # Path
    path = os.path.join(run_dir, f"{label}.parquet")
    # Must exist
    if not os.path.exists(path):
        # Raise
        raise FileNotFoundError(f"no residual table at {path}")
    # Read
    table = pd.read_parquet(path)
    # Empty tables cannot be scored
    if len(table) == 0:
        # Raise
        raise ValueError(f"empty residual table: {path}")
    # Return
    return table


def table_axes(table):
    """Meters and channels in first-seen order from the "<meter>|<channel>|<name>" column names."""
    # Lists
    meters, channels = [], []
    # Loop over columns
    for col in table.columns:
        # Split
        meter, ch, _ = col.split("|")
        # Collect meter
        if meter not in meters:
            # New
            meters.append(meter)
        # Collect channel
        if ch not in channels:
            # New
            channels.append(ch)
    # Return
    return meters, channels


def residual_arrays(table, meters, channels, prefix=""):
    """
    Residuals of one source as (T, M, C) arrays. prefix "" is the model,
    "pers_" is persistence. Returns a dict with mag, ang (float, NaN where
    invalid), mag_valid, ang_valid (bool), and t (the timestamps).
    """
    # Sizes
    T, M, C = len(table), len(meters), len(channels)
    # Output arrays
    out = {"mag": np.full((T, M, C), np.nan, dtype=np.float32), "ang": np.full((T, M, C), np.nan, dtype=np.float32),
           "mag_valid": np.zeros((T, M, C), dtype=bool), "ang_valid": np.zeros((T, M, C), dtype=bool)}
    # Loop over meters
    for mi, meter in enumerate(meters):
        # Loop over channels
        for ci, ch in enumerate(channels):
            # Loop over the four names
            for key, name in (("mag", "mag_res"), ("ang", "ang_res"), ("mag_valid", "mag_valid"), ("ang_valid", "ang_valid")):
                # Column
                col = f"{meter}|{ch}|{prefix}{name}"
                # Present
                if col in table.columns:
                    # Copy
                    out[key][:, mi, ci] = table[col].to_numpy()
    # A residual is valid only where it is also finite
    out["mag_valid"] &= np.isfinite(out["mag"])
    # Angle
    out["ang_valid"] &= np.isfinite(out["ang"])
    # Timestamps
    out["t"] = table.index.to_numpy()
    # Return
    return out


def common_mask(a, b):
    """Mask two sources' residuals to the slots where both are valid, per channel and kind. Edits in place."""
    # Loop over kinds
    for kind in ("mag", "ang"):
        # Shared validity
        both = a[f"{kind}_valid"] & b[f"{kind}_valid"]
        # Apply to both
        for src in (a, b):
            # Validity
            src[f"{kind}_valid"] = both
            # Residual
            src[kind] = np.where(both, src[kind], np.nan)


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------

def conformal_threshold(values, alpha):
    """
    Conformal threshold from a 1-D array of calibration residuals: the
    k-th smallest with k = ceil((n + 1)(1 - alpha)). NaN when there are
    no residuals or k exceeds n (too few residuals for this alpha).
    """
    # Finite values only
    v = values[np.isfinite(values)]
    # Count
    n = len(v)
    # None
    if n == 0:
        # No threshold
        return float("nan")
    # Rank
    k = math.ceil((n + 1) * (1.0 - alpha))
    # Too few
    if k > n:
        # No threshold
        return float("nan")
    # k-th smallest, 1-based
    return float(np.partition(v, k - 1)[k - 1])


def calibrate(arrays, alpha):
    """
    Thresholds per meter and channel for magnitude and angle, shape (M, C)
    each, from one source's calibration arrays. Also the empirical coverage
    per channel: the share of calibration residuals at or below the
    threshold. Returns dict thr_mag, thr_ang, cov_mag, cov_ang, n_mag, n_ang.
    """
    # Sizes
    _, M, C = arrays["mag"].shape
    # Output
    out = {}
    # Loop over kinds
    for kind in ("mag", "ang"):
        # Thresholds
        thr = np.full((M, C), np.nan, dtype=np.float32)
        # Coverage
        cov = np.full((M, C), np.nan, dtype=np.float32)
        # Counts
        n = np.zeros((M, C), dtype=np.int64)
        # Loop over channels
        for mi in range(M):
            # Loop
            for ci in range(C):
                # Residuals of this channel
                v = arrays[kind][:, mi, ci]
                # Valid ones
                v = v[arrays[f"{kind}_valid"][:, mi, ci]]
                # Count
                n[mi, ci] = len(v)
                # Threshold
                thr[mi, ci] = conformal_threshold(v, alpha)
                # Coverage where a threshold exists
                if np.isfinite(thr[mi, ci]) and len(v):
                    # Share at or below
                    cov[mi, ci] = float(np.mean(v <= thr[mi, ci]))
        # Keep
        out[f"thr_{kind}"], out[f"cov_{kind}"], out[f"n_{kind}"] = thr, cov, n
    # Return
    return out


# -----------------------------------------------------------------------------
# Grid statistic
# -----------------------------------------------------------------------------

def references(arrays, percentile, min_count, floor_mag, floor_ang):
    """
    Reference scale per meter and channel for magnitude and angle, from
    the design block's valid residuals: the given percentile, raised to
    the floor. NaN where fewer than min_count residuals exist. Returns
    dict ref_mag, ref_ang (M, C), n_mag, n_ang counts, and floored_mag,
    floored_ang bool (M, C) marking channels whose percentile sat below
    the floor.
    """
    # Sizes
    _, M, C = arrays["mag"].shape
    # Output
    out = {}
    # Loop over kinds
    for kind, floor in (("mag", floor_mag), ("ang", floor_ang)):
        # Reference
        ref = np.full((M, C), np.nan, dtype=np.float32)
        # Floored flags
        floored = np.zeros((M, C), dtype=bool)
        # Counts
        n = np.zeros((M, C), dtype=np.int64)
        # Loop over channels
        for mi in range(M):
            # Loop
            for ci in range(C):
                # Valid residuals
                v = arrays[kind][:, mi, ci][arrays[f"{kind}_valid"][:, mi, ci]]
                # Count
                n[mi, ci] = len(v)
                # Enough
                if len(v) >= min_count:
                    # Percentile
                    q = float(np.percentile(v, percentile))
                    # Below the floor
                    floored[mi, ci] = q < floor
                    # Raised to the floor
                    ref[mi, ci] = max(q, floor)
        # Keep
        out[f"ref_{kind}"], out[f"n_{kind}"], out[f"floored_{kind}"] = ref, n, floored
    # Return
    return out


def grid_scores(arrays, refs, mode, need):
    """
    Grid score per slot: the second-largest meter score among available
    meters, where a meter score is the maximum over its channels of
    residual / reference (mode "mag", "ang", or "both" = the larger of the
    two per channel). Returns (score (T,), available (T,)): score is NaN
    where fewer than `need` meters are available.
    """
    # Channel scores per kind, NaN where no valid residual or no reference
    parts = []
    # Kinds in the mode
    for kind in (["mag", "ang"] if mode == "both" else [mode]):
        # Residuals where valid, NaN elsewhere
        r = np.where(arrays[f"{kind}_valid"], arrays[kind], np.nan)
        # Divided by the reference, NaN where none
        parts.append(r / refs[f"ref_{kind}"][None, :, :])
    # All-NaN slices mean "no score" here, so numpy's warning about them is noise
    with warnings.catch_warnings():
        # Silence
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Larger of the kinds per channel, NaN only where every kind is NaN
        chan = np.nanmax(np.stack(parts), axis=0) if len(parts) > 1 else parts[0]
        # Meter score: maximum over channels, NaN where no channel has a score
        meter = np.nanmax(chan, axis=2)
    # Available meters per slot
    available = np.isfinite(meter).sum(axis=1)
    # Second largest among available: sort with NaN last, take position 1 from the end of the finite part
    ordered = np.sort(np.where(np.isfinite(meter), meter, -np.inf), axis=1)
    # Second largest
    second = ordered[:, -2]
    # NaN where too few meters
    score = np.where(available >= max(2, need), second, np.nan)
    # Return
    return score.astype(np.float32), available


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------

def exceedances(arrays, thresholds, mode):
    """
    (T, M, C) bool: a channel fires in a slot. mode "mag": magnitude above
    its threshold; "ang": angle above its threshold; "both": either. A
    channel with no threshold never fires. NaN residuals never fire.
    """
    # Magnitude exceedance: valid residual above its threshold; a NaN residual or NaN threshold never fires
    mag = arrays["mag_valid"] & (np.nan_to_num(arrays["mag"], nan=-np.inf) > np.nan_to_num(thresholds["thr_mag"], nan=np.inf)[None, :, :])
    # Angle exceedance
    ang = arrays["ang_valid"] & (np.nan_to_num(arrays["ang"], nan=-np.inf) > np.nan_to_num(thresholds["thr_ang"], nan=np.inf)[None, :, :])
    # By mode
    if mode == "mag":
        # Magnitude only
        return mag
    if mode == "ang":
        # Angle only
        return ang
    # Either, once per channel
    return mag | ang


def slot_scores(fires):
    """(T,) number of distinct meters with at least one firing channel."""
    # Any channel per meter, then count meters
    return fires.any(axis=2).sum(axis=1)


def monitored_slots(arrays, thresholds, mode, min_meters):
    """
    (T,) bool: at least min_meters meters hold, in the slot, a valid
    residual on a channel that has a threshold, for the mode's kind(s).
    A channel without a threshold cannot fire, so it is not watched.
    """
    # Watched magnitude channels: valid residual and a threshold
    mag = arrays["mag_valid"] & np.isfinite(thresholds["thr_mag"])[None, :, :]
    # Watched angle channels
    ang = arrays["ang_valid"] & np.isfinite(thresholds["thr_ang"])[None, :, :]
    # By mode
    if mode == "mag":
        # Magnitude
        watched = mag
    elif mode == "ang":
        # Angle
        watched = ang
    else:
        # Either
        watched = mag | ang
    # Meters with any watched channel, counted
    return watched.any(axis=2).sum(axis=1) >= min_meters


def window_mask(t, episodes, before, after):
    """(T,) bool: slots that fall inside the window of any of the given event episodes."""
    # Timestamps
    times = pd.to_datetime(t, utc=True)
    # Start empty
    mask = np.zeros(len(times), dtype=bool)
    # Loop over episodes
    for _, e in episodes.iterrows():
        # Inside the window
        mask |= np.asarray((times >= e["first"] - before) & (times <= e["last"] + after))
    # Return
    return mask


def alarm_episodes(t, alarm, merge_slots):
    """
    Merge alarm slots into episodes. t is the (T,) timestamp array of the
    table (datetime64, UTC), alarm the (T,) bool. Two alarm slots join the
    same episode when the gap between them is at most merge_slots slots.
    Returns a list of (start, end) pandas Timestamps.
    """
    # Alarm times, in order
    times = pd.to_datetime(t[alarm], utc=True)
    # None
    if len(times) == 0:
        # Empty
        return []
    # Largest gap that still merges
    gap = pd.Timedelta(seconds=merge_slots * SLOT_SECONDS)
    # Episodes
    out = []
    # Current episode start and end
    start = end = times[0]
    # Loop over the rest
    for ts in times[1:]:
        # Close enough: extend
        if ts - end <= gap:
            # Extend
            end = ts
        # Otherwise close and start a new one
        else:
            # Close
            out.append((start, end))
            # New
            start = end = ts
    # Last one
    out.append((start, end))
    # Return
    return out


# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------

def load_episodes(path):
    """
    Scored switching events grouped into episodes. Returns a DataFrame
    with one row per episode: episode, first (first scored change),
    last (last scored change), n_changes, devices.
    """
    # Read
    ev = pd.read_csv(path)
    # Timestamps
    ev["t_utc"] = pd.to_datetime(ev["t_utc"], utc=True, format="ISO8601")
    # Scored changes only
    ev = ev[ev["scored"].astype(bool)]
    # Group
    rows = []
    # Loop over episodes
    for ep, g in ev.groupby("episode"):
        # Row
        rows.append({"episode": int(ep), "first": g["t_utc"].min(), "last": g["t_utc"].max(), "n_changes": int(len(g)), "devices": ",".join(sorted(set(g["device"])))})
    # Table sorted by time
    return pd.DataFrame(rows).sort_values("first").reset_index(drop=True)


def episodes_in_scope(episodes, blocks, before, after):
    """
    Split the episodes by how the scored blocks cover their window
    (first - before, last + after): fully inside a block (in scope),
    only partly overlapping a block (partial, excluded from alarms and
    monitored time), or not touching any block (ignored). blocks is a
    list of (start, end) Timestamps. Returns (in_scope, partial).
    """
    # Flags
    inside, partial = [], []
    # Loop over episodes
    for _, e in episodes.iterrows():
        # Window
        w0, w1 = e["first"] - before, e["last"] + after
        # Fully inside a block
        full = any(b0 <= w0 and w1 <= b1 for b0, b1 in blocks)
        # Touching a block at all
        touch = any(w0 <= b1 and w1 >= b0 for b0, b1 in blocks)
        # Record
        inside.append(full)
        # Partly covered
        partial.append(touch and not full)
    # Subsets
    return episodes[np.array(inside, dtype=bool)].reset_index(drop=True), episodes[np.array(partial, dtype=bool)].reset_index(drop=True)


def match(alarms, episodes, before, after):
    """
    One-to-one matching of alarm episodes to event episodes. An event
    episode is detected when an unmatched alarm episode starts inside its
    window; the earliest such alarm is taken. Returns (results, unmatched)
    where results is a list of dicts per event episode (detected, delay in
    seconds, alarm_start) and unmatched the alarm episodes left over.
    """
    # Alarms in time order with a used flag
    pool = [[s, e, False] for s, e in sorted(alarms)]
    # Results
    results = []
    # Loop over event episodes in time order
    for _, ep in episodes.iterrows():
        # Window
        w0, w1 = ep["first"] - before, ep["last"] + after
        # First unmatched alarm starting inside
        hit = next((a for a in pool if not a[2] and w0 <= a[0] <= w1), None)
        # Detected
        if hit is not None:
            # Mark used
            hit[2] = True
            # Record
            results.append({"episode": int(ep["episode"]), "first": ep["first"], "detected": True, "delay_s": float((hit[0] - ep["first"]).total_seconds()), "alarm_start": hit[0]})
        # Missed
        else:
            # Record
            results.append({"episode": int(ep["episode"]), "first": ep["first"], "detected": False, "delay_s": None, "alarm_start": None})
    # Leftovers
    unmatched = [(a[0], a[1]) for a in pool if not a[2]]
    # Return
    return results, unmatched


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

def bootstrap(detected, false_count, n_boot, seed):
    """
    Bootstrap intervals from resampling the event episodes. detected is
    a list of bools, one per episode. The recall interval is a plain
    bootstrap over episodes. The precision and F1 intervals reuse the
    resampled hits against the fixed false-alarm count, so they are
    conditional on that count and carry none of its uncertainty; the
    returned dict says so in "note". 2.5 and 97.5 percentiles.
    """
    # The caveat, carried into the output
    note = "recall: bootstrap over episodes; precision and F1: conditional on the observed false-alarm count"
    # Nothing to resample
    if len(detected) == 0:
        # Empty
        return {"recall_lo": None, "recall_hi": None, "precision_cond_lo": None, "precision_cond_hi": None, "f1_cond_lo": None, "f1_cond_hi": None, "note": note}
    # Generator
    rng = np.random.default_rng(seed)
    # Array
    d = np.array(detected, dtype=float)
    # Resampled recall
    rec = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
    # Resampled hits
    hits = rec * len(d)
    # Precision with the fixed false count
    prec = hits / np.maximum(hits + false_count, 1.0)
    # F1 from the two, zero where both are zero
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    # Percentiles
    return {"recall_lo": float(np.percentile(rec, 2.5)), "recall_hi": float(np.percentile(rec, 97.5)),
            "precision_cond_lo": float(np.percentile(prec, 2.5)), "precision_cond_hi": float(np.percentile(prec, 97.5)),
            "f1_cond_lo": float(np.percentile(f1, 2.5)), "f1_cond_hi": float(np.percentile(f1, 97.5)), "note": note}


def score_source(run_dir, prefix, settings, other_prefix=None):
    """
    The whole pipeline for one source ("" model, "pers_" persistence):
    calibrate on the calibration block, detect on every test table, match
    against the event episodes, count false alarms over monitored time.
    With other_prefix given and settings["common"] true, residuals are
    masked to the slots where both sources are valid. settings is a dict
    with alpha, mode, min_meters, merge_slots, before_minutes,
    after_minutes, calib_label, test_prefixes, events, n_boot, seed,
    common. Returns a dict of results.
    """
    # Windows
    before, after = pd.Timedelta(minutes=settings["before_minutes"]), pd.Timedelta(minutes=settings["after_minutes"])
    # Calibration table
    cal = load_table(run_dir, settings["calib_label"])
    # Axes
    meters, channels = table_axes(cal)
    # Calibration residuals
    arrays = residual_arrays(cal, meters, channels, prefix)
    # Common slots
    if other_prefix is not None and settings["common"]:
        # Mask against the other source
        common_mask(arrays, residual_arrays(cal, meters, channels, other_prefix))
    # Thresholds, per channel
    thr = calibrate(arrays, settings["alpha"])
    # Channels without a threshold, by kind
    missing = {kind: [f"{meters[mi]}|{channels[ci]}" for mi, ci in zip(*np.nonzero(~np.isfinite(thr[f"thr_{kind}"])))] for kind in ("mag", "ang")}
    # Grid statistic: references from the design block, one threshold from the calibration block
    grid = None
    # Asked for
    if settings.get("statistic", "channel") == "grid":
        # Meters needed
        need = max(2, settings["min_meters"])
        # Design table
        des = load_table(run_dir, settings["design_label"])
        # Design residuals
        darr = residual_arrays(des, meters, channels, prefix)
        # Common slots
        if other_prefix is not None and settings["common"]:
            # Mask
            common_mask(darr, residual_arrays(des, meters, channels, other_prefix))
        # References
        refs = references(darr, settings["ref_percentile"], settings["ref_min_count"], settings["ref_floor_mag"], settings["ref_floor_ang"])
        # Grid scores on the calibration block
        gscore, gavail = grid_scores(arrays, refs, settings["mode"], need)
        # One threshold
        gthr = conformal_threshold(gscore, settings["alpha"])
        # Without a threshold the grid statistic cannot alarm; a silent zero-alarm result would mislead, so it raises
        if not np.isfinite(gthr):
            # Raise with the numbers
            raise ValueError(f"grid threshold cannot be calibrated: {int(np.isfinite(gscore).sum())} valid calibration slots on {settings['calib_label']} at alpha {settings['alpha']} (need at least {math.ceil(1.0 / settings['alpha']) - 1})")
        # Empirical calibration coverage
        gcov = float(np.mean(gscore[np.isfinite(gscore)] <= gthr))
        # Record
        grid = {"refs": refs, "threshold": gthr, "coverage": gcov, "n_calibration": int(np.isfinite(gscore).sum()), "need": need}
    # Events
    episodes = load_episodes(settings["events"])
    # Test tables, every requested prefix must have at least one
    labels = [lab for p in settings["test_prefixes"] for lab in list_tables(run_dir, p)]
    # Load every test table once, with its residual arrays
    tables = []
    # Loop over labels
    for label in labels:
        # Table
        table = load_table(run_dir, label)
        # Residuals
        arr = residual_arrays(table, meters, channels, prefix)
        # Common slots
        if other_prefix is not None and settings["common"]:
            # Mask
            common_mask(arr, residual_arrays(table, meters, channels, other_prefix))
        # Keep
        tables.append((label, arr))
    # Block spans, from the first and last scored slot of each table
    blocks = [(pd.to_datetime(arr["t"][0], utc=True), pd.to_datetime(arr["t"][-1], utc=True)) for _, arr in tables]
    # Episodes fully inside the scored blocks, and those only partly covered
    scoped, partial = episodes_in_scope(episodes, blocks, before, after)
    # Alarms, monitored time, per-block records over all test tables
    alarms, monitored_s, per_block = [], 0.0, []
    # Exceedance counts by available-meter count, grid statistic only
    by_avail = {}
    # Loop over tables
    for label, arr in tables:
        # Per-channel statistic
        if grid is None:
            # Firing channels
            fires = exceedances(arr, thr, settings["mode"])
            # Meters firing per slot
            score = slot_scores(fires)
            # Monitored slots: enough meters with a residual on a thresholded channel
            mon = monitored_slots(arr, thr, settings["mode"], settings["min_meters"])
            # Slots inside a partly covered event window are neither alarms nor monitored
            mon &= ~window_mask(arr["t"], partial, before, after)
            # Alarm slots: enough meters firing, and the slot is monitored
            alarm = (score >= settings["min_meters"]) & mon
        # Grid statistic
        else:
            # Scores and availability
            gscore, gavail = grid_scores(arr, grid["refs"], settings["mode"], grid["need"])
            # Monitored: enough available meters, outside partial windows
            mon = np.isfinite(gscore) & ~window_mask(arr["t"], partial, before, after)
            # Alarm: score above the one threshold
            alarm = mon & (np.nan_to_num(gscore, nan=-np.inf) > grid["threshold"])
            # Exceedance by available-meter count
            for a in np.unique(gavail[mon]):
                # Slots with this count
                sel = mon & (gavail == a)
                # Tally
                rec = by_avail.setdefault(int(a), {"slots": 0, "alarm_slots": 0})
                # Add
                rec["slots"] += int(sel.sum()); rec["alarm_slots"] += int((alarm & sel).sum())
        # Episodes
        eps = alarm_episodes(arr["t"], alarm, settings["merge_slots"])
        # Keep
        alarms.extend(eps)
        # Monitored seconds
        monitored_s += float(mon.sum()) * SLOT_SECONDS
        # Merged episode spans in seconds, slot-inclusive
        spans = [float((e - s0).total_seconds()) + SLOT_SECONDS for s0, e in eps]
        # Per-block record, with the alarm-slot fraction and the merged-span summary
        per_block.append({"label": label, "slots": int(len(arr["t"])), "monitored_slots": int(mon.sum()), "alarm_slots": int(alarm.sum()),
                          "alarm_slot_fraction": (float(alarm.sum()) / float(mon.sum())) if mon.sum() else None,
                          "alarm_episodes": len(eps), "merged_span_median_s": float(np.median(spans)) if spans else None, "merged_span_max_s": float(max(spans)) if spans else None})
    # Match
    results, unmatched = match(alarms, scoped, before, after)
    # Counts
    n_ep, n_det, n_false = len(results), sum(r["detected"] for r in results), len(unmatched)
    # False alarms and monitored time per split prefix, so the quiet block's rate is reported on its own
    per_split = {}
    # Loop over prefixes
    for p in settings["test_prefixes"]:
        # Blocks of this split, with their spans
        spans = [(b, blocks[i]) for i, b in enumerate(per_block) if b["label"].startswith(p)]
        # Monitored seconds of this split
        sec = sum(b["monitored_slots"] for b, _ in spans) * SLOT_SECONDS
        # Alarm slots of this split
        a_slots = sum(b["alarm_slots"] for b, _ in spans)
        # Unmatched alarms whose start lies in one of this split's blocks
        n_unm = sum(1 for a0, _ in unmatched if any(b0 <= a0 <= b1 for _, (b0, b1) in spans))
        # Merged spans of every alarm episode (matched or not) starting in this split, seconds
        split_spans = [float((e - a0).total_seconds()) + SLOT_SECONDS for a0, e in alarms if any(b0 <= a0 <= b1 for _, (b0, b1) in spans)]
        # Record
        per_split[p] = {"false_alarm_episodes": n_unm, "monitored_days": sec / 86400.0, "false_alarms_per_day": (n_unm / (sec / 86400.0)) if sec > 0 else None,
                        "alarm_slot_fraction": (a_slots / (sec / SLOT_SECONDS)) if sec > 0 else None,
                        "alarm_episodes": len(split_spans), "merged_span_median_s": float(np.median(split_spans)) if split_spans else None, "merged_span_max_s": float(max(split_spans)) if split_spans else None}
    # Monitored days
    days = monitored_s / 86400.0
    # Rates
    recall = n_det / n_ep if n_ep else None
    # Precision over alarm episodes
    precision = n_det / (n_det + n_false) if (n_det + n_false) else None
    # F1
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else (0.0 if n_ep else None)
    # Delays among detected
    delays = [r["delay_s"] for r in results if r["detected"]]
    # Bootstrap
    boot = bootstrap([r["detected"] for r in results], n_false, settings["n_boot"], settings["seed"])
    # Return
    return {
        # Source
        "source": "persistence" if prefix == "pers_" else "model",
        # Which statistic raised the alarms
        "statistic": settings.get("statistic", "channel"),
        # Grid statistic details, exploratory
        "grid": None if grid is None else {
            # Label
            "note": "exploratory: motivated by test findings; the channel statistic is the primary result",
            # Design and calibration
            "design_label": settings["design_label"], "ref_percentile": settings["ref_percentile"], "ref_min_count": settings["ref_min_count"],
            # Floors
            "ref_floor_mag": settings["ref_floor_mag"], "ref_floor_ang": settings["ref_floor_ang"],
            # References in the (meters x channels) layout
            "ref_mag": grid["refs"]["ref_mag"].tolist(), "ref_ang": grid["refs"]["ref_ang"].tolist(),
            # Channels left out and channels at the floor
            "channels_without_reference": {k: [f"{meters[mi]}|{channels[ci]}" for mi, ci in zip(*np.nonzero(~np.isfinite(grid["refs"][f"ref_{k}"])))] for k in ("mag", "ang")},
            "channels_at_floor": {k: [f"{meters[mi]}|{channels[ci]}" for mi, ci in zip(*np.nonzero(grid["refs"][f"floored_{k}"]))] for k in ("mag", "ang")},
            # Threshold and its empirical calibration coverage
            "threshold": grid["threshold"], "calibration_coverage": grid["coverage"], "n_calibration": grid["n_calibration"], "meters_needed": grid["need"],
            # Test exceedance by available-meter count
            "alarm_slots_by_available_meters": {str(k): dict(v, rate=(v["alarm_slots"] / v["slots"]) if v["slots"] else None) for k, v in sorted(by_avail.items())},
        },
        # Calibration, with the per-channel coverage and counts
        "calibration": {"label": settings["calib_label"], "alpha": settings["alpha"],
                        "coverage_mag_mean": float(np.nanmean(thr["cov_mag"])) if np.isfinite(thr["cov_mag"]).any() else None,
                        "coverage_ang_mean": float(np.nanmean(thr["cov_ang"])) if np.isfinite(thr["cov_ang"]).any() else None,
                        "coverage_mag": thr["cov_mag"].tolist(), "coverage_ang": thr["cov_ang"].tolist(),
                        "n_mag": thr["n_mag"].tolist(), "n_ang": thr["n_ang"].tolist(),
                        "channels_without_threshold": missing},
        # Thresholds for the record, same (meters x channels) layout as the coverage and counts
        "thresholds": {"meters": meters, "channels": channels, "mag": thr["thr_mag"].tolist(), "ang": thr["thr_ang"].tolist()},
        # Blocks
        "blocks": per_block,
        # Episodes
        "episodes_total": int(len(episodes)), "episodes_in_scope": n_ep, "episodes_detected": n_det,
        # Episodes only partly covered by the scored blocks, excluded from alarms and monitored time
        "episodes_partial_excluded": [{"episode": int(e["episode"]), "first": e["first"].isoformat()} for _, e in partial.iterrows()],
        # Per-episode detail
        "episode_results": [{"episode": r["episode"], "first": r["first"].isoformat(), "detected": r["detected"], "delay_s": r["delay_s"], "alarm_start": r["alarm_start"].isoformat() if r["alarm_start"] is not None else None} for r in results],
        # False alarms, pooled over every test table
        "false_alarm_episodes": n_false, "monitored_days": days, "false_alarms_per_day": (n_false / days) if days > 0 else None,
        # The same per split prefix
        "per_split": per_split,
        # Unmatched alarm list, for inspection
        "false_alarm_list": [(s.isoformat(), e.isoformat()) for s, e in unmatched],
        # Scores
        "recall": recall, "precision": precision, "f1": f1,
        # Delay
        "delay_median_s": float(np.median(delays)) if delays else None, "delay_max_s": float(np.max(delays)) if delays else None,
        # Intervals
        "bootstrap": boot,
    }
