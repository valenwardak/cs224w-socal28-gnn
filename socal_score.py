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
#   6b. Per-era calibration (scoring version 2, defined 2026-11-06, the
#      --era-calib option). Version 1 calibrates every test block on the
#      same block. Version 2 maps test-label prefixes to calibration
#      labels, so each test block is calibrated on a held-out quiet block
#      from its own era: the 2023 event blocks on November 1 to 14, 2023
#      (validation block 2 of splits_v2.json), the November 2024 block and
#      the quiet block on February 2025 as before. Each test label must
#      resolve to exactly one rule, or to the default calibration label
#      when no rule matches; two matching rules raise; an unused rule is
#      reported. Thresholds (and the grid threshold) are computed once per
#      calibration block and applied to the blocks assigned to it; the
#      grid references stay on the design block. Every calibration block
#      is checked against the run's training blocks, read from the block's
#      own summary file, and an overlap raises: calibration must be held
#      out. Version 2 is retrospective: the November 2023 calibration
#      follows the September 2023 events in time, and February 2025
#      follows the November 2024 events, so it does not show calibration
#      that would have been available at deployment. The quiet-block
#      result under version 2 is the version-1 result reused, not another
#      evaluation. With no rules the output is version 1, unchanged.
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
# Common-channel restriction (restrict_common=True, a sensitivity analysis).
#   The two statistics watch different channel sets by accident of which
#   meters are present in the design block and in the calibration block.
#   With the restriction, both statistics are limited, per residual kind,
#   to the channels that have both a channel threshold and a grid
#   reference; the grid threshold is then recalibrated on block 1 from the
#   restricted statistic, and the per-channel thresholds keep their own
#   calibration (dropping other channels does not change them). With the
#   same channel sets and min_meters, both statistics monitor the same
#   slots, so the comparison isolates aggregation and calibration from
#   sensor coverage. The channels dropped by the restriction are listed.
#
# Sensitivity exclusion (settings["exclude"], the --exclude option, added
# 2026-11-11 as a labeled sensitivity analysis; the primary results are the
# full-period results and are not changed by it).
#   The quiet block holds a real, unlogged change in the grid (July 29 to
#   about August 10, 2025) that the label-based evaluation counts as false
#   alarms. The course staff's guidance is to keep that period in the
#   primary count and to report a sensitivity analysis without it,
#   separately. The exclusion is a list of [start, end) UTC intervals. With
#   it given: thresholds, references and the grid threshold are exactly the
#   ones of the full-period scoring (calibration blocks are untouched, and
#   an exclusion that touches a calibration or design block raises); the
#   excluded slots are removed from the monitored slots, so monitored
#   exposure (slots and days) is recomputed and an excluded slot can be
#   neither an alarm nor monitored; alarm merging runs separately on each
#   stretch of kept slots, split at every excluded slot and at every
#   exclusion boundary that falls between two recorded slots, so no alarm
#   episode can bridge an exclusion even when no recorded slot lies inside
#   it (an alarm in progress at the boundary ends there, and one after it
#   starts fresh); an event episode whose window touches an exclusion is dropped
#   from scope and listed, and its window is removed from monitoring like a
#   partly covered episode. The exclusions, the excluded slot counts per
#   block and the dropped episodes are recorded in the output. With no
#   exclusion the code path and the output are unchanged. Before a
#   sensitivity result is written, main_score.py checks its protected
#   values (thresholds, missing-threshold lists, references, grid
#   thresholds, calibration assignments, settings other than the exclusion
#   record) against the saved primary file with protected_mismatches below:
#   every requested mode with non-empty model and persistence results and
#   the calibration evidence must be present on both sides (a field missing
#   from both sides fails), and then every protected value must be the
#   same; otherwise nothing is written.
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
#     label_bounds(label)                 UTC bounds of a block from its label
#     check_held_out(run_dir, label)      raise if a calibration block overlaps the run's training blocks
#     resolve_calibration(labels, settings)
#                                         calibration label per test label from the era rules
#     calibrate_source(run_dir, calib_label, prefix, other_prefix, settings, meters, channels, refs)
#                                         thresholds, exclusions, and the grid threshold for one calibration block
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
#     exclusion_bounds(exclusions)        [(start, end), ...] as UTC Timestamps, checked
#     exclusion_mask(t, exclusions)       (T,) bool, slots inside any [start, end) exclusion
#     kept_segments(t, exclusions)        index ranges of the stretches of kept slots, split at every exclusion boundary
#     alarm_episodes_outside(t, alarm, merge_slots, exclusions)
#                                         alarm episodes merged within each kept stretch only
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
#   Sensitivity check
#     evidence_findings(result, side, grid_expected)
#                                         missing or malformed calibration evidence in one saved result
#     protected_mismatches(new, primary, modes, ignore_settings)
#                                         structure, then differences in the values a sensitivity result must share with its primary file
# =============================================================================

# Standard library: ceilings for the conformal rank
import math
# Standard library: block summaries, for the held-out check
import json
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


def exclusion_bounds(exclusions):
    """[(start, end), ...] of the exclusion intervals as UTC Timestamps, naive bounds read as UTC, each start before its end."""
    # Result
    out = []
    # Loop over intervals
    for start, end in exclusions:
        # Bounds as timestamps
        s0, e0 = pd.Timestamp(start), pd.Timestamp(end)
        # Localize naive bounds as UTC
        s0 = s0.tz_localize("UTC") if s0.tzinfo is None else s0.tz_convert("UTC")
        # Same for the end
        e0 = e0.tz_localize("UTC") if e0.tzinfo is None else e0.tz_convert("UTC")
        # Must be ordered
        if not s0 < e0:
            # Raise
            raise ValueError(f"exclusion start must be before its end: {start} to {end}")
        # Keep
        out.append((s0, e0))
    # Return
    return out


def exclusion_mask(t, exclusions):
    """(T,) bool: slots that fall inside any of the [start, end) UTC exclusion intervals. Empty list gives all False."""
    # Timestamps
    times = pd.to_datetime(t, utc=True)
    # Start empty
    mask = np.zeros(len(times), dtype=bool)
    # Loop over intervals
    for s0, e0 in exclusion_bounds(exclusions):
        # Half-open interval
        mask |= np.asarray((times >= s0) & (times < e0))
    # Return
    return mask


def kept_segments(t, exclusions):
    """
    List of (first, last_plus_one) index ranges of the stretches of kept
    slots, in order. A stretch ends at an excluded slot, and also where an
    exclusion boundary falls between two consecutive recorded slots, so two
    kept slots on opposite sides of an exclusion are never in one stretch
    even when no recorded slot lies inside it.
    """
    # Timestamps
    times = pd.to_datetime(t, utc=True)
    # Kept slots
    kept = ~exclusion_mask(t, exclusions)
    # None kept
    if not kept.any():
        # Empty
        return []
    # Side of each slot: how many exclusion ends lie at or before it, so slots across any exclusion differ
    side = np.zeros(len(times), dtype=np.int64)
    # Count per slot, compared as timestamps so the time unit does not matter
    for _, e0 in exclusion_bounds(exclusions):
        # Add
        side += np.asarray(times >= e0, dtype=np.int64)
    # A stretch breaks where the kept flag changes or the side changes
    breaks = np.flatnonzero((kept[1:] != kept[:-1]) | (side[1:] != side[:-1])) + 1
    # Stretch bounds
    bounds = np.concatenate(([0], breaks, [len(kept)]))
    # Kept stretches only
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if kept[a]]


def alarm_episodes_outside(t, alarm, merge_slots, exclusions):
    """
    Alarm episodes merged within each stretch of kept slots only, so no
    episode can bridge an exclusion. Excluded slots are never alarms. With
    no exclusions this is exactly alarm_episodes on the whole table.
    """
    # No exclusions: the unchanged path
    if not exclusions:
        # Whole table
        return alarm_episodes(t, alarm, merge_slots)
    # Episodes
    out = []
    # Each kept stretch on its own
    for a, b in kept_segments(t, exclusions):
        # Merge inside the stretch
        out.extend(alarm_episodes(t[a:b], alarm[a:b], merge_slots))
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


def label_bounds(label):
    """
    UTC bounds [start, end) of a block from its label "<split>_<YYYY-MM-DD>_<YYYY-MM-DD>",
    whose dates are inclusive days in fixed Pacific Standard Time.
    """
    # Last two underscore parts are the dates
    parts = label.rsplit("_", 2)
    # Must have them
    if len(parts) != 3:
        # Raise
        raise ValueError(f"cannot read block dates from label {label}")
    # Start of the first day
    start = pd.Timestamp(parts[1]) + pd.Timedelta(hours=8)
    # Start of the day after the last
    end = pd.Timestamp(parts[2]) + pd.Timedelta(days=1, hours=8)
    # UTC
    return start.tz_localize("UTC"), end.tz_localize("UTC")


def check_held_out(run_dir, label):
    """
    Raise if the calibration block `label` overlaps any training block of
    the run, read from the block's own summary file (<label>.json, which
    score_split writes with the training options). Returns the splits file
    the run was trained with, for the record.
    """
    # Summary path
    path = os.path.join(run_dir, f"{label}.json")
    # Must exist
    if not os.path.exists(path):
        # Raise
        raise FileNotFoundError(f"no summary at {path}; cannot check that calibration block {label} is held out of training")
    # Read
    with open(path) as f:
        # Parse
        summary = json.load(f)
    # Training blocks as "start/end" UTC strings
    train = summary.get("options", {}).get("train_blocks")
    # Must be recorded
    if not train:
        # Raise
        raise ValueError(f"{path} records no training blocks; cannot check that {label} is held out")
    # Bounds of the calibration block
    c0, c1 = label_bounds(label)
    # Loop over training blocks
    for item in train:
        # Bounds
        t0, t1 = (pd.Timestamp(x) for x in item.split("/"))
        # Overlap
        if t0 < c1 and c0 < t1:
            # Raise: in-sample calibration is not allowed
            raise ValueError(f"calibration block {label} overlaps training block {item} of this run: calibration must be held out")
    # Splits file identity
    return summary.get("options", {}).get("splits_file")


def resolve_calibration(labels, settings):
    """
    Calibration label per test label. settings["era_calib"] is an ordered
    list of [prefix, calib_label] rules; a test label takes the rule whose
    prefix it starts with, and the default settings["calib_label"] when no
    rule matches. Two matching rules raise. Returns (mapping, unused rules).
    """
    # Rules
    rules = settings.get("era_calib") or []
    # Mapping
    mapping = {}
    # Rules that matched something
    used = set()
    # Loop over labels
    for label in labels:
        # Matching rules
        hits = [(p, c) for p, c in rules if label.startswith(p)]
        # Ambiguous
        if len(hits) > 1:
            # Raise
            raise ValueError(f"test block {label} matches more than one era rule: {hits}")
        # One rule
        if hits:
            # Record
            mapping[label] = hits[0][1]
            # Used
            used.add(tuple(hits[0]))
        # Default
        else:
            # Record
            mapping[label] = settings["calib_label"]
    # Unused rules
    unused = [list(r) for r in rules if tuple(r) not in used]
    # Done
    return mapping, unused


def calibrate_source(run_dir, calib_label, prefix, other_prefix, settings, meters, channels, refs):
    """
    Thresholds for one source on one calibration block, with the common
    mask, the optional common-channel restriction (which also edits refs
    in place), the channels left without a threshold, and the grid
    threshold when the grid statistic is in use. Returns a dict with thr,
    missing, restriction, grid.
    """
    # Calibration table
    cal = load_table(run_dir, calib_label)
    # Its axes must match
    if table_axes(cal) != (meters, channels):
        # Raise
        raise ValueError(f"calibration block {calib_label} has different meters or channels from the other tables")
    # Calibration residuals
    arrays = residual_arrays(cal, meters, channels, prefix)
    # Common slots
    if other_prefix is not None and settings["common"]:
        # Mask against the other source
        common_mask(arrays, residual_arrays(cal, meters, channels, other_prefix))
    # Thresholds, per channel
    thr = calibrate(arrays, settings["alpha"])
    # Grid statistic asked for
    use_grid = settings.get("statistic", "channel") == "grid"
    # Restriction flag
    restrict = bool(settings.get("restrict_common", False))
    # Restriction: keep only channels with both a threshold and a reference, per kind, for both statistics
    restriction = None
    # Apply
    if restrict:
        # Record
        restriction = {}
        # Per kind
        for kind in ("mag", "ang"):
            # Channels with both
            keep = np.isfinite(thr[f"thr_{kind}"]) & np.isfinite(refs[f"ref_{kind}"])
            # Dropped by the restriction: had one but not the other
            dropped = (np.isfinite(thr[f"thr_{kind}"]) | np.isfinite(refs[f"ref_{kind}"])) & ~keep
            # Apply to the thresholds
            thr[f"thr_{kind}"] = np.where(keep, thr[f"thr_{kind}"], np.nan).astype(np.float32)
            # Apply to the references
            refs[f"ref_{kind}"] = np.where(keep, refs[f"ref_{kind}"], np.nan).astype(np.float32)
            # A dropped channel is no longer "at the floor, still included"
            refs[f"floored_{kind}"] = refs[f"floored_{kind}"] & keep
            # Record
            restriction[kind] = {"channels_kept": int(keep.sum()), "channels_dropped": [f"{meters[mi]}|{channels[ci]}" for mi, ci in zip(*np.nonzero(dropped))]}
    # Channels without a threshold, by kind, after any restriction
    missing = {kind: [f"{meters[mi]}|{channels[ci]}" for mi, ci in zip(*np.nonzero(~np.isfinite(thr[f"thr_{kind}"])))] for kind in ("mag", "ang")}
    # Grid statistic: one threshold from this calibration block
    grid = None
    # Asked for
    if use_grid:
        # Meters needed
        need = max(2, settings["min_meters"])
        # Grid scores on the calibration block
        gscore, gavail = grid_scores(arrays, refs, settings["mode"], need)
        # One threshold
        gthr = conformal_threshold(gscore, settings["alpha"])
        # Without a threshold the grid statistic cannot alarm; a silent zero-alarm result would mislead, so it raises
        if not np.isfinite(gthr):
            # Raise with the numbers
            raise ValueError(f"grid threshold cannot be calibrated: {int(np.isfinite(gscore).sum())} valid calibration slots on {calib_label} at alpha {settings['alpha']} (need at least {math.ceil(1.0 / settings['alpha']) - 1})")
        # Empirical calibration coverage
        gcov = float(np.mean(gscore[np.isfinite(gscore)] <= gthr))
        # Record
        grid = {"refs": refs, "threshold": gthr, "coverage": gcov, "n_calibration": int(np.isfinite(gscore).sum()), "need": need}
    # Done
    return {"thr": thr, "missing": missing, "restriction": restriction, "grid": grid}


def calibration_record(label, cset, settings, meters, channels):
    """The calibration summary of one calibration set: label, alpha, mean and per-channel coverage, counts, channels without a threshold."""
    # Thresholds
    thr = cset["thr"]
    # Record
    return {"label": label, "alpha": settings["alpha"],
            "coverage_mag_mean": float(np.nanmean(thr["cov_mag"])) if np.isfinite(thr["cov_mag"]).any() else None,
            "coverage_ang_mean": float(np.nanmean(thr["cov_ang"])) if np.isfinite(thr["cov_ang"]).any() else None,
            "coverage_mag": thr["cov_mag"].tolist(), "coverage_ang": thr["cov_ang"].tolist(),
            "n_mag": thr["n_mag"].tolist(), "n_ang": thr["n_ang"].tolist(),
            "channels_without_threshold": cset["missing"]}


def score_source(run_dir, prefix, settings, other_prefix=None):
    """
    The whole pipeline for one source ("" model, "pers_" persistence):
    calibrate on each calibration block in use, detect on every test
    table with the thresholds of its own calibration block, match against
    the event episodes, count false alarms over monitored time. With
    other_prefix given and settings["common"] true, residuals are masked
    to the slots where both sources are valid. settings is a dict with
    alpha, mode, min_meters, merge_slots, before_minutes, after_minutes,
    calib_label, test_prefixes, events, n_boot, seed, common, and the
    optional era_calib rules (scoring version 2). Returns a dict of results.
    """
    # Windows
    before, after = pd.Timedelta(minutes=settings["before_minutes"]), pd.Timedelta(minutes=settings["after_minutes"])
    # Axes from the default calibration table
    meters, channels = table_axes(load_table(run_dir, settings["calib_label"]))
    # Grid statistic asked for, or the common-channel restriction, which needs the references too
    use_grid = settings.get("statistic", "channel") == "grid"
    # Restriction flag
    restrict = bool(settings.get("restrict_common", False))
    # Events
    episodes = load_episodes(settings["events"])
    # Sensitivity exclusion intervals, [start, end) UTC; empty by default
    exclusions = list(settings.get("exclude", []) or [])
    # Test tables, every requested prefix must have at least one
    labels = [lab for p in settings["test_prefixes"] for lab in list_tables(run_dir, p)]
    # Calibration block per test label
    calib_of, unused_rules = resolve_calibration(labels, settings)
    # Distinct calibration blocks, the default first when present
    calib_labels = sorted(set(calib_of.values()), key=lambda c: (c != settings["calib_label"], c))
    # Every calibration block must be held out of training; record the splits identity
    splits_files = {c: check_held_out(run_dir, c) for c in calib_labels}
    # An exclusion must not touch a calibration block or the design block: thresholds and references stay those of the full-period scoring
    for c in list(calib_labels) + ([settings["design_label"]] if (use_grid or restrict) else []):
        # Block bounds
        b0, b1 = label_bounds(c)
        # Each interval
        for s0, e0 in exclusion_bounds(exclusions):
            # Overlap
            if s0 < b1 and b0 < e0:
                # Raise
                raise ValueError(f"exclusion {s0.isoformat()} to {e0.isoformat()} overlaps calibration or design block {c}; the sensitivity analysis keeps calibration unchanged")
    # One calibration per block in use; each gets its own copy of the references, since the restriction edits them
    cal_sets = {}
    # Loop
    for c in calib_labels:
        # References from the design block when needed, fresh per calibration block
        refs = None
        # Build
        if use_grid or restrict:
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
        # Calibrate
        cal_sets[c] = calibrate_source(run_dir, c, prefix, other_prefix, settings, meters, channels, refs)
    # The default calibration set, for the top-level fields version 1 readers expect
    main_label = calib_labels[0]
    # Its pieces
    thr, missing, restriction, grid = cal_sets[main_label]["thr"], cal_sets[main_label]["missing"], cal_sets[main_label]["restriction"], cal_sets[main_label]["grid"]
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
    # Episodes whose window touches an exclusion leave the scope and are treated like partly covered ones
    dropped = []
    # Only with exclusions
    if exclusions and len(scoped):
        # Flags
        touch = np.zeros(len(scoped), dtype=bool)
        # Each episode
        for i, (_, e) in enumerate(scoped.iterrows()):
            # Window bounds
            w0, w1 = e["first"] - before, e["last"] + after
            # Each interval
            for s0, e0 in exclusion_bounds(exclusions):
                # Touching
                touch[i] |= bool(w0 < e0 and s0 <= w1)
        # Record
        dropped = [{"episode": int(e["episode"]), "first": e["first"].isoformat()} for _, e in scoped[touch].iterrows()]
        # Their windows join the partial set for monitoring, and they leave the scope
        partial = pd.concat([partial, scoped[touch]]).reset_index(drop=True)
        # Scope
        scoped = scoped[~touch].reset_index(drop=True)
    # Alarms, monitored time, per-block records over all test tables
    alarms, monitored_s, per_block = [], 0.0, []
    # Exceedance counts by available-meter count, grid statistic only: pooled, and per calibration block
    by_avail = {}
    # Per calibration block
    by_avail_cal = {c: {} for c in calib_labels}
    # Excluded slots over all tables
    excluded_total = 0
    # Loop over tables
    for label, arr in tables:
        # Slots inside an exclusion interval, none by default
        excl = exclusion_mask(arr["t"], exclusions)
        # The calibration set this block uses
        cset = cal_sets[calib_of[label]]
        # Its thresholds and grid
        b_thr, b_grid = cset["thr"], cset["grid"]
        # Per-channel statistic
        if b_grid is None:
            # Firing channels
            fires = exceedances(arr, b_thr, settings["mode"])
            # Meters firing per slot
            score = slot_scores(fires)
            # Monitored slots: enough meters with a residual on a thresholded channel
            mon = monitored_slots(arr, b_thr, settings["mode"], settings["min_meters"])
            # Slots inside a partly covered event window are neither alarms nor monitored
            mon &= ~window_mask(arr["t"], partial, before, after)
            # Excluded slots are neither alarms nor monitored
            mon &= ~excl
            # Alarm slots: enough meters firing, and the slot is monitored
            alarm = (score >= settings["min_meters"]) & mon
        # Grid statistic
        else:
            # Scores and availability
            gscore, gavail = grid_scores(arr, b_grid["refs"], settings["mode"], b_grid["need"])
            # Monitored: enough available meters, outside partial windows
            mon = np.isfinite(gscore) & ~window_mask(arr["t"], partial, before, after)
            # Excluded slots are neither alarms nor monitored
            mon &= ~excl
            # Alarm: score above this calibration block's threshold
            alarm = mon & (np.nan_to_num(gscore, nan=-np.inf) > b_grid["threshold"])
            # Exceedance by available-meter count, pooled and for this block's calibration set
            for a in np.unique(gavail[mon]):
                # Slots with this count
                sel = mon & (gavail == a)
                # Counts
                n_sel, n_alarm = int(sel.sum()), int((alarm & sel).sum())
                # Pooled tally
                rec = by_avail.setdefault(int(a), {"slots": 0, "alarm_slots": 0})
                # Add
                rec["slots"] += n_sel; rec["alarm_slots"] += n_alarm
                # Per calibration set
                rec_c = by_avail_cal[calib_of[label]].setdefault(int(a), {"slots": 0, "alarm_slots": 0})
                # Add
                rec_c["slots"] += n_sel; rec_c["alarm_slots"] += n_alarm
        # Episodes, merged within each stretch of kept slots so none bridges an exclusion, with or without recorded slots inside it
        eps = alarm_episodes_outside(arr["t"], alarm, settings["merge_slots"], exclusions)
        # Count
        excluded_total += int(excl.sum())
        # Keep
        alarms.extend(eps)
        # Monitored seconds
        monitored_s += float(mon.sum()) * SLOT_SECONDS
        # Merged episode spans in seconds, slot-inclusive
        spans = [float((e - s0).total_seconds()) + SLOT_SECONDS for s0, e in eps]
        # Per-block record, with its calibration block, the alarm-slot fraction, and the merged-span summary
        per_block.append({"label": label, "calib_label": calib_of[label], "slots": int(len(arr["t"])), "monitored_slots": int(mon.sum()), "alarm_slots": int(alarm.sum()),
                          "alarm_slot_fraction": (float(alarm.sum()) / float(mon.sum())) if mon.sum() else None,
                          "alarm_episodes": len(eps), "merged_span_median_s": float(np.median(spans)) if spans else None, "merged_span_max_s": float(max(spans)) if spans else None,
                          **({"excluded_slots": int(excl.sum())} if exclusions else {})})
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
        # Common-channel restriction, when applied
        "restriction": None if restriction is None else dict(restriction, note="sensitivity analysis: both statistics limited to channels with both a threshold and a reference"),
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
            # Test exceedance by available-meter count, pooled over every test block (and so over calibration sets under version 2)
            "alarm_slots_by_available_meters": {str(k): dict(v, rate=(v["alarm_slots"] / v["slots"]) if v["slots"] else None) for k, v in sorted(by_avail.items())},
            "alarm_slots_by_available_meters_note": "pooled over all test blocks; under version 2 the per-calibration tables are in calibrations",
        },
        # Calibration of the default block, with the per-channel coverage and counts (version 1 layout)
        "calibration": calibration_record(main_label, cal_sets[main_label], settings, meters, channels),
        # Thresholds of the default block for the record, same (meters x channels) layout as the coverage and counts
        "thresholds": {"meters": meters, "channels": channels, "mag": thr["thr_mag"].tolist(), "ang": thr["thr_ang"].tolist()},
        # Scoring version 2: every calibration block in use, the block-to-calibration map, and the held-out check
        "calibrations": {c: dict(calibration_record(c, cal_sets[c], settings, meters, channels),
                                 thresholds={"mag": cal_sets[c]["thr"]["thr_mag"].tolist(), "ang": cal_sets[c]["thr"]["thr_ang"].tolist()},
                                 grid_threshold=None if cal_sets[c]["grid"] is None else cal_sets[c]["grid"]["threshold"],
                                 grid_calibration_coverage=None if cal_sets[c]["grid"] is None else cal_sets[c]["grid"]["coverage"],
                                 grid_n_calibration=None if cal_sets[c]["grid"] is None else cal_sets[c]["grid"]["n_calibration"],
                                 # The references as this set uses them, after any restriction: exclusions and floor flags
                                 grid_channels_without_reference=None if cal_sets[c]["grid"] is None else {k: [f"{meters[mi]}|{channels[ci]}" for mi, ci in zip(*np.nonzero(~np.isfinite(cal_sets[c]["grid"]["refs"][f"ref_{k}"])))] for k in ("mag", "ang")},
                                 grid_channels_at_floor=None if cal_sets[c]["grid"] is None else {k: [f"{meters[mi]}|{channels[ci]}" for mi, ci in zip(*np.nonzero(cal_sets[c]["grid"]["refs"][f"floored_{k}"]))] for k in ("mag", "ang")},
                                 # Test exceedance by available-meter count for the blocks this set calibrated
                                 grid_alarm_slots_by_available_meters=None if cal_sets[c]["grid"] is None else {str(k): dict(v, rate=(v["alarm_slots"] / v["slots"]) if v["slots"] else None) for k, v in sorted(by_avail_cal[c].items())},
                                 restriction=cal_sets[c]["restriction"],
                                 held_out_of_training=True, splits_file=splits_files[c]) for c in calib_labels},
        # Which calibration block each test block used
        "block_calibration": calib_of,
        # Era rules that matched no test block
        "era_rules_unused": unused_rules,
        # Sensitivity exclusion, only when given: intervals, excluded slots, episodes dropped from scope
        **({"exclusions": {"intervals": [[str(a), str(b)] for a, b in exclusions], "excluded_slots": excluded_total, "episodes_dropped": dropped,
                            "note": "sensitivity analysis: excluded slots removed from monitoring and alarms, calibration unchanged, no alarm episode bridges an exclusion; the full-period results are primary"}} if exclusions else {}),
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


# -----------------------------------------------------------------------------
# Sensitivity check
# -----------------------------------------------------------------------------

def evidence_findings(result, side, grid_expected):
    """
    Missing or malformed calibration evidence in one mode/source result,
    as the saved JSON holds it; side names the file for the message. The
    protected values can only be compared when both sides carry them, so a
    field missing from both sides is a finding, not a match. Allowed as the
    scoring writes them: null thresholds and null coverage for channels
    without a threshold, empty missing-channel lists, and under the channel
    statistic a null grid record with null grid fields in the calibration
    records; under the grid statistic the grid record and those fields are
    required.
    """
    # Findings
    out = []
    # Not a record
    if not isinstance(result, dict) or not result:
        # Finding
        return [f"{side}: result record missing or empty"]
    # Per-channel thresholds of the default block: axes and two (meters x channels) lists
    thr = result.get("thresholds")
    # Shape
    if not isinstance(thr, dict) or not all(isinstance(thr.get(k), list) and thr.get(k) for k in ("meters", "channels", "mag", "ang")):
        # Finding
        out.append(f"{side}: thresholds record missing or lacking non-empty meters, channels, mag and ang lists")
    # Missing-threshold lists of the default block
    cal = result.get("calibration")
    # Shape
    if not isinstance(cal, dict) or not isinstance(cal.get("channels_without_threshold"), dict) or not all(isinstance(cal["channels_without_threshold"].get(k), list) for k in ("mag", "ang")):
        # Finding
        out.append(f"{side}: calibration.channels_without_threshold missing or lacking mag and ang lists")
    # Block-to-calibration map and the per-block labels
    bc = result.get("block_calibration")
    # Shape
    if not isinstance(bc, dict) or not bc:
        # Finding
        out.append(f"{side}: block_calibration missing or empty")
    # Blocks
    blocks = result.get("blocks")
    # Shape
    if not isinstance(blocks, list) or not blocks or not all(isinstance(b, dict) and "label" in b and "calib_label" in b for b in blocks):
        # Finding
        out.append(f"{side}: blocks missing, empty, or lacking label and calib_label")
    # Calibration records: one per block in use, each with its thresholds, missing lists and grid fields
    cals = result.get("calibrations")
    # Shape
    if not isinstance(cals, dict) or not cals:
        # Finding
        out.append(f"{side}: calibrations missing or empty")
    # Each record
    else:
        # Must cover exactly the calibration blocks the test blocks use
        if isinstance(bc, dict) and bc and set(cals) != set(bc.values()):
            # Finding
            out.append(f"{side}: calibrations cover {sorted(cals)} but the blocks use {sorted(set(bc.values()))}")
        # Loop
        for c, rec in cals.items():
            # Record
            if not isinstance(rec, dict):
                # Finding
                out.append(f"{side}: calibrations[{c}] is not a record")
                # Next
                continue
            # Always-required fields
            for f in ("label", "alpha", "channels_without_threshold", "thresholds", "held_out_of_training", "splits_file"):
                # Missing
                if f not in rec:
                    # Finding
                    out.append(f"{side}: calibrations[{c}].{f} missing")
            # Thresholds and missing lists in shape
            if not (isinstance(rec.get("thresholds"), dict) and all(isinstance(rec["thresholds"].get(k), list) and rec["thresholds"].get(k) for k in ("mag", "ang"))):
                # Finding
                out.append(f"{side}: calibrations[{c}].thresholds lacking non-empty mag and ang lists")
            # Missing lists
            if not (isinstance(rec.get("channels_without_threshold"), dict) and all(isinstance(rec["channels_without_threshold"].get(k), list) for k in ("mag", "ang"))):
                # Finding
                out.append(f"{side}: calibrations[{c}].channels_without_threshold lacking mag and ang lists")
            # Grid fields: present as keys always; non-null under the grid statistic, null under the channel statistic
            for f in ("grid_threshold", "grid_calibration_coverage", "grid_n_calibration", "grid_channels_without_reference", "grid_channels_at_floor"):
                # Missing key
                if f not in rec:
                    # Finding
                    out.append(f"{side}: calibrations[{c}].{f} missing")
                # Null where the grid statistic needs a value
                elif grid_expected and rec[f] is None:
                    # Finding
                    out.append(f"{side}: calibrations[{c}].{f} is null under the grid statistic")
                # A value where the channel statistic writes null
                elif not grid_expected and rec[f] is not None:
                    # Finding
                    out.append(f"{side}: calibrations[{c}].{f} is set under the channel statistic")
    # Grid record: required under the grid statistic, null under the channel statistic
    grid = result.get("grid", "<missing>")
    # Missing key
    if grid == "<missing>":
        # Finding
        out.append(f"{side}: grid key missing")
    # Grid statistic
    elif grid_expected:
        # Record with references and threshold
        if not isinstance(grid, dict):
            # Finding
            out.append(f"{side}: grid record missing under the grid statistic")
        # Fields
        else:
            # References as non-empty lists, threshold and counts as numbers, channel lists as dicts
            if not all(isinstance(grid.get(k), list) and grid.get(k) for k in ("ref_mag", "ref_ang")):
                # Finding
                out.append(f"{side}: grid.ref_mag and grid.ref_ang must be non-empty lists")
            # Numbers
            for f in ("threshold", "calibration_coverage", "n_calibration", "meters_needed"):
                # Missing or null
                if not isinstance(grid.get(f), (int, float)) or isinstance(grid.get(f), bool):
                    # Finding
                    out.append(f"{side}: grid.{f} missing or not a number")
            # Channel lists and design settings
            for f in ("channels_without_reference", "channels_at_floor", "design_label", "ref_percentile", "ref_min_count", "ref_floor_mag", "ref_floor_ang"):
                # Missing
                if f not in grid:
                    # Finding
                    out.append(f"{side}: grid.{f} missing")
    # Channel statistic
    elif grid is not None:
        # Finding
        out.append(f"{side}: grid record present under the channel statistic")
    # Restriction and era bookkeeping keys must exist (null and empty are their usual values)
    for f in ("restriction", "era_rules_unused"):
        # Missing
        if f not in result:
            # Finding
            out.append(f"{side}: {f} key missing")
    # Return
    return out


def protected_mismatches(new, primary, modes, ignore_settings=("exclude", "exclude_note")):
    """
    Compare a sensitivity result (the dict main_score.py is about to write,
    after its JSON cleaning) with the saved primary result it must agree
    with. First the structure: both files must hold a settings record,
    every requested mode with a non-empty model and persistence result, and
    in every such result the calibration evidence that evidence_findings
    requires (a field missing from both sides fails). Then the values: the
    settings other than the exclusion record, and per mode and source the
    thresholds, the missing-threshold lists, every calibration record's
    thresholds and grid threshold, the grid references and grid threshold,
    and the calibration assignment of every block. Returns a list of
    findings, each naming the mode, source and field; an empty list means
    every protected value is present on both sides and the same. Nothing
    is substituted.
    """
    # Findings
    out = []
    # Settings records
    a_set, b_set = new.get("settings"), primary.get("settings")
    # Both must exist
    for side, rec in (("sensitivity", a_set), ("primary", b_set)):
        # Missing
        if not isinstance(rec, dict) or not rec:
            # Finding
            out.append(f"{side} file has no settings record")
    # Compare settings when both exist
    if isinstance(a_set, dict) and isinstance(b_set, dict):
        # Without the exclusion record
        a = {k: v for k, v in a_set.items() if k not in ignore_settings}
        # Every key on either side
        for k in sorted(set(a) | set(b_set)):
            # Differ or missing
            if k not in a or k not in b_set or a[k] != b_set[k]:
                # Finding
                out.append(f"settings.{k}: sensitivity {a.get(k, '<missing>')!r} against primary {b_set.get(k, '<missing>')!r}")
    # The statistic decides which calibration evidence is required
    grid_expected = isinstance(a_set, dict) and a_set.get("statistic") == "grid"
    # Protected fields of a result, top level
    top = ["thresholds", "block_calibration", "era_rules_unused", "restriction"]
    # Protected fields inside calibration records
    cal_fields = ["label", "alpha", "channels_without_threshold", "thresholds", "grid_threshold", "grid_calibration_coverage", "grid_n_calibration",
                  "grid_channels_without_reference", "grid_channels_at_floor", "restriction", "held_out_of_training", "splits_file"]
    # Protected fields of the grid record
    grid_fields = ["design_label", "ref_percentile", "ref_min_count", "ref_floor_mag", "ref_floor_ang", "ref_mag", "ref_ang",
                   "channels_without_reference", "channels_at_floor", "threshold", "calibration_coverage", "n_calibration", "meters_needed"]
    # Mode tables
    a_modes = new.get("modes") if isinstance(new.get("modes"), dict) else {}
    # Primary
    b_modes = primary.get("modes") if isinstance(primary.get("modes"), dict) else {}
    # Every requested mode
    if not modes:
        # Finding
        out.append("no modes requested for the check")
    # Loop
    for mode in modes:
        # Sources
        for src in ("model", "persistence"):
            # Records
            r = a_modes.get(mode, {}).get(src) if isinstance(a_modes.get(mode), dict) else None
            # Primary
            p = b_modes.get(mode, {}).get(src) if isinstance(b_modes.get(mode), dict) else None
            # Missing or empty on either side
            missing = [name for name, rec in (("sensitivity", r), ("primary", p)) if not isinstance(rec, dict) or not rec]
            # Finding
            if missing:
                # Record
                out.append(f"{mode}/{src}: required mode/source missing or empty in the {' and '.join(missing)} result")
                # Next
                continue
            # Evidence on both sides before any value is compared
            evidence = evidence_findings(r, f"{mode}/{src} sensitivity", grid_expected) + evidence_findings(p, f"{mode}/{src} primary", grid_expected)
            # Stop here for this pair
            if evidence:
                # Record
                out.extend(evidence)
                # Next
                continue
            # Top-level fields
            for f in top:
                # Differ
                if r.get(f, "<missing>") != p.get(f, "<missing>"):
                    # Finding
                    out.append(f"{mode}/{src}: {f} differs from the primary file")
            # Version-1 calibration record of the default block
            if r["calibration"]["channels_without_threshold"] != p["calibration"]["channels_without_threshold"]:
                # Finding
                out.append(f"{mode}/{src}: calibration.channels_without_threshold differs from the primary file")
            # Calibration records, every block in use
            rc, pc = r["calibrations"], p["calibrations"]
            # Same set of blocks
            if set(rc) != set(pc):
                # Finding
                out.append(f"{mode}/{src}: calibration blocks {sorted(rc)} against primary {sorted(pc)}")
            # Fields of the shared ones
            for c in sorted(set(rc) & set(pc)):
                # Each field
                for f in cal_fields:
                    # Differ
                    if rc[c].get(f, "<missing>") != pc[c].get(f, "<missing>"):
                        # Finding
                        out.append(f"{mode}/{src}: calibrations[{c}].{f} differs from the primary file")
            # Grid record under the grid statistic
            if grid_expected:
                # Each field
                for f in grid_fields:
                    # Differ
                    if r["grid"].get(f, "<missing>") != p["grid"].get(f, "<missing>"):
                        # Finding
                        out.append(f"{mode}/{src}: grid.{f} differs from the primary file")
            # Calibration label of every block
            rb = {b["label"]: b["calib_label"] for b in r["blocks"]}
            # Primary
            pb = {b["label"]: b["calib_label"] for b in p["blocks"]}
            # Differ
            if rb != pb:
                # Finding
                out.append(f"{mode}/{src}: per-block calibration labels differ from the primary file")
    # Return
    return out
