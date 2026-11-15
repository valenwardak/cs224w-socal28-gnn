# =============================================================================
# File:        socal_physicsscore.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-15
#
# Purpose:     Helper functions that score the physics baseline alone as an
#              event detector. Imported by main_physicsscore.py. The proposal
#              names a physics state estimator with no learning as a baseline,
#              so it needs a row in the comparison of detectors. Its residual
#              is one number per ten-second slot and circuit side (A and B),
#              read from <YYYY-MM>_physics.npz (main_physicsfeatures.py).
#
#              The alarm rules are those of socal_score.py, and its functions
#              are reused, not rewritten: the conformal threshold, the merging
#              of alarm slots, the episode windows, the one-to-one matching,
#              the sensitivity exclusion and the bootstrap.
#
# The method, in order:
#   1. Calibrate. Each side gets one threshold from its valid residuals on
#      the calibration block (February 2025) at level alpha, by the same
#      conformal rank rule as the models. A side with too few residuals
#      gets no threshold and never raises an alarm.
#   2. Detect. A side is monitored in a slot when its residual is valid and
#      the side has a threshold. It raises an alarm when the residual is
#      above the threshold. Three detectors are scored: "either side"
#      (monitored when at least one side is, alarm when at least one side
#      alarms), "side A" and "side B". The side alarms are combined before
#      the merging rule is applied. A side that is not valid adds no alarm.
#   3. Merge, match and count as socal_score.py does.
#   4. Coverage per episode. For every episode the output holds the slots of
#      its window, how many of them were monitored, and one of three
#      outcomes: caught, watched but not caught, or no valid physics value
#      in its window. A missed episode with no valid value was never seen.
#
# Limits stated in the output:
#   alpha applies to each side separately. The "either side" detector is
#   their combination, so its own false-alarm level is not alpha.
#   The physics residual has no magnitude and angle modes and no per-channel
#   statistic. It is one detector on its own residual.
#   A per-era count needs valid residuals on the quiet block of the era. The
#   output holds the valid count of every validation block, so a reader can
#   see where there are none.
#
# Functions:
#   block_label(split_name, block)        "<split>_<start>_<end>", as the run tables are named
#   load_block(physics_dir, block)        slot times, residuals and validity flags of one block
#   calibrate(arrays, alpha)              per side: threshold, residual count, coverage on the block
#   detector_masks(arrays, thresholds)    per detector: (monitored, alarm) per slot
#   score_detector(blocks, name, episodes, settings)
#                                         merging, matching and counting for one detector
#   score(physics_dir, splits, settings)  everything, returns the body of the score file
#   print_summary(body)                   one line per detector and per episode
# =============================================================================

# Standard library: paths
import os
# Third party: numerics
import numpy as np
# Third party: timestamps
import pandas as pd

# The model data module, for the bounds and the months of a split block
import socal_modeldata as sd
# The scoring module, for the alarm rules shared with the models
import socal_score as ss

# The two circuit sides, as the physics files name them
SIDES = ("a", "b")
# The three detectors and the sides each one watches
DETECTORS = {"either side": ("a", "b"), "side A": ("a",), "side B": ("b",)}
# The three outcomes of an episode
CAUGHT, WATCHED, UNSEEN = "caught", "watched, not caught", "no valid physics value in its window"


def block_label(split_name, block):
    """The label of a split block, in the form the run tables use: "<split>_<start>_<end>"."""
    # Split name and the two dates
    return f"{split_name}_{block['start']}_{block['end']}"


def load_block(physics_dir, block):
    """
    The physics values of one split block. Returns a dictionary: "t", the
    slot times as UTC datetime64 values, and per side "r_<side>" (the
    residual as a fraction) and "valid_<side>". A month file that is
    missing raises, so a gap in the files is never read as "no alarm".
    """
    # Bounds of the block in epoch seconds, end not included
    start_s, end_s = sd.block_seconds(block)
    # Pieces per month
    pieces = {"t": []}
    # Each month the block touches
    for month in sd.months_in_block(block):
        # The physics file of the month
        path = os.path.join(physics_dir, f"{month}_physics.npz")
        # It must be there
        if not os.path.exists(path):
            # Refuse
            raise FileNotFoundError(f"physics file missing for {month}: {path}")
        # Read it
        with np.load(path) as data:
            # Slot times in epoch seconds
            t = data["t"].astype(np.int64)
            # The slots of the block
            keep = (t >= start_s) & (t < end_s)
            # Keep their times
            pieces["t"].append(t[keep])
            # Each side
            for side in SIDES:
                # Its residuals
                pieces.setdefault(f"r_{side}", []).append(data[f"r_{side}"].astype(np.float64)[keep])
                # Its validity flags
                pieces.setdefault(f"valid_{side}", []).append(data[f"valid_{side}"].astype(bool)[keep])
    # One array per name
    arrays = {name: np.concatenate(parts) for name, parts in pieces.items()}
    # A block with no slot cannot be scored
    if len(arrays["t"]) == 0:
        # Refuse
        raise ValueError(f"no physics slots in block {block['start']} to {block['end']}")
    # Times as datetime64 values, which the scoring functions expect
    arrays["t"] = arrays["t"].astype("datetime64[s]").astype("datetime64[ns]")
    # Done
    return arrays


def calibrate(arrays, alpha):
    """
    Per side: the conformal threshold of its valid residuals on the
    calibration block, the number of residuals behind it, and the share of
    them at or below the threshold. The threshold is NaN when the side has
    too few residuals for alpha, and such a side never raises an alarm.
    """
    # Result
    out = {}
    # Each side
    for side in SIDES:
        # Its valid residuals
        values = arrays[f"r_{side}"][arrays[f"valid_{side}"]]
        # The threshold, by the rule the models use
        threshold = ss.conformal_threshold(values, alpha)
        # Record
        out[side] = {"threshold": threshold, "n_calibration": int(len(values)), "calibration_coverage": float((values <= threshold).mean()) if np.isfinite(threshold) else None}
    # Done
    return out


def detector_masks(arrays, thresholds):
    """
    Per detector, two flags per slot: monitored and alarm. A side is
    monitored where its residual is valid and it has a threshold, and
    alarms where the residual is above the threshold. A detector over two
    sides is monitored where at least one side is, and alarms where at
    least one side alarms.
    """
    # Per side
    monitored, alarm = {}, {}
    # Each side
    for side in SIDES:
        # Its threshold
        threshold = thresholds[side]["threshold"]
        # Monitored: a valid residual and a usable threshold
        monitored[side] = arrays[f"valid_{side}"] & bool(np.isfinite(threshold))
        # Alarm: monitored and above the threshold
        alarm[side] = monitored[side] & (arrays[f"r_{side}"] > threshold)
    # Each detector combines its sides
    return {name: (np.any([monitored[s] for s in sides], axis=0), np.any([alarm[s] for s in sides], axis=0)) for name, sides in DETECTORS.items()}


def score_detector(blocks, name, episodes, settings):
    """
    Merge, match and count for one detector, as socal_score.score_source
    does for a model. blocks is a list of (label, arrays, masks) for the
    test blocks. Returns the record of the detector, with the keys a score
    file of a model uses for the same measures.
    """
    # Window lengths
    before, after = pd.Timedelta(minutes=settings["before_minutes"]), pd.Timedelta(minutes=settings["after_minutes"])
    # Sensitivity exclusion intervals, empty by default
    exclusions = list(settings.get("exclude", []) or [])
    # The span of every block, from its first to its last slot
    spans = [(pd.to_datetime(arrays["t"][0], utc=True), pd.to_datetime(arrays["t"][-1], utc=True)) for _, arrays, _ in blocks]
    # Episodes whose whole window lies in a block, and those only partly covered
    scoped, partial = ss.episodes_in_scope(episodes, spans, before, after)
    # Episodes dropped from scope by an exclusion
    dropped = []
    # An episode whose window touches an exclusion leaves the scope, as in the scoring of the models
    if exclusions and len(scoped):
        # One flag per episode
        touch = np.array([any((e["first"] - before) < e0 and s0 <= (e["last"] + after) for s0, e0 in ss.exclusion_bounds(exclusions)) for _, e in scoped.iterrows()], dtype=bool)
        # Record them
        dropped = [{"episode": int(e["episode"]), "first": e["first"].isoformat()} for _, e in scoped[touch].iterrows()]
        # Their windows are removed from monitoring like those of partly covered episodes
        partial = pd.concat([partial, scoped[touch]]).reset_index(drop=True)
        # The scope without them
        scoped = scoped[~touch].reset_index(drop=True)
    # Alarms, records per block, and the final flags per block
    alarms, per_block, kept = [], [], []
    # Each block
    for label, arrays, masks in blocks:
        # Monitored and alarm flags of this detector
        monitored, alarm = masks[name]
        # Slots inside a partly covered window or an exclusion are neither monitored nor alarms
        monitored = monitored & ~ss.window_mask(arrays["t"], partial, before, after) & ~ss.exclusion_mask(arrays["t"], exclusions)
        # Alarms only on monitored slots
        alarm = alarm & monitored
        # Alarm episodes, merged by the shared rule, never across an exclusion
        found = ss.alarm_episodes_outside(arrays["t"], alarm, settings["merge_slots"], exclusions)
        # Keep them
        alarms.extend(found)
        # Keep the flags for the coverage per episode
        kept.append((arrays["t"], monitored))
        # The record of the block
        per_block.append({"label": label, "slots": int(len(arrays["t"])), "monitored_slots": int(monitored.sum()), "alarm_slots": int(alarm.sum()), "alarm_slot_fraction": (float(alarm.sum()) / float(monitored.sum())) if monitored.sum() else None, "alarm_episodes": len(found)})
    # One-to-one matching of alarms to episodes
    results, unmatched = ss.match(alarms, scoped, before, after)
    # Coverage per episode: the slots of its window and how many were monitored
    for result, (_, e) in zip(results, scoped.iterrows()):
        # Counts over the blocks
        window_slots, watched = 0, 0
        # Each block
        for t, monitored in kept:
            # The slots of the window in this block
            inside = ss.window_mask(t, scoped[scoped["episode"] == e["episode"]], before, after)
            # Add
            window_slots, watched = window_slots + int(inside.sum()), watched + int((monitored & inside).sum())
        # Record, with one of the three outcomes
        result.update({"window_slots": window_slots, "monitored_slots_in_window": watched, "outcome": CAUGHT if result["detected"] else (WATCHED if watched else UNSEEN)})
    # False alarms and monitored time per split, so the quiet period has its own rate
    per_split = {}
    # Each test split
    for prefix in settings["test_prefixes"]:
        # The blocks of this split with their spans
        mine = [(record, spans[i]) for i, record in enumerate(per_block) if record["label"].startswith(prefix)]
        # Monitored seconds and alarm slots of the split
        seconds, alarm_slots = sum(record["monitored_slots"] for record, _ in mine) * ss.SLOT_SECONDS, sum(record["alarm_slots"] for record, _ in mine)
        # Unmatched alarms that start in one of its blocks
        false_count = sum(1 for start, _ in unmatched if any(b0 <= start <= b1 for _, (b0, b1) in mine))
        # Record
        per_split[prefix] = {"false_alarm_episodes": false_count, "monitored_days": seconds / 86400.0, "false_alarms_per_day": (false_count / (seconds / 86400.0)) if seconds > 0 else None, "alarm_slot_fraction": (alarm_slots / (seconds / ss.SLOT_SECONDS)) if seconds > 0 else None}
    # Monitored days over all test blocks
    days = sum(record["monitored_slots"] for record in per_block) * ss.SLOT_SECONDS / 86400.0
    # The record of the detector
    return {
        # Which sides it watches
        "detector": name, "sides": list(DETECTORS[name]),
        # Episodes: in scope, caught, and how many had at least one monitored slot in their window
        "episodes_total": int(len(episodes)), "episodes_in_scope": len(results), "episodes_detected": sum(r["detected"] for r in results), "episodes_with_valid_value": sum(1 for r in results if r["monitored_slots_in_window"] > 0),
        # Per episode
        "episode_results": [{"episode": r["episode"], "first": r["first"].isoformat(), "detected": r["detected"], "delay_s": r["delay_s"], "alarm_start": r["alarm_start"].isoformat() if r["alarm_start"] is not None else None, "window_slots": r["window_slots"], "monitored_slots_in_window": r["monitored_slots_in_window"], "outcome": r["outcome"]} for r in results],
        # Episodes only partly covered by the blocks, and episodes dropped by an exclusion
        "episodes_partial_excluded": [{"episode": int(e["episode"]), "first": e["first"].isoformat()} for _, e in partial.iterrows()], "episodes_dropped_by_exclusion": dropped,
        # False alarms, pooled and per split
        "false_alarm_episodes": len(unmatched), "monitored_days": days, "false_alarms_per_day": (len(unmatched) / days) if days > 0 else None, "per_split": per_split,
        # Blocks and the unmatched alarms, for inspection
        "blocks": per_block, "false_alarm_list": [(start.isoformat(), end.isoformat()) for start, end in unmatched],
        # Uncertainty of the recall, by the bootstrap the models use
        "bootstrap": ss.bootstrap([r["detected"] for r in results], len(unmatched), settings["n_boot"], settings["seed"]),
    }


def score(physics_dir, splits, settings):
    """
    Score the physics baseline: calibrate each side on the calibration
    block, build the three detectors on every test block, and merge, match
    and count each one. Returns the body of the score file.
    """
    # Every block of the splits file by its label
    by_label = {block_label(name, block): block for name in ("train", "validation", "test_events", "test_normal") for block in splits.get(name, [])}
    # The calibration block must be one of them
    if settings["calib_label"] not in by_label:
        # Refuse
        raise ValueError(f"calibration block {settings['calib_label']} is not in the splits file")
    # An exclusion must not touch the calibration block, so the thresholds stay those of the full period
    for s0, e0 in ss.exclusion_bounds(list(settings.get("exclude", []) or [])):
        # Bounds of the calibration block
        b0, b1 = (pd.Timestamp(x, unit="s", tz="UTC") for x in sd.block_seconds(by_label[settings["calib_label"]]))
        # Overlap
        if s0 < b1 and b0 < e0:
            # Refuse
            raise ValueError(f"exclusion {s0.isoformat()} to {e0.isoformat()} overlaps the calibration block {settings['calib_label']}")
    # Thresholds per side from the calibration block
    thresholds = calibrate(load_block(physics_dir, by_label[settings["calib_label"]]), settings["alpha"])
    # The test blocks: every block whose label starts with a test prefix, with its values and its detector flags
    blocks = []
    # Each label in the order of the splits file
    for label, block in by_label.items():
        # Test blocks only
        if any(label.startswith(prefix) for prefix in settings["test_prefixes"]):
            # Values
            arrays = load_block(physics_dir, block)
            # Keep
            blocks.append((label, arrays, detector_masks(arrays, thresholds)))
    # There must be at least one
    if not blocks:
        # Refuse
        raise ValueError(f"no block of the splits file starts with {settings['test_prefixes']}")
    # The scored episodes of the event table
    episodes = ss.load_episodes(settings["events"])
    # Valid residuals of every validation block and every test block per side, so a reader sees where there are none
    counts = {label: {side: int(load_block(physics_dir, block)[f"valid_{side}"].sum()) for side in SIDES} for label, block in by_label.items() if label.startswith("validation_") or any(label.startswith(prefix) for prefix in settings["test_prefixes"])}
    # The body of the score file
    return {
        # Settings
        "settings": settings,
        # What kind of detector this is, and its limits
        "note": "physics residual detector: one residual per slot and circuit side, no magnitude or angle mode, no per-channel statistic; alpha applies to each side separately, so the false-alarm level of the either-side detector is not alpha",
        # Thresholds per side, as a fraction and in percent
        "thresholds": {side: dict(record, threshold_percent=(100.0 * record["threshold"]) if np.isfinite(record["threshold"]) else None) for side, record in thresholds.items()},
        # Valid residuals per block and side
        "valid_slots": counts,
        # The three detectors
        "detectors": {name: score_detector(blocks, name, episodes, settings) for name in DETECTORS},
    }


def print_summary(body):
    """Print the thresholds, one line per detector, and one line per detector and episode."""
    # Thresholds
    for side, record in body["thresholds"].items():
        # One line per side
        print(f"side {side.upper()}: threshold {record['threshold_percent']} percent from {record['n_calibration']} calibration residuals, coverage {record['calibration_coverage']}")
    # Each detector
    for name, d in body["detectors"].items():
        # Episodes and monitored time
        print(f"\n{name}: episodes caught {d['episodes_detected']} of {d['episodes_in_scope']} in scope, {d['episodes_with_valid_value']} with a valid value in the window; monitored days {d['monitored_days']:.1f}")
        # Per split
        for prefix, s in d["per_split"].items():
            # One line
            print(f"  {prefix}: {s['false_alarm_episodes']} false alarms over {s['monitored_days']:.1f} monitored days, per day {s['false_alarms_per_day']}, alarm-slot fraction {s['alarm_slot_fraction']}")
        # Per episode
        for r in d["episode_results"]:
            # One line
            print(f"  episode {r['episode']} ({r['first']}): {r['outcome']}; {r['monitored_slots_in_window']} of {r['window_slots']} window slots monitored" + (f"; delay {r['delay_s']:.0f} s" if r["detected"] else ""))
