# =============================================================================
# File:        socal_modeldata.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-22
#
# Purpose:     The data side for every learned model: reading the feature
#              files, scaling, building model inputs, cutting training
#              windows, and iterating over the blocks of a split. Imported
#              by socal_models.py (the GRU baseline, the GNNs, training and
#              scoring) and usable on its own by any other model code.
#
# Data it reads:
#              data/features/<YYYY-MM>.npz from main_features.py, with
#                t        int64   (T,)        slot start, seconds since 1970 UTC
#                rms      float16 (T, M, C)   magnitude, volts or amps
#                ang      float32 (T, M, C)   aligned angle, degrees, NaN where rejected
#                mask     bool    (T, M, C)   reading exists
#                angle_ok bool    (T, M, C)   angle passed the checks
#              data/features/manifest.json   meter order (M) and channel order (C)
#              splits.json                   time blocks per split, UTC bounds
#              graph.json (optional)         clamp ratings, for the current floor
#
# Model input, per slot, meter, channel, F = 5 numbers:
#              [ magnitude / scale,  cos(angle),  sin(angle),  mag_valid,  ang_valid ]
#              The two flags are the validity flags, not the raw presence mask:
#              mag_valid is present and finite, ang_valid is angle_ok and
#              finite and mag_valid. Where a flag is 0 the values before it
#              are 0. So a model, and the persistence baseline, can trust the
#              flags to mean "this number is usable". The raw presence mask
#              stays in the feature files for the silence detector in the
#              scoring script, which reads it there. Per meter that is
#              C * F = 30 numbers per slot.
#
# Scaling:     One positive number per meter and channel, from the train
#              blocks only: the 95th percentile of valid readings, with a
#              floor so a channel that usually sits near zero does not blow
#              up. The floor is 1 volt for voltage channels and 1 percent of
#              the clamp rating for current channels (from graph.json), or
#              --scale-floor when no graph is given. A channel with no valid
#              train reading at all (a meter absent from the train months)
#              gets a nominal fallback instead: the nominal voltage of the
#              meter's bus for voltage channels, a tenth of the clamp rating
#              for current channels, the floor when no graph is given.
#              scale.json records the numbers and which channels used the
#              fallback, so every model and the scoring use the same values.
#
# Windows:     A training example is one meter's last `history` slots and
#              the slot `horizon` steps ahead as the target. A window is
#              usable when no run of `max_gap` or more consecutive slots
#              without a reading lies inside its history (absent slots
#              before the window do not count), and the target slot has at
#              least one valid magnitude. Blocks are loaded one month at a
#              time, so windows never cross a block or month boundary; the
#              first history + horizon - 1 targets of each month are lost,
#              about ten minutes, which is accepted.
#
# Checks:      Every month file is checked on load: slot times strictly
#              increasing in ten-second steps and inside the named month,
#              arrays of matching shape and, with a manifest, the right
#              meter and channel counts.
#              Presence (the feature-file mask) is kept apart from magnitude
#              validity (present and finite). Missing month files raise
#              unless allow_missing is set. scale.json records the meter and
#              channel order and is checked against the manifest on load.
#              Split blocks are checked for overlap.
#
# Functions:
#   Files
#     load_manifest(features_dir)            meter and channel order
#     load_splits(path)                      the splits.json dictionary, overlap checked
#     check_no_overlap(splits)               raise if two blocks overlap
#     block_seconds(block)                   (start, end) of a block as epoch seconds
#     month_bounds(month)                    (start, end) of a feature-file month, fixed PST convention
#     months_in_block(block)                 "YYYY-MM" labels a block touches, same convention
#     load_month(features_dir, month, allow_missing, manifest)
#                                            one month's arrays, checked and cleaned
#     slice_block(arrays, start_s, end_s)    rows of a month inside a block
#   Scaling
#     channel_floor(manifest, graph_path, floor)
#                                            (M, C) floor per channel
#     channel_fallback(manifest, graph_path, floor)
#                                            (M, C) nominal scale for channels with no train data
#     compute_scale(features_dir, splits, manifest, graph_path, percentile, floor, allow_missing)
#                                            (M, C) scale from train blocks, plus counts
#     save_scale(path, ...)                  write scale.json atomically
#     load_scale(path, manifest)             read scale.json, checked against the manifest
#   Model inputs
#     build_features(arrays, scale)          X (T, M, C, F) float32, plus target validity
#     window_starts(present_any, history, horizon, max_gap, target_valid)
#                                            usable window start indices per meter
#   Iteration
#     load_piece(features_dir, block, month, scale, allow_missing, manifest)
#                                            one block-month: (t, X, mag_valid, ang_valid) or None
#     iter_blocks(features_dir, splits, split_name, scale, allow_missing, manifest)
#                                            yield (label, month, t, X, mag_valid, ang_valid) per block-month
#     WindowDataset                          per-meter windows; get_batch(idx) assembles a whole
#                                            batch with array indexing, no per-example loop
# =============================================================================

# Standard library: JSON for manifest, splits, graph, scale
import json
# Standard library: paths
import os
# Third party: numerics
import numpy as np
# Third party: timestamps for month labels
import pandas as pd

# Feature names per channel, in order
FEATURES = ("mag", "cos", "sin", "mag_valid", "ang_valid")
# Number of features per channel
F = len(FEATURES)
# Seconds per slot
SLOT_SECONDS = 10
# The feature files follow the preprocessing grid: fixed Pacific Standard
# Time months, which run from 08:00 UTC on the first day to 08:00 UTC on
# the first day of the next month. This offset turns UTC into that time base.
UTC_OFFSET_SECONDS = 8 * 3600
# Voltage channel names, for the scale floor
VOLTAGE_CHANNELS = ("L1", "L2", "L3")


# -----------------------------------------------------------------------------
# Files
# -----------------------------------------------------------------------------

def load_manifest(features_dir):
    """Read manifest.json: meter order, channel order, array descriptions."""
    # Path
    path = os.path.join(features_dir, "manifest.json")
    # Fail with the path tried
    if not os.path.exists(path):
        # Raise
        raise FileNotFoundError(f"manifest.json not found in {features_dir}")
    # Read
    with open(path) as f:
        # Parse
        return json.load(f)


def load_splits(path):
    """Read splits.json and check that no two blocks overlap."""
    # Fail with the path tried
    if not os.path.exists(path):
        # Raise
        raise FileNotFoundError(f"splits file not found: {path}")
    # Read
    with open(path) as f:
        # Parse
        splits = json.load(f)
    # Overlap check
    check_no_overlap(splits)
    # Return
    return splits


def check_no_overlap(splits):
    """Raise if any two blocks, in any splits, overlap in time."""
    # Every block with its split name
    blocks = [(name, b) for name in ("train", "validation", "test_events", "test_normal") for b in splits.get(name, [])]
    # Bounds
    bounds = [(name, b["start"], *block_seconds(b)) for name, b in blocks]
    # Sort by start
    bounds.sort(key=lambda x: x[2])
    # Each block must start at or after the previous one ends
    for (n1, s1, a1, e1), (n2, s2, a2, e2) in zip(bounds, bounds[1:]):
        # Overlap
        if a2 < e1:
            # Raise
            raise ValueError(f"split blocks overlap: {n1} {s1} and {n2} {s2}")


def block_seconds(block):
    """Start and end (exclusive) of a split block as integer epoch seconds, from its UTC bounds."""
    # Start
    start = int(pd.Timestamp(block["start_utc"]).timestamp())
    # End, exclusive
    end = int(pd.Timestamp(block["end_utc_exclusive"]).timestamp())
    # Return
    return start, end


def month_bounds(month):
    """Start and end (exclusive) of a feature-file month as epoch seconds, in the fixed PST convention."""
    # Midnight UTC on the first day
    first = pd.Timestamp(month + "-01")
    # Midnight UTC on the first day of the next month
    following = first + pd.offsets.MonthBegin(1)
    # Shift both by the offset: the file's month starts at 08:00 UTC
    return int(first.timestamp()) + UTC_OFFSET_SECONDS, int(following.timestamp()) + UTC_OFFSET_SECONDS


def months_in_block(block):
    """
    The "YYYY-MM" feature files a block touches, in order. Months follow
    the files' fixed PST convention, so a UTC time is shifted back by the
    offset before its month is read: 02:00 UTC on October 1 belongs to
    September's file.
    """
    # Bounds
    start, end = block_seconds(block)
    # First month, in the files' time base, as a naive timestamp
    current = pd.Timestamp(start - UTC_OFFSET_SECONDS, unit="s").to_period("M")
    # Last month, end is exclusive so step back one second
    last = pd.Timestamp(end - 1 - UTC_OFFSET_SECONDS, unit="s").to_period("M")
    # Collect
    months = []
    # Loop
    while current <= last:
        # Label
        months.append(current.strftime("%Y-%m"))
        # Advance
        current += 1
    # Return
    return months


def load_month(features_dir, month, allow_missing=False, manifest=None):
    """
    Read one month file, check it, and return its arrays with NaN cleaned.
    With a manifest, also check that the meter and channel counts match it.

    Returns a dictionary:
      t          int64 (T,)        slot times, checked to be strictly increasing by 10 s
      rms        float32 (T, M, C) magnitude, zero where not valid
      ang        float32 (T, M, C) angle in degrees, zero where not valid
      present    bool (T, M, C)    the feature-file mask, untouched: a reading arrived
      mag_valid  bool (T, M, C)    present and the magnitude is finite
      ang_valid  bool (T, M, C)    angle_ok and the angle is finite and mag_valid
    Presence is kept apart from magnitude validity on purpose: a reading
    that arrived with a bad number is not the same as no reading.
    A missing file raises unless allow_missing, in which case None is returned.
    """
    # Path
    path = os.path.join(features_dir, f"{month}.npz")
    # Missing month
    if not os.path.exists(path):
        # Allowed
        if allow_missing:
            # Nothing
            return None
        # Otherwise fail with the path tried
        raise FileNotFoundError(f"feature file missing: {path}")
    # Load inside a context manager so the file is closed
    with np.load(path) as data:
        # Times
        t = data["t"].astype(np.int64)
        # Presence, untouched
        present = data["mask"].astype(bool)
        # Angle flag from the file
        angle_ok = data["angle_ok"].astype(bool)
        # Magnitude
        rms = data["rms"].astype(np.float32)
        # Angle
        ang = data["ang"].astype(np.float32)
    # Shape checks
    if not (rms.shape == ang.shape == present.shape == angle_ok.shape) or rms.shape[0] != len(t):
        # Raise
        raise ValueError(f"{path}: array shapes do not match")
    # Dimension check against the manifest
    if manifest is not None and rms.shape[1:] != (len(manifest["meters"]), len(manifest["channels"])):
        # Raise
        raise ValueError(f"{path}: arrays are {rms.shape[1:]} but the manifest has {len(manifest['meters'])} meters and {len(manifest['channels'])} channels")
    # Month check: every slot time inside the named month, in the files' fixed PST convention
    month_start, month_end = month_bounds(month)
    # Both bounds carry the same offset
    if len(t) and (t.min() < month_start or t.max() >= month_end):
        # Raise
        raise ValueError(f"{path}: slot times fall outside month {month}")
    # Time checks: strictly increasing by one slot
    steps = np.diff(t)
    # Any step that is not exactly one slot
    if len(steps) and not np.all(steps == SLOT_SECONDS):
        # Raise with the first offender
        bad_i = int(np.argmax(steps != SLOT_SECONDS))
        # Message
        raise ValueError(f"{path}: slot times are not contiguous at row {bad_i}, step {int(steps[bad_i])} s")
    # Magnitude validity: present and finite
    mag_valid = present & np.isfinite(rms)
    # Zero where not valid
    rms = np.where(mag_valid, rms, 0.0).astype(np.float32)
    # Angle validity: flagged ok, finite, and the magnitude is valid
    ang_valid = angle_ok & np.isfinite(ang) & mag_valid
    # Zero where not valid
    ang = np.where(ang_valid, ang, 0.0).astype(np.float32)
    # Return
    return {"t": t, "rms": rms, "ang": ang, "present": present, "mag_valid": mag_valid, "ang_valid": ang_valid}


def slice_block(arrays, start_s, end_s):
    """Keep the rows of a month whose slot time lies in [start_s, end_s). Returns None if none."""
    # Row selection
    keep = (arrays["t"] >= start_s) & (arrays["t"] < end_s)
    # Nothing inside
    if not keep.any():
        # None
        return None
    # Slice every array
    return {k: v[keep] for k, v in arrays.items()}


# -----------------------------------------------------------------------------
# Scaling
# -----------------------------------------------------------------------------

def channel_floor(manifest, graph_path, floor):
    """
    Minimum scale per meter and channel, shape (M, C). Voltage channels
    get 1 volt. Current channels get 1 percent of the clamp rating from
    graph.json when it is given, otherwise `floor`.
    """
    # Meters and channels
    meters, channels = manifest["meters"], manifest["channels"]
    # Start everything at the plain floor
    out = np.full((len(meters), len(channels)), float(floor), dtype=np.float32)
    # Voltage channels: 1 volt
    for ci, ch in enumerate(channels):
        # Check
        if ch in VOLTAGE_CHANNELS:
            # Set
            out[:, ci] = 1.0
    # Ratings from the graph, if given
    if graph_path and os.path.exists(graph_path):
        # Read
        with open(graph_path) as f:
            # Parse
            graph = json.load(f)
        # Loop over meters
        for mi, meter in enumerate(meters):
            # Ratings for this meter
            ratings = graph.get("meters", {}).get(meter, {}).get("current_rating", {})
            # Loop over channels
            for ci, ch in enumerate(channels):
                # Current channel with a rating
                if ch in ratings and ratings[ch]:
                    # 1 percent of rating
                    out[mi, ci] = max(out[mi, ci], 0.01 * float(ratings[ch]))
    # Return
    return out


def channel_fallback(manifest, graph_path, floor):
    """
    Nominal scale per meter and channel, shape (M, C), for channels that
    have no valid train reading: the nominal voltage of the meter's bus for
    voltage channels, a tenth of the clamp rating for current channels,
    both from graph.json. The plain floor where the graph is not given or
    has no number for the channel.
    """
    # Meters and channels
    meters, channels = manifest["meters"], manifest["channels"]
    # Start at the floor
    out = np.full((len(meters), len(channels)), float(floor), dtype=np.float32)
    # No graph
    if not graph_path or not os.path.exists(graph_path):
        # Floor everywhere
        return out
    # Read
    with open(graph_path) as f:
        # Parse
        graph = json.load(f)
    # Loop over meters
    for mi, meter in enumerate(meters):
        # Meter record
        rec = graph.get("meters", {}).get(meter, {})
        # Nominal voltage of its bus, through its node
        nominal = None
        # Node and bus known
        if rec.get("node") is not None and rec.get("voltage_bus"):
            # Node record
            node = graph["nodes"][rec["node"]]
            # Nominal voltage of that bus
            nominal = node.get("nominal_voltage", {}).get(rec["voltage_bus"])
        # Ratings
        ratings = rec.get("current_rating", {})
        # Loop over channels
        for ci, ch in enumerate(channels):
            # Voltage channel with a nominal
            if ch in VOLTAGE_CHANNELS and nominal:
                # Set
                out[mi, ci] = float(nominal)
            # Current channel with a rating
            elif ch in ratings and ratings[ch]:
                # A tenth of the rating
                out[mi, ci] = 0.1 * float(ratings[ch])
    # Return
    return out


def compute_scale(features_dir, splits, manifest, graph_path=None, percentile=95.0, floor=1.0, allow_missing=False):
    """
    One positive scale per meter and channel from the train blocks only:
    the given percentile of valid magnitudes, raised to the floor.
    Channels with no valid train reading get the nominal fallback from
    channel_fallback. Returns (scale, count, fallback) where count is the
    true number of valid train readings per channel and fallback marks the
    channels that used the nominal value; the percentile itself is taken
    over every tenth reading to bound memory.
    """
    # Sizes
    M, C = len(manifest["meters"]), len(manifest["channels"])
    # Collect samples per channel, thinned to keep memory small
    samples = [[[] for _ in range(C)] for _ in range(M)]
    # True counts of valid readings
    count = np.zeros((M, C), dtype=np.int64)
    # Loop over train blocks
    for block in splits["train"]:
        # Bounds
        start_s, end_s = block_seconds(block)
        # Loop over months
        for month in months_in_block(block):
            # Load, checked against the manifest
            arrays = load_month(features_dir, month, allow_missing, manifest)
            # Skip missing when allowed
            if arrays is None:
                # Next
                continue
            # Rows in the block
            part = slice_block(arrays, start_s, end_s)
            # Skip empty
            if part is None:
                # Next
                continue
            # True counts of valid readings
            count += part["mag_valid"].sum(axis=0)
            # Every meter and channel
            for mi in range(M):
                # Loop over channels
                for ci in range(C):
                    # Valid magnitudes
                    v = part["rms"][:, mi, ci][part["mag_valid"][:, mi, ci]]
                    # Keep every 10th as a copy, so the full month is not held in memory
                    if len(v):
                        # Store
                        samples[mi][ci].append(v[::10].copy())
    # Floors
    floors = channel_floor(manifest, graph_path, floor)
    # Nominal fallbacks
    fallbacks = channel_fallback(manifest, graph_path, floor)
    # Output starts as the fallback, never below the floor
    scale = np.maximum(fallbacks, floors)
    # Which channels used it
    used_fallback = np.ones((M, C), dtype=bool)
    # Loop
    for mi in range(M):
        # Loop
        for ci in range(C):
            # Nothing valid: keep the fallback
            if not samples[mi][ci]:
                # Next
                continue
            # All samples
            v = np.concatenate(samples[mi][ci])
            # Percentile raised to the floor
            scale[mi, ci] = max(float(np.percentile(v, percentile)), float(floors[mi, ci]))
            # Real data used
            used_fallback[mi, ci] = False
    # Return
    return scale.astype(np.float32), count, used_fallback


def save_scale(path, scale, count, manifest, percentile, floor, fallback=None):
    """Write scale.json with the numbers, the counts, which channels used the fallback, and how they were made."""
    # Body
    body = {
        # How
        "percentile": percentile, "floor": floor,
        # Order
        "meters": manifest["meters"], "channels": manifest["channels"],
        # Values as nested lists
        "scale": scale.tolist(), "count": count.tolist(),
        # Channels with no train data, scaled by their nominal value
        "fallback": fallback.tolist() if fallback is not None else None,
    }
    # Write to a temporary file first, then rename, so a crash never leaves a half file
    tmp = path + ".tmp"
    # Write
    with open(tmp, "w") as f:
        # Pretty JSON
        json.dump(body, f, indent=1)
    # Atomic replace
    os.replace(tmp, path)


def load_scale(path, manifest):
    """
    Read scale.json and return the (M, C) float32 scale array, after
    checking that its meter and channel order match the manifest, its
    shape is right, and every value is finite and positive.
    """
    # Read
    with open(path) as f:
        # Parse
        body = json.load(f)
    # Order must match
    if body["meters"] != manifest["meters"] or body["channels"] != manifest["channels"]:
        # Raise
        raise ValueError(f"{path}: meter or channel order differs from the manifest")
    # Array
    scale = np.array(body["scale"], dtype=np.float32)
    # Shape
    if scale.shape != (len(manifest["meters"]), len(manifest["channels"])):
        # Raise
        raise ValueError(f"{path}: scale shape {scale.shape} does not match the manifest")
    # Values
    if not np.all(np.isfinite(scale)) or not np.all(scale > 0):
        # Raise
        raise ValueError(f"{path}: scale values must be finite and positive")
    # Return
    return scale


# -----------------------------------------------------------------------------
# Model inputs
# -----------------------------------------------------------------------------

def build_features(arrays, scale):
    """
    Turn a block's arrays into the model input X of shape (T, M, C, F) and
    the target validity flags:
      X[..., 0] = rms / scale, zero where not valid
      X[..., 1] = cos(angle), X[..., 2] = sin(angle), zero where not valid
      X[..., 3] = mag_valid as 0/1, X[..., 4] = ang_valid as 0/1
    Returns (X float32, mag_valid bool (T, M, C), ang_valid bool (T, M, C)),
    the same two validity arrays that sit in X, used for targets and loss.
    """
    # Sizes
    T, M, C = arrays["rms"].shape
    # Output
    X = np.zeros((T, M, C, F), dtype=np.float32)
    # Scaled magnitude, already zero where not valid
    X[..., 0] = arrays["rms"] / scale[None, :, :]
    # Angle in radians, already zero where not valid
    rad = np.deg2rad(arrays["ang"])
    # Cos, zero where not valid
    X[..., 1] = np.where(arrays["ang_valid"], np.cos(rad), 0.0)
    # Sin, zero where not valid
    X[..., 2] = np.where(arrays["ang_valid"], np.sin(rad), 0.0)
    # Magnitude validity flag: present and finite
    X[..., 3] = arrays["mag_valid"]
    # Angle validity flag
    X[..., 4] = arrays["ang_valid"]
    # Validity for targets
    return X, arrays["mag_valid"].copy(), arrays["ang_valid"].copy()


def window_starts(present_any, history, horizon, max_gap, target_valid):
    """
    For one meter, the start indices s of usable windows: history slots
    s .. s+history-1 and target slot s+history+horizon-1.

    present_any:  (T,) bool, the meter has at least one reading in the slot
    target_valid: (T,) bool, the target slot has at least one valid magnitude
    A window is usable when no run of max_gap or more consecutive absent
    slots lies inside its history, and its target is valid. Absent slots
    before the window do not count against it.
    """
    # Validate the parameters
    if history < 1 or horizon < 1 or max_gap < 1:
        # Raise
        raise ValueError("history, horizon, and max_gap must be at least 1")
    # Length
    T = len(present_any)
    # Span of one window including the horizon
    span = history + horizon
    # Too short
    if T < span:
        # None
        return np.zeros(0, dtype=np.int64)
    # Absent slots as 1
    absent = (~present_any).astype(np.int32)
    # Running length of the absent run ending at each slot
    run = np.zeros(T, dtype=np.int32)
    # Loop once
    for i in range(T):
        # Extend or reset
        run[i] = run[i - 1] + 1 if (i > 0 and absent[i]) else absent[i]
    # Slot i closes a run of max_gap absent slots that ends at i
    closes = run >= max_gap
    # Such a run lies inside the window [s, s + history) exactly when
    # s <= i - max_gap + 1 and i <= s + history - 1, so for a start s the
    # slots to test are i in [s + max_gap - 1, s + history - 1]
    cum = np.concatenate([[0], np.cumsum(closes)])
    # Candidate starts
    starts = np.arange(0, T - span + 1)
    # A run shorter than the history can fit; otherwise nothing to test
    if max_gap <= history:
        # Count closing slots in the test range
        bad_in_history = cum[starts + history] - cum[starts + max_gap - 1]
    # Window shorter than max_gap can never hold such a run
    else:
        # None
        bad_in_history = np.zeros(len(starts), dtype=np.int64)
    # Target index
    target = starts + span - 1
    # Keep
    keep = (bad_in_history == 0) & target_valid[target]
    # Return
    return starts[keep]


# -----------------------------------------------------------------------------
# Iteration
# -----------------------------------------------------------------------------

def load_piece(features_dir, block, month, scale, allow_missing=False, manifest=None):
    """
    Load the part of one month file that lies inside one block and build
    the model input for it. Returns (t, X, mag_valid, ang_valid), or None
    when the file is missing (allowed) or the block has no rows in it.
    This is the unit the training loop loads, one at a time.
    """
    # Bounds
    start_s, end_s = block_seconds(block)
    # Load, checked
    arrays = load_month(features_dir, month, allow_missing, manifest)
    # Missing when allowed
    if arrays is None:
        # None
        return None
    # Rows in the block
    part = slice_block(arrays, start_s, end_s)
    # Nothing inside
    if part is None:
        # None
        return None
    # Inputs
    X, mag_valid, ang_valid = build_features(part, scale)
    # Return
    return part["t"], X, mag_valid, ang_valid


def iter_blocks(features_dir, splits, split_name, scale, allow_missing=False, manifest=None):
    """
    Yield one piece per (block, month) of a split, in time order:
      (label, month, t, X, mag_valid, ang_valid)
    with t the slot times, X the model input (T, M, C, F), and the two
    validity arrays (T, M, C). Each piece lies inside one block, so windows
    cut from it never cross a block boundary. A missing month file raises
    unless allow_missing.
    """
    # Loop over blocks
    for block in splits[split_name]:
        # Label
        label = f"{split_name}_{block['start']}_{block['end']}"
        # Loop over months
        for month in months_in_block(block):
            # One block-month
            piece = load_piece(features_dir, block, month, scale, allow_missing, manifest)
            # Skip missing or empty
            if piece is None:
                # Next
                continue
            # Hand back
            yield (label, month) + piece


class WindowDataset:
    """
    A torch-style dataset of training examples cut from one piece.

    Each example is one meter and one window start:
      x        (history, C * F)  the meter's history, channels and features flattened
      y        (C, 3)            target: scaled magnitude, cos, sin
      y_valid  (C, 2)            target validity: magnitude, angle
      meter    int               meter index, for the meter embedding
    Built as a plain class so it works without torch for testing;
    torch's DataLoader accepts anything with __len__ and __getitem__.
    """

    def __init__(self, X, mag_valid, ang_valid, history, horizon, max_gap):
        # Keep the arrays
        self.X, self.history, self.horizon = X, history, horizon
        # Sizes
        T, M, C, _ = X.shape
        # Targets: scaled magnitude, cos, sin
        self.y = X[..., 0:3]
        # Target validity
        self.y_valid = np.stack([mag_valid, ang_valid], axis=-1)
        # Examples as (meter, start) pairs
        pairs = []
        # Loop over meters
        for mi in range(M):
            # Any reading in the slot
            present_any = mag_valid[:, mi, :].any(axis=1)
            # Usable starts for this meter
            starts = window_starts(present_any, history, horizon, max_gap, present_any)
            # Store
            pairs.append(np.stack([np.full(len(starts), mi), starts], axis=1))
        # All examples
        self.pairs = np.concatenate(pairs, axis=0) if pairs else np.zeros((0, 2), dtype=np.int64)

    def __len__(self):
        """Number of examples."""
        # Count
        return len(self.pairs)

    def __getitem__(self, i):
        """One example as numpy arrays."""
        # Meter and start
        mi, s = self.pairs[i]
        # Target slot
        target = s + self.history + self.horizon - 1
        # History, flattened per slot
        x = self.X[s:s + self.history, mi].reshape(self.history, -1)
        # Return
        return x, self.y[target, mi], self.y_valid[target, mi], int(mi)

    def get_batch(self, idx):
        """
        A whole batch at once with array indexing, no per-example loop:
          x        (B, history, C * F)
          y        (B, C, 3)
          y_valid  (B, C, 2)
          meter    (B,) int64
        idx is an array of example indices.
        """
        # Meters and starts
        mi, s = self.pairs[idx, 0], self.pairs[idx, 1]
        # Slot index of every history step: (B, history)
        rows = s[:, None] + np.arange(self.history)[None, :]
        # Gather histories: (B, history, C, F) then flatten channels and features
        x = self.X[rows, mi[:, None]].reshape(len(idx), self.history, -1)
        # Target slots
        target = s + self.history + self.horizon - 1
        # Return
        return x, self.y[target, mi], self.y_valid[target, mi], mi.astype(np.int64)
