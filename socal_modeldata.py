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
#              With physics inputs turned on (configure_physics), four more
#              numbers follow, the same for every meter and channel of the
#              slot, F = 9:
#              [ A residual / scale_A, A valid, B residual / scale_B, B valid ]
#              from <YYYY-MM>_physics.npz (main_physicsfeatures.py). Where a
#              valid flag is 0 the residual before it is 0: a missing
#              physics value never looks like a perfect fit. Physics inputs
#              are off unless a main script turns them on, so every earlier
#              command builds exactly the inputs it did before.
#              With the neighbor angle inputs turned on (configure_angdiff,
#              experiment 2, exclusive with physics), three more numbers
#              follow the five standard ones, F = 8:
#              [ cos d, sin d, valid ] where d is the meter's voltage angle
#              minus the equal-weight circular mean of its wired neighbors'
#              angles on the same physical phase, computed at load time by
#              socal_angdiff.neighbor_differences from the month's angles,
#              <YYYY-MM>_late.npz (main_latefeatures.py) and the meter
#              graph's period in force at each slot. Current channels and
#              channels without a phase-to-neutral phase carry zeros with
#              the flag 0. Only history slots ever reach the model, as for
#              every other input.
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
#     save_scale(path, ...)                  write scale.json atomically, with the physics divisors when given
#     load_scale(path, manifest)             read scale.json, checked against the manifest
#     load_physics_scale(path)               the physics divisors from scale.json, or None
#   Physics inputs (optional)
#     configure_physics(features_dir, scale) turn the four physics features on for this process
#     n_features()                           5, or 9 with physics on, or 8 with neighbor angle inputs on
#   Neighbor angle inputs (optional)
#     configure_angdiff(features_dir, graph_body, manifest, min_resultant, late_threshold)
#                                            turn the three neighbor angle features on for this process
#     extra_features()                       names of the added features, in order, or []
#     retained_starts(mag_valid, history, horizon, max_gap, min_meters)
#                                            the graph training set's window selection, shared with socal_models
#     angdiff_training_valid(features_dir, blocks, manifest, history, horizon, max_gap, min_meters)
#                                            valid neighbor angle features inside retained training histories, for the stop-on-empty check
#     compute_physics_scale(features_dir, splits, percentile, allow_missing)
#                                            per side: the percentile of valid residuals over the train blocks
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

# The neighbor angle difference helpers (experiment 2)
import socal_angdiff as sa

# Feature names per channel, in order
FEATURES = ("mag", "cos", "sin", "mag_valid", "ang_valid")
# Number of features per channel without physics inputs
F = len(FEATURES)
# The physics features, appended after the five above when physics inputs are on
PHYSICS_FEATURES = ("phys_a", "phys_a_valid", "phys_b", "phys_b_valid")
# The two sides in array order
PHYSICS_SIDES = ("a", "b")
# Physics configuration for this process: None (off) or {"features_dir", "scale" (2,) float32 or None}
PHYSICS = None
# Neighbor angle configuration for this process: None (off) or {"features_dir", "timeline", "phase_idx", "adjacency", "min_resultant", "late_threshold", "phase_map"}
ANGDIFF = None
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
    # Arrays
    out = {"t": t, "rms": rms, "ang": ang, "present": present, "mag_valid": mag_valid, "ang_valid": ang_valid}
    # Physics inputs, when turned on for this process
    if PHYSICS is not None:
        # Physics file of the month, required: a month without one cannot be trained or scored with physics inputs
        ppath = os.path.join(PHYSICS["features_dir"], f"{month}_physics.npz")
        # Missing
        if not os.path.exists(ppath):
            # Raise
            raise FileNotFoundError(f"physics inputs are on but {ppath} is missing; run main_physicsfeatures.py")
        # Load
        with np.load(ppath) as data:
            # Times must be the feature month's times, slot for slot
            if not np.array_equal(data["t"].astype(np.int64), t):
                # Raise
                raise ValueError(f"{ppath}: slot times differ from {path}")
            # Residual per side, (T, 2)
            r = np.stack([data[f"r_{side}"].astype(np.float32) for side in PHYSICS_SIDES], axis=1)
            # Valid per side, (T, 2)
            v = np.stack([data[f"valid_{side}"].astype(bool) for side in PHYSICS_SIDES], axis=1)
        # A valid residual must be finite; zero where not valid, whatever the file holds
        v &= np.isfinite(r)
        # Zero where not valid
        r = np.where(v, r, 0.0).astype(np.float32)
        # Store
        out["phys_r"], out["phys_valid"] = r, v
    # Neighbor angle inputs, when turned on for this process
    if ANGDIFF is not None:
        # Late flags of the month: slot times, shape and meter order checked against the manifest
        late = sa.load_late(ANGDIFF["features_dir"], month, t, ANGDIFF["meters"])
        # Edge set per slot, from the period in force at that slot
        gs = sa.graph_per_slot(t, ANGDIFF["timeline"])
        # The feature
        nb_cos, nb_sin, nb_valid, _ = sa.neighbor_differences(ang, ang_valid, late, ANGDIFF["phase_idx"], ANGDIFF["adjacency"], gs, ANGDIFF["min_resultant"])
        # Store
        out["nb_cos"], out["nb_sin"], out["nb_valid"] = nb_cos, nb_sin, nb_valid
    # Return
    return out


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


def save_scale(path, scale, count, manifest, percentile, floor, fallback=None, physics=None):
    """
    Write scale.json with the numbers, the counts, which channels used the
    fallback, and how they were made. physics, when given, is
    {"scale": (2,), "count": (2,), "features_dir": str} and is stored under
    "physics" so a run with physics inputs records its divisors.
    """
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
        # Physics divisors, None for a run without physics inputs
        "physics": {"sides": list(PHYSICS_SIDES), "scale": [float(x) for x in physics["scale"]], "count": [int(x) for x in physics["count"]], "features_dir": physics["features_dir"]} if physics is not None else None,
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


def load_physics_scale(path):
    """
    Read the "physics" section of scale.json: the (2,) divisor array, or
    None when the file has none (a run without physics inputs). Checks the
    side order and that the values are finite and positive.
    """
    # Read
    with open(path) as f:
        # Parse
        body = json.load(f)
    # Section
    phys = body.get("physics")
    # None recorded
    if phys is None:
        # None
        return None
    # Side order
    if list(phys.get("sides", [])) != list(PHYSICS_SIDES):
        # Raise
        raise ValueError(f"{path}: physics sides {phys.get('sides')} differ from {list(PHYSICS_SIDES)}")
    # Array
    scale = np.asarray(phys["scale"], dtype=np.float32)
    # Values
    if scale.shape != (len(PHYSICS_SIDES),) or not np.all(np.isfinite(scale)) or not np.all(scale > 0):
        # Raise
        raise ValueError(f"{path}: physics scale values must be {len(PHYSICS_SIDES)} finite positive numbers")
    # Return
    return scale


# -----------------------------------------------------------------------------
# Physics inputs (optional)
# -----------------------------------------------------------------------------

def configure_physics(features_dir, scale=None):
    """
    Turn the physics inputs on for this process: load_month() then reads
    <month>_physics.npz from features_dir and build_features() appends the
    four physics features. scale is the (2,) array of per-side divisors
    (compute_physics_scale or the scale file); it may be set later by
    calling again with the same folder. Pass features_dir None to turn the
    inputs off again. Nothing is on unless a main script calls this.
    """
    # Module state
    global PHYSICS
    # Off
    if features_dir is None:
        # Clear
        PHYSICS = None
        # Done
        return
    # Scale, checked when given
    if scale is not None:
        # Array
        scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        # Two finite positive values
        if scale.shape != (len(PHYSICS_SIDES),) or not np.all(np.isfinite(scale)) or not np.all(scale > 0):
            # Raise
            raise ValueError(f"physics scale must be {len(PHYSICS_SIDES)} finite positive values, got {scale}")
    # Not both
    if ANGDIFF is not None:
        # Raise
        raise ValueError("physics inputs and neighbor angle inputs cannot both be on")
    # Set
    PHYSICS = {"features_dir": features_dir, "scale": scale}


def n_features():
    """Number of features per channel in the model input: F plus the added features that are on."""
    # Count
    return F + len(extra_features())


def retained_starts(mag_valid, history, horizon, max_gap, min_meters=1):
    """
    The window starts the graph training set keeps from one piece, and
    the per-meter usability of each kept window. This is the selection
    rule of socal_models.GraphWindowDataset, kept here so the training set
    and the neighbor-angle training guard use one definition: per meter,
    window_starts() (no run of max_gap absent slots inside the history,
    valid target); a window is kept when at least min_meters meters pass.
    Returns (starts (W,) int64, usable (W, M) bool).
    """
    # Sizes
    T, M = mag_valid.shape[0], mag_valid.shape[1]
    # Slots a window spans
    span = history + horizon
    # Candidate starts
    n_starts = max(T - span + 1, 0)
    # Per-meter usability of each candidate
    usable = np.zeros((n_starts, M), dtype=bool)
    # Each meter
    for mi in range(M):
        # The meter has a reading in the slot
        present_any = mag_valid[:, mi, :].any(axis=1)
        # Its usable starts
        starts = window_starts(present_any, history, horizon, max_gap, present_any)
        # Mark
        usable[starts, mi] = True
    # Kept: enough meters pass
    kept = np.nonzero(usable.sum(axis=1) >= min_meters)[0]
    # Return
    return kept, usable[kept]


def angdiff_training_valid(features_dir, blocks, manifest, history, horizon, max_gap, min_meters=1):
    """
    Valid neighbor angle features inside the history inputs of the
    training windows that would be retained, counted per training block
    piece (the pieces after any --train-months restriction) with the
    training set's own selection rule (retained_starts: history length,
    horizon, gap rule, minimum meters). Only history slots of retained
    windows count, so values that appear only on target rows, or only in
    windows the selection discards, do not make the count positive.
    Returns (valid count, history slots counted, windows retained).
    Independent of the scale: a main script calls it before training
    whether scale.json is computed or loaded. Raises when the inputs are
    off.
    """
    # Must be on
    if ANGDIFF is None:
        # Raise
        raise ValueError("angdiff_training_valid called with the neighbor angle inputs off")
    # Counts
    valid, slots, windows = 0, 0, 0
    # Each block piece
    for block in blocks:
        # Bounds
        start_s, end_s = block_seconds(block)
        # Each month it touches
        for month in months_in_block(block):
            # Month, with the feature computed
            arrays = load_month(features_dir, month, allow_missing=True, manifest=manifest)
            # Missing month
            if arrays is None:
                # Next
                continue
            # Rows inside the piece
            part = slice_block(arrays, start_s, end_s)
            # None
            if part is None:
                # Next
                continue
            # Windows the training set would keep
            starts, _ = retained_starts(part["mag_valid"], history, horizon, max_gap, min_meters)
            # None kept
            if len(starts) == 0:
                # Next
                continue
            # History slots of the kept windows
            covered = np.zeros(len(part["t"]), dtype=bool)
            # Each window
            for s in starts:
                # Its history
                covered[s:s + history] = True
            # Count valid features on those slots, every meter and channel
            valid += int(part["nb_valid"][covered].sum())
            # Slots
            slots += int(covered.sum())
            # Windows
            windows += int(len(starts))
    # Return
    return valid, slots, windows


def extra_features():
    """Names of the features appended after the five standard ones, in order: the physics four, the neighbor angle three, or none."""
    # Physics
    if PHYSICS is not None:
        # Four
        return list(PHYSICS_FEATURES)
    # Neighbor angles
    if ANGDIFF is not None:
        # Three
        return list(sa.ANGDIFF_FEATURES)
    # None
    return []


def configure_angdiff(features_dir, graph_body=None, manifest=None, min_resultant=sa.MIN_RESULTANT, late_threshold=sa.LATE_THRESHOLD):
    """
    Turn the neighbor angle inputs on for this process: load_month() then
    reads <month>_late.npz from features_dir and computes the three
    features from the month's angles and the meter graph (graph_body, a
    fixed or time-indexed graph file already parsed; its phase map gives
    each meter's channels per physical phase). Exclusive with the physics
    inputs. Pass features_dir None to turn the inputs off again. Returns
    the settings recorded with a run.
    """
    # Module state
    global ANGDIFF
    # Off
    if features_dir is None:
        # Clear
        ANGDIFF = None
        # Done
        return None
    # Not both
    if PHYSICS is not None:
        # Raise
        raise ValueError("physics inputs and neighbor angle inputs cannot both be on")
    # Needed
    if graph_body is None or manifest is None:
        # Raise
        raise ValueError("configure_angdiff needs the graph file body and the manifest")
    # Periods and edge sets
    timeline = sa.timeline_from_graph(graph_body)
    # Meters and channels
    meters, channels = list(manifest["meters"]), list(manifest["channels"])
    # Phase map
    phase_idx = sa.phase_channel_index(timeline["meters"], meters, channels)
    # No meter mapped at all is a configuration error
    if not (phase_idx >= 0).any():
        # Raise
        raise ValueError("no meter in the graph file has a phase-to-neutral voltage channel; the phase map is missing")
    # Adjacency per edge set
    adjacency = sa.adjacency_matrices(timeline, timeline["meters"], meters)
    # Cutoffs must make sense
    if not (0.0 < float(min_resultant) <= 1.0) or not np.isfinite(late_threshold) or late_threshold <= 0:
        # Raise
        raise ValueError(f"min_resultant must be in (0, 1] and late_threshold positive, got {min_resultant} and {late_threshold}")
    # Set
    ANGDIFF = {"features_dir": features_dir, "timeline": timeline, "phase_idx": phase_idx, "adjacency": adjacency, "min_resultant": float(min_resultant), "late_threshold": float(late_threshold), "phase_map": sa.phase_map_identity(timeline["meters"], meters), "meters": meters}
    # Settings for the run's options
    return {"min_resultant": float(min_resultant), "late_threshold": float(late_threshold), "phase_map": ANGDIFF["phase_map"], "weights": "equal", "period_rule": "each slot's own period", "n_edge_sets": len(adjacency)}


def compute_physics_scale(features_dir, splits, percentile=95.0, allow_missing=False):
    """
    Per side, the given percentile of the valid physics residuals over the
    train blocks only, the same rule compute_scale() applies to magnitudes.
    Returns (scale (2,) float32, count (2,) int64). Stops with a clear
    message when a side has no valid training value or its scale is zero
    or not finite: a side that cannot be scaled cannot be an input.
    Physics inputs must be on (configure_physics) so load_month reads them.
    """
    # Must be on
    if PHYSICS is None:
        # Raise
        raise ValueError("physics inputs are off; call configure_physics first")
    # Samples per side, thinned like the magnitude scale
    samples = [[] for _ in PHYSICS_SIDES]
    # True counts
    count = np.zeros(len(PHYSICS_SIDES), dtype=np.int64)
    # Train blocks
    for block in splits["train"]:
        # Bounds
        start_s, end_s = block_seconds(block)
        # Months
        for month in months_in_block(block):
            # Load with the physics arrays
            arrays = load_month(features_dir, month, allow_missing)
            # Missing when allowed
            if arrays is None:
                # Next
                continue
            # Rows in the block
            part = slice_block(arrays, start_s, end_s)
            # Empty
            if part is None:
                # Next
                continue
            # Each side
            for k in range(len(PHYSICS_SIDES)):
                # Valid values
                v = part["phys_r"][:, k][part["phys_valid"][:, k]]
                # Count
                count[k] += len(v)
                # Keep every tenth
                if len(v):
                    # Store
                    samples[k].append(v[::10].copy())
    # Scale per side
    scale = np.zeros(len(PHYSICS_SIDES), dtype=np.float32)
    # Each side
    for k, side in enumerate(PHYSICS_SIDES):
        # Nothing valid in training
        if not samples[k]:
            # Stop
            raise ValueError(f"physics side {side} has no valid value in the train blocks; it cannot be scaled or used as an input")
        # Percentile
        scale[k] = float(np.percentile(np.concatenate(samples[k]), percentile))
        # Must be positive and finite
        if not np.isfinite(scale[k]) or scale[k] <= 0:
            # Stop
            raise ValueError(f"physics side {side} scale is {scale[k]} (percentile {percentile} of {int(count[k])} training values); it cannot be used as an input")
    # Return
    return scale, count


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
    With physics inputs on, four more features follow, identical over
    meters and channels: X[..., 5] = A residual / scale_A (0 where not
    valid), X[..., 6] = A valid, X[..., 7] and X[..., 8] the same for B.
    With neighbor angle inputs on instead, three follow per channel:
    X[..., 5] = cos d, X[..., 6] = sin d, X[..., 7] = valid, zero where
    not valid.
    Returns (X float32, mag_valid bool (T, M, C), ang_valid bool (T, M, C)),
    the same two validity arrays that sit in X, used for targets and loss.
    """
    # Sizes
    T, M, C = arrays["rms"].shape
    # Output, F or F plus the physics features
    X = np.zeros((T, M, C, n_features()), dtype=np.float32)
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
    # Physics features, when on
    if PHYSICS is not None:
        # The arrays must be there
        if "phys_r" not in arrays:
            # Raise
            raise ValueError("physics inputs are on but the month arrays carry no physics values")
        # A scale is needed to build inputs
        if PHYSICS["scale"] is None:
            # Raise
            raise ValueError("physics inputs are on but no physics scale is set; compute or load it first")
        # Each side: scaled residual then flag, broadcast over meters and channels
        for k in range(len(PHYSICS_SIDES)):
            # Scaled residual, already zero where not valid
            X[..., F + 2 * k] = (arrays["phys_r"][:, k] / PHYSICS["scale"][k])[:, None, None]
            # Valid flag
            X[..., F + 2 * k + 1] = arrays["phys_valid"][:, k][:, None, None]
    # Neighbor angle features, when on
    if ANGDIFF is not None:
        # The arrays must be there
        if "nb_cos" not in arrays:
            # Raise
            raise ValueError("neighbor angle inputs are on but the month arrays carry no neighbor angle values")
        # Cos of the difference, zero where not valid
        X[..., F] = arrays["nb_cos"]
        # Sin
        X[..., F + 1] = arrays["nb_sin"]
        # Flag
        X[..., F + 2] = arrays["nb_valid"]
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
