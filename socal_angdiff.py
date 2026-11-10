# =============================================================================
# File:        socal_angdiff.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-09
#
# Purpose:     Helper functions for experiment 2, the neighbor angle
#              difference as a model input. The experiment tests whether
#              giving the forecasting model, explicitly, how far each
#              meter's voltage angle sits from its wired neighbors' angles
#              helps forecasting and detection. Used by socal_modeldata.py
#              (which computes the inputs at load time on Colab) and by
#              main_latefeatures.py (which writes the late-capture files on
#              the Mac and prints the availability report), so both sides
#              run the same calculation.
#
#              The feature, per meter, physical phase (a, b, c) and slot:
#                d = wrap( own angle - circular mean of the neighbors' angles
#                          on the same physical phase )
#              given to the model as cos d, sin d and a validity flag, placed
#              on the meter's own voltage channel for that phase. Current
#              channels carry zeros with the flag 0.
#
#              Physical phases, not channel names: each meter's phase map
#              (graph_meters*.json, "phase") says which channel measures
#              which phase. L1 is phase a on most meters but phase c on
#              egauge_13, 14 and 15. A channel whose label is not a
#              phase-to-neutral voltage (egauge_17 and 18 measure the
#              line-to-line voltage ab) has no comparable phase and is left
#              out, as own channel and as neighbor; the availability report
#              shows that loss.
#
#              Neighbors: the meter graph of the period in force at the
#              slot (the time-indexed file's periods, or the one graph of a
#              fixed file), equal weights. The saved meter graph has unit
#              edge weights, so this equals weighting by them.
#
#              Timing validity: an angle is unusable, as own angle and as
#              neighbor, when the meter's capture offset is missing,
#              not finite, or at or beyond the threshold (0.5 s, the
#              whole-slot filter's cutoff; rejecting a missing offset is an
#              extra conservative choice). Passing the cutoff does not
#              guarantee correct alignment. The offset is not in the
#              feature files, so main_latefeatures.py writes it per month
#              as <YYYY-MM>_late.npz (bool (T, M)) from the processed
#              tables. The five standard inputs are not changed by this
#              rule.
#
#              Neighbor mean: the mean resultant length R (length of the
#              mean of the neighbors' unit vectors, 0 to 1) must be at least
#              a cutoff (default 0.5), so two neighbors pointing opposite
#              ways do not define a direction. Where the feature is not
#              valid, both numbers and the flag are 0, never cos 0 = 1.
#
# Functions:
#   Phases and graph
#     phase_channel_index(meter_records, meters, channels)
#                                       (M, 3) channel index per meter and phase, -1 where none
#     phase_map_identity(meter_records, meters)
#                                       short hash of the phase map, for the run's options
#     timeline_from_graph(body)         periods and graphs from a fixed or time-indexed graph file
#     adjacency_matrices(timeline, meter_records, meters)
#                                       one (M, M) equal-weight adjacency per edge set
#     graph_per_slot(t, timeline)       edge-set index per slot time, -1 outside every period
#   Late files
#     read_late_month(processed_dir, month, meters, t, threshold)
#                                       (T, M) late flags from the processed tables' toff
#     write_late(features_dir, month, t, late, meters)
#                                       save <month>_late.npz atomically, with the meter order
#     write_late_manifest(features_dir, months, threshold, processed_dir, meters)
#                                       save late_manifest.json
#     read_late_manifest(features_dir, late_threshold)
#                                       read and validate late_manifest.json (threshold present, finite, positive, equal to the requested one)
#     load_late(features_dir, month, t, meters)
#                                       read <month>_late.npz; slot times, shape (T, M) and meter order checked
#   The feature
#     neighbor_differences(ang, ang_valid, late, phase_idx, adjacency, graph_slot, min_resultant)
#                                       (nb_cos, nb_sin, nb_valid) each (T, M, C), plus reason counts
#   Availability
#     availability(features_dir, splits, episodes, before, after, timeline, manifest, settings)
#                                       rows per block with counts per meter and voltage channel
#     print_availability(rows, manifest)
#                                       per-block totals and per-split meter-by-channel tables
#     save_availability(features_dir, rows, settings)
#                                       angdiff_availability.json
# =============================================================================

# Standard library: hashing for identities
import hashlib
# Standard library: JSON
import json
# Standard library: paths
import os
# Third party: numerics
import numpy as np
# Third party: timestamps and tables
import pandas as pd

# The three added inputs per channel, in order
ANGDIFF_FEATURES = ("nb_cos", "nb_sin", "nb_valid")
# Physical phases in array order
PHASES = ("a", "b", "c")
# Phase-to-neutral voltage labels per phase, as the phase maps write them
PHASE_LABELS = {"ag": 0, "bg": 1, "cg": 2}
# The voltage channel names
VOLTAGE_CHANNELS = ("L1", "L2", "L3")
# Default capture-offset cutoff in seconds, the whole-slot filter's value
LATE_THRESHOLD = 0.5
# Default minimum mean resultant length of the neighbor mean
MIN_RESULTANT = 0.5
# Column separator and offset column name in the processed tables
SEP = "|"
# Offset column
TOFF = "toff"
# Reasons a voltage-channel opportunity has no valid feature, in reporting order
REASONS = ("unmapped", "own_late", "own_angle_invalid", "no_usable_neighbor", "low_resultant")


# -----------------------------------------------------------------------------
# Phases and graph
# -----------------------------------------------------------------------------

def phase_channel_index(meter_records, meters, channels):
    """
    For every meter in manifest order and every physical phase a, b, c:
    the index of the channel that measures that phase-to-neutral voltage,
    or -1 when the meter has no such channel (no phase map, or a label
    that is not ag, bg or cg). Shape (M, 3).
    """
    # Output
    idx = np.full((len(meters), len(PHASES)), -1, dtype=np.int64)
    # Each meter
    for mi, meter in enumerate(meters):
        # Its phase map, if any
        phase = (meter_records.get(meter) or {}).get("phase") or {}
        # Each voltage channel
        for channel, label in phase.items():
            # Only voltage channels
            if channel not in channels or channel not in VOLTAGE_CHANNELS:
                # Skip
                continue
            # Only phase-to-neutral labels
            if label not in PHASE_LABELS:
                # Skip, this channel has no comparable phase
                continue
            # Record
            idx[mi, PHASE_LABELS[label]] = channels.index(channel)
    # Return
    return idx


def phase_map_identity(meter_records, meters):
    """A short hash of the voltage phase map of the listed meters, so a run records which mapping it used."""
    # The map, meters in order, channels sorted
    text = json.dumps({m: {c: l for c, l in sorted(((meter_records.get(m) or {}).get("phase") or {}).items()) if c in VOLTAGE_CHANNELS} for m in meters}, sort_keys=True)
    # Hash
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def timeline_from_graph(body):
    """
    Periods and graphs from a graph file: a time-indexed file gives its own
    periods and edge sets; a fixed file becomes one edge set in force for
    all time. Returns {"graphs": [...], "periods": [...], "meters": {...}}
    with each period holding start_s, end_s (epoch seconds) and graph.
    """
    # Time-indexed layout
    if "periods" in body and "graphs" in body:
        # Periods with epoch bounds
        periods = [{"start_s": int(pd.Timestamp(p["start_utc"]).timestamp()), "end_s": int(pd.Timestamp(p["end_utc_exclusive"]).timestamp()), "graph": int(p["graph"])} for p in body["periods"]]
        # Edge lists
        graphs = [{"edges": g["edges"]} for g in body["graphs"]]
        # Return
        return {"graphs": graphs, "periods": periods, "meters": body["meters"]}
    # Fixed layout: one graph for all time
    return {"graphs": [{"edges": body["edges"]}], "periods": [{"start_s": -2 ** 62, "end_s": 2 ** 62, "graph": 0}], "meters": body["meters"]}


def adjacency_matrices(timeline, meter_records, meters):
    """
    One (M, M) float32 adjacency matrix per edge set of the timeline, in
    manifest meter order, 1 where two meters share an edge, symmetric,
    zero diagonal. Meters without a node in the graph get no edges.
    """
    # Node index per meter, -1 when absent
    node = np.array([int((meter_records.get(m) or {}).get("node", -1)) for m in meters], dtype=np.int64)
    # Meter index per node
    by_node = {int(n): mi for mi, n in enumerate(node) if n >= 0}
    # Matrices
    out = []
    # Each edge set
    for g in timeline["graphs"]:
        # Empty
        A = np.zeros((len(meters), len(meters)), dtype=np.float32)
        # Each edge
        for e in g["edges"]:
            # Meter indices at both ends
            a, b = by_node.get(int(e["source"])), by_node.get(int(e["target"]))
            # Both measured and different
            if a is not None and b is not None and a != b:
                # Equal weight, both directions
                A[a, b] = 1.0
                # Back
                A[b, a] = 1.0
        # Keep
        out.append(A)
    # Return
    return out


def graph_per_slot(t, timeline):
    """Edge-set index per slot time (T,), -1 for a slot outside every period."""
    # Output
    out = np.full(len(t), -1, dtype=np.int64)
    # Each period
    for p in timeline["periods"]:
        # Slots inside
        inside = (t >= p["start_s"]) & (t < p["end_s"])
        # Assign
        out[inside] = p["graph"]
    # Return
    return out


# -----------------------------------------------------------------------------
# Late files
# -----------------------------------------------------------------------------

def read_late_month(processed_dir, month, meters, t, threshold):
    """
    The late flags of one month from the processed tables: late[t, m] is
    True when meter m's capture offset at slot t is missing, not finite,
    or at or beyond the threshold in seconds, and when the meter has no
    table for the month. Aligned to the slot times t of the feature month.
    """
    # Output, everything late until shown otherwise
    late = np.ones((len(t), len(meters)), dtype=bool)
    # Each meter
    for mi, meter in enumerate(meters):
        # Table path
        path = os.path.join(processed_dir, month, f"{meter}.parquet")
        # No table: stays late
        if not os.path.exists(path):
            # Next
            continue
        # Offset column
        col = f"{meter}{SEP}{TOFF}"
        # Read only that column
        table = pd.read_parquet(path, columns=[col])
        # Slot times of the table as epoch seconds, whatever the index resolution; a time zone is converted to UTC, none is taken as UTC
        index = table.index.tz_convert(None) if table.index.tz is not None else table.index
        # Seconds
        ts = np.asarray(index, dtype="datetime64[s]").astype(np.int64)
        # Offsets
        toff = table[col].to_numpy(dtype=np.float64)
        # Usable: finite and below the threshold
        ok = np.isfinite(toff) & (toff < threshold)
        # Map table rows onto the feature grid
        pos = np.searchsorted(t, ts)
        # Rows whose time is on the grid
        hit = (pos < len(t)) & (t[np.minimum(pos, len(t) - 1)] == ts)
        # Not late where usable and on the grid
        late[pos[hit & ok], mi] = False
    # Return
    return late


def write_late(features_dir, month, t, late, meters):
    """Save <month>_late.npz with t, late (T, M) and the ordered meter list the columns follow, written to a temporary name then renamed."""
    # Shape must agree with the meter list
    if late.shape != (len(t), len(meters)):
        # Raise
        raise ValueError(f"late array has shape {late.shape}, expected {(len(t), len(meters))}")
    # Path
    path = os.path.join(features_dir, f"{month}_late.npz")
    # Temporary
    tmp = path + ".tmp.npz"
    # Save
    np.savez_compressed(tmp, t=t.astype(np.int64), late=late.astype(bool), meters=np.array(list(meters), dtype=str))
    # Rename
    os.replace(tmp, path)
    # Return
    return path


def write_late_manifest(features_dir, months, threshold, processed_dir, meters):
    """Save late_manifest.json: the rule, the threshold, the source folder, the ordered meter list and the months written."""
    # The threshold must be usable
    if not np.isfinite(threshold) or float(threshold) <= 0:
        # Raise
        raise ValueError(f"late threshold must be finite and positive, got {threshold}")
    # Path
    path = os.path.join(features_dir, "late_manifest.json")
    # Body
    body = {"rule": "late = capture offset missing, not finite, or >= threshold seconds; a meter without a table for the month is late throughout", "threshold_seconds": float(threshold), "processed_dir": os.path.abspath(processed_dir), "meters": list(meters), "months": list(months), "arrays": {"t": "int64 (T,) slot times", "late": "bool (T, M) in the order of 'meters'", "meters": "str (M,) the meter order of the columns"}}
    # Write
    with open(path, "w") as f:
        # JSON
        json.dump(body, f, indent=2)
    # Return
    return path


def read_late_manifest(features_dir, late_threshold):
    """
    Read late_manifest.json and check it: the threshold must be present,
    finite and positive, and equal to the requested late_threshold; the
    meter list must be present. Returns (body, raw bytes) so a caller can
    hash the bytes for the run's options.
    """
    # Path
    path = os.path.join(features_dir, "late_manifest.json")
    # Missing
    if not os.path.exists(path):
        # Raise
        raise FileNotFoundError(f"no late_manifest.json in {features_dir}; run main_latefeatures.py first")
    # Read
    with open(path, "rb") as f:
        # Bytes
        raw = f.read()
    # Parse
    body = json.loads(raw)
    # Threshold present
    recorded = body.get("threshold_seconds")
    # Must be a finite positive number
    if not isinstance(recorded, (int, float)) or isinstance(recorded, bool) or not np.isfinite(recorded) or recorded <= 0:
        # Raise
        raise ValueError(f"{path}: threshold_seconds is missing or not a finite positive number ({recorded!r})")
    # The requested one too
    if not np.isfinite(late_threshold) or late_threshold <= 0:
        # Raise
        raise ValueError(f"requested late threshold must be finite and positive, got {late_threshold}")
    # Must agree
    if abs(float(recorded) - float(late_threshold)) > 1e-12:
        # Raise
        raise ValueError(f"{path} was written with threshold {recorded} s but the requested threshold is {late_threshold}")
    # Meter list present
    if not isinstance(body.get("meters"), list) or not body["meters"]:
        # Raise
        raise ValueError(f"{path}: the ordered meter list is missing; rewrite the late files with the current main_latefeatures.py")
    # Return
    return body, raw


def load_late(features_dir, month, t, meters):
    """
    Read <month>_late.npz and return late (T, M) bool. The slot times must
    equal t, the stored meter order must equal meters exactly, and the
    array must have shape (T, M); anything else is refused, so a file for
    other meters or another order cannot be read silently.
    """
    # Path
    path = os.path.join(features_dir, f"{month}_late.npz")
    # Missing
    if not os.path.exists(path):
        # Raise
        raise FileNotFoundError(f"neighbor angle inputs are on but {path} is missing; run main_latefeatures.py")
    # Load
    with np.load(path) as data:
        # Meter order recorded
        if "meters" not in data.files:
            # Raise
            raise ValueError(f"{path}: no meter order recorded; rewrite the late files with the current main_latefeatures.py")
        # Must match exactly
        stored = [str(m) for m in data["meters"]]
        # Compare
        if stored != list(meters):
            # Raise
            raise ValueError(f"{path}: meter order {stored} differs from the manifest's {list(meters)}")
        # Times must match
        if not np.array_equal(data["t"].astype(np.int64), t):
            # Raise
            raise ValueError(f"{path}: slot times differ from the feature month")
        # Flags
        late = data["late"]
        # Shape
        if late.shape != (len(t), len(meters)):
            # Raise
            raise ValueError(f"{path}: late array has shape {late.shape}, expected {(len(t), len(meters))}")
        # Bool
        late = late.astype(bool)
    # Return
    return late


# -----------------------------------------------------------------------------
# The feature
# -----------------------------------------------------------------------------

def neighbor_differences(ang, ang_valid, late, phase_idx, adjacency, graph_slot, min_resultant=MIN_RESULTANT):
    """
    Compute the neighbor angle difference for every slot, meter and
    physical phase, and place it on the meter's own channel for that phase.

    ang         (T, M, C) float, aligned angle in degrees
    ang_valid   (T, M, C) bool, the standard angle validity
    late        (T, M) bool, the meter's capture is late (unusable)
    phase_idx   (M, 3) int, channel index per phase, -1 where none
    adjacency   list of (M, M) matrices, one per edge set
    graph_slot  (T,) int, edge-set index per slot, -1 for none
    Returns (nb_cos, nb_sin, nb_valid) each (T, M, C), float32 and bool,
    and a dictionary of reason counts over the voltage-channel
    opportunities (T x M x 3 phases): valid, unmapped, own_late,
    own_angle_invalid, no_usable_neighbor, low_resultant.
    """
    # Sizes
    T, M, C = ang.shape
    # Outputs
    nb_cos = np.zeros((T, M, C), dtype=np.float32)
    # Sin
    nb_sin = np.zeros((T, M, C), dtype=np.float32)
    # Flag
    nb_valid = np.zeros((T, M, C), dtype=bool)
    # Reason counts
    reasons = {k: 0 for k in ("valid",) + REASONS}
    # Each physical phase
    for p in range(len(PHASES)):
        # Channel per meter for this phase
        idx = phase_idx[:, p]
        # Meters with a channel
        mapped = idx >= 0
        # Unmapped opportunities: every slot of every unmapped meter
        reasons["unmapped"] += int(T * (~mapped).sum())
        # Nothing mapped at all
        if not mapped.any():
            # Next phase
            continue
        # Safe index for gathering (unmapped meters read channel 0, masked out below)
        safe = np.where(mapped, idx, 0)
        # Own angle per meter, (T, M)
        own = np.take_along_axis(ang, safe[None, :, None], axis=2)[:, :, 0]
        # Own validity, (T, M)
        own_ok = np.take_along_axis(ang_valid, safe[None, :, None], axis=2)[:, :, 0] & mapped[None, :]
        # Usable: valid and not late
        usable = own_ok & ~late
        # Reasons on the own side, counted on mapped meters only
        reasons["own_late"] += int((mapped[None, :] & late).sum())
        # Angle invalid and not late
        reasons["own_angle_invalid"] += int((mapped[None, :] & ~late & ~own_ok).sum())
        # Own unit vectors, zero where unusable
        rad = np.deg2rad(own)
        # Cos
        c = np.where(usable, np.cos(rad), 0.0).astype(np.float64)
        # Sin
        s = np.where(usable, np.sin(rad), 0.0).astype(np.float64)
        # Usable as float for the counts
        u = usable.astype(np.float64)
        # Neighbor sums, filled per edge set
        sum_c = np.zeros((T, M), dtype=np.float64)
        # Sin sum
        sum_s = np.zeros((T, M), dtype=np.float64)
        # Count of usable neighbors
        cnt = np.zeros((T, M), dtype=np.float64)
        # Each edge set
        for k, A in enumerate(adjacency):
            # Slots under this edge set
            rows = graph_slot == k
            # None
            if not rows.any():
                # Next
                continue
            # Neighbor sums: A is symmetric, so row m of (x @ A) sums over m's neighbors
            sum_c[rows] = c[rows] @ A
            # Sin
            sum_s[rows] = s[rows] @ A
            # Count
            cnt[rows] = u[rows] @ A
        # A usable neighbor exists
        has_nb = cnt > 0
        # Mean resultant length, 0 where no neighbor
        R = np.where(has_nb, np.hypot(sum_c, sum_s) / np.maximum(cnt, 1.0), 0.0)
        # Direction defined
        good = has_nb & (R >= min_resultant)
        # Valid feature: own usable and a defined neighbor direction
        valid = usable & good
        # Reasons on the neighbor side, counted where own is usable
        reasons["no_usable_neighbor"] += int((usable & ~has_nb).sum())
        # Low resultant
        reasons["low_resultant"] += int((usable & has_nb & ~good).sum())
        # Valid
        reasons["valid"] += int(valid.sum())
        # Neighbor mean angle
        mean = np.arctan2(sum_s, sum_c)
        # Difference, wrapped by taking cos and sin directly
        d = rad - mean
        # Place on the own channel, zero where not valid
        vc = np.where(valid, np.cos(d), 0.0).astype(np.float32)
        # Sin
        vs = np.where(valid, np.sin(d), 0.0).astype(np.float32)
        # Scatter into the channel positions, mapped meters only
        for mi in np.nonzero(mapped)[0]:
            # Channel
            ci = idx[mi]
            # Cos
            nb_cos[:, mi, ci] = vc[:, mi]
            # Sin
            nb_sin[:, mi, ci] = vs[:, mi]
            # Flag
            nb_valid[:, mi, ci] = valid[:, mi]
    # Return
    return nb_cos, nb_sin, nb_valid, reasons


# -----------------------------------------------------------------------------
# Availability
# -----------------------------------------------------------------------------

def availability(features_dir, splits, episodes, before, after, timeline, manifest, settings):
    """
    For every block of the four splits: per meter and voltage channel, the
    opportunities (slots where the channel maps to a phase), the valid
    features, and the valid counts inside and outside the scored episode
    windows (scope rule as the scorer's), plus the reason counts over all
    voltage-channel opportunities. Reads the feature months and the late
    files and recomputes the feature with the given settings. Returns a
    list of row dictionaries.
    """
    # The model data and scoring modules, imported here to avoid an import cycle
    import socal_modeldata as sd
    # Scorer
    import socal_score as ss
    # Meters and channels
    meters, channels = list(manifest["meters"]), list(manifest["channels"])
    # Phase map and adjacency from the timeline
    phase_idx = phase_channel_index(timeline["meters"], meters, channels)
    # Adjacency
    adjacency = adjacency_matrices(timeline, timeline["meters"], meters)
    # Voltage channel indices
    vcols = [channels.index(c) for c in VOLTAGE_CHANNELS if c in channels]
    # Rows
    rows = []
    # Each split
    for split_name in ("train", "validation", "test_events", "test_normal"):
        # Each block
        for block in splits.get(split_name, []):
            # Bounds
            start_s, end_s = sd.block_seconds(block)
            # Months
            months = sd.months_in_block(block)
            # Collect
            t_all, valid_all, mapped_all, reasons_all = [], [], [], {k: 0 for k in ("valid",) + REASONS}
            # Missing months
            missing = []
            # Each month
            for month in months:
                # Feature month
                arrays = sd.load_month(features_dir, month, allow_missing=True, manifest=manifest)
                # Late file
                lpath = os.path.join(features_dir, f"{month}_late.npz")
                # Missing either
                if arrays is None or not os.path.exists(lpath):
                    # Record
                    missing.append(month)
                    # Next
                    continue
                # Rows in the block
                part = sd.slice_block(arrays, start_s, end_s)
                # None inside
                if part is None:
                    # Next
                    continue
                # Late flags, meter order checked, sliced the same way
                late = load_late(features_dir, month, arrays["t"], meters)[(arrays["t"] >= start_s) & (arrays["t"] < end_s)]
                # Edge set per slot
                gs = graph_per_slot(part["t"], timeline)
                # The feature
                _, _, nb_valid, reasons = neighbor_differences(part["ang"], part["ang_valid"], late, phase_idx, adjacency, gs, settings["min_resultant"])
                # Keep times
                t_all.append(part["t"])
                # Valid flags on the voltage channels, (T, M, 3)
                valid_all.append(nb_valid[:, :, vcols])
                # Reasons
                for k, v in reasons.items():
                    # Sum
                    reasons_all[k] += v
            # Mapped voltage channels per meter, (M, 3): the channel maps to some phase
            mapped = np.zeros((len(meters), len(vcols)), dtype=bool)
            # Each meter
            for mi in range(len(meters)):
                # Each voltage channel
                for j, ci in enumerate(vcols):
                    # Mapped when any phase points at it
                    mapped[mi, j] = bool((phase_idx[mi] == ci).any())
            # Nothing
            if not t_all:
                # Row
                rows.append({"split": split_name, "block": f"{block['start']}_{block['end']}", "slots": 0, "months_missing": missing})
                # Next
                continue
            # Stack
            t = np.concatenate(t_all)
            # Valid
            valid = np.concatenate(valid_all)
            # Block bounds as timestamps, for the scope rule
            b0, b1 = pd.Timestamp(start_s, unit="s", tz="UTC"), pd.Timestamp(end_s - 10, unit="s", tz="UTC")
            # Episodes in scope and partial
            scoped, partial = ss.episodes_in_scope(episodes, [(b0, b1)], before, after)
            # Window mask
            times = pd.to_datetime(t, unit="s", utc=True)
            # Inside
            inside = np.zeros(len(t), dtype=bool)
            # Each scoped episode
            for _, e in scoped.iterrows():
                # Window
                inside |= np.asarray((times >= e["first"] - before) & (times <= e["last"] + after))
            # Per meter and channel counts
            per = {}
            # Each meter
            for mi, meter in enumerate(meters):
                # Each voltage channel
                for j, ci in enumerate(vcols):
                    # Opportunities: slots where the channel is mapped
                    opp = int(len(t)) if mapped[mi, j] else 0
                    # Valid
                    v = valid[:, mi, j]
                    # Store
                    per[f"{meter}|{channels[ci]}"] = {"opp": opp, "valid": int(v.sum()), "valid_inside": int((v & inside).sum()), "valid_outside": int((v & ~inside).sum())}
            # Totals
            total_opp = int(sum(x["opp"] for x in per.values()))
            # Valid
            total_valid = int(valid.sum())
            # Row
            rows.append({"split": split_name, "block": f"{block['start']}_{block['end']}", "slots": int(len(t)), "inside_slots": int(inside.sum()), "n_meters_mapped": int(mapped.any(axis=1).sum()),
                         "opportunities": total_opp, "valid": total_valid, "valid_inside": int((valid & inside[:, None, None]).sum()), "valid_outside": int((valid & ~inside[:, None, None]).sum()),
                         "opportunities_inside": int(inside.sum()) * int(mapped.sum()), "opportunities_outside": int((~inside).sum()) * int(mapped.sum()),
                         "reasons": reasons_all, "per_channel": per,
                         "episodes_in_scope": [int(x) for x in scoped["episode"]], "episodes_partial": [int(x) for x in partial["episode"]], "months_missing": missing})
    # Return
    return rows


def print_availability(rows, manifest):
    """
    Print, per block, the valid fraction over voltage-channel opportunities,
    inside and outside the episode windows, and the reason shares; then, per
    split, one table with a row per meter and a column per voltage channel
    (valid fraction over that split's blocks), so a meter that never gets
    the feature is visible.
    """
    # Header
    print(f"\n{'split':<12} {'block':<24} {'slots':>8} {'mapped':>6} {'opp':>10} {'valid':>10} {'frac':>6} {'in':>6} {'out':>6}  reasons (share of all slot-meter-phases, mapped or not)")
    # Each row
    for r in rows:
        # No data
        if r["slots"] == 0:
            # Line
            print(f"{r['split']:<12} {r['block']:<24} {0:>8}  no feature or late files for {r['months_missing']}")
            # Next
            continue
        # Fraction helper
        def frac(num, den):
            # Fraction or n/a
            return f"{num / den:6.3f}" if den else f"{'n/a':>6}"
        # Reason text
        reasons = ", ".join(f"{k} {r['reasons'][k] / max(r['opportunities'] + r['reasons']['unmapped'], 1):.3f}" for k in REASONS)
        # Line
        print(f"{r['split']:<12} {r['block']:<24} {r['slots']:>8} {r['n_meters_mapped']:>6} {r['opportunities']:>10} {r['valid']:>10} {frac(r['valid'], r['opportunities'])} {frac(r['valid_inside'], r['opportunities_inside'])} {frac(r['valid_outside'], r['opportunities_outside'])}  {reasons}" + (f"  episodes {r['episodes_in_scope']}" if r["episodes_in_scope"] else "") + (f"  partial (not counted) {r['episodes_partial']}" if r["episodes_partial"] else "") + (f"  missing {r['months_missing']}" if r["months_missing"] else ""))
    # Per split, per meter and channel
    for split_name in ("train", "validation", "test_events", "test_normal"):
        # Rows of the split with data
        part = [r for r in rows if r["split"] == split_name and r["slots"] > 0]
        # None
        if not part:
            # Next
            continue
        # Header
        print(f"\n{split_name}: valid fraction per meter and voltage channel over its blocks (dash = channel not a phase-to-neutral voltage, so never compared)")
        # Column header
        print(f"{'meter':<12} " + " ".join(f"{c:>8}" for c in VOLTAGE_CHANNELS))
        # Each meter
        for meter in manifest["meters"]:
            # Cells
            cells = []
            # Each channel
            for c in VOLTAGE_CHANNELS:
                # Sum over blocks
                opp = sum(r["per_channel"].get(f"{meter}|{c}", {}).get("opp", 0) for r in part)
                # Valid
                v = sum(r["per_channel"].get(f"{meter}|{c}", {}).get("valid", 0) for r in part)
                # Text
                cells.append(f"{v / opp:8.3f}" if opp else f"{'-':>8}")
            # Line
            print(f"{meter:<12} " + " ".join(cells))


def save_availability(features_dir, rows, settings):
    """Save the rows and settings as angdiff_availability.json next to the feature files."""
    # Path
    path = os.path.join(features_dir, "angdiff_availability.json")
    # Write
    with open(path, "w") as f:
        # JSON
        json.dump({"settings": settings, "rows": rows}, f, indent=1)
    # Return
    return path
