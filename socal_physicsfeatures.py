# =============================================================================
# File:        socal_physicsfeatures.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-09
#
# Purpose:     Helper functions that turn the physics baseline's results into
#              model inputs. Imported by main_physicsfeatures.py. For every
#              feature month (<YYYY-MM>.npz from main_features.py) it writes
#              <YYYY-MM>_physics.npz with, per ten-second slot and per circuit
#              side (A and B), the physics residual and a validity flag. The
#              residual is the per-term residual of the least-squares fit of
#              the readings to the fixed reference wiring; a slot whose status
#              is anything but "ok" (no data, too few readings, undetermined,
#              rejected for timing, solver failed) or that has no physics row
#              is NOT valid and gets 0 with valid False. The two sides are
#              kept separate because they come from different circuits and
#              different configurations. Nothing is interpolated or filled.
#
#              It also builds the availability table: for every block of a
#              splits file and each side, how many slots have a valid physics
#              value, and, for the event blocks, how many inside and outside
#              the scored episode windows. That table is read before any
#              model is trained with these inputs.
#
# What it writes, under <features_dir>:
#              <YYYY-MM>_physics.npz  t int64 (T,) the month's slot times, identical to <YYYY-MM>.npz
#                                     r_a float32 (T,) A-side per-term residual (fraction), 0 where not valid
#                                     valid_a bool (T,) A-side value present and status ok
#                                     r_b, valid_b                 the same for the B side
#              physics_manifest.json  the result folders and their settings (config, threshold, policy, rank check),
#                                     the months written, the array names
#
# Functions:
#   load_results(physics_dir)                every result table in the folder as one table by UTC time, each matched to its summary
#   result_settings(physics_dir, required)   the settings recorded in the folder's summary files: present, identical, and
#                                            equal to the required safeguards (rank check on, the agreed threshold and policy)
#   month_arrays(results, t)                 (r, valid) aligned to a month's slot times
#   write_month(features_dir, month, t, sides)
#                                            save <month>_physics.npz
#   write_manifest(features_dir, sides, months, settings)
#                                            save physics_manifest.json
#   availability(features_dir, splits, episodes, before, after)
#                                            rows per block and side with slot and valid counts, episode windows by the
#                                            scorer's scope rule (only episodes whose whole window lies in the block)
#   print_availability(rows)                 the table
#   save_availability(features_dir, rows, settings)
#                                            physics_availability.json
# =============================================================================

# Standard library: file name patterns
import glob
# Standard library: JSON
import json
# Standard library: paths
import os
# Third party: numerics
import numpy as np
# Third party: tables and timestamps
import pandas as pd

# The model data module, for the split-block bounds and the month convention the feature files use
import socal_modeldata as sd
# The scoring module, for its episode scope rule
import socal_score as ss

# The two sides, in the order the arrays carry them
SIDES = ("a", "b")
# Keys of the summary files that must be present and agree across a folder's blocks
SETTING_KEYS = ("config", "solver", "current_weight", "freq_tolerance", "reference_time", "rank_check", "rank_rel_tol", "offset_threshold", "offset_policy", "offset_meters")
# The safeguards an approved feature source must have been produced with
REQUIRED_SETTINGS = {"rank_check": True, "offset_threshold": 0.5, "offset_policy": "whole_slot"}


def load_results(physics_dir):
    """
    Read every result table in the folder as one table indexed by UTC slot
    time with the columns residual_mean and status, sorted, with duplicate
    timestamps refused (overlapping saved windows would double-count).
    Every table must have its summary file of the same name beside it,
    whose label names the table and whose slot count equals the table's
    rows, so a stray or truncated file cannot pass as a result.
    """
    # Files
    paths = sorted(glob.glob(os.path.join(physics_dir, "*.parquet")))
    # None
    if not paths:
        # Refuse
        raise FileNotFoundError(f"no .parquet results in {physics_dir}")
    # Pieces
    pieces = []
    # Each table with its summary
    for p in paths:
        # Summary path
        jpath = p[:-len(".parquet")] + ".json"
        # Must exist
        if not os.path.exists(jpath):
            # Refuse
            raise FileNotFoundError(f"{p} has no summary {jpath}")
        # Read the summary
        with open(jpath) as f:
            # Parse
            body = json.load(f)
        # Label must name the table
        if body.get("label") != os.path.basename(p)[:-len(".parquet")]:
            # Refuse
            raise ValueError(f"{jpath}: label {body.get('label')!r} does not name {os.path.basename(p)}")
        # Read the two needed columns
        table = pd.read_parquet(p, columns=["residual_mean", "status"])
        # Slot count must match
        if int(body.get("slots", -1)) != len(table):
            # Refuse
            raise ValueError(f"{jpath}: {body.get('slots')} slots recorded but {p} has {len(table)} rows")
        # Keep
        pieces.append(table)
    # One table
    res = pd.concat(pieces).sort_index()
    # UTC index
    res.index = pd.to_datetime(res.index, utc=True)
    # Duplicates
    dup = int(res.index.duplicated().sum())
    # Refuse
    if dup:
        # Stop
        raise ValueError(f"{dup} duplicate timestamps across the result files in {physics_dir}; saved windows overlap")
    # An ok slot must carry a finite residual
    bad = int(((res["status"] == "ok") & ~np.isfinite(res["residual_mean"])).sum())
    # Refuse
    if bad:
        # Stop
        raise ValueError(f"{bad} slots marked ok have a non-finite residual in {physics_dir}")
    # Return
    return res


def result_settings(physics_dir, required=None):
    """
    The settings recorded in the folder's summary JSON files. Every
    SETTING_KEY must be present in every file (a summary without them is
    not a result of the current baseline), the settings must be identical
    across every block (one configuration, one filter rule per side), and
    they must equal the required safeguards (REQUIRED_SETTINGS unless
    given): rank check on, the agreed offset threshold and policy. A folder
    of unfiltered or unchecked results is refused as a feature source.
    Returns the dictionary of SETTING_KEYS plus the number of blocks.
    """
    # Required safeguards
    required = REQUIRED_SETTINGS if required is None else required
    # Summary files
    paths = sorted(glob.glob(os.path.join(physics_dir, "*.json")))
    # None
    if not paths:
        # Refuse
        raise FileNotFoundError(f"no .json summaries in {physics_dir}")
    # Settings per file
    seen = []
    # Each
    for p in paths:
        # Read
        with open(p) as f:
            # Parse
            body = json.load(f)
        # Every setting must be recorded
        missing = [k for k in SETTING_KEYS if k not in body or body[k] is None]
        # Refuse
        if missing:
            # Stop
            raise ValueError(f"{p}: settings {missing} are missing or null; not a result of the current baseline (rank check and offset filter recorded)")
        # Keep the setting keys
        seen.append({k: body[k] for k in SETTING_KEYS})
    # All identical
    for s in seen[1:]:
        # Compare
        if s != seen[0]:
            # Refuse
            raise ValueError(f"result files in {physics_dir} were made with different settings: {seen[0]} against {s}")
    # The required safeguards
    for k, want in required.items():
        # Compare
        if seen[0][k] != want:
            # Refuse
            raise ValueError(f"{physics_dir}: {k} is {seen[0][k]!r} but the approved feature source needs {want!r}")
    # Return with the block count
    return dict(seen[0], blocks=len(paths))


def month_arrays(results, t):
    """
    Align the results to a month's slot times t (seconds since 1970 UTC).
    Returns (r, valid): r float32 (T,) the residual_mean where the status is
    ok and 0 elsewhere; valid bool (T,) True only where a row exists with
    status ok and a finite residual. Slots with no physics row are not valid.
    """
    # Month times as UTC timestamps
    times = pd.to_datetime(np.asarray(t, dtype=np.int64), unit="s", utc=True)
    # Rows for those times, NaN where absent
    part = results.reindex(times)
    # Valid: a row with status ok and a finite residual
    valid = (part["status"] == "ok").to_numpy() & np.isfinite(part["residual_mean"].to_numpy(dtype=np.float64))
    # Residual, zero where not valid
    r = np.where(valid, part["residual_mean"].to_numpy(dtype=np.float64), 0.0).astype(np.float32)
    # Return
    return r, valid


def write_month(features_dir, month, t, sides):
    """Save <month>_physics.npz with t and, per side, r_<side> and valid_<side>. sides is {side: (r, valid)}."""
    # Arrays
    arrays = {"t": np.asarray(t, dtype=np.int64)}
    # Each side
    for side in SIDES:
        # Values
        r, valid = sides[side]
        # Lengths must match
        if len(r) != len(t) or len(valid) != len(t):
            # Refuse
            raise ValueError(f"{month}: side {side} arrays do not match the slot count")
        # Store
        arrays[f"r_{side}"] = r.astype(np.float32)
        # Flag
        arrays[f"valid_{side}"] = valid.astype(bool)
    # Path
    path = os.path.join(features_dir, f"{month}_physics.npz")
    # Write to a temporary name, then rename
    tmp = path + ".tmp.npz"
    # Save
    np.savez_compressed(tmp, **arrays)
    # Atomic replace
    os.replace(tmp, path)
    # Return
    return path


def write_manifest(features_dir, sides, months, settings):
    """Save physics_manifest.json: per side the result folder and its settings, the months, the arrays."""
    # Body
    manifest = {
        # Arrays
        "arrays": {"t": "int64 (T,) slot start, seconds since 1970 UTC, identical to the feature month",
                   "r_<side>": "float32 (T,) per-term physics residual as a fraction (0.005 = 0.5 percent), 0 where not valid",
                   "valid_<side>": "bool (T,) True only where a physics row exists with status ok and a finite residual"},
        # Sides
        "sides": {side: {"physics_dir": sides[side], "settings": settings[side]} for side in SIDES},
        # Months
        "months": list(months),
        # Rule
        "validity_rule": "any status other than ok, and any slot without a physics row, is not valid; nothing is interpolated",
    }
    # Path
    path = os.path.join(features_dir, "physics_manifest.json")
    # Write
    with open(path, "w") as f:
        # JSON
        json.dump(manifest, f, indent=2)
    # Return
    return path


def availability(features_dir, splits, episodes, before, after):
    """
    For every block of every split and each side: slots in the block,
    slots with a valid physics value, and for blocks that hold scored
    episodes, the same inside and outside the episode windows (first
    scored change minus before, to last scored change plus after). Which
    episodes count follows the scorer's scope rule
    (socal_score.episodes_in_scope): only an episode whose whole window
    lies inside the block; an episode only partly covered by the block is
    listed in the row as partial and not counted. Reads the
    <month>_physics.npz files. Returns a list of row dictionaries.
    """
    # Rows
    rows = []
    # Each split, by the four names the model data module uses; the splits file also carries other keys (notes, version) that are not block lists
    for split_name in ("train", "validation", "test_events", "test_normal"):
        # Each block
        for block in splits.get(split_name, []):
            # Bounds in epoch seconds, as the model data module reads them
            start_s, end_s = sd.block_seconds(block)
            # Months the block touches, in the feature files' fixed-PST convention
            months = sd.months_in_block(block)
            # Collect
            t_all, valid_all = [], {side: [] for side in SIDES}
            # Each month
            for month in months:
                # Path
                path = os.path.join(features_dir, f"{month}_physics.npz")
                # Missing month: the block is counted on what exists, and the gap is reported
                if not os.path.exists(path):
                    # Next
                    continue
                # Load
                with np.load(path) as data:
                    # Times
                    t = data["t"].astype(np.int64)
                    # Keep rows in the block
                    keep = (t >= start_s) & (t < end_s)
                    # Store
                    t_all.append(t[keep])
                    # Each side
                    for side in SIDES:
                        # Flag
                        valid_all[side].append(data[f"valid_{side}"].astype(bool)[keep])
            # Nothing
            if not t_all:
                # Row saying so
                rows.append({"split": split_name, "block": f"{block['start']}_{block['end']}", "slots": 0, "months_missing": list(months)})
                # Next
                continue
            # Stack
            t = np.concatenate(t_all)
            # Block bounds as timestamps, for the scope rule
            b0, b1 = pd.Timestamp(start_s, unit="s", tz="UTC"), pd.Timestamp(end_s - 10, unit="s", tz="UTC")
            # Episodes whose whole window lies in this block, and those only partly covered
            scoped, partial = ss.episodes_in_scope(episodes, [(b0, b1)], before, after)
            # Episode windows inside the block
            times = pd.to_datetime(t, unit="s", utc=True)
            # Mask
            inside = np.zeros(len(t), dtype=bool)
            # Each scoped episode
            for _, e in scoped.iterrows():
                # Window
                inside |= np.asarray((times >= e["first"] - before) & (times <= e["last"] + after))
            # Row
            row = {"split": split_name, "block": f"{block['start']}_{block['end']}", "slots": int(len(t)), "inside_slots": int(inside.sum()),
                   "episodes_in_scope": [int(x) for x in scoped["episode"]], "episodes_partial": [int(x) for x in partial["episode"]],
                   "months_missing": [m for m in months if not os.path.exists(os.path.join(features_dir, f"{m}_physics.npz"))]}
            # Each side
            for side in SIDES:
                # Flag
                v = np.concatenate(valid_all[side])
                # Counts
                row[f"valid_{side}"] = int(v.sum())
                # Inside
                row[f"valid_{side}_inside"] = int((v & inside).sum())
                # Outside
                row[f"valid_{side}_outside"] = int((v & ~inside).sum())
            # Keep
            rows.append(row)
    # Return
    return rows


def print_availability(rows):
    """Print the availability table: per block and side, valid fraction overall and inside and outside the episode windows."""
    # Header
    print(f"\n{'split':<14} {'block':<24} {'slots':>8} {'A valid':>8} {'frac':>6} {'B valid':>8} {'frac':>6} {'in win':>7} {'A in':>6} {'A out':>6} {'B in':>6} {'B out':>6}  missing months")
    # Each row
    for r in rows:
        # No data
        if r["slots"] == 0:
            # Line
            print(f"{r['split']:<14} {r['block']:<24} {0:>8}  no physics files for {r['months_missing']}")
            # Next
            continue
        # Fractions
        fa, fb = r["valid_a"] / r["slots"], r["valid_b"] / r["slots"]
        # Inside and outside fractions, n/a without windows
        n_in, n_out = r["inside_slots"], r["slots"] - r["inside_slots"]
        # Text
        def frac(num, den):
            # Fraction or n/a
            return f"{num / den:6.3f}" if den else f"{'n/a':>6}"
        # Line
        print(f"{r['split']:<14} {r['block']:<24} {r['slots']:>8} {r['valid_a']:>8} {fa:6.3f} {r['valid_b']:>8} {fb:6.3f} {n_in:>7} {frac(r['valid_a_inside'], n_in)} {frac(r['valid_a_outside'], n_out)} {frac(r['valid_b_inside'], n_in)} {frac(r['valid_b_outside'], n_out)}  {r['months_missing'] if r['months_missing'] else ''}" + (f"  episodes {r['episodes_in_scope']}" if r["episodes_in_scope"] else "") + (f"  partial (not counted) {r['episodes_partial']}" if r["episodes_partial"] else ""))


def save_availability(features_dir, rows, settings):
    """Save the availability rows and the window settings as physics_availability.json next to the physics files."""
    # Path
    path = os.path.join(features_dir, "physics_availability.json")
    # Write
    with open(path, "w") as f:
        # JSON
        json.dump({"settings": settings, "rows": rows}, f, indent=1)
    # Return
    return path
