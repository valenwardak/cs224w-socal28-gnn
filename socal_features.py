# =============================================================================
# File:        socal_features.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-21
#
# Purpose:     Helper functions that turn the processed Parquet tables into
#              compact monthly feature files for model training on Colab.
#              Imported by main_features.py. The Parquet tables stay the
#              authoritative data; these files are a smaller copy holding
#              only what the models read, in fixed-order dense arrays.
#
# What it produces, under <out_dir>:
#              <YYYY-MM>.npz    one compressed file per month, all meters:
#                  t      int64   (T,)        slot time, seconds since 1970 UTC
#                  rms    float16 (T, M, C)   rms magnitude, volts or amps
#                  ang    float32 (T, M, C)   fundamental angle at the slot time, degrees
#                  freq   float32 (T, M)      frequency, hertz, one per meter
#                  mask   bool    (T, M, C)   True where a reading exists
#                  angle_ok bool  (T, M, C)   True where the angle passed the checks below
#                  with --keep-harmonics also, for k in 1, 2:
#                  mag<k> float16 (T, M, C)   harmonic magnitude
#                  ang<k> float32 (T, M, C)   harmonic angle at the slot time, degrees
#              manifest.json    meter order, channel order, array names and
#                               units, dtypes, alignment convention, version
#
#              T is the number of ten-second slots in the month, M the number
#              of meters in fixed order, C the six channels L1 L2 L3 S1 S2 S3.
#
# Precision:   Magnitudes are float16: about three and a half significant
#              digits, so a 9,526 volt reading is kept to within 4 volts.
#              Angles and frequency are float32, since inter-bus angle
#              differences and frequency deviations are smaller than float16
#              can hold. Convert to float32 before any arithmetic.
#
# Alignment:   Each phase angle is measured relative to the meter's own
#              capture time. The stored angle is shifted to the slot time:
#                  ang_slot = ang - 360 * (k + 1) * freq * toff     (degrees)
#              where k is the harmonic index (0 fundamental at 60 Hz, 1 at
#              120 Hz, 2 at 180 Hz, as Caltech's fft_harmonics defines them),
#              freq is the meter's voltage frequency (the true grid
#              frequency), and toff is the capture offset from preprocessing.
#              Angles are wrapped to (-180, 180]. Models should convert to
#              cos and sin.
#
# Frequency and angle_ok:
#              Every channel carries its own frequency estimate, taken by
#              Caltech's code from the strongest peak in the waveform's
#              spectrum. On a current channel carrying only a few amps that
#              peak can land a bin or two away from 60 Hz (verified
#              2026-10-21 on September 2024: whole-hertz disagreements on
#              egauge_6, 16, 18, 20 current channels, never on a voltage
#              channel). The angle reported there is the phase of the wrong
#              component and cannot be repaired by shifting, so:
#                - freq is taken from a present voltage channel only
#                - angle_ok is True only when the reading exists, the angle,
#                  the offset, and both frequencies are finite, and the
#                  channel frequency agrees with the meter frequency within
#                  the tolerance (default 0.01 Hz, meaning "same bin")
#                - angles with angle_ok False are stored as NaN; rms and
#                  mask are untouched
#              angle_ok means "passed these checks", not "certified accurate".
#
# Functions:
#   Reading the processed tables
#     meter_order(meters_file, exclude)          meters in fixed order, excluded ones removed
#     load_meter_month(processed_dir, month, meter, keep_harmonics)
#                                                one meter-month Parquet, needed columns only
#     load_month_tables(processed_dir, month, meters, keep_harmonics)
#                                                every meter-month Parquet, None when missing
#   Building arrays
#     wrap_degrees(angle)                        wrap to (-180, 180]
#     aligned_angle(ang, freq, toff, harmonic)   shift an angle to the slot time
#     meter_frequency(table, meter)              one frequency per slot from the voltage channels
#     build_month_from_tables(tables, month, meters, keep_harmonics, tolerance)
#                                                arrays from already-loaded tables
#     build_month(processed_dir, month, meters, keep_harmonics, tolerance)
#                                                all arrays for one month, plus a stats dict
#   Writing and checking
#     npz_is_complete(path, keep_harmonics)      True when a written .npz is readable and complete
#     save_month(out_dir, month, arrays)         write <month>.npz (ZIP DEFLATE level 1), atomic replace
#     write_manifest(out_dir, meters, keep_harmonics, months, version)
#                                                write manifest.json
#     verify_month(path, processed_dir, month, meters, n_samples, seed, arrays, tables)
#                                                compare stored values against the Parquet
#     export_month(...)                          build, save, verify one month; timing and skip/resume
# =============================================================================

# Standard library: JSON for the manifest
import json
# Standard library: file and folder operations
import os
# Standard library: unique temp files for atomic writes
import tempfile
# Standard library: timing one month's stages
import time
# Standard library: ZIP DEFLATE for .npz
import zipfile
# Third party: numerics
import numpy as np
# Third party: npy format writer used inside .npz
from numpy.lib.format import write_array
# Third party: tables and Parquet reading
import pandas as pd
# Third party: Parquet schema reading, to pick columns before loading
import pyarrow.parquet as pq
# Project: meter list reader and date helpers from preprocessing
import socal_preprocess as spp

# Separator used in processed column names
SEP = "|"
# Name of the per-meter capture-offset column
TOFF = "toff"
# Channel order in the arrays
CHANNELS = ("L1", "L2", "L3", "S1", "S2", "S3")
# Voltage channels, used to pick the frequency
VOLTAGE_CHANNELS = ("L1", "L2", "L3")
# Processed column short names for magnitude and angle per harmonic index
HARMONIC_COLUMNS = {0: ("rms", "ang0"), 1: ("mag1", "ang1"), 2: ("mag2", "ang2")}
# Format version written into the manifest, bump when the layout changes
VERSION = "1"
# ZIP DEFLATE level for .npz; 1 is much faster than numpy's default of 6
# and still np.load-compatible
NPZ_COMPRESSLEVEL = 1
# Arrays every written month file must contain
REQUIRED_NPZ = ("t", "rms", "ang", "freq", "mask", "angle_ok")
# Extra arrays when harmonics were requested
HARMONIC_NPZ = ("mag1", "ang1", "mag2", "ang2")


# -----------------------------------------------------------------------------
# Reading the processed tables
# -----------------------------------------------------------------------------

def meter_order(meters_file, exclude):
    """
    Return the meters in the fixed order used for the arrays: the order of
    meters_file with the excluded names removed.
    """
    # All names in file order
    meters = spp.read_meters(meters_file)
    # Drop excluded ones, keep order
    return [m for m in meters if m not in set(exclude)]


def load_meter_month(processed_dir, month, meter, keep_harmonics):
    """
    Read one meter-month Parquet with only the columns the export needs:
    rms, ang0, freq, toff, masks, and the harmonic columns if asked.
    Returns None when the file does not exist.
    """
    # File path
    path = os.path.join(processed_dir, month, f"{meter}.parquet")
    # Missing file
    if not os.path.exists(path):
        # Nothing
        return None
    # Column names on disk
    names = pq.read_schema(path).names
    # Value short names to read
    values = ["rms", "ang0", "freq"]
    # Harmonics if asked
    if keep_harmonics:
        # Add magnitude and angle of harmonics 1 and 2
        values += ["mag1", "ang1", "mag2", "ang2"]
    # Value columns present
    wanted = [c for c in names if c.startswith(f"{meter}{SEP}") and c.split(SEP)[-1] in values]
    # Mask columns
    wanted += [c for c in names if c.startswith(f"mask{SEP}{meter}{SEP}")]
    # Offset column
    wanted += [c for c in names if c == f"{meter}{SEP}{TOFF}"]
    # pandas restores the datetime index even when columns= is set
    return pd.read_parquet(path, columns=wanted)


def load_month_tables(processed_dir, month, meters, keep_harmonics):
    """Read every meter-month Parquet, None when a meter has no file."""
    return {meter: load_meter_month(processed_dir, month, meter, keep_harmonics) for meter in meters}


# -----------------------------------------------------------------------------
# Building arrays
# -----------------------------------------------------------------------------

def wrap_degrees(angle):
    """Wrap angles in degrees to the range (-180, 180]. NaN stays NaN."""
    # Shift, take modulo 360, shift back
    return -((-angle + 180.0) % 360.0) + 180.0


def aligned_angle(ang, freq, toff, harmonic):
    """
    Shift an angle measured at the capture time to the slot time.

    ang:      angle at capture, degrees
    freq:     fundamental frequency, hertz
    toff:     capture time minus slot time, seconds
    harmonic: 0 for the fundamental, 1 for 120 Hz, 2 for 180 Hz
    Returns degrees wrapped to (-180, 180].
    """
    # The harmonic's own frequency is (k + 1) times the fundamental
    shift = 360.0 * (harmonic + 1) * freq * toff
    # Shift and wrap
    return wrap_degrees(ang - shift)


def _column_to_numpy(table, col, dtype, length, fill=np.nan):
    """
    Convert one table column to a numpy array. Missing columns become
    `fill`. fillna is used only when the series actually has NA.
    """
    # Column absent
    if col not in table.columns:
        # Default
        return np.full(length, fill, dtype=dtype)
    # Series
    series = table[col]
    # Boolean masks and similar need NA filled; value columns keep NaN
    if series.hasnans and not (isinstance(fill, float) and np.isnan(fill)):
        # Fill then convert
        series = series.fillna(fill)
    # copy=False is a no-op when a dtype change is required
    return series.to_numpy(dtype=dtype, copy=False)


def meter_frequency(table, meter):
    """
    One frequency per slot for a meter: the first present voltage
    channel's value (L1, then L2, then L3). Current channels are never
    used, since their estimate is unreliable at small currents.
    Slots with no voltage frequency are NaN. Returns a float64 array.
    """
    # Voltage frequency columns that exist, in order
    cols = [f"{meter}{SEP}{c}{SEP}freq" for c in VOLTAGE_CHANNELS if f"{meter}{SEP}{c}{SEP}freq" in table.columns]
    # Length
    n = len(table)
    # No voltage frequency at all
    if not cols:
        # All NaN
        return np.full(n, np.nan, dtype=np.float64)
    # Empty table
    if n == 0:
        return np.full(0, np.nan, dtype=np.float64)
    # Side-by-side array, same order as L1, L2, L3
    stacked = np.column_stack([table[c].to_numpy(dtype=np.float64, copy=False) for c in cols])
    # Finite flags
    finite = np.isfinite(stacked)
    # First present channel per slot (0 when none are present)
    first = finite.argmax(axis=1)
    # Gather
    out = stacked[np.arange(n, dtype=np.intp), first]
    # Slots with no voltage frequency stay NaN
    out[~finite.any(axis=1)] = np.nan
    # First present value per slot
    return out


def build_month_from_tables(tables, month, meters, keep_harmonics, tolerance):
    """
    Build the arrays for one month from already-loaded tables. Aligns
    tables to the shared grid in place when needed so a later spot-check
    can reuse them. Returns (arrays, stats), or (None, stats) when no
    meter has a file.
    """
    # The grid comes from any meter that exists, all share the same month grid
    present = [t for t in tables.values() if t is not None]
    # Nothing this month
    if not present:
        # Empty
        return None, {"month": month, "meters_with_files": 0}
    # Slot index
    grid = present[0].index
    # Sizes
    T, M, C = len(grid), len(meters), len(CHANNELS)
    # Harmonic indices to export
    harmonics = [0, 1, 2] if keep_harmonics else [0]
    # Magnitude arrays per harmonic, float32 while building
    mags = {k: np.full((T, M, C), np.nan, dtype=np.float32) for k in harmonics}
    # Angle arrays per harmonic
    angs = {k: np.full((T, M, C), np.nan, dtype=np.float32) for k in harmonics}
    # Frequency per meter
    freq = np.full((T, M), np.nan, dtype=np.float32)
    # Mask per meter and channel
    mask = np.zeros((T, M, C), dtype=bool)
    # Angle validity per meter and channel
    angle_ok = np.zeros((T, M, C), dtype=bool)
    # Stats
    stats = {"month": month, "meters_with_files": len(present), "freq_disagreements": {}, "present_fraction": {}, "angle_ok_fraction": {}}
    # Loop over meters
    for mi, meter in enumerate(meters):
        # Table for this meter
        table = tables[meter]
        # Missing meter stays NaN and False
        if table is None:
            # Next
            continue
        # Align to the shared grid only when the index actually differs
        if not table.index.equals(grid):
            # Copy onto the shared slots
            table = table.reindex(grid)
            # Keep the aligned table for a later spot-check
            tables[meter] = table
        # Meter frequency from the voltage channels, float64 for the arithmetic
        meter_freq = meter_frequency(table, meter)
        # Store as float32
        freq[:, mi] = meter_freq.astype(np.float32)
        # Disagreement counter for this meter
        disagreements = 0
        # Capture offset
        toff_col = f"{meter}{SEP}{TOFF}"
        # Offset values, NaN if the column is missing
        toff = _column_to_numpy(table, toff_col, np.float64, T)
        # Loop over channels
        for ci, channel in enumerate(CHANNELS):
            # Mask column
            mask_col = f"mask{SEP}{meter}{SEP}{channel}"
            # Present flags, False if the channel does not exist
            mask[:, mi, ci] = _column_to_numpy(table, mask_col, bool, T, fill=False)
            # This channel's own frequency column
            freq_col = f"{meter}{SEP}{channel}{SEP}freq"
            # Channel frequency, NaN if absent
            channel_freq = _column_to_numpy(table, freq_col, np.float64, T)
            # Agreement with the meter frequency, False where either is NaN
            agree = np.abs(channel_freq - meter_freq) <= tolerance
            # Count disagreements among present readings with both frequencies
            disagreements += int((mask[:, mi, ci] & np.isfinite(channel_freq) & np.isfinite(meter_freq) & ~agree).sum())
            # Loop over harmonics
            for k in harmonics:
                # Column short names
                mag_name, ang_name = HARMONIC_COLUMNS[k]
                # Full column names
                mag_col = f"{meter}{SEP}{channel}{SEP}{mag_name}"
                # Angle
                ang_col = f"{meter}{SEP}{channel}{SEP}{ang_name}"
                # Skip channels without these columns
                if mag_col not in table.columns or ang_col not in table.columns:
                    # Next
                    continue
                # Magnitude
                mags[k][:, mi, ci] = _column_to_numpy(table, mag_col, np.float32, T)
                # Angle at capture, float64
                ang_capture = _column_to_numpy(table, ang_col, np.float64, T)
                # Validity: reading present, angle, offset, both frequencies finite, frequencies agree
                ok = mask[:, mi, ci] & np.isfinite(ang_capture) & np.isfinite(toff) & np.isfinite(meter_freq) & agree
                # The fundamental decides angle_ok for the channel
                if k == 0:
                    # Store
                    angle_ok[:, mi, ci] = ok
                # Angle shifted to the slot time with the meter frequency, computed in float64
                shifted = aligned_angle(ang_capture, meter_freq, toff, k)
                # Rejected angles become NaN
                angs[k][:, mi, ci] = np.where(ok, shifted, np.nan).astype(np.float32)
        # Disagreement count for the stats
        stats["freq_disagreements"][meter] = disagreements
        # Presence fraction for the stats
        stats["present_fraction"][meter] = round(float(mask[:, mi, :].mean()), 4)
        # Share of present readings whose angle passed, for the stats
        stats["angle_ok_fraction"][meter] = round(float(angle_ok[:, mi, :].sum() / max(int(mask[:, mi, :].sum()), 1)), 4)
    # Slot times as whole seconds since 1970 UTC, independent of the index's internal unit
    t = ((grid.tz_convert("UTC") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    # Output arrays, fundamental under the plain names
    arrays = {"t": t, "rms": mags[0].astype(np.float16), "ang": angs[0], "freq": freq, "mask": mask, "angle_ok": angle_ok}
    # Harmonics under numbered names
    for k in harmonics[1:]:
        # Magnitude
        arrays[f"mag{k}"] = mags[k].astype(np.float16)
        # Angle
        arrays[f"ang{k}"] = angs[k]
    # Return
    return arrays, stats


def build_month(processed_dir, month, meters, keep_harmonics, tolerance):
    """
    Build the arrays for one month. Returns (arrays, stats) where arrays
    is a dictionary ready for save_month and stats holds per-meter
    presence fractions and frequency disagreement counts. Returns
    (None, stats) when no meter has a file for the month.
    """
    # Tables per meter, None when missing
    tables = load_month_tables(processed_dir, month, meters, keep_harmonics)
    # Build from the loaded tables
    return build_month_from_tables(tables, month, meters, keep_harmonics, tolerance)


# -----------------------------------------------------------------------------
# Writing and checking
# -----------------------------------------------------------------------------

def npz_is_complete(path, keep_harmonics=False):
    """
    True when path is a readable .npz with the arrays this export writes.
    Uses the ZIP directory only, so a truncated Drive copy fails without
    decompressing. Empty or missing files are False.
    """
    # Missing or empty
    if not os.path.exists(path) or os.path.getsize(path) < 22:
        # Too small to be a zip
        return False
    # Names that must be present
    needed = set(REQUIRED_NPZ)
    # Harmonics if this run asked for them
    if keep_harmonics:
        # Add the four extra arrays
        needed.update(HARMONIC_NPZ)
    try:
        # Central directory sits at the end, so a truncated write fails here
        with zipfile.ZipFile(path, "r") as zf:
            # Member names without the .npy suffix numpy uses
            names = {os.path.splitext(name)[0] for name in zf.namelist()}
        # All required arrays present
        return needed.issubset(names)
    except (zipfile.BadZipFile, OSError, ValueError):
        # Unreadable
        return False


def save_month(out_dir, month, arrays):
    """
    Write the month's arrays to <out_dir>/<month>.npz using ZIP DEFLATE
    level 1 (np.load-compatible). Writes to a unique temp file and replaces
    atomically. Returns (path, bytes).
    """
    # Make sure the folder exists
    os.makedirs(out_dir, exist_ok=True)
    # Path
    path = os.path.join(out_dir, f"{month}.npz")
    # Unique sibling so two processes cannot share a temp name
    fd, tmp_path = tempfile.mkstemp(prefix=f".{month}.", suffix=".npz.tmp", dir=out_dir)
    # Close the fd; ZipFile opens the path itself
    os.close(fd)
    try:
        # ZIP container numpy itself uses for .npz
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=NPZ_COMPRESSLEVEL, allowZip64=True) as zf:
            # One member per array
            for key, val in arrays.items():
                # Same member names as np.savez
                with zf.open(key + ".npy", "w", force_zip64=True) as fid:
                    # Raw npy bytes, no pickles
                    write_array(fid, np.asanyarray(val), allow_pickle=False)
        # Flush to disk before publishing
        with open(tmp_path, "rb") as flush:
            # File contents
            os.fsync(flush.fileno())
        # Atomic publish
        os.replace(tmp_path, path)
    except BaseException:
        # Do not leave a partial destination; temp is dropped too
        if os.path.exists(tmp_path):
            # Best-effort cleanup
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        # Re-raise
        raise
    # Return path and size
    return path, os.path.getsize(path)


def write_manifest(out_dir, meters, keep_harmonics, months, version=VERSION):
    """Write manifest.json describing the layout, so readers never guess."""
    # Array descriptions
    arrays = {
        # Time
        "t": {"dtype": "int64", "shape": "(T,)", "unit": "seconds since 1970-01-01 UTC, slot start"},
        # Magnitude
        "rms": {"dtype": "float16", "shape": "(T, M, C)", "unit": "volts for L channels, amps for S channels"},
        # Angle
        "ang": {"dtype": "float32", "shape": "(T, M, C)", "unit": "degrees in (-180, 180], fundamental, at the slot time"},
        # Frequency
        "freq": {"dtype": "float32", "shape": "(T, M)", "unit": "hertz, first available voltage channel"},
        # Mask
        "mask": {"dtype": "bool", "shape": "(T, M, C)", "unit": "True where a reading exists"},
        # Angle validity
        "angle_ok": {"dtype": "bool", "shape": "(T, M, C)", "unit": "True where the fundamental angle passed the checks: reading present, angle and offset and frequencies finite, channel frequency within tolerance of the meter's voltage frequency. Angles with False are NaN."},
    }
    # Harmonics if exported
    if keep_harmonics:
        # Loop over 1 and 2
        for k in (1, 2):
            # Magnitude
            arrays[f"mag{k}"] = {"dtype": "float16", "shape": "(T, M, C)", "unit": f"harmonic {k} magnitude, {60 * (k + 1)} Hz"}
            # Angle
            arrays[f"ang{k}"] = {"dtype": "float32", "shape": "(T, M, C)", "unit": f"harmonic {k} angle, degrees, at the slot time"}
    # Manifest body
    manifest = {
        # Version of this layout
        "version": version,
        # Meter order, index into M
        "meters": list(meters),
        # Channel order, index into C
        "channels": list(CHANNELS),
        # Arrays
        "arrays": arrays,
        # How angles were shifted
        "alignment": "ang_slot = ang_capture - 360 * (k + 1) * freq * toff, k = harmonic index, freq = the meter's voltage frequency, toff from preprocessing; angles that failed the angle_ok checks are NaN",
        # Why some angles are missing
        "angle_ok_note": "Caltech's frequency estimate takes the strongest spectral peak; on current channels carrying a few amps it can land a bin away from 60 Hz, and the angle there is the phase of the wrong component. angle_ok means passed the checks, not certified accurate.",
        # Months written
        "months": list(months),
        # Where the data came from
        "source": "data/processed/<YYYY-MM>/<meter>.parquet from main_preprocess.py",
    }
    # Path
    path = os.path.join(out_dir, "manifest.json")
    # Write
    with open(path, "w") as f:
        # Pretty JSON
        json.dump(manifest, f, indent=2)
    # Return
    return path


def _load_npz_arrays(path):
    """Load every array in an .npz once, so members are not decompressed per access."""
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def verify_month(path, processed_dir, month, meters, n_samples, seed, arrays=None, tables=None):
    """
    Spot-check a written month against the Parquet it came from. Draws
    n_samples random (slot, meter, channel) cells where a reading exists
    and reports the largest magnitude error (relative), the largest
    angle error (degrees, circular, over cells with angle_ok True), the
    largest frequency error against the meter's L1, and how many sampled
    cells had angle_ok False.

    Expected values are always rebuilt from the Parquet columns (rms,
    capture angle, toff, meter frequency), never copied from the export
    arrays. Stored values come from `arrays` when given, otherwise from
    the written .npz. Passing tables skips Parquet rereads but does not
    skip that independent reconstruction.
    Meter frequency is computed once per sampled meter.
    """
    # Stored arrays, from memory or from disk
    data = arrays if arrays is not None else _load_npz_arrays(path)
    # Random generator, same seed as the original sequential run
    rng = np.random.default_rng(seed)
    # Cells with a reading
    present = np.argwhere(data["mask"])
    # Nothing to check
    if len(present) == 0:
        # Empty report
        return {"checked": 0}
    # Sample rows
    picks = present[rng.choice(len(present), size=min(n_samples, len(present)), replace=False)]
    # Worst errors
    worst = {"checked": int(len(picks)), "rms_rel_err": 0.0, "ang_err_deg": 0.0, "freq_err_hz": 0.0, "angle_rejected": 0}
    # Stored arrays pulled out once
    rms_arr, ang_arr, freq_arr, ok_arr = data["rms"], data["ang"], data["freq"], data["angle_ok"]
    # Cache of loaded meter tables
    cache = {} if tables is None else tables
    # Meter frequency per sampled meter
    freq_cache = {}
    # Loop over samples
    for ti, mi, ci in picks:
        # Meter and channel names
        meter, channel = meters[mi], CHANNELS[ci]
        # Load the meter table once
        if meter not in cache or cache[meter] is None:
            # Read
            cache[meter] = load_meter_month(processed_dir, month, meter, False)
        # Table
        table = cache[meter]
        # Row by slot position
        row = table.iloc[ti]
        # Original magnitude in float64
        rms0 = float(row[f"{meter}{SEP}{channel}{SEP}rms"])
        # Original angle at capture
        ang0 = float(row[f"{meter}{SEP}{channel}{SEP}ang0"])
        # Meter voltage frequency, computed once for this meter
        if meter not in freq_cache:
            # Full series
            freq_cache[meter] = meter_frequency(table, meter)
        # This slot
        freq_meter = float(freq_cache[meter][ti])
        # Offset
        toff0 = float(row[f"{meter}{SEP}{TOFF}"])
        # Stored values
        rms1 = float(rms_arr[ti, mi, ci])
        # Angle
        ang1 = float(ang_arr[ti, mi, ci])
        # Frequency
        freq1 = float(freq_arr[ti, mi])
        # Relative magnitude error, guarded against zero
        worst["rms_rel_err"] = max(worst["rms_rel_err"], abs(rms1 - rms0) / max(abs(rms0), 1e-6))
        # Angle check only where the export accepted the angle
        if ok_arr[ti, mi, ci]:
            # Expected aligned angle with the meter frequency
            expected_ang = aligned_angle(ang0, freq_meter, toff0, 0)
            # Circular angle error
            worst["ang_err_deg"] = max(worst["ang_err_deg"], abs(wrap_degrees(ang1 - expected_ang)))
        # Rejected angles counted
        else:
            # Count
            worst["angle_rejected"] += 1
        # Frequency error against the meter frequency, when both finite
        if np.isfinite(freq1) and np.isfinite(freq_meter):
            # Update
            worst["freq_err_hz"] = max(worst["freq_err_hz"], abs(freq1 - freq_meter))
    # Return
    return worst


def export_month(processed_dir, out_dir, month, meters, keep_harmonics, tolerance, verify_samples, seed, force=False):
    """
    Build, save, and optionally verify one month. Safe to call from a
    worker process. Returns a dict with status, month, and when written
    a log row including stage timings.
    """
    # Destination
    path = os.path.join(out_dir, f"{month}.npz")
    # Resume: a readable complete file is left alone unless forced
    if not force and npz_is_complete(path, keep_harmonics):
        # Same skip the sequential loop used
        return {"status": "exists", "month": month}
    # Load once, reuse for the spot-check
    tables = load_month_tables(processed_dir, month, meters, keep_harmonics)
    # Build
    t0 = time.perf_counter()
    arrays, stats = build_month_from_tables(tables, month, meters, keep_harmonics, tolerance)
    build_s = time.perf_counter() - t0
    # Nothing this month
    if arrays is None:
        # Same skip as the sequential loop
        return {"status": "no_data", "month": month}
    # Save
    t0 = time.perf_counter()
    path, size = save_month(out_dir, month, arrays)
    save_s = time.perf_counter() - t0
    # Spot-check against the in-memory arrays and tables
    t0 = time.perf_counter()
    if verify_samples > 0:
        # Stored values from the written file; expected values from Parquet tables
        check = verify_month(path, processed_dir, month, meters, verify_samples, seed, tables=tables)
    # Skipped
    else:
        # Empty report
        check = {"checked": 0}
    verify_s = time.perf_counter() - t0
    # Log row
    row = {
        # Month
        "month": month,
        # Slots in the month
        "slots": int(len(arrays["t"])),
        # Meters that had a file
        "meters_with_files": stats["meters_with_files"],
        # File size in megabytes
        "size_mb": round(size / 1e6, 1),
        # Spot-check numbers
        "checked": check.get("checked", 0),
        # Worst magnitude error
        "rms_rel_err": check.get("rms_rel_err", ""),
        # Worst angle error
        "ang_err_deg": check.get("ang_err_deg", ""),
        # Worst frequency error
        "freq_err_hz": check.get("freq_err_hz", ""),
        # Frequency disagreements summed over meters
        "freq_disagreements": sum(stats.get("freq_disagreements", {}).values()),
        # Share of present readings whose angle passed, over all meters
        "angle_ok_fraction": round(float(arrays["angle_ok"].sum() / max(int(arrays["mask"].sum()), 1)), 4),
        # Sampled cells whose angle was rejected
        "angle_rejected_in_check": check.get("angle_rejected", 0),
        # Stage timings in seconds
        "build_s": round(build_s, 3),
        "save_s": round(save_s, 3),
        "verify_s": round(verify_s, 3),
    }
    # Hand back
    return {"status": "wrote", "month": month, "row": row}
