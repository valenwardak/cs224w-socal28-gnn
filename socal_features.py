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
#   Building arrays
#     wrap_degrees(angle)                        wrap to (-180, 180]
#     aligned_angle(ang, freq, toff, harmonic)   shift an angle to the slot time
#     meter_frequency(table, meter)              one frequency per slot from the voltage channels
#     build_month(processed_dir, month, meters, keep_harmonics, tolerance)
#                                                all arrays for one month, plus a stats dict
#   Writing and checking
#     save_month(out_dir, month, arrays)         write <month>.npz, return path and size
#     write_manifest(out_dir, meters, keep_harmonics, months, version)
#                                                write manifest.json
#     verify_month(path, processed_dir, month, meters, n_samples, seed)
#                                                compare stored values against the Parquet
# =============================================================================

# Standard library: JSON for the manifest
import json
# Standard library: file and folder operations
import os
# Third party: numerics
import numpy as np
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
    # Read
    return pd.read_parquet(path, columns=wanted)


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


def meter_frequency(table, meter):
    """
    One frequency per slot for a meter: the first present voltage
    channel's value (L1, then L2, then L3). Current channels are never
    used, since their estimate is unreliable at small currents.
    Slots with no voltage frequency are NaN. Returns a float64 array.
    """
    # Voltage frequency columns that exist, in order
    cols = [f"{meter}{SEP}{c}{SEP}freq" for c in VOLTAGE_CHANNELS if f"{meter}{SEP}{c}{SEP}freq" in table.columns]
    # No voltage frequency at all
    if not cols:
        # All NaN
        return np.full(len(table), np.nan, dtype=np.float64)
    # Side-by-side frame
    frame = table[cols].astype("float64")
    # First present value per slot
    return frame.bfill(axis=1).iloc[:, 0].to_numpy(dtype=np.float64)


def build_month(processed_dir, month, meters, keep_harmonics, tolerance):
    """
    Build the arrays for one month. Returns (arrays, stats) where arrays
    is a dictionary ready for save_month and stats holds per-meter
    presence fractions and frequency disagreement counts. Returns
    (None, stats) when no meter has a file for the month.
    """
    # Tables per meter, None when missing
    tables = {m: load_meter_month(processed_dir, month, m, keep_harmonics) for m in meters}
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
        # Align to the shared grid in case of any difference
        table = table.reindex(grid)
        # Meter frequency from the voltage channels, float64 for the arithmetic
        meter_freq = meter_frequency(table, meter)
        # Store as float32
        freq[:, mi] = meter_freq.astype(np.float32)
        # Disagreement counter for this meter
        disagreements = 0
        # Capture offset
        toff_col = f"{meter}{SEP}{TOFF}"
        # Offset values, NaN if the column is missing
        toff = table[toff_col].to_numpy(dtype=np.float64) if toff_col in table.columns else np.full(T, np.nan)
        # Loop over channels
        for ci, channel in enumerate(CHANNELS):
            # Mask column
            mask_col = f"mask{SEP}{meter}{SEP}{channel}"
            # Present flags, False if the channel does not exist
            mask[:, mi, ci] = table[mask_col].fillna(False).to_numpy(dtype=bool) if mask_col in table.columns else False
            # This channel's own frequency column
            freq_col = f"{meter}{SEP}{channel}{SEP}freq"
            # Channel frequency, NaN if absent
            channel_freq = table[freq_col].to_numpy(dtype=np.float64) if freq_col in table.columns else np.full(T, np.nan)
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
                mags[k][:, mi, ci] = table[mag_col].to_numpy(dtype=np.float32)
                # Angle at capture, float64
                ang_capture = table[ang_col].to_numpy(dtype=np.float64)
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


# -----------------------------------------------------------------------------
# Writing and checking
# -----------------------------------------------------------------------------

def save_month(out_dir, month, arrays):
    """Write the month's arrays to <out_dir>/<month>.npz, compressed. Returns (path, bytes)."""
    # Make sure the folder exists
    os.makedirs(out_dir, exist_ok=True)
    # Path
    path = os.path.join(out_dir, f"{month}.npz")
    # Write compressed
    np.savez_compressed(path, **arrays)
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


def verify_month(path, processed_dir, month, meters, n_samples, seed):
    """
    Spot-check a written month against the Parquet it came from. Draws
    n_samples random (slot, meter, channel) cells where a reading exists
    and reports the largest magnitude error (relative), the largest
    angle error (degrees, circular, over cells with angle_ok True), the
    largest frequency error against the meter's L1, and how many sampled
    cells had angle_ok False.
    """
    # Load the file
    data = np.load(path)
    # Random generator
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
    # Cache of loaded meter tables
    cache = {}
    # Loop over samples
    for ti, mi, ci in picks:
        # Meter and channel names
        meter, channel = meters[mi], CHANNELS[ci]
        # Load the meter table once
        if meter not in cache:
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
        # Meter voltage frequency for this slot
        freq_meter = float(meter_frequency(table, meter)[ti])
        # Offset
        toff0 = float(row[f"{meter}{SEP}{TOFF}"])
        # Stored values
        rms1 = float(data["rms"][ti, mi, ci])
        # Angle
        ang1 = float(data["ang"][ti, mi, ci])
        # Frequency
        freq1 = float(data["freq"][ti, mi])
        # Relative magnitude error, guarded against zero
        worst["rms_rel_err"] = max(worst["rms_rel_err"], abs(rms1 - rms0) / max(abs(rms0), 1e-6))
        # Angle check only where the export accepted the angle
        if data["angle_ok"][ti, mi, ci]:
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
