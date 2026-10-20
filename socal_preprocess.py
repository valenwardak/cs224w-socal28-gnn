# =============================================================================
# File:        socal_preprocess.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-18
#
# Purpose:     Helper functions that turn the raw SoCal 28-Bus download
#              into aligned, masked, model-ready tables. Imported by
#              main_preprocess.py. Also imported later by the model code,
#              so the same functions define the data everywhere.
#
# Functions:
#   Reading the raw download
#     read_meters(meters_file)            meter names from a text file
#     parse_date(text)                    "YYYY-MM-DD" -> date
#     month_chunks(start, end)            yield one (start, end) pair per month
#     parse_folder_label(label)           request folder name -> (start, end, channel)
#     find_channel_files(...)             CSV paths for one meter and channel in a range
#     load_channel(paths, value_columns)  read and stack the CSVs, dedupe timestamps
#   Time handling and gridding
#     to_utc(naive_times, offset)         fixed standard time -> UTC
#     month_grid(start, end, offset)      ten-second UTC grid for a month
#     grid_channel(df, grid, offset)      place readings on the grid, add toff
#     build_meter_month(...)              one meter-month table: values, masks, toff
#   Network file
#     load_network(topology_dir)          read the network JSON
#     meter_metadata(network)             meter -> bus, elements, phase map, has_L3
#   Events
#     build_events(...)                   device changes from the operator log
#   Coverage
#     coverage_rows(table, meter, month)  present-reading counts per channel
#     append_coverage(out_dir, rows)      append rows to coverage.csv
#
# What it produces, under <out_dir>:
#              <YYYY-MM>/<meter>.parquet   one file per meter and month:
#                  index  = ten-second UTC grid covering the whole month
#                  columns "<meter>|<channel>|<value>"  float32 readings
#                  columns "mask|<meter>|<channel>"     True where a reading exists
#                  column  "<meter>|toff"               float32 seconds between the
#                                                       exact capture time and the
#                                                       slot time (capture minus slot)
#              meters.json      meter -> voltage bus, current elements, phase map
#              events.csv       every logged device change, UTC and standard time,
#                               with episode id and scored flag
#              coverage.csv     per month, meter, channel: slots, present, fraction
#
# Why toff exists:
#              Each phase angle is measured relative to the exact moment the
#              meter's own capture started. Captures on different meters start
#              a few milliseconds apart, and at 60 Hz one millisecond is 21.6
#              degrees, so angles are only comparable across meters after each
#              is shifted to a common time. The shift to the slot time is
#                  angle_at_slot = angle - 360 * freq * toff   (degrees)
#              All channels of one meter share one capture time, so toff is
#              stored once per meter.
#
# Facts this code relies on (verified 2026-10-18):
#              Meter timestamps and the operator log are both fixed Pacific
#              Standard Time all year, no daylight saving. UTC = time + 8 h.
#              All meter clocks agree to within one ten-second reading.
#              Channels L1 L2 L3 are voltages, S1 S2 S3 currents.
#              egauge_17 and egauge_18 have no L3.
#              Phase labels come from the network file, not from the name.
#              egauge_1 and egauge_2 are not in the network file and are
#              excluded. February 1, 2024 is a records update, not scored.
#
# Raw data layout read:
#              <data_dir>/<meter>/<start>_<end>[_<channel>]/phasors/<meter>-<channel>.csv
#              Columns: t, t_zero_crossing, frequency, rms,
#                       phase_angle_harmonic_k, magnitude_harmonic_k  (k = 0, 1, 2)
#              Empty files are skipped. Overlapping fallback folders are deduplicated.
# =============================================================================

# Standard library: CSV writing
import csv
# Standard library: JSON for metadata
import json
# Standard library: file and folder operations
import os
# Standard library: dates and times
from datetime import date, datetime, timedelta
# Third party: numerics
import numpy as np
# Third party: tables, CSV reading, Parquet writing
import pandas as pd

# Raw CSV column names and the short names used in the output
VALUE_COLUMNS = {
    # Root-mean-square level of the channel
    "rms": "rms",
    # Measured frequency in hertz
    "frequency": "freq",
    # Fundamental (60 Hz) angle and magnitude
    "phase_angle_harmonic_0": "ang0",
    "magnitude_harmonic_0": "mag0",
    # Second harmonic angle and magnitude
    "phase_angle_harmonic_1": "ang1",
    "magnitude_harmonic_1": "mag1",
    # Third harmonic angle and magnitude
    "phase_angle_harmonic_2": "ang2",
    "magnitude_harmonic_2": "mag2",
}
# Grid spacing
GRID_SECONDS = 10
# Separator used in output column names
SEP = "|"
# Name of the per-meter capture-offset column
TOFF = "toff"


# -----------------------------------------------------------------------------
# Reading the raw download
# -----------------------------------------------------------------------------

def read_meters(meters_file):
    """
    Read meter names from a text file, one per line. Blank lines and
    lines starting with # are ignored.
    """
    # Fail with a clear message if the file is missing
    if not os.path.exists(meters_file):
        # Raise so the caller sees the path tried
        raise FileNotFoundError(f"Meters file not found: {meters_file}")
    # Read all lines
    with open(meters_file) as f:
        # Strip whitespace
        lines = [line.strip() for line in f]
    # Keep real entries
    return [line for line in lines if line and not line.startswith("#")]


def parse_date(text):
    """Turn a YYYY-MM-DD string into a date object."""
    # Parse and drop the time part
    return datetime.strptime(text, "%Y-%m-%d").date()


def month_chunks(start_date, end_date):
    """
    Yield (chunk_start, chunk_end) date pairs, one per calendar month,
    both ends inclusive, clipped to start_date and end_date.
    """
    # Begin at the start date
    current = start_date
    # Loop until past the end
    while current <= end_date:
        # First day of the following month
        if current.month == 12:
            # January next year
            next_month = date(current.year + 1, 1, 1)
        # Any other month
        else:
            # First day of next month
            next_month = date(current.year, current.month + 1, 1)
        # Last day of this chunk, clipped
        chunk_end = min(next_month - timedelta(days=1), end_date)
        # Hand back
        yield current, chunk_end
        # Advance
        current = next_month


def parse_folder_label(label):
    """
    Split a request folder name into (start_date, end_date, channel).
    Returns None for names that do not fit, such as "_failed".
    """
    # Split on underscores
    parts = label.split("_")
    # Need two date parts
    if len(parts) < 2:
        # Not a request folder
        return None
    # Try to parse both dates
    try:
        # Start
        start = datetime.strptime(parts[0], "%Y-%m-%d").date()
        # End
        end = datetime.strptime(parts[1], "%Y-%m-%d").date()
    # Not dates
    except ValueError:
        # Not a request folder
        return None
    # Optional channel suffix
    channel = parts[2] if len(parts) > 2 else None
    # Return the pieces
    return start, end, channel


def find_channel_files(data_dir, meter, range_start, range_end, channel):
    """
    Return the paths of every CSV for one meter and channel whose request
    folder overlaps the given date range. Fallback folders are included,
    so timestamps may repeat across files; the caller deduplicates.
    """
    # Meter folder
    meter_dir = os.path.join(data_dir, meter)
    # No folder, no files
    if not os.path.isdir(meter_dir):
        # Empty result
        return []
    # Collect matches
    paths = []
    # Loop over request folders
    for label in os.listdir(meter_dir):
        # Parse the name
        parsed = parse_folder_label(label)
        # Skip non-request folders
        if parsed is None:
            # Next
            continue
        # Unpack
        start, end, folder_channel = parsed
        # Skip folders that do not overlap the range
        if end < range_start or start > range_end:
            # Next
            continue
        # Skip single-channel folders for another channel
        if folder_channel is not None and folder_channel != channel:
            # Next
            continue
        # Expected CSV path
        path = os.path.join(meter_dir, label, "phasors", f"{meter}-{channel}.csv")
        # Keep it if present
        if os.path.exists(path):
            # Add
            paths.append(path)
    # Return all matches
    return paths


def load_channel(paths, value_columns):
    """
    Read t plus the requested value columns from every file, skipping
    empty files, and return one DataFrame sorted by t with duplicate
    timestamps removed. Columns missing from a file are filled with NaN.
    """
    # Collect readable frames
    frames = []
    # Loop over files
    for p in paths:
        # Skip zero-byte files
        if os.path.getsize(p) == 0:
            # Next file
            continue
        # Read the header line to see which columns exist
        try:
            # Column names only
            header = pd.read_csv(p, nrows=0).columns.tolist()
        # Header missing entirely
        except pd.errors.EmptyDataError:
            # Next file
            continue
        # Columns we can actually read from this file
        usecols = ["t"] + [c for c in value_columns if c in header]
        # Read them
        df = pd.read_csv(p, usecols=usecols)
        # Add any missing value columns as NaN so all frames line up
        for c in value_columns:
            # Fill if absent
            if c not in df.columns:
                # NaN column
                df[c] = np.nan
        # Keep
        frames.append(df[["t"] + list(value_columns)])
    # Nothing readable
    if not frames:
        # Empty frame with the right columns
        return pd.DataFrame(columns=["t"] + list(value_columns))
    # Stack
    df = pd.concat(frames, ignore_index=True)
    # Parse timestamps, bad rows become NaT
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    # Drop bad rows and duplicate timestamps, sort
    df = df.dropna(subset=["t"]).drop_duplicates(subset=["t"]).sort_values("t")
    # Return with a fresh index
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Time handling and gridding
# -----------------------------------------------------------------------------

def to_utc(naive_times, utc_offset_hours):
    """
    Convert naive fixed-standard-time timestamps to UTC-aware timestamps.

    The meters and the log run on Pacific Standard Time all year, so UTC
    is a fixed addition of utc_offset_hours (8).
    """
    # Add the offset and label as UTC
    return (naive_times + pd.Timedelta(hours=utc_offset_hours)).dt.tz_localize("UTC")


def month_grid(month_start, month_end, utc_offset_hours):
    """
    Build the ten-second UTC grid covering the month from the first
    second of month_start to the last second of month_end, in standard
    time, then expressed in UTC.
    """
    # Start of the first day, standard time
    start_std = pd.Timestamp(datetime.combine(month_start, datetime.min.time()))
    # Start of the day after the last day, standard time
    end_std = pd.Timestamp(datetime.combine(month_end + timedelta(days=1), datetime.min.time()))
    # Shift both to UTC
    start_utc = (start_std + pd.Timedelta(hours=utc_offset_hours)).tz_localize("UTC")
    # End in UTC
    end_utc = (end_std + pd.Timedelta(hours=utc_offset_hours)).tz_localize("UTC")
    # Grid of slots, end excluded
    return pd.date_range(start_utc, end_utc, freq=f"{GRID_SECONDS}s", inclusive="left")


def grid_channel(df, grid, utc_offset_hours):
    """
    Place one channel's readings onto the grid.

    Each reading is assigned to the ten-second slot it falls in. If two
    readings land in one slot (overlapping fallback files), the first is
    kept. Slots with no reading are NaN. Returns a DataFrame indexed by
    the grid with the value columns plus a "toff" column holding the
    seconds between the exact capture time and the slot time.
    """
    # Names of the value columns, everything except t
    value_names = [c for c in df.columns if c != "t"]
    # No readings at all
    if df.empty:
        # All-NaN frame on the grid, with the toff column included
        return pd.DataFrame(np.nan, index=grid, columns=value_names + [TOFF])
    # UTC-aware timestamps
    t_utc = to_utc(df["t"], utc_offset_hours)
    # Slot each reading belongs to
    slot = t_utc.dt.floor(f"{GRID_SECONDS}s")
    # Value columns only
    out = df.drop(columns=["t"]).copy()
    # Seconds from the slot time to the exact capture time
    out[TOFF] = (t_utc - slot).dt.total_seconds().to_numpy()
    # Index by slot, keeping the UTC label
    out.index = pd.DatetimeIndex(slot)
    # Keep the first reading when two share a slot
    out = out[~out.index.duplicated(keep="first")]
    # Align to the month grid, dropping anything outside it
    return out.reindex(grid)


def build_meter_month(data_dir, meter, channels, month_start, month_end, value_columns, utc_offset_hours):
    """
    Build the aligned table for one meter and one month.

    Returns a DataFrame indexed by the UTC grid with float32 value columns
    "<meter>|<channel>|<value>", boolean columns "mask|<meter>|<channel>",
    and one float32 column "<meter>|toff" with the capture offset in
    seconds, taken from whichever channel has a reading in each slot.
    """
    # The grid for this month
    grid = month_grid(month_start, month_end, utc_offset_hours)
    # Pieces to join
    pieces = []
    # Capture offsets, one series per channel
    offsets = []
    # Loop over channels
    for channel in channels:
        # Files overlapping the month
        paths = find_channel_files(data_dir, meter, month_start, month_end, channel)
        # Raw readings
        raw = load_channel(paths, list(value_columns.keys()))
        # On the grid
        gridded = grid_channel(raw, grid, utc_offset_hours)
        # Pull the offset out before renaming the value columns
        offsets.append(gridded.pop(TOFF))
        # Rename to output names
        gridded.columns = [f"{meter}{SEP}{channel}{SEP}{value_columns[c]}" for c in gridded.columns]
        # Store as float32 to halve the size
        gridded = gridded.astype("float32")
        # Mask: a reading exists if the first value column is present
        gridded[f"mask{SEP}{meter}{SEP}{channel}"] = gridded.iloc[:, 0].notna()
        # Keep
        pieces.append(gridded)
    # Join all channels side by side
    table = pd.concat(pieces, axis=1)
    # Offsets side by side, one column per channel
    offset_table = pd.concat(offsets, axis=1)
    # One offset per slot: the first channel that has a reading in that slot
    table[f"{meter}{SEP}{TOFF}"] = offset_table.bfill(axis=1).iloc[:, 0].astype("float32")
    # Name the index
    table.index.name = "t_utc"
    # Return
    return table


# -----------------------------------------------------------------------------
# Network file: meter metadata and phase labels
# -----------------------------------------------------------------------------

def load_network(topology_dir):
    """
    Load the network description JSON (the single file under
    network_files/circuit3) and return it as a dictionary.
    """
    # Folder holding the network file
    folder = os.path.join(topology_dir, "network_files", "circuit3")
    # The JSON files in it
    names = sorted(n for n in os.listdir(folder) if n.endswith(".json"))
    # Need at least one
    if not names:
        # Raise
        raise FileNotFoundError(f"No network JSON in {folder}")
    # Read the first (only) one
    with open(os.path.join(folder, names[0])) as f:
        # Parse
        return json.load(f)


def meter_metadata(network):
    """
    Build {meter: {...}} from the network file's EgaugeMeter entries.

    For each meter: the bus its voltages are measured on, the elements
    its currents are measured on, a phase map {channel: "a"/"b"/"c"}
    taken from the register element suffixes, and whether L3 exists.
    """
    # Result
    meta = {}
    # Loop over meters
    for m in network.get("EgaugeMeter", []):
        # Name
        name = m["name"]
        # Collect
        voltage_bus = set()
        # Current elements
        current_elements = set()
        # Phase map
        phase = {}
        # Loop over registers
        for r in m.get("registers", []):
            # Element the register measures, if any
            element = r.get("element")
            # Skip registers without one
            if not element:
                # Next
                continue
            # Split "bus_1107.ag" into "bus_1107" and "ag"
            base, _, suffix = element.partition(".")
            # Voltage registers are named L1, L2, L3
            if r.get("unit") == "V" and r.get("name") in ("L1", "L2", "L3"):
                # Record the bus
                voltage_bus.add(base)
                # Phase letter is the first character of the suffix, such as "a" in "ag"
                phase[r["name"]] = suffix[:1]
            # Current registers carry value S1, S2, S3 which match the phasor channel names
            if r.get("unit") == "A" and r.get("value") in ("S1", "S2", "S3"):
                # Record the element
                current_elements.add(base)
                # Phase letter is the suffix itself, such as "a"
                phase[r["value"]] = suffix[:1]
        # Store
        meta[name] = {
            # Bus for voltages (normally exactly one)
            "voltage_bus": sorted(voltage_bus),
            # Elements for currents
            "current_elements": sorted(current_elements),
            # Channel to phase letter
            "phase": phase,
            # Whether the third voltage channel exists
            "has_L3": "L3" in phase,
        }
    # Return
    return meta


# -----------------------------------------------------------------------------
# Events from the operator log
# -----------------------------------------------------------------------------

def build_events(topology_dir, utc_offset_hours, episode_gap_minutes, exclude_dates):
    """
    Read every breaker and switch status file and return a DataFrame of
    device changes: t_std (fixed standard time as logged), t_utc, device,
    from_status, to_status, day, episode, scored.

    The first row of each file is the starting state, not a change.
    Changes closer together than episode_gap_minutes share an episode.
    Changes on an excluded date are marked scored = False.
    """
    # Folder of status files
    folder = os.path.join(topology_dir, "parameter_timeseries")
    # Rows
    rows = []
    # Loop over files
    for fn in sorted(os.listdir(folder)):
        # Only CSVs
        if not fn.endswith(".csv"):
            # Next
            continue
        # Device name is the part before "-"
        device = fn.split("-")[0]
        # Read
        df = pd.read_csv(os.path.join(folder, fn))
        # Each row after the first is a change from the previous status
        for prev, cur in zip(df.itertuples(index=False), df.iloc[1:].itertuples(index=False)):
            # Record
            rows.append({"t_std": pd.Timestamp(cur.t), "device": device, "from_status": prev.str, "to_status": cur.str})
    # Table
    ev = pd.DataFrame(rows).sort_values("t_std").reset_index(drop=True)
    # UTC time
    ev["t_utc"] = to_utc(ev["t_std"], utc_offset_hours)
    # Calendar day in standard time
    ev["day"] = ev["t_std"].dt.date.astype(str)
    # Episode id: new episode when the gap to the previous change exceeds the threshold
    gaps = ev["t_std"].diff() > pd.Timedelta(minutes=episode_gap_minutes)
    # Cumulative count of new episodes, starting at 1
    ev["episode"] = gaps.cumsum() + 1
    # Scored unless the day is excluded
    ev["scored"] = ~ev["day"].isin(set(exclude_dates))
    # Column order
    return ev[["t_std", "t_utc", "device", "from_status", "to_status", "day", "episode", "scored"]]


# -----------------------------------------------------------------------------
# Coverage
# -----------------------------------------------------------------------------

def coverage_rows(table, meter, month_label):
    """
    From one meter-month table, return one coverage row per channel:
    month, meter, channel, slots, present, fraction.
    """
    # Rows
    rows = []
    # Mask columns
    mask_cols = [c for c in table.columns if c.startswith(f"mask{SEP}")]
    # Loop
    for c in mask_cols:
        # Channel name is the last piece
        channel = c.split(SEP)[-1]
        # Count present
        present = int(table[c].sum())
        # Row
        rows.append({"month": month_label, "meter": meter, "channel": channel, "slots": len(table), "present": present, "fraction": round(present / len(table), 4)})
    # Return
    return rows


def append_coverage(out_dir, rows):
    """Append coverage rows to coverage.csv, writing the header if new."""
    # Path
    path = os.path.join(out_dir, "coverage.csv")
    # Header needed only when new
    write_header = not os.path.exists(path)
    # Append
    with open(path, "a", newline="") as f:
        # Writer with fixed columns
        writer = csv.DictWriter(f, fieldnames=["month", "meter", "channel", "slots", "present", "fraction"])
        # Header on first write
        if write_header:
            # Write it
            writer.writeheader()
        # Rows
        writer.writerows(rows)