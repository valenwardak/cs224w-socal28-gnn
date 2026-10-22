# =============================================================================
# File:        socal_download.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-16
# Revised:     2026-10-18  never write into or delete a pre-existing folder;
#                          print error types; tolerate existing _failed entries
#              2026-10-21  validate before DONE; clean partial failures; resume
#                          incomplete folders; expanded download log and missing list
#
# Purpose:     Helper functions for downloading SoCal 28-Bus data from the
#              Caltech API, one meter and one month per request, into the
#              folder layout below, so no download overwrites another and
#              every team member ends up with identical files. This module
#              has no main block; it is imported by main_socal.py.
#
# Folder layout produced:
#              <output_dir>/<meter>/<start>_<end>/request.json
#              <output_dir>/<meter>/<start>_<end>/phasors/<meter>-<channel>.csv
#              <output_dir>/<meter>/<start>_<end>/DONE
#              <output_dir>/<meter>/<start>_<end>_<channel>/...   (fallback, one channel)
#              <output_dir>/<meter>/_failed/<start>_<end>[_<channel>]   (empty marker)
#              <output_dir>/missing_channels.json
#              <output_dir>/download_log.csv
#
# Fallback strategy when a request fails:
#              1. whole month, all channels
#              2. whole month, one channel at a time
#              3. split the date range in half, repeat, down to one day
#
# Safety:      Folders with DONE are never modified. Incomplete folders from a
#              prior run are removed and retried. Failed attempts remove all
#              partial data for that request before continuing.
#
# Requirements:
#              dataset_api_client.py (from caltech-netlab/digital-twin-dataset)
#              in the same folder, plus
#              pip install requests stream-unzip tqdm pyperclip
#
# Access:      Each person needs their own Caltech API access, tied to their
#              GitHub account: https://forms.office.com/r/Ds6rKEtyTV
#              The first run opens a browser to sign in with GitHub and saves
#              dataset_api_credentials.json next to the scripts. Never commit
#              that file.
# =============================================================================

# Standard library: CSV writing for the download log
import csv
# Standard library: JSON for request.json and missing_channels.json
import json
# Standard library: file and folder operations
import os
# Standard library: removing a folder this run created, after a failed attempt
import shutil
# Standard library: pausing between retries
import time
# Standard library: dates for building ranges
from datetime import date, datetime, timedelta
# Caltech client that talks to the SoCal 28-Bus API
from dataset_api_client import DatasetApiClient

# Maps our data type names to the keyword arguments the Caltech client expects
CLIENT_KWARG = {
    # Synchrophasor data, ten-second complex values
    "phasors": "phasors_for",
    # RMS magnitude data, one-second values
    "magnitudes": "magnitudes_for",
    # Raw point-on-wave samples, very large
    "waveforms": "waveforms_for",
}
# Marker file written when a folder is complete
DONE_FILE = "DONE"
# File recording what each folder asked the server for
REQUEST_FILE = "request.json"
# Folder holding empty markers for failed attempts
FAILED_DIR = "_failed"
# File at the root listing channels that never came back
MISSING_FILE = "missing_channels.json"
# Log file at the root, one row per request
LOG_FILE = "download_log.csv"
LOG_COLUMNS = [
    "meter",
    "data_type",
    "channels",
    "start",
    "end",
    "label",
    "status",
    "attempts",
    "error_type",
    "error_message",
    "finished_at",
]
# Subfolder name the API client uses for phasor CSV files
PHASORS_SUBDIR = "phasors"
MAGNITUDES_SUBDIR = "magnitudes"
WAVEFORMS_SUBDIR = "waveforms"
DATA_SUBDIR = {
    "phasors": PHASORS_SUBDIR,
    "magnitudes": MAGNITUDES_SUBDIR,
    "waveforms": WAVEFORMS_SUBDIR,
}


class DownloadValidationError(Exception):
    """Raised when the API response unpacked but expected files are missing."""


def read_meters(meters_file):
    """
    Read meter names from a text file, one per line.

    Blank lines and lines starting with # are ignored, so the file can
    carry comments such as "# egauge_18 has only six channels".
    """
    # Fail with a clear message if the file is missing
    if not os.path.exists(meters_file):
        # Raise so the caller sees the exact path that was tried
        raise FileNotFoundError(f"Meters file not found: {meters_file}")
    # Open the file for reading
    with open(meters_file) as f:
        # Strip whitespace from every line
        lines = [line.strip() for line in f]
    # Keep lines that are not empty and not comments
    meters = [line for line in lines if line and not line.startswith("#")]
    # Refuse to run with an empty list, which would download nothing
    if not meters:
        # Raise so the caller fixes the file
        raise ValueError(f"No meter names found in {meters_file}")
    # Return the cleaned list
    return meters


def parse_date(text):
    """Turn a YYYY-MM-DD string into a date object."""
    # Parse and drop the time part
    return datetime.strptime(text, "%Y-%m-%d").date()


def month_chunks(start_date, end_date):
    """
    Yield (chunk_start, chunk_end) date pairs, one per calendar month,
    both ends inclusive, clipped to start_date and end_date.

    Example: 2025-08-15 to 2025-09-03 gives
             (2025-08-15, 2025-08-31) and (2025-09-01, 2025-09-03).
    """
    # Begin at the start date
    current = start_date
    # Loop until we pass the end date
    while current <= end_date:
        # First day of the following month
        if current.month == 12:
            # January of next year
            next_month = date(current.year + 1, 1, 1)
        # Any other month
        else:
            # First day of next month
            next_month = date(current.year, current.month + 1, 1)
        # Last day of this chunk is the day before next month, or the end date
        chunk_end = min(next_month - timedelta(days=1), end_date)
        # Hand back this chunk
        yield current, chunk_end
        # Move to the first day of the next month
        current = next_month


def range_label(start, end, channel=None):
    """
    Build the folder name for a date range, with an optional channel suffix.

    Example: (2025-07-01, 2025-07-31) gives "2025-07-01_2025-07-31"
             with channel "L2" gives "2025-07-01_2025-07-31_L2"
    """
    # Base label from the two dates
    label = f"{start.isoformat()}_{end.isoformat()}"
    # Add the channel when this is a single-channel folder
    if channel is not None:
        # Append with an underscore
        label = f"{label}_{channel}"
    # Return the label
    return label


def split_range(start, end):
    """
    Split an inclusive date range into two halves.

    Returns ((start, mid), (mid + 1 day, end)).
    """
    # Number of days in the range minus one
    span = (end - start).days
    # Midpoint date
    mid = start + timedelta(days=span // 2)
    # Two halves
    return (start, mid), (mid + timedelta(days=1), end)


def is_done(folder):
    """Return True if the folder has a DONE marker."""
    # Check for the marker file
    return os.path.exists(os.path.join(folder, DONE_FILE))


def mark_done(folder):
    """Create the DONE marker in a folder."""
    # Create an empty file
    open(os.path.join(folder, DONE_FILE), "w").close()


def mark_failed(meter_dir, label):
    """
    Create an empty marker under _failed for a failed attempt.

    Older downloads stored these markers as files rather than folders, so
    an existing entry of either kind is left as it is.
    """
    # Full path of the marker
    marker = os.path.join(meter_dir, FAILED_DIR, label)
    # Leave an existing file or folder alone
    if os.path.exists(marker):
        # Nothing to do
        return
    # Make the marker folder
    os.makedirs(marker)


def write_request(folder, payload):
    """Write request.json describing what was asked of the server."""
    # Make sure the folder exists
    os.makedirs(folder, exist_ok=True)
    # Write the payload as readable JSON
    with open(os.path.join(folder, REQUEST_FILE), "w") as f:
        # Two-space indent for readability
        json.dump(payload, f, indent=2)


def append_log(output_dir, row):
    """
    Append one row to download_log.csv, writing the header if new.

    ``row`` is a dict with keys from ``LOG_COLUMNS`` or a list in that order.
    """
    # Full path to the log
    log_path = os.path.join(output_dir, LOG_FILE)
    # Header needed only when the file does not exist yet
    write_header = not os.path.exists(log_path)
    if isinstance(row, dict):
        row = [row.get(col, "") for col in LOG_COLUMNS]
    # Open in append mode
    with open(log_path, "a", newline="") as f:
        # CSV writer
        writer = csv.writer(f)
        # Header on first write only
        if write_header:
            writer.writerow(LOG_COLUMNS)
        writer.writerow(row)


def channel_csv_name(meter, channel):
    """File name for one channel under the data-type subfolder."""
    return f"{meter}-{channel}.csv"


def validate_download(folder, meter, data_type, channels):
    """
    Return None if every requested channel CSV exists under the folder.

    Empty CSV files are allowed (no data for that range). Raises
    ``DownloadValidationError`` when a file is missing.
    """
    subdir = DATA_SUBDIR[data_type]
    missing = []
    for channel in channels:
        csv_path = os.path.join(folder, subdir, channel_csv_name(meter, channel))
        if not os.path.isfile(csv_path):
            missing.append(channel)
    if missing:
        raise DownloadValidationError(
            f"missing channel file(s) after download: {', '.join(missing)}"
        )


def remove_partial_download(folder):
    """
    Delete an incomplete download folder. Never touches a folder with DONE.
    """
    if not os.path.exists(folder):
        return
    if is_done(folder):
        return
    shutil.rmtree(folder, ignore_errors=True)


def clean_partial_contents(folder):
    """
    Remove downloaded files under ``folder`` but keep local ``request.json``.

    Used between retries for the same request. Never runs on a DONE folder.
    """
    if not os.path.isdir(folder) or is_done(folder):
        return
    for name in os.listdir(folder):
        if name == REQUEST_FILE:
            continue
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)


def prepare_download_folder(folder):
    """
    Return True if this folder should be downloaded into.

    Skips when DONE. Removes and retries incomplete folders from prior runs.
    """
    if is_done(folder):
        return False
    if os.path.exists(folder):
        remove_partial_download(folder)
    return True


def log_request_result(
    output_dir,
    meter,
    data_type,
    channels,
    start,
    end,
    label,
    status,
    attempts,
    error_type="",
    error_message="",
):
    """Append one outcome row for a ``request_range`` call."""
    append_log(
        output_dir,
        {
            "meter": meter,
            "data_type": data_type,
            "channels": " ".join(channels),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "label": label,
            "status": status,
            "attempts": attempts,
            "error_type": error_type,
            "error_message": error_message,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _missing_entries_from_legacy(data):
    """Convert older ``{meter: {label: [channels]}}`` maps to entry dicts."""
    entries = []
    if not isinstance(data, dict):
        return entries
    for meter, labels in data.items():
        if meter == "missing" or not isinstance(labels, dict):
            continue
        for label, chans in labels.items():
            if not isinstance(chans, list):
                continue
            start_s, end_s, _ = _parse_label_dates(label)
            entries.append(
                {
                    "meter": meter,
                    "start": start_s,
                    "end": end_s,
                    "channels": chans,
                    "label": label,
                }
            )
    return entries


def _parse_label_dates(label):
    """
    Parse ``start_end`` or ``start_end_CHANNEL`` folder names.

    Returns (start_iso, end_iso, channel_or_none).
    """
    parts = label.split("_")
    if len(parts) < 2:
        return "", "", None
    start_s = parts[0]
    end_s = parts[1]
    channel = parts[2] if len(parts) > 2 else None
    return start_s, end_s, channel


def load_missing_registry(output_dir):
    """Load persisted missing-download entries from ``missing_channels.json``."""
    path = os.path.join(output_dir, MISSING_FILE)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "missing" in data:
        return list(data["missing"])
    return _missing_entries_from_legacy(data)


def record_missing(output_dir, meter, start, end, channels, label):
    """Record a channel/date range that could not be downloaded."""
    entry = {
        "meter": meter,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "channels": list(channels),
        "label": label,
    }
    entries = load_missing_registry(output_dir)
    key = (entry["meter"], entry["label"], tuple(entry["channels"]))
    seen = {(e["meter"], e["label"], tuple(e["channels"])) for e in entries}
    if key not in seen:
        entries.append(entry)
    _write_missing_file(output_dir, entries)


def find_absent_in_done_folders(output_dir, meters):
    """
    Scan DONE folders whose request.json lists channels with no CSV on disk.

    Returns a list of missing entries (may include multi-channel folders).
    """
    entries = []
    for meter in meters:
        meter_dir = os.path.join(output_dir, meter)
        if not os.path.isdir(meter_dir):
            continue
        for label in sorted(os.listdir(meter_dir)):
            folder = os.path.join(meter_dir, label)
            if not is_done(folder):
                continue
            request_path = os.path.join(folder, REQUEST_FILE)
            if not os.path.exists(request_path):
                continue
            with open(request_path) as f:
                request = json.load(f)
            if "channels" not in request:
                continue
            data_type = request.get("data_type", "phasors")
            try:
                validate_download(folder, meter, data_type, request["channels"])
            except DownloadValidationError:
                present = set()
                subdir = DATA_SUBDIR.get(data_type, PHASORS_SUBDIR)
                sub_path = os.path.join(folder, subdir)
                if os.path.isdir(sub_path):
                    present = set(os.listdir(sub_path))
                absent = [
                    c
                    for c in request["channels"]
                    if channel_csv_name(meter, c) not in present
                ]
                if absent:
                    entries.append(
                        {
                            "meter": meter,
                            "start": request.get("start", _parse_label_dates(label)[0]),
                            "end": request.get("end", _parse_label_dates(label)[1]),
                            "channels": absent,
                            "label": label,
                        }
                    )
    return entries


def _merge_missing_entries(*entry_lists):
    """Deduplicate missing entries by meter, label, and channels."""
    merged = []
    seen = set()
    for entries in entry_lists:
        for entry in entries:
            key = (entry["meter"], entry["label"], tuple(entry["channels"]))
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
    merged.sort(key=lambda e: (e["meter"], e["label"], e["channels"]))
    return merged


def _write_missing_file(output_dir, entries):
    with open(os.path.join(output_dir, MISSING_FILE), "w") as f:
        json.dump({"missing": entries}, f, indent=2)


def make_client(credentials_file):
    """
    Create the Caltech client and confirm the login works.

    The first call triggers the GitHub sign-in in a browser. Returns the
    client, and prints the GitHub user it is signed in as.
    """
    # Build the client pointing at the given credentials file
    client = DatasetApiClient(credentials_file=credentials_file)
    # Ask the API who we are, which also verifies access
    print(f"Signed in as: {client.get_user()}")
    # Return the ready client
    return client


def request_range(client, data_type, meter, channels, start, end, folder, retries, retry_wait):
    """
    Make one request for the given channels and date range, with retries.

    Writes request.json, calls the Caltech client, validates channel CSVs,
    then writes DONE. On any failure, removes partial data for this request.
    Returns (status, attempts, error_type, error_message).
    """
    # Element names such as egauge_1-L1
    elements = [f"{meter}-{channel}" for channel in channels]
    # Keyword argument the client expects for this data type
    kwarg_name = CLIENT_KWARG[data_type]
    # The server's end bound is exclusive, so request through the end of the last day
    time_range = (datetime.combine(start, datetime.min.time()), datetime.combine(end + timedelta(days=1), datetime.min.time()))
    # Record what we are asking for; this also creates the folder
    write_request(folder, {
        # Meter name
        "meter": meter,
        # Data type
        "data_type": data_type,
        # Channels requested
        "channels": channels,
        # Inclusive start date
        "start": start.isoformat(),
        # Inclusive end date
        "end": end.isoformat(),
        # When the request was made
        "requested_at": datetime.now().isoformat(timespec="seconds"),
    })
    last_error_type = ""
    last_error_message = ""
    # Try up to retries + 1 times
    for attempt in range(1, retries + 2):
        # Catch any network or client error
        try:
            # One request to Caltech
            client.download_data(**{kwarg_name: elements}, time_range=time_range, output_dir=folder)
            validate_download(folder, meter, data_type, channels)
            mark_done(folder)
            return "ok", attempt, "", ""
        except Exception as error:
            last_error_type = type(error).__name__
            last_error_message = str(error) or repr(error)
            print(f"    attempt {attempt} failed: {last_error_type}: {last_error_message!r}")
            clean_partial_contents(folder)
            if attempt <= retries:
                time.sleep(retry_wait)
    remove_partial_download(folder)
    return "failed", retries + 1, last_error_type, last_error_message


def download_channel_range(client, output_dir, meter, data_type, channel, start, end, retries, retry_wait, min_days):
    """
    Download one channel over a date range, splitting the range in half on
    failure until it is min_days long or less.
    """
    # Meter folder
    meter_dir = os.path.join(output_dir, meter)
    # Folder name for this range and channel
    label = range_label(start, end, channel)
    # Full folder path
    folder = os.path.join(meter_dir, label)
    if not prepare_download_folder(folder):
        return
    # Progress line
    print(f"  {meter} {label}")
    # Make the request
    status, attempts, error_type, error_message = request_range(
        client, data_type, meter, [channel], start, end, folder, retries, retry_wait
    )
    log_request_result(
        output_dir,
        meter,
        data_type,
        [channel],
        start,
        end,
        label,
        status,
        attempts,
        error_type,
        error_message,
    )
    if status == "ok":
        return
    mark_failed(meter_dir, label)
    if (end - start).days + 1 <= min_days:
        record_missing(output_dir, meter, start, end, [channel], label)
        return
    # Split in half
    (a_start, a_end), (b_start, b_end) = split_range(start, end)
    # First half
    download_channel_range(client, output_dir, meter, data_type, channel, a_start, a_end, retries, retry_wait, min_days)
    # Second half
    download_channel_range(client, output_dir, meter, data_type, channel, b_start, b_end, retries, retry_wait, min_days)


def download_meter_month(client, output_dir, meter, data_type, channels, start, end, retries, retry_wait, min_days):
    """
    Download one meter for one month: all channels together first, then
    fall back to one channel at a time with range splitting.
    """
    # Meter folder
    meter_dir = os.path.join(output_dir, meter)
    # Folder name for the whole month
    label = range_label(start, end)
    # Full folder path
    folder = os.path.join(meter_dir, label)
    if is_done(folder):
        print(f"  skip {meter} {label} (DONE)")
        return
    if not prepare_download_folder(folder):
        return
    # Progress line
    print(f"  {meter} {label} ({len(channels)} channels)")
    # Make the request for all channels
    status, attempts, error_type, error_message = request_range(
        client, data_type, meter, channels, start, end, folder, retries, retry_wait
    )
    log_request_result(
        output_dir,
        meter,
        data_type,
        channels,
        start,
        end,
        label,
        status,
        attempts,
        error_type,
        error_message,
    )
    if status == "ok":
        return
    # Failure: leave a marker
    mark_failed(meter_dir, label)
    # Fall back to one channel at a time
    for channel in channels:
        # Each channel gets its own folder and its own splitting
        download_channel_range(client, output_dir, meter, data_type, channel, start, end, retries, retry_wait, min_days)


def _entry_is_complete(output_dir, entry):
    """Return True when a missing entry now has a valid DONE download."""
    folder = os.path.join(output_dir, entry["meter"], entry["label"])
    if not is_done(folder):
        return False
    data_type = "phasors"
    request_path = os.path.join(folder, REQUEST_FILE)
    if os.path.exists(request_path):
        with open(request_path) as f:
            data_type = json.load(f).get("data_type", data_type)
    try:
        validate_download(folder, entry["meter"], data_type, entry["channels"])
        return True
    except DownloadValidationError:
        return False


def find_missing_channels(output_dir, meters):
    """
    Return all channel/date ranges not successfully downloaded.

    Merges the persisted failure registry with DONE folders that are missing
    expected CSV files (for example from older runs before validation).
    """
    registry = [
        entry
        for entry in load_missing_registry(output_dir)
        if not _entry_is_complete(output_dir, entry)
    ]
    absent_done = find_absent_in_done_folders(output_dir, meters)
    entries = _merge_missing_entries(registry, absent_done)
    return {"missing": entries}


def write_missing_channels(output_dir, meters):
    """Write missing_channels.json at the root of the output folder."""
    missing = find_missing_channels(output_dir, meters)
    _write_missing_file(output_dir, missing["missing"])
    return missing


def download_all(client, output_dir, meters, data_type, channels, start_date, end_date, retries, retry_wait, min_days):
    """
    Download every meter for every month in the date range, then write
    missing_channels.json. Returns the missing-channel map.
    """
    # Make the root output folder
    os.makedirs(output_dir, exist_ok=True)
    # Loop over meters
    for meter in meters:
        # Progress line
        print(f"meter {meter}")
        # Loop over months
        for start, end in month_chunks(start_date, end_date):
            try:
                download_meter_month(
                    client,
                    output_dir,
                    meter,
                    data_type,
                    channels,
                    start,
                    end,
                    retries,
                    retry_wait,
                    min_days,
                )
            except Exception as error:
                label = range_label(start, end)
                print(f"  {meter} {label} unexpected error: {type(error).__name__}: {error!r}")
                log_request_result(
                    output_dir,
                    meter,
                    data_type,
                    channels,
                    start,
                    end,
                    label,
                    "failed",
                    0,
                    type(error).__name__,
                    str(error) or repr(error),
                )
        # Update the missing-channel file after each meter so it is always current
        write_missing_channels(output_dir, meters)
    # Final pass
    return write_missing_channels(output_dir, meters)