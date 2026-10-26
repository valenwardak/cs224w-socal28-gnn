# =============================================================================
# File:        socal_download.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-16
# Revised:     2026-10-25  optional thread pool for meter-month downloads
# Revised:     2026-10-25  skip ranges that already have a _failed marker
# Revised:     2026-10-18  never write into or delete a pre-existing folder;
#                          print error types; tolerate existing _failed entries
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
# Safety:      A folder that already exists is never written into and never
#              deleted, whether or not it has a DONE marker. Only a folder
#              this run creates is downloaded into, and only that folder is
#              removed if its download fails.
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
# Standard library: overlapping network waits without extra processes
from concurrent.futures import ThreadPoolExecutor, as_completed
# Standard library: lock for the shared download log
import threading
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
# Log file at the root, one row per attempt
LOG_FILE = "download_log.csv"
# One lock for download_log.csv so parallel threads do not interleave rows
_LOG_LOCK = threading.Lock()


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


def is_failed(meter_dir, label):
    """Return True if this range has a _failed marker (file or folder)."""
    # Same path mark_failed writes
    return os.path.exists(os.path.join(meter_dir, FAILED_DIR, label))


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

    Columns: meter, data_type, label, status, attempts, finished_at
    """
    # Serialize writers so parallel jobs cannot split a row
    with _LOG_LOCK:
        # Full path to the log
        log_path = os.path.join(output_dir, LOG_FILE)
        # Header needed only when the file does not exist yet
        write_header = not os.path.exists(log_path)
        # Open in append mode
        with open(log_path, "a", newline="") as f:
            # CSV writer
            writer = csv.writer(f)
            # Header on first write only
            if write_header:
                # Column names
                writer.writerow(["meter", "data_type", "label", "status", "attempts", "finished_at"])
            # The data row
            writer.writerow(row)


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

    The folder must not exist yet; the caller checks that. Writes
    request.json first, then calls the Caltech client. On success writes
    DONE and returns ("ok", attempts). On failure removes the folder this
    call created and returns ("failed", attempts).
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
    # Try up to retries + 1 times
    for attempt in range(1, retries + 2):
        # Catch any network or client error
        try:
            # One request to Caltech
            client.download_data(**{kwarg_name: elements}, time_range=time_range, output_dir=folder)
            # Mark complete
            mark_done(folder)
            # Success
            return "ok", attempt
        # Something went wrong on this attempt
        except Exception as error:
            # Report the failure with its type, since some errors carry no message
            print(f"    attempt {attempt} failed: {type(error).__name__}: {error!r}")
            # Wait before trying again, unless this was the last attempt
            if attempt <= retries:
                # Pause
                time.sleep(retry_wait)
    # Remove only the folder this call created
    shutil.rmtree(folder, ignore_errors=True)
    # All attempts failed
    return "failed", retries + 1


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
    # Skip if already complete
    if is_done(folder):
        # Nothing to do
        return
    # Skip if a previous run already failed this exact request
    if is_failed(meter_dir, label):
        # Say so
        print(f"  skip {meter} {label} (_failed)")
        # Do not retry or split again
        return
    # Never touch a folder that exists without a DONE marker
    if os.path.exists(folder):
        # Report and leave it alone
        print(f"  {meter} {label} exists without DONE, skipped")
        # Record in the log
        append_log(output_dir, [meter, data_type, label, "skipped_existing", 0, datetime.now().isoformat(timespec="seconds")])
        # Nothing more
        return
    # Progress line
    print(f"  {meter} {label}")
    # Make the request
    status, attempts = request_range(client, data_type, meter, [channel], start, end, folder, retries, retry_wait)
    # Log the attempt
    append_log(output_dir, [meter, data_type, label, status, attempts, datetime.now().isoformat(timespec="seconds")])
    # Success, done
    if status == "ok":
        # Nothing more
        return
    # Failure: leave a marker
    mark_failed(meter_dir, label)
    # Stop splitting once the range is short enough
    if (end - start).days + 1 <= min_days:
        # Give up on this piece
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
    # Skip if already complete
    if is_done(folder):
        # Say so
        print(f"  skip {meter} {label} (DONE)")
        # Nothing to do
        return
    # Never touch a folder that exists without a DONE marker
    if os.path.exists(folder):
        # Report and leave it alone
        print(f"  {meter} {label} exists without DONE, skipped")
        # Record in the log
        append_log(output_dir, [meter, data_type, label, "skipped_existing", 0, datetime.now().isoformat(timespec="seconds")])
        # Nothing more
        return
    # All-channels request already failed: skip retrying it, still try per channel
    if is_failed(meter_dir, label):
        # Say so
        print(f"  skip {meter} {label} (_failed)")
    # Otherwise try all channels together first
    else:
        # Progress line
        print(f"  {meter} {label} ({len(channels)} channels)")
        # Make the request for all channels
        status, attempts = request_range(client, data_type, meter, channels, start, end, folder, retries, retry_wait)
        # Log the attempt
        append_log(output_dir, [meter, data_type, label, status, attempts, datetime.now().isoformat(timespec="seconds")])
        # Success, done
        if status == "ok":
            # Nothing more
            return
        # Failure: leave a marker
        mark_failed(meter_dir, label)
    # Fall back to one channel at a time
    for channel in channels:
        # Each channel gets its own folder and its own splitting
        download_channel_range(client, output_dir, meter, data_type, channel, start, end, retries, retry_wait, min_days)


def find_missing_channels(output_dir, meters):
    """
    Scan every DONE folder that has a request.json, compare the channels
    requested with the files present, and return
    {meter: {label: [missing channels]}}.

    Folders from older downloads without request.json are skipped.
    """
    # Result dictionary
    missing = {}
    # Loop over meters
    for meter in meters:
        # Meter folder
        meter_dir = os.path.join(output_dir, meter)
        # Skip meters with nothing downloaded
        if not os.path.isdir(meter_dir):
            # Next meter
            continue
        # Loop over range folders
        for label in sorted(os.listdir(meter_dir)):
            # Full path
            folder = os.path.join(meter_dir, label)
            # Only completed folders count
            if not is_done(folder):
                # Next folder
                continue
            # Path of the request record
            request_path = os.path.join(folder, REQUEST_FILE)
            # Skip folders without one
            if not os.path.exists(request_path):
                # Next folder
                continue
            # Read what was requested
            with open(request_path) as f:
                # Parse JSON
                request = json.load(f)
            # Older request files may not have a channels list
            if "channels" not in request:
                # Next folder
                continue
            # Collect every file name under the folder
            present = {name for _, _, files in os.walk(folder) for name in files}
            # Channels whose file did not come back
            absent = [c for c in request["channels"] if f"{meter}-{c}.csv" not in present]
            # Record if any are missing
            if absent:
                # Nested dictionary entry
                missing.setdefault(meter, {})[label] = absent
    # Return the full map
    return missing


def write_missing_channels(output_dir, meters):
    """Write missing_channels.json at the root of the output folder."""
    # Compute the map
    missing = find_missing_channels(output_dir, meters)
    # Write it
    with open(os.path.join(output_dir, MISSING_FILE), "w") as f:
        # Readable JSON
        json.dump(missing, f, indent=2)
    # Return it for the caller to print a summary
    return missing


def download_all(client, output_dir, meters, data_type, channels, start_date, end_date, retries, retry_wait, min_days, workers=4):
    """
    Download every meter for every month in the date range, then write
    missing_channels.json. Returns the missing-channel map.

    workers is how many meter-month requests may run at once. Each worker
    is a thread, so this overlaps waiting on the Caltech API without extra
    processes or many CPUs. 1 keeps the original sequential order. Fallback
    splitting inside a meter-month still runs on that same thread.
    """
    # Make the root output folder
    os.makedirs(output_dir, exist_ok=True)
    # Treat a missing or non-positive value as sequential
    workers = 1 if workers is None else max(1, int(workers))
    # Sequential path: same meter-then-month order as before
    if workers == 1:
        # Loop over meters
        for meter in meters:
            # Progress line
            print(f"meter {meter}")
            # Loop over months
            for start, end in month_chunks(start_date, end_date):
                # Download this meter and month
                download_meter_month(client, output_dir, meter, data_type, channels, start, end, retries, retry_wait, min_days)
            # Update the missing-channel file after each meter so it is always current
            write_missing_channels(output_dir, meters)
        # Final pass
        return write_missing_channels(output_dir, meters)
    # One job per meter and month; folders do not overlap, so threads stay independent
    jobs = [(meter, start, end) for meter in meters for start, end in month_chunks(start_date, end_date)]
    # Say how much work is queued
    print(f"{len(jobs)} meter-month jobs, {workers} threads")
    # Keep Caltech's per-download tqdm bars from tearing each other
    try:
        # Imported here so sequential runs do not need it
        from tqdm import tqdm
        # One lock for all bars
        tqdm.set_lock(threading.RLock())
    # tqdm is a dependency, but a missing install should not block downloads
    except ImportError:
        # Continue without a tqdm lock
        pass
    # Run one meter-month on this thread, including any channel fallback
    def run_job(job):
        # Unpack the job
        meter, start, end = job
        # Same helper as the sequential path
        download_meter_month(client, output_dir, meter, data_type, channels, start, end, retries, retry_wait, min_days)
    # Collect unexpected errors so every job still gets a chance to finish
    errors = []
    # Small pool: threads release the GIL while waiting on the network
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Submit every meter-month
        futures = [pool.submit(run_job, job) for job in jobs]
        # Harvest them as they finish
        for future in as_completed(futures):
            # Catch so one bad job does not discard the rest
            try:
                # Re-raise into this thread
                future.result()
            # download_meter_month already catches request errors; this is extra safety
            except Exception as error:
                # Report the failure with its type
                print(f"  job failed: {type(error).__name__}: {error!r}")
                # Keep it to raise after the pool drains
                errors.append(error)
    # Surface the first unexpected error after the other jobs have finished
    if errors:
        # Raise so the caller sees that something besides a request failure happened
        raise errors[0]
    # Scan once at the end; DONE markers already make a crash restartable
    return write_missing_channels(output_dir, meters)