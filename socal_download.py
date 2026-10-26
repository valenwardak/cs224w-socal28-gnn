# =============================================================================
# File:        socal_download.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-16
# Revised:     2026-10-25  skip empty ~191B ZIPs; retry only transient errors;
#                          keep valid CSVs from truncated multi-channel ZIPs
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
#              <output_dir>/<meter>/_failed/<start>_<end>[_<channel>]   (empty ZIP only)
#              <output_dir>/missing_channels.json
#              <output_dir>/download_log.csv
#
# Fallback strategy when a request fails:
#              1. whole month, all channels
#              2. keep independently validated CSVs; retry missing channels
#              3. split substantial truncated ranges in half, down to one day
#              Tiny empty ZIP responses (~191 bytes) are logged as missing
#              and marked _failed so reruns skip them. Transient network,
#              HTTP, and Drive errors retry with backoff and are not given
#              a permanent _failed marker.
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
# Standard library: local staging so Drive only sees complete files
import tempfile
# Standard library: pausing between retries
import time
# Standard library: overlapping network waits without extra processes
from concurrent.futures import ThreadPoolExecutor, as_completed
# Standard library: lock for the shared download log
import threading
# Standard library: dates for building ranges
from datetime import date, datetime, timedelta
# HTTP client exceptions for selective retries
import requests
# Caltech client module, patched so 429/5xx are retryable
import dataset_api_client
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
# Empty Caltech ZIP responses are ~191 uncompressed bytes. Only those tiny
# payloads skip retry/split; anything larger is split in case data exists.
EMPTY_RESPONSE_MAX_BYTES = 256
# Upper bound for exponential backoff, in seconds
RETRY_BACKOFF_CAP = 300


class RetryableHTTPError(Exception):
    """HTTP 429 or 5xx from the Caltech API; callers back off and retry."""

    def __init__(self, status_code):
        # Preserve the status so is_retryable_error can read it
        self.status_code = status_code
        # Human-readable message
        super().__init__(f"HTTP {status_code}")


def _install_retryable_http_hook():
    """Turn the client's 429/5xx RuntimeError into RetryableHTTPError."""
    # The helper DatasetApiClient already calls on every response
    orig = dataset_api_client.raise_response_error
    # Avoid wrapping twice if the module is reloaded
    if getattr(orig, "_socal_retryable", False):
        # Already installed
        return

    def wrapped(response):
        # Failed HTTP responses are the only ones that need a status check
        if not getattr(response, "ok", True):
            # Status code from requests
            code = getattr(response, "status_code", None)
            # Rate limit or server error: retry, do not treat as missing data
            if code == 429 or (isinstance(code, int) and code >= 500):
                # Typed error so request_range does not inspect the message
                raise RetryableHTTPError(code)
        # All other statuses keep the client's original behavior
        return orig(response)

    # Mark so a second import does not wrap the wrapper
    wrapped._socal_retryable = True
    # Install
    dataset_api_client.raise_response_error = wrapped


# Install once at import; the patch is thread-safe
_install_retryable_http_hook()


def _say(message):
    """Print one line without interleaving output from other workers."""
    # Same lock as the CSV log
    with _LOG_LOCK:
        # One complete line
        print(message)


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


def retry_delay(attempt, retry_wait):
    """Seconds to wait after a failed attempt, capped exponential backoff."""
    # attempt is 1-based; the wait is before the next try
    delay = retry_wait * (2 ** (attempt - 1))
    # Bound so a long retry chain cannot grow without limit
    return min(delay, RETRY_BACKOFF_CAP)


def is_unzip_error(error):
    """True for stream-unzip failures, including TruncatedDataError."""
    # Name check so tests can raise a stub without importing stream_unzip
    if type(error).__name__ in ("TruncatedDataError", "UnzipError", "DataError", "IntegrityError", "SizeIntegrityError", "CRC32IntegrityError"):
        # Deterministic ZIP problem
        return True
    # Also match the real hierarchy when stream-unzip is installed
    try:
        # Local import keeps module load cheap if the extra is missing
        from stream_unzip import UnzipError
    except ImportError:
        # No library, name check above is enough
        return False
    # True for TruncatedDataError and every other UnzipError
    return isinstance(error, UnzipError)


def is_retryable_error(error):
    """
    True for transient network, HTTP 429/5xx, and Drive I/O failures.

    ZIP corruption is not retryable: the same payload will fail again.
    """
    # Status-typed HTTP errors from the client hook
    if isinstance(error, RetryableHTTPError):
        # Always retry 429/5xx
        return True
    # Timeouts and dropped connections
    if isinstance(error, (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError)):
        # Transient
        return True
    # requests.HTTPError if a test or wrapper raised one directly
    if isinstance(error, requests.exceptions.HTTPError):
        # Status if the response object is present
        code = getattr(getattr(error, "response", None), "status_code", None)
        # Same rule as RetryableHTTPError
        return code == 429 or (isinstance(code, int) and code >= 500)
    # Google Drive FUSE I/O errors are OSError with errno 5, 110, etc.
    if isinstance(error, OSError) and not isinstance(error, (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)):
        # Retry the transfer
        return True
    # Not a transient failure
    return False


def folder_payload_bytes(folder):
    """Uncompressed bytes written under folder, excluding request.json and DONE."""
    # Running total
    total = 0
    # Missing folder is an empty payload
    if not os.path.isdir(folder):
        # Nothing to count
        return 0
    # Walk every extracted file
    for root, _, files in os.walk(folder):
        # Each file
        for name in files:
            # Markers are not payload
            if name in (REQUEST_FILE, DONE_FILE):
                # Skip
                continue
            # Add size; ignore files that vanished mid-walk
            try:
                # Bytes on disk
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                # Treat as zero
                continue
    # Total extracted bytes
    return total


def csv_is_complete(path):
    """
    True if the CSV ends on a complete row.

    This catches mid-row truncation. It cannot tell a finished ZIP member
    from a prefix that happens to end on a newline; discard_newest_csv
    handles that case after an unzip error. Timestamp coverage and row
    counts are not used: SoCal current channels and outage windows are
    legitimately sparse.
    """
    # Missing or tiny files are incomplete
    try:
        # Size in bytes
        size = os.path.getsize(path)
    except OSError:
        # Cannot read
        return False
    # Smaller than "t\\nx\\n"
    if size < 4:
        # Incomplete
        return False
    # Inspect header, a data row, and the tail without loading the file
    try:
        # Binary so a truncated last byte is visible
        with open(path, "rb") as f:
            # Header line
            header = f.readline()
            # Empty or whitespace-only header
            if not header.strip():
                # Incomplete
                return False
            # Columns in the header
            n_cols = header.count(b",") + 1
            # First data row
            data = f.readline()
            # Need at least one record
            if not data.strip():
                # Header only
                return False
            # File must end on a line boundary
            f.seek(0, os.SEEK_END)
            # Byte length
            end = f.tell()
            # Last byte
            f.seek(-1, os.SEEK_END)
            # Mid-row truncation leaves a non-newline tail
            if f.read(1) not in (b"\n", b"\r"):
                # Incomplete last row
                return False
            # Last 8 KB is enough to recover the last line
            f.seek(max(0, end - 8192))
            # Tail bytes
            tail = f.read()
        # Normalize newlines and drop the final terminator
        last = tail.replace(b"\r\n", b"\n").replace(b"\r", b"\n").rstrip(b"\n")
        # Nothing left
        if not last:
            # Incomplete
            return False
        # Last physical line
        last_line = last.rsplit(b"\n", 1)[-1]
        # Column count must match the header
        return last_line.count(b",") + 1 == n_cols
    except OSError:
        # Unreadable
        return False


def discard_newest_csv(folder):
    """
    Delete the most recently written CSV in folder.

    stream-unzip CRC-checks each member only after it finishes. When the
    stream dies, the open member is incomplete even if its last bytes are
    a valid CSV row. That file is the newest one on disk.
    """
    # Path of the newest CSV, if any
    newest = None
    # Modification time of that file, nanoseconds
    newest_mtime = None
    # Nothing to drop
    if not os.path.isdir(folder):
        # Done
        return
    # Find the newest CSV
    for root, _, files in os.walk(folder):
        # Each name
        for name in files:
            # Only extracted CSVs
            if not name.endswith(".csv"):
                # Skip
                continue
            # Full path
            path = os.path.join(root, name)
            # Stat may fail if the file vanished
            try:
                # Nanosecond mtime distinguishes files written in one second
                mtime = os.stat(path).st_mtime_ns
            except OSError:
                # Skip unreadable
                continue
            # Keep the latest; >= so a later path wins a tie
            if newest_mtime is None or mtime >= newest_mtime:
                # Update
                newest_mtime = mtime
                # Path to remove
                newest = path
    # Remove the in-progress member
    if newest is not None:
        # Ignore a race with a parallel cleanup
        try:
            # Drop it so promote_complete_csvs cannot copy a prefix
            os.remove(newest)
        except OSError:
            # Already gone
            pass


def present_channels(folder, meter, channels):
    """Channels whose CSV in folder passes csv_is_complete."""
    # Map file name back to channel suffix
    wanted = {f"{meter}-{channel}.csv": channel for channel in channels}
    # Channels that validated
    found = set()
    # Missing folder has nothing
    if not os.path.isdir(folder):
        # Empty
        return found
    # Walk extracted files
    for root, _, files in os.walk(folder):
        # Each name
        for name in files:
            # Channel this file belongs to, if any
            channel = wanted.get(name)
            # Skip timestamps and unrelated files
            if channel is None:
                # Next file
                continue
            # Independently validate; incomplete CSVs do not count
            if csv_is_complete(os.path.join(root, name)):
                # Keep
                found.add(channel)
    # Validated subset
    return found


def promote_complete_csvs(src, dest):
    """Copy only CSVs that pass csv_is_complete from src into dest."""
    # Nothing to copy
    if not os.path.isdir(src):
        # Done
        return
    # Walk staging
    for root, _, files in os.walk(src):
        # Each extracted file
        for name in files:
            # Only CSVs; request.json is written by the caller
            if not name.endswith(".csv"):
                # Skip
                continue
            # Full path in staging
            path = os.path.join(root, name)
            # Drop the file that was open when the ZIP died
            if not csv_is_complete(path):
                # Incomplete
                continue
            # Keep the phasors/ relative path
            rel = os.path.relpath(path, src)
            # Destination path
            out = os.path.join(dest, rel)
            # Make the data-type subfolder
            os.makedirs(os.path.dirname(out), exist_ok=True)
            # Copy onto the real output (Drive sees a complete file)
            shutil.copy2(path, out)


def request_range(client, data_type, meter, channels, start, end, folder, retries, retry_wait):
    """
    Make one request for the given channels and date range, with retries.

    The folder must not exist yet; the caller checks that. Writes
    request.json first, then downloads into a local staging directory.
    Returns (status, attempts) where status is:
      ok        every requested channel has a validated CSV; DONE written
      partial   some channels recovered; folder kept; no DONE
      missing   empty/tiny ZIP, no validated CSVs; folder removed
      truncated substantial ZIP corruption, no validated CSVs; folder removed
      failed    retries exhausted on a transient or unknown error
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
    # Last attempt number, for the return value
    attempts = 0
    # Try up to retries + 1 times
    for attempt in range(1, retries + 2):
        # Record which attempt this is
        attempts = attempt
        # Unzip to local disk, then copy complete files to the destination
        staging = tempfile.mkdtemp(prefix="socal_dl_")
        # Error from this attempt, or None on a clean unzip
        download_error = None
        try:
            # Catch network, HTTP, unzip, and Drive errors
            try:
                # One request to Caltech, streamed into staging
                client.download_data(**{kwarg_name: elements}, time_range=time_range, output_dir=staging)
            except Exception as error:
                # Keep the error to classify after we inspect staging
                download_error = error
                # Report the failure with its type, since some errors carry no message
                _say(f"    attempt {attempt} failed: {type(error).__name__}: {error!r}")
            # Bytes actually unzipped, used to tell empty ZIPs from truncated data
            payload = folder_payload_bytes(staging)
            # The open ZIP member is incomplete even if it ends on a newline
            if download_error is not None and is_unzip_error(download_error):
                # Drop it before any copy onto the destination
                discard_newest_csv(staging)
            # Keep only independently validated CSVs; Drive sees complete files
            try:
                # Copy into the request folder
                promote_complete_csvs(staging, folder)
            except OSError as error:
                # Interrupted copy onto Drive: treat like a failed transfer
                if download_error is None:
                    # Promote failed after a clean unzip
                    download_error = error
                    # Same log line as a download failure
                    _say(f"    attempt {attempt} failed: {type(error).__name__}: {error!r}")
            # Channels whose files survived validation
            have = present_channels(folder, meter, channels)
            # Every requested channel is present: success even if the ZIP
            # complained after the last member (missing central directory)
            if set(channels) <= have:
                # Mark complete
                mark_done(folder)
                # Success
                return "ok", attempt
            # Clean unzip that silently omitted some or all channels
            if download_error is None:
                # Some channels recovered
                if have:
                    # Caller will retry the rest one channel at a time
                    return "partial", attempt
                # Nothing came back; do not call this verified success
                shutil.rmtree(folder, ignore_errors=True)
                # Suspected missing data
                return "missing", attempt
            # Transient network / HTTP / Drive: bounded exponential backoff
            if is_retryable_error(download_error) and attempt <= retries:
                # Wait, then try the same request again
                time.sleep(retry_delay(attempt, retry_wait))
                # Next attempt
                continue
            # Tiny truncated ZIP: empty range, not a flaky download
            if is_unzip_error(download_error) and payload <= EMPTY_RESPONSE_MAX_BYTES and not have:
                # Remove the empty request folder
                shutil.rmtree(folder, ignore_errors=True)
                # Record as missing; caller must not split
                return "missing", attempt
            # Substantial ZIP corruption, with or without recovered files
            if is_unzip_error(download_error):
                # Some channels are usable
                if have:
                    # Keep them; caller retries the rest
                    return "partial", attempt
                # Nothing valid; remove the folder and let the caller split
                shutil.rmtree(folder, ignore_errors=True)
                # Split may recover a shorter range
                return "truncated", attempt
            # Retries exhausted, or an unknown error
            if have:
                # Keep recovered files
                return "partial", attempt
            # Nothing to keep
            shutil.rmtree(folder, ignore_errors=True)
            # Caller may still split
            return "failed", attempt
        finally:
            # Always drop staging, including on return
            shutil.rmtree(staging, ignore_errors=True)
    # Loop ended without a return (retries of a retryable error)
    shutil.rmtree(folder, ignore_errors=True)
    # Transient failure
    return "failed", attempts


def download_channel_range(client, output_dir, meter, data_type, channel, start, end, retries, retry_wait, min_days):
    """
    Download one channel over a date range, splitting substantial truncated
    or failed ranges in half until they are min_days long or less. Empty
    ZIP responses are recorded as missing and are not split.
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
        _say(f"  skip {meter} {label} (_failed)")
        # Do not retry or split again
        return
    # Never write into a folder that exists without a DONE marker
    if os.path.exists(folder):
        # Already has a validated CSV for this channel
        if channel in present_channels(folder, meter, [channel]):
            # Nothing more
            return
        # Report and leave the folder alone
        _say(f"  {meter} {label} exists without DONE, skipped")
        # Record in the log
        append_log(output_dir, [meter, data_type, label, "skipped_existing", 0, datetime.now().isoformat(timespec="seconds")])
        # Stop if the range is too short to split
        if (end - start).days + 1 <= min_days:
            # Give up on this piece
            return
        # Shorter folders have different names, so they can still recover
        (a_start, a_end), (b_start, b_end) = split_range(start, end)
        # First half
        download_channel_range(client, output_dir, meter, data_type, channel, a_start, a_end, retries, retry_wait, min_days)
        # Second half
        download_channel_range(client, output_dir, meter, data_type, channel, b_start, b_end, retries, retry_wait, min_days)
        # Do not request this exact folder
        return
    # Progress line
    _say(f"  {meter} {label}")
    # Make the request
    status, attempts = request_range(client, data_type, meter, [channel], start, end, folder, retries, retry_wait)
    # Log the attempt
    append_log(output_dir, [meter, data_type, label, status, attempts, datetime.now().isoformat(timespec="seconds")])
    # Success, done
    if status == "ok":
        # Nothing more
        return
    # Empty ZIP: suspected missing data; skip this exact request on rerun
    if status == "missing":
        # Permanent skip for this label only
        mark_failed(meter_dir, label)
        # Do not split
        return
    # Stop splitting once the range is short enough
    if (end - start).days + 1 <= min_days:
        # Transient failures stay unmarked so a later run can retry
        return
    # Substantial truncation or a failed transfer: shorter ranges may work
    (a_start, a_end), (b_start, b_end) = split_range(start, end)
    # First half
    download_channel_range(client, output_dir, meter, data_type, channel, a_start, a_end, retries, retry_wait, min_days)
    # Second half
    download_channel_range(client, output_dir, meter, data_type, channel, b_start, b_end, retries, retry_wait, min_days)


def download_meter_month(client, output_dir, meter, data_type, channels, start, end, retries, retry_wait, min_days):
    """
    Download one meter for one month: all channels together first, keep any
    validated CSVs from a truncated ZIP, then fetch missing channels one at
    a time with range splitting.
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
        _say(f"  skip {meter} {label} (DONE)")
        # Nothing to do
        return
    # Channels already on disk with a complete CSV
    have = present_channels(folder, meter, channels)
    # Channels still needed
    need = [channel for channel in channels if channel not in have]
    # Folder already has every channel: finish the DONE marker and stop
    if not need:
        # Complete
        mark_done(folder)
        # Nothing more
        return
    # Try all channels together unless the folder already exists or failed
    if not os.path.exists(folder) and not is_failed(meter_dir, label):
        # Progress line
        _say(f"  {meter} {label} ({len(channels)} channels)")
        # Make the request for all channels
        status, attempts = request_range(client, data_type, meter, channels, start, end, folder, retries, retry_wait)
        # Log the attempt
        append_log(output_dir, [meter, data_type, label, status, attempts, datetime.now().isoformat(timespec="seconds")])
        # Success, done
        if status == "ok":
            # Nothing more
            return
        # Skip redoing a large truncated or empty combined request; still
        # fetch missing channels. Transient failures stay unmarked so a
        # later run can retry the combined request.
        if status != "failed":
            # missing, partial, or truncated
            mark_failed(meter_dir, label)
        # Recompute from whatever the ZIP left behind
        have = present_channels(folder, meter, channels)
        # Channels still missing
        need = [channel for channel in channels if channel not in have]
        # Truncated ZIP that still yielded every channel
        if not need:
            # Treat as complete
            mark_done(folder)
            # Nothing more
            return
    elif is_failed(meter_dir, label):
        # Say so
        _say(f"  skip {meter} {label} (_failed)")
    else:
        # Pre-existing folder: never write into it; still fetch missing channels
        _say(f"  {meter} {label} exists without DONE, skipped")
        # Record in the log
        append_log(output_dir, [meter, data_type, label, "skipped_existing", 0, datetime.now().isoformat(timespec="seconds")])
    # Fall back to one channel at a time for whatever is still missing
    for channel in need:
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
            _say(f"meter {meter}")
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
    _say(f"{len(jobs)} meter-month jobs, {workers} threads")
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
                _say(f"  job failed: {type(error).__name__}: {error!r}")
                # Keep it to raise after the pool drains
                errors.append(error)
    # Surface the first unexpected error after the other jobs have finished
    if errors:
        # Raise so the caller sees that something besides a request failure happened
        raise errors[0]
    # Scan once at the end; DONE markers already make a crash restartable
    return write_missing_channels(output_dir, meters)