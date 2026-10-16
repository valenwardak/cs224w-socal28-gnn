# CS224W Project: GNN Event Detection on a Real Electric Grid

Stanford CS224W, Fall 2026, Project Type 1 (Real-World Applications of GNNs)

**Authors:** Valen Wardak and Landon Choy

## What this project does

We test whether a graph neural network (GNN) that knows the wiring of a real distribution grid detects switching events better than the same model without the graph. The data is the SoCal 28-Bus dataset from Caltech: 28 buses, 22 metered, ten-second synchrophasor readings, time-varying topology, and operator-logged switch and breaker events used only for evaluation. The proposal PDF is in this repo.

## Dataset

- Paper: https://arxiv.org/abs/2504.06588
- IEEE DataPort page: https://dx.doi.org/10.21227/hsfq-fq78
- Caltech repo with the API client: https://github.com/caltech-netlab/digital-twin-dataset

The data is not stored in this repo. Each team member downloads it with the scripts here, which produce identical files and folder layout for everyone.

## Step 1: get access (once per person)

Access is granted per person and tied to your GitHub account.

1. Submit the request form: https://forms.office.com/r/Ds6rKEtyTV
2. Use your own name and your own GitHub username.
3. Wait for approval. It has taken from a few hours to a few days.

## Step 2: install

Open a terminal and run:

```bash
git clone https://github.com/valenwardak/cs224w-socal28-gnn.git
cd cs224w-socal28-gnn
pip install -r requirements.txt
```

`dataset_api_client.py` is Caltech's client, copied unchanged from their repo. Python 3.10 or newer is required (the client uses the `X | None` type syntax).

## Step 3: run the download

### First run

From inside the repo folder:

```bash
python main_socal.py
```

What happens:

1. The script reads `meters.txt` (22 meters).
2. It creates the Caltech client. On the first run there is no credentials file, so the client prints a code, opens GitHub in your browser, and asks you to enter the code. Approve it. The client saves `dataset_api_credentials.json` next to the scripts and prints `Signed in as: <your GitHub username>`.
3. It downloads phasors for every meter, one meter and one month at a time, from August 1, 2023 through September 3, 2025, into `data/`.
4. It prints one line per request, for example `egauge_1 2024-09-01_2024-09-30 (6 channels)`, and `skip ... (DONE)` for anything already complete.
5. When finished it prints how many meters have missing channels and writes `data/missing_channels.json` and `data/download_log.csv`.

Never commit `dataset_api_credentials.json`. It is in `.gitignore`.

### Stopping and resuming

You can stop the script at any time (Ctrl+C). Run the same command again and it skips every folder that has a `DONE` marker and continues where it left off. Only a complete, successful request gets a `DONE` marker, so a stopped run never leaves half-written data.

### Options

Every setting has a project default and can be changed on the command line. See them all with:

```bash
python main_socal.py --help
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--start-date` | `2023-08-01` | First day to download, inclusive, YYYY-MM-DD |
| `--end-date` | `2025-09-03` | Last day to download, inclusive, YYYY-MM-DD |
| `--output-dir` | `data` | Folder for downloaded data |
| `--meters-file` | `meters.txt` | Text file with one meter name per line |
| `--data-type` | `phasors` | One of `phasors`, `magnitudes`, `waveforms` |
| `--channels` | `L1 L2 L3 S1 S2 S3` | Channel suffixes per meter |
| `--credentials-file` | `dataset_api_credentials.json` | Where the Caltech login tokens are stored |
| `--retries` | `2` | Retries per request before it counts as failed |
| `--retry-wait` | `30` | Seconds to wait between retries |
| `--min-days` | `1` | Stop splitting a failed range at this many days |

Examples:

```bash
# Only the recommended final year
python main_socal.py --start-date 2024-09-01 --end-date 2025-09-03

# One meter, for testing (make a file test_meters.txt with one line: egauge_1)
python main_socal.py --meters-file test_meters.txt --end-date 2023-08-31

# Magnitude data instead of phasors (magnitude current channels are named CT1 to CT3)
python main_socal.py --data-type magnitudes --channels L1 L2 L3 CT1 CT2 CT3 --output-dir data_magnitudes

# Be more patient with a slow server
python main_socal.py --retries 4 --retry-wait 120
```

### How failures are handled

For each meter and month the script:

1. Requests the whole month with all channels together.
2. If that fails after the retries, requests each channel alone for the whole month.
3. If a channel fails, splits the date range in half and tries each half, repeating down to a single day.

Every failed attempt leaves an empty marker folder under `<meter>/_failed/`. Every successful request gets its own folder with a `DONE` marker. All of this is recorded in `data/download_log.csv`.

### Checking the result

After a run, open `data/missing_channels.json`. It lists, per meter and per folder, any channel that was requested but never came back from the server. An empty `{}` means everything requested was delivered. Then compare your `data/` tree with a teammate's: the folder names and file names should match exactly.

### Folder layout produced