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

## Pipeline

1. Download: `main_socal.py` pulls raw phasor CSVs from the Caltech API into `data/`.
2. Preprocess: `main_preprocess.py` turns the raw files into aligned, masked Parquet tables in `data/processed/`, plus the event table and a coverage report.
3. Split: `main_splits.py` writes `splits.json`, the fixed train / validation / test blocks that every later script reads.
4. Graph and models: built from `data/processed/`, `topology/`, and `splits.json` (next stage).

## Step 1: get access (once per person)

Access is granted per person and tied to your GitHub account.

1. Submit the request form: https://forms.office.com/r/Ds6rKEtyTV
2. Use your own name and your own GitHub username.
3. Wait for approval. It has taken from a few hours to a few days.

## Step 2: install

```bash
git clone https://github.com/valenwardak/cs224w-socal28-gnn.git
cd cs224w-socal28-gnn
pip install -r requirements.txt
```

`dataset_api_client.py` is Caltech's client, copied unchanged from their repo. Python 3.10 or newer is required.

## Step 3: download the raw data

From inside the repo folder:

```bash
python main_socal.py
```

The first run opens a browser to sign in with GitHub and saves `dataset_api_credentials.json` next to the scripts. Never commit that file (it is in `.gitignore`).

Defaults: phasors for the 22 meters in `meters.txt` (egauge_3 through egauge_24), August 1, 2023 through September 3, 2025, into `data/`. The script requests one meter and one month at a time. If a request fails it retries, then falls back to one channel at a time, then splits the date range in half down to a single day. Rerunning skips anything already marked `DONE`. A folder that exists without `DONE` is never touched.

Options: `python main_socal.py --help`

| Option | Default | Meaning |
|--------|---------|---------|
| `--start-date` | `2023-08-01` | First day, inclusive |
| `--end-date` | `2025-09-03` | Last day, inclusive |
| `--output-dir` | `data` | Folder for downloaded data |
| `--meters-file` | `meters.txt` | One meter name per line |
| `--data-type` | `phasors` | `phasors`, `magnitudes`, or `waveforms` |
| `--channels` | `L1 L2 L3 S1 S2 S3` | Channel suffixes per meter |
| `--retries` | `2` | Retries per request |
| `--retry-wait` | `30` | Seconds between retries |
| `--min-days` | `1` | Stop splitting a failed range at this many days |

### Raw folder layout

```
data/
  egauge_3/
    2024-09-01_2024-09-30/        one month, all channels
      request.json
      phasors/
        t.csv
        egauge_3-L1.csv ...       one file per channel
      DONE
    2025-07-01_2025-07-31_L1/     fallback: one channel for the month
    _failed/                      markers for failed attempts
  ...
  missing_channels.json
  download_log.csv
```

Channels: L1, L2, L3 are the three voltage phases. S1, S2, S3 are the three current phases. egauge_17 and egauge_18 have no L3. Each CSV has columns `t, t_zero_crossing, frequency, rms`, then angle and magnitude for harmonics 0, 1, 2.

## Step 4: preprocess

Needs the `topology/` folder, which is Caltech's `sample_dataset/topology` copied into this repo (network description, breaker status files, change logs). It is committed, so a clone already has it.

```bash
python main_preprocess.py
```

Rerunning skips months already built. `--force` rebuilds. Options: `python main_preprocess.py --help`. The full run takes about an hour and produces about 25 GB.

Output under `data/processed/`:

| File | Content |
|------|---------|
| `<YYYY-MM>/<meter>.parquet` | One file per meter and month. Index: ten-second UTC grid for the whole month. Columns `<meter>\|<channel>\|<value>` as float32 for values `rms, freq, ang0, mag0, ang1, mag1, ang2, mag2`, and `mask\|<meter>\|<channel>` as boolean, True where a reading exists. Gaps are never filled. |
| `meters.json` | For each meter: the bus its voltages are measured on, the elements its currents are measured on, the channel-to-phase map from the network file, and whether L3 exists. |
| `events.csv` | Every logged device change with time in standard time and UTC, device, from and to status, day, episode id, and a scored flag. |
| `coverage.csv` | Per month, meter, channel: slots, slots with a reading, fraction. |

Read a table with `pandas.read_parquet`. Drop the columns you do not need; the harmonics 1 and 2 columns are kept so nobody has to rerun preprocessing to test them.

## Step 5: splits

```bash
python main_splits.py
```

Writes `splits.json` in the repo root and prints, per split, the days, coverage, events, and episodes it contains. `splits.json` is committed: it is the single definition of which dates are train, validation, and test, and every model and evaluation script reads it. Dates in it are inclusive calendar days in fixed Pacific Standard Time, with UTC bounds given for each block.

The split is by time blocks, fixed before any model is trained:

- test_events: each event day with a 2-day margin on both sides; November 26 to December 7, 2023 is one block.
- test_normal: April 1 to August 31, 2025, where false alarms per day are measured.
- validation: January 8 to 21, 2024 and February 3 to 16, 2025, for early stopping and conformal calibration.
- train: everything else from August 15, 2023 to September 3, 2025.

Train and validation contain no scored event. The event log is used only to define normal data and to score; the models never see a label.

## Data facts the code relies on

Established by the scripts in `checks/`, which can be rerun with `--data-dir ../data`, and by the coverage report:

- Meter timestamps are fixed Pacific Standard Time all year, no daylight saving shift (`main_dst.py`: 15 of 15 clock-change nights show normal row counts).
- All meter clocks agree to within one ten-second reading (`main_clocks.py`: 20 meters stop at 17:52:10 on November 13, 2024).
- The operator log uses the same time base as the meters (`main_clocks.py`: 14 meters stop at 14:04:10 on September 17, 2023, logged 14:04:17). UTC is therefore time plus 8 hours for readings and log alike.
- egauge_1 and egauge_2 are not in the network file and are excluded. egauge_23 and egauge_24 are in it and are downloaded.
- Meters present over time: egauge_3 and 4 report from September 2024 only; egauge_15 stops after September 2024; egauge_23 stops after June 2024. So 20 meters report in each year, not the same 20.
- Phase labels for egauge_13, 14, 15, 20 follow the network file, not the L1 L2 L3 order. `meters.json` carries the map.
- February 1, 2024 is logged at exactly midnight with nothing visible in the readings and is not scored. That leaves 32 changes on 5 days in about 7 episodes.
- September 26, 2023 shows large current steps one hour before the logged times on several meters. The logged times for that day are unverified against the readings.
- Site-wide outages: September 28 to October 27, 2023; July 18 to September 4, 2024; October 30 to November 6 and December 12 to 22, 2024. August 2025 has meter-specific gaps on egauge_4, 10, 11, 12, 14, 16.

## Files

| File | Purpose |
|------|---------|
| `main_socal.py` | Download main script |
| `socal_download.py` | Download helper functions |
| `dataset_api_client.py` | Caltech's API client, unchanged |
| `main_preprocess.py` | Preprocessing main script |
| `socal_preprocess.py` | Preprocessing helper functions |
| `main_splits.py` | Writes `splits.json` and reports each split |
| `splits.json` | The fixed train / validation / test blocks |
| `merge_years.sh` | Merges two per-year download folders into one |
| `meters.txt` | The 22 meters, one per line |
| `requirements.txt` | Python packages |
| `topology/` | Caltech's network description, breaker status files, change logs |
| `checks/` | One-off scripts that established the data facts above |
| `notes/` | Data notes: gaps, outages, known issues. Read before analysis. |