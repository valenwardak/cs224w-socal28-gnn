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
4. Physics baseline: `main_physics.py` runs a least-squares state estimator with no learning over any window or split and writes a residual per ten-second slot to `data/processed/physics/`.
5. Feature tables: `main_features.py` exports the processed tables as compact monthly files in `data/features/` for training on Colab.
6. Learned models: GRU baseline and the GNNs, trained on Colab from `data/features/`, `topology/`, and `splits.json` (next stage).

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

Rerunning skips months already built. `--force` rebuilds. Options: `python main_preprocess.py --help`. The full run takes from twenty minutes to an hour and produces about 25 GB.

Output under `data/processed/`:

| File | Content |
|------|---------|
| `<YYYY-MM>/<meter>.parquet` | One file per meter and month. Index: ten-second UTC grid for the whole month. Columns `<meter>\|<channel>\|<value>` as float32 for values `rms, freq, ang0, mag0, ang1, mag1, ang2, mag2`, `mask\|<meter>\|<channel>` as boolean, True where a reading exists, and one column `<meter>\|toff` as float32: seconds from the slot time to the meter's exact capture time. Gaps are never filled. |
| `meters.json` | For each meter: the bus its voltages are measured on, the elements its currents are measured on, the channel-to-phase map from the network file, and whether L3 exists. |
| `events.csv` | Every logged device change with time in standard time and UTC, device, from and to status, day, episode id, and a scored flag. |
| `coverage.csv` | Per month, meter, channel: slots, slots with a reading, fraction. |

Read a table with `pandas.read_parquet`. Drop the columns you do not need; the harmonics 1 and 2 columns are kept so nobody has to rerun preprocessing to test them.

Phase angles are measured relative to each meter's own capture time. Captures on different meters start a few milliseconds apart, and at 60 Hz one millisecond is 21.6 degrees, so angles are only comparable across meters after each is shifted to the slot time: `angle_at_slot = angle - 360 * freq * toff` in degrees. Angles within one meter share one capture time and need no shift.

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

## Step 6: physics baseline

A least-squares state estimator, following Caltech's bus injection model in their `code_examples/state_estimation_phasor.py`. For every ten-second slot it takes the measured voltage and current phasors and the admittance matrix of a fixed reference wiring (the normal switch state), solves for the bus voltages that best fit the circuit equations, and reports the residual: the leftover gap between the readings and the wiring, averaged over readings and expressed in percent of nominal. When a switch changes state the readings stop fitting the fixed wiring and the residual shifts. No training.

Needs `caltech_utils/`, which is Caltech's `utils` folder copied unchanged into this repo. Their `global_param.py` expects that folder to be named `utils/` and the topology to sit under `sample_dataset/`; `socal_physics.py` redirects those paths at startup, so their files are not edited.

The reference wiring, the connected component to keep, and the map from meter currents to bus injections (with the clamp sign for each) are in `physics_config.json`. The map and signs are copied from Caltech's example and cover the A-side circuit, meters 3, 7, 9, 11, 13, 15, 19, 21. Phase angles are shifted to the slot time with `toff` before use. Every angle is shifted with the meter's voltage frequency, and a reading whose own frequency estimate differs from it by more than `--freq-tolerance` (default 0.01 Hz, the same FFT bin) is dropped from that slot's equations rather than used or read as zero, for the reason given in Step 7.

First check, Caltech's normal hour (paper Section VI-A reports 0.851 percent; this code gives 0.55 percent on the same hour with two more voltage meters):

```bash
python main_physics.py --start 2024-11-14T15:00:00 --end 2024-11-14T16:00:00
```

A whole split:

```bash
python main_physics.py --split test_events
```

Options: `python main_physics.py --help`. `--solver cvxpy` (default) matches Caltech's objective, about 200 slots per second. `--solver numpy` is the plain least-squares version, much faster, for long windows.

Output under `data/processed/physics/`, one pair of files per block:

| File | Content |
|------|---------|
| `<label>.parquet` | Per slot: `residual` (objective value), `residual_mean` (per reading, fraction), `n_voltage`, `n_current`, `status` (`ok`, `no_data`, `too_few`, `solver_failed`). Slots without readings are marked, never scored as zero. |
| `<label>.json` | Summary: scored slots, mean, median, and 99th percentile residual in percent, status counts, options used. |

`checks/main_physics_events.py` lines the residual up against `events.csv` and prints the level before, during, and after each episode.

What it finds on the event blocks: the November 13 to 14, 2024 episode (swmp_17 and cb_121, which change the feed of the whole A-side) raises the residual from 0.55 percent to 20 to 27 percent for the five and a half hours the switch is open, and the restoration brings it back within minutes. The 2023 episodes are building transfers (one breaker opens, the tie breaker to the B-side closes) and leave the A-side residual flat: to the A-side model a transferred building looks like a building that switched off, since its breaker current drops to zero and its bus voltage stays at the same campus level. Detecting transfers from physics would need the B-side circuit in the model as well, which is a later extension.

## Step 7: feature tables for training

Training runs on Google Colab, so the data has to fit on Google Drive and load fast. `main_features.py` reads the Parquet tables and writes one compressed file per month holding only what the models use, in fixed-order dense arrays. The Parquet tables stay the authoritative data.

```bash
python main_features.py
```

Rerunning skips months already written. `--force` rebuilds. `--keep-harmonics` adds harmonics 1 and 2. Options: `python main_features.py --help`. The full run takes about an hour. A full month is about 130 MB, the whole period 2.9 GB.

Output under `data/features/`:

| File | Content |
|------|---------|
| `<YYYY-MM>.npz` | Arrays for the month. `t` int64 (T,) slot start in seconds since 1970 UTC. `rms` float16 (T, M, C) magnitude in volts or amps. `ang` float32 (T, M, C) fundamental angle in degrees at the slot time, NaN where rejected. `freq` float32 (T, M) hertz, from the meter's voltage channels. `mask` bool (T, M, C) True where a reading exists. `angle_ok` bool (T, M, C) True where the angle passed the checks. With `--keep-harmonics` also `mag1, ang1, mag2, ang2`. T is slots in the month, M the 22 meters in `meters.txt` order, C the six channels L1 L2 L3 S1 S2 S3. |
| `manifest.json` | Meter order, channel order, array names, units, dtypes, the alignment rule, and the format version. Read it instead of guessing. |
| `build_log.csv` | Per month: slots, meters with files, size, spot-check errors against the Parquet, frequency disagreements, share of readings with a usable angle. |

Precision: magnitudes are float16, about three and a half significant digits, so a 9,526 volt reading is kept to within 4 volts. Angles and frequency are float32. Convert to float32 before arithmetic. Models should turn angles into cos and sin, since degrees wrap at 180.

Angle validity: Caltech's frequency estimate takes the strongest peak in each channel's spectrum. On current channels carrying only a few amps that peak can land a bin or two away from 60 Hz, and the angle reported there is the phase of the wrong component. The export aligns every angle with the meter's voltage frequency, accepts it only when the channel's own frequency agrees within 0.01 Hz (the same FFT bin), and stores rejected angles as NaN with `angle_ok` False. The rms and the presence mask are untouched. From October 2024 on, 98 percent of readings keep their angle. From August 2023 to July 2024 only 78 to 84 percent do, so the first year has far more small-current channels, which is noted but not yet explained. `angle_ok` means passed the checks, not certified accurate.

For Colab: upload `data/features/` to Google Drive once, and at the start of each session copy the month files to the local disk before training. Reading training windows through the Drive mount is slow.

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
- Captures land about 36 milliseconds after each ten-second boundary on a typical meter (median `toff`), with rare captures up to a full slot late.
- Voltage channels always agree on frequency within a meter. Current channels carrying a few amps (egauge_6 at 4.5 A on 55 to 71 percent of slots, egauge_18 and egauge_20 on 2 to 10 percent, egauge_16 rarely) report a frequency 1 to 4 Hz off, a wrong FFT bin, and their angles on those slots are not usable (`main_freqcheck.py`, September 2024).

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
| `main_physics.py` | Physics baseline main script |
| `socal_physics.py` | Physics baseline helper functions |
| `physics_config.json` | Reference wiring, kept component, injection map and signs for the physics baseline |
| `caltech_utils/` | Caltech's `utils` folder, unchanged: graph and admittance code |
| `main_features.py` | Feature export main script |
| `socal_features.py` | Feature export helper functions |
| `merge_years.sh` | Merges two per-year download folders into one |
| `meters.txt` | The 22 meters, one per line |
| `requirements.txt` | Python packages |
| `topology/` | Caltech's network description, breaker status files, change logs |
| `checks/` | One-off scripts that established the data facts above, plus `main_physics_events.py` and `main_freqcheck.py` |
| `notes/` | Data notes: gaps, outages, known issues. Read before analysis. |