# Optical flow from YAML: `aurora flow run`

*AuroralProcessing · 2026-11-17*

Flow jobs work like pipeline jobs:
- You write **flow settings files** (reusable recipes, like chain files).
- You write **flow job files**, which say which inputs to run with which flows, what to sweep, and where the results go.
- You run `aurora flow run job.yaml`.

Every run records **warp_mae, zero_mae and fb_error**. One `summary.csv` compares every input × flow setting.

---

## 1. Quick start

```zsh
conda activate sp
aurora flow run flow_jobs/example_chain_compare.yaml -n     # list the runs and where they go
aurora flow run flow_jobs/example_chain_compare.yaml        # run them
```

The example runs TV-L1 with default settings on every pipeline output of the 2-minute test clip (first 100 pairs). Results:

| | Path |
|---|---|
| **IN** | `optical_flow/processed/2013-03-27-CamSer7196_015400-015600/*/frames.h5` |
| **OUT** (job folder) | `optical_flow/flow/example_chain_compare/`: `summary.csv`, `runs.csv`, `runs/`, `job.yaml` |
| **OUT** (each run) | `optical_flow/flow/example_chain_compare/2013-03-27-CamSer7196_015400-015600/<output>/tvl1_default/flow.h5` |

---

## 2. Flow settings files (`flows/*.yaml`)

A flow settings file is one recipe: the method, its settings, and how frames are paired. The file name is its label, which is used in run names.

```yaml
# flows/tvl1_default.yaml
method: tvl1              # tvl1 | piv | flct
params:                   # any setting from `aurora flow compute -h`, without the --tvl1- prefix
  lambda: 0.15
  gamma: 0
gap: 1                    # frames between the two images of a pair
```

| Key | Default | Meaning |
|---|---|---|
| `method` | `tvl1` | `tvl1`, `piv` or `flct` |
| `params` | the method's defaults | Settings named as in the flags: `--tvl1-scale-step` → `scale_step`, `--piv-first-window` → `first_window` |
| `gap` | 1 | Frames between the two images of a pair |
| `stride` | = gap | Frames between successive pairs |
| `warm_start` | false | Start each pair from the previous pair's flow (TV-L1 only) |
| `clip` | `[0.5, 99.5]` | Percentiles for the global input scaling |
| `input_dtype` | `float32` | `uint8` reproduces the old prototype |

- **Typos are errors:** an unknown key or setting stops the job before anything runs. For example, `lamda` gives *unknown TV-L1 setting 'lamda'*.
- **Writing a flow inline:** you can put a flow directly in the job file instead of a separate file. Give it a `label:`.

---

## 3. Flow job files (`flow_jobs/*.yaml`)

```yaml
# flow_jobs/chain_compare.yaml          the job's name is the file name: chain_compare
output: ../flow                         # where the job folder goes (default optical_flow/flow)
inputs:                                 # files, folders, globs: pipeline outputs or raw sources
  - ../filtered/2013-04-14-CamSer1387_extract/*/frames.h5
flows:
  - ../flows/tvl1_default.yaml          # a settings file
  - label: piv16                        # or inline
    method: piv
    params: {window: 16}
grid:                                   # optional sweep
  lambda: [0.05, 0.15, 0.3]
metrics: [warp_mae, zero_mae, fb_error] # the default
workers: 4
video: [quiver]                         # optional: quiver and/or speed
score: false                            # also run `aurora flow score`
```

| Key | Default | Meaning |
|---|---|---|
| `inputs` | (required) | Same rules as pipeline `sources:`: a file, a folder (its `*.h5`), a glob, or `{dir:, glob:}` |
| `flows` | (required) | Settings files and/or inline flows |
| `output` | `optical_flow/flow` | Parent of the job folder |
| `frames`, `times` | `auto` | `auto` picks `frames`/`frames_ut1_unix` in pipeline outputs and `rawimg`/`ut1_unix` otherwise, so you don't pass dataset names |
| `start`, `stop` | whole file | Frame range, e.g. `stop: 101` for the first 100 pairs |
| `grid` | none | Settings to sweep (§5) |
| `metrics` | all three | `fb_error` doubles the run time. Drop it with `metrics: [warp_mae, zero_mae]`. |
| `workers` | 1 | Parallel processes per run |
| `video`, `video_options` | none | Videos per run, saved in the job's `videos/` folder, and options for them, e.g. `video_options: {grid: 32, fps: 10, vmax: 3}` |
| `score` | false | Run `aurora flow score` on all runs afterwards |

Relative paths are relative to the job file's folder.

---

## 4. Where results go

Everything goes in the job folder, `<output>/<job>/`. Nothing is written next to your inputs.

```
flow/chain_compare/                              <- the job folder
  job.yaml       copy of the job file (plus any command-line --set/--grid)
  runs.csv       status of each run from the last `aurora flow run`
  summary.csv    one row per run (below)
  runs/          a link to every run: <output>-<run>
  videos/        <output>-<run>_quiver.mp4, <output>-<run>_speed.mp4 (with `video:`)
  2013-04-14-CamSer1387_extract/                 <- the source
    norm_only-4427dcf1/                          <- the pipeline output
      tvl1_default_lambda0.05/flow.h5
      tvl1_default_lambda0.15/flow.h5
      piv16/flow.h5
    full_stage_bin_size10-0bc12f19/
      ...
```

- **Pipeline outputs:** `<job folder>/<source>/<output>/<run>/flow.h5`.
- **Other inputs** (not pipeline outputs): `<job folder>/<file name>/raw/<run>/flow.h5`.
- **Run name:** the flow's label plus each grid value, e.g. `tvl1_default_lambda0.05`.
- **Videos:** all in `videos/`, named like the `runs/` links. A video found in a run folder (where they used to go) is moved there on the next run rather than made again.
- **To delete a job, delete its folder.** Your pipeline outputs aren't affected. Deleting a pipeline output doesn't delete its flow, but its chain columns in `summary.csv` will be empty, because the chain is read from the output.
- **Two inputs with the same source and output names** (for example, copies in two folders) would share run folders. That's an error, so put them in separate jobs.

---

## 5. Sweeps, overrides, dry runs

**Grids.** A grid key applies to **every flow whose method has that setting**: `lambda` applies to TV-L1 flows and is ignored by PIV. `gap` and `stride` apply to all flows. When `gap` is swept, all gaps use the same first frames, as in `aurora flow sweep`.

```yaml
grid:
  lambda: [0.05, 0.15]
  gap: [1, 2, 4]
```

**From the command line:**

```zsh
aurora flow run JOB.yaml -n                          # list runs, folders and status; computes nothing
aurora flow run JOB.yaml --grid lambda=0.05,0.15     # add to the job's grid
aurora flow run JOB.yaml --set gap=2                 # change a setting in every flow that has it
aurora flow run JOB.yaml -j 8                        # more workers
aurora flow run JOB.yaml --score                     # also score
aurora flow run JOB.yaml --force                     # recompute runs that exist
```

**Without a job file:** give inputs and a settings file. The job is named after the flow, unless you pass `--job-name`.

```zsh
aurora flow run filtered/clip/*/frames.h5 -f flows/tvl1_default.yaml -o flow --job-name quick_look
```

**Typo protection:** a `--set` or grid name that no flow has is an error, e.g. *--set ['lamda']: no flow has that setting*.

---

## 6. Rerunning and changing settings

| Situation | What happens |
|---|---|
| Stop a job | Press Ctrl+C once, even with several workers. Saved pairs are kept and the file is closed properly. Pairs that workers were still computing are lost. Don't `kill` the process, because that can leave `flow.h5` unreadable. |
| Rerun the same job | Finished runs say `exists` and are skipped; interrupted runs resume where they stopped |
| You edited the job so a run folder would get different settings | That run fails with `exists with different settings`, and the old result is kept. `-n` shows `exists, different settings`. |
| You want the new settings in the same folders | `--force` recomputes those runs |
| You want to keep both | Copy the job file under a new name; a new job gets new folders |

Adding `fb_error` later counts as a different setting. Decide on the metrics before a long run.

---

## 7. Reading `summary.csv`

There's one row per run. The first columns are the ones you compare:

| Column | Meaning |
|---|---|
| `run` | `<output>-<run>` |
| `status` | `complete`, `partial` or `missing` |
| `output`, `chain` | The pipeline output and its chain, e.g. `normalize > binned_temporal_average(bin_size=10) > …` |
| **`warp_over_zero`** | mean warp_mae / mean zero_mae. **The fit score: lower is better.** |
| `warp_mae_mean`, `zero_mae_mean`, `fb_error_mean` | Means over pairs (medians are also included) |
| `<stage>.<setting>`, `<stage>.on` | Each pre-processing setting, e.g. `binned_temporal_average.bin_size`, `wiener.on` |
| `method`, `lambda`, `gap`, … | The flow settings |
| `dt` | Seconds per pair |

Example from two `bin10` outputs of the test clip (first 10 pairs):

| run | wiener.on | warp_over_zero | warp_mae | zero_mae | fb_error |
|---|---|---|---|---|---|
| …bin_size10_nowiener…-tvl1_default | False | 0.692 | 0.0434 | 0.0628 | 0.91 px |
| …bin_size10_wiener…-tvl1_default | True | 0.721 | 0.0318 | 0.0441 | 0.88 px |

The Wiener filter lowers both residuals, because the frames are less noisy. But the flow explains a slightly smaller share of the change (0.72 vs 0.69), and the forward and backward flows agree slightly better (fb_error 0.88 vs 0.91). The *Flow Quality Metrics* guide explains how to weigh these.

In Python:

```python
import pandas as pd
s = pd.read_csv("flow/chain_compare/summary.csv")
s.sort_values("warp_over_zero")[["output", "lambda", "warp_over_zero", "fb_error_mean"]]
```

**Scoring.** For region-by-region scores (lanes vs diffuse glow, speeds, star pinning), set `score: true` or pass `--score`. This writes `scores.csv` and `scores_heatmaps.png` into the job folder. Afterwards you can also run `aurora flow score <job folder>/runs`.

---

## 8. Recipes

**Compare all your pre-processing chains with one flow:**

```yaml
inputs: [../filtered/2013-04-14-CamSer1387_extract/*/frames.h5]
flows: [../flows/tvl1_default.yaml]
```

**Tune TV-L1 on the chain you picked:**

```yaml
inputs: [../filtered/2013-04-14-CamSer1387_extract/full_stage_bin_size10-0bc12f19/frames.h5]
flows: [../flows/tvl1_default.yaml]
grid: {lambda: [0.05, 0.1, 0.15, 0.3], gamma: [0, 1]}
```

**TV-L1 against PIV on the same frames:**

```yaml
inputs: [../filtered/2013-04-14-CamSer1387_extract/norm_only-4427dcf1/frames.h5]
flows: [../flows/tvl1_default.yaml, ../flows/piv16.yaml]
```

**Quick test on a few pairs before a long run:** add `stop: 11` (10 pairs), check `summary.csv`, then remove it and run again. The short runs' folders are refused as "different settings", so either use `--force` or rename the job.
