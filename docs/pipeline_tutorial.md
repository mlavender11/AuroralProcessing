# Pre-processing with YAML: tutorial

*AuroralProcessing — `aurora pipeline`*

This guide covers running pre-processing chains from YAML files: what the files look like, how to run them, how to sweep settings, where the results go, and how to find them again.

---

## 1. The idea in one minute

- A **source** is a HiST HDF5 file with raw frames (`rawimg`) and times (`ut1_unix`). It is **only ever read**.
- A **chain** is an ordered list of **stages** (normalize, bin, median, …). You write it once in a **chain file** (`chains/*.yaml`).
- Running a chain on a source writes one **output folder**:

  ```
  <output>/<source stem>/<label>-<hash8>/
      frames.h5      processed frames + times, and everything needed to reproduce them
      preview.mp4    optional
  ```

- `hash8` is computed from the chain's settings (with defaults filled in) and the source file name. So:
  - running the same chain on the same source again finds the existing folder and **skips it**, even under a different name (e.g. v4 with 15-frame bins *is* v2, so it is reported as the existing `v2-ac2b49cf`);
  - a different chain always gets a **different folder**, so nothing is ever overwritten;
  - moving the output folder or the data drive doesn't change the hash.
- A **job file** (`jobs/*.yaml`) says which sources to run with which chains, what to sweep, and where to put the results. It replaces editing `run_pipeline.py`.
- `index.csv` at the top of the output folder lists every output with its settings. It is rebuilt from the files, so it is never out of date. It replaces `experiments.xlsx`.

---

## 2. Setup

```bash
cd AuroralProcessing
pip install -e .          # now also installs pyyaml and scipy
aurora pipeline -h
```

The default output folder is `optical_flow/processed/` inside the repo (it is git-ignored). Chain and job files live in `chains/` and `jobs/` and *should* be committed: they are the record of what you ran.

---

## 3. Your first run

Reproduce the old **v4** on the test clip:

```bash
aurora pipeline run optical_flow/data/2013-03-27-CamSer7196_015400-015600.h5 --chain chains/v4.yaml
```

You'll see:

```
exists, skipping: optical_flow/processed/2013-03-27-CamSer7196_015400-015600/v4-ad2d874b
```

It's skipped because v4 was migrated into that folder, and `chains/v4.yaml` reproduces it exactly (checked pixel for pixel). Add `--force` to rebuild it anyway.

To see what a command *would* do without running anything, add `--dry-run` (`-n`):

```bash
aurora pipeline run optical_flow/data/2013-03-27-CamSer7196_015400-015600.h5 \
    --chain chains/v4.yaml --chain chains/v6.yaml --dry-run
```

---

## 4. Chain files

A chain file is a YAML list. Each item is one stage: `stage:` names it, and any other keys are its settings. Settings you leave out take their defaults.

```yaml
# chains/bin_med_wiener.yaml
# Starting point for new outputs: v6's order with the full 16-bit range.
- stage: normalize               # vmin/vmax fitted from the source if not given
- stage: binned_temporal_average
  bin_size: 10
- stage: spatial_median
  k: 3
- stage: wiener
- stage: to_uint16               # out_max defaults to 65535
```

Rules:

- **Order matters**: frames flow top to bottom.
- A stage whose settings are estimated from the data (`normalize` without `vmin`/`vmax`) **must be first**, because the estimate is made from the source frames. You'll get a clear error otherwise.
- **End with `to_uint16`** if you want previews and videos. The video code needs uint16. Without it, the frames are stored as float (nothing is truncated).
- Unknown stage names or settings are errors, so typos are caught before anything runs.

### Two optional keys on any stage

| key | meaning |
|---|---|
| `id: name` | A name for this stage, used by overrides when the same stage appears twice (see §6). |
| `enabled: false` | Leave the stage out. Same as deleting it, but handy for sweeps (`wiener: [on, off]`). |

### A label for the chain

The outputs of a chain file are named after the file: `chains/v4.yaml` gives `v4-<hash>`. To use a different name, write the chain as a mapping:

```yaml
label: wide_median
stages:
  - stage: normalize
  - stage: spatial_median
    k: 7
  - stage: to_uint16
```

Inline chains in a job file with no `label:` get a name built from their stages, e.g. `norm_bin30_med3_wiener`.

---

## 5. Stage reference

| stage | settings (default) | what it does |
|---|---|---|
| `normalize` | `vmin` (fitted), `vmax` (fitted), `curve` (`linear`; or `log`, `sqrt`) | Clips to [vmin, vmax] and maps to 0–1. Unset limits are the 1st/99th percentiles of the source's nonzero pixels, sampled at one frame per second. The fitted values are stored in the output (`chain_fitted`). `log` needs vmin > 0. |
| `binned_temporal_average` | `bin_size` (15) | Averages each run of `bin_size` frames into one, stamped with their mean time. Drops the partial last bin. Divides the frame rate by `bin_size` (the test clip is ~53 fps, so bin 10 gives ~0.19 s per frame). |
| `sliding_temporal_median` | `window` (5, odd) | Running median over `window` frames, stamped with the centre frame's time. Outputs N − window + 1 frames, so the frame rate is unchanged. (The old name `sliding_temporal_medial` is still accepted.) |
| `spatial_median` | `k` (3, odd ≥ 3) | k × k median filter. k = 3, 5 use OpenCV; larger k uses SciPy (slower). |
| `wiener` | `size` (`null` = 3), `noise` (`null`) | `scipy.signal.wiener`. `noise: null` re-estimates the noise on **every frame** (what v2–v6 used). A number fixes it. `noise: auto` estimates it once from the first 5 frames and uses it for all frames (consistent smoothing). The estimated value is stored in `chain_fitted`. |
| `to_uint16` | `low` (0.0), `high` (1.0), `out_max` (65535) | Maps [low, high] to 0..out_max as uint16. Use the default 65535 for full precision. v2–v6 used `out_max: 255`, i.e. 8-bit. |

---

## 6. Overrides: changing one setting without editing a file

An override is `<stage>.<setting>=<value>`. Each dot steps one level in:

```bash
# v4 with 15-frame bins instead of 30
aurora pipeline run SRC.h5 --chain chains/v4.yaml --set binned_temporal_average.bin_size=15

# v4 without the Wiener filter
aurora pipeline run SRC.h5 --chain chains/v4.yaml --set wiener=off
```

| key form | meaning | example |
|---|---|---|
| `<stage>.<setting>=V` | set a setting | `spatial_median.k=5` |
| `<stage>=on/off` | switch a stage on or off | `wiener=off` |
| `<stage>.enabled=on/off` | same as above | `wiener.enabled=false` |
| `<id>.<setting>=V` | for a stage with an `id:` | `post.k=7` |

Values are read the way YAML reads them: `15` is a number, `off`/`false` mean off, `null` means unset. Each value is then checked against the stage's setting type, so `bin_size=2.5` is an error.

When an override changes a chain that has an explicit label, the label gets a tag. `chains/v4.yaml --set spatial_median.k=5` is saved as `v4_k5-<hash>`, so it never looks like the real v4.

If the changed chain turns out to be one you've already run, the existing output is used instead. Try `chains/v4.yaml --set binned_temporal_average.bin_size=15 --dry-run`: that chain is exactly v2, so it reports `v2-ac2b49cf  exists`.

**Repeated stages.** If a chain has two `spatial_median` stages, `spatial_median.k=5` is ambiguous and is rejected. Give them ids:

```yaml
- stage: normalize
- stage: spatial_median
  id: pre
- stage: binned_temporal_average
  bin_size: 10
- stage: spatial_median
  id: post
  k: 5
- stage: to_uint16
```

Then use `--set post.k=7`.

---

## 7. Grids: sweeping settings

A grid tries every combination of the listed values. On the command line, each `--grid` takes comma-separated values:

```bash
aurora pipeline run SRC.h5 --chain chains/bin_med_wiener.yaml \
    --grid binned_temporal_average.bin_size=5,10,30 \
    --grid wiener=on,off \
    --dry-run
```

That's 3 × 2 = 6 outputs, named like `bin_med_wiener_bin_size5_nowiener-<hash>`. Remove `--dry-run` to run them.

How grids behave:

- A grid applies to **every chain that has that stage**. A chain without the stage runs once, unchanged. One grid can therefore cover several chains.
- A grid key that matches **no** chain is an error, which catches typos like `wienr`.
- Combinations that come out identical (e.g. different settings of a stage that's switched off) run only once.
- To sweep different settings on different chains, use separate job files.

---

## 8. Job files

A job file holds a whole batch: sources, chains, grid, output folder and preview. Paths inside it are **relative to the job file's folder**.

```yaml
# jobs/example_bin_sweep.yaml
# Bin size x Wiener on/off on the 2-minute test clip, with previews.
output: ../optical_flow/processed       # optional; default optical_flow/processed
sources:
  - ../optical_flow/data/2013-03-27-CamSer7196_015400-015600.h5
chains:
  - ../chains/bin_med_wiener.yaml
grid:
  binned_temporal_average.bin_size: [10, 30]
  wiener: [on, off]
preview: true
```

```bash
aurora pipeline run jobs/example_bin_sweep.yaml --dry-run
aurora pipeline run jobs/example_bin_sweep.yaml
```

This job has already been run; its four outputs are in `optical_flow/processed/`.

### Every key

| key | required | meaning |
|---|---|---|
| `sources` | yes | List of sources (see below). |
| `chains` | yes | List of chain files and/or inline chains. |
| `output` | no | Output folder. Default `optical_flow/processed`. |
| `grid` | no | `<key>: [values]`, as in §7. |
| `preview` | no | `true` to render `preview.mp4` for each output. |
| `frames`, `times` | no | Dataset names in the sources. Default `rawimg`, `ut1_unix`. |

### Sources: files, folders, patterns

```yaml
sources:
  - /Volumes/Elements/HiST/2013-03-27/2013-03-27-CamSer7196.h5    # one file
  - /Volumes/Elements/HiST/2013-03-28                              # every *.h5 in a folder
  - /Volumes/Elements/HiST/2013-04-*/*.h5                          # a glob pattern
  - dir: /Volumes/Elements/HiST/2013-03-29                         # a folder with a pattern
    glob: "*CamSer7196*.h5"
```

### Chains: files and inline

```yaml
chains:
  - ../chains/v4.yaml                       # a chain file (label: v4)
  - file: ../chains/v6.yaml                 # a chain file with a new label
    label: v6_again
  - label: bin10_nowiener                   # an inline chain
    stages:
      - stage: normalize
      - stage: binned_temporal_average
        bin_size: 10
      - stage: to_uint16
```

The job makes **every source × every chain × every grid combination**. Use `--dry-run` to check the count before a big run.

### Command-line options on top of a job file

Options override the job file without editing it:

```bash
aurora pipeline run jobs/example_bin_sweep.yaml --out /Volumes/Elements/processed   # different output folder
aurora pipeline run jobs/example_bin_sweep.yaml --grid spatial_median.k=3,5        # add to the grid
aurora pipeline run jobs/example_bin_sweep.yaml --chain chains/v6.yaml             # replace the chains
aurora pipeline run jobs/example_bin_sweep.yaml --preview --force                  # previews; rebuild everything
```

If one source fails (say, a corrupt file), the error is printed, the rest of the job keeps going, and the final table marks that row `failed: …`.

---

## 9. Command reference

```
aurora pipeline run TARGET [options]
    TARGET              a job file (.yaml/.yml), or a source: .h5 file, folder of .h5 files, or glob
    -c, --chain FILE    chain file (repeatable). Required with a source; replaces a job file's chains
    --set NAME=VALUE    override a setting in every chain that has the stage (repeatable)
    --grid NAME=V1,V2   sweep a setting (repeatable; adds to a job file's grid)
    --label NAME        name for the outputs (single chain only)
    -o, --out DIR       output folder
    --frames, --times   dataset names in the sources
    --preview           render preview.mp4 per output
    --force             rebuild outputs that already exist
    -n, --dry-run       list what would be made, run nothing
    -q, --quiet         no progress output

aurora pipeline list [DIR] [--all]
    rebuild DIR/index.csv and print a summary (--all: every column)
```

---

## 10. Finding outputs: `index.csv`

```bash
aurora pipeline list
```

This rebuilds `optical_flow/processed/index.csv` and prints one line per output: folder, frame count, seconds per frame (`dt`), dtype, whether a preview exists, when it was made, and the chain. The CSV also has:

- one column per setting, e.g. `binned_temporal_average.bin_size` and `normalize.vmin` (the fitted value);
- `<stage>.on` columns (True/False) for whether each stage is in the chain;
- `git_commit` (the code version that made it; `-dirty` means there were uncommitted changes);
- `migrated_from` for v2–v6.

Open it in Excel or pandas and sort or filter as you like. It's regenerated from the files, so there's nothing to keep in sync by hand.

---

## 11. What's inside an output

```
frames.h5
  frames            (N, H, W) uint16 (or float if the chain doesn't end in to_uint16)
  frames_ut1_unix   (N,) float64, seconds since 1970 (UT1)
  attrs:
    chain           JSON: the chain as specified, defaults filled in (this is what the hash uses)
    chain_fitted    JSON: the chain as run, with fitted values (normalize vmin/vmax, wiener auto noise)
    label, hash
    source          full path of the source when it was made
    source_name, source_frames, source_times
    git_commit, created, complete
    migrated_from   (only v2–v6)
```

In Python:

```python
from auroral_processing.pipeline.processing_chain import read_output_attrs
import h5py

out = "optical_flow/processed/2013-03-27-CamSer7196_015400-015600/v4-ad2d874b"
a = read_output_attrs(out)
print(a["chain_fitted"], a["n_frames"], a["dt"])
with h5py.File(f"{out}/frames.h5") as f:
    frames, t = f["frames"][:], f["frames_ut1_unix"][:]
```

A run writes `frames.partial.h5` and renames it only when it's finished, so a crash never leaves something that looks complete. Delete a leftover `frames.partial.h5` if you find one.

---

## 12. Running from Python or a notebook

```python
from auroral_processing.pipeline.processing_chain import run_chain
from auroral_processing.pipeline.config import load_chain, load_job, apply_overrides
from auroral_processing.pipeline.jobs import run_job

SRC = "optical_flow/data/2013-03-27-CamSer7196_015400-015600.h5"

# one chain
out = run_chain(SRC, load_chain("chains/v4.yaml").stages, "optical_flow/processed", label="v4", preview=True)

# a chain written in Python, with an override
chain = [{"stage": "normalize"}, {"stage": "binned_temporal_average", "bin_size": 30}, {"stage": "to_uint16"}]
run_chain(SRC, apply_overrides(chain, {"binned_temporal_average.bin_size": 15}), "optical_flow/processed")

# a whole job file; returns a table of what happened
table = run_job(load_job("jobs/example_bin_sweep.yaml"), dry_run=True)
```

The older `apply_processing(hdf_fn, chain, "v7")` still works and writes into the source file. It now refuses to overwrite an existing dataset unless you pass `overwrite=True`, and stores the chain as JSON.

---

## 13. Feeding outputs to optical flow

Every output is an ordinary HDF5 frame stack, so the flow commands work on it directly:

```bash
OUT=optical_flow/processed/2013-03-27-CamSer7196_015400-015600/bin_med_wiener_bin_size10_wiener-b04243cc
aurora flow sweep $OUT/frames.h5 $OUT/flow/gamma_lambda \
    --frames frames --times frames_ut1_unix \
    --tvl1-scale-step 0.5 --grid gamma=1,3 --grid lambda=0.15,0.2
```

Keeping flow sweeps inside the output folder (`$OUT/flow/<sweep name>`) means:

- `aurora flow compare` names each run after its output;
- `aurora flow score` adds the pre-processing settings as columns, and heatmaps can use them, e.g. `gamma` × `wiener.on`.

The companion document *Changes and Flow Next Steps* has the full flow workflow.

---

## 14. YAML gotchas

| write | not | why |
|---|---|---|
| `null` (or leave the key out) | `None` | YAML's empty value is `null`. (`None` is accepted for settings that allow it, but `null` is standard.) |
| `on` / `off`, `true` / `false` | `"on"` with quotes | Unquoted, these are booleans, which is what `enabled` and grid on/off want. |
| `1.0e-3` | | PyYAML reads `1e-3` as text. The pipeline converts it for you, but `1.0e-3` is a real number everywhere. |
| two-space indents, consistent | tabs | YAML forbids tabs. |
| `[10, 30]` or a `- ` list | `10, 30` | A bare `10, 30` is one string. |
| quotes around globs: `"*.h5"` | `*.h5` | A leading `*` means something else in YAML. |

---

## 15. Troubleshooting

| message | fix |
|---|---|
| `unknown stage name 'wienr'` | Check the spelling. The message lists the known stages. |
| `unknown settings ['size'] … allowed settings: ['k']` | That stage doesn't have that setting. See §5. |
| `stage 1 (normalize) estimates its settings from the source frames, so it must be the first stage` | Move `normalize` to the top, or give it `vmin` and `vmax`. |
| `grid ['wienr']: no chain has that stage` | Typo in a grid/--set key, or none of the chains has that stage. |
| `the chain has 2 'spatial_median' stages; give them an id` | See §6, repeated stages. |
| `no preview … frames are float64` | End the chain with `to_uint16`. |
| `source not found` / `no source files match` | Paths in a job file are relative to the job file, not to where you run the command. |
| Output wasn't rebuilt after changing code | Outputs are keyed by chain settings, not by code. Use `--force` after a code change that alters results. `git_commit` in `index.csv` shows which code made each output. |
