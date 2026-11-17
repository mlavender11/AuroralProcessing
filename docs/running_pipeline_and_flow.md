# Running the Pipeline and Optical Flow

*A practical guide with many option setups · AuroralProcessing, branch `pipeline-yaml`*

Every recipe below was run on this machine. Each has a **Reads / Writes** box: those are the input and output files and folders. Paths are relative to the repo folder, `~/Developer/AuroralProcessing`.

---

## 0. Where everything lives

```
AuroralProcessing/
├── chains/                           IN   chain files (stages) you write; committed to git
│   ├── v2.yaml … v6.yaml                  reproduce the old versions (8-bit)
│   └── bin_med_wiener.yaml                16-bit starting point
├── jobs/                             IN   job files (sources × chains × grid); committed
│   └── example_bin_sweep.yaml
├── optical_flow/
│   ├── data/                         IN   raw HiST source files (read only)
│   │   └── 2013-03-27-CamSer7196_015400-015600.h5      rawimg, ut1_unix
│   ├── processed/                    OUT  pipeline outputs (default; git-ignored)
│   │   ├── index.csv                      registry of every output
│   │   └── <source stem>/
│   │       └── <label>-<hash8>/           ONE pre-processing output
│   │           ├── frames.h5              frames, frames_ut1_unix   ← input to flow
│   │           ├── preview.mp4
│   │           ├── benchmark.h5           (if you build one)        ← input to flow
│   │           └── flow/                  OUT  flow sweeps of THIS output
│   │               └── <sweep>/
│   │                   ├── summary.csv, scores.csv, scores_heatmaps.png
│   │                   └── <run>/flow.h5, flow_quiver.mp4
│   ├── compare/                      OUT  cross-output comparisons (aurora flow compare)
│   │   └── <name>/  <output>-<sweep>-<run> → links, scores.csv, scores_heatmaps.png
│   │                (names that would collide get their parent folder added)
│   └── sweep/                             your older flow sweeps (on copy.h5:v4)
```

**The rule of thumb.** Sources are only read. The pipeline writes under `optical_flow/processed/` (or `--out`). Put flow results *inside* the output they were computed from (`<output>/flow/…`), and comparisons across outputs in `optical_flow/compare/`. All of these output locations are git-ignored.

| command | reads | writes (default location) |
|---|---|---|
| `aurora pipeline run` | source `.h5`, chain/job `.yaml` | `optical_flow/processed/<src>/<label>-<hash>/frames.h5` (+ `preview.mp4`) |
| `aurora pipeline list` | `optical_flow/processed/*/*/frames.h5` | `optical_flow/processed/index.csv` |
| `aurora flow compute` | any frame stack `.h5` | `<source stem>_flow.h5` **next to the source**, or `-o` |
| `aurora flow video` | a flow `.h5` (+ its source) | `<flow stem>_<kind>.mp4` next to the flow file, or `-o` |
| `aurora flow sweep` | frame stack `.h5` | `OUTDIR/<run>/flow.h5`, `flow_quiver.mp4`, `OUTDIR/summary.csv` |
| `aurora flow compare` | run folders | `OUTDIR/` of links |
| `aurora flow score` | a sweep or compare folder (+ the sources) | `scores.csv`, `scores_heatmaps.png` in that folder |
| `aurora flow benchmark` | frame stack `.h5` | the `-o` file (required) |
| `aurora flow repack` | flow `.h5` | rewrites it in place (gzip) |

---

## 1. Setup (once per terminal)

```bash
conda activate sp
cd ~/Developer/AuroralProcessing

# shorthands used throughout
SRC=optical_flow/data/2013-03-27-CamSer7196_015400-015600.h5
P=optical_flow/processed/2013-03-27-CamSer7196_015400-015600
FR=(--frames frames --times frames_ut1_unix)      # dataset names inside a pipeline output
```

> **zsh note.** Keep groups of flags in an **array**, `FR=(…)`, and write `"${FR[@]}"`. A plain string such as `FR="--frames frames"` is passed as *one* argument in zsh and the command fails with "unrecognized arguments". The array form works in zsh and bash.

Check everything is installed: `aurora -h`, `aurora pipeline run -h`, `aurora flow sweep -h`.

---

## Part A: Pre-processing pipeline

### A1. Reproduce an old version

```bash
aurora pipeline run $SRC --chain chains/v4.yaml
```

| Reads | Writes |
|---|---|
| `$SRC` (`rawimg`, `ut1_unix`), `chains/v4.yaml` | `$P/v4-ad2d874b/frames.h5`. It already exists, so this prints `exists, skipping` |

### A2. Preview without running (`--dry-run`)

```bash
aurora pipeline run $SRC -c chains/v2.yaml -c chains/v4.yaml -c chains/v6.yaml -n
```

| Reads | Writes |
|---|---|
| source, chain files, existing outputs | nothing. Prints a table with each folder and `exists` / `would run` |

### A3. Change one setting (`--set`)

```bash
aurora pipeline run $SRC -c chains/bin_med_wiener.yaml --set binned_temporal_average.bin_size=15
aurora pipeline run $SRC -c chains/bin_med_wiener.yaml --set wiener=off
aurora pipeline run $SRC -c chains/bin_med_wiener.yaml --set wiener.noise=auto --set spatial_median.k=5
```

| Reads | Writes |
|---|---|
| source, chain | `$P/bin_med_wiener_bin_size15-<hash>/`, `$P/bin_med_wiener_nowiener-<hash>/`, `$P/bin_med_wiener_noiseauto_k5-<hash>/` |

### A4. Sweep settings (`--grid`)

```bash
aurora pipeline run $SRC -c chains/bin_med_wiener.yaml \
    --grid binned_temporal_average.bin_size=5,10,15,30 --grid wiener=on,off --preview
```

| Reads | Writes |
|---|---|
| source, chain | 8 folders `$P/bin_med_wiener_bin_size<N>_<wiener\|nowiener>-<hash>/`, each with `frames.h5` + `preview.mp4`. The four already made by the example job (bin 10 and 30, with and without Wiener) are skipped. |

### A5. Several chains at once, with your own names

```bash
aurora pipeline run $SRC -c chains/v6.yaml -c chains/bin_med_wiener.yaml          # labels: v6, bin_med_wiener
aurora pipeline run $SRC -c chains/bin_med_wiener.yaml --label tenframe            # label: tenframe
```

### A6. A different output folder (`--out`)

```bash
aurora pipeline run $SRC -c chains/bin_med_wiener.yaml --out /Volumes/Elements/processed
aurora pipeline list /Volumes/Elements/processed
```

| Reads | Writes |
|---|---|
| source, chain | `/Volumes/Elements/processed/2013-03-27-CamSer7196_015400-015600/bin_med_wiener-<hash>/` and `/Volumes/Elements/processed/index.csv` |

The hash ignores the output folder, so the same chain gets the same folder name on either drive.

### A7. A whole night, or many nights (folder and glob sources)

```bash
aurora pipeline run /Volumes/Elements/HiST/2013-03-27 -c chains/bin_med_wiener.yaml -n         # every *.h5 in the folder
aurora pipeline run '/Volumes/Elements/HiST/2013-03-*/*.h5' -c chains/bin_med_wiener.yaml -n    # a glob; quote it
```

| Reads | Writes |
|---|---|
| every matching `.h5` | `optical_flow/processed/<each source stem>/bin_med_wiener-<hash>/` |

### A8. Job files (the reproducible way)

```bash
aurora pipeline run jobs/example_bin_sweep.yaml -n                     # what would it do?
aurora pipeline run jobs/example_bin_sweep.yaml                        # run it
aurora pipeline run jobs/example_bin_sweep.yaml --grid spatial_median.k=3,5 --out /Volumes/Elements/processed
aurora pipeline run jobs/example_bin_sweep.yaml --force                # rebuild everything in it
```

A new job file, e.g. `jobs/night_0327.yaml`. Paths are relative to the job file's folder:

```yaml
output: /Volumes/Elements/processed          # OUT
sources:                                     # IN
  - dir: /Volumes/Elements/HiST/2013-03-27
    glob: "*CamSer7196*.h5"
chains:                                      # IN
  - ../chains/bin_med_wiener.yaml
  - ../chains/v6.yaml
grid:
  binned_temporal_average.bin_size: [10, 30]
preview: true
```

### A9. Sources whose datasets aren't `rawimg` / `ut1_unix`

```bash
aurora pipeline run some_file.h5 -c chains/bin_med_wiener.yaml --frames imgs --times t
```

### A10. Find what you've made

```bash
aurora pipeline list                 # rebuilds optical_flow/processed/index.csv and prints a summary
aurora pipeline list --all           # every column: settings, fitted vmin/vmax, git commit, …
```

### A11. From Python or a notebook

```python
from auroral_processing.pipeline.processing_chain import run_chain
from auroral_processing.pipeline.config import load_chain, load_job, apply_overrides
from auroral_processing.pipeline.jobs import run_job

out = run_chain(SRC, load_chain("chains/bin_med_wiener.yaml").stages, "optical_flow/processed", preview=True)
print(out)                            # IN: SRC → OUT: optical_flow/processed/<src>/<label>-<hash>
run_job(load_job("jobs/example_bin_sweep.yaml"), dry_run=True)
```

The full YAML syntax is in the separate *YAML Pipeline Tutorial*.

---

## Part B: Optical flow

The flow commands work on **any** frame stack: a raw source (`rawimg`, `ut1_unix`, the defaults), a pipeline output (`frames`, `frames_ut1_unix` → add `"${FR[@]}"`), or a benchmark (`synth`, `synth_t`).

Pick an output to work on:

```bash
O=$P/bin_med_wiener_bin_size10_wiener-b04243cc      # 636 frames, 0.189 s apart, 16-bit
```

### B1. One flow run with defaults

```bash
aurora flow compute $O/frames.h5 "${FR[@]}"
```

| Reads | Writes |
|---|---|
| `$O/frames.h5` | `$O/frames_flow.h5`: flow for every consecutive pair (gap 1), TV-L1 with OpenCV defaults, metrics `warp_mae`, `zero_mae` |

Flow is in px per pair (u = +x right, v = +y down). The run resumes: interrupt it and run the same command again, and it continues where it stopped.

### B2. Choosing the output file and adding videos

```bash
aurora flow compute $O/frames.h5 "${FR[@]}" -o $O/flow/single/gamma3.h5 --video quiver speed \
    --tvl1-gamma 3 --tvl1-lambda 0.2 --tvl1-scale-step 0.5
```

| Reads | Writes |
|---|---|
| `$O/frames.h5` | `$O/flow/single/gamma3.h5`, `gamma3_quiver.mp4`, `gamma3_speed.mp4` |

### B3. Which frames get paired: gap, stride, start, stop

| option | meaning | example |
|---|---|---|
| `--gap G` | frames between the two images of a pair | `--gap 3` → (0,3), (3,6), … |
| `--stride S` | frames between successive pairs (default = gap) | `--gap 3 --stride 1` → (0,3), (1,4), … overlapping |
| `--start A --stop B` | only frames A … B−1 | `--start 100 --stop 200` |

```bash
aurora flow compute $O/frames.h5 "${FR[@]}" --gap 3 -o $O/flow/gap3.h5                   # 0.57 s per pair
aurora flow compute $O/frames.h5 "${FR[@]}" --gap 3 --stride 1 -o $O/flow/gap3_s1.h5     # 3× the pairs, overlapping
aurora flow compute $O/frames.h5 "${FR[@]}" --start 300 --stop 360 -o $O/flow/burst.h5   # a short window
```

**Seconds per pair = gap × frame spacing.** On this output the spacing is 0.189 s, so gap 3 is 0.57 s, the same as gap 1 on a bin-30 output.

### B4. Quality metrics, warm start and speed

```bash
aurora flow compute $O/frames.h5 "${FR[@]}" --metrics warp_mae zero_mae fb_error -o $O/flow/fb.h5   # fb_error doubles run time
aurora flow compute $O/frames.h5 "${FR[@]}" --warm-start -o $O/flow/warm.h5     # each pair starts from the previous flow (TV-L1)
aurora flow compute $O/frames.h5 "${FR[@]}" -j 4 -o $O/flow/par.h5              # 4 processes
```

### B5. Input scaling options

```bash
aurora flow compute $O/frames.h5 "${FR[@]}" --clip 1 99 -o $O/flow/clip1_99.h5      # percentiles for the global lo/hi (default 0.5 99.5)
aurora flow compute $O/frames.h5 "${FR[@]}" --input-dtype uint8 -o $O/flow/u8.h5    # quantize the TV-L1 input to 8-bit (old prototype)
```

### B6. Every TV-L1 setting

`--tvl1-<name> VALUE`, with OpenCV defaults:

| flag | default | effect |
|---|---|---|
| `--tvl1-lambda` | 0.15 | data weight: higher follows the images more and smooths less |
| `--tvl1-gamma` | 0.0 | tolerance to brightness change (> 0 helps with brightening aurora) |
| `--tvl1-tau` | 0.25 | solver step (keep ≤ 0.25) |
| `--tvl1-theta` | 0.3 | coupling of data and smoothness terms |
| `--tvl1-scales` | 5 | pyramid levels: more can follow larger motions |
| `--tvl1-scale-step` | 0.8 | size ratio between levels (0.5 did best on your benchmark) |
| `--tvl1-warps` | 5 | warps per level |
| `--tvl1-epsilon` | 0.01 | convergence threshold |
| `--tvl1-inner-iterations` / `--tvl1-outer-iterations` | 30 / 10 | iterations per warp |
| `--tvl1-median-filtering` | 5 | median filter on the flow (1 = off) |

### B7. PIV and FLCT instead of TV-L1 (`--method`)

```bash
aurora flow compute $O/frames.h5 "${FR[@]}" --method piv  --piv-window 16 --piv-first-window 64 -o $O/flow/piv16.h5
aurora flow compute $O/frames.h5 "${FR[@]}" --method piv  --piv-window 8 --piv-overlap 0.75 --piv-max-disp 20 -o $O/flow/piv8.h5
aurora flow compute $O/frames.h5 "${FR[@]}" --method flct --flct-sigma 8 -o $O/flow/flct8.h5
aurora flow compute $O/frames.h5 "${FR[@]}" --method flct --flct-sigma 12 --flct-skip 4 -o $O/flow/flct12_fast.h5
```

| PIV flag (default) | | FLCT flag (default) | |
|---|---|---|---|
| `--piv-window` (16) | final window, px | `--flct-sigma` (8) | Gaussian window, px |
| `--piv-first-window` (64) | first pass; trackable motion ≈ ⅓ of it | `--flct-thresh` (0.05) | skip pixels darker than this fraction |
| `--piv-overlap` (0.5) | window overlap | `--flct-kr` (0 = off) | low-pass filter |
| `--piv-max-disp` (40) | largest accepted px/pair | `--flct-skip` (1) | compute every n-th pixel (faster) |
| `--piv-s2n-threshold` (1.05) | signal/noise cutoff | `--flct-biascor` (1) | bias correction |
| `--piv-median-threshold` (3) | outlier test | | |
| `--piv-normalized` (1) | normalized correlation | | |

PIV and FLCT also store `invalid_frac` per pair. `--warm-start` is TV-L1 only.

### B8. Flow on raw frames (no pre-processing)

```bash
aurora flow compute $SRC --stop 500 -o optical_flow/compare/raw_test/flow.h5      # rawimg / ut1_unix are the defaults
```

| Reads | Writes |
|---|---|
| `$SRC` `rawimg` | `optical_flow/compare/raw_test/flow.h5`. Without `-o` it would go **next to the source** in `optical_flow/data/` |

### B9. Videos of an existing flow file

```bash
aurora flow video $O/flow/gap3.h5                                   # quiver → $O/flow/gap3_quiver.mp4
aurora flow video $O/flow/gap3.h5 --kind speed --vmax 3 --cmap viridis -o $O/flow/gap3_speed.mp4
aurora flow video $O/flow/gap3.h5 --grid-arrows 48 --arrow-scale 8 --fps 15
aurora flow video old_run/flow.h5 --source /new/path/source.h5      # the source moved since the flow was computed
```

Options:
- `--vmax` is the top of the colour scale in px/pair (default: 99th percentile).
- `--grid-arrows` sets the number of arrows across the image.
- `--arrow-scale` is the arrow length in px per px/pair.

### B10. Sweeps (a grid of settings)

```bash
aurora flow sweep $O/frames.h5 $O/flow/gamma_lambda "${FR[@]}" --tvl1-scale-step 0.5 \
    --grid gamma=0,1,3 --grid lambda=0.1,0.15,0.2,0.3 -j 4
```

| Reads | Writes |
|---|---|
| `$O/frames.h5` | `$O/flow/gamma_lambda/<run>/flow.h5` + `flow_quiver.mp4` for 12 runs (`gamma0_lambda0.1`, …), and `$O/flow/gamma_lambda/summary.csv` |

More sweep setups:

```bash
# gap: runs share the same first frames; per-second scores compare them
aurora flow sweep $O/frames.h5 $O/flow/gaps "${FR[@]}" --tvl1-gamma 3 --grid gap=1,2,3,6 --video

# gap × stride
aurora flow sweep $O/frames.h5 $O/flow/gap_stride "${FR[@]}" --grid gap=2,4 --grid stride=1,4 --video

# pyramid
aurora flow sweep $O/frames.h5 $O/flow/pyramid "${FR[@]}" --grid scale_step=0.5,0.65,0.8 --grid scales=3,5,7

# PIV window sizes / FLCT sigma
aurora flow sweep $O/frames.h5 $O/flow/piv_win "${FR[@]}" --method piv --grid window=8,12,16,24
aurora flow sweep $O/frames.h5 $O/flow/flct_sig "${FR[@]}" --method flct --grid sigma=4,8,12 --flct-skip 2

# videos: none, both, fixed colour scale
aurora flow sweep $O/frames.h5 $O/flow/novid "${FR[@]}" --grid gamma=1,3 --video
aurora flow sweep $O/frames.h5 $O/flow/vids  "${FR[@]}" --grid gamma=1,3 --video quiver speed --vmax 2
```

Complete runs are skipped, and partial runs resume, so rerunning a sweep only fills in what's missing.

### B11. Scoring a sweep

```bash
aurora flow score $O/flow/gamma_lambda
```

| Reads | Writes |
|---|---|
| every `<run>/flow.h5` and the source frames | `$O/flow/gamma_lambda/scores.csv`, `scores_heatmaps.png` (a heatmap when exactly two settings vary, otherwise bar charts) |

Use `--source PATH` if the source has moved, and `--no-plot` to skip the image. The metrics are explained in the *Metrics and Tests* document.

### B12. Comparing across outputs (pre-processing choices)

```bash
for o in $P/bin_med_wiener_bin_size10_*; do
  aurora flow sweep $o/frames.h5 $o/flow/best_0p57s "${FR[@]}" --gap 3 \
      --tvl1-gamma 3 --tvl1-lambda 0.2 --tvl1-scale-step 0.5 --grid gamma=3 --video -j 4
done
for o in $P/bin_med_wiener_bin_size30_*; do
  aurora flow sweep $o/frames.h5 $o/flow/best_0p57s "${FR[@]}" --gap 1 \
      --tvl1-gamma 3 --tvl1-lambda 0.2 --tvl1-scale-step 0.5 --grid gamma=3 --video -j 4
done
aurora flow compare optical_flow/compare/bin_wiener $P/bin_med_wiener_*/flow/best_0p57s/gamma3
aurora flow score   optical_flow/compare/bin_wiener
```

| Reads | Writes |
|---|---|
| four outputs' `frames.h5` | `<output>/flow/best_0p57s/gamma3/flow.h5` in each output; `optical_flow/compare/bin_wiener/` with links named `<output>-best_0p57s-gamma3`, plus `scores.csv` and `scores_heatmaps.png` (`bin_size` × `wiener.on`) |

Bin 10 with gap 3 and bin 30 with gap 1 are both 0.57 s per pair, so the comparison is at equal physical time.

### B13. Benchmarks (known truth)

```bash
aurora flow benchmark $O/frames.h5 -o $O/benchmark.h5 --frames frames
aurora flow benchmark $O/frames.h5 -o $O/benchmark_fast.h5 --frames frames --speeds 5 10 20 30 --brightness 0 0 --seed 1
aurora flow benchmark $O/frames.h5 -o $O/benchmark_real.h5 --frames frames --realistic-flow $O/flow/gap3.h5

aurora flow sweep $O/benchmark.h5 $O/flow/bench_gl --frames synth --times synth_t --gap 1 --stride 2 \
    --tvl1-scale-step 0.5 --grid gamma=1,3 --grid lambda=0.15,0.2,0.3 --video
aurora flow score $O/flow/bench_gl
```

| Reads | Writes |
|---|---|
| `$O/frames.h5` | `$O/benchmark.h5` (`synth`, `synth_t`, `true_flow`, masks). The sweep writes `$O/flow/bench_gl/…`; score adds the truth columns (`epe_lane`, `speed_recovery`, …) |

A benchmark sweep **must** use `--frames synth --times synth_t --gap 1 --stride 2`, so that each pair is one (a, b) benchmark pair. `--brightness 0 0` turns off the brightening patches. `--realistic-flow` adds a case built from a real flow file of the same frames.

### B14. Comparing flow methods

```bash
aurora flow sweep $O/benchmark.h5 $O/flow/bench_piv  --frames synth --times synth_t --gap 1 --stride 2 --method piv  --grid window=8,16 --video
aurora flow sweep $O/benchmark.h5 $O/flow/bench_flct --frames synth --times synth_t --gap 1 --stride 2 --method flct --grid sigma=6,10 --video
aurora flow compare optical_flow/compare/methods $O/flow/bench_gl/gamma3_lambda0.2 $O/flow/bench_piv/window16 $O/flow/bench_flct/sigma6
aurora flow score optical_flow/compare/methods
```

### B15. MATLAB

```bash
aurora flow repack $O/flow/gap3.h5          # LZF → gzip, in place, so MATLAB can read it without the LZF plugin
```

### B16. From Python

```python
from auroral_processing.flow.compute_flow import compute_flow
from auroral_processing.flow.params import TVL1Params, PIVParams
from auroral_processing.flow.sweep import run_sweep
from auroral_processing.flow.score import score_sweep
from auroral_processing.products.flow_video import make_flow_video

O = "optical_flow/processed/2013-03-27-CamSer7196_015400-015600/bin_med_wiener_bin_size10_wiener-b04243cc"
best = TVL1Params(gamma=3, lambda_=0.2, scale_step=0.5)
f = compute_flow(f"{O}/frames.h5", f"{O}/flow/py/best.h5", frames="frames", times="frames_ut1_unix",
                 params=best, gap=3, workers=4)
make_flow_video(f, kind="quiver")
run_sweep(f"{O}/frames.h5", {"lambda": [0.15, 0.2]}, f"{O}/flow/py_sweep", base_params=best,
          frames="frames", times="frames_ut1_unix", gap=3, videos=())
score_sweep(f"{O}/flow/py_sweep")
```

Scripts that use `workers > 1` must run under `if __name__ == "__main__":` on macOS.

---

## Part C: Complete workflows

### W1. 8-bit vs 16-bit (one variable)

`v6` and `bin_med_wiener_bin_size10_wiener` differ only in `out_max`.

```bash
for o in $P/v6-9359fc6d $P/bin_med_wiener_bin_size10_wiener-b04243cc; do
  aurora flow sweep $o/frames.h5 $o/flow/best_gap3 "${FR[@]}" --gap 3 --tvl1-gamma 3 --tvl1-lambda 0.2 \
      --tvl1-scale-step 0.5 --grid gamma=3 --video -j 4
done
aurora flow compare optical_flow/compare/quantization $P/v6-*/flow/best_gap3/gamma3 $P/bin_med_wiener_bin_size10_wiener-*/flow/best_gap3/gamma3
aurora flow score optical_flow/compare/quantization
```

### W2. New pre-processing idea → flow verdict

```bash
aurora pipeline run $SRC -c chains/bin_med_wiener.yaml --set spatial_median.k=7 --preview     # 1. make it
aurora pipeline list                                                                           # 2. find its folder
o=$P/bin_med_wiener_k7-<hash>                                                                  #    (from the list)
aurora flow sweep $o/frames.h5 $o/flow/best_gap3 "${FR[@]}" --gap 3 --tvl1-gamma 3 \
    --tvl1-lambda 0.2 --tvl1-scale-step 0.5 --grid gamma=3 --video                            # 3. same flow as W1
aurora flow compare optical_flow/compare/median_k $P/bin_med_wiener_bin_size10_wiener-*/flow/best_gap3/gamma3 $o/flow/best_gap3/gamma3
aurora flow score optical_flow/compare/median_k                                                # 4. verdict
```

### W3. Re-tune TV-L1 for 16-bit data

Run B13 on the 16-bit bin-10 output, then read `combined_rank` in `$O/flow/bench_gl/scores.csv`.

### W4. A full night, start to finish

```bash
aurora pipeline run jobs/night_0327.yaml -n && aurora pipeline run jobs/night_0327.yaml   # A8's job file
for f in /Volumes/Elements/processed/*/bin_med_wiener_bin_size10-*/frames.h5; do
  aurora flow compute $f "${FR[@]}" --gap 3 --tvl1-gamma 3 --tvl1-lambda 0.2 --tvl1-scale-step 0.5 \
      -o ${f:h}/flow/best_gap3.h5 --video quiver -j 4
done
```

In zsh, `${f:h}` is the folder containing `$f`; in bash use `$(dirname $f)`.

---

## Part D: Troubleshooting

| symptom | cause / fix |
|---|---|
| `unrecognized arguments: --frames frames --times …` | flags kept in a string variable in zsh; use an array (§1) |
| `KeyError: … (object 'rawimg' doesn't exist)` | a pipeline output or benchmark without `"${FR[@]}"` / `--frames synth --times synth_t` |
| `… exists with different settings; pass overwrite=True` | that flow file was made with other settings; pick a new `-o` or add `--overwrite` |
| `benchmark sweeps must use --gap 1 --stride 2` | add those flags when sweeping a benchmark |
| a flow file landed in `optical_flow/data/` | `flow compute` without `-o` writes next to the source |
| `flow score`: "no <run>/flow.h5" | point it at the sweep folder (holding run folders), not at a run |
| video: "Unable to open file" for the source | the source moved; add `--source NEW_PATH` |
| `PIV first-pass window … needs at least 4x4` | use a smaller `--piv-first-window` |
| A grid run was skipped | complete runs are reused; delete the run folder or use `--overwrite` to redo it |
| `-o some/new/folder/flow.h5` | missing folders are created (since commit `26209d7`) |
