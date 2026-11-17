# How the Optical Flow Code Works

*A file-by-file, function-by-function walkthrough · `auroral_processing/flow/`, `products/flow_video.py`, `cli/flow_cli.py`*

---

## 1. The big picture

```
                    aurora flow …  (cli/flow_cli.py)
                               │
     ┌─────────────┬───────────┼─────────────┬──────────────┬──────────────┐
     ▼             ▼           ▼             ▼              ▼              ▼
  compute        sweep       video         score         benchmark       repack
     │             │           │             │              │              │
     ▼             ▼           ▼             ▼              ▼              ▼
 compute_flow  run_sweep  make_flow_video score_sweep  make_benchmark  repack_flow
     │       (calls compute_flow     (reads flow     (synthetic frames   (io.py)
     │        per grid point)         files only)     + true flow)
     ├── params.py      settings dataclasses (TV-L1 / PIV / FLCT)
     ├── methods.py     estimators: one .calc(a, b) → (H, W, 2) flow
     ├── metrics.py     per-pair quality numbers
     └── io.py          FlowWriter: the flow file (resume, config, metrics)
```

**One flow file per run.** Every computation writes one HDF5 file with all its pairs, per-pair metrics, and a JSON `config` describing exactly how it was made. Everything downstream (videos, sweeps, scoring, comparisons) reads these files and never recomputes flow.

**Conventions used everywhere:**

- Flow is an `(H, W, 2)` float32 array: `u = flow[..., 0]` is +x (right), `v = flow[..., 1]` is +y (down).
- Units are **pixels per pair**, from the pair's first frame `i0` to its second `i1 = i0 + gap`. Divide by the pair's time `t1 − t0` for px/s.
- Images reach the estimators as float32 in 0–1 (or uint8 0–255), all scaled with **one global lo/hi**. Per-frame stretching would turn brightness changes into fake motion.

---

## 2. Data flow of one run (`compute_flow`)

```
source.h5 [frames] ──fit_scaler──► FrameScaler(lo, hi)            (once, from 20 sample frames)
          │
          ├─ make_pairs(gap, stride, start, stop) ──► pairs (P, 2) = (i0, i1)
          │
          └─ FlowWriter(out, config, pairs, …) ──► existing file with the same config?  resume
                     │                                     different config?  error (unless overwrite)
                     ▼
            todo = pairs not yet `done`
                     │
       workers == 1  │  workers > 1: chunks → ProcessPoolExecutor(_run_chunk)
                     ▼
            iter_flow: for each pair k
                a, b = scaler(frames[i0]), scaler(frames[i1])      (2-frame cache)
                flow = estimator.calc(a, b, init)                  (init = previous flow if warm_start)
                metrics = estimator.last + warp_mae, zero_mae, fb_error
                     │
                     ▼
            writer.write(k, flow, metrics) → flow[k], metrics/*[k], done[k] = True
                     │
            writer.finish() → attrs["complete"] = all done
```

---

## 3. `flow/params.py`: settings

Each method's settings are a dataclass. Field metadata holds the help text (used by the CLI) and, for TV-L1, the name of the OpenCV setter.

| name | what it does |
|---|---|
| `_param(default, setter, help)` | Builds a dataclass `field` with `metadata={"setter", "help"}`. |
| `class _Params` | Shared base class: saving, loading and copying settings. |
| `_Params.to_dict()` | Plain dict with *external* names (`lambda`, not `lambda_`). This goes into the flow file's config. |
| `_Params.from_dict(d)` | Inverse of `to_dict`. Accepts `lambda` or `lambda_`, casts values to int/float, rejects unknown keys with the allowed list. |
| `_Params.replace(**changes)` | Copy with some settings changed (used by sweeps). |
| `class TVL1Params` | `lambda_`, `gamma`, `tau`, `theta`, `scales`, `scale_step`, `warps`, `epsilon`, `inner_iterations`, `outer_iterations`, `median_filtering`. Defaults equal OpenCV's. `METHOD="tvl1"`. |
| `class PIVParams` | `window`, `first_window`, `overlap`, `max_disp`, `s2n_threshold`, `median_threshold`, `normalized`. `METHOD="piv"`. |
| `class FLCTParams` | `sigma`, `thresh`, `kr`, `skip`, `biascor`. `METHOD="flct"`. |
| `METHODS` | `{"tvl1": TVL1Params, "piv": PIVParams, "flct": FLCTParams}`. |
| `params_from_dict(method, d)` | Settings object for a method name, e.g. when reading a config back. |
| `external_name(name)` | `"lambda_" → "lambda"` (strips the trailing underscore). |
| `_cast(f, v)` | `int` or `float` according to the field's type. |
| `make_tvl1(params, use_initial_flow=False)` | Creates `cv2.optflow.DualTVL1OpticalFlow_create()` and calls each field's setter (`setLambda`, `setGamma`, …). Raises a clear `ImportError` if OpenCV-contrib is missing. |

---

## 4. `flow/methods.py`: the estimators

Every estimator has the same interface:

- `calc(a, b, init=None) -> (H, W, 2) float32`, in px/pair, +x right, +y down;
- `.last`: a dict of per-pair extras (e.g. `invalid_frac`);
- `supports_init`: whether warm start works;
- `extra_metrics`: the names of those extras.

| name | what it does |
|---|---|
| `_to_unit(img)` | float32 0–1 whichever dtype it gets (uint8 is divided by 255). |
| **`class TVL1Estimator`** | Dual TV-L1 variational optical flow (OpenCV). `supports_init = True`. |
| `TVL1Estimator.__init__(params)` | Builds the OpenCV object once with `make_tvl1`. |
| `TVL1Estimator.calc(a, b, init)` | Turns on `setUseInitialFlow` when `init` is given, then calls `tvl1.calc(a, b, init)`. OpenCV accepts uint8 0–255 or float32 0–1. |
| `pass_windows(window, first_window)` | PIV window size per pass: halve from `first_window` while it's above `window`, then end at `window`. `(16, 64) → (64, 32, 16)`, `(12, 64) → (64, 32, 16, 12)`. |
| **`class PIVEstimator`** | Multi-pass, window-deforming cross-correlation PIV (`openpiv.windef`), interpolated to every pixel. `extra_metrics = ("invalid_frac",)`. |
| `PIVEstimator.__init__(params)` | Validates overlap ∈ [0, 1) and `first_window ≥ window`. Fills `windef.PIVSettings`: window sizes and overlaps per pass, displacement limits ±`max_disp`, signal-to-noise and median thresholds, normalized correlation. Plots off. |
| `PIVEstimator._grid(a, b)` | 1. Checks that the first-pass grid is at least 4×4. 2. `first_pass`. 3. Validation (`typical_validation`): displacement limits, global std, signal-to-noise and local-median tests. 4. `replace_outliers`. 5. `multipass_img_deform` for each later pass, which shifts and deforms windows by the flow so far. Returns window centres `x, y`, vectors `u, v`, and the final-pass `flags`. |
| `PIVEstimator.calc(a, b)` | Runs `_grid` on 0–1 images. If no vector survives validation, it writes zero flow with a warning (`invalid_frac = 1`). Otherwise `invalid_frac = mean(flags)`. It maps every pixel to fractional grid coordinates and bilinearly interpolates `u, v` with `cv2.remap` (edge replicate). |
| **`class FLCTEstimator`** | Fourier local correlation tracking (`pyflct`): a Gaussian-windowed cross-correlation centred on every pixel. `extra_metrics = ("invalid_frac",)`. |
| `FLCTEstimator.__init__(params)` | Imports `pyflct` (failing early if it's missing) and makes a private temp folder. pyflct writes `temp.dat` to the working directory, and parallel workers would otherwise delete each other's file. |
| `FLCTEstimator.__del__` | Removes the temp folder. |
| `FLCTEstimator.calc(a, b)` | Calls `pyflct.flct` with Δt = Δs = 1 (so px/pair), `sigma`, `thresh`, `kr`, `skip` (+ interpolation) and `biascor`, using `order="column"` and transposing the result. pyflct 0.3's default row order returns no y motion. Pixels FLCT skipped (`vm == 0`) get zero flow and count toward `invalid_frac`; NaNs become 0. |
| `ESTIMATORS` | `{"tvl1": …, "piv": …, "flct": …}`. |
| `make_estimator(params)` | The estimator for a settings object (`ESTIMATORS[params.METHOD](params)`). |

---

## 5. `flow/compute_flow.py`: running a method over a frame stack

| name | what it does |
|---|---|
| `INPUT_DTYPES` | `("float32", "uint8")`, the image types TV-L1 accepts. |
| **`class FrameScaler(lo, hi, dtype)`** | Frozen dataclass. `scaler(frame)` = `clip((frame − lo)/(hi − lo), 0, 1)` as float32, or ×255 as uint8. The same `lo`/`hi` are used for every frame of a run. |
| `fit_scaler(ds, start, stop, percentiles=(0.5, 99.5), dtype, n_sample=20)` | Picks `lo`/`hi` as percentiles over 20 evenly spaced frames of the range. Errors if `hi ≤ lo`. |
| `make_pairs(n_frames, gap=1, stride=None, start=0, stop=None)` | `(P, 2)` array of `(i0, i0 + gap)`, with `i0` stepping by `stride` (default = gap) inside [start, stop). |
| `default_out_path(src)` | `<src stem>_flow.h5` next to the source. |
| `_to_unit(img)` | float32 0–1 for the metrics, whichever dtype TV-L1 got. |
| **`iter_flow(src, frames, pairs, ks, params, scaler, metrics, warm_start)`** | Generator over pair indices `ks`. It opens the source, builds the estimator (plus a second one for backward flow if `fb_error` is requested), keeps the **last two scaled frames** in a cache (with stride = gap each frame is loaded once), passes the previous flow as `init` when `warm_start` and the pairs are consecutive, computes the metrics, and yields `(k, flow, metrics)`. |
| `_init_worker(threads)` | Worker-process setup: `cv2.setNumThreads(threads)`, so workers share the cores. |
| `_run_chunk(*args)` | `list(iter_flow(*args))`, i.e. one chunk of pairs in a worker. |
| **`compute_flow(src, out, *, frames, times, params, gap, stride, start, stop, clip_percentiles, input_dtype, metrics, warm_start, workers, chunk_size, overwrite, progress)`** | The main entry point. 1. Validates the options. 2. Reads the frame and time datasets. 3. Makes the pairs and fits the scaler. 4. Builds the **config** dict (`source` resolved path, `frames`, `times`, `gap`, `stride`, `start`, `stop`, `clip_percentiles`, `input_dtype`, `metrics`, `warm_start`, the method's settings under its name, and `method` for non-TV-L1). 5. Opens a `FlowWriter`. 6. Takes `lo`/`hi` from the writer, so a resumed run keeps its original scaling. 7. Computes only pairs not yet done, serially or in parallel chunks (about 4 per worker, each with `cpu_count // workers` OpenCV threads). 8. Writes each pair as it arrives and finishes. Returns the flow file path. |

**Design notes:**
- TV-L1 configs have no `method` key, so runs made before PIV/FLCT existed still resume.
- In parallel mode, `warm_start` only carries over inside a chunk.
- Parallel and serial runs give identical flow; a test checks this.

---

## 6. `flow/io.py`: the flow file

File layout:

```
flow        (P, H, W, 2) float32, LZF-compressed, one chunk per pair, NaN until written
i0, i1      (P,) int64      source frame indices
t0, t1      (P,) float64    source times
done        (P,) bool       pair written (resume)
metrics/<name>  (P,) float32   per-pair metrics, NaN until written
attrs: config (JSON, sorted keys), <method>_<setting> for each setting, lo, hi,
       cv2_version, created, complete
```

| name | what it does |
|---|---|
| **`class FlowWriter`** | Creates a flow file, or reopens one to continue writing pairs into it. |
| `FlowWriter.__init__(path, *, config, pairs, t0, t1, shape, metrics, scaling, overwrite, compression="lzf")` | If the file exists and `overwrite` is false: reopen it if its `config` JSON is identical (**resume**), otherwise raise `FileExistsError`. If not: create the parent folder and the datasets, and write the attrs, including a flat copy of each setting (`tvl1_gamma`, `piv_window`, …) for MATLAB/HDFView. |
| `done` (property) | The `done` array. |
| `scaling` (property) | `{"lo", "hi"}` from the attrs. |
| `write(k, flow, metrics)` | Stores pair `k`'s flow and metrics and marks it done. |
| `finish()` | Sets `complete` = all pairs done, then closes. |
| `close()`, `__enter__`, `__exit__` | File handling / `with` support. |
| `open_flow(path)` | Opens read-only. |
| `read_config(f)` | The JSON config as a dict. |
| `read_method(config)` | `(method, settings)`. Configs without `method` are TV-L1. |
| `repack_flow(path, out=None)` | Copies every dataset and attribute into a gzip-compressed file and replaces the original (or writes `out`). MATLAB can read gzip without the LZF plugin. |

---

## 7. `flow/metrics.py`: per-pair quality metrics

The metrics are computed during `compute_flow` and stored in `metrics/<name>`. They are explained in detail in the *Metrics and Tests* document.

| name | what it does |
|---|---|
| `METRICS`, `DEFAULT_METRICS` | `("warp_mae", "zero_mae", "fb_error")`; the default leaves out `fb_error` (it costs a second flow). |
| `_sample_coords(flow)` | Where each vector lands (`x + u`, `y + v`) and a mask of landings inside the frame. |
| `warp_mae(i0, i1, flow)` | Mean \|I0 − I1 warped back by the flow\| over pixels that land inside the frame. |
| `zero_mae(i0, i1)` | Mean \|I1 − I0\|: the residual if there were no motion (the baseline `warp_mae` should beat). |
| `fb_error(flow, back_flow)` | Forward-backward consistency: mean \|f(x) + b(x + f(x))\| in px. |

---

## 8. `flow/sweep.py`: grids of runs

| name | what it does |
|---|---|
| `PAIRING` | `("gap", "stride")`: grid keys that change the pairs rather than the method. |
| `run_name(settings)` | Folder name from the grid values: `{"gamma": 1, "lambda": 0.15}` → `gamma1_lambda0.15`. |
| `summarise(flow_path)` | `pairs_done`, `complete`, and the mean and median of each metric, for `summary.csv`. |
| **`run_sweep(src, grid, outdir, *, base_params, videos, video_kw, progress, **compute_kw)`** | For every combination in `grid`: apply the method settings with `base_params.replace`, run `compute_flow` into `outdir/<run>/flow.h5` (resuming or skipping), render missing videos under a `.partial.mp4` name then rename, and add a row to `summary.csv`. **Gap sweeps** keep runs comparable: `stride` defaults to the largest gap, and `stop` is shortened per gap, so every gap uses the same first frames `i0`. A video `vmax` given for the base gap is scaled with the gap (and `arrow_scale` inversely), so colours mean the same px/s. |

---

## 9. `flow/score.py`: scoring runs without recomputing them

| name | what it does |
|---|---|
| `BORDER` | 16 px ignored at the frame edges. |
| `REAL_PANELS`, `GAP_PANELS`, `TRUTH_PANELS` | `(column, title, lower_is_better)` lists for the heatmap figure. |
| `structured_mask(img, frac=0.15)` | Pixels whose gradient magnitude (after a σ = 3 blur) is in the top 15%: auroral structure, i.e. lanes and edges. |
| `_derivatives(flow, sigma=4)` | Vorticity (∂v/∂x − ∂u/∂y) and shear magnitude √((∂u/∂x − ∂v/∂y)² + (∂u/∂y + ∂v/∂x)²) of the σ = 4-smoothed flow. |
| `_corr(a, b)` | Pearson correlation, or NaN if either input is constant. |
| **`class _Inputs`** | Everything shared by runs on the same source and pairs. Loads the scaled frames A (i0) and B (i1). For a benchmark: the truth, cases, lane, bright-patch and star masks (and it checks gap 1 / stride 2). For real data: stars found on a 50-frame mean. Also builds the noise floor, the edge mask, the structure masks `L` (minus stars), the ring around the stars, and a pixel grid. |
| `_real_scores(F, I)` | Real-data scores for one run's flows `F` (warp residuals by region, speeds, roughness, star test, per-second versions, temporal change). |
| `_truth_scores(F, I)` | Benchmark scores against the true flow (EPE, speed recovery, vorticity/shear correlation, false motion, EPE per case). |
| `_runs(sweep_dir)` | `(name, path, settings)` per run: in `summary.csv` order when there is one, else every `*/flow.h5`. The settings are method, gap, stride, the method's settings, and the source's pre-processing. |
| `_preprocessing(source)` | If the source is a pipeline output: `output` (folder), `chain` (summary), `<stage>.on`, `<stage>.<setting>`. Otherwise empty. |
| **`score_sweep(sweep_dir, *, source, plot, progress)`** | Skips incomplete runs. Caches `_Inputs` by (source, frames, i0, i1). Scores each run and adds the truth scores and ranks for benchmarks. Writes `scores.csv` and, optionally, `scores_heatmaps.png`. |
| `varied_settings(df)` | The settings that differ between runs (method, gap, stride, method settings, pre-processing columns). |
| `_tick(v)` | Tick-label formatting. |
| `plot_scores(df, out, title)` | One panel per main score. Truth panels for benchmarks; per-second (gap) panels when the time per pair differs; otherwise the real-data panels. A heatmap when exactly two settings vary (best cell outlined), otherwise horizontal bars sorted by the score. The noise floor is drawn on the warp-residual panel. |

---

## 10. `flow/benchmark.py`: synthetic data with a known answer

It takes real frames, separates stars from aurora, moves the aurora by a **known** flow, and adds back fixed stars and fresh noise matched to the source. The result is a normal frame stack (`synth`, `synth_t`) plus `true_flow` and masks.

| name | what it does |
|---|---|
| `measure_noise(ds, idx, mask)` | Noise σ per frame and its spatial correlation, from differences of consecutive frames in smooth regions (the half of pixels with the lowest gradient, stars removed). σ = clipped std / √2. The lag-1 correlation r is converted to an equivalent Gaussian blur, `sqrt(−1/(4 ln r))`. |
| `_clipped_std(x, n_sigma=4, iterations=3)` | Standard deviation with outliers (stars, changing aurora) clipped iteratively. |
| `correlated_noise(rng, shape, sigma, blur)` | Gaussian noise, blurred to the measured correlation and rescaled to σ. |
| `find_stars(mean_frame, threshold, dilate=5)` | Compact bright spots: (mean frame − 9 px median) above the threshold, dilated. |
| `find_lanes(img, frac=0.12, min_area=300)` | Dark lanes: the top 12% black-top-hat response of the smoothed frame, opened, keeping components ≥ 300 px. |
| `across_lane_direction(img, lanes)` | Unit vector across the lanes: the dominant gradient direction (structure-tensor eigenvector). |
| `lane_weight(lanes, speed)` | 1 on the lanes, falling linearly to 0 over max(15, 1.5·speed) px, then blurred, so the flow gradient stays gentle and nothing folds over. |
| `vortex(shape, centre, radius, peak)` | Rankine vortex: solid-body inside `radius`, 1/r outside, faded beyond ~3 radii. |
| `backward_map(u, iterations=50)` | Fixed-point inversion: for a forward flow u, finds w with w(y) = u(y − w(y)), so b(y) = a(y − w(y)). Returns w and the residual. |
| `warp(img, w)` | b = a(y − w(y)) with cubic interpolation and reflected borders. |
| `gain_field(rng, still, n, sigma, amplitude)` | n Gaussian blobs on still pixels with random ± gain: brightness change without motion. |
| `pick_base_frames(ds, n, lo, hi, step=4)` | The n frames with the most dark-lane structure, kept at least N/(2n) apart. |
| `max_gradient(u)` | Largest spatial derivative of a flow (fold-over check). |
| `_realistic_field(flow_path, base, sigma=8, max_grad=0.6)` | A real flow from a flow file (the pair nearest the base frame), smoothed by σ, and more if needed, until its gradient is ≤ 0.6. |
| **`make_benchmark(src, out, *, frames, base_frames, n_base=4, speeds=(5, 15, 25), shear=20, vortex_peak=10, realistic_flow, brightness=(0.05, 0.15), noise, seed, overwrite)`** | For each base frame (averaged with its two neighbours to reduce noise): split out the stars, find the lanes and the across-lane direction, and build the cases: `rigid<speed>`, `shear` (across-lane speed ramping 0 → `shear` along the lanes), `vortex` (on a lane near the centre), and optionally `realistic`. For each case: invert to a backward map, warp, add brightness blobs, then add stars and independent noise to both frames. Writes the benchmark file (layout in the module docstring). |
| `is_benchmark(path)` | True if the file has `true_flow`. |

---

## 11. `products/flow_video.py`: rendering flow files

| name | what it does |
|---|---|
| `KINDS` | `("quiver", "speed")`. |
| `colour_lut(name="turbo")` | 256-entry BGR lookup table from a matplotlib colormap. |
| `block_mean(flow, step)` | Averages the flow over step × step blocks and returns the block centres and mean u, v. Partial edge blocks are dropped. |
| `_time_label(t0, i0, i1)` | `"YYYY-MM-DD HH:MM:SS.mmm UTC   frames i0->i1"` (or `t=…` for non-epoch times). |
| `_draw_colourbar(img, lut, vmax, ref_len)` | Colour scale 0…vmax px/pair, plus a reference arrow of length vmax. |
| `_draw_label(img, label)` | Black bar with the time label. |
| `draw_quiver(background, flow, step, arrow_scale, vmax, lut, label, dim=0.55)` | The dimmed first frame with one arrow per block, coloured by block-mean speed. Tiny arrows are skipped. |
| `draw_speed(flow, vmax, lut, label)` | \|flow\| heatmap. |
| `global_vmax(flow_ds, ks, step, percentile=99, n_sample=50)` | A fixed colour/arrow limit for the whole video: the 99th-percentile speed over up to 50 pairs (block means for quivers). |
| **`make_flow_video(flow_path, out, *, kind, grid=32, vmax, arrow_scale, fps=10, dim, cmap, quality, source, progress)`** | Opens the flow file and its source (or `source=` if it moved). Step = max(H, W) // grid. `arrow_scale` defaults to making a vmax arrow 0.9 of a cell. Backgrounds use the run's own `lo`/`hi` (as uint8). Writes one video frame per computed pair with libx264. |

---

## 12. `cli/flow_cli.py`: the `aurora flow` commands

| name | what it does |
|---|---|
| `_method_fields(method)` | `(flag name, field)` per setting, e.g. `("scale-step", …)`. Used to generate `--tvl1-scale-step` and the others. |
| `METHOD_NAMES` | `("tvl1", "piv", "flct")`. |
| `_cast(f)` | `int` or `float` for argparse. |
| **`add_parser(sub)`** | Builds `aurora flow` with subcommands `compute`, `video`, `repack`, `sweep`, `compare`, `score` and `benchmark`. A shared `inputs` parent holds the source and pairing options. A generated parameter parent has `--<method>-<setting>` flags for all three methods, with help and defaults taken from the dataclasses. |
| `_add_video_options(p)` | `--grid-arrows`, `--vmax`, `--arrow-scale`, `--fps`, `--cmap`. |
| `_params(args)` | The settings object for `--method`, from its flags. |
| `_compute_kw(args)`, `_video_kw(args)` | The `compute_flow` and `make_flow_video` keyword arguments from the flags. |
| `run_compute(args)` | `compute_flow`, then the requested videos. |
| `run_video(args)` | `make_flow_video`. |
| `run_repack(args)` | `repack_flow` for each file. |
| `parse_grid(items, method)` | `["gamma=1,3", "gap=2,4"]` → `{"gamma": [1.0, 3.0], "gap": [2, 4]}`, typed by the method's fields. Unknown names exit with the allowed list. |
| `run_sweep(args)` | `run_sweep` with the parsed grid. |
| `run_compare(args)` | Creates `OUTDIR` with one symlink per run folder. Links are named `<sweep>-<run>`, or `<output>-<sweep>-<run>` for runs inside a pipeline output. If two names still collide, parent folders are added until they're unique (`night_a-best-gamma1`, `night_b-best-gamma1`). |
| `run_score(args)` | `score_sweep`. |
| `run_benchmark(args)` | `make_benchmark` from the flags (`--brightness 0 0` → none). |

---

## 13. How it fits with the pre-processing pipeline

- A pipeline output's `frames.h5` is an ordinary frame stack, so `compute_flow` reads it with `frames="frames", times="frames_ut1_unix"`. The flow config records that file's path.
- `score._preprocessing` reads the output's `chain` attribute, so scores carry the pre-processing settings, and `plot_scores` can use them as heatmap axes.
- Because outputs are never overwritten (hash-named folders), a flow file's `source` always points at the frames it was computed from.
