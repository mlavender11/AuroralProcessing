# Optical Flow Metrics and Tests

*What every number means, how it's computed, how to read it, and what the test suite checks*

---

## 1. Three levels of measurement

| level | where it's computed | where it's stored | needs truth? |
|---|---|---|---|
| **Per-pair metrics** | during `compute_flow` (`flow/metrics.py`, plus the estimator's own) | `metrics/<name>` in each `flow.h5` (one value per pair) | no |
| **Sweep summary** | `run_sweep` → `summarise` | `summary.csv` (mean/median of each per-pair metric per run) | no |
| **Scores** | `aurora flow score` (`flow/score.py`), afterwards, from saved flows | `scores.csv` + `scores_heatmaps.png` | real-data scores: no. Truth scores: a benchmark |

Notation used below:
- I0, I1 are the pair's frames, scaled to 0–1 with the run's global lo/hi.
- F = (u, v) is the estimated flow and T the true flow, both in px/pair.
- \|·\| is a magnitude.
- *dt* is the pair's time in seconds (t1 − t0).

---

## 2. Per-pair metrics (`flow.h5` → `metrics/`)

| metric | formula | meaning | good |
|---|---|---|---|
| `warp_mae` | mean over valid x of \|I1(x + F(x)) − I0(x)\| | How well the flow explains the change: pull I1 back along the flow and compare with I0. Only pixels whose vector lands inside the frame count. | Low, and **below `zero_mae`** |
| `zero_mae` | mean \|I1 − I0\| | The same residual for *no* motion: the baseline. If `warp_mae ≥ zero_mae`, the flow explains nothing. | (reference) |
| `fb_error` *(opt-in: `--metrics … fb_error`)* | mean \|F(x) + B(x + F(x))\|, where B is the flow I1 → I0 | Forward-backward consistency in px. Going forward and then back should return to the start. | Near 0; high at occlusions and in unreliable regions. Doubles the run time. |
| `invalid_frac` *(PIV, FLCT)* | PIV: share of final-pass vectors that failed validation and were replaced. FLCT: share of pixels skipped (below `thresh`) and set to zero flow | How much of the field is filled in rather than measured. | Low |

`summary.csv` has `<metric>_mean` and `<metric>_median` per run, plus `pairs_done` and `complete`.

**Caveats:**
- `warp_mae` rewards *any* flow that makes the images match, including flow that fits noise.
- On featureless glow, a wild flow can match just as well as the true one.

That is why the scores below split the frame into regions and compare against a noise floor.

---

## 3. Regions used by the scores

Frame-wide averages are dominated by featureless glow and noise. `score.py` therefore splits each frame:

| region | definition | why |
|---|---|---|
| **edge** | everything except a 16 px border (`BORDER`) | flow at frame edges is unreliable |
| **structure** ("lanes") `L` | top 15% of gradient magnitude of the σ = 3-blurred I0 (`structured_mask`), minus stars, per pair | where motion is actually observable: lanes and edges |
| **stars** | bright compact spots: (mean of 50 frames − 9 px median) > 8% of (hi − lo), dilated 5 px | stars don't move, so flow there should be ≈ 0 |
| **ring** | 15 × 15 dilation of the stars, minus the stars | the aurora just around each star, for comparison |
| **diffuse** | edge ∧ not structure ∧ not stars | glow with little to track: any motion there is suspect |

**Noise floor.** `measure_noise` estimates the per-frame noise σ from consecutive-frame differences in smooth regions (clipped std / √2). For two independent Gaussian noisy frames, E\|n0 − n1\| = 2σ/√π ≈ 1.128 σ. So:

    noise_floor = 1.128 · σ / (hi − lo)          (in the same 0–1 units as the warp residuals)

Two noisy images of the same scene can't match better than this. A warp residual **below** the noise floor means the flow is fitting noise.

---

## 4. Real-data scores (`scores.csv`, any source)

| column | computed as | read it as |
|---|---|---|
| `noise_floor` | above | the reference for the warp residuals |
| `pair_dt` | median t1 − t0, s | the time per pair (changes with gap and with bin size) |
| `warp_lane` | mean over pairs of mean\|I1(x + F) − I0\| on structure (out-of-frame → ignored) | how well the motion of the structure is explained. Lower is better, down to the noise floor |
| `warp_diffuse` | the same on diffuse glow | lower is better; well below the floor means fitting noise |
| `warp_excess_lane` | `warp_lane − noise_floor` | what the flow *can't* explain on the lanes. Grows with gap as lanes change shape. **Lower is better**; negative means noise-fitting |
| `speed_lane_p50`, `speed_lane_p95` | median / 95th percentile of \|F\| on structure, px/pair | how fast the structure moves (descriptive, not good or bad) |
| `speed_diffuse_p50` | median \|F\| on diffuse glow | **false motion** in glow. Lower is better |
| `lane_over_diffuse` | `speed_lane_p50 / speed_diffuse_p50` | real motion against background noise motion. **Higher is better** |
| `rough_diffuse` | mean over diffuse pixels of \|∂F/∂x\| + \|∂F/∂y\| (both components, 1-px differences) | how noisy or speckled the flow is in the glow. Lower is better |
| `star_over_ring` | median \|F\| on stars / median \|F\| in the ring | < 1: static stars pin the flow, as they should. ≈ 1: the flow ignores them |
| `temporal_change_lane` | median over structure (in both pairs) of \|F_{k+1} − F_k\|, divided by median lane speed. Only when stride = gap (consecutive pairs) | pair-to-pair jitter relative to the speed. Lower is better (aurora changes, but smoothly) |
| `*_s` columns | `speed_lane_p50_s`, `speed_lane_p95_s`, `speed_diffuse_p50_s`, `rough_diffuse_s` = the value / `pair_dt` | px/s versions. **Use these to compare runs with different gap or bin size** |
| pre-processing columns | `output`, `chain`, `<stage>.on`, `<stage>.<setting>` | present when the source is a pipeline output; used as heatmap axes |

---

## 5. Truth scores (benchmark sources only)

A benchmark (`aurora flow benchmark`) is real frames moved by a known flow T. It contains these cases per base frame:

| case | true motion |
|---|---|
| `rigid<s>` | uniform motion of `s` px/pair across the lanes (default 5, 15, 25), fading to 0 away from them |
| `shear` | across-lane speed ramping from 0 to 20 px/pair along the lanes |
| `vortex` | Rankine vortex on a lane (solid-body core, 1/r outside, peak ≤ 10 px/pair) |
| `realistic` *(opt.)* | a real flow (from `--realistic-flow`), smoothed until it can't fold over |

Each benchmark also has still glow with 2–3 patches that **brighten or dim without moving** (5–15%), fixed stars, and fresh noise matched to the source (level and spatial correlation).

Regions for truth scoring: **moving** = lanes ∧ \|T\| > 1 px ∧ edge; **fast** = moving ∧ \|T\| > 2; **still** = \|T\| < 0.2 ∧ not bright patch ∧ not stars ∧ edge.

| column | computed as | read it as |
|---|---|---|
| `epe_lane` | mean endpoint error \|F − T\| on moving lanes, px | **The main accuracy number.** Lower is better |
| `epe_lane_p95` | 95th percentile of the same | the worst errors (outliers, failures) |
| `epe_<case>` | `epe_lane` per case (`epe_rigid5`, `epe_shear`, `epe_vortex`, …) | which kind of motion a setting handles |
| `speed_recovery` | median \|F\| / \|T\| on fast lanes | 1 = full speed. < 1 = underestimates (over-smoothing) |
| `vorticity_corr`, `shear_corr` | Pearson correlation of vorticity / shear magnitude of the σ = 4-smoothed F and T, inside the moving area dilated by 21 px (pairs with ≥ 100 px) | whether the flow's *structure* (rotation, shear) is right, not only its speed. Higher is better |
| `structure_corr` | mean of the two | |
| `false_still` | median \|F\| in still glow | false motion where nothing moves. Lower is better |
| `false_bright` | median \|F\| in the brightening/dimming patches | **brightness change mistaken for motion.** Lower is better. TV-L1's `gamma` exists to reduce this |
| `drift_rank` | rank of `epe_lane` (1 = lowest) | |
| `structure_rank` | rank of `structure_corr` (1 = highest) | |
| `combined_rank` | mean of the two ranks | the single ordering used to pick the best run |

Your v4 benchmark example: `gamma3_lambda0.2` has the best `combined_rank`. `gamma3` has much lower `false_bright` than `gamma1` (0.74 vs 3.07 px), which shows `gamma` working as intended.

---

## 6. The heatmap figure (`scores_heatmaps.png`)

`plot_scores` picks a panel set:

| when | panels |
|---|---|
| benchmark | `epe_lane`, `epe_lane_p95`, `speed_recovery`, `structure_corr`, `false_still`, `false_bright`, `combined_rank` |
| time per pair differs (gap, stride or bin size varied) | `warp_excess_lane`, `speed_diffuse_p50_s`, `lane_over_diffuse`, `speed_lane_p50_s`, `speed_lane_p95_s`, `rough_diffuse_s` |
| otherwise | `warp_diffuse`, `speed_diffuse_p50`, `speed_lane_p95`, `temporal_change_lane`, `rough_diffuse`, `star_over_ring` |

- **Exactly two settings vary:** a heatmap, with the best cell outlined in red (for scores that have a direction).
- **Otherwise:** horizontal bars sorted by the score.
- Each panel title says "(lower better)" or "(higher better)".
- The noise floor is drawn as a dashed line on the diffuse warp panel.

---

## 7. Reading scores: a checklist

1. **Did the flow explain anything?** `warp_mae < zero_mae` on every pair (`summary.csv`).
2. **Is it fitting noise?** `warp_excess_lane` shouldn't be strongly negative, and `warp_diffuse` shouldn't sit far below `noise_floor`.
3. **False motion:** `speed_diffuse_p50(_s)`, `rough_diffuse(_s)`, `false_still`, and `false_bright` (benchmark): lower.
4. **Signal against noise:** `lane_over_diffuse` higher. `star_over_ring` < 1.
5. **Stability:** `temporal_change_lane` lower (contiguous pairs only).
6. **Accuracy (benchmark):** `epe_lane`, `speed_recovery` ≈ 1, `structure_corr` higher, then `combined_rank`.
7. **Comparing different gaps or bin sizes:** use the `_s` columns and `warp_excess_lane`, or match seconds per pair.

---

## 8. The test suite

Run all of it:

```bash
conda activate sp
cd ~/Developer/AuroralProcessing
python -m pytest tests                 # 43 tests, about 15 s
python -m pytest tests/test_flow.py -v # the flow tests, with names
python -m pytest tests -k benchmark    # tests matching a name
```

All 43 pass on this machine, including PIV and FLCT. Those two skip automatically if `openpiv` / `pyflct` aren't installed. Every test writes into pytest's own temporary folders and never touches your data.

### 8.1 Test fixtures (synthetic data)

| fixture (file) | what it is |
|---|---|
| `src` (test_flow) | 8 frames, 96 × 128, of a smooth random texture **translating by a known (1.5, −0.75) px per frame**, as uint16 with times 0.5 s apart. Ground truth for exact-motion checks. |
| `bench` (test_flow) | a benchmark built from `src` (base frames 2 and 5, speed 3, brightness 10%). |
| `aurora_src` (test_flow) | 6 frames, 160 × 160, aurora-like: diffuse glow, two dark diagonal lanes moving 1 px/frame, 25 point stars, slight noise. |
| `src` (test_pipeline) | 24 frames of random uint16 noise, 32 × 40, at 50 fps (pipeline mechanics only). |
| inline (test_pipeline, last test) | 12 frames, 96 × 112, of a texture shifting 1 px/frame plus noise. |

### 8.2 `tests/test_flow.py`: optical flow (28 tests)

**Settings and building blocks**

| test | checks |
|---|---|
| `test_params_round_trip_and_replace` | `TVL1Params` → dict → `TVL1Params` is lossless; `lambda`/`lambda_` naming; `replace`; unknown settings rejected. |
| `test_make_pairs` | Pair indices for gap, stride and start; no pairs when the gap is longer than the stack. |
| `test_scaler_outputs_tv_l1_types` | `FrameScaler` gives float32 0–1 (or uint8), clipping outside lo/hi, and works with numpy-scalar lo/hi as read from HDF5 attributes. |
| `test_tv_l1_float_input_is_0_to_1` | OpenCV TV-L1 gives the same flow for uint8 x and float32 x/255, which confirms the 0–1 float convention. |
| `test_block_mean` | Block averaging for quiver arrows: values and block centres. |

**Correctness of computed flow**

| test | checks |
|---|---|
| `test_recovers_known_shift` | On the translating texture, the median flow equals the true shift within 0.1 px. `warp_mae < zero_mae` on every pair, `fb_error < 0.2` px, and t1 − t0 = 0.5 s. |
| `test_correlation_recovers_known_shift[piv]`, `[flct]` | PIV and FLCT (gap 2) recover 2 × the shift within 0.25 px, **with the same +x right / +y down convention as TV-L1**. The config records the method, and `invalid_frac` is stored. |
| `test_parallel_matches_serial` | 2 workers give **bit-identical** flow to 1 worker. |

**Files, resuming, safety**

| test | checks |
|---|---|
| `test_resume_and_settings_mismatch` | A run interrupted after 3 pairs resumes and completes. Rerunning into the same file with different settings raises `FileExistsError` unless `overwrite`. |
| `test_repack_flow_to_gzip` | `aurora flow repack` turns LZF into gzip with identical data and attributes, and the repacked file is still recognised as the same complete run. |
| `test_outputs_into_new_folders` | Flow files and videos can be written into folders that don't exist yet. |
| `test_compare_names_runs_with_the_same_sweep_name` | `aurora flow compare` of two sweeps with the same name in different folders gives distinct links (`a-best-gamma1`, `b-best-gamma1`). |

**Videos**

| test | checks |
|---|---|
| `test_video[quiver]`, `test_video[speed]` | One video frame per pair for both kinds. |
| `test_video_with_moved_source` | `aurora flow video --source` works after the source file has moved. |

**Sweeps and CLI**

| test | checks |
|---|---|
| `test_sweep` | Python `run_sweep`: run names, `summary.csv` with complete runs and metric means. |
| `test_cli` | `aurora flow compute` with `--gap`, TV-L1 flags and `--video`: settings stored as attrs, right number of pairs, video written. |
| `test_cli_sweep` | `aurora flow sweep` with a grid and videos; base flags apply to all runs; no `.partial.mp4` left behind. |
| `test_sweep_over_gap_and_score` | A gap sweep: all gaps share the same first frames, and gap-3 flow ≈ 3 × gap-1 flow. Scoring gives the right `pair_dt` and per-second columns. |
| `test_sweep_over_stride` | Gap × stride sweep: correct pairs; `temporal_change_lane` only for stride = gap. |
| `test_correlation_params_and_warm_start` | PIV settings round-trip, `pass_windows` sequences, unknown PIV settings rejected, warm start refused for PIV. |
| `test_cli_piv_sweep_compare_and_score` | A PIV sweep and a TV-L1 sweep, gathered with `aurora flow compare`, scored together: both methods appear with warp scores and a heatmap. |

**Benchmarks and scoring**

| test | checks |
|---|---|
| `test_benchmark_layout` | Benchmark file layout: cases in order, `synth` uint16 with 2 frames per pair, truth and mask shapes, backward-map inversion residual < 0.1 px; refuses to overwrite. |
| `test_benchmark_truth_explains_frames` | **The truth is right.** With no noise or brightness change, warping b back by `true_flow` reproduces a: residual < 30% of \|b − a\| on moving pixels, stars excluded. |
| `test_cli_benchmark` | `aurora flow benchmark` with `--base-frames` and `--speeds` gives the expected cases. |
| `test_score_sweep_with_truth` | Scoring a benchmark sweep fills every truth column (`epe_lane`, `speed_recovery`, `structure_corr`, `false_bright`, `epe_rigid3`, `combined_rank`) and the heatmap. |
| `test_score_real_sweep_and_cli` | Scoring a real-data sweep: 4 runs, real-data columns present, no truth columns, heatmap written. |

### 8.3 `tests/test_pipeline.py`: pre-processing, and flow on pipeline outputs (15 tests)

| test | checks |
|---|---|
| `test_from_dict_does_not_modify_input_and_casts` | Building a stage leaves your dict unchanged and is repeatable; values are cast (`"1e-3"`, `5.0` → 5); `noise: auto`; `null`; bad names, settings and types rejected. |
| `test_to_uint16_full_range_maps_high_to_65535` | `high` maps to 65535, **not 0** (the `2**16` overflow); `out_max = 2**16` is rejected. |
| `test_spatial_median_large_kernel` | k = 3, 5, 7, 9 all run (OpenCV vs SciPy paths). |
| `test_old_stage_name_alias_and_normalize_checks` | The old `sliding_temporal_medial` name still works; log curve with vmin ≤ 0 and bad curve names are rejected. |
| `test_wiener_auto_fixes_one_noise_value` | `noise: auto` estimates one positive noise value and passes every frame through. |
| `test_fit_stage_must_be_first` | `normalize` without limits after another stage raises an error; with explicit limits it's allowed. |
| `test_write_h5_keeps_float_and_refuses_silent_overwrite` | Float frames stay float (no truncation); an existing dataset needs `overwrite=True`. |
| `test_hash_ignores_defaults_ids_and_output_folder` | Explicit defaults, ids and the source's folder don't change the hash; the source name and settings do; disabled stages are dropped. Checks automatic labels. |
| `test_run_chain_writes_skips_and_forces` | An output folder with the right frames, times and attrs (unfitted and fitted chains); the source is untouched; a rerun is skipped; a changed chain gets a new folder; the same chain under another label reuses the folder; a new output folder gives the same hash. |
| `test_apply_processing_json_and_overwrite_guard` | The older in-file interface stores JSON and refuses to overwrite. |
| `test_overrides_and_grid` | Dotted overrides, on/off, `id`s for repeated stages, errors for unknown stages or settings, grid expansion ignoring keys not in the chain, value parsing. |
| `test_job_file_expands_sources_chains_grid` | A job file with a file and a folder source, a chain file and an inline chain, and a grid: 15 outputs. Labels; dry run writes nothing; run, then rerun → all `exists`; `index.csv` complete. |
| `test_job_errors` | Unknown job keys, misspelt grid keys, missing sources and malformed chains give clear errors. |
| `test_cli_run_and_list` | `aurora pipeline run` with `--grid`, `--set` labels, `aurora pipeline list`, missing `--chain` exits. |
| `test_flow_on_pipeline_outputs_compare_and_score` | Flow sweeps inside two pipeline outputs (Wiener on/off) → `compare` makes 2 distinct links → `score` has `wiener.on` as a varied setting, and **each run is scored against its own output's frames** (different noise floors). |

### 8.4 What the tests don't cover

- **Real auroral accuracy.** Tests use synthetic textures. Accuracy on real aurora is what the benchmark plus scoring workflow is for.
- **Speed or performance.** No timing assertions.
- **MATLAB reading repacked files.** Only the gzip conversion itself is tested.
- **Score values as "correct".** The tests check that the scores exist and are internally consistent (e.g. the gap scaling). They don't check thresholds for good or bad flow.

Adding a test: put it in `tests/test_flow.py` or `tests/test_pipeline.py`, reuse a fixture (`src`, `bench`, `aurora_src`), write into `tmp_path`, and run `python -m pytest tests -k <name>`.
