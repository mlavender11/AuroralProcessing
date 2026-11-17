import numpy as np
import pandas as pd
import h5py
import imageio
import cv2
import pytest

from auroral_processing.flow.params import TVL1Params, make_tvl1
from auroral_processing.flow.compute_flow import compute_flow, make_pairs, fit_scaler, FrameScaler
from auroral_processing.flow.sweep import run_sweep, run_name
from auroral_processing.products.flow_video import make_flow_video, block_mean
from auroral_processing.cli import main

SHIFT = (1.5, -0.75)  # px per frame, (dx, dy)
N, H, W = 8, 96, 128
FAST = TVL1Params(scales=3, warps=3)


def _smooth_texture(rng, h, w):
    return cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 3)


@pytest.fixture
def src(tmp_path):
    """uint16 frames of a texture translating by SHIFT each frame, unix-second times."""
    base = _smooth_texture(np.random.default_rng(0), H, W)
    frames = [
        cv2.warpAffine(base, np.float32([[1, 0, SHIFT[0] * n], [0, 1, SHIFT[1] * n]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        for n in range(N)
    ]
    frames = (np.array(frames) / base.max() * 3000 + 500).astype(np.uint16)
    path = tmp_path / "src.h5"
    with h5py.File(path, "w") as f:
        f["img"] = frames
        f["t"] = 1.364e9 + np.arange(N) * 0.5
    return path


KW = dict(frames="img", times="t", params=FAST, progress=False)


def test_params_round_trip_and_replace():
    p = TVL1Params(lambda_=0.3, gamma=1)
    assert p.to_dict()["lambda"] == 0.3
    assert TVL1Params.from_dict(p.to_dict()) == p
    assert p.replace(**{"lambda": 0.05}).lambda_ == 0.05
    with pytest.raises(ValueError):
        TVL1Params.from_dict({"lamda": 1})


def test_make_pairs():
    np.testing.assert_array_equal(make_pairs(10, gap=4), [[0, 4], [4, 8]])
    np.testing.assert_array_equal(make_pairs(10, gap=2, stride=1, start=5), [[5, 7], [6, 8], [7, 9]])
    assert len(make_pairs(3, gap=3)) == 0


def test_scaler_outputs_tv_l1_types():
    frame = np.array([[0, 100], [200, 1000]], dtype=np.uint16)
    f32 = FrameScaler(np.float64(100), np.float64(200))(frame)  # numpy scalars, as read from HDF5 attrs
    assert f32.dtype == np.float32
    np.testing.assert_allclose(f32, [[0, 0], [1, 1]])
    assert FrameScaler(100, 200, "uint8")(frame).dtype == np.uint8


def test_tv_l1_float_input_is_0_to_1():
    """OpenCV scales float32 input by 255, so uint8 x and float32 x/255 give the same flow."""
    rng = np.random.default_rng(1)
    a = (_smooth_texture(rng, 64, 64) * 255 / 0.6).clip(0, 255).astype(np.uint8)
    b = np.roll(a, 1, axis=1)
    tvl1 = make_tvl1(FAST)
    np.testing.assert_allclose(
        tvl1.calc(a, b, None), tvl1.calc(a.astype(np.float32) / 255, b.astype(np.float32) / 255, None), atol=1e-3
    )


def test_recovers_known_shift(src):
    out = compute_flow(src, metrics=("warp_mae", "zero_mae", "fb_error"), **KW)
    with h5py.File(out) as f:
        assert f.attrs["complete"]
        flow = f["flow"][:]
        assert flow.shape == (N - 1, H, W, 2)
        median = np.median(flow[:, 16:-16, 16:-16], axis=(1, 2))
        np.testing.assert_allclose(median, np.broadcast_to(SHIFT, median.shape), atol=0.1)
        m = {k: f["metrics"][k][:] for k in f["metrics"]}
        assert (m["warp_mae"] < m["zero_mae"]).all()
        assert (m["fb_error"] < 0.2).all()
        np.testing.assert_array_equal(f["t1"][:] - f["t0"][:], 0.5)


def test_parallel_matches_serial(src, tmp_path):
    serial = compute_flow(src, tmp_path / "serial.h5", **KW)
    parallel = compute_flow(src, tmp_path / "parallel.h5", workers=2, chunk_size=2, **KW)
    with h5py.File(serial) as a, h5py.File(parallel) as b:
        np.testing.assert_array_equal(a["flow"][:], b["flow"][:])


def test_resume_and_settings_mismatch(src, tmp_path):
    out = tmp_path / "flow.h5"
    compute_flow(src, out, **KW)
    with h5py.File(out, "r+") as f:  # simulate an interrupted run
        f["done"][3:] = False
        f["flow"][3:] = np.nan
        f.attrs["complete"] = False
    compute_flow(src, out, **KW)
    with h5py.File(out) as f:
        assert f.attrs["complete"]
        assert np.isfinite(f["flow"][:]).all()

    with pytest.raises(FileExistsError):
        compute_flow(src, out, gap=2, **KW)
    compute_flow(src, out, gap=2, overwrite=True, **KW)


@pytest.mark.parametrize("kind", ["quiver", "speed"])
def test_video(src, kind):
    flow = compute_flow(src, **KW)
    video = make_flow_video(flow, kind=kind, grid=8, progress=False)
    with imageio.get_reader(video) as r:
        assert r.count_frames() == N - 1


def test_block_mean():
    flow = np.zeros((10, 12, 2), np.float32)
    flow[:5, :, 0] = 2
    X, Y, u, v = block_mean(flow, 5)
    assert u.shape == (2, 2)
    np.testing.assert_array_equal(u, [[2, 2], [0, 0]])
    np.testing.assert_array_equal(X[0], [2, 7])


def test_sweep(src, tmp_path):
    grid = {"gamma": [0, 1], "lambda": [0.15]}
    kw = {k: v for k, v in KW.items() if k != "params"}
    summary = run_sweep(src, grid, tmp_path / "sweep", base_params=FAST, videos=(), **kw)
    df = pd.read_csv(summary)
    assert list(df["run"]) == ["gamma0_lambda0.15", "gamma1_lambda0.15"]
    assert df["complete"].all() and "warp_mae_mean" in df
    assert run_name({"lambda_": 0.05}) == "lambda0.05"


def test_cli(src, tmp_path):
    out = tmp_path / "cli.h5"
    main(["flow", "compute", str(src), "--frames", "img", "--times", "t", "--gap", "2",
          "--tvl1-gamma", "1", "--tvl1-scales", "3", "-o", str(out), "--video", "speed"])
    with h5py.File(out) as f:
        assert f.attrs["tvl1_gamma"] == 1 and f.attrs["tvl1_scales"] == 3
        assert len(f["i0"]) == 3
    assert out.with_name("cli_speed.mp4").exists()


def test_cli_sweep(src, tmp_path):
    outdir = tmp_path / "sweep"
    main(["flow", "sweep", str(src), str(outdir), "--frames", "img", "--times", "t", "--gap", "3",
          "--tvl1-scales", "3", "--grid", "gamma=0,1", "--grid", "lambda=0.15", "--video", "quiver", "--grid-arrows", "8"])
    df = pd.read_csv(outdir / "summary.csv")
    assert list(df["run"]) == ["gamma0_lambda0.15", "gamma1_lambda0.15"]
    assert (df["scales"] == 3).all()
    assert (outdir / "gamma1_lambda0.15" / "flow_quiver.mp4").exists()
    assert not list(outdir.rglob("*.partial.mp4"))


# ---- synthetic benchmark and scoring ----

from auroral_processing.flow.benchmark import make_benchmark
from auroral_processing.flow.score import score_sweep


@pytest.fixture
def bench(src, tmp_path):
    return make_benchmark(src, tmp_path / "bench.h5", frames="img", base_frames=[2, 5], speeds=(3,), brightness=(0.1, 0.1))


def test_benchmark_layout(bench):
    with h5py.File(bench) as f:
        P = len(f["case"])
        assert list(f["case"].asstr()[:]) == ["rigid3", "shear", "vortex"] * 2
        assert f["synth"].shape == (2 * P, H, W) and f["synth"].dtype == np.uint16
        assert f["true_flow"].shape == (P, H, W, 2)
        assert f["masks/lanes"].shape == f["masks/bright_patch"].shape == (P, H, W)
        assert f.attrs["inversion_residual"] < 0.1
    with pytest.raises(FileExistsError):
        make_benchmark(bench.parent / "src.h5", bench, frames="img", base_frames=[2])


@pytest.fixture
def aurora_src(tmp_path):
    """Small aurora-like stack: diffuse glow, two dark diagonal lanes, point stars, slight noise."""
    h, w = 160, 160
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    rng = np.random.default_rng(2)
    glow = 1000 + 400 * np.exp(-((xx - 70) ** 2 + (yy - 90) ** 2) / (2 * 60**2))
    stars = np.zeros((h, w), np.float32)
    stars[rng.integers(5, h - 5, 25), rng.integers(5, w - 5, 25)] = 4000
    stars = cv2.GaussianBlur(stars, (0, 0), 1)
    frames = []
    for n in range(6):
        d = (xx - 0.6 * yy) / np.hypot(1, 0.6)
        lanes = 250 * (np.exp(-((d - 20 - n) ** 2) / (2 * 4**2)) + np.exp(-((d + 25 - n) ** 2) / (2 * 6**2)))
        frames.append(glow - lanes + stars + rng.normal(0, 5, (h, w)))
    path = tmp_path / "aurora.h5"
    with h5py.File(path, "w") as f:
        f["img"] = np.clip(frames, 0, 65535).astype(np.uint16)
    return path


def test_benchmark_truth_explains_frames(aurora_src, tmp_path):
    """Without noise or brightness changes, warping b back by the true flow reproduces a."""
    out = make_benchmark(aurora_src, tmp_path / "clean.h5", frames="img", base_frames=[3], speeds=(3, 8),
                         brightness=None, noise=(0, 0))
    with h5py.File(out) as f:
        S, T, stars = f["synth"][:].astype(np.float32), f["true_flow"][:], f["masks/stars"][:]
    yy, xx = np.mgrid[: S.shape[1], : S.shape[2]].astype(np.float32)
    check = ~cv2.dilate(stars.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)  # stars stay put
    check[:12], check[-12:], check[:, :12], check[:, -12:] = False, False, False, False
    for k in range(len(T)):
        a, b = S[2 * k], S[2 * k + 1]
        back = cv2.remap(b, xx + T[k, ..., 0], yy + T[k, ..., 1], cv2.INTER_CUBIC)
        m = check & (np.hypot(T[k, ..., 0], T[k, ..., 1]) > 0.5)
        assert m.sum() > 200
        assert np.abs(back - a)[m].mean() < 0.3 * np.abs(b - a)[m].mean()


def test_score_sweep_with_truth(bench, tmp_path):
    kw = {k: v for k, v in KW.items() if k not in ("params", "frames", "times")}
    run_sweep(bench, {"lambda": [0.05, 0.3]}, tmp_path / "sweep", base_params=FAST, videos=(),
              frames="synth", times="synth_t", gap=1, stride=2, **kw)
    df = pd.read_csv(score_sweep(tmp_path / "sweep", progress=False))
    assert list(df["run"]) == ["lambda0.05", "lambda0.3"]
    for col in ("epe_lane", "speed_recovery", "structure_corr", "false_bright", "epe_rigid3", "combined_rank", "warp_diffuse"):
        assert df[col].notna().all(), col
    assert (tmp_path / "sweep" / "scores_heatmaps.png").exists()


def test_score_real_sweep_and_cli(src, tmp_path):
    outdir = tmp_path / "sweep"
    main(["flow", "sweep", str(src), str(outdir), "--frames", "img", "--times", "t", "--tvl1-scales", "3",
          "--grid", "gamma=0,1", "--grid", "lambda=0.1,0.3", "--video"])
    main(["flow", "score", str(outdir)])
    df = pd.read_csv(outdir / "scores.csv")
    assert len(df) == 4 and "temporal_change_lane" in df and "epe_lane" not in df
    assert (outdir / "scores_heatmaps.png").exists()


def test_cli_benchmark(src, tmp_path):
    out = tmp_path / "b.h5"
    main(["flow", "benchmark", str(src), "-o", str(out), "--frames", "img", "--base-frames", "3", "--speeds", "2", "4"])
    with h5py.File(out) as f:
        assert list(f["case"].asstr()[:]) == ["rigid2", "rigid4", "shear", "vortex"]


def test_sweep_over_gap_and_score(src, tmp_path):
    outdir = tmp_path / "sweep"
    main(["flow", "sweep", str(src), str(outdir), "--frames", "img", "--times", "t", "--tvl1-scales", "3",
          "--grid", "gap=1,3", "--grid", "scale_step=0.5,0.8", "--video", "quiver", "--vmax", "2"])
    df = pd.read_csv(outdir / "summary.csv")
    assert list(df["run"]) == ["gap1_scale_step0.5", "gap1_scale_step0.8", "gap3_scale_step0.5", "gap3_scale_step0.8"]
    assert list(df["gap"]) == [1, 1, 3, 3]
    with h5py.File(outdir / "gap1_scale_step0.5" / "flow.h5") as a, h5py.File(outdir / "gap3_scale_step0.5" / "flow.h5") as b:
        np.testing.assert_array_equal(a["i0"][:], b["i0"][:])  # same first frames, stride = largest gap
        np.testing.assert_array_equal(b["i1"][:] - b["i0"][:], 3)
        flow1, flow3 = a["flow"][:], b["flow"][:]
    assert np.median(flow3[..., 0]) == pytest.approx(3 * np.median(flow1[..., 0]), rel=0.2)
    scores = pd.read_csv(score_sweep(outdir, progress=False))
    assert scores["pair_dt"].tolist() == pytest.approx([0.5, 0.5, 1.5, 1.5])
    assert scores["speed_lane_p50_s"].notna().all() and "lane_over_diffuse" in scores
    assert (outdir / "scores_heatmaps.png").exists()


def test_sweep_over_stride(src, tmp_path):
    outdir = tmp_path / "sweep"
    main(["flow", "sweep", str(src), str(outdir), "--frames", "img", "--times", "t", "--tvl1-scales", "3",
          "--grid", "gap=1,2", "--grid", "stride=1,2", "--video"])
    df = pd.read_csv(outdir / "summary.csv")
    assert list(df["stride"]) == [1, 2, 1, 2] and list(df["gap"]) == [1, 1, 2, 2]
    with h5py.File(outdir / "gap2_stride1" / "flow.h5") as f:
        assert list(f["i0"][:]) == [0, 1, 2, 3, 4, 5]  # stop shortened so gap 1 and 2 share i0
    scores = pd.read_csv(score_sweep(outdir, progress=False))
    assert len(scores) == 4 and scores["temporal_change_lane"].notna().sum() == 2  # only stride == gap


def test_outputs_into_new_folders(src, tmp_path):
    flow = compute_flow(src, tmp_path / "a" / "b" / "flow.h5", **KW)
    assert flow.exists()
    assert make_flow_video(flow, tmp_path / "c" / "v.mp4", grid=8, progress=False).exists()


def test_video_with_moved_source(src, tmp_path):
    flow = compute_flow(src, tmp_path / "f.h5", **KW)
    moved = tmp_path / "moved.h5"
    src.rename(moved)
    out = tmp_path / "v.mp4"
    main(["flow", "video", str(flow), "-o", str(out), "--source", str(moved), "--grid-arrows", "8"])
    assert out.exists()


def test_repack_flow_to_gzip(src, tmp_path):
    flow = compute_flow(src, tmp_path / "f.h5", **KW)
    with h5py.File(flow) as f:
        assert f["flow"].compression == "lzf"
        data = f["flow"][:]
    main(["flow", "repack", str(flow)])
    with h5py.File(flow) as f:
        assert f["flow"].compression == "gzip"
        np.testing.assert_array_equal(f["flow"][:], data)
        assert f.attrs["complete"] and "config" in f.attrs and "warp_mae" in f["metrics"]
    assert compute_flow(src, flow, **KW) == flow  # still recognised as the same, complete run


# ---- correlation methods (OpenPIV, FLCT) ----

from auroral_processing.flow.params import PIVParams, FLCTParams

PIV_FAST = PIVParams(window=8, first_window=32)  # 32 -> 16 -> 8 px windows


@pytest.mark.parametrize("params", [PIV_FAST, FLCTParams(sigma=10)], ids=["piv", "flct"])
def test_correlation_recovers_known_shift(src, tmp_path, params):
    pytest.importorskip({"piv": "openpiv", "flct": "pyflct"}[params.METHOD])
    out = compute_flow(src, tmp_path / "f.h5", **{**KW, "params": params}, gap=2)
    with h5py.File(out) as f:
        flow = f["flow"][:, 16:-16, 16:-16]
        cfg = __import__("json").loads(f.attrs["config"])
        assert cfg["method"] == params.METHOD and params.METHOD in cfg
        assert f"{params.METHOD}_{list(params.to_dict())[0]}" in f.attrs
        assert np.isfinite(f["metrics/invalid_frac"][:]).all()
    # +x right, +y down, the same convention as TV-L1
    assert np.median(flow[..., 0]) == pytest.approx(2 * SHIFT[0], abs=0.25)
    assert np.median(flow[..., 1]) == pytest.approx(2 * SHIFT[1], abs=0.25)


def test_correlation_params_and_warm_start(src, tmp_path):
    assert PIVParams.from_dict(PIV_FAST.to_dict()) == PIV_FAST
    assert PIV_FAST.replace(window=12).window == 12
    from auroral_processing.flow.methods import pass_windows
    assert pass_windows(16, 64) == (64, 32, 16) and pass_windows(12, 64) == (64, 32, 16, 12)
    assert pass_windows(24, 64) == (64, 32, 24) and pass_windows(16, 16) == (16,)
    with pytest.raises(ValueError, match="unknown PIV setting"):
        PIVParams.from_dict({"gamma": 1})
    with pytest.raises(ValueError, match="only supported for TV-L1"):
        compute_flow(src, tmp_path / "f.h5", **{**KW, "params": PIV_FAST}, warm_start=True)


def test_cli_piv_sweep_compare_and_score(src, tmp_path):
    pytest.importorskip("openpiv")
    main(["flow", "sweep", str(src), str(tmp_path / "piv"), "--frames", "img", "--times", "t", "--method", "piv",
          "--piv-first-window", "32", "--grid", "window=8,16", "--video", "quiver", "--grid-arrows", "8"])
    df = pd.read_csv(tmp_path / "piv" / "summary.csv")
    assert list(df["run"]) == ["window8", "window16"] and (df["method"] == "piv").all()
    assert (df["first_window"] == 32).all() and "invalid_frac_mean" in df
    main(["flow", "sweep", str(src), str(tmp_path / "tvl1"), "--frames", "img", "--times", "t",
          "--tvl1-scales", "3", "--grid", "gamma=1", "--video"])
    main(["flow", "compare", str(tmp_path / "both"), str(tmp_path / "piv" / "window8"), str(tmp_path / "tvl1" / "gamma1")])
    scores = pd.read_csv(score_sweep(tmp_path / "both", progress=False))
    assert sorted(scores["run"]) == ["piv-window8", "tvl1-gamma1"]
    assert sorted(scores["method"]) == ["piv", "tvl1"] and scores["warp_lane"].notna().all()
    assert (tmp_path / "both" / "scores_heatmaps.png").exists()
