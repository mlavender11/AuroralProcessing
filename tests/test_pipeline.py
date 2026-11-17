import json
import numpy as np
import pandas as pd
import h5py
import pytest

from auroral_processing.pipeline.stages import Stage, ToUint16, SpatialMedian, Wiener, Normalize, STAGES
from auroral_processing.pipeline.processing_chain import (
    run_chain, chain_hash, canonical_chain, auto_label, fit_stages, build_stages, apply_processing,
    read_output_attrs, OUTPUT_FILE,
)
from auroral_processing.pipeline.drivers import write_h5
from auroral_processing.pipeline.config import (
    apply_overrides, expand_grid, load_chain, load_job, parse_grid, parse_assignments, resolve_sources,
)
from auroral_processing.pipeline.jobs import build_index, run_job
from auroral_processing.cli import main

N, H, W = 24, 32, 40
CHAIN = [
    {"stage": "normalize"},
    {"stage": "binned_temporal_average", "bin_size": 4},
    {"stage": "spatial_median", "k": 3},
    {"stage": "wiener"},
    {"stage": "to_uint16"},
]


def _write_src(path, seed=0):
    rng = np.random.default_rng(seed)
    frames = (2000 + 800 * rng.random((N, H, W))).astype(np.uint16)
    with h5py.File(path, "w") as f:
        f["rawimg"] = frames
        f["ut1_unix"] = 1.364e9 + np.arange(N) / 50
    return path


@pytest.fixture
def src(tmp_path):
    return _write_src(tmp_path / "2013-03-27-CamTest.h5")


# ---------- stages ----------
def test_from_dict_does_not_modify_input_and_casts():
    d = {"stage": "wiener", "noise": "1e-3", "size": 5.0, "id": "w1", "enabled": True}
    s = Stage.from_dict(d)
    assert d == {"stage": "wiener", "noise": "1e-3", "size": 5.0, "id": "w1", "enabled": True}
    assert s.noise == 1e-3 and s.size == 5 and isinstance(s.size, int)
    assert Stage.from_dict(d) == s  # reusable
    assert Stage.from_dict({"stage": "wiener", "noise": "auto"}).noise == "auto"
    assert Stage.from_dict({"stage": "normalize", "vmin": "null"}).vmin is None
    with pytest.raises(ValueError, match="unknown settings"):
        Stage.from_dict({"stage": "spatial_median", "kk": 3})
    with pytest.raises(ValueError, match="not a valid"):
        Stage.from_dict({"stage": "binned_temporal_average", "bin_size": 2.5})
    with pytest.raises(ValueError, match="unknown stage"):
        Stage.from_dict({"stage": "nope"})


def test_to_uint16_full_range_maps_high_to_65535():
    out = [f for _, f in ToUint16().run([(0.0, np.array([[0.0, 0.5, 1.0, 2.0]]))])][0]
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, [[0, 32768, 65535, 65535]])
    with pytest.raises(ValueError):
        ToUint16(out_max=2**16)


def test_spatial_median_large_kernel():
    frame = np.random.default_rng(0).random((20, 20)).astype(np.float32)
    for k in (3, 5, 7, 9):
        (_, out), = SpatialMedian(k=k).run([(0.0, frame)])
        assert out.shape == frame.shape


def test_old_stage_name_alias_and_normalize_checks():
    assert STAGES["sliding_temporal_medial"] is STAGES["sliding_temporal_median"]
    with pytest.raises(ValueError, match="vmin > 0"):
        Normalize(vmin=0, vmax=10, curve="log")
    with pytest.raises(ValueError, match="curve"):
        Normalize(curve="sqrtt")


def test_wiener_auto_fixes_one_noise_value():
    rng = np.random.default_rng(1)
    w = Wiener(noise="auto")
    out = list(w.run((float(i), rng.random((16, 16))) for i in range(8)))
    assert len(out) == 8 and isinstance(w.noise, float) and w.noise > 0


def test_fit_stage_must_be_first():
    stages = build_stages([{"stage": "spatial_median"}, {"stage": "normalize"}])
    with pytest.raises(ValueError, match="must be"):
        fit_stages(stages, None, None)
    fit_stages(build_stages([{"stage": "spatial_median"}, {"stage": "normalize", "vmin": 1, "vmax": 2}]), None, None)


def test_write_h5_keeps_float_and_refuses_silent_overwrite(tmp_path):
    with h5py.File(tmp_path / "x.h5", "w") as f:
        write_h5([(0.0, np.full((2, 2), 0.5, np.float32))], f, "a", progress=False)
        assert f["a"].dtype == np.float32 and f["a"][0, 0, 0] == 0.5
        with pytest.raises(FileExistsError):
            write_h5([(0.0, np.zeros((2, 2)))], f, "a", progress=False)
        write_h5([(0.0, np.zeros((2, 2)))], f, "a", overwrite=True, progress=False)


# ---------- hashing and run_chain ----------
def test_hash_ignores_defaults_ids_and_output_folder():
    explicit = [{"stage": "spatial_median", "k": 3, "id": "m"}, {"stage": "to_uint16", "out_max": 65535}]
    short = [{"stage": "spatial_median"}, {"stage": "to_uint16"}]
    assert chain_hash(explicit, "a.h5") == chain_hash(short, "/some/where/a.h5")
    assert chain_hash(short, "a.h5") != chain_hash(short, "b.h5")
    assert chain_hash(short, "a.h5") != chain_hash([{"stage": "spatial_median", "k": 5}, {"stage": "to_uint16"}], "a.h5")
    assert auto_label(CHAIN) == "norm_bin4_med3_wiener"
    disabled = [*short, {"stage": "wiener", "enabled": False}]
    assert canonical_chain(disabled) == canonical_chain(short)


def test_run_chain_writes_skips_and_forces(src, tmp_path):
    out_root = tmp_path / "processed"
    out = run_chain(src, CHAIN, out_root, progress=False)
    assert out.parent == out_root / src.stem and out.name.startswith("norm_bin4_med3_wiener-")
    with h5py.File(out / OUTPUT_FILE) as f:
        assert f["frames"].shape == (N // 4, H, W) and f["frames"].dtype == np.uint16
        assert len(f["frames_ut1_unix"]) == N // 4
        assert json.loads(f["frames"].attrs["preprocessing_chain"])[0]["vmin"] is not None
    with h5py.File(src) as f:
        assert set(f) == {"rawimg", "ut1_unix"}  # source untouched
    a = read_output_attrs(out)
    assert a["chain"] == canonical_chain(CHAIN) and a["chain"][0]["vmin"] is None
    assert a["chain_fitted"][0]["vmin"] is not None and a["complete"] and a["source_name"] == src.name
    created = a["created"]

    run_chain(src, CHAIN, out_root, progress=False)  # skipped: unchanged
    assert read_output_attrs(out)["created"] == created
    assert not (out / "frames.partial.h5").exists()

    other = run_chain(src, apply_overrides(CHAIN, {"binned_temporal_average.bin_size": 2}), out_root, progress=False)
    assert other != out and read_output_attrs(other)["n_frames"] == N // 2

    elsewhere = run_chain(src, CHAIN, tmp_path / "elsewhere", label="mine", progress=False)
    assert elsewhere.name.split("-")[-1] == out.name.split("-")[-1] and elsewhere.name.startswith("mine-")


def test_apply_processing_json_and_overwrite_guard(src):
    apply_processing(src, CHAIN, "v1")
    with h5py.File(src) as f:
        assert json.loads(f["v1"].attrs["preprocessing_chain"])[1]["bin_size"] == 4
    with pytest.raises(FileExistsError):
        apply_processing(src, CHAIN, "v1")
    apply_processing(src, CHAIN, "v1", overwrite=True)


# ---------- YAML, overrides, grids, jobs ----------
def test_overrides_and_grid():
    c = apply_overrides(CHAIN, {"binned_temporal_average.bin_size": 8, "wiener": "off"})
    assert c[1]["bin_size"] == 8 and c[3]["enabled"] is False and CHAIN[1]["bin_size"] == 4
    with pytest.raises(ValueError, match="no stage"):
        apply_overrides(CHAIN, {"gaussian.sigma": 1})
    with pytest.raises(ValueError, match="unknown settings"):
        apply_overrides(CHAIN, {"spatial_median.size": 5})

    twice = [{"stage": "spatial_median", "id": "pre"}, {"stage": "spatial_median", "id": "post", "k": 5}]
    with pytest.raises(ValueError, match="give them an id"):
        apply_overrides([{"stage": "spatial_median"}, {"stage": "spatial_median"}], {"spatial_median.k": 5})
    assert apply_overrides(twice, {"post.k": 7})[1]["k"] == 7

    grid = parse_grid(["binned_temporal_average.bin_size=2,4", "wiener=on,off", "sliding_temporal_median.window=3,5"])
    combos = expand_grid(CHAIN, grid)
    assert len(combos) == 4  # the window key isn't in this chain, so it is ignored
    assert parse_assignments(["a.b=1e-3", "w=off", "x.y=null"]) == {"a.b": "1e-3", "w": False, "x.y": None}


def test_job_file_expands_sources_chains_grid(src, tmp_path):
    folder = tmp_path / "night"
    folder.mkdir()
    _write_src(folder / "a.h5", 1)
    _write_src(folder / "b.h5", 2)
    (tmp_path / "chains").mkdir()
    (tmp_path / "chains" / "base.yaml").write_text(
        "- stage: normalize\n- stage: binned_temporal_average\n  bin_size: 4\n- stage: wiener\n- stage: to_uint16\n"
    )
    (tmp_path / "job.yaml").write_text(f"""
output: out
sources:
  - {src.name}
  - dir: night
chains:
  - chains/base.yaml
  - label: nobin
    stages:
      - stage: normalize
      - stage: to_uint16
grid:
  binned_temporal_average.bin_size: [2, 4]
  wiener: [on, off]
""")
    job = load_job(tmp_path / "job.yaml")
    assert job.output == tmp_path / "out" and [p.name for p in job.sources] == [src.name, "a.h5", "b.h5"]
    tasks = job.tasks()
    assert len(tasks) == 3 * (4 + 1)  # base: 2 bins x wiener on/off; nobin has neither stage
    labels = {t.label for t in tasks}
    assert "base_bin_size2_nowiener" in labels and "nobin" in labels

    plan = run_job(job, dry_run=True, progress=False)
    assert (plan["status"] == "would run").all() and not job.output.exists()
    done = run_job(job, progress=False)
    assert (done["status"] == "done").all()
    again = run_job(job, progress=False)
    assert (again["status"] == "exists").all()

    index = pd.read_csv(job.output / "index.csv")
    assert len(index) == 15 and index["complete"].all()
    assert {"binned_temporal_average.bin_size", "normalize.vmin"} <= set(index.columns)


def test_job_errors(tmp_path, src):
    (tmp_path / "bad.yaml").write_text(f"sources: [{src}]\nchains:\n  - [{{stage: normalize}}]\ngird: {{}}\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_job(tmp_path / "bad.yaml")
    (tmp_path / "typo.yaml").write_text(f"sources: [{src}]\nchains:\n  - [{{stage: normalize}}]\ngrid: {{wienr: [on, off]}}\n")
    with pytest.raises(ValueError, match="no chain has that stage"):
        load_job(tmp_path / "typo.yaml").tasks()
    with pytest.raises(FileNotFoundError):
        resolve_sources(["missing.h5"], tmp_path)
    with pytest.raises(ValueError, match="'stage' key"):
        load_chain([{"bin_size": 3}])


def test_cli_run_and_list(src, tmp_path, capsys):
    chain = tmp_path / "c.yaml"
    chain.write_text("- stage: normalize\n- stage: binned_temporal_average\n  bin_size: 4\n- stage: to_uint16\n")
    out = tmp_path / "processed"
    main(["pipeline", "run", str(src), "--chain", str(chain), "--out", str(out), "--grid",
          "binned_temporal_average.bin_size=2,4", "-q"])
    folders = sorted(p.name for p in (out / src.stem).iterdir())
    assert len(folders) == 2 and all(f.startswith("c_bin_size") for f in folders)
    main(["pipeline", "list", str(out)])
    assert "c_bin_size2" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["pipeline", "run", str(src), "--out", str(out)])  # no chain


def test_flow_on_pipeline_outputs_compare_and_score(tmp_path):
    """Flow sweeps inside two pipeline outputs, gathered with compare and scored with their pre-processing."""
    import cv2
    from auroral_processing.flow.score import varied_settings

    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.random((96, 112)).astype(np.float32), (0, 0), 3)
    frames = [np.roll(base, n, axis=1) * 3000 + 500 + rng.normal(0, 20, base.shape) for n in range(12)]
    src = tmp_path / "moving.h5"
    with h5py.File(src, "w") as f:
        f["rawimg"] = np.clip(frames, 0, 65535).astype(np.uint16)
        f["ut1_unix"] = 1.364e9 + np.arange(12) * 0.5

    chain = [{"stage": "normalize"}, {"stage": "binned_temporal_average", "bin_size": 2}, {"stage": "wiener"},
             {"stage": "to_uint16"}]
    runs = []
    for wiener in (True, False):
        out = run_chain(src, apply_overrides(chain, {"wiener": wiener}), tmp_path / "processed", progress=False)
        main(["flow", "sweep", str(out / "frames.h5"), str(out / "flow" / "s"), "--frames", "frames", "--times",
              "frames_ut1_unix", "--tvl1-scales", "2", "--grid", "gamma=0", "--video"])
        runs.append(str(out / "flow" / "s" / "gamma0"))
    main(["flow", "compare", str(tmp_path / "cmp"), *runs])
    assert len(list((tmp_path / "cmp").iterdir())) == 2  # same sweep/run names, distinct links
    main(["flow", "score", str(tmp_path / "cmp"), "--no-plot"])
    df = pd.read_csv(tmp_path / "cmp" / "scores.csv")
    assert sorted(df["wiener.on"]) == [False, True] and "wiener.on" in varied_settings(df)
