"""Flow job files: aurora flow run."""

import cv2
import h5py
import numpy as np
import pandas as pd
import pytest

from auroral_processing.cli import main
from auroral_processing.flow.jobs import (
    FlowJob, auto_datasets, build_summary, is_pipeline_output, load_flow, load_flow_job, run_flow_job,
)
from auroral_processing.pipeline.processing_chain import run_chain

N, H, W = 8, 96, 128
SHIFT = (1.5, -0.75)
FAST = {"scales": 3, "warps": 3}


@pytest.fixture
def raw(tmp_path):
    """Raw-style source (rawimg/ut1_unix) of a texture translating by SHIFT each frame."""
    base = cv2.GaussianBlur(np.random.default_rng(0).random((H, W)).astype(np.float32), (0, 0), 3)
    frames = [cv2.warpAffine(base, np.float32([[1, 0, SHIFT[0] * n], [0, 1, SHIFT[1] * n]]), (W, H),
                             borderMode=cv2.BORDER_REFLECT) for n in range(N)]
    path = tmp_path / "data" / "2013-04-14-CamTest.h5"
    path.parent.mkdir()
    with h5py.File(path, "w") as f:
        f["rawimg"] = (np.array(frames) / base.max() * 3000 + 500).astype(np.uint16)
        f["ut1_unix"] = 1.364e9 + np.arange(N) * 0.5
    return path


@pytest.fixture
def outputs(raw, tmp_path):
    """Two pipeline outputs of the raw source."""
    root = tmp_path / "filtered"
    a = run_chain(raw, [{"stage": "normalize"}, {"stage": "to_uint16"}], root, label="norm", progress=False)
    b = run_chain(raw, [{"stage": "normalize"}, {"stage": "spatial_median"}, {"stage": "to_uint16"}], root,
                  label="med", progress=False)
    return [a / "frames.h5", b / "frames.h5"]


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def job_file(tmp_path):
    _write(tmp_path / "flows" / "tvl1_fast.yaml", "method: tvl1\nparams: {scales: 3, warps: 3}\n")
    return _write(tmp_path / "flow_jobs" / "compare.yaml",
                  "output: ../flow\n"
                  "inputs: [../filtered/*/*/frames.h5]\n"
                  "flows: [../flows/tvl1_fast.yaml]\n"
                  "grid: {lambda: [0.1, 0.3]}\n")


# ---------- loading ----------
def test_load_flow_file_inline_and_errors(tmp_path):
    f = load_flow(_write(tmp_path / "tv.yaml", "method: tvl1\nparams: {lambda: '1e-1', scales: 3}\ngap: 2\n"))
    assert f.label == "tv" and f.params.lambda_ == 0.1 and f.params.scales == 3 and f.gap == 2
    piv = load_flow({"label": "piv16", "method": "piv", "params": {"window": 16}})
    assert piv.label == "piv16" and piv.params.window == 16
    assert load_flow({"file": str(tmp_path / "tv.yaml"), "label": "renamed"}).label == "renamed"
    with pytest.raises(ValueError, match="unknown keys"):
        load_flow({"method": "tvl1", "lamda": 0.1})
    with pytest.raises(ValueError, match="unknown TV-L1 setting"):
        load_flow({"method": "tvl1", "params": {"lamda": 0.1}})
    with pytest.raises(ValueError, match="unknown method"):
        load_flow({"method": "farneback"})


def test_grid_and_set_apply_only_to_flows_that_have_them(outputs):
    tv, piv = load_flow({"label": "tv", "params": FAST}), load_flow({"label": "piv", "method": "piv"})
    job = FlowJob("j", outputs[:1], [tv, piv], grid={"lambda": [0.1, 0.3], "gap": [1, 2]}, sets={"window": 32})
    runs = job.runs()
    names = sorted(r.name for r in runs)
    assert names == ["piv_gap1", "piv_gap2", "tv_lambda0.1_gap1", "tv_lambda0.1_gap2", "tv_lambda0.3_gap1",
                     "tv_lambda0.3_gap2"]
    assert all(r.params.window == 32 for r in runs if r.flow.label == "piv")
    gap2 = [r for r in runs if r.settings.get("gap") == 2]
    assert all(r.compute_kw["gap"] == 2 and r.compute_kw["stride"] == 2 for r in gap2)  # shared i0 (stride = max gap)
    with pytest.raises(ValueError, match="no flow has that setting"):
        FlowJob("j", outputs, [tv], grid={"lamda": [0.1]}).runs()


def test_auto_datasets_and_pipeline_detection(raw, outputs):
    assert auto_datasets(outputs[0]) == ("frames", "frames_ut1_unix") and is_pipeline_output(outputs[0])
    assert auto_datasets(raw) == ("rawimg", "ut1_unix") and not is_pipeline_output(raw)


# ---------- running ----------
def test_dry_run_writes_nothing(outputs, job_file, tmp_path):
    job = load_flow_job(job_file)
    table = run_flow_job(job, dry_run=True, progress=False)
    assert len(table) == 4 and set(table["status"]) == {"would run"}
    assert not (tmp_path / "flow").exists()
    assert not any((p.parent / "flow").exists() for p in outputs)


def test_run_layout_summary_rerun_and_force(outputs, job_file, tmp_path):
    job = load_flow_job(job_file)
    table = run_flow_job(job, progress=False)
    assert set(table["status"]) == {"done"}

    # everything in the job folder: <job>/<source>/<output>/<run>, nothing next to the inputs
    folder = tmp_path / "flow" / "compare"
    for out in outputs:
        assert not (out.parent / "flow").exists()
        for lam in ("0.1", "0.3"):
            assert (folder / out.parent.parent.name / out.parent.name / f"tvl1_fast_lambda{lam}" / "flow.h5").exists()
    assert (folder / "job.yaml").read_text() == job_file.read_text()
    links = sorted(p.name for p in (folder / "runs").iterdir())
    assert links == sorted(f"{o.parent.name}-tvl1_fast_lambda{lam}" for o in outputs for lam in ("0.1", "0.3"))

    s = pd.read_csv(folder / "summary.csv")
    assert len(s) == 4 and set(s["status"]) == {"complete"}
    for m in ("warp_mae", "zero_mae", "fb_error"):
        assert s[f"{m}_mean"].notna().all()
    assert (s["warp_over_zero"] < 1).all()  # the flow explains the translation
    assert {"output", "chain", "spatial_median.on", "lambda", "dt"} <= set(s.columns)
    assert sorted(s["spatial_median.on"]) == [False, False, True, True]
    assert np.allclose(s["dt"], 0.5)

    # rerun: everything exists
    assert set(run_flow_job(load_flow_job(job_file), progress=False)["status"]) == {"exists"}

    # edited settings in the same run folders: refused, then --force recomputes
    job_file.write_text(job_file.read_text() + "metrics: [warp_mae, zero_mae]\n")
    edited = load_flow_job(job_file)
    assert set(run_flow_job(edited, dry_run=True, progress=False)["status"]) == {"exists, different settings"}
    failed = run_flow_job(edited, progress=False)
    assert failed["status"].str.contains("different settings").all()
    assert set(run_flow_job(edited, force=True, progress=False)["status"]) == {"done"}
    assert "fb_error_mean" not in pd.read_csv(folder / "summary.csv")


def test_raw_source_goes_to_job_folder_and_score(raw, tmp_path):
    flow = load_flow({"label": "tv", "params": FAST})
    job = FlowJob("rawjob", [raw], [flow], output=tmp_path / "flow", metrics=("warp_mae", "zero_mae"), score=True)
    assert set(run_flow_job(job, progress=False)["status"]) == {"done"}
    run = tmp_path / "flow" / "rawjob" / raw.stem / "raw" / "tv" / "flow.h5"
    assert run.exists() and not (raw.parent / "flow").exists()  # nothing written next to the data
    folder = tmp_path / "flow" / "rawjob"
    assert (folder / "runs" / f"{raw.stem}-tv").is_symlink()
    assert (folder / "scores.csv").exists() and not (folder / "runs" / "scores.csv").exists()
    s = pd.read_csv(folder / "summary.csv")
    assert "output" not in s and s.loc[0, "warp_over_zero"] < 1


def test_summary_reports_deleted_runs(outputs, job_file, tmp_path):
    import shutil

    run_flow_job(load_flow_job(job_file), progress=False)
    folder = tmp_path / "flow" / "compare"
    shutil.rmtree(folder / outputs[0].parent.parent.name / outputs[0].parent.name)  # delete one output's runs
    (folder / "runs" / "._metadata").write_bytes(b"")  # what macOS leaves beside links on exFAT drives
    s = pd.read_csv(build_summary(folder))
    assert (s["status"] == "missing").sum() == 2 and (s["status"] == "complete").sum() == 2 and len(s) == 4


def test_inputs_that_would_share_a_run_folder_are_an_error(outputs, tmp_path):
    import shutil

    twin = tmp_path / "other" / outputs[0].parent.parent.name / outputs[0].parent.name
    shutil.copytree(outputs[0].parent, twin)  # same source and output names, different place
    job = FlowJob("j", [outputs[0], twin / "frames.h5"], [load_flow({"label": "tv", "params": FAST})])
    with pytest.raises(ValueError, match="would share the run folder"):
        job.runs()


def test_cli_flow_run(outputs, job_file, tmp_path, capsys):
    main(["flow", "run", str(job_file), "-n", "-q"])
    out = capsys.readouterr().out
    assert "would run" in out and not (tmp_path / "flow").exists()

    main(["flow", "run", str(job_file), "-q", "--set", "lambda=0.2"])  # --set replaced by the grid's lambda
    assert (tmp_path / "flow" / "compare" / "summary.csv").exists()
    assert "--set lambda=0.2" in (tmp_path / "flow" / "compare" / "job.yaml").read_text()

    flows = tmp_path / "flows" / "tvl1_fast.yaml"
    main(["flow", "run", str(outputs[0]), "-f", str(flows), "--set", "gap=2", "-o", str(tmp_path / "quick"), "-q"])
    quick = tmp_path / "quick" / "tvl1_fast"
    assert (quick / outputs[0].parent.parent.name / outputs[0].parent.name / "tvl1_fast" / "flow.h5").exists()
    assert (quick / "summary.csv").exists()

    with pytest.raises(SystemExit, match="no flow has that setting"):
        main(["flow", "run", str(job_file), "--set", "lamda=0.2", "-q"])
    with pytest.raises(SystemExit, match="--flow"):
        main(["flow", "run", str(outputs[0]), "-q"])
