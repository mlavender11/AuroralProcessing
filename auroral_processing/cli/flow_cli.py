import argparse
from dataclasses import fields
from pathlib import Path


def _method_fields(method):
    """(flag name, dataclass field) for each setting of a flow method, e.g. ("scale-step", ...)."""
    from auroral_processing.flow.params import METHODS, external_name

    return [(external_name(f.name).replace("_", "-"), f) for f in fields(METHODS[method])]


METHOD_NAMES = ("tvl1", "piv", "flct")


def _cast(f):
    return int if f.type in ("int", int) else float


def add_parser(sub):
    p = sub.add_parser("flow", help="TV-L1 optical flow", description="TV-L1 optical flow on HDF5 frame stacks")
    flow_sub = p.add_subparsers(dest="flow_command", required=True, metavar="COMMAND")

    # ---- shared options ----
    inputs = argparse.ArgumentParser(add_help=False)
    inputs.add_argument("filename", help="source HDF5 file")
    inputs.add_argument("--frames", default="rawimg", help="frame dataset (N, H, W), default rawimg")
    inputs.add_argument("--times", default="ut1_unix", help="time dataset (N,), default ut1_unix")
    inputs.add_argument("--gap", type=int, default=1, help="frames between the two images of a pair")
    inputs.add_argument("--stride", type=int, default=None, help="frames between successive pairs (default = gap)")
    inputs.add_argument("--start", type=int, default=0, help="first frame index")
    inputs.add_argument("--stop", type=int, default=None, help="frame index to stop before")
    inputs.add_argument("--clip", type=float, nargs=2, default=(0.5, 99.5), metavar=("PLO", "PHI"),
                        help="percentiles used for the global input scaling")
    inputs.add_argument("--input-dtype", choices=["float32", "uint8"], default="float32",
                        help="image type given to TV-L1 (uint8 matches the old prototype)")
    inputs.add_argument("--metrics", nargs="*", default=["warp_mae", "zero_mae"],
                        help="per-pair metrics: warp_mae zero_mae fb_error (fb_error doubles run time)")
    inputs.add_argument("--warm-start", action="store_true", help="start each pair from the previous pair's flow")
    inputs.add_argument("--method", choices=METHOD_NAMES, default="tvl1",
                        help="tvl1 (OpenCV TV-L1), piv (OpenPIV multi-pass), flct (Fourier local correlation tracking)")
    inputs.add_argument("-j", "--workers", type=int, default=1, help="parallel processes")
    inputs.add_argument("--overwrite", action="store_true", help="replace an existing output with different settings")

    tvl1 = argparse.ArgumentParser(add_help=False)
    titles = {"tvl1": "TV-L1 parameters (OpenCV defaults shown)", "piv": "PIV parameters (--method piv)",
              "flct": "FLCT parameters (--method flct)"}
    for method in METHOD_NAMES:
        g = tvl1.add_argument_group(titles[method])
        for name, f in _method_fields(method):
            g.add_argument(f"--{method}-{name}", dest=f"{method}_{f.name}", type=_cast(f), default=f.default,
                           metavar=name.upper(), help=f"{f.metadata['help']} (default {f.default})")

    # ---- aurora flow compute ----
    c = flow_sub.add_parser("compute", parents=[inputs, tvl1], help="compute flow into a flow .h5 file")
    c.add_argument("-o", "--out", default=None, help="output flow file (default <source>_flow.h5)")
    c.add_argument("--video", nargs="*", choices=["quiver", "speed"], default=[], help="also render these videos")
    c.set_defaults(func=run_compute)

    # ---- aurora flow video ----
    v = flow_sub.add_parser("video", help="render a flow file as an MP4")
    v.add_argument("flow_file", help="flow .h5 file from 'aurora flow compute'")
    v.add_argument("-o", "--out", default=None, help="output video (default <flow file>_<kind>.mp4)")
    v.add_argument("--kind", choices=["quiver", "speed"], default="quiver")
    v.add_argument("--source", default=None, help="source file, if it has moved since the flow was computed")
    _add_video_options(v)
    v.set_defaults(func=run_video)

    # ---- aurora flow repack ----
    r = flow_sub.add_parser("repack", help="rewrite flow files with gzip so MATLAB can read and write them")
    r.add_argument("flow_files", nargs="+", help="flow .h5 files (rewritten in place)")
    r.set_defaults(func=run_repack)

    # ---- aurora flow sweep ----
    s = flow_sub.add_parser("sweep", parents=[inputs, tvl1], help="run a grid of TV-L1 settings")
    s.add_argument("outdir", help="folder for the runs and summary.csv")
    s.add_argument("--grid", action="append", required=True, metavar="NAME=V1,V2,...",
                   help="setting of --method to sweep, e.g. --grid gamma=1,3 --grid lambda=0.05,0.15, or for piv "
                        "--grid window=8,16; also gap=2,4,8 and stride=4,8 "
                        "(then --stride defaults to the largest gap and --vmax is px/pair at --gap, scaled per gap)")
    s.add_argument("--video", nargs="*", choices=["quiver", "speed"], default=["quiver"],
                   help="videos to render per run (default quiver; give none to skip)")
    _add_video_options(s)
    s.set_defaults(func=run_sweep)

    # ---- aurora flow compare ----
    cmp = flow_sub.add_parser("compare", help="gather runs (e.g. the best of each method) into one folder to score together")
    cmp.add_argument("outdir", help="folder to create; each run is linked in as <sweep>-<run>")
    cmp.add_argument("run_dirs", nargs="+", help="run folders (each holding a flow.h5)")
    cmp.set_defaults(func=run_compare)

    # ---- aurora flow score ----
    sc = flow_sub.add_parser("score", help="score the runs of a sweep (scores.csv + heatmaps)")
    sc.add_argument("sweep_dir", help="folder from 'aurora flow sweep'")
    sc.add_argument("--source", default=None, help="source file, if it has moved since the sweep ran")
    sc.add_argument("--no-plot", action="store_true", help="skip scores_heatmaps.png")
    sc.set_defaults(func=run_score)

    # ---- aurora flow benchmark ----
    b = flow_sub.add_parser("benchmark", help="build a synthetic benchmark with known flow from real frames")
    b.add_argument("filename", help="source HDF5 file")
    b.add_argument("-o", "--out", required=True, help="benchmark file to write")
    b.add_argument("--frames", default="v4", help="frame dataset to build from (default v4)")
    b.add_argument("--base-frames", type=int, nargs="+", default=None,
                   help="source frames to build from (default: the 4 with the most lane structure)")
    b.add_argument("--speeds", type=float, nargs="+", default=[5, 15, 25], help="rigid lane speeds, px/pair")
    b.add_argument("--realistic-flow", default=None, help="flow file of the same frames, used (smoothed) as an extra case")
    b.add_argument("--brightness", type=float, nargs=2, default=[0.05, 0.15], metavar=("MIN", "MAX"),
                   help="fractional brightness change of still patches (0 0 for none)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--overwrite", action="store_true", help="replace an existing benchmark")
    b.set_defaults(func=run_benchmark)

    # ---- aurora flow run ----
    j = flow_sub.add_parser(
        "run",
        help="run a flow job file (inputs x flows x grid) and summarise the metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run a flow job file, or flow settings files on inputs. Each run goes to\n"
        "<pipeline output>/flow/<job>/<run>/flow.h5 (other inputs: <job folder>/sources/<name>/<run>/);\n"
        "the job folder <output>/<job>/ gets runs/ links, runs.csv and summary.csv.",
        epilog=RUN_EPILOG,
    )
    j.add_argument("target", nargs="+", help="flow job file (.yaml), or inputs: .h5 files, folders or globs")
    j.add_argument("-f", "--flow", action="append", default=[], metavar="FLOW.yaml",
                   help="flow settings file (repeatable). Required with inputs; replaces a job file's flows")
    j.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="set a flow setting in every flow that has it, e.g. lambda=0.1 or gap=2")
    j.add_argument("--grid", action="append", default=[], metavar="NAME=V1,V2",
                   help="sweep a setting, e.g. lambda=0.05,0.15 (repeatable; adds to a job file's grid)")
    j.add_argument("--job-name", default=None, help="job name with inputs (default: the first flow's label)")
    j.add_argument("-o", "--out", default=None, help="parent folder of the job folder (default: job file's output, "
                                                     "else optical_flow/flow)")
    j.add_argument("-j", "--workers", type=int, default=None, help="parallel processes per run")
    j.add_argument("--score", action="store_true", default=None, help="also run 'aurora flow score' on the runs")
    j.add_argument("--force", action="store_true", help="recompute runs that already exist")
    j.add_argument("-n", "--dry-run", action="store_true", help="list the runs and their folders without computing")
    j.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    j.set_defaults(func=run_job)


RUN_EPILOG = """\
examples:
  aurora flow run flow_jobs/chain_compare.yaml
  aurora flow run flow_jobs/chain_compare.yaml -n                       # what would run, where
  aurora flow run flow_jobs/chain_compare.yaml --grid lambda=0.05,0.15  # add a sweep
  aurora flow run filtered/clip/*/frames.h5 -f flows/tvl1_default.yaml --set gap=2

flow settings file (flows/tvl1_default.yaml):
  method: tvl1                  # tvl1 | piv | flct
  params: {lambda: 0.15}        # names as in the --tvl1-* / --piv-* / --flct-* flags
  gap: 1                        # optional; also stride, warm_start, clip, input_dtype

flow job file (its file name is the job name):
  output: ../flow
  inputs: [../filtered/clip/*/frames.h5]
  flows: [../flows/tvl1_default.yaml]
  grid: {lambda: [0.05, 0.15]}
  metrics: [warp_mae, zero_mae, fb_error]   # the default; fb_error doubles run time
  workers: 4
  video: [quiver]
  score: false

Rerunning skips finished runs and resumes partial ones. A run whose folder holds a
flow file made with other settings fails with "different settings"; use --force.
"""


def build_flow_job(args):
    """A FlowJob from a job file or inputs, with the command-line options applied on top."""
    from auroral_processing.flow.jobs import FlowJob, load_flow_job, load_flow
    from auroral_processing.pipeline.config import resolve_sources, parse_assignments, parse_grid

    first = Path(args.target[0])
    if len(args.target) == 1 and first.suffix.lower() in (".yaml", ".yml"):
        job = load_flow_job(first)
        if args.flow:
            job.flows = [load_flow(f) for f in args.flow]
        if args.job_name:
            job.name = args.job_name
    else:
        if not args.flow:
            raise SystemExit("give at least one --flow FLOW.yaml with inputs (or pass a flow job file)")
        flows = [load_flow(f) for f in args.flow]
        job = FlowJob(name=args.job_name or flows[0].label, inputs=resolve_sources(args.target), flows=flows)
    job.sets.update(parse_assignments(args.set))
    job.grid.update(parse_grid(args.grid))
    if args.out:
        job.output = Path(args.out).resolve()
    if args.workers:
        job.workers = args.workers
    if args.score:
        job.score = True
    extra = [f"--set {s}" for s in args.set] + [f"--grid {g}" for g in args.grid] + [f"--flow {f}" for f in args.flow]
    job.note = ("command-line: " + " ".join(extra)) if extra else ""
    return job


def run_job(args):
    import pandas as pd
    from auroral_processing.flow.jobs import run_flow_job

    try:
        job = build_flow_job(args)
        table = run_flow_job(job, dry_run=args.dry_run, force=args.force, progress=not args.quiet)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"error: {e}")

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200,
                           "display.max_colwidth", 70):
        print(f"\njob folder: {job.folder}")
        print(table.drop(columns="folder").to_string(index=False))
    if not args.dry_run and (job.folder / "summary.csv").exists():
        print(f"summary: {job.folder / 'summary.csv'}")
    failed = table["status"].str.startswith("failed").sum()
    if failed:
        raise SystemExit(f"{failed} of {len(table)} failed")


def _add_video_options(p):
    g = p.add_argument_group("video options")
    g.add_argument("--grid-arrows", type=int, default=32, help="arrows across the larger image dimension")
    g.add_argument("--vmax", type=float, default=None, help="top of the speed scale, px/pair (default 99th pct)")
    g.add_argument("--arrow-scale", type=float, default=None, help="arrow px per px/pair")
    g.add_argument("--fps", type=float, default=10)
    g.add_argument("--cmap", default="turbo")


def _params(args):
    from auroral_processing.flow.params import METHODS

    return METHODS[args.method](**{f.name: getattr(args, f"{args.method}_{f.name}") for _, f in _method_fields(args.method)})


def _compute_kw(args):
    return dict(
        frames=args.frames,
        times=args.times,
        gap=args.gap,
        stride=args.stride,
        start=args.start,
        stop=args.stop,
        clip_percentiles=tuple(args.clip),
        input_dtype=args.input_dtype,
        metrics=args.metrics,
        warm_start=args.warm_start,
        workers=args.workers,
        overwrite=args.overwrite,
    )


def _video_kw(args):
    return dict(grid=args.grid_arrows, vmax=args.vmax, arrow_scale=args.arrow_scale, fps=args.fps, cmap=args.cmap)


def run_compute(args):
    from auroral_processing.flow.compute_flow import compute_flow
    from auroral_processing.products.flow_video import make_flow_video

    out = compute_flow(args.filename, args.out, params=_params(args), **_compute_kw(args))
    print(out)
    for kind in args.video:
        print(make_flow_video(out, kind=kind))


def run_video(args):
    from auroral_processing.products.flow_video import make_flow_video

    print(make_flow_video(args.flow_file, args.out, kind=args.kind, source=args.source, **_video_kw(args)))


def run_repack(args):
    from auroral_processing.flow.io import repack_flow

    for path in args.flow_files:
        print(repack_flow(path))


def parse_grid(items, method="tvl1"):
    """["gamma=1,3", "lambda=0.05,0.15"] -> {"gamma": [1.0, 3.0], "lambda": [0.05, 0.15]}"""
    types = {name.replace("-", "_"): _cast(f) for name, f in _method_fields(method)} | {"gap": int, "stride": int}
    grid = {}
    for item in items:
        name, _, values = item.partition("=")
        name = name.strip().replace("-", "_")
        if name not in types or not values:
            raise SystemExit(f"bad --grid {item!r}; use NAME=V1,V2 with NAME one of {sorted(types)} (--method {method})")
        grid[name] = [types[name](v) for v in values.split(",")]
    return grid


def run_sweep(args):
    from auroral_processing.flow.sweep import run_sweep as sweep

    summary = sweep(
        args.filename,
        parse_grid(args.grid, args.method),
        args.outdir,
        base_params=_params(args),
        videos=args.video,
        video_kw=_video_kw(args),
        **_compute_kw(args),
    )
    print(summary)


def run_compare(args):
    from pathlib import Path

    out = Path(args.outdir)
    runs = [Path(r) for r in args.run_dirs]
    for run in runs:
        if not (run / "flow.h5").exists():
            raise SystemExit(f"{run} has no flow.h5")

    def link_name(run, levels):
        """<sweep>-<run>, prefixed by ``levels`` more parent folders (skipping a pipeline output's "flow")."""
        parts = [p for p in run.resolve().parts[-(2 + levels) - 1 :] if p != "flow"]
        return "-".join(parts[-(2 + levels) :])

    def default_levels(run):
        sweep = run.resolve().parent
        in_output = sweep.parent.name == "flow" and (sweep.parent.parent / "frames.h5").exists()
        return 1 if in_output else 0  # pipeline output/flow/<sweep>/<run>: name the output too

    levels = {run: default_levels(run) for run in runs}
    for _ in range(4):  # add parent folders to names that collide, until they are unique
        names = {}
        for run in runs:
            names.setdefault(link_name(run, levels[run]), []).append(run)
        clashes = [rs for rs in names.values() if len(rs) > 1]
        if not clashes:
            break
        for rs in clashes:
            for run in rs:
                levels[run] += 1
    else:
        raise SystemExit(f"can't give these runs distinct names: {[str(r) for rs in clashes for r in rs]}")
    names = {name: rs[0] for name, rs in names.items()}
    out.mkdir(parents=True, exist_ok=True)
    for name, run in names.items():
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(run.resolve(), target_is_directory=True)
        print(link)
    print(f"now: aurora flow score {out}")


def run_score(args):
    from auroral_processing.flow.score import score_sweep

    print(score_sweep(args.sweep_dir, source=args.source, plot=not args.no_plot))


def run_benchmark(args):
    from auroral_processing.flow.benchmark import make_benchmark

    print(make_benchmark(
        args.filename,
        args.out,
        frames=args.frames,
        base_frames=args.base_frames,
        speeds=args.speeds,
        realistic_flow=args.realistic_flow,
        brightness=tuple(args.brightness) if max(args.brightness) > 0 else None,
        seed=args.seed,
        overwrite=args.overwrite,
    ))
