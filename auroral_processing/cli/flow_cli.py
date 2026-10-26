import argparse
from dataclasses import fields


def _tvl1_fields():
    from auroral_processing.flow.params import TVL1Params, external_name

    return [(external_name(f.name).replace("_", "-"), f) for f in fields(TVL1Params)]


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
    inputs.add_argument("-j", "--workers", type=int, default=1, help="parallel processes")
    inputs.add_argument("--overwrite", action="store_true", help="replace an existing output with different settings")

    tvl1 = argparse.ArgumentParser(add_help=False)
    g = tvl1.add_argument_group("TV-L1 parameters (OpenCV defaults shown)")
    for name, f in _tvl1_fields():
        g.add_argument(f"--tvl1-{name}", dest=f.name, type=_cast(f), default=f.default, metavar=name.upper(),
                       help=f"{f.metadata['help']} (default {f.default})")

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
    _add_video_options(v)
    v.set_defaults(func=run_video)

    # ---- aurora flow sweep ----
    s = flow_sub.add_parser("sweep", parents=[inputs, tvl1], help="run a grid of TV-L1 settings")
    s.add_argument("outdir", help="folder for the runs and summary.csv")
    s.add_argument("--grid", action="append", required=True, metavar="NAME=V1,V2,...",
                   help="setting to sweep, e.g. --grid gamma=1,3 --grid lambda=0.05,0.15; also gap=2,4,8 "
                        "(then --stride defaults to the largest gap and --vmax is px/pair at --gap, scaled per gap)")
    s.add_argument("--video", nargs="*", choices=["quiver", "speed"], default=["quiver"],
                   help="videos to render per run (default quiver; give none to skip)")
    _add_video_options(s)
    s.set_defaults(func=run_sweep)

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


def _add_video_options(p):
    g = p.add_argument_group("video options")
    g.add_argument("--grid-arrows", type=int, default=32, help="arrows across the larger image dimension")
    g.add_argument("--vmax", type=float, default=None, help="top of the speed scale, px/pair (default 99th pct)")
    g.add_argument("--arrow-scale", type=float, default=None, help="arrow px per px/pair")
    g.add_argument("--fps", type=float, default=10)
    g.add_argument("--cmap", default="turbo")


def _params(args):
    from auroral_processing.flow.params import TVL1Params

    return TVL1Params(**{f.name: getattr(args, f.name) for _, f in _tvl1_fields()})


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

    print(make_flow_video(args.flow_file, args.out, kind=args.kind, **_video_kw(args)))


def parse_grid(items):
    """["gamma=1,3", "lambda=0.05,0.15"] -> {"gamma": [1.0, 3.0], "lambda": [0.05, 0.15]}"""
    types = {name.replace("-", "_"): _cast(f) for name, f in _tvl1_fields()} | {"gap": int}
    grid = {}
    for item in items:
        name, _, values = item.partition("=")
        name = name.strip().replace("-", "_")
        if name not in types or not values:
            raise SystemExit(f"bad --grid {item!r}; use NAME=V1,V2 with NAME one of {sorted(types)}")
        grid[name] = [types[name](v) for v in values.split(",")]
    return grid


def run_sweep(args):
    from auroral_processing.flow.sweep import run_sweep as sweep

    summary = sweep(
        args.filename,
        parse_grid(args.grid),
        args.outdir,
        base_params=_params(args),
        videos=args.video,
        video_kw=_video_kw(args),
        **_compute_kw(args),
    )
    print(summary)


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
