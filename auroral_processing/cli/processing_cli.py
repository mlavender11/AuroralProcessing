import argparse
from pathlib import Path


def add_parser(sub):
    p = sub.add_parser(
        "pipeline",
        help="run pre-processing chains (YAML chain and job files)",
        description="Run pre-processing chains on HDF5 frame stacks. Each result goes to its own output "
        "folder <out>/<source stem>/<label>-<hash8>/frames.h5; existing results are skipped.",
    )
    pipe_sub = p.add_subparsers(dest="pipeline_command", required=True, metavar="COMMAND")

    # ---- aurora pipeline run ----
    r = pipe_sub.add_parser(
        "run",
        help="run a job file, or chain files on source files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run a job file, or chain files on sources.",
        epilog="examples:\n"
        "  aurora pipeline run jobs/bin_sweep.yaml\n"
        "  aurora pipeline run data/night.h5 --chain chains/v4.yaml\n"
        "  aurora pipeline run data/night.h5 --chain chains/v4.yaml --set binned_temporal_average.bin_size=15\n"
        "  aurora pipeline run data/ --chain chains/v4.yaml --grid spatial_median.k=3,5 --grid wiener=on,off --dry-run",
    )
    r.add_argument("target", help="job file (.yaml/.yml), or a source: .h5 file, folder of .h5 files, or glob")
    r.add_argument("-c", "--chain", action="append", default=[], metavar="CHAIN.yaml",
                   help="chain file (repeatable). Required with a source; replaces a job file's chains")
    r.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="override a setting in every chain, e.g. binned_temporal_average.bin_size=15 or wiener=off")
    r.add_argument("--grid", action="append", default=[], metavar="NAME=V1,V2",
                   help="sweep a setting, e.g. spatial_median.k=3,5 (repeatable; adds to a job file's grid)")
    r.add_argument("--label", default=None, help="name for the outputs (default: chain file name, or built from the stages)")
    r.add_argument("-o", "--out", default=None, help="output folder (default: job file's output, else optical_flow/processed)")
    r.add_argument("--frames", default=None, help="frame dataset in the sources (default rawimg)")
    r.add_argument("--times", default=None, help="time dataset in the sources (default ut1_unix)")
    r.add_argument("--preview", action="store_true", default=None, help="also render preview.mp4 for each output")
    r.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    r.add_argument("-n", "--dry-run", action="store_true", help="list the outputs that would be made, without running")
    r.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    r.set_defaults(func=run)

    # ---- aurora pipeline list ----
    ls = pipe_sub.add_parser("list", help="rebuild and show index.csv of an output folder")
    ls.add_argument("out", nargs="?", default=None, help="output folder (default optical_flow/processed)")
    ls.add_argument("--all", action="store_true", help="show every column")
    ls.set_defaults(func=run_list)


def build_job(args):
    """A Job from a job file or a source, with the command-line options applied on top."""
    from auroral_processing.pipeline.config import (
        Job, load_job, load_chain, resolve_sources, parse_assignments, parse_grid,
    )

    target = Path(args.target)
    if target.suffix.lower() in (".yaml", ".yml"):
        job = load_job(target)
        if args.chain:
            job.chains = [load_chain(c) for c in args.chain]
    else:
        if not args.chain:
            raise SystemExit("give at least one --chain CHAIN.yaml with a source (or pass a job file)")
        job = Job(sources=resolve_sources([args.target]), chains=[load_chain(c) for c in args.chain])

    if args.label:
        if len(job.chains) > 1:
            raise SystemExit("--label needs a single chain")
        job.chains[0].label = args.label
    job.sets.update(parse_assignments(args.set))
    job.grid.update(parse_grid(args.grid))
    if args.out:
        job.output = Path(args.out)
    if args.frames:
        job.frames = args.frames
    if args.times:
        job.times = args.times
    if args.preview:
        job.preview = True
    return job


def run(args):
    import pandas as pd
    from auroral_processing.pipeline.jobs import run_job

    try:
        job = build_job(args)
        table = run_job(job, dry_run=args.dry_run, force=args.force, progress=not args.quiet)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"error: {e}")

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200,
                           "display.max_colwidth", 80):
        print(f"\noutput folder: {job.output}")
        print(table.to_string(index=False))
    failed = table["status"].str.startswith("failed").sum()
    if failed:
        raise SystemExit(f"{failed} of {len(table)} failed")


def run_list(args):
    import pandas as pd
    from auroral_processing.pipeline.config import DEFAULT_OUTPUT
    from auroral_processing.pipeline.jobs import build_index

    out = Path(args.out) if args.out else DEFAULT_OUTPUT
    if not out.is_dir():
        raise SystemExit(f"no output folder {out}")
    index = build_index(out)
    df = pd.read_csv(index) if index.stat().st_size > 1 else pd.DataFrame()
    print(index)
    if df.empty:
        print("(no outputs)")
        return
    cols = df.columns if args.all else [c for c in ("folder", "n_frames", "dt", "dtype", "preview", "created", "chain")
                                         if c in df]
    with pd.option_context("display.max_rows", None, "display.width", 250, "display.max_colwidth", 120):
        print(df[list(cols)].to_string(index=False))
