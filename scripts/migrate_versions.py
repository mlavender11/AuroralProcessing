"""
One-off: move the old pre-processing versions (v2, v3, ... datasets written into a source file by
``apply_processing``) into output folders, as if ``run_chain`` had made them.

Each ``vN`` becomes ``<out>/<source stem>/vN-<hash8>/frames.h5``. The hash is that of the chain
with its fitted normalize limits removed, on the *original* source, so running ``chains/vN.yaml``
on that source later finds the migrated output and skips it. A ``vN.mp4`` next to the file is
copied in as ``preview.mp4``. The file holding the versions is only read, never changed (existing
flow sweeps still point at it).

    python scripts/migrate_versions.py optical_flow/data/2013-03-27-CamSer7196_015400-015600-copy.h5
"""

import argparse
import ast
import json
import re
import shutil
from pathlib import Path
import h5py

from auroral_processing.pipeline.config import DEFAULT_OUTPUT
from auroral_processing.pipeline.drivers import read_h5
from auroral_processing.pipeline.jobs import build_index
from auroral_processing.pipeline.processing_chain import (
    PREVIEW_FILE, canonical_chain, is_complete, output_dir, write_output,
)


def parse_chain(text):
    """The stored chain: JSON, or the Python repr older versions wrote with str()."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def unfitted(chain):
    """The chain as it was specified: normalize limits were fitted, not given."""
    out = []
    for s in chain:
        s = dict(s)
        if s["stage"] == "normalize":
            s["vmin"] = s["vmax"] = None
        out.append(s)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("versions_file", help="HDF5 file holding the vN datasets")
    p.add_argument("--source", default=None,
                   help="original source file the versions were made from (default: versions_file without '-copy')")
    p.add_argument("-o", "--out", default=str(DEFAULT_OUTPUT), help="output folder (default optical_flow/processed)")
    p.add_argument("--versions", nargs="*", default=None, help="datasets to migrate (default: every vN with a chain)")
    p.add_argument("--frames", default="rawimg", help="source frame dataset the versions were made from")
    p.add_argument("--times", default="ut1_unix", help="source time dataset")
    args = p.parse_args(argv)

    vfile = Path(args.versions_file)
    source = Path(args.source) if args.source else vfile.with_name(vfile.name.replace("-copy", ""))
    if not source.exists():
        raise SystemExit(f"original source {source} not found; pass --source")
    out_root = Path(args.out)

    with h5py.File(vfile, "r") as f:
        names = args.versions or sorted(
            (k for k in f if re.fullmatch(r"v\d+", k) and "preprocessing_chain" in f[k].attrs),
            key=lambda k: int(k[1:]),
        )
        for name in names:
            ds = f[name]
            fitted = canonical_chain(parse_chain(ds.attrs["preprocessing_chain"]))
            chain = canonical_chain(unfitted(fitted))
            out = output_dir(source, chain, out_root, label=name, frames=args.frames, times=args.times)
            if is_complete(out, chain):
                print(f"{name}: already migrated -> {out}")
            else:
                times = ds.attrs.get("time_dataset", f"{name}_ut1_unix")
                write_output(
                    out,
                    read_h5(ds, f[times]),
                    total_frames=len(ds),
                    chain=chain,
                    chain_fitted=fitted,
                    source=source,
                    frames=args.frames,
                    times=args.times,
                    label=name,
                    extra_attrs={"migrated_from": f"{vfile.name}:{name}"},
                )
                print(f"{name}: {len(ds)} frames -> {out}")
            video = vfile.with_name(f"{name}.mp4")
            if video.exists() and not (out / PREVIEW_FILE).exists():
                shutil.copy2(video, out / PREVIEW_FILE)
                print(f"{name}: copied {video.name} as {PREVIEW_FILE}")

    print(build_index(out_root))


if __name__ == "__main__":
    main()
