import h5py
from functools import partial
from auroral_processing.utils.norm import compute_norm
from auroral_processing.pipeline.generators import (
    spatial_median,
    frame_binning_avg,
    normalize,
    to_uint16,
    read_h5,
    apply_wiener,
)
from auroral_processing.pipeline.drivers import write_h5

# TODO read chain from setup file


def chain(src, *stages):
    # Chain together processing stage generators
    # src should be hdf reader or other
    # Intermediate stages apply affects
    # Final stage is a driver from drivers.py
    # src is the original source,
    # the source of each stage is the prior stage
    for stage in stages:
        src = stage(src)
    # return the final stage, pull from it to operate the chain
    return src


def apply_processing(hdf_fn, iteration):
    with h5py.File(hdf_fn, "r+") as f:
        frames = f["rawimg"]
        ut = f["ut1_unix"]
        norm = compute_norm(frames, ut, 1)
        vmin, vmax = norm.vmin, norm.vmax
        total_frames = len(frames)

        # bin_size = 15
        # pipeline = chain(
        #     read_h5(frames),
        #     partial(normalize, vmin=vmin, vmax=vmax),
        #     partial(spatial_median, k=3),
        #     partial(frame_binning_avg, bin_size=bin_size),
        #     to_uint16,
        # )

        bin_size = 30
        pipeline = chain(
            read_h5(frames),
            partial(normalize, vmin=vmin, vmax=vmax),
            partial(spatial_median, k=3),
            partial(frame_binning_avg, bin_size=bin_size),
            partial(apply_wiener),
            to_uint16,
        )

        n = write_h5(pipeline, f, iteration, total_frames=total_frames)
        print(f"wrote {n} frames")
        # ut = f["ut1_unix"][:]
        # ut_binned = ut[: n * bin_size].reshape(n, bin_size).mean(axis=1)

        # name = "pipeline-full-ut1_unix"
        # if name in f:
        #     del f[name]
        # f.create_dataset(name, data=ut_binned)

        # f["pipeline-full"].attrs["bin_size"] = bin_size
