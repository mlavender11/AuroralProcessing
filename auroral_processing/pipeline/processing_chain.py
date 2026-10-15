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
    """
    Chain processing stages together.

    src is the original source (e.g. read_h5); each stage takes the previous
    stage as its source. Every stage passes (t, frame) pairs, so time follows
    the frames through binning etc. Pull from the returned generator (with a
    driver from drivers.py) to run the chain.
    """
    for stage in stages:
        src = stage(src)
    # return the final stage, pull from it to operate the chain
    return src


def apply_processing(hdf_fn, iteration, frames_name="rawimg", time_name="ut1_unix"):
    """
    Run the processing chain on `frames_name` and write the result to
    `iteration`, with its own time dataset f"{iteration}_ut1_unix".
    """
    with h5py.File(hdf_fn, "r+") as f:
        frames = f[frames_name]
        ut = f[time_name]
        norm = compute_norm(frames, ut, 1)
        vmin, vmax = norm.vmin, norm.vmax
        total_frames = len(frames)

        # pipeline = chain(
        #     read_h5(frames, ut),
        #     partial(normalize, vmin=vmin, vmax=vmax),
        #     partial(spatial_median, k=3),
        #     partial(frame_binning_avg, bin_size=bin_size),
        #     apply_wiener,
        #     to_uint16,
        # )
        bin_size = 15
        pipeline = chain(
            read_h5(frames, ut),
            # partial(normalize, vmin=vmin, vmax=vmax),
            partial(spatial_median, k=3),
            partial(frame_binning_avg, bin_size=bin_size),
            apply_wiener,
            to_uint16,
        )

        stages = (  # TODO UPDATE TO DO AUTO FROM ABOVE CHAIN
            f"normalize(vmin={vmin:.6g}, vmax={vmax:.6g}) -> spatial_median(k=3) "
            f"-> frame_binning_avg({bin_size}) -> wiener -> to_uint16"
        )

        time_attrs = dict(ut.attrs)
        time_attrs.update(
            units="seconds since 1970-01-01",
            time_scale="UT1",
            source_dataset=time_name,
        )

        n = write_h5(pipeline, f, iteration, total_frames=len(frames) // bin_size, time_attrs=time_attrs)

        img = f[iteration]
        img.attrs["source_dataset"] = frames_name
        img.attrs["bin_size"] = bin_size
        img.attrs["stages"] = stages
        print(f"wrote {n} frames to /{iteration}, times to /{img.attrs['time_dataset']}")
        return n
