import h5py
import copy
import json
import numpy as np
from auroral_processing.utils.norm import compute_norm
from auroral_processing.pipeline.generators import STAGES, read_h5
from auroral_processing.pipeline.drivers import write_h5

IMAGE_ATTRS = {
    "CLASS": np.bytes_("IMAGE"),
    "IMAGE_VERSION": np.bytes_("1.2"),
    "IMAGE_SUBCLASS": np.bytes_("IMAGE_GRAYSCALE"),
    "DISPLAY_ORIGIN": np.bytes_("UL"),
    "IMAGE_WHITE_IS_ZERO": np.uint8(0),
}


def build_chain(src, chain_dict):
    """
    Chain processing stages together.

    src is the original source (e.g. ``read_h5``); each stage takes the previous
    stage as its source. Every stage passes ``(t, frame)`` pairs, so time follows
    the frames through binning etc. Pull from the returned generator (with a
    driver from ``drivers.py``) to run the chain.

    ``chain_dict`` is a dictionary containing each step and its settings

    Example:
    ```chain_dict = {
        "normalize": {"vmin": norm.vmin, "vmax": norm.vmax}, #TODO does norm.vmin work here
        "spatial_median": {"k": 3},
        "frame_binning_avg": {"bin_size": 15},
        "apply_wiener": {},
        "to_uint16": {},
    }
    ```
    """

    for step, settings in chain_dict.items():
        src = STAGES[step](src, **settings)

    return src


def resolve(chain_dict, values):
    "Copy chain dict, replacing values such as $name with values[name]"
    "e.x. values[vmin] = norm.vmin, so $vmin becomes norm.vmin"
    chain_dict = copy.deepcopy(chain_dict)
    for settings in chain_dict.values():
        for key, val in settings.items():
            if isinstance(val, str) and val.startswith("$"):
                settings[key] = values[val[1:]]
    return chain_dict


def num_output_frames(n_frames, chain_dict):
    """Number of frames the chain outputs from n_frames input frames."""
    n = n_frames
    for stage, settings in chain_dict.items():
        if stage == "frame_binning_avg":
            n //= settings.get("bin_size", 15)
        elif stage == "sliding_temporal_median":
            n -= settings.get("window", 5) - 1
    return n


def write_attrs(f, src_frame_name, src_ut_name, dest_frame_name, dest_ut_name, chain_dict):
    dest_imgs = f[dest_frame_name]
    dest_ut = f[dest_ut_name]
    src_ut = f[src_ut_name]

    time_attrs = dict(src_ut.attrs)
    time_attrs.update(
        units="seconds since 1970-01-01",
        time_scale="UT1",
        source_dataset=src_ut_name,
    )

    for key, val in time_attrs.items():
        dest_ut.attrs[key] = val

    dest_imgs.attrs["source_dataset"] = src_frame_name
    dest_imgs.attrs["preprocessing_chain"] = str(chain_dict)
    dest_imgs.attrs["time_dataset"] = dest_ut_name
    dest_ut.attrs["frame_dataset"] = dest_frame_name

    for key, val in IMAGE_ATTRS.items():
        dest_imgs.attrs[key] = val


def apply_processing(hdf_fn, chain_dict, iteration, frames_name="rawimg", time_name="ut1_unix"):
    """
    Run the processing chain on `frames_name` and write the result to
    `iteration`, with its own time dataset f"{iteration}_ut1_unix".
    """
    with h5py.File(hdf_fn, "r+") as f:
        frames = f[frames_name]
        ut = f[time_name]

        norm = compute_norm(frames, ut, 1)
        chain = resolve(chain_dict, {"vmin": norm.vmin, "vmax": norm.vmax})

        pipeline = build_chain(read_h5(frames, ut), chain)

        n = write_h5(
            pipeline,
            f,
            iteration,
            total_frames=num_output_frames(len(frames), chain),
        )

        write_attrs(f, frames_name, time_name, iteration, f"{iteration}_ut1_unix", chain)
        print(f"wrote {n} frames to /{iteration}, times to /{f[iteration].attrs['time_dataset']}")
        return n
