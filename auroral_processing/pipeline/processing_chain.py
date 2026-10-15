import h5py
import copy
import json
import numpy as np
from auroral_processing.utils.norm import compute_norm
from auroral_processing.pipeline.drivers import write_h5, read_h5
from auroral_processing.pipeline.stages import Stage

IMAGE_ATTRS = {
    "CLASS": np.bytes_("IMAGE"),
    "IMAGE_VERSION": np.bytes_("1.2"),
    "IMAGE_SUBCLASS": np.bytes_("IMAGE_GRAYSCALE"),
    "DISPLAY_ORIGIN": np.bytes_("UL"),
    "IMAGE_WHITE_IS_ZERO": np.uint8(0),
}

def build_stages(chain):
    """Accepts Stage objects, dicts, or a mix"""
    return [copy.deepcopy(s) if isinstance(s, Stage) else Stage.from_dict(s) for s in chain]


def connect_stages(stages, src):
    """Needs a driver at the end - e.g. write_h5"""
    for stage in stages:
        src = stage.run(src)
    return src

def num_output_frames(stages, n):
    for stage in stages:
        n = stage.n_out(n)
    return n

def fit_stages(stages, frames, times):
    """Calculate needed values for each stage, e.g. norm for normalize"""
    for i, stage in enumerate(stages):
        if type(stage).fit is Stage.fit: # TODO what if we didn't have this line?
            continue
        else: # TODO have it fit on the output of the previous stage
            stage.fit(frames, times)
    # TODO need return stages?


def apply_processing(hdf_fn, chain, iteration, frames_name='rawimg', times_name='ut1_unix'):
    stages = build_stages(chain)

    with h5py.File(hdf_fn, 'r+') as f:
        frames = f[frames_name]
        ut = f[times_name]
        fit_stages(stages, frames, ut)

        n = write_h5(
            connect_stages(stages, read_h5(frames, ut)),
            f,
            iteration,
            total_frames=num_output_frames(stages, len(frames))
        )

        write_attrs(f, frames_name, times_name, iteration, f'{iteration}_ut1_unix',
                    [s.to_dict() for s in stages])
    

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
