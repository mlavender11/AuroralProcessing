from auroral_processing.pipeline.frame_ops import apply_norm_to_frame
import numpy as np
from scipy.signal import wiener

# Chained together to apply processing steps
# TODO make a class for these?


def read_h5(frame_dataset, block=64):
    # TODO read time ?
    """
    read in HDF5 file frame by frame
    reads frams from HDF5 in blocks

    Parameters
    ----------
    frame_dataset :
        dataset within HDF5 file containing frames
    block : int, optional
        number of frames to read from drive at once, by default 64

    Yields
    ------
    individual frames
    """
    for start in range(0, len(frame_dataset), block):
        # pull of size chunk, for memory performance
        chunk = frame_dataset[start : start + block]
        for frame in chunk:
            yield frame


def normalize(src, vmin, vmax, curve="linear"):
    """
    apply normalize to sequence of frames, pulled from an iterable

    Parameters
    ----------
    src : iterable
        iterable containing frames, intended to be np.ndarray
    vmin : float
        vmin value used for normalization
    vmax : float
        vmax value used for normalization
    curve : str, optional
        curve type, see apply_norm_to_frame, by default "linear"

    Yields
    ------
    individual frame
    """
    for frame in src:
        yield apply_norm_to_frame(frame, vmin, vmax, curve=curve)


def spatial_median(src, k=3):
    # TODO use cv2.spatial median directly?
    """
    apply spatial median to a sequence of frames, pulled from an iterable

    Parameters
    ----------
    src : iterable
        iterable containing frames, intended to be np.ndarray
    k : int, optional
        kernel size for median, must be odd, by default 3

    Yields
    ------
    individual frame
    """
    # TODO k > 5 for float32? Maybe use scipy?
    for frame in src:
        yield apply_spatial_median_to_frame(frame, k)


def sliding_temporal_median(src, window=5):  # TODO need to pad edges?
    # Must window be odd?
    if window % 2 == 0:
        raise ValueError(f"window must be odd: current window size = {window}")

    k = window // 2
    ring, n = None, 0
    for frame in src:
        if ring is None:  # Why not do this above?
            ring = np.empty((window,) + frame.shape, dtype=np.float32)

        ring[n % window] = frame  # why this and not overwrite n?
        n += 1
        if n >= window:
            yield np.partition(ring, k, axis=0)[k]


def frame_binning_avg(src, bin_size=15, frame_shape=(512, 512)):
    counter = 0
    frame_sum = None
    for frame in src:
        if frame_sum is None:
            frame_sum = np.zeros(frame_shape, dtype=np.float64)

        # Accumulate frames, continue until bin is full
        frame_sum += frame
        counter += 1

        # if bin is full, average together and reset
        if counter == bin_size:
            frame_avg = frame_sum / bin_size
            yield frame_avg
            frame_sum, counter = None, 0


def to_uint16(src, lo=0.0, hi=1.0):
    scale = 255.0 / (hi - lo)
    for frame in src:
        clipped = np.clip((frame - lo) * scale, 0, 255)
        rounded = np.rint(clipped)
        int_frame = rounded.astype(np.uint16)
        yield int_frame


def apply_wiener(src, size=None, noise=None):
    for frame in src:
        yield wiener(frame, size, noise)
