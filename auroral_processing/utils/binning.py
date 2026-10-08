"""
Arithmetic for video and keogram binning: output frame rate, frames per bin,
and keogram column counts.
"""

import numpy as np


def calculate_fps(ut, bin_size, playback_speed):
    """
    Calculates the fps of output video based on bin size and playback speed.

    Parameters
    ----------
    ut : array_like
        List of unix epoch time, corresponding to each frame
    bin_size : int
        Number of source frames in each bin, to be averaged together into one output frame.
    playback_speed : float
        Playback speed multiplier.

    Returns
    -------
    float
        Output frames per second.
    """
    source_seconds_per_frame = np.median(np.diff(ut))
    source_seconds_per_output_bin = source_seconds_per_frame * bin_size
    fps = playback_speed / source_seconds_per_output_bin

    return fps


def calculate_bin_size(ut, fps, playback_speed):
    """
    Calculates the frame bin size required for specified output fps and playback speed
    bins are collections of source frames to be averaged into one output frame.

    Parameters
    ----------
    ut : array_like
        List of unix epoch time, corresponding to each frame.
    fps : float
        Output frames per second.
    playback_speed : float
        Playback speed multiplier.

    Returns
    -------
    int
        Frames per bin. Minimum of one.
    """

    source_seconds_per_frame = np.median(np.diff(ut))
    bin_size = playback_speed / (fps * source_seconds_per_frame)
    return max(1, round(bin_size))


def compute_keogram_bins(n_frames, ut, bin_width_seconds=None):
    """
    Given the number of frames to be processed, a list of unix epoch times of
    each frame, and the width, in seconds, of each bin, return the number of
    bins and the frames in each bin. If bin_width_seconds is None, assumes 1
    frame per bin.

    Parameters
    ----------
    n_frames : int
        Number of frames to be processed.
    ut : array_like
        List of unix epoch times.
    bin_width_seconds : float, optional
        Number of seconds to be averaged together in each bin, by default None.

    Returns
    -------
    tuple[int, int]
        n_bins, frames_per_bin
    """

    import math

    if bin_width_seconds is None:
        return n_frames, 1
    num_seconds = ut[-1] - ut[0]
    n_bins = max(1, int(num_seconds / bin_width_seconds))
    frames_per_bin = max(1, n_frames // n_bins)
    n_bins = math.ceil(n_frames / frames_per_bin)  # actual columns
    return n_bins, frames_per_bin
