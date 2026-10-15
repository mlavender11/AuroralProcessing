import numpy as np
import cv2


def get_frame_to_rgb(cmap, norm):  # TODO Move to video?
    """
    Build a function that converts a raw frame to an RGB uint8 image.
    Used for videos and keograms
    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Color map applied to normalized pixel values.
    norm : matplotlib.colors.Normalize
        Normalization applied to raw pixel values before colormapping.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Function that maps a raw frame array to an (H, W, 3) uint8 RGB array.
    """

    def frame_to_rgb(raw_frame):
        rgba = cmap(norm(raw_frame))
        rgb = rgba[:, :, :3]
        return (rgb * 255).astype(np.uint8)

    return frame_to_rgb


# ------------ Pre-Processing Utils -----------------
def apply_norm_to_frame(frame, vmin, vmax, curve="linear"):
    """
    apply normalization to frame based on vmin and vmax, type based on curve

    Parameters
    ----------
    frame :
        video/image frame
    vmin : _type_
        min value for normalization scaling
    vmax : _type_
        max value for normalization scaling
    curve : str, optional
        type of normalization (linear, log, sqrt), by default "linear"

    Returns
    -------
    np.ndarray
        normalized frame, same dimensions as input

    Raises
    ------
    ValueError
        if curve is not of a valid type
    """
    frame = np.clip(frame, vmin, vmax).astype(np.float32)
    if curve == "linear":
        return (frame - vmin) / (vmax - vmin)
    elif curve == "log":
        return (np.log(frame) - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
    elif curve == "sqrt":
        return (np.sqrt(frame) - np.sqrt(vmin)) / (np.sqrt(vmax) - np.sqrt(vmin))
    raise ValueError(curve)


def apply_spatial_median(frame, k=3):
    """
    use cv2.medianBlur on frame - applies spatial blurring

    Parameters
    ----------
    frame : np.ndarray
        input frame
    k : int, optional
        kernel size, must be odd, by default 3

    Returns
    -------
    np.ndarray
        blurred frame, same size as input
    """
    return cv2.medianBlur(frame, k)


def apply_temporal_median(frame_stack):
    # TODO pass UT through, keep track of time
    """
    Apply temporal median to a stack of frames

    Parameters
    ----------
    frame_stack : list[np.ndarray]
        stack of frames to merge together

    Returns
    -------
    np.ndarray
        frame the same size as input
    """
    return np.median(frame_stack, axis=0)
