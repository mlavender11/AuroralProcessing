import numpy as np


def get_frame_to_rgb(cmap, norm):
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
