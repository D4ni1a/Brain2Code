import scipy.signal as signal
from scipy.ndimage import zoom


def downsample_with_zoom(arr, target_size, poly = True):
    """
    Downsample using scipy's function

    Parameters:
    -----------
    arr : numpy array, shape (d1, d2, d3)
    target_d4 : int, target size for third dimension

    Returns:
    --------
    downsampled array, shape (d1, d2, target_d4)
    """
    d3 = arr.shape[-1]

    if poly:
        return signal.resample_poly(arr, target_size, d3, axis=1)


    # Calculate zoom factors
    # First dimensions remain the same, last dimension is scaled
    zoom_factors = [1.0]*(len(arr.shape)-1) + [target_size / d3]

    # Apply zoom
    downsampled = zoom(arr, zoom_factors, order=1)
    return downsampled
