import numpy as np
from scipy.ndimage import binary_dilation, median_filter


def noise_floor_vad(frame_energy, k=3.0, floor_pct=10, hang=8):
    e = median_filter(np.asarray(frame_energy, float), 5)
    floor = np.percentile(e, floor_pct) + 1e-12
    sp = e > k * floor
    return binary_dilation(sp, np.ones(2 * hang + 1, bool))
