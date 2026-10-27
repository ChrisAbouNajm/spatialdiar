import numpy as np

C_SOUND = 343.0


def mic_xy(angles_deg, radius):
    a = np.deg2rad(np.asarray(angles_deg, float))
    return radius * np.stack([np.cos(a), np.sin(a)], 1)


AMI8 = dict(
    name='ami8',
    xy=mic_xy(np.arange(0, 360, 45), 0.10),
    files=[f'{i:02d}' for i in range(1, 9)],
)
AMI4 = dict(
    name='ami4',
    xy=mic_xy([0, 90, 180, 270], 0.10),
    files=['01', '03', '05', '07'],
)
RESPEAKER4 = dict(
    name='respeaker4',
    xy=mic_xy([45, 135, 225, 315], 0.0457 * np.sqrt(2) / 2),
    files=None,
)

ARRAYS = {a['name']: a for a in (AMI8, AMI4, RESPEAKER4)}
