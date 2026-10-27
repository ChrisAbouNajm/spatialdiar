import argparse
import os
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from spatialdiar.spatial import doa, geometry, rttm, seats, vad

FS = 16000


def load_channels(paths):
    import soundfile as sf
    if len(paths) == 1:
        x, fs = sf.read(paths[0], dtype='float32', always_2d=True)
        return x.T, fs
    chans, fs = [], None
    for p in paths:
        c, r = sf.read(p, dtype='float32')
        if fs is not None and r != fs:
            raise SystemExit(f'{p}: {r} Hz, expected {fs} Hz')
        fs = r
        chans.append(c if c.ndim == 1 else c[:, 0])
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans]), fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio', nargs='+',
                    help='one multichannel WAV, or one file per channel in '
                         'array order')
    ap.add_argument('--array', default='ami8', choices=sorted(geometry.ARRAYS),
                    help='microphone geometry (default: ami8)')
    a = ap.parse_args()

    x, fs = load_channels(a.audio)
    arr = geometry.ARRAYS[a.array]
    if x.shape[0] != len(arr['xy']):
        raise SystemExit(f'{a.array} expects {len(arr["xy"])} channels, '
                         f'got {x.shape[0]}')
    uri = os.path.splitext(os.path.basename(a.audio[0]))[0]
    out = os.path.splitext(a.audio[0])[0] + '.rttm'
    print(f'{x.shape[0]} ch x {x.shape[1] / fs:.1f} s at {fs} Hz, '
          f'geometry {a.array}')
    if fs != FS:
        g = gcd(FS, fs)
        x = resample_poly(x, FS // g, fs // g, axis=1).astype(np.float32)
        print(f'resampled to {FS} Hz')

    proc = doa.SpatialProcessor(arr['xy'], fs=FS)
    res = proc.process(x)
    hist = doa.angle_hist(res, doa.bin_weights(res))

    found, info = seats.discover_ridges(hist.mean(0), res['angs'])
    if not len(found):
        raise SystemExit('no seats found -- is there speech, and are the '
                         'channels in array order?')
    print(f'seats: {np.round(found, 1)} '
          f'({info["n_found"]} found, closest pair '
          f'{info["min_sep_found"]:.1f} deg)')
    if info['at_floor']:
        print('  warning: two seats sit at the separation floor, so they may '
              'be one speaker split in two')

    speech = vad.noise_floor_vad(res['energy'].sum(1))
    det = seats.detect(seats.seat_mass(hist, res['angs'], found), speech,
                       q=seats.GUIDE_Q)
    labels = [f'{uri}_seat{i}' for i in range(len(found))]
    n = rttm.write_rttm(out, uri,
                        rttm.det_to_intervals(det, res['times'], labels),
                        gap=0.4)
    ovl = ((det.sum(0) >= 2) & speech).mean()
    print(f'speech {speech.mean() * 100:.0f}% of frames, '
          f'{ovl * 100:.1f}% with two or more seats active')
    print(f'-> {out} ({n} segments)')


if __name__ == '__main__':
    main()
