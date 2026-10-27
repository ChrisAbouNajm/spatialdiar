import numpy as np

from .angles import angdiff

GUIDE_Q = 0.15


def discover_ridges(hist_s, angs, *, min_sep=20.0, prom_frac=0.03,
                    height_frac=0.08):
    from scipy.signal import find_peaks
    gh = np.asarray(hist_s, float)
    gh = gh.sum(0) if gh.ndim > 1 else gh
    A = len(gh)
    if gh.max() <= 0:
        return np.zeros(0), dict(n_found=0, min_sep_found=np.nan,
                                 at_floor=False)
    step = float(np.median(np.diff(angs)))
    dist = max(int(round(min_sep / step)), 1)
    tri = np.concatenate([gh, gh, gh])
    pk, props = find_peaks(tri, distance=dist,
                           prominence=prom_frac * gh.max(),
                           height=height_frac * gh.max())
    keep = (pk >= A) & (pk < 2 * A)
    pk, prom = pk[keep] - A, props['prominences'][keep]
    if len(pk) == 0:
        pk = np.array([int(gh.argmax())])
        prom = np.array([0.0])
    order = np.argsort(prom)[::-1]
    chosen = []
    for i in order:
        a = float(angs[pk[i]])
        if all(angdiff(a, angs[pk[c]]) >= min_sep for c in chosen):
            chosen.append(i)
    seats = np.sort(np.array([float(angs[pk[i]]) for i in chosen]))
    if len(seats) > 1:
        ms = min(angdiff(seats[i], seats[j]) for i in range(len(seats))
                 for j in range(i + 1, len(seats)))
    else:
        ms = np.nan
    return seats, dict(n_found=len(seats), min_sep_found=float(ms),
                       at_floor=bool(ms == ms and ms < min_sep + 2 * step))


def seat_mass(hist_s, angs, seats, half=15.0):
    return np.stack([hist_s[:, angdiff(angs, s) < half].sum(1)
                     for s in seats], 0)


def detect(mass, speech, q=0.35):
    mass_n = mass / (mass.sum(0) + 1e-9)
    det = (mass_n > q) & speech[None, :]
    dom = mass_n.argmax(0)
    idx = np.where(speech)[0]
    det[dom[idx], idx] = True
    return det
