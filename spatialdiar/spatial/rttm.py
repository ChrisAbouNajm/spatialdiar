import numpy as np
from scipy.ndimage import median_filter


def det_to_intervals(det, frame_times, labels, min_dur=0.2, med_frames=15):
    hop = float(np.median(np.diff(frame_times)))
    per = {}
    for i, lab in enumerate(labels):
        d = median_filter(det[i].astype(float), med_frames) > 0.5
        idx = np.where(np.diff(np.r_[0, d.astype(int), 0]))[0].reshape(-1, 2)
        for s, e in idx:
            if (e - s) * hop >= min_dur:
                per.setdefault(lab, []).append(
                    [float(frame_times[s]),
                     float(frame_times[min(e, len(frame_times) - 1)])])
    return per


def merge_intervals(iv, gap):
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def write_rttm(path, uri, per_spk, gap, min_dur=0.1):
    n = 0
    with open(path, 'w') as f:
        for lab in sorted(per_spk):
            if not per_spk[lab]:
                continue
            for s, e in merge_intervals(per_spk[lab], gap):
                if e - s > min_dur:
                    f.write(f'SPEAKER {uri} 1 {s:.3f} {e-s:.3f} '
                            f'<NA> <NA> {lab} <NA> <NA>\n')
                    n += 1
    return n
