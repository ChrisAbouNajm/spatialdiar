import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import stft

from .geometry import C_SOUND


class SpatialProcessor:
    def __init__(self, mic_xy, fs=16000, nfft=512, hop=256,
                 flo=300.0, fhi=4000.0, ang_step=2.0,
                 coh_tc=7, block=4096, chunk=256):
        self.mic_xy = np.asarray(mic_xy, float)
        self.fs, self.nfft, self.hop = fs, nfft, hop
        self.coh_tc, self.block, self.chunk = coh_tc, block, chunk
        self.nch = len(mic_xy)

        freqs = np.fft.rfftfreq(nfft, 1 / fs)
        self.band = (freqs >= flo) & (freqs <= fhi)
        self.fb = freqs[self.band].astype(np.float32)

        self.angs = np.arange(0, 360, ang_step, dtype=np.float32)
        ang_rad = np.deg2rad(self.angs)
        dvec = np.stack([np.cos(ang_rad), np.sin(ang_rad)], 1)
        tau = (self.mic_xy @ dvec.T) / C_SOUND
        self.pairs = [(i, j) for i in range(self.nch)
                      for j in range(i + 1, self.nch)]
        self.Pi = np.array([p[0] for p in self.pairs])
        self.Pj = np.array([p[1] for p in self.pairs])
        dt = np.stack([tau[i] - tau[j] for i, j in self.pairs])
        steer = np.exp(-2j * np.pi * self.fb[None, :, None]
                       * dt[:, None, :]).astype(np.complex64)
        self.steer = np.ascontiguousarray(
            np.concatenate([steer.real, -steer.imag], 0).transpose(1, 0, 2))

    def stft_frames(self, x):
        freqs, times, S = stft(x, fs=self.fs, nperseg=self.nfft,
                               noverlap=self.nfft - self.hop,
                               boundary=None, padded=False)
        Sb = np.ascontiguousarray(
            S[:, self.band, :].transpose(2, 0, 1)).astype(np.complex64)
        return Sb, times.astype(np.float64)

    def _process_block(self, Sb):
        Tb = Sb.shape[0]
        CS = Sb[:, self.Pi, :] * np.conj(Sb[:, self.Pj, :])
        CPH = (CS / (np.abs(CS) + 1e-12)).astype(np.complex64)

        num = uniform_filter1d(CS.real, self.coh_tc, axis=0) ** 2 + \
              uniform_filter1d(CS.imag, self.coh_tc, axis=0) ** 2
        pw = np.abs(Sb) ** 2
        pw_s = uniform_filter1d(pw, self.coh_tc, axis=0)
        den = pw_s[:, self.Pi, :] * pw_s[:, self.Pj, :] + 1e-20
        coh = np.sqrt(num / den).mean(1).astype(np.float32)
        del CS, num, den, pw_s

        FB, A = len(self.fb), len(self.angs)
        doa = np.zeros((Tb, FB), np.int16)
        conf = np.zeros((Tb, FB), np.float32)
        resp = np.zeros((Tb, A), np.float32)
        for t0 in range(0, Tb, self.chunk):
            t1 = min(t0 + self.chunk, Tb)
            c = CPH[t0:t1]
            r = np.concatenate([c.real, c.imag], 1).transpose(2, 0, 1)
            r = r @ self.steer
            doa[t0:t1] = r.argmax(2).T
            conf[t0:t1] = r.max(2).T / len(self.pairs)
            resp[t0:t1] = r.sum(0)
        return doa, conf, coh, pw.mean(1).astype(np.float32), resp

    def process(self, x):
        Sb, times = self.stft_frames(x)
        T = Sb.shape[0]
        m = self.coh_tc
        outs = {k: [] for k in ('doa_bin', 'conf', 'coh', 'energy', 'resp_bb')}
        for b0 in range(0, T, self.block):
            b1 = min(b0 + self.block, T)
            lo, hi = max(b0 - m, 0), min(b1 + m, T)
            doa, conf, coh, en, resp = self._process_block(Sb[lo:hi])
            s = slice(b0 - lo, b0 - lo + (b1 - b0))
            for k, v in zip(outs, (doa, conf, coh, en, resp)):
                outs[k].append(v[s])
        res = {k: np.concatenate(v, 0) for k, v in outs.items()}
        res['times'] = times
        res['angs'] = self.angs
        return res


def bin_weights(res, coh_floor=0.35, energy_exp=0.3):
    return ((res['energy'] ** energy_exp)
            * np.clip(res['coh'] - coh_floor, 0, None)
            * np.clip(res['conf'], 0, None))


def angle_hist(res, w, t_smooth=9, a_sigma_bins=3.0):
    doa_bin, angs = res['doa_bin'], res['angs']
    T, FB = doa_bin.shape
    A = len(angs)
    hist = np.zeros((T, A), np.float32)
    tt = np.repeat(np.arange(T), FB)
    np.add.at(hist, (tt, doa_bin.ravel()), w.ravel())
    hist = uniform_filter1d(hist, t_smooth, axis=0)
    r = int(3 * a_sigma_bins)
    ker = np.exp(-0.5 * (np.arange(-r, r + 1) / a_sigma_bins) ** 2)
    ker = (ker / ker.sum()).astype(np.float32)
    pad = np.concatenate([hist[:, -r:], hist, hist[:, :r]], 1)
    out = np.apply_along_axis(lambda row: np.convolve(row, ker, 'same'), 1, pad)
    return out[:, r:-r]
