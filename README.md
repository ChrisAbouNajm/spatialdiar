# spatialdiar

Training-free spatial diarization for clinical speech. Multichannel audio
in; **who spoke when**, including during overlap, out.

No model, no training data, no enrolment corpus, no per-room calibration, and
no speaker count: the number of seats comes from the recording itself.
numpy and scipy only, which is why it runs in real time on a Raspberry Pi 5 in
311 MB.

Reference implementation for *Training-Free Spatial Diarization for Overlap
Detection and Source Separation in Clinical Speech*.

## The idea

Most diarizers decide, per frame, who is speaking. That forces a choice when
two people speak at once, and in a consultation they often do.

This one never makes that choice. Every time-frequency bin votes for a
direction, and the votes accumulate into an angle histogram over the whole
recording. Speakers who stay in their seats show up as ridges. Because
activity is then read off per seat, two speakers can be marked at the same
instant — which a single-argmax direction estimator cannot do at all, and
which is what makes the output usable as a guide for a separator.


## Use it

```bash
pip install -r requirements.txt
python diarize.py meeting.wav --array ami8   # who spoke when -> meeting.rttm
```

The RTTM is written next to the input, named after it (after the first file
when each channel is its own file). `diarize.py` resamples any other rate to
16 kHz, the rate every frame setting assumes. The library does not, so pass it
16 kHz audio.

Or as a library:

```python
from spatialdiar.spatial import doa, geometry, rttm, seats, vad

arr = geometry.ARRAYS['ami8']                      # or 'ami4', 'respeaker4'
proc = doa.SpatialProcessor(arr['xy'], fs=16000)

res = proc.process(x)                              # x: (channels, samples)
hist = doa.angle_hist(res, doa.bin_weights(res))   # (frames, angle bins)

found, info = seats.discover_ridges(hist.mean(0), res['angs'])   # no speaker count
speech = vad.noise_floor_vad(res['energy'].sum(1))
det = seats.detect(seats.seat_mass(hist, res['angs'], found), speech,
                   q=seats.GUIDE_Q)

rttm.write_rttm('out.rttm', 'meeting', 
                rttm.det_to_intervals(det, res['times'], 
                                      [f'seat{i}' for i in range(len(found))]),
                gap=0.4)
```

You never pass a speaker count. `discover_ridges` keeps every peak of the
pooled angle profile that survives a prominence and height test, so the number
of seats is decided by the audio — forcing a fixed count is what splits one
acoustic ridge into two spurious seats, and downstream makes two speakers share
a seat. If no peak passes, the single strongest direction is returned. `info`
carries the diagnostics: `n_found`, the closest pair, and whether any pair sits
at the separation floor. There is no count or cap argument at all.

`q=seats.GUIDE_Q` (0.15) is the paper's activity threshold for separation
guides: a seat counts as active when it holds more than 15% of the frame's
spatial mass. `detect`'s own default, 0.35, is the oracle-seat diagnostic.

These defaults make a separation guide, not a stand-alone diarizer. They
favour catching every stretch of speech, overlap included, over precision,
because a separator such as GSS forgives false activity but loses whatever is
missed. That is why the guide's DER looks poor next to a neural diarizer's
while the transcripts it leads to are better.

If nothing separates the voices afterwards — this is the only diarization step,
or each word of one transcript has to go to one speaker — overlap marking works
against you. At q=0.15 a seat often still counts as active while only its
neighbour speaks, so most words touch two speakers. Use q=0.5 instead (in
`diarize.py`, change `q=seats.GUIDE_Q` to `q=0.5`; in the library, pass
`q=0.5` to `seats.detect`): at most one seat can hold more
than half of a frame's spatial mass and the strongest is always marked, so
exactly one seat is active in each speech frame. Giving words to seats this way
was right more often than at q=0.15 on 27 four-mic AMI windows (91.1% of words
against 88.9%) and in simulated quiet two-person rooms at the 32 mm geometry
(83.0% against 68.7%).

The defaults are the paper's frozen operating point — prominence 0.03, height
0.08, minimum separation 20° — recovered unchanged by a held-out
development/test split by meeting. `default.yaml` lists them in one place. It
is documentation, not a loader: every published number came from the keyword
defaults in the code.

The paper's features were computed with an earlier form of the steering
product in `doa.py` (`np.einsum`). The current matrix-product form is 4–6×
faster and rounds differently, so a near-tie between two directions can resolve
the other way and move a segment edge by a frame. On the 27 AMI windows checked
against the published features, the seats were identical and DER stayed within
0.15 points in every window.

## What is here

```
spatialdiar/
  spatial/
    geometry.py   array layouts and the plane-wave convention
    angles.py     angular distance between azimuths
    doa.py        STFT, per-bin DOA, coherence gating, angle histogram
    seats.py      blind seat discovery, seat mass, per-seat activity
    vad.py        noise-floor voice activity detection
    rttm.py       activity -> timed segments -> RTTM

default.yaml      the frozen operating point, for reading (not loaded)
diarize.py        CLI: audio -> RTTM
```

## Scope

This is the part of the paper that is ours: the spatial method, and nothing
else. Guided source separation ([desh2608/gss](https://github.com/desh2608/gss))
and Whisper are other people's work and are not vendored here; the RTTM this
produces is what you feed the first. The MUSIC, single-argmax SRP-PHAT and TDOA
baselines the paper compares against are not here either — they are baselines,
not the contribution.

The experiments, corpus plumbing, baselines and evaluation that produced the
paper's numbers are not part of this repository.

## Operating condition

The method assumes speakers stay in approximately fixed seats, which holds for
a seated consultation and is the setting the paper evaluates.

Two separate limits are worth knowing. `discover_ridges` will not place two
seats closer than `min_sep` (20°), so speakers nearer than that merge into one
acoustic ridge — and no label-free test can distinguish a merged pair from a
genuinely smaller group, which is why the paper reports the deployment gate as
a negative result. `info['at_floor']` flags when a pair sits at that floor, and
is the signal to treat the output with suspicion.

Empirically, on the 32 mm device geometry the paper measures clean behaviour at
45° separation and above (seat error within 3.5°), and clear degradation at 25°
with RT60 ≥ 0.4 s (DER 35.9–80.7%). Treat sufficient angular separation as an
operating condition, not something the method recovers from.
