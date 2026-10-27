# spatialdiar

Training-free spatial diarization for multichannel audio. Estimates who spoke when, including overlapping speech, using the spatial information from a microphone array.

The method estimates directions of arrival independently across frequency bins, then combines them to identify speaker locations and track their activity. It does not require training, speaker enrolment, or a predefined speaker count.

Implemented in NumPy and SciPy. The spatial front end runs in real time on a Raspberry Pi 5, with 311 MB peak memory.

Reference implementation for *Training-Free Spatial Diarization for Overlap Detection and Source Separation for Clinical Speech*.

## How it works

Instead of selecting one direction per frame, the method lets each time-frequency bin contribute to a spatial histogram. Speakers who remain in approximately fixed positions form persistent peaks, which are used to identify seats and estimate activity independently for each speaker.

This allows multiple speakers to be active at the same time. The resulting activity tracks can be used directly for diarization or as guides for source separation.

## Installation and usage

```bash
pip install -r requirements.txt
python diarize.py meeting.wav --array ami8
```

This produces `meeting.rttm` alongside the input audio.

Supported array configurations include `ami8`, `ami4`, and `respeaker4`. The CLI resamples audio to 16 kHz when necessary.

The method can also be used directly in Python:

```python
from spatialdiar.spatial import doa, geometry, rttm, seats, vad

arr = geometry.ARRAYS['ami8']
proc = doa.SpatialProcessor(arr['xy'], fs=16000)

res = proc.process(x)  # x: (channels, samples)
hist = doa.angle_hist(res, doa.bin_weights(res))

found, info = seats.discover_ridges(hist.mean(0), res['angs'])
speech = vad.noise_floor_vad(res['energy'].sum(1))
mass = seats.seat_mass(hist, res['angs'], found)

det = seats.detect(mass, speech, q=seats.GUIDE_Q)

rttm.write_rttm(
    'out.rttm', 'meeting',
    rttm.det_to_intervals(
        det, res['times'],
        [f'seat{i}' for i in range(len(found))]
    ),
    gap=0.4
)
```

Library inputs should already be sampled at 16 kHz.

## Activity threshold

The default threshold, `q=0.15`, is the operating point used in the paper for guided source separation (GSS). It prioritizes speech coverage, particularly during overlap, at the expense of additional false activity.

For applications requiring a single speaker label per frame, `q=0.5` is an alternative (for the CLI, change `q=seats.GUIDE_Q` in `diarize.py`):

```python
det = seats.detect(mass, speech, q=0.5)
```

This produces one active seat per detected speech frame. It is useful for single-speaker attribution but does not preserve overlap.

The other main parameters are fixed at the paper's operating point: peak prominence `0.03`, peak height `0.08`, and minimum seat separation `20°`. They are documented in `default.yaml`, although the implementation reads its defaults from the Python code.

## Repository structure

```text
spatialdiar/
  spatial/
    geometry.py   Microphone array geometries
    angles.py     Angular distance calculations
    doa.py        STFT and spatial direction estimation
    seats.py      Seat discovery and activity detection
    vad.py        Voice activity detection
    rttm.py       RTTM output

default.yaml      Parameter reference
diarize.py        Command-line interface
```

## Limitations

The method assumes speakers remain approximately stationary and are sufficiently separated in angle. Closely spaced speakers may be detected as a single seat, particularly in reverberant rooms.

The paper evaluates fixed-seat recordings from the AMI corpus. The implementation is intended for similar settings, including seated clinical conversations.

This repository contains the spatial diarization method only. GSS, Whisper, the experimental baselines, and the evaluation pipeline are not included.

For source separation, the generated RTTM can be used with [desh2608/gss](https://github.com/desh2608/gss).

## Reproducibility

The repository provides the spatial method, not a complete reproduction of the paper's transcription experiments.
