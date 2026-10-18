# Frozen host-batch comparison — 2026-10-18

Status: prepared before timing. Separate foreground authorization is required.
Production signal reference remains493cdca; candidate base55fddfe plus the
archived host-batch patch. Frozen rich-mid-v3 Basis model SHA256:
`e2ca85b64521ca62d72278b33177b426d4386bc0add0ff2d94526f76a5d09ae3`.

## Question and invariants

Does increasing the host feature/predict-call limit from8192 to32768/65536
reduce whole-chunk processing time enough to justify18/40MiB extra feature
scratch? This tests batching, not neural acceleration. Device capacity remains
8192; model weights,184-feature semantics, row order, Euler8/NFE8, strength1,
1500–8000Hz band, projection and10s/2s WOLA remain unchanged. Cache remains
64MiB/serial. No384MiB full-cache arm is reconsidered.

The new raw-FLAC inputs are converted to native48k stereo float32 without EQ,
limiting, normalization or resampling. Previous Stage2/intermediate WAVs were
absent after cleanup. All new arms use identical newly pinned inputs; do not
compare absolute times with the previous different-input experiment.

## Qualification

Development input: the first10 seconds of the retained original Feelin' Catchy
FLAC. Three repeats per host limit, fixed balanced Latin-square order:
8192/32768/65536;65536/8192/32768;32768/65536/8192. Fresh processes/outputs,
one new release/OpenCL binary and same model/input SHA256 for all arms.

Require all nine float WAV SHA256 values and terminal full-state FNV fingerprints
to agree; FNV is noncryptographic corroboration, while trained-fixture tests
directly compare full states. NFE must remain8. Host predict-call count must be
`8*ceil(active_rows/host_limit)`; internal GPU segmentation stays8192. Any
nonfinite/bound failure, unexpected cache use, input mutation or mismatch fails.

A candidate must reduce median `chunk_materialization_ms` by at least10% against
the matched8192 control and stay below384MiB observed RSS/process HWM.
If both qualify, choose32768 unless65536 is at least5% quicker than32768.
This is a development choice; no target/perceptual measurements select a sample.

## Conditional validation, only after qualification

The authorization request will explicitly cover these follow-ups:

- Paired control/candidate ten-second held-out excerpts at48s from Chasing
  Horizons and Singularity. Require exact file/state identity, same NFE and
  correct host/device segmentation;384MiB process bound.
- Matched CPU control/candidate on the development excerpt, requiring exact
  waveform/state equality. Compare CPU/OpenCL waveform deviation against the
  existing2e-5 tolerance, reporting the actual value;384MiB process bound.
- Paired full131-second Singularity Mid-CFM passes,8192 versus chosen host
  limit, same10s/2s WOLA. Require complete waveform SHA256 and every corresponding
  chunk state fingerprint to agree;512MiB process bound and at least10% reduction
  in the full-stage restoration elapsed time. This is not full auto-master timing.

The runner enforces180s per process and checks sampled memory plus terminal
profile HWM. Every path is new; no disk chunk-cache hits are valid timing data.
Capture raw temperature/frequency snapshots and affinity, but do not infer causal
thermal behavior from them. Stop rather than relax a failed gate after results.
Single-track paired validation is limited evidence, not universal performance.

## Verification before timing

Release CPU tests exercise an18492-row trained-model fixture across six cached/
fallback configurations and three host limits over two Euler steps, comparing
complete states/waveforms and host counts. Configuration/default/reset/foreground
checks and eight mocked runner tests pass. Actual large-host OpenCL parity is
still unmeasured at plan freeze. Ordinary defaults remain8192 and Euler8.
