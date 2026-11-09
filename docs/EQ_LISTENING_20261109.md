# Isolated EQ listening session — 2026-11-09

## Status and interpretation

The corrected EQ has earned a small exploratory owner listening test. The package is delivered and verified; **human judgments are pending**. No listening results, subjective improvement, general transparency or production promotion are claimed. The failed short Mid pilots remain unpromoted, and the high-field synthesis stage is excluded so this comparison answers one question.

The fixed local plan in `artifacts/eq-listening-20261109/plan.json` was written before candidate renders. It was not precommitted. Three already available local recordings supply exactly their 48–60 second regions, without candidate-driven excerpt selection. The three arms are legacy EQ, corrected EQ and EQ bypass, with identical shared gain-only finalization. Neural stages, denoising, microstructure, spatial processing, morphic processing and glue are off. Zero limiter budget and a −18 LUFS render target are used. EQ bypass is not untouched source: it retains that common finalization. This is excerpt-conditioned global EQ, not a whole-song mastering comparison. The recordings do not represent all musical genres, pristine commercial masters or intentional lo-fi styles.

The pinned executable is the prior revision binary from source commit `56a3275`, SHA-256 `c245db19c7ac91b7aa724326c7211fb49a5c7d768e232d385d6d05c7f069eee2`. Current Rust changes add tests and correct terminology; they do not change its DSP. The eight broad EQ bands are hand-selected analysis bands, not an implemented auditory critical-band filterbank or masking model.

## Readiness checks

Nine focused CPU release EQ tests pass, including opposing band corrections and support continuity/bounds, silence, impulse/geometry, quiet-to-loud channel swaps, and equal-channel consistency. The listening tool passes 25 tests. A first test invocation failed due to module discovery; the documented discovery command resolved it. A later regression expected the historical provenance error text; keeping that message for procedural mode fixed the test while giving recording mode its own message. These failures were tooling issues, not audio-quality results.

All nine real renders are finite 48 kHz stereo with exactly 576,000 frames. Native and independent FFmpeg meters satisfy the predeclared peak ceiling and −18 LUFS tolerance. The local 800–2500 Hz cross-correlation timing proxy peaks at zero integer lag for every arm relative to the EQ-bypass render. This is neither sub-sample verification nor proof of transient, phase or perceptual preservation. Numeric qualification is recorded in `artifacts/eq-listening-20261109/qualification.json`; output is not selected by a quality score. Source hashes and full render provenance remain locally retained alongside the public command receipts; no song waveforms are tracked in Git.

The new `recording_diagnostic` listening mode accepts 8–48 second real-recording excerpts, requires explicit nonprocedural provenance and common source/render target, and restricts fatigue to `not_assessed`. Historical/procedural modes retain their distinct rules. See [listening protocol](BLIND_LISTENING_PROTOCOL.md).

## Delivery and response

Listener folder: `/sdcard/Download/Roach_EQ_Blind_20261109/`.

Twelve randomized A/B trials contain 24 native-rate 24-bit WAVs. There is a hidden identity pair for each source alongside all three distinct variant comparisons. Pair matching applies constant attenuation only, uses peak headroom, and verifies encoded loudness within 0.15 LU. No additional limiting, compression, EQ, resampling or candidate selection is performed during packaging. The private seed is drawn once; assignments and gains remain in the organizer directory. All delivered files were compared by hash with the qualified listener folder; the key was not copied. Verification reports `ready=true trials=12 audio_files=24`.

Read `INSTRUCTIONS.txt`, check a comfortable playback level before judging, disable playback normalization/enhancement/crossfade, then keep settings fixed. Start with `T0001/A.wav` and `B.wav`. For each trial, record transient, tonal, spatial and overall preference using `A`, `B`, `indistinguishable` or `abstain`; confidence is low/medium/high and fatigue stays `not_assessed`. Put prose in a separate `notes.txt`, keyed by trial. Take breaks. Directional choices on identity trials and low-confidence/indistinguishable responses must be preserved, not discarded.

Return the completed `responses.csv` and optional notes. The organizer will validate the unchanged audio and completed categories, freeze the response file and only then reveal assignments. Do not inspect file hashes, metrics or the private key before judging. This is a small local preference experiment, not population validation or a fatigue study. No automatic responses or inferred human judgments will be generated.

The private manifest/key and source audio remain under ignored `runs/eq-listening-20261109/private/`; the listener-only copy is in Download. Do not publish or move the organizer files into that folder. Listening judgments will determine whether corrected EQ deserves broader full-track validation, redesign, or rejection. Larger training and default changes are not justified by this readiness check.
