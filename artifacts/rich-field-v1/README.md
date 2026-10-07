# Rich Field V1: Coupled Complex Field Operators

This artifact directory freezes the 1,000-step coupled complex field operator models,
their 24-scene development benchmarks, and the provenance receipts for the full-song
restoration of the user's reference song (`Verse 1 v 77.wav`).

## Models

- `basis.json`: The winning 1,000-step **Basis** coupled field model (Kind 3).
  Combines phase-aligned temporal transport $\tau \cdot z(t-1)$ and constant-energy
  rotation $\omega \mathbf{J} z$, trained with exact flow-velocity matching on the
  gated prior. This model produced the successful full-length restoration:
  `/sdcard/Download/Verse 1 v 77 - Basis Restored.wav`.
- `transport.json`: Phase-aligned temporal transport sibling operator.
- `rotation.json`: Constant-energy orthogonal rotation sibling operator.
- `diagonal.json`: Direct diagonal operator (baseline flow matching).

## Evaluation Receipts

- `development24.json`: 24-scene development evaluation (`runs/rich-sprint/field-dev1000`).
  - `Basis`: **0.71319** missing-band NMSE, **10.41 dB** LSD.
  - `Transport`: **0.5893** NMSE on cymbal bursts (strongest cymbal burst texture).
  - `Rotation`: Constrained energy via $\omega \mathbf{J} z$, reducing overshoot by $3\times$–$5\times$.
- `full-song-receipt.json`: Provenance receipt for the complete 285.04s full-song restoration run
  (36 chunks of 10s, 2s cosine equal-power crossfade, peak 0.6783, zero clipping).
- `trackgleam-comparison.json`: Formal comparison between the raw original Suno generation (`Verse 1 v 77.wav`),
  the external commercial master (`TrackGleam Master`), and our `Basis Restored` track.

## Listening Verification

- **Owner Verdict**: The owner confirmed the full-length Basis restoration feels punchier,
  highs are definitely better, and definition is very good. The characteristic bite/crispness
  of Basis was preferred for maintaining high-band strength over heavier attenuation.
- **TrackGleam Comparison**: TrackGleam achieved $-14.8\text{ dBFS}$ RMS via standard commercial
  mastering gain (+2.8 dB) with an exciter boost concentrated at 8–12 kHz, but rolled off
  sharply above 16 kHz ($0.094\%$). Basis Restored preserved original dynamic headroom ($-17.6\text{ dBFS}$)
  while expanding crest factor by $+0.9\text{ dB}$ (punchier transients) and doubling top-octave air
  energy ($0.223\%$).
