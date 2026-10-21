# PH spectrum lifetime qualification

Four explicit `drop` calls release full-track spectra after their final reads:
the two controller M/S spectra after synthesis, and the two masked low-band
spectra after their synthesis. Arithmetic, FFT call order, metrics, controller
settings and all quality audits remain unchanged. Independent review confirmed
that synthesis returns owned waveform buffers and no later code uses the spectra.

Twenty-seven focused GTF Rust tests pass. A pinned-before/after CPU comparison on
a procedural peak/multitone control yields exact SHA equality for48 WAV files
across13 Morphic modes. All non-timing comparison fields also match. A separate
synthetic auto-master check matches the complete PH audit, finalizer metadata,
float master and listen WAV exactly. These are bounded parity controls, not proof
across arbitrary audio or of subjective preference.

The fresh140.4-second Chasing Horizons Smooth-settings control now completes
with all native peak, mono, known-band and transient checks, at observed process
HWM1274.39MiB under the unchanged1536MiB watchdog. Previously the unchanged
control was stopped at observed1583.93MiB before completion. That earlier peak
is censored: no exact percentage reduction in full-run peak memory is claimed.
A static four-spectrum budget is about412MiB, not a measured peak delta.
No speed experiment or quality-audit bypass was performed.

[Identities and checks](../artifacts/ph-memory-20261021/verification.json),
[parity receipt](../artifacts/ph-memory-20261021/parity_complete.json), and
[full control receipt](../artifacts/ph-memory-20261021/full_control_receipt.json)
preserve commands, sources, output hashes and scope. The compiled memory-fixed
binary is `runs/ph-memory-20261021/lifetime-candidate-v1`; its SHA256 is
`7e4c47fe19e589553a8c4e999f988b76683de2967c7b5547f2b58fb9484cf161`.

One coherence candidate now uses that same binary and all matched settings,
adding only `--mid-coherence`. It still requires its own full-song quality gates.
The production champion remains unchanged; the owner has not judged this candidate.
