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
adding only `--mid-coherence`. It passes its own full-song quality gates; the paired settings match exactly
except the declared intervention.
The production champion remains unchanged. The owner has now judged the package;
see [the response and limitations](OWNER_LISTENING_RESULTS_20261021.md).

## Listening package delivered

Exactly one full-song candidate and one matched current control completed.
Both pass native peak, known-band, mono and transient checks. Maximum observed
process HWM across the pair is1274.39MiB. The candidate differs from the control
and remains an experimental version; the archived dirty-build Smooth master is
not claimed byte-reproduced. The pair uses one fixed binary and differs only by
`--mid-coherence`. No production promotion has occurred.

The owner package is at `/sdcard/Download/Roach Listening 20261021` (same content
as `runs/listening-coherence-20261020/listen/session1`). It has four A/B trials:
full-song and fixed30s excerpts, plus identity controls. Transparent attenuation
matches playback loudness. Private source assignments remain outside the listener
folder; the seed was drawn once and saved only privately. The owner has already
seen mechanism measurements, so this is source-assignment blinding, not naive
expectation blinding. The independent boundary audit verifies all11 delivered
files, eight matching-format WAVs, paired lengths and no source/metric/key leaks.

The owner returned `responses (1).csv`: B on all four trials. The two actual
comparisons select the matched control; the two identical-file trials also
receive B choices. Prose aspect/fatigue fields and percentage confidence are
preserved verbatim in a separate receipt, without weakening categorical reveal
validation. Comparative superiority, prolonged-fatigue reduction and population
benefit remain unestablished. The revealed session cannot serve as fresh blind data.

[Full master receipts](../artifacts/ph-memory-20261021/fullsong_complete.json),
[delivery hashes](../artifacts/ph-memory-20261021/delivery.json), and
[independent boundary review](../artifacts/ph-memory-20261021/listener_boundary_review.json)
provide organizer evidence. Do not consult technical diagnostics or private keys
before completing listening judgments. Optional cover-art extraction found no
image stream in the source WAV; audio FLAC encoding and all qualification gates
completed normally.
