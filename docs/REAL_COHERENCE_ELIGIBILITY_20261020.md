# Real-song coherence eligibility — 2026-10-20

A default-off combined path now passes the predeclared real-song screen on
Chasing Horizons: 98.419% lower layout RMS, maximum combined protected-band RMS
error 3.11e-9, maximum residual growth 2.44% against a 10% cap, exact short disabled
CPU parity, CPU/OpenCL maximum difference 2.98e-8, and zero silence output.
Eleven native children cover shifted 26-second views at actual 10/2 geometry,
an irregular 20.4-second tail, and short numerical/null controls. Observed child
HWM peaks at 225.64 MiB; Python HWM is 198.34 MiB. This is not a speed experiment.

[Protocol](../artifacts/listening-coherence-20261020/eligibility_protocol.json),
[receipts](../artifacts/listening-coherence-20261020/real_gate_complete.json), and
[identities](../artifacts/listening-coherence-20261020/milestone.json) preserve the
inputs, exact origins, model/binary/source hashes, gates and two stopped harness
runs. One stopped on strict Python-versus-native float32 serialization; the other
assumed new receipt fields existed in an older baseline. Their data remains local;
no acoustic threshold was relaxed. Seventeen focused Rust checks and eleven
Python gate checks pass. No training or solver/weight change occurred.

`auto-master --mid-coherence` opts into absolute-origin excitation and the actual
native whole-track trusted-band projection after WOLA, before polish. It requires
Euler8, 10/2 geometry, Basis mode, a fresh output directory, <=180 seconds,
`--quiet-mid-denoise`, and no `--force-specialists`. Routing abstention stays intact.
The quiet gate is soft; it does not establish zero false edits. Defaults stay
unchanged. Standalone controls retain the historical unprojected path and expose
explicit quiet/origin/final-lock settings. Model and conditioning receipts matter:
the flag alone is not a universal qualification for arbitrary configurations.

This earned one controlled full-song listening experiment, not restoration accuracy
or promotion. Initially, two matched Smooth-control attempts halted in the existing PH stage before a
candidate started: first at a1GiB guard, then at the documented1.5GiB retry.
The second watchdog observed1583.93MiB at182.84s. At that checkpoint no listening candidate or
package existed and final mastering eligibility was unearned. See
[the stop and next experiment](../artifacts/listening-coherence-20261020/fullsong_verdict.json).
The archived Smooth FLAC remains a historical comparison with incomplete original
input/executable provenance; the newly rendered control supplies an exact anchor.
The subsequent [parity-qualified lifetime fix](PH_MEMORY_RESULTS_20261021.md)
allowed both full masters and a listening package under the unchanged memory cap.
[Owner responses are now recorded](OWNER_LISTENING_RESULTS_20261021.md): the
control was selected in both actual comparisons, but both identical-file trials
also received directional choices; comparative superiority remains unestablished.
Procedural complex-error conflict and coherent
healthy-input false additions remain negative evidence. The 1e-6 protection gate
applies to Mid output; ordinary downstream gain/EQ/dynamics intentionally changes
bands, so no final-output relock is applied.
