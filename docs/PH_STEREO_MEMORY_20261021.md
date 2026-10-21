# Sequential stereo-audit memory qualification

Feelin’ Catchy's 179.902375-second control stopped in PH at sampled process
HWM1549.117MiB under a1536MiB cap. No candidate was started. The original latch
and failure remain intact; the cap and all audio-quality checks are unchanged.

The earlier four-drop change qualified the shorter140.4s pair but retained four
full masked stereo spectra simultaneously in the audit. The new change prepares
one mask at a time in the same Llow/Rlow/Lhi/Rhi synthesis order, retaining both
repeated original analyses and every reduction/metric. Owned low-band waveforms,
last-used stereo spectra and the controller's original M/S vectors are released
after their final reads. No high=full-low shortcut, FFT removal, model change,
quality-audit bypass or arithmetic change is introduced.

[Design](../artifacts/ph-stereo-memory-20261021/design.json),
[exact patch](../artifacts/ph-stereo-memory-20261021/native_change.patch) and
[independent structural review](../artifacts/ph-stereo-memory-20261021/static_review.json)
precede qualification. A33735×513 Complex32 spectrum is132.035MiB on this full
track. Static liveness estimates are not measured peak savings.

Twenty-seven focused GTF Rust tests pass. Pinned-before/after CPU qualification
matches all48 procedural comparison WAV hashes across13 modes and every
non-timing comparison field. Separate synthetic and10-second real pre-PH
auto-master controls match the complete PH audit, master.json, float master and
PCM listening WAV exactly. No audio or source input was overwritten. The maximum
sampled child HWM among these six bounded calls is94.293MiB, below512MiB.
These controls establish bounded exact parity, not proof across arbitrary inputs.

The [qualifier](../artifacts/ph-stereo-memory-20261021/qualify.py) reuses the
established13-mode method with fresh paths. The real control is fixed60..70s of
the failed run's cached pre-PH spatial signal, selected before candidate outputs.
[Fixture receipt](../artifacts/ph-stereo-memory-20261021/real_fixture.json) and
[parity receipt](../artifacts/ph-stereo-memory-20261021/parity_complete.json)
retain commands and hashes. The new binary is
`runs/ph-stereo-memory-20261021/sequential-candidate-v1`, SHA256
`c2c5b14f77d0275f1b2f3fd94f028efd2253479a6477f94729d7366ebb37b511`.
The original fixed binary7e4c47fe… remains unchanged. Rust1.99.1, locked release
OpenCL build and Cargo-j8 are recorded in [build.json](../artifacts/ph-stereo-memory-20261021/build.json).

For a clone without the old procedural WAV, the
[fixture-only generator](../artifacts/ph-stereo-memory-20261021/make_procedural_control.py)
reuses its original formula and canonical WAV writer, with create-only output.
It regenerates the current device's reference SHA exactly
([check](../artifacts/ph-stereo-memory-20261021/procedural_fixture_reproduction.json)):

```sh
LD_PRELOAD=/data/data/com.termux/files/usr/lib/libc++_shared.so \
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python artifacts/ph-stereo-memory-20261021/make_procedural_control.py \
  --output runs/stage-accountability-20261020/master-observer-engaged-v1/known-peak-control.wav
```

It does not invoke old renders or need their binaries. Historical binary hashes
are local receipts; another environment must rebuild and repin both reference
and candidate explicitly before reproducing numerical comparisons.

The fresh same-source full-track control and one coherence candidate now complete
under the unchanged cap. Maximum observed process HWM across the pair is
1057.277MiB; all native quality gates pass. All five full-track pre-PH stage WAV
hashes match the old stopped control exactly. The additional parity receipt and
its binary binding are checked before dispatch; prefix equality is checked before
starting the candidate. Twenty-two runner tests cover these gates and containment.

The old full control never completed, so no full old final waveform exists for
comparison. Exact short audio/audit parity plus full-prefix equality must not be
called whole-track before/after waveform parity. The old1549.117MiB peak is
censored; no exact percentage peak-memory reduction or speedup is claimed.
[Full pair and listening receipts](../artifacts/listening-feelin-retry-20261021/milestone.json)
record provenance and the delivered blind package. Owner preference is pending;
production sound remains unpromoted.
