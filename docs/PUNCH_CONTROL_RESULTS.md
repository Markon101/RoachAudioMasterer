# Punch controls and propagation audit checkpoint

Parent: `40a76b1`; implementation: `9dcfae3`. Exact executable/source identities are
recorded in `artifacts/punch-controls-v1/provenance.json`. The
[frozen plan](PUNCH_CONTROL_PLAN.md) preceded candidate evaluation.

## Implemented behavior

`auto-master` and `master` share three optional controls:

- `--no-glue`: bypass glue, including under auto-tune.
- `--glue-attack-ms`: 0.1–200 ms; default remains 20 ms. Knee stays 8 dB and
  release stays 160 ms. A slower attack changes compressor response; it is not
  a calibrated guarantee of preferred punch.
- `--max-limiter-gr-db`: optional 0–12 dB maximum limiter-envelope reduction.
  Omitted remains unlimited. Zero uses common gain only after optional glue.
  The budget can lower achieved LUFS; the measured true-peak ceiling retains
  priority. Lookahead/release and raw peak detection remain intact.

For nonzero budget B, pre-gain is capped relative to the same compressed sample
peak used by the limiter. Required gains are at least `10^(-B/20)`, with a small
rounding margin. Sliding minima, averages and one-sided release retain that
lower bound. A final common attenuation, reflected in reported pre-gain, handles
remaining measured true-peak overshoot without adding envelope compression.
This is a project loudness/peak policy, not an amplitude-expansion head or an
inverse of prior compression. The peak claim concerns this existing 4x FIR
meter; arbitrary downstream encoders/playback devices are not evaluated.

Explicit CLI strength, glue ratio/threshold and morphic mode/coupling/quartic/
shelf now override auto-tune and intent, including explicitly supplied legacy
defaults. Manual glue ratio is the actual applied ratio; `--no-glue` wins.
Updated advisory stage actions follow intent/manual changes rather than stale
base values. JSON distinguishes requested/applied ratio and records attack,
budget, loudness shortfall and parameter sources. There is no new neutral preset.

## Confirmed propagation repairs (Tier 2)

1. **SFHT training clamp derivative:** forward velocity is
   `16*tanh(raw/16)`. Training omitted `1-(velocity/16)^2` before dense backprop.
   The old saturated regression produced analytic 0.3 versus finite difference
   0.02448263 for one coordinate. All four output-coordinate derivative checks
   now pass. Per-output and global optimizer clipping remain after the chain
   rule; clipped update directions are not claimed to be exact loss gradients.
2. **Mid waveform adjoint:** forward synthesis protects both `[0, cutoff]` and
   `[8000, 24000]` Hz through a full-length Fourier projection. The old adjoint
   removed only the low band. A protected 8050 Hz test leaked RMS 0.7071067 into
   the old adjoint. Applying the same complementary, self-adjoint projection
   before STFT synthesis VJP passes the upper-band null and waveform directional
   finite-difference checks. Existing encoder/head gradient checks still pass.
3. **SFHT checkpoint/resume validation:** loaders now reject incompatible input/
   output dimensions and corrupt/mismatched Adam moments or counters before
   propagation. A malformed moment-vector regression is rejected cleanly.

The legacy `highband` launcher also now returns failure if the canonical child
dies by signal, rather than falling through to success. A local synthetic child
terminated by SIGTERM yields launcher exit code 1.

Model sizes, frozen weights, inference equations and checkpoint schemas are
unchanged. Future training summaries record the gradient rule and source
provenance. No new training or song adaptation ran, and improved learning or
restoration quality is unestablished. Old training trajectories cannot be
relabelled as if they used the corrected rules. Legacy SFHT resume state also
lacks RNG progress; exact resumed-versus-uninterrupted scheduling remains an
open audit item for a later training-state version and frozen training plan.

## Validation

- The full OpenCL-enabled ordinary suite passed 118 tests, with seven manual
  diagnostics/benchmarks ignored. After adding resume hardening, eight focused
  SFHT tests and the CLI override test passed. The two new gradient regressions
  were observed failing before their repairs. Mastering tests cover bypass,
  timing response, budgets 0/0.5/2/6 dB, sparse peaks, silence and mono/stereo.
- The separately selected native CPU/OpenCL parity diagnostic passed; largest
  absolute discrepancy was 5.96e-8. SFHT dense parity also ran with the Termux
  ICD environment enabled. Backend layouts/kernels were not changed.
- On the fixed 12-second procedural fixture, omitted controls give byte-identical
  float masters against the retained parent executable for forced auto-tune
  with SFHT/CFM specialists, legacy bypass routing and standalone mastering.
  This is a tested fixture, not universal bit-identity on every recording.
- The CLI integration check confirms explicit default-valued ratio/threshold/
  strength and a bypass mode survive auto-tune. The canonical and legacy CLI
  aliases expose the new controls. Fresh-output rules and existing recordings
  remain intact.

The debug CLI-parser regression initially exceeded Android's default test-thread
stack; it now uses an explicit bounded 8 MiB test thread. The normal release CLI
passed direct command execution. This test-harness issue was not presented as
an audio or ML result.

## Fixed local song excerpts (Tier 3, not listening evidence)

Both owner-selected songs use fixed 30–42 second source excerpts. Parent uses
auto-tune; candidate adds `--no-glue --max-limiter-gr-db 2`. The restored input
and frozen models are the same, but finalizers may differ in loudness.

| Track / variant | LUFS | True peak dBTP | Crest dB | Max glue GR dB | Max limiter GR dB |
|---|---:|---:|---:|---:|---:|
| Chasing Horizons / parent | -11.0125 | -1.0000 | 12.7910 | 0.5859 | 0.7874 |
| Chasing Horizons / controls | -11.0121 | -1.0000 | 12.7717 | 0 | 0.7854 |
| Feelin' Catchy / parent | -11.0894 | -0.9998 | 12.0379 | 0.5064 | 1.4840 |
| Feelin' Catchy / controls | -11.0847 | -1.0000 | 12.0178 | 0 | 1.4486 |

**Limited/negative result:** the 2 dB budget is not binding on either fixed
excerpt. Glue bypass yields little change and crest decreases about 0.02 dB.
These excerpts therefore establish integration and bypass, not better bass
attack or proof that the limiter budget helps these passages. Binding-budget
behavior and lower loudness are established on the fixed sparse-peak procedural
control. No owner audition or new full-song preference exists for this checkpoint.
Do not select replacement excerpts based on these outcomes.

All audio, source identities, logs and intermediate files stay in ignored local
`runs/punch-controls-20261012T022505Z/`. Public artifacts contain scalar
measurements and code/model provenance only. Compilation uses Cargo `-j 8`, with
sequential runs/builds. No speed experiment ran. About 31 GB remains free; no
package cache or frozen/run evidence cleanup was necessary.

## Research guidance and independent challenge

The earlier sidechain/bass-transient experiment supports retaining the glue
detector highpass. Next-evolution validation explicitly did not establish that
slow clocks prevent crescendo clamping. The new budget makes that tradeoff
inspectable without changing learned architectures. Primary references and
implemented/proposed distinctions are in
[REFERENCES.md](REFERENCES.md#punch-control-checkpoint-provenance).

Giannoulis/Massberg/Reiss provides detector and timing mathematics, supporting
controlled attack comparisons. Zaunschirm/Reiss/Klapuri motivates separate
sub-band onset/body diagnostics before a transient-expansion experiment.
Gorlow/Reiss motivates a procedural known-compressor inverse control, subject to
known model/parameter assumptions. Neither new transient modification nor
compression inversion is implemented. Paper results do not prove this project's
sound quality; no large pretrained model or natural-audio training is introduced.

A bounded Hermes math review completed through agent-orchestra. A broader
Antigravity audit timed out, and a separate Hermes file review returned a failed
task despite partial text; neither is treated as a completed verification. The
completed review's five-point assessment, with coordinator corrections, is:

1. Strongest evidence: exact clamp calculus, matched Fourier adjoint and measured
   parent parity/peak-budget tests.
2. Uncertainty: the existing true-peak meter and fixed fixtures do not establish
   every signal's behavior or audible bass-attack quality.
3. Overclaims to avoid: clipping-free playback everywhere, restored dynamics,
   better learning, or listener preference. The gain proof needs a **lower**
   bound on gain, not merely `gain <= 1`; optimizer clipping modifies gradients.
4. Confounds: lowering loudness changes auditions; equalize comparisons with
   a common downward gain policy. Existing full-band energy correlation is not
   an onset-sharpness metric.
5. Next discriminating work: input-selected bass/upper-bass onset attack/body
   diagnostics and controlled listening on full tracks. Later compare corrected
   training rules at matched model size and fixed procedural schedules, with
   fresh seed/family and null controls. Model enlargement is not supported yet.
