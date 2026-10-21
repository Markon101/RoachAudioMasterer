# Source-triggered output-window diagnostic, V2

The single V2 attempt completed with 320 shared source-selected events. These
fixed windows show near-zero median candidate/control RMS attack-to-sustain
differences, with localized changes retained in the distributions. They do not
establish or refute the owner's perceived punch difference, and do not establish
source-relative attack preservation, restoration, superiority or stage causation.

V1 remains stopped: its source/output alignment gate failed. Its nine retained
files are unchanged from commit `5c55880`; exact V1 code/tests are preserved in
that commit. V2 is a separately declared **comparative output-window** scope,
not a successful retry of the source-anchored claim. The source's waveform
windows are used only for event selection and shared activity/null rules.
Output-minus-source ratios are omitted from raw data and summary.

`design.json` was frozen at 2026-10-21 03:51:04.875567 UTC, before any source
event analysis. Its detector, windows, bands, FIR and shared null rules exactly
match V1 and are regression-tested. The detector uses source stereo energy in
5 ms frames against the preceding 40 ms, a 3 dB rise threshold, fixed local
maxima and 200 ms strongest-first suppression. It produced 700 local-maximum
candidates: 320 accepted, 380 suppressed and zero boundary exclusions. No event
was selected or removed using either output. The earlier independent 518/519
output-onset counts are not used.

All three WAVs are native 48 kHz stereo, 6,739,200 frames (140.4 seconds). The
same half-open pre [-80,-20), attack [-5,35) and sustain [60,160) ms windows are
applied at each input-derived onset, with full FIR support. The bands are
fullband, 30–200, 200–2000, 2000–8000 and 8000–20000 Hz. Each filtered band uses
the same centered 4097-tap Hamming-windowed sinc coefficients for every file.
RMS pools both channels' squared samples; peaks take the largest absolute
native sample across either channel. These are finite-filter proxies, not
instrument stems or continuous-time/true-peak measurements.

## Paired levels and contrast

Every value below is candidate minus control in dB. Columns are medians of
per-event differences, so medians in separate columns need not subtract exactly.
Attack/sustain RMS ratios are invariant to independent positive uniform gains
with this frozen source mask; absolute level differences remain level-dependent.

| Band | Attack RMS median | Sustain RMS median | Attack/sustain RMS median | Attack/sustain p10 to p90 | Attack/sustain min to max |
| --- | ---: | ---: | ---: | --- | --- |
| Fullband | -0.000245 | -0.000295 | +0.000004 | -0.003395 to +0.003861 | -0.101561 to +0.014181 |
| 30–200 Hz | -0.000277 | -0.000283 | +0.000008 | -0.000966 to +0.001203 | -0.059328 to +0.007802 |
| 200–2000 Hz | -0.000294 | -0.000256 | -0.000030 | -0.005709 to +0.005592 | -0.103849 to +0.021662 |
| 2000–8000 Hz | +0.000089 | -0.000286 | +0.000005 | -0.030402 to +0.035942 | -0.369629 to +0.117267 |
| 8000–20000 Hz | -0.000935 | -0.000324 | -0.000517 | -0.004883 to +0.004361 | -0.169865 to +0.085764 |

Fullband attack/sustain differences are negative for 157 and positive for 163
events; bass counts are 155/165 and presence counts 156/164. These counts are
descriptive, not event-level replications or significance tests. No audibility
threshold or composite score is introduced.

Peak and crest behavior is less concentrated than RMS contrast. Presence attack
peak differences span -2.019723 to +0.664873 dB, with p10/p90
-0.137408/+0.138959 dB; fullband attack peaks span -0.264329 to +0.238946 dB.
The complete summary retains all pre/attack/sustain RMS, peak and crest deltas,
all paired ratios, signs, means, quantiles, extrema and null counts.

Three retained examples illustrate why an aggregate or ratio cannot be called
attack loss:

- E0093, 41.710 s: fullband attack RMS -0.183060 dB and sustain RMS
  -0.081499 dB, giving attack/sustain -0.101561 dB.
- E0315, 137.200 s: presence attack RMS **+0.042070 dB** and sustain RMS
  **+0.411699 dB**, giving attack/sustain -0.369629 dB. Here the lower ratio
  accompanies an increase in both measured windows.
- E0281, 120.375 s: presence attack peak -2.019723 dB, attack RMS
  -0.308176 dB and sustain RMS -0.007458 dB. This localized change remains
  visible despite the near-zero medians; it is not assigned an audible cause.

The previously listened-to [48,78) s excerpt contains 82 of the same events.
Its attack/sustain RMS medians are -0.000074 dB fullband, +0.000020 dB bass,
-0.000571 dB low-mid, -0.000835 dB presence and -0.000564 dB high band.
It is a dependent subset, not new validation. A single low-activity source bass
pre-window nulls its attack/pre RMS and crest-difference ratios in all outputs;
the event remains. All attack/sustain ratios have 320 valid pairs.

## Sample alignment and treatment-band timing

The existing five 80–1200 Hz control/candidate anchors pass their unchanged
zero-lag/correlation/margin gates, with correlations >=0.9999992457. This band
is below the Mid treatment and cannot alone validate treatment-band timing.
V1 source/output alignment remains failed: best source-relative lags were
[-3,0,0,-1,-1] samples at its five anchors, with some correlations below 0.9.
Those failures were not relabeled as passes or corrected with fitted shifts.

V2 additionally retains descriptive control/candidate correlation searches in
the fixed 2–8 kHz **attack** window for every event, over +/-10 ms. No tolerance
was fitted, no gate was applied, no event was dropped and no waveform shifted.
All 320 searches peak at zero integer lag. Zero-lag correlation is median
0.999132, p10 0.995433, minimum 0.856401 and maximum 0.999999999883.
The [48,78) s subset also has all 82 best lags zero; minimum correlation is
0.981102. Every raw timing observation, including the weakest match, is retained.

These observations provide no evidence of a relative integer-sample shift in
these particular presence-band windows. Band-limited correlation mixes phase,
spectrum and timing, and does not guarantee sub-sample/every-event acoustic
alignment, continuous-time timing, or perceived transient identity. The source
detector itself has 5 ms resolution and the symmetric filters have pre/post
ringing. Windows can contain multiple instruments and decays.

## Reproduction, evidence and limits

```sh
LD_PRELOAD=/data/data/com.termux/files/usr/lib/libc++_shared.so \
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python -m unittest scripts.test_matched_attack_audit -v
```

All 16 tests pass, including the twelve V1 controls, unchanged scientific
definitions/default V1 scope, omitted V2 source-relative ratios, a deliberately
seven-sample-shifted presence event retained descriptively, and degenerate timing
nulls without event removal. `tests.json` and `tests.txt` preserve the command
and result. V1's own test receipt remains its original 12-test result.

The real invocation is retained in `invocation.json`:

```sh
LD_PRELOAD=/data/data/com.termux/files/usr/lib/libc++_shared.so \
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python scripts/matched_attack_audit.py \
  --design artifacts/matched-attack-paired-20261021/design.json
```

This completed directory is create-only and must not be rerun in place. A repeat
requires separately authorized fresh provenance/output paths. Artifact-local
HALT/STOP latches, periodic pins, finite/geometry/alignment checks and a 512 MiB
observed process-HWM guard remain enforced. The conservative observed HWM is
153.31 MiB; `/proc` VmHWM reports 47.37 MiB while `getrusage` reports a larger
maximum. Both are retained without exact allocation attribution. Timestamps and
test durations are provenance, with no speed claim. NumPy-only processing reads
bounded <=65,536-frame blocks and processes one event/recording/band at a time.

`selection.json`, `summary.json`, `alignment.json`, `source_hashes.json` and
`COMPLETE.json` retain definitions, source identities, raw selection/exclusions,
paired distributions and artifact hashes. Detailed paired observations stay
local in ignored `runs/matched-attack-paired-20261021/paired_events.jsonl`
(4,893,892 bytes; SHA256
`0500a8aee8d4ca0d068386e2a45a2ddc7c54352095e22267b765990a573010f6`),
with an artifact path/size/count/hash receipt. They contain measurements, not
audio copies. `integrity_audit.json` verifies every saved row's source selection,
coordinates, pair arithmetic, scope omissions, summary recomputation, hashes,
V1 preservation and unchanged frozen executable/audio.

Only one source song and final masters are involved; pre-limiter/intermediate
signals are unavailable to this audit. The owner feedback and its listening
limitations remain unchanged. These descriptions cannot identify a stage cause,
settle perceptual punch/coherence, justify DSP tuning, establish restoration or
support production promotion. No audio was rerendered or altered.
