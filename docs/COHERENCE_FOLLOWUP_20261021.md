# Coherence followup: fixed attacks and cross-song gates

The first listening session suggested a possible high-coherence/bass-punch
tradeoff, with identity-control and playback-level limitations. No DSP retuning
or production promotion follows. The new work preserves that feedback and tests
cheaper objective uncertainties before further listening.

## Fixed attack windows on the existing Chasing Horizons pair

The first preregistered source-anchored diagnostic **stopped** before event
analysis: source-to-master alignment checks failed. Shared mastering changes can
alter band-limited correlation peaks; the result does not establish literal file
delay. [V1 evidence](../artifacts/matched-attack-20261021/README.md) remains
immutable at commit `5c55880`.

Independent review identified that passing 80–1200 Hz output-pair anchors did
not cover the treated Mid band. A separately declared narrower experiment kept
the source-only detector, windows, filters and null rules unchanged, omitted
source-relative reconstruction claims, and retained event-local 2–8 kHz timing
checks without shifts or output-dependent event exclusions.

[V2](../artifacts/matched-attack-paired-20261021/README.md) completed on 320
shared events, including 82 in the prior excerpt. RMS attack/sustain median
differences are near zero; localized presence-band changes remain, including
an attack-peak delta of -2.0197 dB. All 320 presence timing searches peak at zero
integer lag; this is not a sub-sample or perceptual guarantee. A lower ratio can
reflect increased sustain: one event has attack RMS +0.0421 dB and sustain
+0.4117 dB. The principal investigator independently replicated two adversarial
events with direct convolution, agreeing with the worker FFT deltas within
1e-9 dB. Sixteen synthetic/containment/scope tests pass. Detailed 4.9 MB event
rows remain local with committed hashes and compact summaries.

Tier 2 covers the tests and arithmetic replication; Tier 3 covers these bounded
measurements. The owner's perceived punch/coherence tradeoff remains unresolved;
this is not evidence of broad attack loss, perceptual equivalence or superiority.

## Singularity Moment Remix v2: control fails before candidate

The unchanged pinned control completed native processing, then failed the
predeclared known-band proxy: **1.141326 dB versus <1 dB**. Mono, transient-energy
agreement and peak checks passed. No candidate or listening package was produced.
The original run remains latched. [Stop receipt and recipe](../artifacts/listening-singularity-20261021/README.md)
retain all hashes and the explicit pre-render prerequisite revision. That revision
permitted inconclusive prior-song diagnostics; no audio/resource tolerance changed.
The failed path did not persist its sampled process HWM, so none is invented.

The native proxy is a first-channel filtered-RMS ratio normalized by broadband
RMS, with 1–2.5 kHz passband and tapered skirts. Its legacy name is not a maximum
over frequencies or a trusted-bin preservation guarantee. Source comments now
state the actual <1 dB threshold and >=0.90 block-energy cosine threshold;
arithmetic and historical binaries stay unchanged.

[Cached-stage trace](../artifacts/singularity-preservation-20261021/trace-v1/report.md)
reaches 1.140267 dB at final mastering, 0.001059 dB below the saved native result.
Clean polish contributes the largest signed increment (+0.714347 dB), followed
by final mastering (+0.360222 dB). The first analytical crossing is the mastered
output. These are cached-boundary observations, not denoise/EQ/limiter ablations.
The float64 Fourier/Parseval calculation approximates native f32 operations;
the native failure remains binding. Twelve known-answer/containment tests pass;
observed process HWM is 225.223 MiB. No audio was rerendered for this trace.

## One additional source, then stop source selection if gates fail

Feelin’ Catchy is the next mission-priority reference, fixed before inspecting
any of its outputs. The complete 179.902375-second original is used, with the
same pinned binary, weights, settings, 1536 MiB process ceiling and native gates.
[Plan](../artifacts/listening-feelin-20261021/plan.json) and
[prerequisite](../artifacts/listening-feelin-20261021/attack_prerequisite.json)
are recorded before rendering. The shared runner now accepts an explicit plan
and preserves child execution/memory receipts even when quality validation fails.
Eleven containment/plan/execution tests and five package-provenance tests pass.
No further source search is authorized by this experiment if this control fails.

Feelin's first control stopped in the existing PH stage at observed process HWM
1549.117 MiB against the unchanged 1536 MiB cap. Its execution receipt survives
the stop; no candidate or listening package exists. No more sources will be tried
under this plan. That stop motivated sequential stereo-audit buffer preparation,
with exact short audio/audit parity before the fresh retry of this same source.
No cap increase or skipped quality audit was authorized.
Any eventual package uses fresh counterbalanced source positions, transparent
attenuation near native master loudness, categorical CSV plus optional notes,
and a comfortable playback check. The 24-test listening harness preserves legacy
seeded assignments by default. Every new speed experiment still needs fresh
foreground confirmation; these are quality runs, with no speedup claim.

## Same-source retry earns a new listening package

The [sequential stereo-audit memory change](PH_STEREO_MEMORY_20261021.md), commit
`fc92ea1`, passed27 focused Rust tests,48 procedural WAV comparisons across13 modes
and exact synthetic/real10s PH/finalizer audit and audio parity. A fresh Feelin
control then passed all native quality gates under the same1536MiB cap; its five
full pre-PH stage hashes match the stopped run. Only after those checks did the
single coherence candidate run; it also passed. Maximum observed process HWM
across the completed pair is1057.277MiB. The old stopped peak is censored, so no
exact percentage full-run memory reduction is reported. No complete old final
waveform exists; bounded parity is not whole-track before/after waveform parity.

The verified package is delivered locally at
`/sdcard/Download/Roach Listening Feelin Catchy 20261021`: four trials/eight48k
stereo PCM24 WAVs, categorical CSV, optional notes and neutral instructions.
Fresh source positions are counterbalanced; the old Chasing trial mapping does
not apply. Matching uses transparent attenuation near the native master level,
without remastering or added limiting. The owner should lower the previous
max-volume setting before the playback-level check. All12 delivered files match
source hashes. Independent review verifies real audio formats, lengths, identity
controls, source-position balance, receipt chain and absence of source labels or
technical metrics in the listener payload.

[Milestone](../artifacts/listening-feelin-retry-20261021/milestone.json),
[full native receipts](../artifacts/listening-feelin-retry-20261021/fullsong_complete.json),
[boundary audit](../artifacts/listening-feelin-retry-20261021/listener_boundary_review.json)
and [delivery](../artifacts/listening-feelin-retry-20261021/delivery.json) preserve
the exact binary/model/input/configuration and checks. Original stops, Chasing
audio and raw responses remain intact. Owner Feelin judgments and preference are
pending. No subjective production promotion occurred.

## Independent clean-stage counterfactuals after delivery

While the Feelin package stayed fixed, a bounded native example replayed the
cached Singularity Mid input through the actual `clean_audio_v1_3` function.
Its combined denoise+EQ output had to match the saved clean WAV **byte-for-byte**
before any branch contrasts were accepted. It passed. The preregistered arms
were raw identity, guard-only, denoise-only, EQ-only and combined; the global
0.99 peak guard is distinguished from a true clean bypass. On this input the
guard-only output equals identity exactly.

Signed analytical shape increments versus Mid are +0.099314 dB for denoise,
+0.607245 dB for EQ and +0.714347 dB combined; the explicit factorial interaction
is +0.007788 dB. These isolate effects on this local normalized-RMS proxy, not
audible damage, preference or the final mastered spectrum. Denoise includes both
high-band and dual-zone quiet-mid processing. EQ sees a different input after
denoise; the finalizer is not included in these counterfactuals. Do not subtract
these numbers from the failed final-master score to predict a new master.

[Plan and report](../artifacts/singularity-clean-factorial-20261021/report.md),
[verification](../artifacts/singularity-clean-factorial-20261021/verification.json)
and [root reconciliation](../artifacts/singularity-clean-factorial-20261021/root_reconciliation.json)
retain exact source/output hashes and all five controls. Nine focused Rust
example/module tests and five Python tests pass; native child HWM stays below
388.879 MiB and analysis below225.043 MiB. The original failure, production code
and delivered listening package remain unchanged. Raw intermediate WAVs stay
local; no additional master or listening candidate was generated.

The strongest next independent engineering test is a matched native finalizer
replay on the cached pre-master input, requiring exact baseline reproduction
before glue/limiter counterfactuals. Declare it separately and preserve the
unchanged native gate. The next perceptual decision remains the owner's Feelin
judgments; do not promote or retune based on these proxy measurements alone.
