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

Feelin's control stopped in the existing PH stage at observed process HWM
1549.117 MiB against the unchanged 1536 MiB cap. Its execution receipt survives
the stop; no candidate or listening package exists. No more sources will be tried
under this plan. The next bounded engineering step is sequential stereo-audit
buffer preparation, with exact short audio/audit parity before any fresh retry of
this same source. No cap increase or skipped quality audit is authorized.
Any eventual package uses fresh counterbalanced source positions, transparent
attenuation near native master loudness, categorical CSV plus optional notes,
and a comfortable playback check. The 24-test listening harness preserves legacy
seeded assignments by default. Every new speed experiment still needs fresh
foreground confirmation; these are quality runs, with no speedup claim.
