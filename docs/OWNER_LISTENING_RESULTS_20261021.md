# Chasing Horizons owner listening response — 2026-10-21

The owner returned `responses (1).csv` and explicitly confirmed listening to both
A and B in every trial. All four preference entries are B. Both actual comparisons
therefore select the matched Smooth-style control. Both identity trials also
select B despite byte-identical A/B files. This records the owner's choices;
neither control superiority nor candidate superiority is established by this
session. The default-off coherence candidate remains unpromoted and Smooth
`493cdca` remains the production champion.

| Trial | Duration | A | B | Submitted preference | Interpretation |
| --- | --- | --- | --- | --- | --- |
| T0001 | 140.4 s | Candidate | Candidate | B | Exact identity control |
| T0002 | 30 s | Candidate | Candidate | B | Exact identity control |
| T0003 | 30 s | Candidate | Matched control | B | Control selected |
| T0004 | 140.4 s | Candidate | Matched control | B | Control selected |

The 30-second excerpt begins at sample 2,304,000 (48 seconds at 48 kHz).
Full-song and excerpt observations are dependent, drawn from one song and listener.
Both nonidentity comparisons happened to assign the candidate to A; the all-B
pattern cannot separate source preference from order/label/playback effects.
The owner later described difficulty distinguishing the files.

## Receipt and response format

[Receipt](../artifacts/listening-results-20261021/receipt.json) preserves every
response field verbatim, revealed source labels, audio hashes and raw CSV hash.
The original bytes remain local at
`runs/listening-results-20261021/submission-v1/responses-original.csv`:
`f9c788751806ea62552d472c9d31d20bfd4dc7130d02f53dc5d5d3b431fa876f`.
The [create-only recorder](../artifacts/listening-results-20261021/record_submission.py)
verifies all eight source and delivered WAVs against the original private key
before recording. A separate native agent independently verified the delivered
hashes, response hash, source mapping and limitations; no paid external calls.
The existing listening harness passes all 18 tests via
`PYTHONDONTWRITEBYTECODE=1 python scripts/test_blind_listening.py`.
A separate receipt check confirms verbatim field equality with the archived CSV
and that the complete append-only research ledger still parses.
The original private key/seed, audio and raw archive remain local/ignored.

The CSV contains valid preference choices, but prose transient/tonal/spatial
fields, percentage confidence (75/90/90/80%) and free-text fatigue. It therefore
does not satisfy the strict categorical `blind_listening.py reveal` contract.
That contract was not weakened, no aspect preferences were inferred, and no
percentages were mapped to low/medium/high categories. The separate receipt is
an explicit free-text ingestion, not a successful strict categorical reveal.

Comments include favorable transient/tonal/spatial impressions, low fatigue
descriptions, and T0004's "slight high end hum" / "slight high band hum".
The aspect comments do not explicitly name a member. The preference entry alone
does not establish which member each adjective describes, or which produced the
hum. No causal hum or prolonged-fatigue conclusion follows.

After source assignments were revealed, the owner clarified that A sounded less
punchy than B. This explicitly supplies a perceived difference for the actual
comparisons; preserve it as a post-reveal explanation of the submitted preference,
not a new independent blind judgment. A was the coherence intervention and B the
Smooth-style control. A method name does not establish smoothing or attack loss.
The owner then reported that A's highs sounded more coherent, explaining that
attention had initially focused on the bass. Both observations are retained:
more coherent highs perceived in A and more punch perceived in B. This suggests
a possible perceptual tradeoff to investigate (Tier 5 hypothesis), without
rewriting the original B preferences or asserting a replicated mechanism effect.
The later high-band observation was made after the source reveal.

[Followup receipt](../artifacts/listening-results-20261021/followup.json) checks
the existing native final-master diagnostics without rerendering. Candidate minus
control mean attack crest is +0.00239 dB, attack-minus-pre-onset crest is -0.00529 dB,
and maximum limiter reduction is about +0.02309 dB. These are mixed aggregate
proxies, not proof or refutation of perceived punch loss. Onsets were detected
independently (519 versus 518), so these are not matched-event comparisons.
The field `attack_crest_expansion_db` describes attack versus pre-onset crest
within one output; it is not input-to-output expansion. The strongest next cheap
diagnostic is a fixed input-derived onset mask applied to both existing masters,
including final-output attack/sustain behavior. Do not tune the event detector
or thresholds to obtain the owner's reported difference.

## Evidence and next decision

Tier 2: package integrity verified; raw submission preserved without recoding.
Recorded source-blinded human choices exist, but **Tier 4 comparative superiority
is not established** because identity controls and the all-B pattern confound
interpretation. General preference, audible improvement, target fidelity and
hum attribution remain Tier 7 (unestablished). Prior exposure to mechanism metrics
also prevents a claim of naive-expectation blinding. No significance claim is made.

The objective numerical/continuity and memory gates remain valid within their
original scope; they never established preference. The coherent-but-wrong
procedural errors and healthy-input false additions remain open negative evidence.

The highest-information listening followup, if undertaken, uses these existing
masters in a fresh concealed session with deliberately counterbalanced source
positions and identity controls. Do not reuse revealed trial assignments as
blind data, rerender another candidate merely to obtain a favorable response,
tune DSP against these four judgments, or pool this with historical auditions.
A low-burden workflow should accept an optional notes field from the outset and
make indistinguishable/abstain easy. Inspect source/control/candidate hum spectra
only as an independent diagnostic, with listening attribution still unresolved.
Any new speed experiment continues to require fresh foreground confirmation.
