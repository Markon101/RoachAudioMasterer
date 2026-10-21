# Local blind A/B listening protocol

This organizer guide describes `scripts/blind_listening.py`. Give listeners only
the generated listener folder and its neutral `INSTRUCTIONS.txt`. Keep this guide,
the input manifest and the organizer folder away from participants until judgments
are complete. All audio stays local. The tool uses Python's standard library and
installed FFmpeg/ffprobe; it does not call a model or online service.

The tool packages already rendered WAV/FLAC files. It does not run mastering,
choose favorable passages, infer missing provenance or establish content alignment.
Before creating the manifest, verify that all variants refer to the same source,
have the same musical time origin and have no relative processing latency. Fix
excerpts before listening, using source-based musical landmarks without consulting
candidate deltas or listener preferences. A common sample interval does not prove
that different renders have the same musical alignment.

## Two separate questions

`historical_playback` compares completed historical masters as they exist, with
playback loudness matched by attenuation. Different render targets and processing
settings can remain confounded. Preference applies to those particular masters;
it cannot isolate an architecture's effect.

`equal_target_architecture` requires every variant in a track to declare the same
original-source SHA-256 and requested render target, plus a full source git commit
for each variant. Keep other render settings and controls matched and retain their
provenance. These declarations are checked for completeness and equality, not
independently authenticated; the organizer must verify them against run records.
Matching requested targets alone does not isolate an architecture.

`procedural_diagnostic` is for short, fixed procedural passages used to judge a
specific local mechanism, such as a transient or texture change. It accepts only
excerpts labeled `diagnostic` with durations from 8 through 48 seconds. It does not
request a long excerpt, assess fatigue, or support long-session claims. Each
variant must declare `procedural: true`, a nonempty `run_record`, a full git commit
and a source SHA-256. Variants in one track must declare the same source hash.
These are metadata declarations: the tool checks their shape and same-source
equality, but does not authenticate them or inspect the run record. The organizer
must check each declaration against retained run evidence. Short procedural
judgments apply only to the prepared excerpts and listening setup; they do not
establish long-term preference, fatigue, general audio quality, or population
preference. The CSV fixes fatigue to `not_assessed`, and reveal rejects any other
fatigue value in this mode.

Keep these modes in separate sessions and interpret them separately; do not pool
their responses.

## Private manifest

Use a fresh output directory pair for every preparation. Paths below are illustrative;
the script resolves relative source paths against the manifest directory. Excerpt
bounds and any common trim use the original decoded sample grid, with an inclusive
start and exclusive end. The tool never resamples, shifts, pads or automatically
shortens audio. Historical and equal-target tracks require an exactly 30-second
`diagnostic` excerpt and a `long` excerpt lasting at least 120 seconds. Procedural
diagnostic tracks instead require only `diagnostic` excerpts of 8 through 48
seconds. Multiple excerpts are allowed, with at most 64 trials per session and
two to six variants per track.

```json
{
  "version": 1,
  "mode": "historical_playback",
  "seed": 7419321,
  "target_lufs": -23.0,
  "tracks": [
    {
      "id": "local_track_1",
      "variants": [
        {
          "label": "historical_variant_1",
          "path": "masters/variant1.flac",
          "provenance": {
            "run_record": "runs/variant1/summary.json",
            "git_commit": "FULL_40_CHARACTER_GIT_COMMIT",
            "source_sha256": "ORIGINAL_INPUT_64_CHARACTER_SHA256",
            "target_lufs": -11.0,
            "qualification": "Record any missing or uncertain history explicitly."
          }
        },
        {
          "label": "historical_variant_2",
          "path": "masters/variant2.flac",
          "provenance": {
            "run_record": "runs/variant2/summary.json",
            "git_commit": "FULL_40_CHARACTER_GIT_COMMIT",
            "source_sha256": "ORIGINAL_INPUT_64_CHARACTER_SHA256",
            "target_lufs": -13.0
          }
        }
      ],
      "excerpts": [
        {"id": "fixed_short", "kind": "diagnostic", "start_sample": 1440000, "frames": 1440000},
        {"id": "fixed_long", "kind": "long", "start_sample": 0, "frames": 5760000}
      ]
    }
  ]
}
```

This example assumes 48 kHz. The seed is private and must be fixed before audition;
do not try seeds to obtain a preferred order. Source files must have identical
sample rates, channel counts and decoded sample counts. For intentional end-length
differences only, declare a track-level `"common_trim": {"start_sample": 0,
"frames": 5760000}` containing every excerpt. The same original sample interval
then applies to every variant. Other rate/channel mismatches are rejected.

A short procedural mechanism check can use a separate manifest like this:

```json
{
  "version": 1,
  "mode": "procedural_diagnostic",
  "seed": 7419322,
  "target_lufs": -23.0,
  "tracks": [{
    "id": "fixed_procedural_phrase",
    "variants": [
      {"label": "arm_a", "path": "renders/arm-a.wav", "provenance": {
        "procedural": true, "run_record": "runs/arm-a/summary.json",
        "git_commit": "FULL_40_CHARACTER_GIT_COMMIT",
        "source_sha256": "SAME_64_CHARACTER_PROCEDURAL_SOURCE_SHA256"}},
      {"label": "arm_b", "path": "renders/arm-b.wav", "provenance": {
        "procedural": true, "run_record": "runs/arm-b/summary.json",
        "git_commit": "FULL_40_CHARACTER_GIT_COMMIT",
        "source_sha256": "SAME_64_CHARACTER_PROCEDURAL_SOURCE_SHA256"}}
    ],
    "excerpts": [
      {"id": "short_attack", "kind": "diagnostic", "start_sample": 432000, "frames": 624000},
      {"id": "longer_phrase", "kind": "diagnostic", "start_sample": 0, "frames": 1248000}
    ]
  }]
}
```

At 48 kHz these example excerpts are 13 and 26 seconds. Before preparation,
independently confirm that both renders use the same procedural source, musical
time origin and alignment. The run-record, commit and source-hash declarations
are retained privately but are not authenticated by the listening tool.

## Prepare and hand off

```sh
python scripts/blind_listening.py prepare /private/session.json \
  --listener /local/fresh-listener-session \
  --organizer /private/fresh-organizer-session
```

The organizer directory must be outside the listener directory. Both destinations
must be new; the command refuses existing paths. The private directory is created
with owner-only permissions. Directory access remains an operating-system trust
boundary: a participant who can access the organizer's account can read its key.
For self-listening, have another person hold the organizer directory and seed.

Each fixed excerpt gets all distinct variant pairs and one randomly selected
identity pair. Pair order, orientation and trial order are deterministic for the
private seed. Public names are `T0001/A.wav`, `T0001/B.wav`, and so on. Trial IDs
carry no track, variant, excerpt-kind or control identifiers. Participants may
still recognize music; this protocol conceals variant identity, not song identity.
This is a listening protocol, not concealment against digital inspection. Exact
identity controls necessarily have equal samples and hashes; file comparison can
identify them. Participants must defer file comparisons and diagnostics until
judgments are frozen. The private key is not a technical barrier on a shared account.

For new sessions, the optional private manifest field
`"assignment_policy": "counterbalanced_pairs"` randomizes each track/pair's
initial A/B orientation and alternates it across excerpts before shuffling trial
order. Two excerpts expose each variant once in each slot; an odd number leaves
an imbalance of at most one per pair. Identity pairs remain one randomly selected
variant per excerpt. This controls source-position imbalance, not expectations or
independent replication. Omitting the field, or specifying `"independent"`,
preserves the legacy seeded assignment sequence exactly. Other values are
rejected. Draw a private seed once; do not reroll it for a favored order.

For each pair, FFmpeg `ebur128` measures integrated loudness and true peak. The
shared playback target is no higher than the manifest ceiling, either member's
measured loudness, or the level needed to leave peak headroom. Each member receives
only constant attenuation, with no amplification, limiter, dynamics, `loudnorm`,
EQ or resampling. Output is native-rate 24-bit PCM WAV with source metadata removed.
The script verifies 24-bit PCM encoding and checks the encoded audio, requiring loudness within 0.15 LU
of the shared target and of the pair partner, exact sample counts/rate/channels,
and true peak below clipping. Measurements use the installed FFmpeg scanner's
reported precision. Silence/unmeasurable excerpts and failed gates abort; no READY
marker is left and incomplete listener output is removed. Failed private material
is retained, so retry with fresh directory names. The tool checks free storage for the
planned uncompressed pairs plus a safety margin; monitor phone storage during larger
sessions. Each pair contains two uncompressed excerpts.

The manifest target is a ceiling, not a requirement to use -23 LUFS. Minimize
attenuation when matching comparable masters, retaining encoded loudness/peak
checks. An earlier owner session required maximum M50x playback volume at
-23 LUFS, so check a comfortable listening level before judgments. If files remain
too quiet, report that and stop instead of silently accepting an inadequate
level. Then keep playback settings fixed. Do not remaster or add limiting just
to raise listening copies.

`READY` records successful package creation; it does not prove the WAVs still
exist. External cleanup can remove audio while leaving this marker behind. Before
hand-off, the organizer must verify the package against its private key:

```sh
python scripts/blind_listening.py verify \
  --organizer /private/fresh-organizer-session
```

Verification requires `READY`, every expected A/B WAV, and an exact match to each
recorded audio hash. It prints only a neutral readiness flag and trial/audio-file
counts. It does not print the key or source identities. Keep the organizer folder
private. If verification fails, restore the missing or changed package from
trusted retained files and verify again before hand-off. Run verification again
after any external copy or cleanup that could affect the listener folder.

Distribute the listener folder only after verification succeeds. It contains neutral
instructions, the WAV pairs and `responses.csv`; all labels, source paths, source
and output hashes, gains, measurements, provenance, seed, installed FFmpeg/ffprobe versions and assignments stay in
the organizer folder. The tool checks source hashes again before declaring ready.
Never add metrics, reference descriptions, delta tracks, processing claims or
expectation hints to the listener folder. Disable playback loudness normalization,
enhancement and crossfade, and keep the listening setup fixed. Take breaks; do not
pressure listeners to hear a difference.

## Responses and reveal

The response CSV has one fixed row per trial. `transient`, `tonal`, `spatial` and
`preference` each accept `A`, `B`, `indistinguishable` or `abstain`. `confidence`
accepts `low`, `medium`, `high`; `fatigue` accepts `none`, `mild`, `high`, `not_assessed`. Use `not_assessed` for short trials
or whenever fatigue cannot be assessed. Short trials cannot establish prolonged
fatigue; even a 120-second passage provides only a limited session impression.
These are categorical subjective judgments. Confidence and fatigue apply to the
whole trial. In `procedural_diagnostic`, fatigue is prefilled as `not_assessed`
and cannot be changed; the mode does not collect fatigue judgments.
The generated instructions give the same definitions without identifying controls.
Keep free-text comments in a separate `notes.txt`, keyed by trial identifier;
do not replace categorical CSV values with prose or percentage confidence.

```sh
python scripts/blind_listening.py reveal \
  --organizer /private/fresh-organizer-session \
  --responses /local/fresh-listener-session/responses.csv
```

Reveal requires every trial exactly once and every judgment to contain an allowed
category. It checks the READY marker and that listener audio still matches the
recorded hashes. Only then does it create `revealed.private.json` in the private
folder, containing the joined key and completed responses plus a response-file
hash. It refuses to overwrite an earlier reveal. The listener folder never receives
an answer key automatically. Freeze the first completed response set; do not revise
judgments after reveal or substitute seed orders after looking at results.

Report every trial, abstention, indistinguishable judgment, identity-control outcome
and fatigue qualification. This is a small local listening aid, not a calibrated
population study or an automatic statistical estimator. Identity controls alone
do not supply positive/near-null calibration; any later calibrated study needs a
separately frozen design. Automated tests establish implementation behavior
(Tier 2); scanner readings are measurements (Tier 3). Completed human judgments
can support a qualified report of subjective preference (Tier 4), without proving
lost-information recovery, generalization or architectural causation. Missing
historical provenance remains unknown (Tier 7).

Validation uses only mocked I/O and procedural audio:

```sh
python -m unittest discover -s scripts -p test_blind_listening.py -v
```
