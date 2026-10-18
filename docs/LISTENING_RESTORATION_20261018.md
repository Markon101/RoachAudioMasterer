# Listening package restoration — 2026-10-18

External cleanup removed the previously prepared WAVs while leaving their READY
markers. A marker alone is insufficient evidence that a package is playable.
The missing packages were rebuilt from retained, hash-verified historical master
FLACs using the original private assignments, excerpts and deterministic seeds.
The old folders retain a renamed stale marker rather than claiming readiness.

The current listener folders are:

- `runs/mission-20261018/listen/session1`
- `runs/mission-20261018/listen/session2`
- `runs/mission-20261018/listen/session3`

Each contains eight A/B trials, neutral instructions and `responses.csv`. The
organizer verified all 48 audio files against their private SHA256 records with
the new `blind_listening.py verify` command. All 13 harness tests passed,
including a procedural FFmpeg round trip and missing/changed-audio rejection.
Repeat verification before handing off a copied or cleaned package.

For listening, follow `INSTRUCTIONS.txt` in each session, keep playback settings
and volume fixed, and fill in `responses.csv` before seeking any answer key.
Indistinguishable and abstain are valid responses. Short trials and the longer
session excerpts cannot establish prolonged fatigue. This asks whether the
historical outputs differ or are preferred at matched playback loudness.

Keep the organizer folders under
`runs/mission-20261018/listening-private/` private; they contain identities and
measurements. Do not publish audio, keys or private manifests. The full local
restoration receipt is `listening-private/restoration_receipt.json`; the
[public receipt](../artifacts/mission-20261018/listening_restoration.json)
contains only readiness and scope.

These remain historical-playback comparisons of the three available versions.
The original alpha delivery is still unavailable at its documented paths, and
equal-target architecture renders remain unprepared. Original Feelin' Catchy
and Singularity FLAC inputs are present and hash-verified; the old Chasing source
WAV and Stage2 WAV intermediates are absent. No owner judgments have been
received, no key has been revealed, and no Tier 4 result or promotion is claimed.
