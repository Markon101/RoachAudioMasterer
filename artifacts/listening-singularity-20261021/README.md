# Singularity control stopped before a candidate

The pinned Smooth-style control completed native processing, but failed the
unchanged native `known_band_preserved` gate: 1.141326 dB versus the <1 dB
criterion. Mono compatibility, transient timing and final native peak gates pass.
The session is latched at `runs/listening-singularity-20261021/fullsong-v1/HALTED.json`.
No candidate or listening package was produced. Do not resume or clear the latch.

`stop_receipt.json` retains source, binary, protocol, output and summary hashes.
The failed path did not persist the sampled process high-water mark; it is unknown,
and reaching validation without a resource stop does not supply an exact peak.
Native stage audio stays local. An offline stage trace is investigating where
the first-channel broadband-normalized filtered-RMS shape proxy changes. It is
not a maximum over passbands or the protected-band Mid projection error.

The original plan is preserved. `plan_revision_v2.json` explicitly permits an
inconclusive prior-song punch diagnostic, after independent review, before this
render. It changes no native quality tolerance, weights, configuration or memory
ceiling. V1's source-alignment failure remains recorded; the narrower paired-only
diagnostic is a separate experiment.

Reproduction uses the committed recipe, fresh output paths and original pins:
`PYTHONDONTWRITEBYTECODE=1 python artifacts/listening-singularity-20261021/render_pair.py`.
This exact output directory is immutable; repeat invocation refuses it.
Eight mocked containment/revision tests pass in
`scripts/test_render_listening_pair.py`; five qualification-chain tests pass in
`scripts/test_package_listening_pair.py`. Package/delivery recipes remain unused
because eligibility was not earned. No speed comparison or promotion occurred.
