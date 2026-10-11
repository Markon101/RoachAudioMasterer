# Next evolution: frozen implementation and validation plan

Date: 2026-10-11. Parent: `c67e21a`; champion: `8570929`
(`v1.1.0-alpha.1-champion`). This plan fixes the experiment before results.

1. Validate the handover intent binary on the complete local Fourier Longing
   input, with auto-tuning, frozen SFHT/CFM-v2 models and phone export disabled.
   Inspect intent, actual stage settings, loudness, peak and preservation.
2. Add opt-in input-only coupled envelope clocks (5 ms, 150 ms, 2.5 s).
   Use a cascade of exponential energy trackers, not new learned dynamics or a
   claim of Titan reproduction. Slow phrase energy conditions the allowed fast
   Port-Hamiltonian actuator excursion; leave the existing resonator unchanged.
3. Add a separate opt-in contrast budget. Use input-relative medium-envelope
   quantiles, a silence gate and continuous slew-limited interpolation to let
   quieter passages receive less width, room and actuator excursion. Keep the
   base shelf, bass guard, known-band projection and final limiter intact.
   This is heuristic spatial/modulation allocation, not semantic verse/chorus
   recognition, amplitude expansion or recovered dynamics.
4. Test disabled-path parity, silence, constant-level abstention, gain/channel
   invariance, impulse/ramp/section transitions, unseen procedural seeds and
   families, known-band preservation and mono behavior. Compare against static
   champion and simple single-clock controls; no best-seed selection.
5. Build the champion source in an isolated local directory and compare complete
   masters on the same source/settings. Record all negative and ambiguous
   results. No new training, speed benchmark, external audio upload or champion
   promotion. Compilation uses at most two jobs, following AGENTS.md.

Provenance repairs needed for validation: retain the original tuning card and
write the intent-adjusted effective card; record actual applied stage settings.
Phone export must create a fresh full-song FLAC destination without WAV copying
or overwriting existing files. Research audio stays in ignored local runs.

The new envelope cascade and allocation rules are project-specific DSP. The
unchanged Port-Hamiltonian/AVF method provenance remains in
[REFERENCES.md](REFERENCES.md). Tests establish implementation behavior (Tier 2),
measurements establish change (Tier 3); perceptual benefit remains unestablished.

## Reproduction

```sh
cargo test --locked --features opencl --bin roach-audio-masterer -j 2 -- --test-threads=1
# Only this named quality diagnostic is enabled; speed tests remain ignored.
ROACH_EVOLUTION_AUDIT_OUT=runs/fresh-evolution-audit \
  cargo test --release --locked --features opencl --bin roach-audio-masterer -j 2 \
  next_evolution_procedural_audit -- --ignored --nocapture
cargo build --release --locked --features opencl -j 2
./target/release/roach-audio-masterer auto-master --input song.wav \
  --auto-tune --coupled-clocks --contrast-budget --backend opencl \
  --out runs/fresh-evolution-master --no-export-sdcard
cargo run --release --locked --example compare_evolution -j 2 -- \
  --reference frozen-champion-master.wav --candidate new-master.wav \
  --out runs/fresh-master-comparison.json
```

For CPU-only work omit the OpenCL feature and select `--backend cpu`.
The procedural audit fixes seeds 950021, 950089, 960031 across tones and noise,
uses all six scenes, and compares static PH settings, clocks only, budget only,
both together and a matched static shelf. It also writes one procedural float
input for disabled-path CLI checks. These are local DSP controls, not independent
natural recordings or an established model-family generalization result.
