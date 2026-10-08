# Sprint 3 Plan: Low-End Flow Restoration & Tri-Band Cross-Signal Integration

## 1. Objectives & Motivation
Sprint 1 delivered high-band denoise, auto-EQ, multithreading, and OpenCL acceleration.
Sprint 2 extended conditional flow matching to the midrange ($500\text{ Hz}\text{--}6\text{ kHz}$) with clean harmonic priors, dual-boundary locking, and an 8-core bounded solver delivering 13.1x speedup.

Sprint 3 completes the full acoustic spectrum by:
1. **Eliminating the Synthetic LFO Texture**: Removing the hardcoded 1.7 Hz pitch vibrato and rigid FM modulations in `src/rich_synth.rs`, shifting to 75% rock-solid steady acoustic pitches and 25% subtle non-periodic human micro-drift ($0.05\text{--}0.2\text{ Hz}$).
2. **Low-End Flow Matching ($20\text{ Hz}\text{--}500\text{ Hz}$)**: Formulating the inverse problem of sub-bass & fundamental restoration from surviving higher harmonic overtones ($m \cdot f_k$, $m \in \{2, 3, 4, 5, 6\}$).
3. **Cross-Signal Tri-Band Architecture**: Building unified multi-band coordination where Low, Mid, and High restoration share global spectral context and phase coherence without crossover collision.
4. **Extended Procedural Training**: Scaling procedural training to 2,000 steps with analytic Adam backpropagation.
5. **Audition on Real Audio**: Restoring full-spectrum depth on `Feelin' Catchy` and verifying mastering metrics.

---

## 2. Mathematical Formulation

### Low-End Damage Model (`Damage::low_cut`)
In phone recordings, tinny playback captures, vinyl high-pass filtering, or weak mixes, energy is lost below a high-pass cutoff $f_{\text{cut}} \in [40.0, 350.0]\text{ Hz}$:
$$D(k) = \begin{cases} 0.0 & f_k \le f_{\text{cut}} - \Delta f \\ \frac{1}{2} \left[1 + \cos\left(\pi \frac{f_{\text{cut}} - f_k}{\Delta f}\right)\right]^p & f_{\text{cut}} - \Delta f < f_k \le f_{\text{cut}} \\ 1.0 & f_k > f_{\text{cut}} \end{cases}$$
Where $f_k > f_{\text{cut}}$ is the surviving trusted band, and $f_k \le f_{\text{cut}}$ is the missing fundamental/sub-bass region.

### Superharmonic Comb Feature Routing (Inverse Physics)
- In high-band completion, higher partials ($k \cdot f_0$) are inferred from lower fundamentals ($f_0$).
- In low-band restoration, the missing fundamental $f_k \in [20, 500]\text{ Hz}$ is inferred from surviving **superharmonics** in the low-mids:
$$f_{\text{overtone}} = m \cdot f_k, \quad m \in \{2, 3, 4, 5, 6\}$$
When $f_k < f_{\text{cut}}$, the overtones $2f_k, 3f_k, 4f_k$ lie above $f_{\text{cut}}$ in the surviving audio. The network extracts peak magnitudes and phases at these exact harmonic locations to calculate the fundamental's amplitude, pitch, and phase.

### Clean Sub-Bass Prior
White noise in sub-bass produces unwanted rumble, muddy DC offsets, and phase flutter. The prior $z(0)$ for low-end restoration uses:
$$z(0) = \text{harmonic\_projection}(p)$$
with **zero white noise** ($p.noise = 0$), guaranteeing tight, clean, punchy sub-bass transient attacks.

### Tri-Band Boundary Locking (`triband_waveform`)
Dual/tri-band locking via `dsp::lock_known_bands`:
- Low restoration locks everything above $low\_ceiling$ ($f \ge 350\text{--}500\text{ Hz}$).
- Mid restoration locks both below $mid\_cutoff$ and above $mid\_ceiling$.
- High completion locks everything below $high\_cutoff$.
This guarantees 100% bit-exact preservation of all trusted regions.

---

## 3. Work Breakdown & Milestones

### Milestone 1: De-LFO Synthetic Generators & Low-End Curriculum
- Modify `src/rich_synth.rs`:
  - Eliminate the fixed 1.7 Hz sine vibrato.
  - Implement 75% rock-solid steady pitches + 25% subtle non-periodic human drift.
  - Add sub-bass and kick fundamental synthetic families (808 sine drops, acoustic upright bass, kick transients, synth sub).
- Add unit tests verifying determinism, bounds, and absence of rigid periodic modulations.

### Milestone 2: Low-End Feature Routing & Damage in `src/scene_features.rs`
- Add `Damage::low_cut(seed)`.
- Implement superharmonic feature extraction for bins $f_k < 500\text{ Hz}$.
- Extend multi-band energy metrics: `(low_rms, mid_rms, high_rms)`.
- Verify with unit tests.

### Milestone 3: Low-End Model & Tri-Band Pipeline (`src/rich_low.rs` & `src/rich_triband.rs`)
- Implement `LowField` Basis model schema.
- Implement procedural training pipeline `rich_low::train` (2,000 steps).
- Implement `low_waveform` with upper-band locking ($f \ge low\_ceiling$).
- Add unified CLI commands: `rich-low-train`, `rich-low-restore`, and `rich-triband-restore` in `src/main.rs`.

### Milestone 4: Training & Verification
- Train 2,000 steps for `rich-low-v1` Basis model (`artifacts/rich-low-v1/basis.json`).
- Verify finite-difference gradients and known-band preservation tests.
- Record SHA256 checksums in `artifacts/rich-low-v1/SHA256SUMS`.

### Milestone 5: Full-Spectrum Audition on `Feelin' Catchy`
- Audition low-end restoration and tri-band pipeline on `Feelin' Catchy`.
- Measure sub-bass punch, transient crest factor, and spectral balance.
- Document results in `docs/LOW_BAND_FLOW_RESULTS.md` and `RESULTS.md`.
