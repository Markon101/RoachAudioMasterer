# Sprint 2 Plan: Extending Flow Matching to Mid Frequencies (500 Hz – 6 kHz)

## 1. Motivation & Context
In Sprint 1, high-band adaptive denoising, bounded auto-EQ, and 8-core concurrency were implemented and verified on the Adreno 830 / Oryon architecture.

Currently, the frozen high-band flow models (`artifacts/native-longer-v2/flow3000.json`, `artifacts/rich-field-v1/basis.json`) operate exclusively on frequencies above $3.5\text{--}6\text{ kHz}$. Audio deficiencies in damaged, band-limited, or overprocessed material frequently affect the **midrange** ($500\text{ Hz}\text{--}6000\text{ Hz}$), where core vocal body, snare fundamental attacks, guitar resonance, and instrument timbre reside.

This sprint extends the conditional flow matching framework into the midrange, fulfilling the core architectural requirement in `docs/ARCHITECTURE.md` of a unified multi-band restoration system.

---

## 2. Mathematical Formulation

### Generalized Damage & Mid-Band Distribution
Instead of fixing cutoff in the high band ($3500\text{--}8000\text{ Hz}$), `MidDamage` randomizes cutoffs into the core midrange:
- Cutoff: $f_{cut} \sim \mathcal{U}(500.0, 3500.0)\text{ Hz}$
- Transition: $\Delta f \sim \mathcal{U}(150.0, 600.0)\text{ Hz}$
- Power: $p \sim \mathcal{U}(1.0, 3.5)$

At $f_{cut} = 1000\text{ Hz}$, all frequencies from $1000\text{ Hz}$ to $24000\text{ Hz}$ are untrusted and reconstructed by the flow-matching operator.

### Subharmonic Feature Routing
The shared frequency head (`HEAD_INPUT = 184`) takes subharmonic peak features routed through frequency ratios $m \in \{1, 2, 3, 4, 6\}$:
$$f_{sub} = \frac{f_k}{m}$$
For any target mid-band frequency $f_k \in [500, 3500]\text{ Hz}$:
- For $f_k = 1500\text{ Hz}$: $f_{sub} \in \{1500, 750, 500, 375, 250\}\text{ Hz}$.
- When $f_{cut} = 600\text{ Hz}$, the subharmonics $500\text{ Hz}, 375\text{ Hz}, 250\text{ Hz}$ lie strictly within the trusted low band ($< f_{cut}$).
- The network directly observes these surviving bass/low-mid fundamental partials to guide the trajectory of mid-band overtones!

### Continuous Flow Matching
Linear interpolation between complex DSP prior $z(0)$ and target complex spectrum $y$:
$$z(t) = (1 - t) z(0) + t y, \quad t \in [0, 1]$$
Target velocity:
$$v^*(z(t), t) = y - z(0)$$
Objective minimized via explicit Adam backpropagation:
$$\mathcal{L} = \mathbb{E}_{t, y, z(0)} \left[ \| v_\theta(z(t), t, c) - v^*(z(t), t) \|^2 \right]$$

---

## 3. Implementation Architecture

1. **`src/scene_features.rs` / `src/rich_synth.rs`**:
   - Add `Damage::mid_range(seed)` generating cutoffs in $500\text{--}3500\text{ Hz}$.
   - Ensure procedural pitch synthesis provides rich fundamental excitation in $50\text{--}400\text{ Hz}$ so that mid-band partials have strong physical coupling.
2. **`src/rich_mid.rs`**:
   - `train_mid_flow`: Dedicated mid-band flow training pipeline.
   - Persisted Adam optimizer, CPU/GPU OpenCL parity, 8-core concurrency.
   - `restore_mid`: Mid-band restoration subcommand and pipeline integration.
3. **CLI Integration (`src/main.rs`)**:
   - `RichMidTrain`: Subcommand to train mid-band flow models.
   - `RichMidRestore`: Subcommand to test mid-band restoration on excerpts and songs.
   - Preserves all frozen high-band checkpoints intact.

---

## 4. Verification Plan
- [x] Analytical gradient check on mid-band velocity loss.
- [x] Zero-update control test: un-updated model equals DSP prior.
- [x] Known-band preservation test: frequencies below $f_{cut}$ remain 100% unchanged.
- [x] GPU/CPU numerical parity check under 8-core OpenCL batching.
