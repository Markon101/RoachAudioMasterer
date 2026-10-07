# Sprint 1 Plan: High-Band Denoise, Bounded Auto-EQ, and 4-Core Concurrency

## Objectives
1. **High-Band Denoise & Adaptive De-Fizz**:
   - Suppress stationary background hiss/fizz in the reconstructed high frequencies ($>14\text{--}16\text{ kHz}$) during quiet or sparse sections.
   - Preserve transient attacks and harmonic definition when active events occur.
2. **Bounded Auto-EQ (Spectral Balance Correction)**:
   - Implement the Spectral Balance stage described in `docs/ARCHITECTURE.md`.
   - Measure running long-term spectral tilt against an acoustic reference slope (pink-noise / natural music slope: $-3.5\text{ to } -4.5\text{ dB/octave}$).
   - Apply bounded, smooth, wide-Q correction (strictly capped at $\pm 1.5\text{ dB}$) to remove mud/boxiness and harshness without altering artistic tone.
3. **4-Core Multithreading & Pipeline Concurrency**:
   - Comply with the Termux/S25 Ultra 4-core mobile limit (`AGENTS.md`).
   - Use `std::thread::scope` (zero external dependencies) to parallelize:
     - Feature tensor preparation across frequency bins and time frames.
     - Iterative state updates in `refine_mode`.
     - Channel processing (Mid/Side and Left/Right).

## Mathematical Formulation

### 1. High-Band Denoise (Spectral Floor Tracking & Soft Gating)
For STFT frames $t$ and bins $k$ corresponding to $f_k \ge 14\text{ kHz}$:
- Maintain a running minimum spectral noise floor estimate $\hat{N}(t, k)$ with asymmetric exponential smoothing:
  $$\hat{N}(t, k) = \begin{cases} \alpha_{down} \hat{N}(t-1, k) + (1-\alpha_{down}) |X(t, k)|, & |X(t, k)| \le \hat{N}(t-1, k) \\ \alpha_{up} \hat{N}(t-1, k) + (1-\alpha_{up}) |X(t, k)|, & |X(t, k)| > \hat{N}(t-1, k) \end{cases}$$
  where $\alpha_{up} \gg \alpha_{down}$ (slow rise, fast fall).
- Compute instantaneous a priori SNR: $\gamma(t, k) = \frac{|X(t, k)|^2}{\hat{N}(t, k)^2 + \epsilon}$.
- Soft suppression gain:
  $$G(t, k) = \left( \frac{\gamma(t, k)}{\gamma(t, k) + \beta} \right)^\nu$$
  clamped to a configurable floor $G_{min} \approx -6\text{ dB}$ ($0.5$).
- Below $14\text{ kHz}$, $G(t, k) \equiv 1.0$ (known content and lower harmonics are untouched).

### 2. Bounded Auto-EQ (Spectral Tilt & Resonance Balancing)
- Compute octave/sub-band average energy across 8 critical acoustic bands:
  - Sub ($20\text{--}60\text{ Hz}$), Low ($60\text{--}250\text{ Hz}$), Low-Mid ($250\text{--}600\text{ Hz}$), Mid ($600\text{--}2000\text{ Hz}$),
  - High-Mid ($2\text{--}4\text{ kHz}$), Presence ($4\text{--}8\text{ kHz}$), Brilliance ($8\text{--}14\text{ kHz}$), Air ($14\text{--}20\text{ kHz}$).
- Measure deviation $\Delta_b$ from natural acoustic balance reference.
- Compute smooth correction curve $C(f)$ using cubic spline / raised cosines, clamped to $[-1.5\text{ dB}, +1.5\text{ dB}]$.
- Apply via zero-phase linear filtering during finalization.

### 3. 4-Core Concurrency
- `scene_engine::CachedField` template preparation: divide `count` rows into 4 chunks processed in parallel with `std::thread::scope`.
- `rich_gate::inference_tiled`: divide feature matrix extraction across 4 threads before batched OpenCL predictor evaluation.
- Chunk synthesis: parallelize Mid and Side channel IFFT and WOLA accumulation.

## Verification & Parity
- Unit test for identity (zero damage / flat input yields unity gain).
- Bound test: verify Auto-EQ never exceeds $\pm 1.5\text{ dB}$.
- Parity test: verify 4-core multithreaded outputs exactly match single-threaded reference within float32 numerical precision.
