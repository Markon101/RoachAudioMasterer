# Pipeline and command catalog

This is a concise map of operational paths, not a claim that each is a
production-ready quality improvement. Use `--help` for the complete controls.
The production champion is the explicitly configured `auto-master` invocation
in the project README; opting into research stages changes that configuration.

| CLI | Actual scope and contract |
| --- | --- |
| `auto-master` | Read/transcode, assess, optional SFHT and mid specialists, clean/EQ and microstructure, spatial and optional morphic/dynamics, final loudness/peak control, audit/export. Routing depends on assessment and explicit options. Low/mid checkpoint defaults may differ from the README champion. Summary records applied settings; later stages can alter earlier protected bands. |
| `auto-master --high-model FIELD` | Opt-in learned high-field residual after mid and before clean. No manufactured lowpass; fresh directory; <=48 s; bypass at `--high-strength 0`; parent checkpoints checked. See [high-field note](OPT_IN_HIGH_FIELD_PIPELINE.md). |
| `auto-master --smooth-auto-eq` | Opt-in continuous interpolation for auto-EQ; legacy interpolation stays default. Has no effect when auto-EQ is disabled. |
| `rich-triband-restore` | Historical command name: low/SFHT and mid restoration with optional polish and spatial processing. It does not call the high field or the complete mastering finalizer. |
| `rich-field-restore` / `rich-field-restore-song` | Standalone frozen high-field research. Controlled mode deliberately low-passes a supplied reference for evaluation; undegraded mode adds a speculative residual to observed audio. The historical song command may reuse chunk caches with `--resume`; cache hits do not authenticate input or parameters. |
| `rich-mid-restore` | Standalone mid flow; coherence is a separate experimental option. Per-chunk projection differs from a final whole-track lock and downstream DSP may alter either result. |
| `scene-restore` / `flow-restore` / `restore` | Earlier research paths for synthetic/known-band completion with explicit cutoff and residual controls; their distinct model formats and sample-rate assumptions apply. |
| `microstructure` / `master` | Standalone processing commands; `master` can also apply optional spatial processing and finalizes loudness and peak level. Clean and spatial DSP are available through pipeline options, not separate commands. None establishes restoration quality. |
| `dynamics-restore` | Explicit bounded dynamics model experiment; not enabled in the champion unless requested. |
| `rich-mid-train` / `rich-mid-evaluate` | Procedural research. Optional `--damage-seed` selects independently seeded target/damage experiments; independent training is fresh and cannot resume a legacy state. This flag does not retrain any existing checkpoint by itself. |
| `dynamics-train --corrected-objective` | Opt-in corrected objective for fresh experiments; legacy objective and its checkpoints remain separate. |

Models synthesize or reshape plausible residuals, not historically authenticated
missing details. Normal stage outputs and numeric checks do not prove audible
improvement. Select fresh outputs and retain identity, DSP and matched-level
listening controls. Full-track resource use and backend placement require their
own qualification; no speed claim follows from this catalog.
