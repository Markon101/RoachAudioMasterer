# DeepSeek/Hermes migration — 2026-10-20

The two installed DeepSeek helpers and Hermes main/default now select
`deepseek/deepseek-v4.1-flash-1017` through the existing OpenRouter account.
The duplicate primary fallback was removed; Xiaomi remains the existing fallback.
Inactive MoA settings were preserved. Exact active paths, hashes, model checks,
costs and limitations are in [migration.json](../artifacts/agent-migration-20261020/migration.json).
Two bounded live helper smokes requested and returned the versioned model.
The catalog advertises 1,048,576 context tokens and tools/structured output;
this does not establish long-context competence or justify larger contexts.

Hermes generated two adversarial audit-checker tests through its actual tool
runner. The job exhausted its four-call budget without completing its summary.
A native worker and root separately reproduced the failures. The semantic checker
fix is recorded in `16afda5`; all seven focused tests and ten saved native traces
validate. Do not attribute the native rerun to Hermes.

The file tool wrote its new untracked test into the main checkout despite the
assigned worktree. That file was preserved, moved, reviewed, and later integrated.
This exposed a real containment gap. Hermes worker calls are now locally latched
at `runs/agent-migration-20261020/HALT`; no automatic clearing or resume is allowed.
DSP research continues independently.

The job-local [guard](../artifacts/agent-migration-20261020/hermes_job_guard.py)
allows two exact reads, one create-only exact write and one fixed foreground test
command/workdir. It checks HALT before dispatch, refuses traversal/symlinks and
existing output, and preserves evidence on violations. Ten independent offline
checks pass. It is a narrow tool-dispatch boundary, not an OS sandbox for generated
Python. Existing halted runs must remain intact; future guarded work needs an
explicitly approved fresh task/worktree/output and reviewed allowlists.

This Hermes build uses `HERMES_HOME/config.yaml` and `HERMES_HOME/.env`.
`HERMES_CONFIG_PATH` and `HERMES_ENV_PATH` are not honored overrides. Temporary
retry fields were restored after the failed override assumption; future overlays
must verify the loaded path and unchanged main config before making calls.

A worker exposed a legacy embedded credential constant during source inspection.
It differs from all current configured credentials. Embedded fallbacks were
removed; no key values belong in public artifacts. If that old key remains valid,
its credential owner must revoke it. Model-field rollback copies remain private
with mode 600 under the ignored migration run directory. Never restore an exposed
embedded fallback. No new credits/subscription were purchased; the observed account
usage delta was about $0.01344, below the $0.10 session test ceiling.
