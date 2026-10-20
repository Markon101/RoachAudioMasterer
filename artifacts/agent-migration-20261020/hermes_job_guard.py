"""One-job allowlist around the installed Hermes registry; no global Hermes changes."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Callable, Iterable

TARGET_MODEL = "deepseek/deepseek-v4.1-flash-1017"
WORKTREE = Path("/data/data/com.termux/files/usr/tmp/roach-hermes-stage-adversary-20261020")
RUN_DIR = Path("/data/data/com.termux/files/home/projects/RoachAudioMasterer/runs/agent-migration-20261020")
HALT_PATH = RUN_DIR / "HALT"
READ_PATHS = (
    WORKTREE / "scripts/analyze_master_stage_audit.py",
    WORKTREE / "scripts/test_analyze_master_stage_audit.py",
)
WRITE_PATH = WORKTREE / "scripts/test_master_audit_adversarial.py"
TEST_COMMAND = (
    "LD_PRELOAD=/data/data/com.termux/files/usr/lib/libc++_shared.so "
    "PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=scripts "
    "python -m unittest scripts.test_master_audit_adversarial"
)
TEST_TIMEOUT_SECONDS = 120
MAX_WRITE_CHARS = 16_384
TASK_PROMPT = f"""Perform only this isolated adversarial analyzer task. Read these exact full files:
{READ_PATHS[0]}
{READ_PATHS[1]}
Create only {WRITE_PATH}. Use synthetic fixtures from `from test_analyze_master_stage_audit import make_audit, analysis` and temporary JSONL files only. Write exactly two unittest cases that assert rejection for (1) changing only final_output.meta.final_trim_gain_linear from 1 to 0.5 without changing audio, and (2) with dynamic=True changing both paired limiter.kind fields to `unknown_backend`. Preserve any failing regression and counterexample. Then run exactly `{TEST_COMMAND}` as a foreground terminal call with workdir `{WORKTREE}` and timeout {TEST_TIMEOUT_SECONDS}. No other files, tools, commands, networks, audio, Cargo, commits, or model workloads."""


class JobToolGuard:
    """Allow two exact reads, one exact write, and one exact foreground test command."""

    def __init__(
        self, *, root: Path, read_paths: Iterable[Path], write_path: Path,
        terminal_command: str, terminal_workdir: Path, halt_path: Path,
        delegate: Callable,
    ):
        self.root = Path(root).resolve(strict=True)
        self.read_paths = {self._absolute(p) for p in read_paths}
        self.write_path = self._absolute(write_path)
        self.terminal_command = terminal_command
        self.terminal_workdir = self._absolute(terminal_workdir)
        self.halt_path = Path(halt_path).absolute()
        self.delegate = delegate
        self._reads_done: set[str] = set()
        self._write_done = False
        self._terminal_done = False
        self._local_halted = False
        self._lock = threading.RLock()
        for allowed in (*self.read_paths, self.write_path, self.terminal_workdir):
            if not self._is_under_root(allowed):
                raise ValueError("allowlist path escapes the fixed worktree")

    @staticmethod
    def _absolute(path: Path) -> str:
        p = Path(path)
        if not p.is_absolute() or ".." in p.parts:
            raise ValueError("allowlist paths must be absolute and traversal-free")
        return str(p)

    def _is_under_root(self, raw: str) -> bool:
        try:
            return Path(raw).is_relative_to(self.root)
        except (OSError, ValueError):
            return False

    def preflight(self) -> bool:
        """A latched marker is authoritative and is never removed or rewritten."""
        return not self._local_halted and not os.path.lexists(self.halt_path)

    def _persist_halt(self, code: str, tool: str) -> bool:
        payload = json.dumps({
            "schema": "hermes-job-halt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reason_code": code,
            "tool": tool,
        }, sort_keys=True).encode("utf-8") + b"\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.halt_path, flags, 0o600)
        except FileExistsError:
            return True
        except OSError:
            return False
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            return True
        except OSError:
            return False

    def _halted(self, tool: str) -> str:
        return json.dumps({"error": "job halted", "tool": tool}, sort_keys=True)

    def _reject(self, tool: str, code: str) -> str:
        self._local_halted = True
        self._persist_halt(code, tool)
        return self._halted(tool)

    def _create_allowlisted_file(self, content: str) -> str:
        """Create the one authorized /usr/tmp output without Hermes' /usr denylist bypass."""
        relative = Path(self.write_path).relative_to(self.root)
        parts = relative.parts
        if not parts or any(part in ("", ".", "..") for part in parts):
            raise OSError("invalid guarded output path")
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise OSError("safe no-follow file creation is unavailable")
        dir_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        file_fd = None
        try:
            for component in parts[:-1]:
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
                os.close(dir_fd)
                dir_fd = next_fd
            file_fd = os.open(
                parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=dir_fd,
            )
            with os.fdopen(file_fd, "w", encoding="utf-8", newline="") as stream:
                file_fd = None
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(dir_fd)
        finally:
            if file_fd is not None:
                os.close(file_fd)
            os.close(dir_fd)
        return json.dumps({
            "success": True,
            "path": self.write_path,
            "resolved_path": self.write_path,
            "files_modified": [self.write_path],
        }, sort_keys=True)

    def _path_is_exact_and_safe(
        self, raw: object, expected: str, *, must_exist: bool, must_be_dir: bool = False,
    ) -> bool:
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            return False
        p = Path(raw)
        if not p.is_absolute() or ".." in p.parts or raw != str(p) or raw != expected:
            return False
        if not self._is_under_root(expected):
            return False
        try:
            resolved = p.resolve(strict=must_exist)
            if str(resolved) != expected:
                return False
            if must_exist and must_be_dir and not p.is_dir():
                return False
            if must_exist and not must_be_dir and not p.is_file():
                return False
            return True
        except (OSError, RuntimeError, ValueError):
            return False

    def __call__(self, name: str, args: dict, **kwargs):
        with self._lock:
            if not self.preflight():
                return self._halted(str(name))
            if not isinstance(args, dict):
                return self._reject(str(name), "arguments_not_mapping")

            if name == "read_file":
                raw = args.get("path")
                if not any(raw == p for p in self.read_paths):
                    return self._reject(name, "read_outside_allowlist")
                if not self._path_is_exact_and_safe(raw, raw, must_exist=True):
                    return self._reject(name, "read_path_resolution_mismatch")
                if args.get("offset", 1) != 1 or args.get("limit", 2000) != 2000:
                    return self._reject(name, "read_must_be_complete")
                if raw in self._reads_done:
                    return self._reject(name, "duplicate_read")
                self._reads_done.add(raw)
                return self.delegate(name, args, **kwargs)

            if name == "write_file":
                raw = args.get("path")
                if raw != self.write_path:
                    return self._reject(name, "write_outside_allowlist")
                if not self._path_is_exact_and_safe(raw, self.write_path, must_exist=False):
                    return self._reject(name, "write_path_resolution_mismatch")
                content = args.get("content")
                if not isinstance(content, str) or len(content) > MAX_WRITE_CHARS:
                    return self._reject(name, "write_content_out_of_bounds")
                if self._write_done:
                    return self._reject(name, "duplicate_write")
                if self._reads_done != self.read_paths:
                    return self._reject(name, "write_before_complete_reads")
                if os.path.lexists(self.write_path):
                    return self._reject(name, "write_target_preexisting")
                self._write_done = True
                try:
                    return self._create_allowlisted_file(content)
                except OSError:
                    return self._reject(name, "write_create_failed")

            if name == "terminal":
                allowed_keys = {"command", "workdir", "timeout", "background", "pty"}
                if set(args) - allowed_keys:
                    return self._reject(name, "terminal_arguments_out_of_bounds")
                if args.get("command") != self.terminal_command:
                    return self._reject(name, "terminal_command_mismatch")
                workdir = args.get("workdir")
                if not self._path_is_exact_and_safe(
                    workdir, self.terminal_workdir, must_exist=True, must_be_dir=True,
                ):
                    return self._reject(name, "terminal_workdir_mismatch")
                if args.get("timeout") != TEST_TIMEOUT_SECONDS:
                    return self._reject(name, "terminal_timeout_mismatch")
                if args.get("background", False) is not False or args.get("pty", False) is not False:
                    return self._reject(name, "terminal_must_be_foreground")
                if self._terminal_done:
                    return self._reject(name, "duplicate_terminal")
                if self._reads_done != self.read_paths or not self._write_done:
                    return self._reject(name, "terminal_before_reads_or_write")
                self._terminal_done = True
                return self.delegate(name, args, **kwargs)

            return self._reject(str(name), "tool_not_allowlisted")

    @contextmanager
    def install(self, registry):
        """Patch only this process's installed registry instance for the guarded job."""
        original = registry.dispatch
        registry.dispatch = self
        try:
            yield
        finally:
            registry.dispatch = original


def build_job_guard(registry):
    delegate = registry.dispatch
    return JobToolGuard(
        root=WORKTREE, read_paths=READ_PATHS, write_path=WRITE_PATH,
        terminal_command=TEST_COMMAND, terminal_workdir=WORKTREE,
        halt_path=HALT_PATH, delegate=delegate,
    )


def preflight_private_overlay() -> tuple[dict, str]:
    """Validate the actual private Hermes overlay before creating a model client."""
    if os.path.lexists(HALT_PATH):
        raise RuntimeError("existing HALT marker: this run remains stopped")
    from hermes_cli.config import get_config_path, load_config
    path = str(get_config_path())
    config = load_config()
    if not os.path.samefile(path, os.path.join(os.environ.get("HERMES_HOME", ""), "config.yaml")):
        raise RuntimeError("Hermes loaded an unexpected config path")
    model = config.get("model", {})
    if model.get("default") != TARGET_MODEL or model.get("provider") != "openrouter":
        raise RuntimeError("Hermes overlay model/provider do not match the approved route")
    if int(config.get("agent", {}).get("api_max_retries", 3)) != 0:
        raise RuntimeError("Hermes overlay is not configured for a single attempt")
    if int(config.get("agent", {}).get("auto_recovery_cycles", 5)) != 0:
        raise RuntimeError("Hermes overlay recovery is enabled")
    if config.get("fallback_providers"):
        raise RuntimeError("Hermes overlay has an external fallback route")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("existing route credential is unavailable")
    return config, path


def run_guarded_worker() -> dict:
    """One bounded AIAgent run; deliberately refuses the old latched run directory."""
    _, config_path = preflight_private_overlay()
    if os.path.lexists(WRITE_PATH):
        raise RuntimeError("allowlisted output already exists; preserving it and refusing overwrite")
    import model_tools
    from tools.registry import registry
    from run_agent import AIAgent

    for source in READ_PATHS:
        if source.is_symlink() or not source.is_file() or len(source.read_text(encoding="utf-8").splitlines()) > 2000:
            raise RuntimeError("an allowlisted source is missing, linked, or exceeds full-read bounds")
    if not WORKTREE.is_dir() or WORKTREE.is_symlink():
        raise RuntimeError("fixed worktree is unavailable or linked")

    agent = AIAgent(
        model=TARGET_MODEL, provider="openrouter", requested_provider="openrouter",
        max_iterations=4, max_tokens=500, run_budget_seconds=180,
        enabled_toolsets=["file", "terminal"], provider_require_parameters=True,
        quiet_mode=True, skip_context_files=True, load_soul_identity=False,
        skip_memory=True, skip_background_review=True, save_trajectories=False,
        cwd=str(WORKTREE),
    )
    if getattr(agent, "_api_max_retries", None) != 1 or getattr(agent, "_auto_recovery_cycles", None) != 0:
        agent.close()
        raise RuntimeError("AIAgent did not load single-attempt retry bounds")

    guard = build_job_guard(registry)
    try:
        with guard.install(registry):
            result = agent.run_conversation(TASK_PROMPT)
    finally:
        agent.close()
    return {
        "requested_model": TARGET_MODEL,
        "agent_model_after": getattr(agent, "model", None),
        "agent_provider_after": getattr(agent, "provider", None),
        "served_model": getattr(agent, "last_served_model", None),
        "api_calls": result.get("api_calls") if isinstance(result, dict) else None,
        "completed": result.get("completed") if isinstance(result, dict) else None,
        "failed": result.get("failed") if isinstance(result, dict) else None,
        "tool_counts": {"reads": len(guard._reads_done), "writes": int(guard._write_done), "terminal": int(guard._terminal_done)},
        "effective_config_path": config_path,
        "halted": os.path.lexists(HALT_PATH),
        "input_tokens": getattr(agent, "session_input_tokens", None),
        "output_tokens": getattr(agent, "session_output_tokens", None),
        "estimated_cost_usd": getattr(agent, "session_estimated_cost_usd", None),
    }
