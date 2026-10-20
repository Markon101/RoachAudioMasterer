import json
from pathlib import Path
import tempfile
import unittest
import sys

import model_tools
from tools.registry import registry
sys.path.insert(0, str(Path(__file__).parent))
from hermes_job_guard import JobToolGuard, TEST_TIMEOUT_SECONDS, HALT_PATH, preflight_private_overlay


class InstalledRegistryGuardTests(unittest.TestCase):
    def setUp(self):
        private_tmp = Path(__file__).resolve().parents[2] / "runs/agent-migration-20261020"
        self.temp = tempfile.TemporaryDirectory(dir=str(private_tmp), prefix="guard-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "wt"
        (self.root / "scripts").mkdir(parents=True)
        self.read_a = self.root / "scripts" / "a.py"
        self.read_b = self.root / "scripts" / "b.py"
        self.write = self.root / "scripts" / "new.py"
        self.read_a.write_text("fixture alpha\n", encoding="utf-8")
        self.read_b.write_text("fixture beta\n", encoding="utf-8")
        self.halt = self.base / "HALT"
        self.assertIsNotNone(registry.get_entry("read_file"))
        self.assertIsNotNone(registry.get_entry("write_file"))
        self.assertIsNotNone(registry.get_entry("terminal"))

    def guard(self, delegate=None, *, read_paths=None, write_path=None):
        if delegate is None:
            delegate = registry.dispatch
        return JobToolGuard(
            root=self.root,
            read_paths=read_paths or (self.read_a, self.read_b),
            write_path=write_path or self.write,
            terminal_command="fixed offline command",
            terminal_workdir=self.root,
            halt_path=self.halt,
            delegate=delegate,
        )

    def test_actual_registry_reads_exact_allowlist_and_restores_hook(self):
        guard = self.guard()
        original_func = registry.dispatch.__func__
        with guard.install(registry):
            first = registry.dispatch("read_file", {"path": str(self.read_a), "offset": 1, "limit": 2000}, task_id="g1")
            second = registry.dispatch("read_file", {"path": str(self.read_b)}, task_id="g1")
        self.assertIn("fixture alpha", first)
        self.assertIn("fixture beta", second)
        self.assertEqual(registry.dispatch.__func__, original_func)
        self.assertFalse(self.halt.exists())
        self.assertEqual(guard._reads_done, {str(self.read_a), str(self.read_b)})

    def test_registry_write_hook_creates_only_the_exact_output_file(self):
        guard = self.guard()
        with guard.install(registry):
            registry.dispatch("read_file", {"path": str(self.read_a), "offset": 1, "limit": 2000}, task_id="g2")
            registry.dispatch("read_file", {"path": str(self.read_b), "offset": 1, "limit": 2000}, task_id="g2")
            result = registry.dispatch("write_file", {"path": str(self.write), "content": "safe fixture\n"}, task_id="g2")
        payload = json.loads(result)
        self.assertFalse(payload.get("error"))
        self.assertEqual(self.write.read_text(encoding="utf-8"), "safe fixture\n")
        self.assertFalse(self.halt.exists())

    def test_outside_write_is_denied_before_registry_and_latches_halt(self):
        outside = self.base / "outside.txt"
        outside.write_text("keep", encoding="utf-8")
        calls = []
        guard = self.guard(delegate=lambda *a, **kw: calls.append((a, kw)) or "delegated")
        result = guard("write_file", {"path": str(outside), "content": "overwrite"})
        self.assertEqual(json.loads(result)["error"], "job halted")
        self.assertEqual(outside.read_text(encoding="utf-8"), "keep")
        self.assertEqual(calls, [])
        self.assertTrue(self.halt.exists())
        self.assertEqual(self.halt.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(self.halt.read_text())["reason_code"], "write_outside_allowlist")

    def test_dotdot_path_is_denied_and_latches_halt(self):
        bad = str(self.root / "scripts" / ".." / "outside.txt")
        calls = []
        guard = self.guard(delegate=lambda *a, **kw: calls.append((a, kw)) or "delegated")
        result = guard("write_file", {"path": bad, "content": "no"})
        self.assertEqual(json.loads(result)["error"], "job halted")
        self.assertEqual(calls, [])
        self.assertTrue(self.halt.exists())

    def test_symlink_alias_is_denied_even_when_allowlisted_lexically(self):
        outside = self.base / "outside.txt"
        outside.write_text("keep", encoding="utf-8")
        alias = self.root / "scripts" / "alias.py"
        alias.symlink_to(outside)
        guard = self.guard(write_path=alias, delegate=lambda *a, **kw: self.fail("delegate ran"))
        result = guard("write_file", {"path": str(alias), "content": "no"})
        self.assertEqual(json.loads(result)["error"], "job halted")
        self.assertEqual(outside.read_text(encoding="utf-8"), "keep")
        self.assertTrue(self.halt.exists())

    def test_existing_partial_output_is_never_overwritten(self):
        self.write.write_text("preserve prior regression", encoding="utf-8")
        guard = self.guard(delegate=lambda *a, **kw: "delegated")
        guard("read_file", {"path": str(self.read_a)})
        guard("read_file", {"path": str(self.read_b)})
        result = guard("write_file", {"path": str(self.write), "content": "replacement"})
        self.assertEqual(json.loads(result)["error"], "job halted")
        self.assertEqual(self.write.read_text(encoding="utf-8"), "preserve prior regression")
        self.assertTrue(self.halt.exists())

    def test_existing_halt_blocks_allowlisted_read_without_rewriting_marker(self):
        marker = b"supervisor halt\n"
        self.halt.write_bytes(marker)
        calls = []
        guard = self.guard(delegate=lambda *a, **kw: calls.append((a, kw)) or "delegated")
        result = guard("read_file", {"path": str(self.read_a)})
        self.assertEqual(json.loads(result)["error"], "job halted")
        self.assertEqual(self.halt.read_bytes(), marker)
        self.assertEqual(calls, [])

    def test_local_latch_stops_followup_calls_if_halt_file_cannot_be_written(self):
        calls = []
        missing_parent_marker = self.base / "missing-parent" / "HALT"
        guard = JobToolGuard(
            root=self.root,
            read_paths=(self.read_a, self.read_b),
            write_path=self.write,
            terminal_command="fixed offline command",
            terminal_workdir=self.root,
            halt_path=missing_parent_marker,
            delegate=lambda *a, **kw: calls.append((a, kw)) or "delegated",
        )
        first = guard("write_file", {"path": str(self.base / "outside.txt"), "content": "no"})
        second = guard("read_file", {"path": str(self.read_a)})
        self.assertEqual(json.loads(first)["error"], "job halted")
        self.assertEqual(json.loads(second)["error"], "job halted")
        self.assertFalse(missing_parent_marker.exists())
        self.assertEqual(calls, [])

    def test_actual_latched_run_preflight_stops_before_config_or_client_setup(self):
        self.assertTrue(HALT_PATH.exists(), "the failed run's supervisor HALT must remain latched")
        with self.assertRaisesRegex(RuntimeError, "existing HALT marker"):
            preflight_private_overlay()

    def test_terminal_requires_exact_command_workdir_timeout_and_foreground(self):
        calls = []
        guard = self.guard(delegate=lambda *a, **kw: calls.append((a, kw)) or "terminal-result")
        guard("read_file", {"path": str(self.read_a)})
        guard("read_file", {"path": str(self.read_b)})
        guard("write_file", {"path": str(self.write), "content": "new file"})
        accepted = guard("terminal", {
            "command": "fixed offline command",
            "workdir": str(self.root),
            "timeout": TEST_TIMEOUT_SECONDS,
            "background": False,
            "pty": False,
        })
        self.assertEqual(accepted, "terminal-result")
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[-1][0][0], "terminal")
        self.assertFalse(self.halt.exists())

        other = self.base / "other"
        other.mkdir()
        second_halt = self.base / "HALT-2"
        second_calls = []
        second = JobToolGuard(
            root=self.root,
            read_paths=(self.read_a, self.read_b),
            write_path=self.write,
            terminal_command="fixed offline command",
            terminal_workdir=self.root,
            halt_path=second_halt,
            delegate=lambda *a, **kw: second_calls.append((a, kw)) or "delegated",
        )
        second("read_file", {"path": str(self.read_a)})
        second("read_file", {"path": str(self.read_b)})
        second("write_file", {"path": str(self.write), "content": "new file"})
        blocked = second("terminal", {
            "command": "fixed offline command",
            "workdir": str(other),
            "timeout": TEST_TIMEOUT_SECONDS,
            "background": False,
            "pty": False,
        })
        self.assertEqual(json.loads(blocked)["error"], "job halted")
        self.assertEqual(len(second_calls), 2)
        self.assertTrue(second_halt.exists())


if __name__ == "__main__":
    unittest.main()
