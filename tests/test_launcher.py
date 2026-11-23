"""Launcher tests that run without Qt or machine-learning dependencies."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import venv

import main as launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix="yolo 启动 ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root / "main.py"
        self.script.touch()
        for target, value in (
            ("_ROOT", str(self.root)),
            ("sys.prefix", "/system"),
            ("sys.base_prefix", "/system"),
            ("sys.platform", "linux"),
            ("sys.argv", [str(self.script), "train", "--name", "有 空格"]),
        ):
            patcher = patch("main." + target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        frozen = patch.object(launcher.sys, "frozen", False, create=True)
        frozen.start()
        self.addCleanup(frozen.stop)
        execv = patch("main.os.execv")
        self.execv = execv.start()
        self.addCleanup(execv.stop)

    def make_environment(self, name=".venv", windows=False):
        environment = self.root / name
        executable = environment / ("Scripts/python.exe" if windows else "bin/python")
        executable.parent.mkdir(parents=True)
        executable.touch()
        (environment / "pyvenv.cfg").touch()
        return executable

    def test_linux_relaunch_preserves_arguments(self):
        executable = self.make_environment()
        launcher._relaunch_in_project_venv()
        self.execv.assert_called_once_with(
            str(executable), [str(executable), str(self.script), "train", "--name", "有 空格"]
        )

    def test_windows_uses_scripts_python(self):
        executable = self.make_environment(windows=True)
        with patch("main.sys.platform", "win32"):
            launcher._relaunch_in_project_venv()
        self.assertEqual(self.execv.call_args.args[0], str(executable))

    def test_dot_venv_takes_priority(self):
        executable = self.make_environment()
        self.make_environment("venv")
        launcher._relaunch_in_project_venv()
        self.assertEqual(self.execv.call_args.args[0], str(executable))

    def test_legacy_venv_is_supported(self):
        executable = self.make_environment("venv")
        launcher._relaunch_in_project_venv()
        self.assertEqual(self.execv.call_args.args[0], str(executable))

    def test_no_environment_keeps_current_interpreter(self):
        launcher._relaunch_in_project_venv()
        self.execv.assert_not_called()

    def test_incomplete_environment_is_skipped(self):
        executable = self.make_environment()
        executable.unlink()
        fallback = self.make_environment("venv")
        launcher._relaunch_in_project_venv()
        self.assertEqual(self.execv.call_args.args[0], str(fallback))

    def test_existing_virtual_environment_is_respected(self):
        self.make_environment()
        with patch("main.sys.prefix", str(self.root / ".venv")):
            launcher._relaunch_in_project_venv()
        self.execv.assert_not_called()

    def test_frozen_runtime_is_respected(self):
        self.make_environment()
        with patch("main.sys.frozen", True):
            launcher._relaunch_in_project_venv()
        self.execv.assert_not_called()


class LauncherSubprocessTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "Docker GUI launcher is Linux-only")
    def test_linux_gui_uses_docker_without_a_virtual_environment_or_terminal(self):
        with TemporaryDirectory(prefix="yolo 启动 ") as directory:
            root = Path(directory)
            shutil.copyfile(launcher.__file__, root / "main.py")
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copyfile(Path(launcher.__file__).parent / "scripts" / "run_gui.sh", scripts / "run_gui.sh")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for name, body in (
                ("docker", 'printf "%s\\n" "$@"'),
                ("xhost", "exit 0"),
            ):
                executable = bin_dir / name
                executable.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
                executable.chmod(0o755)

            result = subprocess.run(
                [sys._base_executable, str(root / "main.py")],
                cwd=root.parent,
                env={**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = result.stdout.splitlines()
            self.assertEqual(arguments[-4:], ["yolo-tool:latest", "python3", "-m", "yolo_studio.qt_app"])
            self.assertIn(f"{root}:/workspace", arguments)
            self.assertNotIn("-it", arguments)

    def test_real_environment_relaunch_without_activation(self):
        with TemporaryDirectory(prefix="yolo 启动 ") as directory:
            root = Path(directory)
            shutil.copyfile(launcher.__file__, root / "main.py")
            environment = root / ".venv"
            venv.EnvBuilder(with_pip=False).create(environment)
            package = root / "yolo_studio"
            package.mkdir()
            (package / "__init__.py").touch()
            (package / "__main__.py").write_text(
                "import json, sys\n"
                "def main():\n"
                "    print(json.dumps({'prefix': sys.prefix, 'args': sys.argv[1:]}))\n"
                "    sys.exit(7)\n",
                encoding="utf-8",
            )
            # Run from outside the project to verify paths are script-relative.
            result = subprocess.run(
                [sys.executable, str(root / "main.py"), "train", "--name", "有 空格"],
                cwd=root.parent,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 7, result.stderr)
            output = json.loads(result.stdout)
            self.assertEqual(Path(output["prefix"]), environment)
            self.assertEqual(output["args"], ["train", "--name", "有 空格"])


if __name__ == "__main__":
    unittest.main()
