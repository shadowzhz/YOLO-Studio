"""Convenience root launcher for YOLO Studio."""

from __future__ import annotations

import os
from pathlib import Path
import sys

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _relaunch_in_project_venv() -> None:
    """Use a local environment without requiring shell activation.

    Respect explicitly selected virtual environments and packaged runtimes.
    Check sys.prefix, not the executable's resolved path: POSIX virtual
    environments often symlink their Python executable to the system Python.
    """
    if getattr(sys, "frozen", False) or sys.prefix != sys.base_prefix:
        return

    root = Path(_ROOT)
    for name in (".venv", "venv"):
        environment = root / name
        if not (environment / "pyvenv.cfg").is_file():
            continue
        executable = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if executable.is_file():
            os.execv(str(executable), [str(executable), str(root / "main.py"), *sys.argv[1:]])
            return


def _launch_docker_gui() -> None:
    """Open the Linux desktop through Docker when started with system Python."""
    if sys.argv[1:] or sys.platform != "linux" or getattr(sys, "frozen", False):
        return
    if sys.prefix != sys.base_prefix:
        return

    script = Path(_ROOT) / "scripts" / "run_gui.sh"
    if script.is_file():
        os.execvp("bash", ["bash", str(script)])


def main(argv: list[str] | None = None) -> None:
    """Launch the PySide6 desktop UI with no arguments, or delegate CLI commands when given."""
    args = sys.argv[1:] if argv is None else argv
    if args:
        from yolo_studio.__main__ import main as cli_main

        if argv is not None:
            saved = sys.argv[:]
            sys.argv = [sys.argv[0], *args]
            try:
                cli_main()
            finally:
                sys.argv = saved
        else:
            cli_main()
        return

    from yolo_studio.qt_app import main as qt_main

    qt_main()


if __name__ == "__main__":
    _relaunch_in_project_venv()
    _launch_docker_gui()
    main()
