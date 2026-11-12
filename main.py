"""Convenience root launcher for YOLO Studio."""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


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
    main()
