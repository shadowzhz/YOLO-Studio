"""Shared cross-platform entry point for portable YOLO Studio distributions."""

from __future__ import annotations

from datetime import datetime
import io
import os
from pathlib import Path
import sys
import traceback


class _TeeTextIO(io.TextIOBase):
    """Write text to both the portable log file and an attached console stream."""

    def __init__(self, log_file, console_stream) -> None:
        self._log_file = log_file
        self._console_stream = console_stream

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._log_file.write(text)
        self._log_file.flush()
        if self._console_stream is not None:
            try:
                self._console_stream.write(text)
                self._console_stream.flush()
            except OSError:
                pass
        return len(text)

    def flush(self) -> None:
        self._log_file.flush()
        if self._console_stream is not None:
            try:
                self._console_stream.flush()
            except OSError:
                pass


def _is_writable_dir(path: Path) -> bool:
    """Verify actual write capability without relying on os.access."""
    probe = path / f".yolo-studio-write-probe-{os.getpid()}"
    try:
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _writable_data_dir(bundle_root: Path) -> Path:
    """Store state inside the extracted bundle when writable, else per-user data dir."""
    if _is_writable_dir(bundle_root):
        return bundle_root
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        target = base / "YOLO-Studio"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
        target = base / "yolo-studio"
    target.mkdir(parents=True, exist_ok=True)
    return target


def main(argv: list[str] | None = None) -> None:
    bundle_root = Path(__file__).resolve().parent.parent
    app_dir = str(bundle_root / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    data_root = _writable_data_dir(bundle_root)
    config_dir = data_root / "config"
    os.environ["YOLO_STUDIO_HOME"] = str(bundle_root)
    os.environ["YOLO_STUDIO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["MPLCONFIGDIR"] = str(config_dir / "matplotlib")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    fonts_dir = bundle_root / "runtime" / "fonts"
    if fonts_dir.is_dir():
        os.environ.setdefault("YOLO_STUDIO_FONTS_DIR", str(fonts_dir))

    args = sys.argv[1:] if argv is None else argv
    if args:
        from yolo_studio.__main__ import main as cli_main

        if argv is not None:
            saved_argv = sys.argv[:]
            sys.argv = [sys.argv[0], *args]
            try:
                cli_main()
            finally:
                sys.argv = saved_argv
        else:
            cli_main()
        return

    os.chdir(data_root)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    with (data_root / "launch.log").open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout = _TeeTextIO(log, orig_stdout)
        sys.stderr = _TeeTextIO(log, orig_stderr)
        print(f"\n[{datetime.now().isoformat(timespec='seconds')}] YOLO Studio ({sys.platform})")
        try:
            from yolo_studio.qt_app import main as qt_main

            qt_main()
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr


if __name__ == "__main__":
    main()
