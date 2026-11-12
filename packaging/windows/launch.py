"""Entry point for the portable Windows distribution."""

from datetime import datetime
import os
from pathlib import Path
import sys
import traceback


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    os.environ["YOLO_CONFIG_DIR"] = str(root / "config")
    os.environ["MPLCONFIGDIR"] = str(root / "config" / "matplotlib")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    with (root / "launch.log").open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout = sys.stderr = log
        print(f"\n[{datetime.now().isoformat(timespec='seconds')}] YOLO Studio")
        try:
            from yolo_studio.qt_app import main

            main()
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
