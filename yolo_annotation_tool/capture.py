"""Camera capture primitives used by the annotation workflow."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4


class CameraCapture:
    """Read camera frames in a daemon thread without blocking the UI."""

    def __init__(self, source: int) -> None:
        self.source = source
        self._stop = Event()
        self._lock = Lock()
        self._frame = None
        self._status = "正在连接摄像头…"
        self._thread = Thread(target=self._run, daemon=True, name="camera-capture")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self):
        with self._lock:
            return (self._frame.copy() if self._frame is not None else None, self._status)

    def _run(self) -> None:
        cap = None
        try:
            import cv2
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise RuntimeError("无法打开摄像头，请检查设备编号、权限或是否被其他程序占用。")
            failures = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 30:
                        raise RuntimeError("无法读取摄像头画面，请检查设备连接。")
                    self._stop.wait(0.05)
                    continue
                failures = 0
                with self._lock:
                    self._frame = frame
                    self._status = "摄像头已连接"
        except Exception as exc:
            with self._lock:
                self._frame = None
                self._status = f"采集失败：{exc}"
        finally:
            if cap is not None:
                cap.release()


def save_frame(frame, directory: Path) -> Path:
    """Save a JPEG without overwriting existing images, including on Windows."""
    import cv2
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise OSError("图片编码失败")
    directory.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        name = f"capture_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}.jpg"
        path = directory / name
        try:
            with path.open("xb") as output:
                output.write(encoded.tobytes())
            return path
        except FileExistsError:
            continue
    raise OSError("无法生成唯一的图片文件名")
