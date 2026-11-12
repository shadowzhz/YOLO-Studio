"""Modular PySide6 UI components for YOLO Studio."""

from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

__all__ = ["ASSETS_DIR", "IMAGE_SUFFIXES"]
