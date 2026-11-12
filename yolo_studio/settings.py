"""Small local settings store for the desktop application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import sys


def resolve_model_path(model: str | Path) -> str:
    """Resolve weight names or relative paths, checking ``models/`` when present."""
    text = str(model).strip()
    if not text:
        return text
    candidate = Path(text).expanduser()
    if candidate.exists() or candidate.is_absolute():
        return text
    roots: list[Path] = [Path.cwd()]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if exe_dir not in roots:
            roots.append(exe_dir)
    repo_root = Path(__file__).resolve().parent.parent
    if repo_root not in roots:
        roots.append(repo_root)
    for root in roots:
        for base in (root / "models", root):
            option = base / candidate
            if option.exists():
                return str(option)
    return text


def default_model_dialog_path(model: str | Path) -> str:
    """Return an existing model path or ``models/`` directory for file dialogs."""
    resolved = resolve_model_path(model)
    if resolved and Path(resolved).exists():
        return resolved
    roots: list[Path] = [Path.cwd()]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if exe_dir not in roots:
            roots.append(exe_dir)
    repo_root = Path(__file__).resolve().parent.parent
    if repo_root not in roots:
        roots.append(repo_root)
    for root in roots:
        models_dir = root / "models"
        if models_dir.is_dir():
            return str(models_dir)
    return str(model).strip()


@dataclass
class AppSettings:
    theme: str = "深色"
    show_guide_lines: bool = True
    zoom_step_percent: int = 25
    keep_zoom_on_image_switch: bool = False
    annotation_line_width: int = 2
    annotation_fill_enabled: bool = False
    previous_image_shortcut: str = "A"
    next_image_shortcut: str = "D"
    save_annotation_shortcut: str = "Ctrl+S"
    last_project: str = ""
    sam_model: str = "sam2.1_t.pt"
    training_model: str = "yolo11n.pt"
    training_data: str = "data.yaml"
    training_project: str = "runs/train"
    validation_model: str = "best.pt"
    validation_output_dir: str = "runs/validation"


class SettingsStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path.cwd() / "config" / "app_settings.json"

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return AppSettings()
        if not isinstance(raw, dict):
            return AppSettings()
        defaults = AppSettings()
        values = {}
        for field in fields(AppSettings):
            value = raw.get(field.name)
            default = getattr(defaults, field.name)
            if isinstance(default, bool) and isinstance(value, bool):
                values[field.name] = value
            elif isinstance(default, int) and not isinstance(default, bool) and isinstance(value, int) and not isinstance(value, bool):
                values[field.name] = value
            elif isinstance(default, str) and isinstance(value, str):
                values[field.name] = value
        return AppSettings(**values)

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
