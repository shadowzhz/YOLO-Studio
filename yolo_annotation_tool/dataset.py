"""YOLO dataset validation and label writing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
import shutil
import tempfile
import yaml
from typing import Iterable, Sequence


class DatasetError(ValueError):
    """Raised when a dataset does not satisfy the YOLO contract."""


@dataclass(frozen=True)
class DatasetReport:
    images: int
    labels: int
    missing_labels: tuple[str, ...]
    malformed_labels: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_labels and not self.malformed_labels and self.images > 0


@dataclass(frozen=True)
class DatasetSplitReport:
    source_images: int
    train_images: int
    val_images: int
    skipped_unlabeled: tuple[str, ...]
    yaml_path: Path


def prepare_project_structure(root: str | Path) -> Path:
    """Create the standard project folders without touching existing files."""
    root = Path(root).expanduser()
    if root.exists() and not root.is_dir():
        raise DatasetError(f"项目路径不是文件夹：{root}")
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("images", "labels", "train/images", "train/labels", "val/images", "val/labels"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def _image_files(path: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def _source_images(root: Path) -> tuple[Path, list[Path]]:
    images_dir = root / "images"
    if images_dir.exists():
        files = _image_files(images_dir)
        if files:
            return images_dir, files
    excluded = {"images", "labels", "train", "val", "runs"}
    files = [p for p in _image_files(root) if p.relative_to(root).parts[0] not in excluded]
    return (root, files) if files else (images_dir, [])


def _valid_label_line(line: str, class_count: int | None) -> bool:
    fields = line.split()
    is_detection = len(fields) == 5
    is_polygon = len(fields) >= 7 and len(fields[1:]) % 2 == 0
    if not (is_detection or is_polygon):
        return False
    try:
        cls = int(fields[0])
        values = [float(v) for v in fields[1:]]
    except ValueError:
        return False
    if class_count is not None and not 0 <= cls < class_count:
        return False
    return all(0.0 <= value <= 1.0 for value in values)


def validate_dataset(root: str | Path, class_count: int | None = None) -> DatasetReport:
    root = Path(root)
    image_root, images = _source_images(root)
    labels_root = root / "labels"
    missing: list[str] = []
    malformed: list[str] = []
    labels = 0
    for image in images:
        label = labels_root / image.relative_to(image_root).with_suffix(".txt") if labels_root.exists() else image.with_suffix(".txt")
        if not label.exists():
            missing.append(str(image))
            continue
        labels += 1
        if any(not _valid_label_line(line, class_count) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()):
            malformed.append(str(label))
    return DatasetReport(len(images), labels, tuple(missing), tuple(malformed))


def write_detection_labels(path: str | Path, boxes: Iterable[Sequence[float]], classes: Iterable[int]) -> None:
    """Write normalized ``class x_center y_center width height`` rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls, box in zip(classes, boxes):
        if len(box) != 4:
            raise DatasetError("A detection box must contain four normalized values")
        values = [float(v) for v in box]
        if not all(0 <= value <= 1 for value in values):
            raise DatasetError("YOLO box values must be normalized to [0, 1]")
        rows.append(f"{int(cls)} " + " ".join(f"{value:.6f}" for value in values))
    from .annotations import Detection, save_annotations
    save_annotations(path, [Detection(int(row.split()[0]), *(float(v) for v in row.split()[1:])) for row in rows])

def remove_unused_class(root: Path, names: Sequence[str], index: int) -> None:
    """Remove an unused class and keep all project label IDs consistent."""
    changes: dict[Path, bytes] = {}
    for directory in (root / "labels", root / "train" / "labels", root / "val" / "labels"):
        for path in directory.rglob("*.txt"):
            text = path.read_text(encoding="utf-8")
            rows = []
            for line in text.splitlines(keepends=True):
                fields = line.split()
                if not fields:
                    rows.append(line)
                    continue
                if not _valid_label_line(line, len(names)):
                    raise DatasetError(f"标注格式有误，无法安全删除类别：{path}")
                class_id = int(fields[0])
                if class_id == index:
                    raise DatasetError(f"类别“{names[index]}”仍被标注使用：{path}")
                rows.append(re.sub(r"^(\s*)\S+", lambda match: match.group(1) + str(class_id - 1), line, count=1) if class_id > index else line)
            updated = "".join(rows)
            if updated != text:
                changes[path] = updated.encode("utf-8")

    remaining = list(names)
    del remaining[index]
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        text = yaml_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        expected = {i: name for i, name in enumerate(names)}
        if not isinstance(data, dict) or data.get("names") not in (expected, list(names)):
            raise DatasetError("data.yaml 类别与当前列表不一致")
        data.update(nc=len(remaining), names=dict(enumerate(remaining)))
        changes[yaml_path] = yaml.safe_dump(data, allow_unicode=True, sort_keys=False).encode("utf-8")
    changes[root / "classes.txt"] = ("\n".join(remaining) + "\n").encode("utf-8")

    backup = Path(tempfile.mkdtemp(prefix=".class-backup-", dir=root))
    (backup / "files.txt").write_text("".join(f"{number}.original: {path.relative_to(root)}\n" for number, path in enumerate(changes)), encoding="utf-8")
    replaced: list[tuple[Path, Path | None]] = []
    try:
        for number, (path, content) in enumerate(changes.items()):
            original = backup / f"{number}.original"
            if path.exists():
                shutil.copy2(path, original)
            else:
                original = None
            staged = backup / f"{number}.new"
            staged.write_bytes(content)
            staged.replace(path)
            replaced.append((path, original))
    except OSError as exc:
        try:
            for path, original in reversed(replaced):
                if original is None:
                    path.unlink()
                else:
                    shutil.copy2(original, path)
        except OSError as rollback_error:
            raise DatasetError(f"保存失败且恢复未完成，原文件备份位于 {backup}：{rollback_error}") from exc
        shutil.rmtree(backup, ignore_errors=True)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def make_dataset_yaml(root: str | Path, names: Sequence[str], output: str | Path | None = None) -> Path:
    """Create a minimal Ultralytics dataset YAML without requiring PyYAML."""
    root = prepare_project_structure(root).resolve()
    target = Path(output) if output else root / "data.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    data = dict(path=root.as_posix(), train="train/images", val="val/images", nc=len(names), names=dict(enumerate(names)))
    target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def split_annotated_dataset(
    root: str | Path,
    names: Sequence[str],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    stop_event=None,
) -> DatasetSplitReport:
    """Copy labeled images and labels into the standard train/val folders.

    Images without a label file are deliberately excluded so unfinished images
    are not silently treated as negative samples. Empty label files are kept as
    valid negative samples.
    """
    if not 0 <= val_fraction < 1:
        raise DatasetError("验证集比例必须大于等于 0 且小于 1")
    root = prepare_project_structure(root)
    image_root, images = _source_images(root)
    label_root = root / "labels"
    labeled: list[tuple[Path, Path, Path]] = []
    skipped: list[str] = []
    for image in images:
        if stop_event is not None and stop_event.is_set():
            raise DatasetError("数据集划分已取消")
        relative = image.relative_to(image_root)
        label = label_root / relative.with_suffix(".txt")
        if not label.exists():
            skipped.append(relative.as_posix())
            continue
        if any(not _valid_label_line(line, len(names)) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()):
            raise DatasetError(f"标注格式或类别有误：{label}")
        labeled.append((image, label, relative))
    if not labeled:
        raise DatasetError("没有找到已标注图片。请先完成至少一张图片的标注。")

    random.Random(seed).shuffle(labeled)
    if len(labeled) < 2 or val_fraction == 0:
        val_count = 0
    else:
        val_count = min(len(labeled) - 1, max(1, round(len(labeled) * val_fraction)))
    val_items = labeled[:val_count]
    train_items = labeled[val_count:]

    with tempfile.TemporaryDirectory(prefix=".split-", dir=root) as temp:
        staging = Path(temp)
        for split_name, items in (("train", train_items), ("val", val_items)):
            split_images = staging / split_name / "images"
            split_labels = staging / split_name / "labels"
            split_images.mkdir(parents=True, exist_ok=True)
            split_labels.mkdir(parents=True, exist_ok=True)
            for image, label, relative in items:
                if stop_event is not None and stop_event.is_set():
                    raise DatasetError("数据集划分已取消")
                destination_image = split_images / relative
                destination_label = split_labels / relative.with_suffix(".txt")
                destination_image.parent.mkdir(parents=True, exist_ok=True)
                destination_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, destination_image)
                shutil.copy2(label, destination_label)

        if stop_event is not None and stop_event.is_set():
            raise DatasetError("数据集划分已取消")
        moved = []
        try:
            for split_name in ("train", "val"):
                for kind in ("images", "labels"):
                    target = root / split_name / kind
                    backup = staging / (split_name + "_" + kind + "_backup")
                    target.rename(backup)
                    moved.append((target, backup))
                    (staging / split_name / kind).rename(target)
        except OSError:
            for target, backup in reversed(moved):
                if target.exists():
                    shutil.rmtree(target)
                backup.rename(target)
            raise

    yaml_path = make_dataset_yaml(root, names)
    return DatasetSplitReport(len(labeled), len(train_items), len(val_items), tuple(skipped), yaml_path)
