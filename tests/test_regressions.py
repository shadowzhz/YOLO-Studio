import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from yolo_studio.__main__ import main as cli_main
from yolo_studio.annotations import Detection, load_annotations
from yolo_studio.dataset import (
    DatasetError,
    make_dataset_yaml,
    prepare_project_structure,
    remove_unused_class,
    split_annotated_dataset,
)
from yolo_studio.maixcam import (
    collect_calibration_images,
    detect_model_info,
    generate_maixpy_script,
    generate_mud_file,
    prune_and_clean_onnx,
)
from yolo_studio.qt_app import (
    AnnotationPage,
    MainWindow,
    TrainingPage,
    ValidationPage,
    _relaunch_with_cuda_runtime,
)
from yolo_studio.ui.main_window import _resolve_saved_path


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = prepare_project_structure(self.tmp.name)
        self.page = AnnotationPage()
        self.addCleanup(self.page._sam_timer.stop)
        self.page.project_root.setText(str(self.root))
        for name in ("a.png", "b.png"):
            image = QImage(32, 32, QImage.Format.Format_RGB32)
            image.fill(0)
            image.save(str(self.root / "images" / name))
        self.page.images = sorted((self.root / "images").glob("*.png"))
        self.page.apply_image_filter()
        self.page.update_stats()

    def test_draft_does_not_cross_image(self):
        self.page.canvas._polygon_draft = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)]
        self.page.select_image(1)
        self.assertEqual(self.page.canvas._polygon_draft, [])

    def test_negative_keeps_dirty_annotation(self):
        label = self.root / "labels/a.txt"
        label.write_text("")
        self.page.canvas._annotations = [Detection(0, 0.5, 0.5, 0.2, 0.2)]
        self.page.canvas.changed.emit()
        self.page.unset_negative()
        self.page.select_image(1)
        self.assertEqual(len(load_annotations(label)), 1)

    def test_class_order_rejected(self):
        self.page.class_names.setText("cat,dog")
        self.page.save_classes()
        self.page.class_names.setText("dog,cat")
        self.page.save_classes()
        self.assertEqual(self.page.current_class_names(), ["cat", "dog"])

    def test_save_classes_ignores_blank_lines(self):
        (self.root / "classes.txt").write_text("cat\n\ndog\n\n", encoding="utf-8")
        self.page.load_classes()
        self.page.class_names.setText("cat,dog,bird")
        self.page.save_classes()
        self.assertEqual(self.page.current_class_names(), ["cat", "dog", "bird"])

    def test_navigation_shortcut_scoped_to_canvas(self):
        self.assertIs(self.page._previous_shortcut.parent(), self.page.canvas)
        self.assertIs(self.page._next_shortcut.parent(), self.page.canvas)
        self.assertEqual(self.page._previous_shortcut.context(), Qt.ShortcutContext.WidgetShortcut)
        self.assertEqual(self.page._next_shortcut.context(), Qt.ShortcutContext.WidgetShortcut)

    def test_panning_in_segment_and_sam_modes(self):
        self.page.canvas.resize(400, 400)
        for mode in ("segment", "sam"):
            self.page.canvas.set_mode(mode)
            press = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(100, 100),
                QPointF(100, 100),
                Qt.MouseButton.MiddleButton,
                Qt.MouseButton.MiddleButton,
                Qt.KeyboardModifier.NoModifier,
            )
            move = QMouseEvent(
                QEvent.Type.MouseMove,
                QPointF(140, 125),
                QPointF(140, 125),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.MiddleButton,
                Qt.KeyboardModifier.NoModifier,
            )
            release = QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                QPointF(140, 125),
                QPointF(140, 125),
                Qt.MouseButton.MiddleButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.page.canvas.mousePressEvent(press)
            self.page.canvas.mouseMoveEvent(move)
            self.page.canvas.mouseReleaseEvent(release)
            self.assertEqual((self.page.canvas._pan_x, self.page.canvas._pan_y), (40.0, 25.0))
            self.assertEqual(self.page.canvas.sam_points, [])
            self.assertEqual(self.page.canvas._polygon_draft, [])
            self.page.canvas.reset_view()

    def test_undo_preserved_across_mode_switch(self):
        self.page.canvas._record_change()
        self.page.canvas._annotations.append(Detection(0, 0.5, 0.5, 0.2, 0.2))
        self.page.canvas.set_mode("segment")
        self.page.canvas._polygon_draft = [(0.1, 0.1), (0.2, 0.2)]
        self.page.canvas.undo()
        self.assertEqual(self.page.canvas.annotations, [])
        self.assertEqual(self.page.canvas._polygon_draft, [])

    def test_yaml_roundtrip_and_delete(self):
        names = ["true", "#hidden", "a: b"]
        path = make_dataset_yaml(self.root, names)
        self.assertEqual(yaml.safe_load(path.read_text())["names"], dict(enumerate(names)))
        remove_unused_class(self.root, names, 1)
        self.assertEqual(yaml.safe_load(path.read_text())["names"], {0: "true", 1: "a: b"})

    def test_invalid_labels_rejected(self):
        (self.root / "labels/a.txt").write_text("99 2 .5 .2 .2")
        with self.assertRaises(DatasetError):
            split_annotated_dataset(self.root, ["cat"])

    def test_empty_labels_valid(self):
        (self.root / "labels/a.txt").write_text("")
        self.assertEqual(split_annotated_dataset(self.root, ["cat"]).train_images, 1)

    def test_missing_label_cached(self):
        self.page._state_cache.clear()
        self.page._image_state(self.page.images[0])
        with patch.object(Path, "exists", side_effect=AssertionError("unexpected disk lookup")):
            self.assertEqual(self.page._image_state(self.page.images[0]), "unlabeled")

    def test_sam_stale_result_rejected(self):
        self.page._sam_request = 2
        self.page._apply_sam_result((1, self.page.images[0], 0, ((0, 0), (1, 0), (1, 1)), "segment"))
        self.assertEqual(self.page.canvas.annotations, [])

    def test_sam_busy_keeps_pending(self):
        self.page._sam_busy = True
        self.page._sam_point_clicked(0.1, 0.1, 1)
        self.assertTrue(self.page._sam_pending)

    def test_no_global_calibration_search(self):
        with patch.object(Path, "cwd", side_effect=AssertionError("global search")):
            self.assertEqual(collect_calibration_images(self.root / "missing", self.root / "calib"), [])

    def test_validation_preserves_input(self):
        v = ValidationPage()
        messages = []
        v.status_changed.connect(messages.append)
        image = self.page.images[0]
        before = image.read_bytes()
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=lambda _: object())}):
            v._work("dummy", image, "image", 0.25, image.parent)
        self.assertEqual(image.read_bytes(), before)
        self.assertTrue(any("输出目录不能" in m for m in messages))

    def test_maixcam_prune_and_artifacts(self):
        import onnx
        from onnx import TensorProto, helper

        x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 224, 320])
        dfl_out = helper.make_tensor_value_info("/model.23/dfl/conv/Conv_output_0", TensorProto.FLOAT, [1, 4, 10])
        sig_out = helper.make_tensor_value_info("/model.23/Sigmoid_output_0", TensorProto.FLOAT, [1, 1, 10])
        final_out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 10])
        w = helper.make_tensor("w", TensorProto.FLOAT, [4, 3, 1, 1], [0.1] * 12)
        n_dfl = helper.make_node("Conv", ["images", "w"], ["/model.23/dfl/conv/Conv_output_0"], name="/model.23/dfl/conv/Conv")
        n_sig = helper.make_node("Sigmoid", ["images"], ["/model.23/Sigmoid_output_0"], name="/model.23/Sigmoid")
        n_cat = helper.make_node(
            "Concat",
            ["/model.23/dfl/conv/Conv_output_0", "/model.23/Sigmoid_output_0"],
            ["output0"],
            name="/model.23/Concat",
            axis=1,
        )
        graph = helper.make_graph([n_dfl, n_sig, n_cat], "toy", [x], [final_out], initializer=[w], value_info=[dfl_out, sig_out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        raw_onnx = self.root / "raw.onnx"
        clean_onnx = self.root / "clean.onnx"
        onnx.save(model, str(raw_onnx))

        outputs = prune_and_clean_onnx(raw_onnx, clean_onnx, "yolo11", 23)
        self.assertEqual(outputs, ["/model.23/dfl/conv/Conv_output_0", "/model.23/Sigmoid_output_0"])
        self.assertTrue(clean_onnx.is_file())

        mud_path = self.root / "test.mud"
        generate_mud_file(mud_path, "test_int8.cvimodel", "yolo11", ["apple", "pear"])
        mud_text = mud_path.read_text(encoding="utf-8")
        self.assertIn("model = test_int8.cvimodel", mud_text)
        self.assertIn("labels = apple, pear", mud_text)

        script_path = self.root / "main.py"
        generate_maixpy_script(script_path, "test.mud", "yolo11")
        self.assertIn("nn.YOLO11", script_path.read_text(encoding="utf-8"))

    def test_maixcam_missing_dfl_head_raises(self):
        import onnx
        from onnx import TensorProto, helper

        x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])
        y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3, 32, 32])
        n = helper.make_node("Relu", ["images"], ["output0"], name="relu")
        graph = helper.make_graph([n], "no_dfl", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        raw_onnx = self.root / "nodfl.onnx"
        onnx.save(model, str(raw_onnx))
        with self.assertRaises(RuntimeError):
            prune_and_clean_onnx(raw_onnx, self.root / "clean.onnx", "yolo11", 23)

    def test_maixcam_rejects_non_detect_task(self):
        weight = self.root / "seg.pt"
        weight.write_bytes(b"stub")
        fake_yolo = SimpleNamespace(task="segment", names={0: "obj"}, model=SimpleNamespace(model=[]))
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=lambda _: fake_yolo)}):
            with self.assertRaises(ValueError):
                detect_model_info(weight)

    def test_cli_export_supports_maixcam(self):
        captured = []
        with patch("yolo_studio.__main__.export_model", side_effect=lambda cfg: captured.append(cfg) or "out"):
            with patch.object(
                sys,
                "argv",
                ["yolo_studio", "export", "--model", "best.pt", "--format", "maixcam", "--imgsz", "224x320", "--quantize", "BF16"],
            ):
                cli_main()
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].format, "maixcam")
        self.assertEqual(captured[0].imgsz, "224x320")
        self.assertEqual(captured[0].quantize, "BF16")

        import io
        from contextlib import redirect_stderr

        with patch.object(sys, "argv", ["yolo_studio", "export", "--model", "best.pt", "--format", "onnx", "--imgsz", "224x320"]):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli_main()

    def test_training_page_rejects_invalid_params(self):
        page = TrainingPage()
        page.epochs.setText("0")
        with patch.object(QMessageBox, "warning") as warn:
            page.run()
        warn.assert_called_once()

    def test_workspace_path_bridge_and_no_hardcoded_relaunch(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("YOLO_STUDIO_PYTHON", None)
            with patch("os.execv", side_effect=AssertionError("unexpected execv")):
                _relaunch_with_cuda_runtime()

        with patch("pathlib.Path.cwd", return_value=self.root):
            resolved = _resolve_saved_path("/workspace/images", "")
            self.assertEqual(Path(resolved), self.root / "images")
            self.assertEqual(_resolve_saved_path("/workspace/nonexistent_dir", "fallback"), "fallback")

    def test_main_window_instantiates_and_shares_config_dir(self):
        with patch("pathlib.Path.cwd", return_value=self.root):
            window = MainWindow()
            try:
                self.assertEqual(window.about._config_dir, window.settings_store.path.parent)
            finally:
                window.annotation._sam_timer.stop()
                window.close()


if __name__ == "__main__":
    unittest.main()
