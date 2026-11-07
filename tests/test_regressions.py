import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import yaml
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
from yolo_annotation_tool.qt_app import AnnotationPage, ValidationPage
from yolo_annotation_tool.annotations import Detection, save_annotations, load_annotations
from yolo_annotation_tool.dataset import prepare_project_structure, make_dataset_yaml, split_annotated_dataset, DatasetError, remove_unused_class
from yolo_annotation_tool.maixcam import collect_calibration_images

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
        for name in ('a.png', 'b.png'):
            image = QImage(32, 32, QImage.Format.Format_RGB32)
            image.fill(0)
            image.save(str(self.root / 'images' / name))
        self.page.images = sorted((self.root / 'images').glob('*.png'))
        self.page.apply_image_filter()
        self.page.update_stats()
    def test_draft_does_not_cross_image(self):
        self.page.canvas._polygon_draft = [(.1,.1),(.5,.1),(.5,.5)]
        self.page.select_image(1)
        self.assertEqual(self.page.canvas._polygon_draft, [])
    def test_negative_keeps_dirty_annotation(self):
        label = self.root/'labels/a.txt'
        label.write_text('')
        self.page.canvas._annotations = [Detection(0,.5,.5,.2,.2)]
        self.page.canvas.changed.emit()
        self.page.unset_negative()
        self.page.select_image(1)
        self.assertEqual(len(load_annotations(label)), 1)
    def test_class_order_rejected(self):
        self.page.class_names.setText('cat,dog'); self.page.save_classes()
        self.page.class_names.setText('dog,cat'); self.page.save_classes()
        self.assertEqual(self.page.current_class_names(), ['cat','dog'])
    def test_yaml_roundtrip_and_delete(self):
        names=['true','#hidden','a: b']
        path=make_dataset_yaml(self.root,names)
        self.assertEqual(yaml.safe_load(path.read_text())['names'],dict(enumerate(names)))
        remove_unused_class(self.root,names,1)
        self.assertEqual(yaml.safe_load(path.read_text())['names'],{0:'true',1:'a: b'})
    def test_invalid_labels_rejected(self):
        (self.root/'labels/a.txt').write_text('99 2 .5 .2 .2')
        with self.assertRaises(DatasetError): split_annotated_dataset(self.root,['cat'])
    def test_empty_labels_valid(self):
        (self.root/'labels/a.txt').write_text('')
        self.assertEqual(split_annotated_dataset(self.root,['cat']).train_images,1)
    def test_missing_label_cached(self):
        self.page._state_cache.clear()
        self.page._image_state(self.page.images[0])
        with patch.object(Path,'exists',side_effect=AssertionError('unexpected disk lookup')):
            self.assertEqual(self.page._image_state(self.page.images[0]),'unlabeled')
    def test_sam_stale_result_rejected(self):
        self.page._sam_request=2
        self.page._apply_sam_result((1,self.page.images[0],0,((0,0),(1,0),(1,1)),'segment'))
        self.assertEqual(self.page.canvas.annotations,[])
    def test_sam_busy_keeps_pending(self):
        self.page._sam_busy=True
        self.page._sam_point_clicked(.1,.1,1)
        self.assertTrue(self.page._sam_pending)
    def test_no_global_calibration_search(self):
        with patch.object(Path,'cwd',side_effect=AssertionError('global search')):
            self.assertEqual(collect_calibration_images(self.root/'missing',self.root/'calib'),[])
    def test_validation_preserves_input(self):
        from types import SimpleNamespace
        v=ValidationPage(); messages=[]; v.status_changed.connect(messages.append)
        image=self.page.images[0]; before=image.read_bytes()
        with patch.dict('sys.modules',{'ultralytics':SimpleNamespace(YOLO=lambda _:object())}):
            v._work('dummy',image,'image',.25,image.parent)
        self.assertEqual(image.read_bytes(),before)
        self.assertTrue(any('输出目录不能' in m for m in messages))

if __name__ == '__main__': unittest.main()
