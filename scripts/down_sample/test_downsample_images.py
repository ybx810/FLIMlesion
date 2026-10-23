"""Synthetic regression tests; never reads or writes the real data directory."""
import contextlib
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zlib
import struct

import cv2
import numpy as np
from PIL import Image, ImageCms

import downsample_images as d


class DownsamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='FLIM_中文 空格_')
        self.root = Path(self.temp.name)
        self.source = self.root / '原图 空格'
        self.source.mkdir()
        self.out = self.root / 'outputs'

    def tearDown(self):
        self.temp.cleanup()

    def run_batch(self, *extra):
        args = d.parser().parse_args(['--input-dir', str(self.source), '--output-dir', str(self.out), '--sizes', '32', '64', '256', *extra])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return d.run(args)

    def image(self, name, size=(129, 85), color=(240, 80, 15), **kwargs):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new('RGB', size, color) as im:
            im.save(path, **kwargs)
        return path

    def test_rounding_and_dimensions(self):
        for size, expected in [(1024, (1024, 683)), (2048, (2048, 1365)), (4096, (4096, 2731))]:
            self.assertEqual(d.target_dimensions(30000, 20000, size), expected)
            self.assertEqual(d.target_dimensions(20000, 30000, size), expected[::-1])
            self.assertEqual(d.target_dimensions(5000, 5000, size), (size, size))
        self.assertEqual(d.target_dimensions(10000, 1, 1024), (1024, 1))
        with self.assertRaises(ValueError):
            d.target_dimensions(20, 30, 40)

    def test_exact_algorithms_and_independence(self):
        rng = np.random.default_rng(42)
        for shape in [(257, 389, 3), (257, 257, 3), (1, 389, 3)]:
            src = rng.integers(0, 256, shape, dtype=np.uint8)
            baseline = src.copy()
            src.flags.writeable = False
            for target in [32, 64, 128]:
                dims = d.target_dimensions(shape[1], shape[0], target)
                for method, flag in [('area', cv2.INTER_AREA), ('lanczos', cv2.INTER_LANCZOS4)]:
                    arr, _ = d.resample(src, method, target)
                    np.testing.assert_array_equal(arr, cv2.resize(src, dims, interpolation=flag))
                expected = src
                levels = 0
                while (max(expected.shape[:2]) + 1) // 2 >= target:
                    expected = cv2.pyrDown(expected, borderType=cv2.BORDER_REFLECT_101)
                    levels += 1
                if expected.shape[1::-1] != dims:
                    expected = cv2.resize(expected, dims, interpolation=cv2.INTER_LINEAR)
                arr, meta = d.resample(src, 'gaussian_pyramid', target)
                np.testing.assert_array_equal(arr, expected)
                self.assertEqual(meta['pyramid_levels'], levels)
            np.testing.assert_array_equal(src, baseline)

    def test_batch_paths_color_collision_skip_overwrite_failure_and_integrity(self):
        self.image('类别 一/same.PNG')
        self.image('类别 一/same.tif', size=(128, 128))
        (self.source / 'broken.jpg').write_bytes(b'not an image')
        before = {p: hashlib.sha256(p.read_bytes()).digest() for p in self.source.rglob('*') if p.is_file()}
        summary = self.run_batch('--comparisons', '1', '--comparison-size', '32')
        self.assertEqual((summary['success'], summary['skipped'], summary['failed']), (12, 6, 9))
        paths = list((self.out / 'area/32').rglob('*.png'))
        self.assertEqual(len(paths), 2)
        for p in paths:
            with Image.open(p) as im:
                self.assertEqual(im.mode, 'RGB')
                self.assertEqual(im.getpixel((0, 0)), (240, 80, 15))
        times = {p: p.stat().st_mtime_ns for p in paths}
        summary = self.run_batch()
        self.assertEqual(summary['success'], 0)
        self.assertEqual(summary['skipped'], 18)
        self.assertEqual(times, {p: p.stat().st_mtime_ns for p in paths})
        self.assertEqual(self.run_batch('--overwrite')['success'], 12)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).digest() for p in before})
        with (self.out / 'processing_log.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 81)
        with Image.open(next((self.out / 'comparisons').rglob('*.png'))) as im:
            self.assertEqual(im.size, (96, 69))

    def test_exif_icc_and_rejections(self):
        exif = Image.Exif()
        exif[274] = 6
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        self.image('oriented.jpg', size=(100, 60), exif=exif, icc_profile=icc)
        with Image.new('RGBA', (100, 60)) as im:
            im.save(self.source / 'alpha.png')
        with Image.fromarray(np.zeros((60, 100), dtype=np.uint16)) as im:
            im.save(self.source / '16bit.tif')
        # Construct valid RGB16 PNG, which Pillow exposes as RGB8: must reject before load.
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        raw = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 100, 60, 16, 2, 0, 0, 0))
        raw += chunk(b'IDAT', zlib.compress((b'\0' + b'\0' * 600) * 60)) + chunk(b'IEND', b'')
        (self.source / 'rgb16.png').write_bytes(raw)
        self.assertEqual(self.run_batch()['failed'], 27)
        for p in (self.out / 'area/32').rglob('*.png'):
            with Image.open(p) as im:
                self.assertEqual(im.size, (19, 32))
                self.assertEqual(im.info['icc_profile'], icc)
                self.assertNotIn(274, im.getexif())

    def test_dry_run_budget_and_source_protection(self):
        self.image('sample.png')
        self.assertEqual(self.run_batch('--dry-run')['planned'], 6)
        self.assertFalse(self.out.exists())
        summary = self.run_batch('--memory-budget-gb', '0.001')
        self.assertEqual(summary['failed'], 6)
        self.assertEqual(summary['skipped'], 3)
        self.assertEqual(len(list(self.out.rglob('*.png'))), 0)
        with self.assertRaises(ValueError):
            self.run_batch('--output-dir', str(self.source / 'bad'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
