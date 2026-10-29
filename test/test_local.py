"""Check raw Baidu headings and the bundled local panorama example."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from run_local import run_local


class LocalTests(unittest.TestCase):
    def test_baidu_heading_wraps_at_north(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            metadata = root / 'metadata.csv'
            pd.DataFrame({'pid': ['0', '1', '2'], 'lat': 31.23, 'lng': 121.48,
                          'heading': [0, math.pi / 2, 2 * math.pi]}).to_csv(metadata, index=False)

            def calculate(cleaned, footprints, **kwargs):
                data = pd.read_csv(cleaned)
                np.testing.assert_allclose(data.computed_compass_angle, [270, 0, 270])
                pd.DataFrame(columns=['pid']).to_csv(kwargs['out_dir'], index=False)

            with patch('run.download_and_merge_buildings'), patch(
                    'openfacades.streetview.geo_aov_calculator', side_effect=calculate):
                run_local(images, metadata, root / 'output', source='baidu')

    def test_baidu_example_has_three_images_and_raw_headings(self):
        example = Path(__file__).parent / 'baidu'
        metadata = pd.read_csv(example / 'metadata.csv', dtype={'pid': str})
        self.assertEqual(metadata.columns.tolist(), ['pid', 'lat', 'lng', 'heading'])
        self.assertEqual(len(metadata), 3)
        self.assertEqual(metadata.pid.nunique(), 3)
        for pid in metadata.pid:
            with Image.open(example / 'images' / f'{pid}.jpg') as image:
                self.assertEqual(image.size, (2048, 1024))
                image.verify()

        def calculate(cleaned, footprints, **kwargs):
            data = pd.read_csv(cleaned, dtype={'id': str})
            self.assertEqual(data.id.tolist(), metadata.pid.tolist())
            np.testing.assert_allclose(data.computed_compass_angle, [258.3, 260, 257.52])
            pd.DataFrame(columns=['pid']).to_csv(kwargs['out_dir'], index=False)

        with tempfile.TemporaryDirectory() as output, patch(
                'run.download_and_merge_buildings') as download, patch(
                'openfacades.streetview.geo_aov_calculator', side_effect=calculate):
            run_local(example / 'images', example / 'metadata.csv', output,
                      example / 'footprint.geojson', source='baidu')
            download.assert_not_called()

    def test_unknown_source_fails_before_downloading(self):
        with patch('run.download_and_merge_buildings') as download:
            with self.assertRaises(ValueError):
                run_local('.', 'missing.csv', source='unknown')
            download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
