import tempfile
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

import run_local
from openfacades.footprint.downloader import BuildingDataDownloader
from openfacades.streetview.aovcalculator import process_point
from openfacades.detection.pano2pers import _find_image_file, save_bounding_boxes_as_images


class GSVTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        self.metadata = self.root / 'metadata.csv'
        pd.DataFrame([{'pid': '001_GSV', 'lat': 48.148, 'lng': 11.568, 'heading': math.radians(112)}]).to_csv(self.metadata, index=False)

    def test_auto_footprints_and_100_metre_boundary(self):
        downloaded = []
        def download(bounds, output, overwrite=False):
            downloaded.append((bounds, overwrite))
        def calculate(metadata, footprints, **kwargs):
            clean = pd.read_csv(metadata, dtype={'id': str})
            self.assertEqual(clean.id.iloc[0], '001_GSV')
            self.assertAlmostEqual(clean.computed_compass_angle.iloc[0], 112)
            pd.DataFrame(columns=['pid']).to_csv(kwargs['out_dir'], index=False)
        with patch('run.download_and_merge_buildings', side_effect=download), patch('openfacades.streetview.geo_aov_calculator', side_effect=calculate):
            run_local.run_local(self.images, self.metadata, self.root / 'output')
        self.assertEqual(len(downloaded), 1)
        self.assertTrue(downloaded[0][1])
        points = gpd.GeoSeries([Point(11.568, 48.148)], crs='EPSG:4326')
        boundary = gpd.read_file(self.root / 'output/01_data/bbox.geojson').to_crs(points.estimate_utm_crs())
        np.testing.assert_allclose(boundary.total_bounds[2:] - boundary.total_bounds[:2], [200, 200], atol=.01)

    def test_heading_only_is_converted_to_degrees(self):
        pd.DataFrame([{'pid': str(i), 'lat': 48.148, 'lng': 11.568,
                       'heading': heading, 'degree': -999}
                      for i, heading in enumerate([0, math.pi / 2, math.pi, 2 * math.pi])]).to_csv(self.metadata, index=False)
        def calculate(metadata, footprints, **kwargs):
            clean = pd.read_csv(metadata)
            np.testing.assert_allclose(clean.computed_compass_angle, [0, 90, 180, 360])
            pd.DataFrame(columns=['pid']).to_csv(kwargs['out_dir'], index=False)
        with patch('run.download_and_merge_buildings'), patch('openfacades.streetview.geo_aov_calculator', side_effect=calculate):
            run_local.run_local(self.images, self.metadata, self.root / 'heading_output')

    def test_local_footprints_skip_download(self):
        footprint = self.root / 'local.geojson'
        gpd.GeoDataFrame({'osmid': [123], 'geometry': [box(11.568, 48.148, 11.569, 48.149)]}, crs='EPSG:4326').to_file(footprint, driver='GeoJSON')
        def calculate(metadata, buildings, **kwargs):
            loaded = gpd.read_file(buildings)
            self.assertIn('building_id', loaded.columns)
            pd.DataFrame(columns=['pid']).to_csv(kwargs['out_dir'], index=False)
        with patch('run.download_and_merge_buildings') as download, patch('openfacades.streetview.geo_aov_calculator', side_effect=calculate):
            run_local.run_local(self.images, self.metadata, self.root / 'local_output', footprint)
            download.assert_not_called()

    def test_invalid_metadata_fails_before_download(self):
        pd.DataFrame([{'pid':'bad', 'lat':999, 'lng':11, 'heading':0}]).to_csv(self.metadata, index=False)
        with patch('run.download_and_merge_buildings') as download:
            with self.assertRaises(ValueError):
                run_local.run_local(self.images, self.metadata, self.root / 'bad')
            download.assert_not_called()

    def test_osmnx_2_bounds_and_building_id(self):
        frame = gpd.GeoDataFrame({'geometry': [box(11,48,12,49)]}, index=pd.MultiIndex.from_tuples([('way', 123)], names=['element', 'id']), crs='EPSG:4326')
        with patch('openfacades.footprint.downloader.ox.features_from_bbox', return_value=frame) as fetch, patch('openfacades.footprint.downloader.ox.__version__', '2.1.0'):
            result = BuildingDataDownloader('osm').download_buildings([11,48,12,49])
        self.assertEqual(fetch.call_args.args[0], (11,48,12,49))
        self.assertEqual(result.building_id.iloc[0], 123)

    def test_unobstructed_and_blocked_buildings(self):
        row = pd.Series({'geometry':Point(0,0), 'pid':'001_GSV', 'lat':0, 'lng':0, 'compass_angle':0})
        visible = gpd.GeoDataFrame({'building_id':['target'], 'geometry':[box(10,-5,20,5)]}, crs='EPSG:32632')
        result = process_point(row, visible, 30, 2)
        self.assertTrue(any(r['building_id']=='target' and r['aov_geo']>0 for r in result))
        blocked = gpd.GeoDataFrame({'building_id':['target','blocker'], 'geometry':[box(10,-5,20,5),box(4,-10,6,10)]}, crs='EPSG:32632')
        result = process_point(row, blocked, 30, 2)
        self.assertFalse(any(r['building_id']=='target' for r in result))

    def test_osm_and_overture_are_both_merged(self):
        import run
        osm = gpd.GeoDataFrame({'building_id':['osm1'], 'building_type':['yes'],
                               'geometry':[box(11.568,48.148,11.5681,48.1481)]}, crs='EPSG:4326')
        overture = gpd.GeoDataFrame({'building_id':['ext1'], 'building_type':['residential'],
                                    'geometry':[box(11.5683,48.148,11.5684,48.1481)]}, crs='EPSG:4326')
        output = self.root / 'merged'
        output.mkdir()
        sources = []
        def download(instance, bounds):
            sources.append(instance.source)
            return (osm if instance.source == 'osm' else overture).copy()
        with patch.object(BuildingDataDownloader, 'download_buildings', download):
            run.download_and_merge_buildings([11.567,48.147,11.569,48.149],str(output))
        self.assertEqual(sources, ['osm', 'overture'])
        merged = gpd.read_file(output/'footprint.geojson')
        self.assertEqual(set(merged.building_id), {'osm1', 'ext1'})

    def test_raw_metadata_reaches_existing_detection_and_filtering(self):
        from concurrent.futures import ThreadPoolExecutor
        footprint = self.root / 'visible.geojson'
        gpd.GeoDataFrame({'osmid':[123], 'geometry':[box(11.5682,48.14795,11.56835,48.14805)]}, crs='EPSG:4326').to_file(footprint, driver='GeoJSON')
        def detect(aov, output, images_folder=None):
            candidates = pd.read_csv(aov, dtype={'pid':str})
            self.assertIn('001_GSV',set(candidates.pid))
            self.assertEqual(images_folder,str(self.images))
            pd.DataFrame([{'image_name':'sample.png','pid':'001_GSV','building_id':'123'}]).to_csv(Path(output)/'individual_building.csv',index=False)
        with patch('openfacades.streetview.aovcalculator.ProcessPoolExecutor', ThreadPoolExecutor), patch('run.run_object_detection', side_effect=detect) as detection, patch('run.select_suitable_building_image') as filtering:
            run_local.run_local(self.images,self.metadata,self.root/'pipeline',footprint)
            detection.assert_called_once()
            filtering.assert_called_once()

    def test_jpg_detection_preserves_string_ids(self):
        from PIL import Image
        from openfacades.detection.detector import GroundingDinoDetector
        Image.fromarray(np.full((180,360,3),127,dtype=np.uint8)).save(self.images/'001_GSV.jpg')
        metadata = self.root/'aov.csv'
        pd.DataFrame([{'pid':'001_GSV','building_id':'001','compass_angle':0,
                       'left_angle_geo':350,'right_angle_geo':10,'aov_geo':20}]).to_csv(metadata,index=False)
        detector = object.__new__(GroundingDinoDetector)
        detector.model = None
        with patch('openfacades.detection.detector.load_image',return_value=(None,None)), patch('openfacades.detection.detector.predict',return_value=(np.array([[.5,.5,.8,.5]]),np.array([.9]),['building'])):
            output = self.root/'detected'
            detector.run_detection(str(metadata),str(self.images),str(output))
        results = pd.read_csv(output/'individual_building.csv',dtype={'pid':str,'building_id':str})
        self.assertEqual(results.pid.iloc[0],'001_GSV')
        self.assertEqual(results.building_id.iloc[0],'001')
        self.assertTrue((output/'individual_building/pid_001_GSV_bdid_001.png').exists())

    def test_no_detections_produce_empty_result(self):
        from PIL import Image
        from openfacades.detection.detector import GroundingDinoDetector
        Image.fromarray(np.full((180, 360, 3), 127, dtype=np.uint8)).save(self.images / '001_GSV.jpg')
        metadata = self.root / 'aov.csv'
        pd.DataFrame([{'pid': '001_GSV', 'building_id': '001', 'compass_angle': 0,
                       'left_angle_geo': 350, 'right_angle_geo': 10, 'aov_geo': 20}]).to_csv(metadata, index=False)
        detector = object.__new__(GroundingDinoDetector)
        detector.model = None
        with patch('openfacades.detection.detector.load_image', return_value=(None, None)), patch('openfacades.detection.detector.predict', return_value=(np.empty((0, 4)), np.array([]), [])):
            output = self.root / 'no_detections'
            detector.run_detection(str(metadata), str(self.images), str(output))
        self.assertTrue(pd.read_csv(output / 'building_bbox.csv').empty)
        self.assertTrue(pd.read_csv(output / 'individual_building.csv').empty)

    def test_missing_images_produce_empty_result(self):
        from openfacades.detection.detector import GroundingDinoDetector
        metadata = self.root/'aov.csv'
        pd.DataFrame([{'pid':'missing','building_id':'001','compass_angle':0,
                       'left_angle_geo':350,'right_angle_geo':10,'aov_geo':20}]).to_csv(metadata,index=False)
        detector = object.__new__(GroundingDinoDetector)
        output = self.root/'empty'
        detector.run_detection(str(metadata),str(self.images),str(output))
        self.assertTrue(pd.read_csv(output/'individual_building.csv').empty)

    def test_jpg_lookup_and_string_pid_projection(self):
        from PIL import Image
        image = self.images / '001_GSV.JPG'
        Image.fromarray(np.full((180,360,3),127,dtype=np.uint8)).save(image)
        self.assertEqual(_find_image_file('001_GSV',self.images),str(image))
        results = self.root / 'boxes.csv'
        pd.DataFrame([{'pid':'001_GSV','building_id':'001','boxes':'[0.5,0.5,0.1,0.2,0.9]'}]).to_csv(results,index=False)
        output = self.root / 'projected'
        frame = save_bounding_boxes_as_images(results,self.images,output)
        self.assertEqual(frame.pid.iloc[0],'001_GSV')
        self.assertTrue((output/'individual_building/pid_001_GSV_bdid_001.png').exists())


if __name__ == '__main__':
    unittest.main()
