"""Offline geometry/selection tests; external services and model inference are mocked."""
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch, Mock
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box, MultiPolygon, Polygon

import run_building as query
from openfacades.streetview.aovcalculator import process_point, geo_aov_calculator


class BuildingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.local = gpd.GeoDataFrame({'building_id': ['way_123'], 'osm_id': ['way/123'],
            'geometry': [box(500000, 5300000, 500020, 5300020)]}, crs=32632)
        self.target = self.local.to_crs(4326)

    def metadata(self, points):
        points = gpd.GeoSeries(points, crs=self.local.crs).to_crs(4326)
        return pd.DataFrame({'id': [f'{i:03d}' for i in range(len(points))],
            'lng': points.x, 'lat': points.y, 'computed_compass_angle': 0, 'quality_score': .8})

    def test_id_types_and_invalid_input(self):
        for value, expected in [('123','way/123'),('way/123','way/123'),('relation/123','relation/123')]:
            self.assertEqual(query.normalize_id(value), expected)
        for value in ['0','-1','node/1','way/0','123.0','way/../1','１２３']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                query.normalize_id(value)

    def test_target_full_osm_way_and_relation(self):
        # OSMnx performs real XML assembly, including the relation member ring.
        nodes = ''.join(f'<node id="{i}" lat="{lat}" lon="{lon}"/>' for i,lat,lon in
                        [(1,48,11),(2,48,11.001),(3,48.001,11.001),(4,48.001,11)])
        way = '<way id="123"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/><tag k="building" v="yes"/></way>'
        relation = '<relation id="123"><member type="way" ref="123" role="outer"/><tag k="type" v="multipolygon"/><tag k="building" v="yes"/></relation>'
        response = Mock(content=f'<osm version="0.6">{nodes}{way}{relation}</osm>'.encode())
        with patch('requests.get', return_value=response) as request:
            for kind in ['way','relation']:
                target = query.get_target(f'{kind}/123', self.root)
                self.assertEqual(target.osm_id.iloc[0], f'{kind}/123')
                self.assertEqual(target.building_id.iloc[0], f'{kind}_123')
                self.assertAlmostEqual(target.geometry.iloc[0].area, .000001)
                self.assertTrue(request.call_args.args[0].endswith(f'/{kind}/123/full'))

    def test_target_network_error_is_not_empty_success(self):
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError('network unavailable')
        with patch('requests.get',return_value=response), self.assertRaises(RuntimeError):
            query.get_target('way/123',self.root)

    def test_true_30m_distance_not_bounding_box(self):
        data = self.metadata([Point(499970,5300010), Point(499969,5300010),
            Point(499971,5299971), Point(500010,5300010), Point(500000,5300010)])
        path = self.root/'metadata.csv'
        data.to_csv(path,index=False)
        filtered = query.filter_metadata(path,self.local)
        self.assertEqual(filtered.id.tolist(),['000'])

    def test_southern_hemisphere_buffers(self):
        target = gpd.GeoDataFrame({'geometry':[box(151.2,-33.86,151.201,-33.859)]},crs=4326)
        local, pano, context = query.search_boundaries(target)
        self.assertEqual(local.crs.to_epsg(),32756)
        bounds = local.total_bounds
        np.testing.assert_allclose(pano.to_crs(local.crs).total_bounds, bounds + [-30,-30,30,30], atol=.02)
        np.testing.assert_allclose(context.to_crs(local.crs).total_bounds, bounds + [-100,-100,100,100], atol=.02)

    def test_metadata_dedup_invalid_positions_and_headings(self):
        data = self.metadata([Point(499980,5300010)]*4)
        data.loc[1,'id'] = '000'
        data.loc[2,'lng'] = np.inf
        data.loc[3,'computed_compass_angle'] = -1
        path = self.root/'meta.csv'
        data.to_csv(path,index=False)
        filtered = query.filter_metadata(path,self.local)
        self.assertEqual(filtered.id.tolist(), ['000'])

    def test_restore_exact_target_preserves_same_number_neighbour(self):
        neighbours = gpd.GeoDataFrame({'building_id':['123','123','outside'],
            'geometry':[self.local.geometry.iloc[0],box(500030,5300000,500040,5300010),
                        box(500300,5300000,500310,5300010)]},crs=32632).to_crs(4326)
        path = self.root/'footprint.geojson'
        neighbours.to_file(path,driver='GeoJSON')
        query.restore_target(path,self.target,self.local)
        restored = gpd.read_file(path)
        self.assertEqual(len(restored),2)
        self.assertEqual((restored.building_id=='way_123').sum(),1)
        self.assertTrue(restored[restored.building_id=='way_123'].geometry.iloc[0].equals(self.target.geometry.iloc[0]))

    def view(self, point, geometries=None, target='way_123', buffer=30):
        buildings = geometries if geometries is not None else self.local
        row={'geometry':point,'pid':'001','lat':48,'lng':11,'compass_angle':0}
        return process_point(row,buildings,buffer,2,target_building_id=target)

    def test_different_facades_and_self_occlusion(self):
        west=self.view(Point(499980,5300010))[0]
        east=self.view(Point(500040,5300010))[0]
        corner=self.view(Point(499980,5299980))[0]
        self.assertTrue(set(json.loads(west['visible_sides'])).isdisjoint(json.loads(east['visible_sides'])))
        self.assertEqual(len(json.loads(west['visible_sides'])),1)
        self.assertEqual(len(json.loads(corner['visible_sides'])),2)
        self.assertGreater(west['frontality'],.9)

    def test_obstruction_outside_old_30m_viewpoint_buffer(self):
        buildings=gpd.GeoDataFrame({'building_id':['way_123','obstacle'],
            'geometry':[box(500000,5300000,500010,5300200),box(499980,5300050,499995,5300150)]},crs=32632)
        values=self.view(Point(499980,5299990),buildings)
        no_obstacle=self.view(Point(499980,5299990),buildings.iloc[:1])
        self.assertLess(values[0]['aov_geo'],no_obstacle[0]['aov_geo'])

    def test_fully_blocked_target(self):
        buildings=pd.concat([self.local,gpd.GeoDataFrame({'building_id':['wall'],
            'geometry':[box(499985,5299990,499990,5300030)]},crs=32632)],ignore_index=True)
        self.assertEqual(self.view(Point(499980,5300010),buildings),[])

    def test_multipolygon_returns_one_target(self):
        target=self.local.copy()
        target['geometry']=[MultiPolygon([box(500000,5300000,500010,5300010),box(500000,5300020,500010,5300030)])]
        views=self.view(Point(499980,5300015),target)
        self.assertEqual(len(views),1)
        self.assertTrue(any(side.startswith('1:') for side in json.loads(views[0]['visible_sides'])))

    def test_original_aov_mode_keeps_original_schema(self):
        rows=self.view(Point(499980,5300010),target=None)
        self.assertEqual(set(rows[0]),{'pid','lat','lng','compass_angle','building_id',
            'left_angle_geo','right_angle_geo','aov_geo','distance'})

    def test_north_facing_target_angle_crosses_zero(self):
        result=self.view(Point(500010,5299980))[0]
        self.assertGreater(result['left_angle_geo'],result['right_angle_geo'])
        self.assertGreater(result['aov_geo'],10)
        self.assertLess(result['aov_geo'],130)

    def test_concave_building_uses_visible_walls_not_center_quadrants(self):
        target=self.local.copy()
        target['geometry']=[Polygon([(500000,5300000),(500030,5300000),
            (500030,5300010),(500010,5300010),(500010,5300030),(500000,5300030)])]
        result=self.view(Point(500020,5300020),target)[0]
        self.assertEqual(len(json.loads(result['visible_sides'])),2)

    def test_projection_caps_wide_view_and_preserves_center_and_vertical_view(self):
        from PIL import Image
        import math
        from openfacades.detection.pano2pers import save_bounding_boxes_as_images
        images=self.root/'images'
        images.mkdir()
        Image.new('RGB',(720,360),'white').save(images/'001.jpg')
        boxes=self.root/'boxes.csv'
        for width, cap, expected in [(150/360,135,135),(90/360,135,90),(150/360,None,150)]:
            with self.subTest(width=width,cap=cap):
                pd.DataFrame([{'pid':'001','building_id':'123',
                    'boxes':str([.7,.4,width,.5,.9])}]).to_csv(boxes,index=False)
                with patch('openfacades.detection.pano2pers.get_perspective',return_value=Image.new('RGB',(10,10))) as project:
                    save_bounding_boxes_as_images(boxes,images,self.root/'crop',max_fov=cap)
                args=project.call_args.kwargs
                self.assertEqual(args['FOV'],expected)
                self.assertEqual(args['THETA'],int((.7-.5)*360))
                self.assertEqual(args['PHI'],int((.5-.4)*180))
                original_fov=int(width*360)
                original_width=width*720
                self.assertAlmostEqual(args['width']/math.tan(math.radians(expected/2)),
                    original_width/math.tan(math.radians(original_fov/2)))

    def test_existing_mapillary_entry_keeps_its_pipeline(self):
        import run
        previous=Path.cwd()
        os.chdir(self.root)
        try:
            names=['download_and_merge_buildings','prepare_mly_metadata','compute_and_filter_aov',
                   'download_selected_images','run_object_detection','select_suitable_building_image']
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(patch('sys.argv',['run.py','--bbox','[11,48,11.001,48.001]','--api_key','local-test-token']))
                functions={name:stack.enter_context(patch.object(run,name)) for name in names}
                functions['prepare_mly_metadata'].return_value='clean.csv'
                functions['compute_and_filter_aov'].return_value=('aov.csv','selected.csv')
                run.main()
                for function in functions.values():
                    function.assert_called_once()
                functions['run_object_detection'].assert_called_with('aov.csv',str(self.root.resolve()/'output/02_img'))
        finally:
            os.chdir(previous)

    def test_candidates_prefer_side_coverage_and_limit_downloads(self):
        rows=[{'pid':str(i),'visible_sides':json.dumps(['west']), 'frontality':1.,'quality_score':1.,
               'aov_geo':50.,'distance':10.} for i in range(30)]
        rows.append(dict(rows[0],pid='east',visible_sides='["east"]',frontality=.5))
        selected=query.select_views(pd.DataFrame(rows),20)
        self.assertEqual(len(selected),20)
        self.assertEqual(selected.pid.iloc[1],'east')
        self.assertEqual(selected.pid.nunique(),20)

    def test_final_limit_dedup_and_deterministic_quality(self):
        data=pd.DataFrame([{'pid':str(i),'visible_sides':'["west"]','blur_score':i+40,
            'building_ratio':.5,'frontality':1.,'quality_score':.9,'aov_geo':50,'distance':20} for i in range(9)])
        data=pd.concat([data,data.iloc[:1]],ignore_index=True)
        result=query.select_views(data,5,final=True)
        self.assertEqual(result.pid.tolist(),['8','7','6','5','4'])
        pd.testing.assert_frame_equal(result, query.select_views(data.sample(frac=1,random_state=3),5,final=True))
        self.assertEqual(len(query.select_views(data.iloc[:2],5)),2)

    def test_limits_checked_before_any_network(self):
        with patch.object(query,'get_target') as lookup:
            for limit in [0,-1,21,2.5,True]:
                with self.assertRaises(ValueError):
                    query.run_building('123','local',limit)
            with self.assertRaises(ValueError):
                query.run_building('123','')
            lookup.assert_not_called()

    def pipeline(self, empty=False, blocked=False, failure=False, no_crops=False, no_quality=False):
        from PIL import Image
        def metadata(bounds, directory, key):
            path=Path(directory)/'pids_urls_clean.csv'
            data=self.metadata([Point(499980,5300010),Point(500040,5300010)])
            if empty:
                data=data.iloc[:0]
            data.to_csv(path,index=False)
            return path
        def footprints(bounds,directory,overwrite=False):
            buildings=self.target.copy()
            if blocked:
                wall=gpd.GeoDataFrame({'building_id':['a','b'],'geometry':[
                    box(499985,5299990,499990,5300030),box(500030,5299990,500035,5300030)]},crs=32632).to_crs(4326)
                buildings=pd.concat([buildings,wall],ignore_index=True)
            buildings.to_file(Path(directory)/'footprint.geojson',driver='GeoJSON')
        def download(directory,path,key):
            if failure:
                return
            folder=Path(directory)/'mly_svi'; folder.mkdir()
            for pid in pd.read_csv(path,dtype={'id':str}).id:
                Image.new('RGB',(1024,512),'white').save(folder/f'{pid}.jpg')
        def detect(path,directory,max_fov=None):
            self.assertEqual(max_fov,135)
            rows=[]
            if not no_crops:
                for pid in pd.read_csv(path,dtype={'pid':str}).pid:
                    name=f'pid_{pid}_bdid_way_123.png'
                    Image.new('RGB',(256,256),'white').save(Path(directory)/'individual_building'/name)
                    rows.append({'image_name':name,'pid':pid,'building_id':'way_123'})
            pd.DataFrame(rows,columns=['image_name','pid','building_id']).to_csv(Path(directory)/'individual_building.csv',index=False)
        def quality(directory):
            crops=pd.read_csv(Path(directory)/'individual_building.csv',dtype={'pid':str})
            if no_quality:
                crops=crops.iloc[:0]
            crops.to_csv(Path(directory)/'individual_building_select.csv',index=False)
            pd.DataFrame({'filename_key':crops.image_name.apply(lambda x:Path(x).stem),'Building':.8}).to_csv(Path(directory)/'pixel_ratios.csv',index=False)
        fake_filter=types.ModuleType('openfacades.detection.filter')
        fake_filter.calculate_blur_score=lambda path:100.
        previous=Path.cwd()
        os.chdir(self.root)
        try:
            with patch.object(query,'get_target',return_value=self.target), \
                 patch.object(query,'prepare_mly_metadata',side_effect=metadata), \
                 patch.object(query,'download_and_merge_buildings',side_effect=footprints), \
                 patch.object(query,'download_selected_images',side_effect=download) as downloads, \
                 patch.object(query,'run_object_detection',side_effect=detect), \
                 patch.object(query,'select_suitable_building_image',side_effect=quality) as filters, \
                 patch('openfacades.streetview.aovcalculator.ProcessPoolExecutor',ThreadPoolExecutor), \
                 patch.dict('sys.modules',{'openfacades.detection.filter':fake_filter}):
                path=query.run_building('way/123','local-test-token')
                return path.resolve(), downloads.call_count, filters.call_count
        finally:
            os.chdir(previous)

    def test_pipeline_real_geometry_output_isolation_and_no_secret(self):
        first,downloads,filters=self.pipeline()
        second,_,_=self.pipeline()
        self.assertNotEqual(first,second)
        self.assertTrue(first.exists())
        self.assertEqual(downloads,1)
        self.assertEqual(filters,1)
        result=pd.read_csv(first,dtype={'pid':str})
        self.assertEqual(len(result),2)
        self.assertEqual(set(result.osm_id),{'way/123'})
        self.assertEqual(result.pid.tolist(),['000','001'])
        self.assertFalse((self.root/'output/01_data').exists())
        for path in first.parents[1].rglob('*'):
            if path.suffix in ['.csv','.json']:
                self.assertNotIn('local-test-token',path.read_text())

    def test_empty_metadata_writes_empty_result_without_models(self):
        path,downloads,filters=self.pipeline(empty=True)
        self.assertTrue(pd.read_csv(path).empty)
        self.assertEqual((downloads,filters),(0,0))
        self.assertEqual(json.loads((path.parents[1]/'summary.json').read_text())['status'],'no_panoramas')

    def test_no_visibility_does_not_download_images(self):
        path,downloads,filters=self.pipeline(blocked=True)
        self.assertEqual((downloads,filters),(0,0))
        self.assertEqual(json.loads((path.parents[1]/'summary.json').read_text())['status'],'no_visible_views')

    def test_download_failure_is_error(self):
        with self.assertRaises(RuntimeError):
            self.pipeline(failure=True)
        summaries=list((self.root/'output').rglob('summary.json'))
        self.assertEqual(json.loads(summaries[0].read_text())['status'],'error')

    def test_no_detections_skips_quality_models(self):
        path,_,filters=self.pipeline(no_crops=True)
        self.assertEqual(filters,0)
        self.assertTrue(pd.read_csv(path).empty)

    def test_all_rejected_by_quality_is_valid_empty_result(self):
        path,_,filters=self.pipeline(no_quality=True)
        self.assertEqual(filters,1)
        self.assertTrue(pd.read_csv(path).empty)
        self.assertEqual(list(pd.read_csv(path)),query.RESULT_COLUMNS)
        self.assertEqual(json.loads((path.parents[1]/'summary.json').read_text())['status'],'no_quality_views')


if __name__ == '__main__':
    unittest.main()
