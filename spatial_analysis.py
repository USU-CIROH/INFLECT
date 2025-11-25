"""
Bankfull Intersection Identification

Map-based visualization functions for bankfull method development and refinement.

Noelle Patterson, USU 
February 2024
"""

import geopandas as gpd
from shapely.geometry import Polygon, MultiPoint, LineString, Point
import pandas as pd
import numpy as np
import rasterio
from sklearn.linear_model import LinearRegression


"""Function for creating mapping points from bankfull results to overlay on cross-sections"""
def create_bankfull_pts(cross_sections, dem_fp, thalweg, d_interval, spatial_plot_interval, reach_name):
    all_widths_df = pd.read_csv('data_outputs/{}/all_widths.csv'.format(reach_name))
    # Create a multipoint object of bankfull extent from the cross-section and aggregate bankfull elevation data
    inflections = pd.read_csv('data_outputs/{}/max_inflections.csv'.format(reach_name))
    dem = rasterio.open(dem_fp)

    pos_inflections = inflections['pos_inflections']
    neg_inflections = inflections['neg_inflections']

    # Use thalweg elevs to un-detrend inflection point result for map-based plotting
    x = np.cumsum(all_widths_df['thalweg_distance'].values).reshape((-1,1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]
    
    def map_inflections(inflections_series, sign):
        # this function is supposed to create a set of points along each transect. Can you help fix it? It currently only saves points for the first transect..
        intersection_pts = []
        for inf_index, inflection in enumerate(inflections_series):
            # Loop through cross_sections.
            for transect_index, row in cross_sections.iterrows():
                line = gpd.GeoDataFrame({'geometry': [row['geometry']]}, crs=cross_sections.crs)
                # Assign elevational data to line segment. Need to discretize line into closely-spaced points first. 
                tot_len = line.length
                distances = np.arange(0, tot_len[0], spatial_plot_interval) 
                stations = row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
                stations = gpd.GeoDataFrame(geometry=stations, crs=cross_sections.crs)
                # Extract z elevation at each station along transect
                elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))
                # current_topo_bf = topo_bankfull['bankfull'].iloc[transect_index]

                # Un-detrend inflection point results
                current_inflection = inflection * d_interval + fit_slope[transect_index]

                # loop through each elev in elevs
                current_intersect_pts = []
                for index, elev in enumerate(elevs):
                    val = elev - current_inflection
                    if index > 0:
                        prev_val = elevs[index - 1] - current_inflection
                        if val * prev_val < 0:
                            current_intersect_pts.append(stations['geometry'][index])
                            # intersection_pts.append(stations['geometry'][index])
                # If more than two intersection points identified in the transect drop all except two closest to thalweg
                if len(current_intersect_pts) > 2:
                    # Code from ChatGPT
                    line = thalweg['geometry']
                    # 1. select closest segment in thalweg for distance comparison
                    def closest_segment(line, point):
                        coords = line.get_coordinates()
                        min_dist = float('inf')
                        closest_seg = None
                        for i in range(len(coords) - 1):
                            seg = LineString([coords.iloc[i], coords.iloc[i+1]])
                            dist = point.distance(seg)
                            if dist < min_dist:
                                min_dist = dist
                                closest_seg = seg
                        return closest_seg
                    # 2. determine if points are on left or right side of nearest thalweg segment
                    def side_of_segment(segment, point):
                        x1, y1 = segment.coords[0]
                        x2, y2 = segment.coords[1]
                        x, y = point.x, point.y
                        cross = (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1) # calculate cross product of vectors
                        return 'left' if cross > 0 else 'right'
                    
                    point_info = []
                    for pt in current_intersect_pts:
                        seg = closest_segment(line, pt)
                        dist = pt.distance(seg)
                        side = side_of_segment(seg, pt)
                        point_info.append({
                            'point': pt,
                            'distance': dist,
                            'side': side,
                            'segment': seg
                        })
                    # 3. select the closest points left and right of the thalweg to keep for mapping 
                    
                    left_points = [p for p in point_info if p['side'] == 'left']
                    right_points = [p for p in point_info if p['side'] == 'right']

                    if not left_points and not right_points:
                        # No valid points found, skip to next transect
                        continue

                    closest_left = min(left_points, key=lambda p: p['distance'], default=None)
                    closest_right = min(right_points, key=lambda p: p['distance'], default=None)
                    if closest_left is not None:
                        intersection_pts.append(closest_left['point'])
                    if closest_right is not None:
                        intersection_pts.append(closest_right['point'])
                else:
                    if current_intersect_pts:
                        intersection_pts.append(current_intersect_pts[0])
                        if len(current_intersect_pts) > 1:
                            intersection_pts.append(current_intersect_pts[1])
        multipoint_geom = MultiPoint(intersection_pts)
        multipoint = gpd.GeoDataFrame(index=[0], crs=cross_sections.crs, geometry=[multipoint_geom])
        multipoint.to_file(filename='data_outputs/{}/spatial/inflections_{}_multipoint.shp'.format(reach_name, sign), driver="ESRI Shapefile")
    print('positive inflections')
    map_inflections(pos_inflections, 'positive') 
    print('negative inflections')   
    map_inflections(neg_inflections, 'negative')    
