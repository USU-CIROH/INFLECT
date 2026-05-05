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

    # Put inflections back in units of d_interval
    def convert_d_int(inflect_ls, d_interval):
        inflections_dint = []
        for inflection in inflect_ls:
            inflection_dint = inflection / d_interval
            inflections_dint.append(inflection_dint)
        return inflections_dint
    pos_inflections = convert_d_int(pos_inflections, d_interval)
    neg_inflections = convert_d_int(neg_inflections, d_interval)

    # Use thalweg elevs to un-detrend inflection point result for map-based plotting
    x = np.cumsum(all_widths_df['thalweg_distance'].values).reshape((-1,1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]
    
    def map_inflections(inflections_series, sign):
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
                # If more than two intersection points identified in the transect keep the two closest to center of transect
                if len(current_intersect_pts) > 2:
                    # Get the center point and direction of the transect line
                    line_geom = row['geometry']
                    line_center = line_geom.interpolate(0.5, normalized=True)
                    
                    # Get line direction from start to end
                    coords = list(line_geom.coords)
                    start = coords[0]
                    end = coords[-1]
                    
                    # Determine which side of the line each point is on using cross product around center
                    def side_of_line(point, line_center, line_start, line_end):
                        x_start, y_start = line_start
                        x_end, y_end = line_end
                        x_center, y_center = line_center.x, line_center.y
                        x, y = point.x, point.y
                        # Use the transect orientation and translate relative to center
                        cross = (x_end - x_start) * (x - x_center) + (y_end - y_start) * (y - y_center)
                        return 'left' if cross > 0 else 'right'
                    
                    # Classify points by side and calculate distance to center
                    left_points = []
                    right_points = []
                    for pt in current_intersect_pts:
                        dist = pt.distance(line_center)
                        side = side_of_line(pt, line_center, start, end)
                        if side == 'left':
                            left_points.append((pt, dist))
                        else:
                            right_points.append((pt, dist))
                    
                    # Get the closest point on each side
                    if left_points:
                        closest_left = min(left_points, key=lambda x: x[1])
                        intersection_pts.append(closest_left[0])
                    if right_points:
                        closest_right = min(right_points, key=lambda x: x[1])
                        intersection_pts.append(closest_right[0])
                else:
                    if current_intersect_pts:
                        intersection_pts.append(current_intersect_pts[0])
                        if len(current_intersect_pts) > 1:
                            intersection_pts.append(current_intersect_pts[1])
        multipoint_geom = MultiPoint(intersection_pts)
        multipoint = gpd.GeoDataFrame(index=[0], crs=cross_sections.crs, geometry=[multipoint_geom])
        multipoint.to_file(filename='data_outputs/{}/spatial/inflections_{}_multipoint.shp'.format(reach_name, sign), driver="ESRI Shapefile")
    # To map only one inflection from the list, specify here, e.g. [pos_inflections[1]] or [neg_inflections[0]]
    print('mapping positive inflections')
    map_inflections([pos_inflections[1]], 'positive') 
    print('mapping negative inflections')   
    map_inflections([neg_inflections[0]], 'negative')    
