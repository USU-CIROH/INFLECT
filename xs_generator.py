from shapely.geometry import LineString, Point
import math
import geopandas as gpd
import pandas as pd
import glob
import os
from pynhd import NLDI
nldi = NLDI()

# Bring in centerline(s)
centerline_fp = glob.glob('data_inputs/Willamette/Thalweg/Thalweg1.shp')
centerline_fps = [centerline_fp]
name = 'Willamette_TW1'

# XS generation parameters
xs_num = 60
width_multiplier = 15 # Set cross-section width as a scalar multiplied by mean river width estimate
river_widths = pd.read_csv('data_inputs/Bankfull_Meanflow_CONUS_Stream_Reaches.txt', sep=',')

# Define XS-generation function 
# Code from Chat GPT
def perpendicular_station_line(
    line: LineString,
    distance: float,
    half_length: float = 50
) -> LineString:
    if distance <= 0:
        seg_start = Point(line.coords[0])
        seg_end = Point(line.coords[1])
        station_pt = seg_start
    elif distance >= line.length:
        seg_start = Point(line.coords[-2])
        seg_end = Point(line.coords[-1])
        station_pt = seg_end
    else:
        # Walk segments until we find the one where we draw the cross-section
        coords = list(line.coords)
        remaining = distance

        for i in range(len(coords) - 1):
            p0 = Point(coords[i])
            p1 = Point(coords[i + 1])
            seg_len = p0.distance(p1)

            if remaining <= seg_len:
                frac = remaining / seg_len
                station_pt = LineString([p0, p1]).interpolate(frac, normalized=True)
                seg_start, seg_end = p0, p1
                break

            remaining -= seg_len

    # Segment direction
    dx = seg_end.x - seg_start.x
    dy = seg_end.y - seg_start.y

    angle = math.atan2(dy, dx)
    perp_angle = angle + math.pi / 2

    cx, cy = station_pt.coords[0]

    x1 = cx + half_length * math.cos(perp_angle)
    y1 = cy + half_length * math.sin(perp_angle)
    x2 = cx - half_length * math.cos(perp_angle)
    y2 = cy - half_length * math.sin(perp_angle)

    return LineString([(x1, y1), (x2, y2)])


'''
Loop through list of centerlines, and generate cross-sections
'''

for index, centerline_fp in enumerate(centerline_fps):
    segment_gdf = gpd.read_file(centerline_fp[index])
    segment = segment_gdf.geometry.iloc[0]
    xs_crs = segment_gdf.crs
    # get nearest comid using midpoint of centerline:
    center_point = segment.interpolate(segment.length / 2)
    center_coords = (center_point.x, center_point.y)
    comid_closest = nldi.comid_byloc((center_coords), xs_crs)
    comid = comid_closest['comid'][0]
    river_width = river_widths[river_widths['COMID'] == comid]['mf_width']
    river_width = river_width.values[0] # extract numeric from Series
    xs_per_comid = xs_num # try measuring xs spacing as function of comid
    xs_spacing = segment.length/(xs_per_comid - 1)
    xs_len = river_width * width_multiplier 

    station_geoms = []
    station_dist = []
    d = 0

    for i in range(xs_per_comid):
        d = i * xs_spacing
        station_geoms.append(
            perpendicular_station_line(segment, d, half_length=xs_len/2)
        )
        station_dist.append(d)

    cross_sections = gpd.GeoDataFrame(
        {"station": station_dist},
        geometry=station_geoms,
        crs=xs_crs
    )
  
    # Export xs and segment as shapefiles
    output_dir = 'data_outputs/{}/cross_sections'.format(name)
    os.makedirs(output_dir, exist_ok=True)
    cross_sections.to_file('{}/{}_xs.shp'.format(output_dir, name))