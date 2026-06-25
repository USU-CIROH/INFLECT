from rasterio import coords
from shapely.geometry import LineString, Point
import math
import geopandas as gpd
import pandas as pd
from pathlib import Path
import glob
import os
from pynhd import NLDI
from xarray.plot import line
nldi = NLDI()

# # Bring in centerline(s)
# centerline_fp = glob.glob('data_inputs/Willamette/Thalweg/Thalweg1.shp')
# centerline_fps = [centerline_fp]
# name = 'Willamette_TW1'

# Identify the path to the parent directory containing subfolders for each reach. 
# Each reach folder should contain a 'Thalweg' subfolder with the centerline shapefile(s). 
# The script will generate cross-sections for each centerline and save them in a 'Cross_Sections' 
# subfolder within each reach folder.
parent_dir = Path(r"/Users/ahurst/Documents/USU/Benchmark_sites_from_SNR") 

# XS generation parameters
xs_num = 60
width_multiplier = 15 # Set cross-section width as a scalar multiplied by mean river width estimate
river_widths = pd.read_csv('/Users/ahurst/Documents/USU/Bankfull_Meanflow_CONUS_Stream_Reaches.txt', sep=',')

# Define XS-generation function 
# Code from Chat GPT
# def perpendicular_station_line(
#     line: LineString,
#     distance: float,
#     half_length: float = 50
# ) -> LineString:
#     if distance <= 0:
#         seg_start = Point(line.coords[0][:2])
#         seg_end = Point(line.coords[1][:2])
#         station_pt = seg_start
#     elif distance >= line.length:
#         seg_start = Point(line.coords[-2][:2])
#         seg_end = Point(line.coords[-1][:2])
#         station_pt = seg_end
#     else:
#         # Walk segments until we find the one where we draw the cross-section
#         coords = list(line.coords)
#         remaining = distance

#         for i in range(len(coords) - 1):
#             p0 = Point(coords[i][:2])
#             p1 = Point(coords[i + 1][:2])
#             seg_len = p0.distance(p1)

#             if remaining <= seg_len:
#                 frac = remaining / seg_len
#                 station_pt = LineString([p0, p1]).interpolate(frac, normalized=True)
#                 seg_start, seg_end = p0, p1
#                 break

#             remaining -= seg_len

#     # Segment direction
    
#     dx = seg_end.x - seg_start.x
#     dy = seg_end.y - seg_start.y

#     angle = math.atan2(dy, dx)
#     perp_angle = angle + math.pi / 2

#     cx, cy = station_pt.coords[0][:2]

#     x1 = cx + half_length * math.cos(perp_angle)
#     y1 = cy + half_length * math.sin(perp_angle)
#     x2 = cx - half_length * math.cos(perp_angle)
#     y2 = cy - half_length * math.sin(perp_angle)

#     return LineString([(x1, y1), (x2, y2)])

def perpendicular_station_line(line: LineString,
                               distance: float,
                               half_length: float = 50) -> LineString:
    """
    Generate a perpendicular cross-section at a given distance along a LineString.
    Works for any vertex density and avoids manual segment traversal.
    """

    # ---- clamp distance to valid range ----
    distance = max(0, min(distance, line.length))

    # ---- get point on line ----
    p = line.interpolate(distance)

    # ---- small offset to compute local direction ----
    eps = max(line.length * 1e-6, 0.01)

    d1 = max(0, distance - eps)
    d2 = min(line.length, distance + eps)

    p_before = line.interpolate(d1)
    p_after = line.interpolate(d2)

    # ---- direction vector ----
    dx = p_after.x - p_before.x
    dy = p_after.y - p_before.y

    # ---- perpendicular direction ----
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length

    # perpendicular vector
    px, py = -uy, ux

    cx, cy = p.x, p.y

    # ---- build cross-section line ----
    x1 = cx + px * half_length
    y1 = cy + py * half_length
    x2 = cx - px * half_length
    y2 = cy - py * half_length

    return LineString([(x1, y1), (x2, y2)])

'''
Loop through subfolders and generate cross-sections
'''

for reach_dir in parent_dir.iterdir():

    if not reach_dir.is_dir():
        continue
    
    thalweg_dir = reach_dir / 'Thalweg'

    if not thalweg_dir.exists():
        print(f"No 'Thalweg' directory found in {reach_dir}. Skipping.")
        continue    

    output_dir = reach_dir / 'Cross_Sections'
    output_dir.mkdir(exist_ok=True)

    shp_files = list(thalweg_dir.glob('*.shp'))

    for shp_file in shp_files:
        print(f"Processing {shp_file}...")

        segment_gdf = gpd.read_file(shp_file)
        segment = segment_gdf.geometry.iloc[0]
        xs_crs = segment_gdf.crs

        print(f"Geometry type: {segment.geom_type}")
        print(f"Length: {segment.length}")
        print(f"Num coords: {len(segment.coords)}")

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

        output_name = f"{shp_file.stem}_xs.shp"
        cross_sections.to_file(output_dir / output_name)
        print(f" Saved: {output_name}")
    
        # # Export xs and segment as shapefiles
        # output_dir = 'data_outputs/{}/cross_sections'.format(name)
        # os.makedirs(output_dir, exist_ok=True)
        # cross_sections.to_file('{}/{}_xs.shp'.format(output_dir, name))