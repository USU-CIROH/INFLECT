import rasterio
from rasterio.features import shapes
from shapely.geometry import LineString, shape
from shapely.ops import unary_union
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


#Cross-section filter function to remove cross-sections with more than 30% of the line outside of the extents of the DEM
def cross_section_within_dem(line, dem_valid_poly, threshold = 0.7):
    """
    Returns True if >= threshold fraction of line lies inside the valid DEM cells
    """

    if line.length == 0:
        return False
    
    intersection = line.intersection(dem_valid_poly)

    return(intersection.length / line.length >= threshold)

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

    # ---- get point on line that represents center of the cross-section ----
    p = line.interpolate(distance)

    # ---- small offset to compute local direction ----
    eps = max(line.length * 1e-6, 0.01)

    d1 = max(0, distance - eps)
    d2 = min(line.length, distance + eps)

    p_before = line.interpolate(d1)
    p_after = line.interpolate(d2)

    # ---- direction vector to approximate tangent to the centerline----
    dx = p_after.x - p_before.x
    dy = p_after.y - p_before.y

    # ---- perpendicular direction ----
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length

    # perpendicular vector for cross-section direction
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
    
    dem_dir = reach_dir / 'dem'
    # select 1m DEM if available, otherwise use 10m DEM; this can be edited later when we want to run 10m for comparison
    dems = list(dem_dir.glob("*.tif"))

    if len(dems) == 0:
        print(f"No DEM found for {station_id}")
        continue

    dem_1m = [d for d in dems if '_1m' in d.stem]

    if len(dem_1m) > 0:
        dem_fp = str(dem_1m[0])
    else:
        dem_fp = str(dems[0])  # Use the first available DEM if no 1m DEM is found


    with rasterio.open(dem_fp) as src:
        mask = src.dataset_mask()

        valid_polys = [shape(geom) for geom, val in shapes(mask, mask=mask > 0, transform = src.transform)]

        dem_valid_poly = unary_union(valid_polys)


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

            xs_line = perpendicular_station_line(segment, d, half_length=xs_len/2)

            #DEM boundary filter
            if not cross_section_within_dem(xs_line, dem_valid_poly, threshold = 0.9):
                print(f"Skipping XS at distance {d:.2f} (outside DEM bounds)")
                continue

            station_geoms.append(
                xs_line
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