"""
INFLECT (INFLection-based Elevations from Channel Topography) Method

Primary implementation script.

Noelle Patterson, USU
Funded by CIROH (Cooperative Institute for Research to Operations in Hydrology) 
2024/2025
"""

import os
import glob
import geopandas as gpd
import pandas as pd
import glob
import rasterio
from rasterio import features
import ast
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
from shapely.ops import unary_union
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import sys
import time
from analysis import calc_dwdh, inflect, get_raster_boundary
from visualization import output_record, plot_bankfull_increments, plot_longitudinal_profile, transect_plot, plot_inflections
from spatial_analysis import create_bankfull_pts
from pathlib import Path

# Steps for INFLECT analysis:
# 1. Measure channel width along a depth interval for each cross-section (Analysis.py -> calc_dwdh)
#    Use either partial width method (default) or continuous width method (INFLECT algorithm parameters, width_calc_method)
# 2. Calculate first and second order derivatives of the channel widths to identify topographic inflection points (Analysis.py -> inflect)
# 3. Identify peaks in inflection curves representing channel elevation of major width inflection. 
#    Method either finds peaks from individual curves (cross-section method, default) or from an aggregated curve
#    representing the entire reach (aggregate method). See INFLECT parameters, inflect_calc_method. 
# 4. Post-processing: plot results for cross-section width measurements and inflection curve calculation (Visualization.py)
# 5. Generate spatial points where INFLECT points align with land surface elevations along input cross-sections (spatial_analysis.py)

# Set INFLECT algorithm parameters
units = 'meters' # based on units in DEM
sampling_interval = 1 # set elevation sampling interval along transect in units of meters
d_interval = 10/100 # Set intervals to step up in cross-section depth. Default is units meters and 1/10m intervals
slope_window = 10 # Set window size for calculating slope for derivatives. Set in d_interval units. 
lower_bound = 5 # Set lower vertical boundary for inflection id within cross-section, in units of d_interval. Default 5 = 50cm
upper_bound = 100 # Set upper vertical boundary for inflection id within cross-section, in units of d_interval. Default 100 = 10m
spatial_plot_interval = 0.5 # interval to measure elevations along transects, default units meters.
width_calc_method = 'partial' # 'continuous' 'partial' - choose from either partial additive widths or continuous-only methods of width calculation.
# peak detection parameters
inflect_calc_method = 'cross-section' # 'cross-section' or 'aggregate' method for inflection calculation. 'cross-section' is default. 
max_peak_ratio = 2 # The ratio of max peak:detected peak. Default val 2 means the detected peak must be one half the magnitude of the maximum peak. 
distance_val = 5 # The minimum distance required between individual peaks, unitless. Must be greater or equal to 1. 
width_val = 2 # The minumum width of an individual peak at the base, unitless
prominence_val = 10 # optional, the prominence required for an individual peak, unitless
bankfull = 'yes' # *In development*. 'yes' or 'no', whether to return a bankfull estimate only (yes) or return all major inflections (no)

# Specify input data file paths in correct input folder directories. This pulls from a root directory with subfolders of
# input files and then creates a dataframe for the inputs.
root_dir = Path(r'/Users/ahurst/Documents/USU/Benchmark_sites_from_SNR')

records = []

for reach_dir in root_dir.iterdir():

    if not reach_dir.is_dir():
        continue

    station_id = reach_dir.name

    dem_dir = reach_dir / 'dem'
    thalweg_dir = reach_dir / 'Thalweg'
    xs_dir = reach_dir / 'Cross_Sections'

    if not dem_dir.exists():
        print(f"Missing DEM folder: {station_id}")
        continue

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

    #Find thalwegs
    thalwegs = list(thalweg_dir.glob("*.shp"))

    for tw in thalwegs:
        tw_name = tw.stem

        #Find matching cross-sections

        xs_match = list(xs_dir.glob(f"{tw_name}_xs.shp"))

        if len(xs_match) == 0:
            print(f"No cross-section found for {tw_name} in {station_id}")
            continue

        records.append({
            "station_ids": f"{station_id}_{tw_name}",
            "dems": dem_fp,
            "thalwegs": str(tw),
            "cross-sections": str(xs_match[0])
        })

inputs_ls = pd.DataFrame(records)
bad_sites = []

# station_id = 'Miranda'
# dem_fp = '/Users/ahurst/Downloads/INFLECT_inputs/Miranda/dem/miranda_1m_bathy.tif'
# thalweg_fp = ['/Users/ahurst/Downloads/INFLECT_inputs/Miranda/thalweg/thalweg.shp']
# cross_section_fp = ['/Users/ahurst/Downloads/INFLECT_inputs/Miranda/cross-sections/Thalweg_10m_adjusted.shp']
# inputs_ls = pd.DataFrame({'station_ids':station_id, 'dems':dem_fp, 'thalwegs':thalweg_fp, 'cross-sections':cross_section_fp})
# inputs_ls = inputs_ls.reset_index()

for index, row in inputs_ls.iterrows():
    # Start tracking execution time
    start_time = time.time()
    reach_name = row['station_ids']

    try:

        # Create output folders if needed, delete old files in output folders if they exist.
        if not os.path.exists('data_outputs/{}'.format(reach_name)):
            os.makedirs('data_outputs/{}'.format(reach_name))
        if not os.path.exists('data_outputs/{}/transect_plots'.format(reach_name)):
            os.makedirs('data_outputs/{}/transect_plots'.format(reach_name))
        else:
            files = glob.glob('data_outputs/{}/transect_plots/*'.format(reach_name))
            for f in files:
                os.remove(f)
        if not os.path.exists('data_outputs/{}/second_order_roc'.format(reach_name)):
            os.makedirs('data_outputs/{}/second_order_roc'.format(reach_name))
        else:
            files = glob.glob('data_outputs/{}/second_order_roc/*'.format(reach_name))
            for f in files:
                os.remove(f)
        if not os.path.exists('data_outputs/{}/all_widths'.format(reach_name)):
            os.makedirs('data_outputs/{}/all_widths'.format(reach_name))
        else:        
            files = glob.glob('data_outputs/{}/all_widths/*'.format(reach_name))  
            for f in files:
                os.remove(f)  
        if not os.path.exists('data_outputs/{}/spatial'.format(reach_name)):
            os.makedirs('data_outputs/{}/spatial'.format(reach_name))

        print('Running INFLECT for station: {}'.format(reach_name))
        # Read in data: thalweg, and cross-sections 
        thalweg = gpd.read_file(row['thalwegs'])
        cross_sections = gpd.read_file(row['cross-sections'])
        dem_fp = row['dems']
        all_widths_df = calc_dwdh(reach_name, cross_sections, dem_fp, sampling_interval, d_interval, width_calc_method) # calc widths array for each cross-section
        print('width calcs done!')
        inflect(reach_name, inflect_calc_method, d_interval, all_widths_df, slope_window, max_peak_ratio, distance_val, width_val, prominence_val)
        # Calculate and save execution time in minutes
        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"\nExecution completed in {execution_time_minutes:.2f} minutes")
        output_record(reach_name, slope_window, d_interval, sampling_interval, width_calc_method, units)
        
        # Plotting functions
        print('Generating visualizations...')
        plot_longitudinal_profile(reach_name)
        plot_bankfull_increments(reach_name, d_interval)
        transect_plot(cross_sections, dem_fp, sampling_interval, d_interval, reach_name)
        plot_inflections(d_interval, reach_name)
        # Spatial analysis
        create_bankfull_pts(cross_sections, dem_fp, thalweg, d_interval, spatial_plot_interval, reach_name)

    except ValueError as e:
        print(f"\n FAILED: {reach_name}")
        print(f"Reason: {e}\n")
        bad_sites.append((reach_name, str(e)))
        continue

print("\n FAILED SITES ")
for site, err in bad_sites:
    print(f"{site}: {err}")
