"""
INFLECT (INFLection-based Elevations from Channel Topography) Method

Primary implementtion script.

Noelle Patterson, USU
Funded by CIROH (Cooperative Institute for Research to Operations) 
2024/2025
"""

import os
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
# import dataretrieval.nwis as nwis
from datetime import datetime
import sys
import time
from analysis import calc_dwdh, inflect, get_raster_boundary
from visualization import output_record, plot_bankfull_increments, plot_longitudinal_profile, transect_plot, plot_inflections
from spatial_analysis import create_bankfull_pts

# Steps for bankfull analysis:
# 1. Identify benchmark bankfull using inundation rasters (Analysis.py -> id_benchmark_bankfull)
# 2. Measure channel width along a depth interval for each cross-section (Analysis.py -> calc_dwdh)
# 3. Calculate first and second order derivatives of the channel widths to identify topographic bankfull (Analysis.py -> calc_derivatives)
# 4. Post-processing: plot results (Visualization.py -> plot_bankfull_increments, plot_longitudinal_bf)

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
distance_val = 5 # The minimum distance required between individual peaks, unitless
width_val = 2 # The minumum width of an individual peak at the base, unitless
prominence_val = 20 # optional, the prominence required for an individual peak, unitless

# Start tracking execution time
start_time = time.time()

# Specify input data file paths in correct input folder directories
dem_fp = 'data_inputs/Iowa/New_Hartford/dem/dem_10m_HAND.tif' # file in 'data_inputs/dem/...' folder
thalweg_fp = ['data_inputs/Iowa/New_Hartford/thalweg/thalweg.shp']
cross_sections_fp = ['data_inputs/Iowa/New_Hartford/cross-sections/cross-sections.shp']

inputs_ls = pd.DataFrame({'thalwegs':thalweg_fp, 'cross-sections':cross_sections_fp})
inputs_ls = inputs_ls.reset_index()

for index, row in inputs_ls.iterrows():
    # get reach name from thalweg file path third element when split by'/'  
    # reach_name = row['thalwegs'].split('/')[2]
    reach_name = 'New_Hartford_HAND'

    # Create output folders if needed
    if not os.path.exists('data_outputs/{}'.format(reach_name)):
        os.makedirs('data_outputs/{}'.format(reach_name))
    if not os.path.exists('data_outputs/{}/transect_plots'.format(reach_name)):
        os.makedirs('data_outputs/{}/transect_plots'.format(reach_name))
    if not os.path.exists('data_outputs/{}/second_order_roc'.format(reach_name)):
        os.makedirs('data_outputs/{}/second_order_roc'.format(reach_name))
    if not os.path.exists('data_outputs/{}/all_widths'.format(reach_name)):
        os.makedirs('data_outputs/{}/all_widths'.format(reach_name))
    if not os.path.exists('data_outputs/{}/spatial'.format(reach_name)):
        os.makedirs('data_outputs/{}/spatial'.format(reach_name))

    # Read in data: thalweg, and cross-sections 
    thalweg = gpd.read_file(row['thalwegs'])
    cross_sections = gpd.read_file(row['cross-sections'])

    all_widths_df = calc_dwdh(reach_name, cross_sections, dem_fp, sampling_interval, d_interval, width_calc_method) # calc widths array for each cross-section
    print('done!')
    inflect(reach_name, inflect_calc_method, d_interval, all_widths_df, slope_window, max_peak_ratio, distance_val, width_val, prominence_val)
    # Calculate and save execution time in minutes
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\nExecution completed in {execution_time_minutes:.2f} minutes")
    output_record(reach_name, slope_window, d_interval, sampling_interval, width_calc_method, units, execution_time_minutes)
    
    # Plotting functions
    plot_longitudinal_profile(reach_name, all_widths_df, dem_fp, cross_sections, sampling_interval)
    plot_bankfull_increments(reach_name, d_interval)
    transect_plot(cross_sections, dem_fp, sampling_interval, d_interval, reach_name)
    plot_inflections(d_interval, reach_name)
    # Spatial analysis
    create_bankfull_pts(cross_sections, dem_fp, thalweg, d_interval, spatial_plot_interval, reach_name)
