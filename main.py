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
from analysis import calc_dwdh, inflect, get_raster_boundary
from visualization import output_record, plot_bankfull_increments, plot_longitudinal_profile, transect_plot, plot_inflections
from spatial_analysis import create_bankfull_pts

# Steps for bankfull analysis:
# 1. Identify benchmark bankfull using inundation rasters (Analysis.py -> id_benchmark_bankfull)
# 2. Measure channel width along a depth interval for each cross-section (Analysis.py -> calc_dwdh)
# 3. Calculate first and second order derivatives of the channel widths to identify topographic bankfull (Analysis.py -> calc_derivatives)
# 4. Post-processing: plot results (Visualization.py -> plot_bankfull_increments, plot_longitudinal_bf)



# Set INFLECT algorithm parameters
plot_interval = 1 # set plotting interval along transect in units of meters
d_interval = 10/100 # Set intervals to step up in depth (in units meters). Default is 10cm intervals
slope_window = 10 # Set window size for calculating slope for derivatives
lower_bound = 5 # Set lower vertical boundary for inflection id within cross-section, in units of d_interval. Default 5 = 50cm
upper_bound = 100 # Set upper vertical boundary for inflection id within cross-section, in units of d_interval. Default 100 = 10m
spatial_plot_interval = 0.5 # interval for finding inflection elevation along cross_sections.
width_calc_method = 'partial' # 'continuous' 'partial' - choose from either partial additive widths or continuous-only methods of width calculation.
# peak detection parameters
inflect_calc_method = 'cross-section' # 'cross-section' or 'aggregate' method for inflection calculation. 'cross-section' is default. 
max_peak_ratio = 2 # The ratio of max peak:detected peak. Default val 2 means the detected peak must be one half the magnitude of the maximum peak. 
distance_val = 5 # The minimum distance required between individual peaks
width_val = 2 # The minumum width of an individual peak at the base
prominence_val = 20 # optional, the prominence required for an individual peak

# Specify input data file paths in correct input folder directories
dem_fp = 'data_inputs/dem/USGS_1M_Klamath_merged/USGS_1M_Klamath_merged.tif' # file in 'data_inputs/dem/...' folder
thalweg_fp = ['data_inputs/klamath_INFLECT_inputs/reach_18010206000027/18010206000027_line.shp']
cross_sections_fp = ['data_inputs/klamath_INFLECT_inputs/reach_18010206000027/18010206000027_transects.shp']


inputs_ls = pd.DataFrame({'thalwegs':thalweg_fp, 'cross-sections':cross_sections_fp})
inputs_ls = inputs_ls.reset_index()

for index, row in inputs_ls.iterrows():
    # get reach name from thalweg file path third element when split by'/'  
    # reach_name = row['thalwegs'].split('/')[2]
    reach_name = 'example'

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

    all_widths_df = calc_dwdh(reach_name, cross_sections, dem_fp, plot_interval, d_interval, width_calc_method) # calc widths array for each cross-section
    inflect(reach_name, inflect_calc_method, d_interval, all_widths_df, slope_window, max_peak_ratio, distance_val, width_val, prominence_val)
    output_record(reach_name, slope_window, d_interval, lower_bound, upper_bound, width_calc_method)

    # Plotting functions
    plot_longitudinal_profile(reach_name, dem_fp, cross_sections, plot_interval)
    plot_bankfull_increments(reach_name, d_interval)
    transect_plot(cross_sections, dem_fp, plot_interval, d_interval, reach_name)
    plot_inflections(d_interval, reach_name)
    # Spatial analysis
    create_bankfull_pts(cross_sections, dem_fp, thalweg, d_interval, spatial_plot_interval, reach_name)