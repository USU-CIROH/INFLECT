import pandas as pd
import numpy as np
from numpy import nan
import re
import glob
import geopandas as gpd
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_longitudinal_profile(reach_name, dem, cross_sections, plot_interval):
    # Extract and detrend thalweg for plotting
    thalweg_distances = []
    thalweg_line = []
    for cross_sections_index, cross_sections_row in cross_sections.iterrows():
        line = gpd.GeoDataFrame({'geometry': [cross_sections_row['geometry']]}, crs=cross_sections.crs) 
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = cross_sections_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=cross_sections.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))

        thalweg = min(elevs) # track this for use later in detrending
        thalweg_line.append(thalweg)
        # find station coordinates at thalweg
        if cross_sections_index > 0: # measure distances for all but first (most upstream) transect
            thalweg_index = elevs.index(thalweg)
            thalweg_coords = stations.geometry[thalweg_index]
            # Get distance from thalweg to next thalweg
            next_transect = cross_sections.iloc[cross_sections_index - 1]
            next_line = gpd.GeoDataFrame({'geometry': [next_transect['geometry']]}, crs=cross_sections.crs)
            next_tot_len = next_line.length
            next_distances = np.arange(0, next_tot_len[0], plot_interval) 
            next_stations = next_transect['geometry'].interpolate(next_distances) # specify stations in transect based on plotting interval
            next_stations = gpd.GeoDataFrame(geometry=next_stations, crs=cross_sections.crs)
            next_elevs = list(dem.sample([(point.x, point.y) for point in next_stations.geometry]))
            next_thalweg = min(next_elevs)
            next_thalweg_index = next_elevs.index(next_thalweg)
            next_thalweg_coords = next_stations.geometry[next_thalweg_index]
            thalweg_distance = next_thalweg_coords.distance(thalweg_coords) # distance to next thalweg, in meters
        else: 
            thalweg_distance = 0
        if cross_sections_index == 0:
            thalweg_distances.append(thalweg_distance)
        else:
            thalweg_distances.append(thalweg_distance + thalweg_distances[cross_sections_index-1])

    thalweg_detrend = []
    x_vals_thalweg = np.arange(0, len(thalweg_line))
    x = np.array(x_vals_thalweg).reshape(-1, 1)
    y = np.array(thalweg_line)
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope =  slope*x
    fit_slope = [val[0] for val in fit_slope]
    # pairwise subtract fit from thalwegs
    for index, val in enumerate(thalweg_line):
        thalweg_detrend.append(val - fit_slope[index])
    # Plot logitudinal profile
    # breakpoint()
    fig, ax = plt.subplots()
    plt.xlabel('Cross-sections from upstream to downstream (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Logitudinal profile, {}'.format(reach_name))
    plt.plot(thalweg_distances, thalweg_line, color='grey', label='Thalweg')
    plt.plot(thalweg_distances, fit_slope + intercept, linestyle='--', color='black', label='Linear detrend')
    plt.legend(loc='upper right')
    plt.savefig('data_outputs/{}/Longitudinal_profile'.format(reach_name))
    plt.close()

def plot_bankfull_increments(reach_name, d_interval):
    all_widths_df = pd.read_csv('data_outputs/{}/all_widths.csv'.format(reach_name))
    for index, row in all_widths_df.iterrows():
        all_widths_df.at[index, 'widths'] = eval(row['widths'])

    # Detrend widths before plotting based on thalweg elevation, and start plotting point based on detrend
    x = np.array(all_widths_df['transect_id']).reshape((-1, 1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope] # unnest the array
    
    # Create color ramp 
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=len(all_widths_df)-1)
    # Plot all widths spaghetti style
    fig, ax = plt.subplots()
    plt.ylabel('Channel width (m)')
    plt.xlabel('Detrended elevation (m)')
    plt.title('Incremental channel widths for {}'.format(reach_name))
    # run loop to find where to set left limit on plot
    left_lims = []
    for index, val in enumerate(all_widths_df.iterrows()):
        for i, v in enumerate(val[1]['widths']):
            if v != 0: # mark first time inflection line exceeds zero (i.e. begins)
                left_lims.append(i)  
                break
    left_lim = min(left_lims)

    for index, row in all_widths_df.iterrows(): 
        row = row['widths']
        x_len = round(len(row) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        # apply detrend shift to xvals
        x_vals = [x_val - fit_slope[index] for x_val in x_vals] # - intercept, optionally, to center at zero
        plt.plot(x_vals, row, alpha=0.3, color=cmap(norm(index)), linewidth=0.75) 
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Set array to avoid warnings
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Downstream distance (m)")
    plt.xlim(left=left_lim*d_interval)
    plt.savefig('data_outputs/{}/all_widths.jpeg'.format(reach_name), dpi=400)
    plt.close()

def transect_plot(cross_sections, dem, plot_interval, d_interval, reach_name):
    # topo_bankfull = pd.read_csv('data/data_outputs/{}/bankfull_topo.csv'.format(reach_name))
    inflections = pd.read_csv('data_outputs/{}/max_inflections_aggregate.csv'.format(reach_name))
    all_widths_df = pd.read_csv('data_outputs/{}/all_widths.csv'.format(reach_name))

    # Use thalweg to detrend elevation on y-axes for transect plotting. Don't remove intercept (keep at elevation) 
    x = np.array(all_widths_df['transect_id']).reshape((-1, 1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]
    
    # for cross_section in cross_sections:
    for transects_index, transects_row in cross_sections.iterrows():
        line = gpd.GeoDataFrame({'geometry': [transects_row['geometry']]}, crs=cross_sections.crs) 
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = transects_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=cross_sections.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))

        # Arrange points together for plotting
        def get_x_vals(y_vals):
            x_len = round(len(y_vals) * d_interval, 4)
            x_vals = np.arange(0, x_len, d_interval)
            return(x_vals)
        fig = plt.figure(figsize=(6,8))
        plt.plot(distances, elevs, color='black', linestyle='-', label='Cross section')
        for index, x in enumerate(inflections['pos_inflections']):
            if index == 0:
                plt.axhline(x*d_interval, color='red', label='positive inflections', linewidth=2)
            else:
                plt.axhline(x*d_interval, color='red', linewidth=2)
        for index, x in enumerate(inflections['neg_inflections']):
            if index == 0:
                plt.axhline(x*d_interval, color='blue', label='negative inflections', linewidth=2)
            else:
                plt.axhline(x*d_interval, color='blue', linewidth=2)

        plt.xlabel('Cross section distance (meters)', fontsize=16)
        plt.ylabel('Elevation (meters)', fontsize=16)
        plt.legend(fontsize=16)
        # increase font size for axes and labels
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig('data_outputs/{}/transect_plots/bankfull_transect_{}.jpeg'.format(reach_name, transects_index))
        plt.close()

def plot_inflections(d_interval, reach_name):
    all_widths_df = pd.read_csv('data_outputs/{}/all_widths.csv'.format(reach_name))
    inflections = pd.read_csv('data_outputs/{}/max_inflections.csv'.format(reach_name))
    def get_x_vals(y_vals, d_interval):
        x_len = round(len(y_vals) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        return(x_vals)
    
    # bring in 2nd derivative files
    inflections_fp = glob.glob('data_outputs/{}/second_order_roc/*'.format(reach_name))
    # Sort the inflections files numerically
    def extract_num(path):
        match = re.search(r'\d+', path)
        return int(match.group()) if match else np.nan 
    inflections_fp_sorted = sorted(inflections_fp, key=extract_num)
    # bring in aggregated inflections array for plotting
    inflections_array_agg = pd.read_csv('data_outputs/{}/inflections_array_agg.csv'.format(reach_name))
    # Use thalweg elevs to detrend 2nd derivatives. Don't remove intercept (keep at elevation) 
    x = np.cumsum(all_widths_df['thalweg_distance'].values).reshape((-1,1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]
    # Set up plot and create color ramp 
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=len(inflections_fp_sorted)-1)
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.ylabel(r'Inflection magnitude $(1/m)$')
    plt.xlabel('Detrended elevation (m)')
    plt.title('Cross section width inflections for {}'.format(reach_name))
    ax.tick_params(axis='both', which='major')
    # Set x-axis labels to show only integer values
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    # loop through files and get left limit for plot
    left_lims = []
    for index, inflection_fp in enumerate(inflections_fp_sorted): 
        inflection = pd.read_csv(inflection_fp)
        inflection = inflection['ddw']
        for i, val in enumerate(inflection):
            if abs(val) > 0:
                left_lims.append(i)
                break
    left_lim = min(left_lims)
    # loop through files and plot
    for index, inflection_fp in enumerate(inflections_fp_sorted): 
        inflection = pd.read_csv(inflection_fp)
        inflection = inflection['ddw']
        # detrend inflections so they all plot at the same starting point
        offset = fit_slope[index]
        offset = offset / d_interval
        offset_int = int(offset)
        if offset_int < 0:
            inflection = [0] * abs(offset_int) + inflection.tolist() 
        else: # Only other case is no detrend (first transect)
            inflection = inflection
        # plot all inflections spaghetti style
        x_vals = get_x_vals(inflection, d_interval)
        plt.plot(x_vals, inflection, alpha=0.5, color=cmap(norm(index)), linewidth=1.25) 
        # run loop to find where to set left limit on plot
        for index, val in enumerate(inflection):
            if val != 0: # mark first time inflection line exceeds zero (i.e. begins)
               left_lims.append(index)  
               break
    for index, x in enumerate(inflections['pos_inflections']):
        if index == 0:
            plt.axvline(x*d_interval, color='red', label='positive inflections', linewidth=2)
        else:
            plt.axvline(x*d_interval, color='red', linewidth=2)
    for index, x in enumerate(inflections['neg_inflections']):
        if index == 0:
            plt.axvline(x*d_interval, color='blue', label='negative inflections', linewidth=2)
        else:
            plt.axvline(x*d_interval, color='blue', linewidth=2)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Set array to avoid warnings
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Downstream distance (m)")
    # overlay aggregate inflections
    x_vals_overlay = get_x_vals(inflections_array_agg, d_interval)
    plt.plot(x_vals_overlay, inflections_array_agg, color='black', linewidth=1.5)
    # set plot xlim as range of x_vals_overlay
    plt.xlim(left=left_lim * d_interval, right=x_vals_overlay[-1])
    plt.tight_layout()
    plt.savefig('data_outputs/{}/inflections_all.jpeg'.format(reach_name))
    return

def output_record(reach_name, slope_window, d_interval, lower_bound, upper_bound, width_calc_method):
    inflections = pd.read_csv('data_outputs/{}/max_inflections.csv'.format(reach_name))
    # Consolidate 'pos_inflections' and 'neg_inflections' columns into single lists
    pos_inflections_all = []
    neg_inflections_all = []
    for val in inflections['pos_inflections']:
        pos_inflections_all.append(val)
    for val in inflections['neg_inflections']:
        neg_inflections_all.append(val)
    record_df = pd.DataFrame({'positive inflections':[pos_inflections_all], 'negative inflections':[neg_inflections_all], \
                              'width calc method':[width_calc_method], 'derivative slope_window': [slope_window], 'width calc interval (m)': [d_interval], 'lower_search_bound': [lower_bound], \
                                'upper_search_bound': [upper_bound]})
    record_df.to_csv('data_outputs/{}/Summary_results.csv'.format(reach_name))
