import pandas as pd
import numpy as np
import math
import os
import glob
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
from sklearn.linear_model import LinearRegression
import raster_footprint
from scipy.signal import find_peaks
import seaborn as sns
import rasterio
from rasterio import features
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import shape
import numpy as np

def get_x_vals(y_vals, d_interval):
        x_len = round(len(y_vals) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        return(x_vals)

def get_raster_boundary(dem_fp):
    with rasterio.open(dem_fp) as src:
        # Read at lower resolution, e.g. 1/100th (adjust as needed)
        scale = 100
        new_height = src.height // scale
        new_width = src.width // scale
        data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.nearest
        )
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        # Build mask: True where data != nodata
        if src.nodata is not None:
            mask = data != src.nodata
        else:
            mask = ~np.isnan(data)
        # Polygonize valid area
        shapes = features.shapes(mask.astype("uint8"), mask=mask, transform=transform)
        polygons = [shape(geom) for geom, val in shapes if val == 1]
    footprint = unary_union(polygons)
    gdf = gpd.GeoDataFrame(geometry=[footprint], crs=src.crs)
    return gdf
    
def multipoint_slope(windowsize, timeseries, xvals):
    dw = np.zeros(len(timeseries))
    lr_window = int(windowsize/2) # indexing later requires this to be an integer
    for n in range(lr_window, len(timeseries) - lr_window):
        x = xvals[n - lr_window:n + lr_window]
        y = timeseries[n - lr_window:n + lr_window]
        # remove nans with a mask, if there are at least two real data points
        nancount = sum(1 for x in y if isinstance(x, float) and math.isnan(x))
        if nancount < 3:
            mask = ~np.isnan(x) & ~np.isnan(y)
            slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x[mask], np.array(y)[mask])
        else: 
            slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x, np.array(y))
        dw[n] = slope1  
    return dw   
    
def find_boundary(xsection, bound):
    for index, val in enumerate(xsection):
        if val > bound: # find first instance of exceeding lower width bound
            bound_index = index
            break
    # if upper bound not exceeded, set to last width index of cross section
    bound_index = index 
    return bound_index

def calc_dwdh(reach_name, cross_sections, dem_fp, plot_interval, d_interval, width_calc_method):
    # Loop through xsections and create dw/dh array for each xsection
    all_widths_df = pd.DataFrame(columns=['widths']) # df to store width arrays 
    incomplete_intersection_counter = 0
    total_measurements = 0

    # for cross_section in cross_sections:
    for cross_sections_index, cross_sections_row in cross_sections.iterrows():
        wh_ls = []
        line = gpd.GeoDataFrame({'geometry': [cross_sections_row['geometry']]}, crs=cross_sections.crs) 
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = cross_sections_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        
        # Extract z elevation at each station along transect
        with rasterio.open(dem_fp) as src:
            # sample() yields pixel values at the given coordinates
            elevs = list(src.sample([(point.x, point.y) for point in stations]))

        # remove elevs that were sampled in nodata zone of raster (vals are > 3.4e38)
        elevs = [elev for elev in elevs if elev < 3e38]

        # Determine total depth of iterations based on max rise on the lower bank
        min_z = min(elevs)
        min_y = list(elevs).index(min_z)
        max_left_bank = max(list(elevs)[0:min_y])
        max_right_bank = max(list(elevs)[min_y:])

        if max_right_bank < max_left_bank:
            max_depth = max_right_bank
        else:
            max_depth = max_left_bank

        # Shorten cross_sections if necessary to only include rising banks. Will not affect max depth. 
        while elevs[0] < max_left_bank:
            elevs = elevs[1:] # remove left-most point if it is below maximum left bank elevation (left bank drop off)
        while elevs[-1] < max_right_bank:# remove right-most point if it is below maximum right bank elevation (right bank drop off)
            elevs = elevs[:-1]

        depths = max_depth//d_interval
        # If depth is improperly assigned skip to next cross-section
        if depths[0] == float('inf'):
            continue
        
        # calc width at the current depth 
        if width_calc_method == 'partial':
            for index, depth in enumerate(range(int(depths[0]))):
                total_measurements += 1
                # find intercepts of current d with bed profile (as locs where normalized profile pts have a sign change)
                wat_level = [x - (d_interval * index) for x in elevs]
                intercepts = []
                for i, val in enumerate(elevs[0:-1]):
                    if np.sign(wat_level[i] * wat_level[i + 1]) < 0:
                        intercepts.append(distances[i])
                
                # Find distances between intercept points
                if len(intercepts) == 2: # most common case, one intercept on each side of channel
                    width = intercepts[1] - intercepts[0]
                elif (len(intercepts) % 2) == 0: # other common case, bed elevation has at least one extra pair of intercepts
                    partial_widths = []
                    for int_index in range(0, len(intercepts), 2):
                        w = intercepts[int_index + 1] - intercepts[int_index]
                        partial_widths.append(w)
                    width = sum(partial_widths)
                elif len(intercepts) == 3:
                    left_bank = min(intercepts)
                    right_bank = max(intercepts)
                    width = right_bank - left_bank
                else:
                    # print("Cannot accurately determine width with incomplete xsection intersection. Num intersections = {}.".format(len(intercepts)))
                    width = np.nan
                    incomplete_intersection_counter += 1 
                wh_ls.append(width)

        elif width_calc_method == 'continuous':
            thalweg = min(elevs) 
            thalweg_index = elevs.index(thalweg)
            for index, depth in enumerate(range(int(depths[0]))):
                total_measurements += 1
                # find intercepts of current d with bed profile (as locs where normalized profile pts have a sign change)
                wat_level = [x - (d_interval * index) for x in elevs]
                intercepts = []
                for i, val in enumerate(elevs[0:-1]):
                    if np.sign(wat_level[i] * wat_level[i + 1]) < 0:
                        intercepts.append(distances[i])
                
                # Find distances between intercept points
                width = np.nan
                if len(intercepts) == 0: 
                    width = 0
                elif len(intercepts) == 2: # most common case, one intercept on each side of channel
                    width = intercepts[1] - intercepts[0]
                elif (len(intercepts) % 2) == 0: # other common case, bed elevation has at least one extra pair of intercepts
                    # identify which pair of intercepts contains thalweg. select that pair for width calc. 
                    for int_index in range(0, len(intercepts), 2): # loop through each pair of intercepts
                        if intercepts[int_index] <= thalweg_index <= intercepts[int_index + 1]:
                            # found the intercepts pair that contains the thalweg index
                            width = intercepts[int_index + 1] - intercepts[int_index]
                            break
                elif len(intercepts) == 3: # This is assumed to mean that left and right banks are separated by a single point
                    left_bank = min(intercepts)
                    right_bank = max(intercepts)
                    width = right_bank - left_bank
                else:
                    print("Cannot accurately determine width with incomplete xsection intersection at cross-section {}. Num intersections = {}.".format(transects_index, len(intercepts)))
                    width = np.nan
                    incomplete_intersection_counter += 1 
                wh_ls.append(width)

        thalweg = min(elevs) # track this for use later in detrending
        # find station coordinates at thalweg
        if cross_sections_index > 0: # measure distances for all but first (most upstream) transect
            thalweg_index = elevs.index(thalweg)
            thalweg_coords = stations[thalweg_index]
            # Get distance from thalweg to next thalweg
            next_transect = cross_sections.iloc[cross_sections_index - 1]
            next_line = gpd.GeoDataFrame({'geometry': [next_transect['geometry']]}, crs=cross_sections.crs)
            next_tot_len = next_line.length
            next_distances = np.arange(0, next_tot_len[0], plot_interval) 
            next_stations = next_transect['geometry'].interpolate(next_distances) # specify stations in transect based on plotting interval
            next_stations = gpd.GeoDataFrame(geometry=next_stations, crs=cross_sections.crs)
            with rasterio.open(dem_fp) as src:
                next_elevs = list(src.sample([(point.x, point.y) for point in next_stations.geometry]))
            next_thalweg = min(next_elevs)
            next_thalweg_index = next_elevs.index(next_thalweg)
            next_thalweg_coords = next_stations.geometry[next_thalweg_index]
            thalweg_distance = next_thalweg_coords.distance(thalweg_coords) # distance to next thalweg, in meters
        else: 
            thalweg_distance = 0

        wh_ls_df = pd.DataFrame({'widths':wh_ls})
        wh_ls_df.to_csv('data_outputs/{}/all_widths/widths_{}.csv'.format(reach_name, cross_sections_index))
        wh_append = pd.DataFrame({'widths':[wh_ls], 'transect_id':cross_sections_index, 'thalweg_elev':thalweg, 'thalweg_distance':thalweg_distance})
        all_widths_df = pd.concat([all_widths_df, wh_append], ignore_index=True)

    all_widths_df.to_csv('data_outputs/{}/all_widths.csv'.format(reach_name))
    return(all_widths_df)
                                
def inflect(reach_name, inflect_calc_method, d_interval, all_widths_df, slope_window, max_peak_ratio, distance_val, width_val, prominence_val):
    # Function for identifying top inflection point peaks
    def top_peaks_id(peaks_array, num_peaks):
        if len(peaks_array[0]) < num_peaks:
            peak_range = len(peaks_array[0])
        else: 
            peak_range = num_peaks
        peak_indices = peaks_array[0]
        max_peaks = []
        for i in range(0, peak_range): # Here is where to define number of peaks looking for 
            current_max = 0 
            current_max_index = 0
            for j in range(len(peak_indices)):
                if abs(peaks_array[1]['peak_heights'][j]) > current_max:
                    current_max = abs(peaks_array[1]['peak_heights'][j])
                    current_max_index = j
            peaks_array[1]['peak_heights'] = np.delete(peaks_array[1]['peak_heights'], current_max_index)
            max_peaks.append(peak_indices[current_max_index])
            peak_indices = np.delete(peak_indices, current_max_index)
        return max_peaks

    # Use thalweg elevs to detrend 2nd derivatives. Don't remove intercept (keep at elevation) 
    x = np.cumsum(all_widths_df['thalweg_distance'].values).reshape((-1,1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]

    if inflect_calc_method == 'cross-section':
        '''
        Inflection Point Methodology: Calc inflections from each cross-section then take the average of results. 
        '''
        # Calculate 2nd derivatives to get inflections of width arrays
        ddw_ls = []
        for x_index, xsection in enumerate(all_widths_df['widths']): # loop through all x-sections
            dw = []
            ddw = []
            xs_xvals = get_x_vals(xsection, d_interval)
            dw = multipoint_slope(slope_window, xsection, xs_xvals)
            ddw = multipoint_slope(slope_window, dw, xs_xvals)
            pd.DataFrame({'ddw':ddw}).to_csv('data_outputs/{}/second_order_roc/{}_roc.csv'.format(reach_name, x_index))
            ddw_ls.append(ddw)

        inflections_ls = []
        for index, inflections in enumerate(ddw_ls):
            # Incorporate detrend as a shift in 2nd derivative array
            # Should be raising all values after first transect. So they start later. 
            if not isinstance(inflections, list):
                inflections = inflections.tolist()
            offset = fit_slope[index]
            offset = offset / d_interval
            offset_int = int(offset)
            if offset_int < 0:
                inflections = [0] * abs(offset_int) + inflections
            else: # Only other case is no detrend (first transect)
                inflections = inflections
            inflections_ls.append(inflections)
            
        # convert list of lists into dataframe
        inflections_df = pd.DataFrame(inflections_ls)
        n_xs = len(inflections_df.index)
        inflections_df = inflections_df.dropna(axis=1, thresh=n_xs * 0.5) # drop columns with less than 50% of values present

        # Aggregate all arrays together by averaging across all rows in df
        inflections_array = inflections_df.mean(axis=0, skipna=True)

        # identify top three peaks (across positive and negative)
        peaks_pos = find_peaks(inflections_array, height=max(inflections_array)/max_peak_ratio, distance=distance_val, width=width_val) #, prominence=prominence_val) # require peaks to be at least half the mag of max peak
        inflections_array_neg = [-i for i in inflections_array] # invert all signs to detect negative peaks
        peaks_neg = find_peaks(inflections_array_neg, height=max(inflections_array_neg)/max_peak_ratio, distance=distance_val, width=width_val) #, prominence=prominence_val) # require peaks to be at least half the mag of max peak
        # save peak locs for plotting along wd and cross_sections
        # ID top 3 peaks in each category - positive
        max_pos_peak = top_peaks_id(peaks_pos, 3)
        # ID top 3 peaks in each category - negative
        max_neg_peak = top_peaks_id(peaks_neg, 3)
        
        # Save max positive and negative inflections (bankfull range)
        max_len = max(len(max_pos_peak), len(max_neg_peak))
        pos_peak_indices_pad = max_pos_peak + [np.nan] * (max_len - len(max_pos_peak))
        neg_peak_indices_pad = max_neg_peak + [np.nan] * (max_len - len(max_neg_peak))
        max_inflections_df = pd.DataFrame({'pos_inflections':pos_peak_indices_pad, 'neg_inflections':neg_peak_indices_pad})
        max_inflections_df.to_csv('data_outputs/{}/max_inflections.csv'.format(reach_name))
        inflections_array.to_csv('data_outputs/{}/inflections_array.csv'.format(reach_name), index=False)
        # there it is! 

    if inflect_calc_method == 'aggregate':    
        '''
        Alt inflection point method: calc inflection points from aggregated width/depth curve
        '''
        # Detrended, aggregated cross_sections using padded-zeros approach
        all_widths_df['widths_detrend'] = [[] for _ in range(len(all_widths_df))] 
        # Loop through all_widths
        for index, row in all_widths_df.iterrows():
            offset = fit_slope[index]
            offset = offset / d_interval
            offset_int = int(offset)
            if offset_int < 0: # most likely case, downstream xsections are lower elevation than furthest upstream
                # populate new column of df with width values
                all_widths_df.loc[index, 'widths_detrend'].extend([0] * abs(offset_int) + row['widths']) # add zeros to beginning of widths list. Need to unnest when using.
            elif offset_int > 0: # this probably won't come up
                all_widths_df.loc[index, 'widths_detrend'].extend(row[abs(offset_int):])
            else:
                all_widths_df.loc[index, 'widths_detrend'].extend(row['widths'])
        # Once all offsets applied, use zero-padding aggregation method just like with non-detrended widths.
        n_xs = len(all_widths_df.index) # number of cross_sections to use when applying requirements for number of cross_sections in aggregation
        max_len = max(all_widths_df['widths_detrend'].apply(len)) # find the longest row in df
        all_widths_df['widths_padded_detrend'] = all_widths_df['widths_detrend'].apply(lambda x: np.pad(x, (0, max_len - len(x)), constant_values=np.nan)) # pad all shorter rows with nan
        padded_df_detrend = pd.DataFrame(all_widths_df['widths_padded_detrend'].tolist())
        # drop columns element-wise in which more than half of values are nan
        padded_df_detrend = padded_df_detrend.dropna(axis=1, thresh=n_xs * 0.5) # drop columns with less than 50% of values present
        # calculate transect_50_detrend as the median of each column
        # this is the aggregate cross-section for the reach
        transect_50_detrend = padded_df_detrend.apply(lambda row: np.nanpercentile(row, 50), axis=0)
        
        # take second derivative of transect_50_detrend
        xvals_agg = get_x_vals(transect_50_detrend, d_interval)
        dw = multipoint_slope(slope_window, transect_50_detrend, xvals_agg)
        ddw = multipoint_slope(slope_window, dw, xvals_agg)
        # use inflection pt method to find top pos/neg peaks
        inflections_array_agg = ddw
        peaks_pos_agg = find_peaks(inflections_array_agg, height=max(inflections_array_agg)/max_peak_ratio, distance=distance_val, width=width_val, prominence=prominence_val) # , prominence=20) # require peaks to be at least half the mag of max peak
        inflections_array_neg_agg = [-i for i in inflections_array_agg] # invert all signs to detect negative peaks
        peaks_neg_agg = find_peaks(inflections_array_neg_agg, height=max(inflections_array_neg_agg)/max_peak_ratio, distance=distance_val, width=width_val, prominence=prominence_val) # prominence=20) # require peaks to be at least half the mag of max peak
        # ID top 3 peaks in each category - positive
        max_pos_peak_agg = top_peaks_id(peaks_pos_agg, 3)
        # ID top 3 peaks in each category - negative
        max_neg_peak_agg = top_peaks_id(peaks_neg_agg, 3)

        # Save values and plot results
        max_len_agg = max(len(peaks_pos_agg[0]), len(peaks_neg_agg[0]))
        pos_peak_indices_pad_agg = max_pos_peak_agg + [np.nan] * (max_len_agg - len(max_pos_peak_agg))
        neg_peak_indices_pad_agg = max_neg_peak_agg + [np.nan] * (max_len_agg - len(max_neg_peak_agg))
        pd.DataFrame(inflections_array_agg).to_csv('data_outputs/{}/inflections_array_alt.csv'.format(reach_name), index=False)
        max_inflections_df_agg = pd.DataFrame({'pos_inflections':pos_peak_indices_pad_agg, 'neg_inflections':neg_peak_indices_pad_agg})
        max_inflections_df_agg.to_csv('data_outputs/{}/max_inflections_alt.csv'.format(reach_name))

        # Determine x-vals for plotting
        x_range = range(0, len(inflections_array_agg))
        x_vals = list(x_range)
        x_vals = [i * d_interval - intercept for i in x_vals]
        fig, ax = plt.subplots()
        plt.plot(x_vals, inflections_array_agg, color='black')
        plt.xlim(left=-5)

        for index, peak in enumerate(max_pos_peak_agg):
            if index == 0:
                plt.axvline(peak/10 - intercept, color='red', label='positive inflections')
            else:
                plt.axvline(peak/10 - intercept, color='red')
        for index, peak in enumerate(max_neg_peak_agg):
            if index == 0:
                plt.axvline(peak/10 - intercept, color='blue', label='negative inflections')
            else:
                plt.axvline(peak/10 - intercept, color='blue')
        
        plt.title('Inflection Points Density, Aggregate Method')
        plt.xlabel('Detrended elevation (m)')
        plt.ylabel('Second derivative')
        # plt.text(.5, .5, str(round(max(inflections_array), 3)))
        plt.legend()
        plt.savefig('data_outputs/{}/inflection_pt_density_plot_agg.jpeg'.format(reach_name))
        plt.close()
    
    
    