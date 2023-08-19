import sys
import os

import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import datetime
from datetime import datetime, timedelta

# This allows us to import tools from the dvm-dos-tem/scripts directory
sys.path.insert(0, '/work/scripts')
import output_utils as ou
from output_utils import load_trsc_dataframe #loads transient and scenario data, columns are layers
from output_utils import stitch_stages

def plot_data(site_folder, run_folder, timeres, output_var, y, x, layer, year, end_year):
    '''
    This will plot UNSYNTHESIZED/UN-INTERPOLATED data, ie; for all layers. 
    loads and plots data on a yearly basis (regardless of time res of data)
    
    site_folder: name of site ex:'MD_deep_soil'
    run_folder: name of folder containing the run ex:'control_allout_monthly_11px'
    timeres: data resolution ex:'monthly'
    output_var: output to plot ex:'TMINEC'
    y: y pixel
    x: x pixel
    layer: layer number to plot for data that is by layer
    year='1995'
    end_year='2030'

    '''
    os.chdir('/data/workflows/'+site_folder+'/'+run_folder)
    if timeres=='monthly':
        df, meta = load_trsc_dataframe(var=output_var, timeres='monthly', px_y=y, px_x=x, fileprefix='output')
        df_yearly = df.resample('Y').mean()
    elif timeres=='yearly':
        df_yearly, meta = load_trsc_dataframe(var=output_var, timeres='yearly', px_y=y, px_x=x, fileprefix='output')
    else: 
        print('Specified Time Resolution not recognized - use monthly or yearly data')
    try:
        plt.plot(df_yearly.loc[year:end_year].index, df_yearly.loc[year:end_year][layer], label=output_var)
    except:
        plt.plot(df_yearly.loc[year:end_year].index, df_yearly.loc[year:end_year], label=output_var)
    plt.xlabel('Year')
    plt.ylabel(output_var+'({})'.format(meta['var_units']))
    return df_yearly

def plot_synth_data(df,output_var, y, x, layer=0,year=1901, end_year=2015):
    '''
    This will plot DEPTH INTERPOLATED output data, ie; use when you have created depth interpolated output files,
    using Output_Synthesis_Valeria.sh
    df,output_var, y, x
    '''
    tlayer = df[output_var]
    time = df['time'][:]
    base_date = datetime(year, month=1, day=1)  
    time = [base_date + timedelta(days=int(t)) for t in time]
    filtered_time = [t for t in time if t.year <= end_year]
    plt.plot(filtered_time, tlayer[:len(filtered_time), y, x, layer])  # Assuming you want to plot the first spatial point (0, 0)
    plt.xlabel('Time (Years)')
    plt.ylabel(output_var)
    return 

def soil_contour(data_path,output_var,res,px,py,output_folder, start_time=None, end_time=None, 
                  layer_start=None, layer_end=None, n=100):
    '''
    This function will plot the soil contour BY LAYER for temperature (or another monhtly output var)
    
    data_path: path to model run folder, ex: data_path='/data/workflows/MD_deep_soil/control_allout_monthly_11px/'
    output_var: variable to plot by depth,
    res: time resolution of data,
    px,py: pixel location
    output_folder: name of folder containing output from model run
    start_time: start time for data selection
    end_time: end time for data selection
    layer_start: starting depth for plotting
    layer_end: ending depth for plottin
    n: number of levels in colorbar
    
    Ex:soil_contour(data_path,'TLAYER','monthly',1,1,'output',cmap='seismic',n=100)
    This is currently plotting by layer - update to use actual depth
    '''    
    #Plot master branch run
    os.chdir(data_path)
    df, meta = load_trsc_dataframe(var=output_var, timeres=res, px_y=py, px_x=px, fileprefix=output_folder)
    layers = df.columns.astype(float)
    times = pd.to_datetime(df.index)
    
    # Filter data based on start_time and end_time
    if start_time is not None and end_time is not None:
        mask = (times >= start_time) & (times <= end_time)
        df = df.loc[mask]
        times = times[mask]
        
    # Filter data based on depth_start and depth_end
    if layer_start is not None and layer_end is not None:
        depth_indices = np.where((layers >= layer_start) & (layers <= layer_end))[0]
        layers = layers[depth_indices]
        df = df.iloc[:, depth_indices]

    layer_mesh, time_mesh = np.meshgrid(layers, times)
    temperature = df.values
    color_axes=max(abs(np.max(temperature)),abs(np.min(temperature))) 
    plt.contourf(time_mesh, layer_mesh, temperature, cmap='seismic', vmin=-color_axes, vmax=color_axes, levels=n)

    # Add color bar
    plt.colorbar(label='Temperature (C)')
    plt.gca().invert_yaxis()

    # Set labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Layer number')

    # Show the plot
#     plt.show()
    
    return

def soil_contourbydepth(data_path,output_var,res,px,py,output_folder, start_time=None, end_time=None, 
                  depth_start=None, depth_end=None, n=100):
    '''
    data_path: path to output data to plot
    output_var: variable to plot (technically, doesn't need to be temperature)
    res: time resolution of output data
    px, py: pixel location of data
    output_folder: name of output folder
    start_time, end_time: start and end year of data to plot
    depth_start, depth_end: starting and ending depth to be plotted (in meters)
    n: levels on the colobar
    '''
    
    os.chdir(data_path)
    df, meta = load_trsc_dataframe(var=output_var, timeres=res, px_y=py, px_x=px, fileprefix=output_folder) 
    df_depth, meta_depth = load_trsc_dataframe(var='LAYERDEPTH',  timeres=res, px_y=py, px_x=px, fileprefix=output_folder)
    df_dz, meta_dzh = load_trsc_dataframe(var='LAYERDZ',  timeres=res, px_y=py, px_x=px, fileprefix=output_folder)
    
    layers = df.columns.astype(float)
    times = pd.to_datetime(df.index)
    
    # Filter data based on start_time and end_time
    if start_time is not None and end_time is not None:
        mask = (times >= start_time) & (times <= end_time)
        df = df.loc[mask]
        times = times[mask]

    # Extract necessary data
    depths = df_depth.iloc[:, :-2].values
    dz = df_dz.iloc[:, :-2].values
    temperature = df.iloc[:, :-2].values
    xp = depths + dz / 2  # Center of each layer, x-coordinates of the data points for interp1d

    # Create a regular grid of depth values
    ii=np.unravel_index(np.argmax(depths), depths.shape)
    maxd=depths.max()+(dz[ii]/2)
    regular_depths = np.arange(0, maxd, 0.01)

    # Interpolate temperature onto the regular grid
    interp_temperature = np.empty((temperature.shape[0], regular_depths.shape[0]))
    for i in range(temperature.shape[0]):
        f = interp1d(xp[i], temperature[i], kind='linear', fill_value='extrapolate')
        interp_temperature[i] = f(regular_depths)

    # Create contour plot
    color_axes = max(np.max(temperature), np.abs(np.min(temperature)))
    vmax=color_axes
    vmin=-vmax
    plt.contourf(times, regular_depths, interp_temperature.T, cmap='seismic', vmin=vmin, vmax=vmax, levels=n)

    # Add colorbar
    plt.colorbar(label='Temperature (C)')

    # Set labels and title
    plt.xlabel('Time')
    plt.ylabel('Depth (m)')

    # Show the plot
    plt.ylim(depth_start, depth_end)
    plt.gca().invert_yaxis()
    return


def layertype_vis(data_path,output_var,res,px,py,output_folder, start_time=None, end_time=None, 
                  layer_start=None, layer_end=None, n=3):
    '''
    data_path: path to model run folder, ex: data_path='/data/workflows/MD_deep_soil/control_allout_monthly_11px/'
    output_var: variable to plot by depth,
    res: time resolution of data,
    px,py: pixel location
    output_folder: name of folder containing output from model run
    start_time: start time for data selection
    end_time: end time for data selection
    layer_start: starting depth for plotting
    layer_end: ending depth for plotting
    n: number of levels in colorbar
    
    Ex:soil_contour(data_path,'TLAYER','monthly',1,1,'output',cmap='seismic',n=100)
    '''    
    #Plot master branch run
    os.chdir(data_path)
    df, meta = load_trsc_dataframe(var=output_var, timeres=res, px_y=py, px_x=px, fileprefix=output_folder)
    df=df.iloc[:,0:-4] #use for plotting layertype instead of temp
    layers = df.columns.astype(float)
    times = pd.to_datetime(df.index)
    
    # Filter data based on start_time and end_time
    if start_time is not None and end_time is not None:
        mask = (times >= start_time) & (times <= end_time)
        df = df.loc[mask]
        times = times[mask]
        
    # Filter data based on depth_start and depth_end
    if layer_start is not None and layer_end is not None:
        depth_indices = np.where((layer_start >= layer_start) & (layers <= layer_end))[0]
        layers = layers[depth_indices]
        df = df.iloc[:, depth_indices]

    depth_mesh, time_mesh = np.meshgrid(depths, times)
    temperature = df.values
    color_axes=max(abs(np.max(temperature)),abs(np.min(temperature))) 
    plt.contourf(time_mesh, depth_mesh, temperature, cmap='seismic', vmin=-color_axes, vmax=color_axes, levels=n)

    # Add color bar
#     plt.colorbar()
    plt.gca().invert_yaxis()

    # Set labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Layer number')
    
    # Show the plot
#     plt.show()
    
    return

def seasonal_profile(VAR, depth, thickness, resolution, time_range, months):
    '''
    VAR : variable dataframe
    depth : associated LAYERDEPTH dataframe
    thickness : associated LAYERDZ dataframe
    resolution : total number of points to interpolate through soil column/spacing between points if using np.arange
    time_range : time period to be calculated over 
    months : months included in calculation - ['Jan', 'Feb', 'Dec'] (e.g winter season)
    '''
    month_range = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    startyr, endyr = time_range.split('-')
#     startyr, endyr = time_range[0], time_range[1]

    #Setting time range    
    range_series = VAR[startyr:endyr]
    LD = depth[startyr:endyr] #depth from surface
    LZ = thickness[startyr:endyr] #thickness of each layer

    #Seasonal exclusion
    range_series = range_series[range_series.index.month.isin([i+1 for i, e in enumerate(month_range) if e in months])]
    LD = (LD[LD.index.month.isin([i+1 for i, e in enumerate(month_range) if e in months])]).mean()
    LZ = (LZ[LZ.index.month.isin([i+1 for i, e in enumerate(month_range) if e in months])]).mean()
    mean = range_series.mean().values.tolist()
        
    maxi = range_series.max().values.tolist()
    mini = range_series.min().values.tolist()
    std =  range_series.std().values.tolist()

    xp = (LD + LZ/2).values.tolist() #center of each layer,  x-coordinates of the data points for np.interp
    
    indexes = [i for i,v in enumerate(xp) if v < 0]

    for index in sorted(indexes, reverse=True):
        del xp[index]
        del mean[index] #y-coordinates of the data points in np.interp, same length as xp
        del mini[index]
        del maxi[index]
        del std[index]

    #creating depth array to interpolate through
#     x=np.linspace(min(xp), max(xp), resolution)
    x=np.arange(0,max(LD),resolution) #x-coordinates at which to evaluate the interpolated values
    
    interp_mean=np.interp(x,xp,mean) 
    interp_mini=np.interp(x,xp,mini)
    interp_maxi=np.interp(x,xp,maxi)
    interp_std=np.interp(x,xp,std)
    
    return x,interp_mean, interp_std, interp_mini, interp_maxi


def seasonal_stats(VAR, time_range, months):
    '''
    VAR : variable dataframe
    time_range : time period to be calculated over 
    months : months included in calculation - ['Jan', 'Feb', 'Dec'] (e.g winter season)
    '''
    month_range = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Setting time range
    range_series = VAR[time_range[0]:time_range[1]]

    # Seasonal exclusion
    range_series = range_series[range_series.index.month.isin([i+1 for i, e in enumerate(month_range) if e in months])]
    mean = range_series.mean().values.tolist()
    maxi = range_series.max().values.tolist()
    mini = range_series.min().values.tolist()
    std = range_series.std().values.tolist()

    return mean, std, mini, maxi
