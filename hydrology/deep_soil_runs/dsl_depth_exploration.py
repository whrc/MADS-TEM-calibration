#!/usr/bin/env python
# coding: utf-8

import sys
import os

import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import datetime
from datetime import datetime, timedelta

# This allows us to import tools from the dvm-dos-tem/scripts directory
sys.path.insert(0, '/work/scripts')
import output_utils as ou
from output_utils import load_trsc_dataframe #loads transient and scenario data, columns are layers


def plot_data(site_folder, run_folder, timeres, output_var, y, x, layer, year, end_year):
    '''
    This will plot UNSYNTHESIZED/UN-INTERPOLATED data, ie; for all layers. 
    loads and plots data on a yearly basis (regardless of time res of data)
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
    This will plot DEPTH INTERPOLATED data, ie; for depths specified in Output_Synthesis_Valeria.sh
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


# ### No DSL:
# DSL = dynamic soil layering module
# This module contains the set of equations that relate C stocks to layer thickness. If C stocks are too small in a given layer, they may be merged with other layers - this causes the layers to be reordered. Turning off the dsl module removes the appearance of spikes in the layerdepth data
# 
# ##### To Turn off DSL: 
#     - change all instances of runner.cohort.md->set_dslmodule(true);  to runner.cohort.md->set_dslmodule(false);  in TEM.cpp in the src directory - and then recompile.
#     - layer thicknesses should then match the values in the parameter files cmt_dimground.txt
#     
# NOTE: Model runs with DSL turned off are tagged "nodsl" in the workflow folder name

#PLOT ALD

# df_yearly, df_ds, df_yearly_m, df_m=plot_experiment('ALD','MD_deep_soil',0)

fig, ax = plt.subplots(1,2,figsize=(15, 5))

#Plot master branch run
os.chdir('/data/workflows/MD_deep_soil/control')
df, meta = load_trsc_dataframe(var='ALD', timeres='yearly', px_y=0, px_x=0, fileprefix='output')
ax[0].plot(df.loc['2010':'2100'].index, df.loc['2010':'2100'][0], label='ALD')
ax[0].set_xlabel('year')
ax[0].set_ylabel('ALD ({})'.format(meta['var_units']))
ax[0].set_title('Master branch')

#plot deep soil branch run
os.chdir('/data/workflows/MD_deep_soil/experiment')
df, meta = load_trsc_dataframe(var='ALD', timeres='yearly', px_y=0, px_x=0, fileprefix='output')
ax[1].plot(df.loc['2010':'2100'].index, df.loc['2010':'2100'][0], label='ALD')
ax[1].set_xlabel('year')
ax[1].set_ylabel('ALD ({})'.format(meta['var_units']))
ax[1].set_title('Deep Soil branch')


folder='BNZ_deep_soil'
run_folder_control='control_allout_monthly_00px'
run_folder_experiment='experiment_allout_monthly_00px'
output_var='LAYERDEPTH'
layer=7
year='1950'
end_year='2100'
px_y=0
px_x=0

fig, ax = plt.subplots(1,1,figsize=(15, 5))

plot_data(folder, run_folder_control, 'monthly', output_var, px_y, px_x, layer, year, end_year)
# plot_data(folder, run_folder_experiment, 'monthly', output_var, px_y, px_x, layer, year, end_year)
ax.legend(['Master (unchanged)', 'Deep Soil'])
plt.title(folder+' Pixel 0,0')


folder='BNZ_deep_soil'
run_folder_control='control_allout_monthly_11px'
run_folder_experiment='experiment_allout_monthly_11px'
output_var='RH'
layer=7
year='1950'
end_year='2100'
px_y=1
px_x=1

fig, ax = plt.subplots(1,1,figsize=(15, 5))

plot_data(folder, run_folder_control, 'monthly', output_var, px_y, px_x, layer, year, end_year)
plot_data(folder, run_folder_experiment, 'monthly', output_var, px_y, px_x, layer, year, end_year)
ax.legend(['Master (unchanged)', 'Deep Soil'])
plt.title(folder+' Pixel 1,1')


folder='MD_deep_soil'
run_folder_control='control_allout_monthly_11px'
run_folder_experiment='experiment_allout_monthly_11px'
output_var='SHLWC'
layer=15
year='1950'
end_year='2100'
px_y=1
px_x=1

fig, ax = plt.subplots(1,1,figsize=(15, 5))

plot_data(folder, run_folder_control, 'monthly', output_var, px_y, px_x, layer, year, end_year)
# plot_data(folder, run_folder_experiment, 'monthly', output_var, px_y, px_x, layer, year, end_year)
ax.legend(['Master (unchanged)', 'Deep Soil'])
plt.title(folder+' Pixel 1,1, cmt 1')


folder='MD_deep_soil'
run_folder_control='control_allout_monthly_11px_31'
# run_folder_experiment='experiment_allout_monthly_11px'
output_var='SHLWC'
layer=15
year='1950'
end_year='2100'
px_y=1
px_x=1

fig, ax = plt.subplots(1,1,figsize=(15, 5))

plot_data(folder, run_folder_control, 'monthly', output_var, px_y, px_x, layer, year, end_year)
# plot_data(folder, run_folder_experiment, 'monthly', output_var, px_y, px_x, layer, year, end_year)
ax.legend(['Master (unchanged)', 'Deep Soil'])
plt.title(' MD with dsl on Pixel 1,1, cmt 31')


folder='MD_deep_soil'
run_folder_control='control_allout_monthly_11px_2'
# run_folder_experiment='experiment_allout_monthly_11px'
output_var='SHLWC'
layer=15
year='1950'
end_year='2100'
px_y=1
px_x=1

fig, ax = plt.subplots(1,1,figsize=(15, 5))

plot_data(folder, run_folder_control, 'monthly', output_var, px_y, px_x, layer, year, end_year)
# plot_data(folder, run_folder_experiment, 'monthly', output_var, px_y, px_x, layer, year, end_year)
ax.legend(['Master (unchanged)', 'Deep Soil'])
plt.title(' MD with dsl on Pixel 1,1, cmt 2')


folder='MD_deep_soil'
run_folder_control='control_allout_monthly_11px_3'
output_var='LTRFALC'
layer=15
year='1950'
end_year='2015'
px_y=1
px_x=1

fig, ax = plt.subplots(1,1,figsize=(15, 5))

plot_data(folder, run_folder_control, 'monthly', output_var, px_y, px_x, layer, year, end_year)
ax.legend(['LTRFALC', 'LAYERDEPTH'])
plt.title(' MD with dsl on Pixel 1,1, cmt 3')

fig, ax = plt.subplots(1,1,figsize=(15, 5))
plot_data(folder, run_folder_control, 'monthly', 'LAYERDEPTH', px_y, px_x, layer, year, end_year)


folder='BNZ_deep_soil'
run_folder_control='control_allout_monthly_34px'
run_folder_experiment='experiment_allout_monthly_34px'
output_var='LAYERDEPTH'
layer=7
year='1950'
end_year='2100'
px_y=3
px_x=4

fig, ax = plt.subplots(1,1,figsize=(15, 5))

plot_data(folder, run_folder_control, 'monthly', output_var, px_y, px_x, layer, year, end_year)
# plot_data(folder, run_folder_experiment, 'monthly', output_var, px_y, px_x, layer, year, end_year)
ax.legend(['Master (unchanged)', 'Deep Soil'])
plt.title(folder+' Pixel 3,4')


#WITH UPDATED COEFFICIENTS
folder='MD_deep_soil'
run_folder_control='control_allout_monthly_11px'
run_folder_experiment='experiment_allout_monthly_11px'
output_var='LAYERDEPTH'
layer=17
year='1950'
end_year='2100'
px_y=1
px_x=1

fig, ax = plt.subplots(1,1,figsize=(15, 5))

plot_data(folder, run_folder_control, 'monthly', output_var, px_y, px_x, layer, year, end_year)
plot_data(folder, run_folder_experiment, 'monthly', output_var, px_y, px_x, layer, year, end_year)
ax.legend(['Master (unchanged)', 'Deep Soil'])
plt.title(folder+' Pixel 1,1')


folder='MD_deep_soil'
run_folder_control='control_allout_monthly_34px'
# run_folder_experiment='control_allout_monthly_34px_nodsl'
output_var='SHLWC'
layer=2
year='1950'
end_year='2100'
px_y=3
px_x=4

fig, ax = plt.subplots(1,1,figsize=(15, 5))

plot_data(folder, run_folder_control, 'monthly', output_var, px_y, px_x, layer, year, end_year)
# plot_data(folder, run_folder_experiment, 'monthly', 'LAYERDEPTH', px_y, px_x, layer, year, end_year)
ax.legend(['Master (unchanged)', 'No DSL module'])
plt.title('MD Pixel 3,4')


folder='MD_deep_soil'
run_folder_control='control_allout_monthly_34px'
run_folder_experiment='control_allout_monthly_34px_nodsl'
output_var='SHLWC'
layer=4
year='1950'
end_year='2100'
px_y=3
px_x=4

fig, ax = plt.subplots(1,1,figsize=(15, 5))

plot_data(folder, run_folder_control, 'monthly', 'LAYERDEPTH', px_y, px_x, layer, year, end_year)
plot_data(folder, run_folder_experiment, 'monthly', 'LAYERDEPTH', px_y, px_x, layer, year, end_year)
ax.legend(['Master (unchanged)', 'No DSL module'])
plt.title('MD Pixel 3,4')




