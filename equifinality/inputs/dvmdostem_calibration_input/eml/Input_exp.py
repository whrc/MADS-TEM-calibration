#!/usr/bin/env python

# Author: Helene Genet, UAF
# Creation date: Dec. 9 2021
# Purpose: this script generates modified input files for DVM-DOS-TEM simulations, following three options (see bellow comments for details)


import os
from os.path import exists
import netCDF4 as nc
import pandas as pd
import numpy as np
import json
import xarray as xr
import argparse
import matplotlib.pyplot as plt
import datetime


ORGIN = "historic-climate.nc"
MODIN = "historic-climate-sw.nc"

inpath = '/Users/helenegenet/Helene/TEM/INPUT/production/downscaling_test/eml'

### VARIABLE OF INTEREST
vmod = "tair"
### PIXEL OF INTEREST
xmod = 0
ymod = 0
## First month (1=january ...) and year of the modified time series
start_month = 1
start_year = 2009
## Last month (1=january ...) and year of the modified time series
end_month = 12
end_year = 2022


series = [0,0,0,0,2,2,2,2,2,0,0,0]
start_org = 1901
end_org = 2022

start = (start_year-start_org)*12+start_month-1
end = (end_year-start_org)*12+end_month-1


### Read the original input data
ds=xr.open_dataset(os.path.join(inpath, ORGIN))
### Loop through the twelve months and check that change is required, if not, then pass
for i in range(0,12):
  if series[i] ==0 :
    print ('month',i+1,': no change for this month, pass!')
  ## if change is required, only change the data for the time period of choice
  else :
    print ('month',i + 1,': change =', series[i])
    for j in range(i,len(org['time'][:]),12):
      if j >= start and j <= end:
        print (j)
        ds[vmod][j, ymod, xmod]= ds[vmod][j, ymod, xmod] + series[i]

ds.to_netcdf(os.path.join(inpath, MODIN)) 

original = nc.Dataset(os.path.join(inpath, ORGIN))
modified = nc.Dataset(os.path.join(inpath, MODIN))
plt.plot(original.variables['tair'][(start-24):end,0,0], label='original',marker='o',alpha=.5)
plt.plot(modified.variables['tair'][(start-24):end,0,0], label='modified',marker='^',alpha=.5)
plt.legend()
#plt.show()
#print(inpath)
#Save figure plotting original vs modified csv to be saved in workshop-lab2/modopt1/ path
newfilename=os.path.join(inpath,'orig_vs_sw.png')
plt.savefig(newfilename)
plt.close()









ORGIN = "historic-climate.nc"
MODIN = "historic-climate-ww.nc"

inpath = '/Users/helenegenet/Helene/TEM/INPUT/production/downscaling_test/eml'

### VARIABLE OF INTEREST
vmod = "precip"
### PIXEL OF INTEREST
xmod = 0
ymod = 0
## First month (1=january ...) and year of the modified time series
start_month = 1
start_year = 2009
## Last month (1=january ...) and year of the modified time series
end_month = 12
end_year = 2022


series = [10,10,10,10,0,0,0,0,0,10,10,10]
start_org = 1901
end_org = 2022

start = (start_year-start_org)*12+start_month-1
end = (end_year-start_org)*12+end_month-1


### Read the original input data
ds=xr.open_dataset(os.path.join(inpath, ORGIN))
### Loop through the twelve months and check that change is required, if not, then pass
for i in range(0,12):
  if series[i] ==0 :
    print ('month',i+1,': no change for this month, pass!')
  ## if change is required, only change the data for the time period of choice
  else :
    print ('month',i + 1,': change =', series[i])
    for j in range(i,len(org['time'][:]),12):
      if j >= start and j <= end:
        print (j)
        ds[vmod][j, ymod, xmod]= ds[vmod][j, ymod, xmod] * (1+0.01*series[i])

ds.to_netcdf(os.path.join(inpath, MODIN)) 

original = nc.Dataset(os.path.join(inpath, ORGIN))
modified = nc.Dataset(os.path.join(inpath, MODIN))
plt.plot(original.variables['precip'][(start-24):end,0,0], label='original',marker='o',alpha=.5)
plt.plot(modified.variables['precip'][(start-24):end,0,0], label='modified',marker='^',alpha=.5)
plt.legend()
#plt.show()
#print(inpath)
#Save figure plotting original vs modified csv to be saved in workshop-lab2/modopt1/ path
newfilename=os.path.join(inpath,'orig_vs_ww.png')
plt.savefig(newfilename)
plt.close()









ORGIN = "historic-climate.nc"
MODIN = "historic-climate-aw.nc"

inpath = '/Users/helenegenet/Helene/TEM/INPUT/production/downscaling_test/eml'

### VARIABLE OF INTEREST
vmod1 = "precip"
vmod2 = "tair"
### PIXEL OF INTEREST
xmod = 0
ymod = 0
## First month (1=january ...) and year of the modified time series
start_month = 1
start_year = 2009
## Last month (1=january ...) and year of the modified time series
end_month = 12
end_year = 2022


series1 = [10,10,10,10,0,0,0,0,0,10,10,10]
series2 = [0,0,0,0,2,2,2,2,2,0,0,0]
start_org = 1901
end_org = 2022

start = (start_year-start_org)*12+start_month-1
end = (end_year-start_org)*12+end_month-1


### Read the original input data
ds=xr.open_dataset(os.path.join(inpath, ORGIN))
### Loop through the twelve months and check that change is required, if not, then pass
for i in range(0,12):
  if series1[i] ==0 :
    print ('month',i+1,': no change for this month, pass!')
  ## if change is required, only change the data for the time period of choice
  else :
    print ('month',i + 1,': change =', series1[i])
    for j in range(i,len(org['time'][:]),12):
      if j >= start and j <= end:
        print (j)
        ds[vmod1][j, ymod, xmod]= ds[vmod1][j, ymod, xmod] * (1+0.01*series1[i])

for i in range(0,12):
  if series2[i] ==0 :
    print ('month',i+1,': no change for this month, pass!')
  ## if change is required, only change the data for the time period of choice
  else :
    print ('month',i + 1,': change =', series2[i])
    for j in range(i,len(org['time'][:]),12):
      if j >= start and j <= end:
        print (j)
        ds[vmod2][j, ymod, xmod]= ds[vmod2][j, ymod, xmod] + series2[i]

ds.to_netcdf(os.path.join(inpath, MODIN)) 





original = nc.Dataset(os.path.join(inpath, ORGIN))
modified = nc.Dataset(os.path.join(inpath, MODIN))
plt.plot(original.variables['precip'][(start-24):end,0,0], label='original',marker='o',alpha=.5)
plt.plot(modified.variables['precip'][(start-24):end,0,0], label='modified',marker='^',alpha=.5)
plt.plot(original.variables['tair'][(start-24):end,0,0], label='original',marker='o',alpha=.5)
plt.plot(modified.variables['tair'][(start-24):end,0,0], label='modified',marker='^',alpha=.5)
plt.legend()
newfilename=os.path.join(inpath,'orig_vs_aw.png')
plt.savefig(newfilename)
plt.close()





