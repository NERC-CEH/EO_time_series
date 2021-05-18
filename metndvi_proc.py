#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that attributes a 2016 CSS file with met office data 
and S2 derived NDVI. This must be pointed at the 

@author: Ciaran Robb
"""

import os 
from src.downloader import dloadbatch, setup_sesh
from src.met_tseries import met_time_series
from tqdm import tqdm
import src.eotimeseries as eot
import ee
import geopandas as gpd
import argparse
ee.Initialize()


parser = argparse.ArgumentParser()
parser.add_argument("-user", "--usr", type=str, required=True, 
                    help="CEDA username - required for MET data")

parser.add_argument("-pass", "--pss", type=str, required=True, 
                    help="CEDA password - required for MET data")

parser.add_argument("-cssfile", "--css", type=str, required=True, 
                    help="countryside survey point shapefile")

parser.add_argument("-folder", "--fldr", type=str, required=True, 
                    help="The path of the folder to work in")



args = parser.parse_args() 


#A bit of a kludge solution but a one off....
setup_sesh(args.usr, args.pss)

os.chdir(args.fldr)

# The input data
inShp = args.css

#template url - MET seem to update this semi regularly
rain_url16 = ('https://dap.ceda.ac.uk/badc/ukmo-hadobs/data/insitu/MOHC/'
 'HadOBS/HadUK-Grid/v1.0.2.1/1km/rainfall/mon/v20200731/'
 'rainfall_hadukgrid_uk_1km_mon_201601-201612.nc')
# prelim yr vars
years = ['201701-201712', '201801-201812', '201901-201912']

yrlist = [rain_url16.replace('201601-201612', y) for y in years]

# stick 2016 in
yrlist.insert(0, rain_url16)

# list of vars
clim_vars = ['groundfrost', 'hurs', 'psl', 'pv', 'sfcWind', 'sun', 'tas',
             'rainfall']

# wherever we are!
outdirs = ['2016', '2017', '2018', '2019']
_ = [os.mkdir(d) for d in outdirs if not os.path.isdir(d)]

# final list of urls
finalproc = []
for y in yrlist:
    sublist = [y.replace('rainfall', c) for c in clim_vars]
    finalproc.append(sublist)

# the met stuff
ootlists = []

for idx, yr in enumerate(finalproc):
    
    imlist = dloadbatch(yr, outdirs[idx])
    ootlists.append(imlist)
    for d,p in tqdm(zip(imlist, clim_vars)):
        met_time_series(d, inShp, inShp, p) 
        
  
# Now for the Sentinel 2 data. Rather difficult to eliminate cloud without also
# elimating half the data. This is the TOA data as the SR is only available '17
# onwards 
# list of start and end dates

# Due to how I implemented the agg of data has to be done per year
# TODO  change this!!
dates = [('2016-01-01','2016-12-31'), ('2017-01-01','2017-12-31'), 
         ('2018-01-01','2018-12-31'), ('2019-01-01','2019-12-31')]
        
gdf = gpd.read_file(inShp)
# get the coords
lons, lats = eot.points_to_pixel(gdf, espgin='epsg:27700', espgout='epsg:4326')
del gdf

# Tis a question as to whether mean, max, or upper 95th perc is the best monthly
# agg to go with? 95th perc is used here in case of spurious NDVI vals
# The below DOES NOT mask clouds via percentage or bit mask (but could), but 
# this always eliminates potentially useful data. 
# You will have to decide afterwards which vals look plausible in the time series
# It may be possible to smooth it to get a 'better curve'. 

# See the README.rst for an explanation of why client parallelism is used here
# Updates/improvements welcome....
for d in dates:
    _ = eot.S2_ts(inShp, collection="COPERNICUS/S2", start_date=d[0],
      end_date=d[1], dist=10, cloud_mask=False, stat='perc', 
      cloud_perc=100, para=True, outfile=inShp)
    
# Now for a ratio of polarisation from the Sentinel 1 GRD data. 
# This will be labelled VVVH in the attributes
for d in dates:
    print(d)
    _ = eot.S1_ts(inShp, start_date=d[0],
               end_date=d[1], dist=20,  polar='VVVH',
               orbit='ASCENDING', stat='mean', outfile=inShp, month=True,
               para=True)
        
        

        
        
        
    