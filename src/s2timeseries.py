"""
A few functions for extracting of time series for Sentinel 2 data with the

view to comparing it with something else 

"""
import ee
import pandas as pd
from datetime import datetime
from pyproj import Transformer
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def _points_to_pixel(gdf, espgin='epsg:27700', espgout='epsg:4326'):
    """
    convert some points from one proj to the other and return pixel coords
    and lon lats for some underlying raster
    
    returns both to give choice between numpy/pandas or xarray
    
    """
    transformer = Transformer.from_crs(espgin, espgout, always_xy=True) 
    xin = gdf.POINT_X
    yin = gdf.POINT_Y
    
    # better than the old pyproj way
    points = list(zip(xin, yin))
    
    # output order is lon, lat
    coords_oot = np.array(list(transformer.itransform(points)))
    
    # for readability only
    lats = coords_oot[:,1]
    lons = coords_oot[:,0]
    
    return  lons, lats


# conver S2 name to date
def _conv_date(entry):
    
    # TODO I'm sure there is a better way....
    # split based on the _ then take the first block of string
    dt = entry.split(sep='_')[0][0:8]
    
    # the dt entry
    oot = datetime.strptime(dt, "%Y%m%d")
    
    return oot
    

# The main process
    
def _s2_tseries(lat, lon, collection="COPERNICUS/S2", start_date='2016-01-01',
               end_date='2016-12-31', dist=20, cloud_mask=True, 
               stat='max', cloud_perc=100, para=False):
    
    """
    Time series from a single coordinate using gee
    
    Parameters
    ----------
    
    lat: int
             lat for point
    
    lon: int
              lon for pont
              
    collection: string
                    the S2 collection either.../S2 or .../S2_SR
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
                    
    dist: int
             the distance around point e.g. 20m
             
    cloud_mask: int
             whether to mask cloud
             
    cloud_perc: int
             the acceptable cloudiness per pixel in addition to prev arg
              
    """
    # joblib hack - this is goona need to run in gee directly i think
    if para == True:
        ee.Initialize()
    
    
    S2 = ee.ImageCollection(collection).filterDate(start_date,
                       end_date).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',
                       cloud_perc))


    # a point
    geometry = ee.Geometry.Point(lon, lat)
    
    # give the point a distance to ensure we are on a pixel
    s2list = S2.filterBounds(geometry).getRegion(geometry, dist).getInfo()
    
    # the headings of the data
    cols = s2list[0]
    
    # the rest
    rem=s2list[1:len(s2list)]
    
    # now a big df to reduce somewhat
    df = pd.DataFrame(data=rem, columns=cols)

    # get a proper date
    df['Date'] = df['id'].apply(_conv_date)
    
    if cloud_mask == True:
        
        # Have kept the bitwise references here
        cloudBitMask = 1024 #1 << 10 
        cirrusBitMask = 2048 # 1 << 11 
        
        df = df.drop(df[df.QA60 == cloudBitMask].index)
        df = df.drop(df[df.QA60 == cirrusBitMask].index)
    
    # Don't bother as this occasionally scrubs legit values!!!    
    # there are 2 granules - likely unfortunate loc.
    #df = df.drop_duplicates(subset=['Date'])
    
    # ndvi whilst we are here
    df['ndvi'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'])
    
    # May change to this for merging below
    df = df.set_index(df['Date'])
    nd = pd.Series(df['ndvi'])
    # return nd
    # may be an idea to label the 'id' as a contant val or something
    # dump the redundant  columns
    
    # A monthly dataframe 
    # Should be max or upper 95th looking at NDVI
    if stat == 'max':
        nd = nd.resample(rule='M').max()
    if stat == 'perc':
        # avoid potential outliers
        nd = nd.resample(rule='M').quantile(.95)
    
    # For entry to a shapefile must be this way up
    return nd.transpose()
    
# Main
    
#inShp = '/home/ciaran/SOC-D/SOC-D-CSS/CSVeg20161819.shp'
#
#outfile = '/home/ciaran/SOC-D/SOC-D-CSS/CSVeg20161819_NDVI_S2.shp'
#
#gdf = gpd.read_file(inShp)

# the parallel version
    

      
#lons, lats = _points_to_pixel(gdf)

# Until I figure this out a quick dirty answer - this took almost 2mins....
# ... so is very slow....

def S2_ts(lons, lats, gdf, collection="COPERNICUS/S2", start_date='2016-01-01',
               end_date='2016-12-31', dist=20, cloud_mask=True, 
               stat='max', cloud_perc=100, para=False, outfile=None):
    
    
    """
    Time series from a single coordinate using gee
    
    Parameters
    ----------
    
    lons: int
              lon for pont
    
    lats: int
             lat for point

              
    collection: string
                    the S2 collection either.../S2 or .../S2_SR
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
                    
    dist: int
             the distance around point e.g. 20m
             
    cloud_mask: int
             whether to mask cloud
             
    cloud_perc: int
             the acceptable cloudiness per pixel in addition to prev arg
             
    Returns
    -------
         
    """
    
    idx = np.arange(0, len(lons))
    
    wcld = Parallel(n_jobs=-1, verbose=2)(delayed(_s2_tseries)(lats[p], lons[p],
                   cloud_mask=False, para=True) for p in idx) 
    
    finaldf = pd.DataFrame(wcld)
    
    finaldf.columns = finaldf.columns.strftime("%Y-%m-%d").to_list()

    newdf = pd.merge(gdf, finaldf, on=gdf.index)
    
    if outfile != None:
        newdf.to_file(outfile)
    
    return newdf

def _plotsquare(df, square):
    
    # Quick dirty time series plotting

    sqr = df[df.SQUARE==square]
    
    yrcols = [y for y in sqr.columns if '2016' in y]
    
    ndplotvals = sqr[yrcols]
    
    ndplotvals.transpose().plot.line()

## This is for a shapefile
#wdf.index = gdf.index
#
## For plotting reasons
#fwdft = wdf.transpose()
#
## eg - well that isn't much good is it lots of missing values
#fwdft.loc[:,0:100].plot.line()
#
## Without cloud ######################################
#wocld = Parallel(n_jobs=-1, verbose=2)(delayed(s2_tseries)(lats[p], lons[p],
#               para=True) for p in idx) 
## With the cloud mask - not sure whether this is helpful
## Should probably rename to months......
#finaldf = pd.DataFrame(wocld)#.reset_index()
## This is for a shapefile
#finaldf.index = gdf.index
#
## For plotting reasons
#fdft = finaldf.transpose()
#
## eg - well that isn't much good is it lots of missing values
#fdft.loc[:,0:100].plot.line()
#
## Comapre the 2 
#
#fdft.loc[:,3].plot.line()
#fwdft.loc[:,3].plot.line()
#
#### Save a new shape
#
## Sort columns back out agan
#
#finaldf.columns = finaldf.columns.strftime("%Y-%m-%d").to_list()
#
#newdf = pd.merge(gdf, finaldf, on=gdf.index)
#newdf.to_file(outfile)
#
## Quick dirty time series plotting
## One assumes things will be similar in the same square!
#sqr = newdf[newdf.SQUARE==637]
#
#yrcols = [y for y in sqr.columns if '2016' in y]
#
#ndplotvals = sqr[yrcols]
#
#ndplotvals.transpose().plot.line()






 

