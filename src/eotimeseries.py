"""
A few functions for extracting of time series from S1/S2 data with GEE with a

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
import geemap
from osgeo import ogr, osr
import os
import json

ogr.UseExceptions()
osr.UseExceptions()

def get_month_ndperc(start_date, end_date, geom, collection="COPERNICUS/S2"):
    
    """
    Make a monthly ndvi 95th percentile composite collection
    
    Parameters
    ----------
              
    start_date: string
                    start date of time series
                    
    end_date: string
                    end date of time series
    
    geom: string
          an ogr compatible file with an extent
          
    collection: string
                the ee image collection  
    
    Returns
    -------
    
    GEE collection of monthly NDVI images
    
    """
    

    roi = extent2poly(geom)
    
    imcol = ee.ImageCollection(
            collection).filterDate(start_date, end_date).filterBounds(roi)
    
    months = ee.List.sequence(1, 12)
    
    def _funcmnth(m):
        
        filtered = imcol.filter(ee.Filter.calendarRange(start=m, field='month'))
        
        composite = filtered.reduce(ee.Reducer.percentile([95]))
        
        return composite.normalizedDifference(
                ['B8_p95', 'B4_p95']).rename('NDVI').set('month', m)
    
    composites = ee.ImageCollection.fromImages(months.map(_funcmnth))
    
    return composites

def extent2poly(inshp, outfile=None, filetype="ESRI Shapefile", 
                   geecoord=True):
    
    """
    Get the coordinates of a layers extent, save a polygon and return a GEE geom
    
    This will convert to wgs84 lat lon by default
    
    Parameters
    ----------
    
    inshp: string
            input ogr compatible geometry file
    
    outfile: string
            the path of the output file, if not specified, it will be input file
            with 'extent' added on before the file type
    
    filetype: string
            ogr comapatible file type (see gdal/ofr docs) default 'ESRI Shapefile'
            ensure your outfile string has the equiv. e.g. '.shp'
    
    geecoord: bool
           convert to WGS84 lat,lon - necessary for GEE
           
    Returns
    -------
    
    a GEE polygon geometry
    
    """
    # ogr read in etc
    vds = ogr.Open(inshp)
    lyr = vds.GetLayer()
    ext = lyr.GetExtent()
    
    # make the linear ring 
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(ext[0],ext[2])
    ring.AddPoint(ext[1], ext[2])
    ring.AddPoint(ext[1], ext[3])
    ring.AddPoint(ext[0], ext[3])
    ring.AddPoint(ext[0], ext[2])
    
    # drop the geom into poly object
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    if geecoord == True:
        # Getting spatial reference of input 
        srs = lyr.GetSpatialRef()
    
        # make WGS84 projection reference3
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
    
        # OSR transform
        transform = osr.CoordinateTransformation(srs, wgs84)
        # apply
        poly.Transform(transform)
    
    # in case we wish to write it for later....    
    if outfile == None:
        outfile = inshp[:-4]+'extent.shp'
        
    out_drv = ogr.GetDriverByName(filetype)
    
    # remove output shapefile if it already exists
    if os.path.exists(outfile):
        out_drv.DeleteDataSource(outfile)
    
    # create the output shapefile
    ootds = out_drv.CreateDataSource(outfile)
    ootlyr = ootds.CreateLayer("extent", geom_type=ogr.wkbPolygon)
    
    # add an ID field
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    ootlyr.CreateField(idField)
    
    # create the feature and set values
    featureDefn = ootlyr.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    feature.SetGeometry(poly)
    feature.SetField("id", 1)
    ootlyr.CreateFeature(feature)
    feature = None
    
    # Save and close 
    ootds.FlushCache()
    ootds = None
    # gee doesn't like the json for some weird reason     
    # outpoly = json.loads(poly.ExportToJson())
    # hence we have this hack solution
    outpoly = geemap.shp_to_ee(outfile)
    
    return outpoly


def zonal_tseries(collection, start_date, end_date, inShp, bandnm='NDVI',
                  attribute='id'):
    
    
    """
    Zonal Time series for a feature collection 
    
    Parameters
    ----------
              
    collection: string
                    the image collection  best if this is agg'd monthly or 
                    something
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
             
    bandnm: string
             the bandname of choice that exists or has been created in 
             the image collection  e.g. B1 or NDVI
            
    attribute: string
                the attribute for filtering (required for GEE)
                
    Returns
    -------
    
    pandas dataframe
    
    Notes
    -----
    
    Unlike the other tseries functions here, this operates server side, meaning
    the bottleneck is in the download/conversion to dataframe.
    
    This function is not reliable with every image collection at present
    
    """
    # Overly elaborate, unreliable and fairly likely to be dumped 

    # shp/json to gee feature here
    # if #filetype is something:
    'converting to ee feature'
    shp = geemap.shp_to_ee(inShp)
    # else # it is a json:
    
    
    # select the band and perform a spatial agg
    # GEE makes things ugly/hard to read
    def _imfunc(image):
      return (image.select(bandnm)
        .reduceRegions(
          collection=shp.select([attribute]),
          reducer=ee.Reducer.mean(),
          scale=30
        )
        .filter(ee.Filter.neq('mean', None))
        .map(lambda f: f.set('imageId', image.id())))

    # now map the above over the collection
    # we have a triplet of attributes, which can be rearranged to a table
    # below,
    triplets = collection.map(_imfunc).flatten()
    
    
    def _fmt(table, row_id, col_id):
      """
      arrange the image stat values into a table of specified order
      """ 
      def _funcfeat(feature):
              feature = ee.Feature(feature)
              return [feature.get(col_id), feature.get('mean')]
          
      def _rowfunc(row):
          
          values = ee.List(row.get('matches')).map(_funcfeat)
          
          return row.select([row_id]).set(ee.Dictionary(values.flatten()))
    
      rows = table.distinct(row_id)
      # Join the table to the unique IDs to get a collection in which
      # each feature stores a list of all features having a common row ID.
      joined = ee.Join.saveAll('matches').apply(primary=rows,
                              secondary=table,
                              condition=ee.Filter.equals(leftField=row_id, 
                                                         rightField=row_id))
      
      t_oot = joined.map(_rowfunc)
      
      return t_oot 
    
    # run the above to produce the table where the columns are attributes
    # and the rows the image ID
    table = _fmt(triplets, attribute, 'imageId')
    
    print('converting to pandas df')
    df = geemap.ee_to_pandas(table)
    # bin the const GEE index
    df = df.drop(columns=['system:index'])
    #Nope.....
    #geemap.ee_export_vector(table, outfile)
    
    
    return df
    

def points_to_pixel(gdf, espgin='epsg:27700', espgout='epsg:4326'):
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
    S2 Time series from a single coordinate using gee
    
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
    

# A quick/dirty answer but not an efficient one - this took almost 2mins....

def S2_ts(lons, lats, inshp, collection="COPERNICUS/S2", start_date='2016-01-01',
               end_date='2016-12-31', dist=20, cloud_mask=True, 
               stat='max', cloud_perc=100, para=False, outfile=None):
    
    
    """
    Monthly time series from a point shapefile 
    
    Parameters
    ----------
    
    lons: int
              lon for pont
    
    lats: int
             lat for point
             
    inshp: string
                a shapefile to join the results to 
              
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
    
    outfile: string
           the output shapefile if required
             
             
    Returns
    -------
         
    geopandas dataframe
    
    Notes
    -----
    
    This spreads the point queries client side, meaning the bottleneck is the 
    of threads you have. This is maybe 'evened out' by returning the full dataframe 
    quicker than dowloading it all from the server side
    
    """
    
    idx = np.arange(0, len(lons))
    
    gdf = gpd.read_file(inshp)
    
    if 'key_0' in gdf.columns:
        gdf = gdf.drop(columns=['key_0'])
    
    datalist = Parallel(n_jobs=-1, verbose=2)(delayed(_s2_tseries)(lats[p], lons[p],
                    collection=collection,
                    start_date=start_date,
                    end_date=end_date,
                    stat=stat,
                    cloud_perc=cloud_perc,
                    cloud_mask=cloud_mask, para=True) for p in idx) 
    
    finaldf = pd.DataFrame(datalist)
    
    finaldf.columns = finaldf.columns.strftime("%y-%m").to_list()
    
    finaldf.columns = ["nd-"+c for c in finaldf.columns]

    newdf = pd.merge(gdf, finaldf, on=gdf.index)
    
    if outfile != None:
        newdf.to_file(outfile)
    
    return newdf

def plot_group(df, group, index, name):
    
    """
    Plot time series per CSS square eg for S2 ndvi or met var
    
    Parameters
    ----------
    
    df: byte
        input pandas/geopandas df
    
    group: string
          the attribute to group by
          
    index: int
            the index of interest
            
    name: string
            the name of interest

    
    """
    
    # Quick dirty time series plotting

    sqr = df[df[group]==index]
    
    yrcols = [y for y in sqr.columns if name in y]
    
    ndplotvals = sqr[yrcols]
    
    ndplotvals.transpose().plot.line()
    
    
def _S1_date(entry):
    
    # TODO I'm sure there is a better way....
    # split based on the _ then take the first block of string
    dt = entry.split(sep='_')[0][0:8]
    
    # the dt entry
    oot = datetime.strptime(dt, "%Y%m%d")
    
    return oot


def _s1_tseries(lat, lon, start_date='2016-01-01',
               end_date='2016-12-31', dist=20,  polar='VVVH',
               orbit='ASCENDING', stat='mean', month=True, para=True):
    
    """
    Time series from a single coordinate using gee
    
    Parameters
    ----------
    
    lat: int
             lat for point
    
    lon: int
              lon for pont
             
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
                    
    dist: int
             the distance around point e.g. 20m
             
    polar: string
             send receive characteristic - VV, VH or both
             
    orbit: string
             the orbit direction - either 'ASCENDING' or 'DESCENDING'
    
    month: bool
            aggregate to month
              
    """
    # joblib hack - this is gonna need to run in gee directly i think
    if para == True:
        ee.Initialize()
    
    # a point
    geometry = ee.Geometry.Point(lon, lat)
    
    # give the point a distance to ensure we are on a pixel
    df = _get_s1_prop(start_date, end_date, geometry,
                          polar=polar, orbit=orbit, dist=dist)
    
    # a band ratio that will be useful to go here
    # the typical one that is used
    # TODO - is this really informative? - more filters required?
    if polar == 'VVVH':
        df['VVVH'] = (df['VV'] / df['VH'])
    
    # May change to this for merging below
    df = df.set_index(df['Date'])
    
    # ratio
    nd = pd.Series(df[polar])
    

    # A monthly dataframe 
    # Should be max or upper 95th looking at NDVI
    if stat == 'max':
        nd = nd.resample(rule='M').max()
    if stat == 'perc':
        # avoid potential outliers - not relevant to SAR....
        nd = nd.resample(rule='M').quantile(.95)
    if stat == 'mean':
        nd = nd.resample(rule='M').mean()
    if stat == 'median':
        nd = nd.resample(rule='M').median()
    
    # For entry to a shapefile must be this way up
    return nd.transpose()


def _get_s1_prop(start_date, end_date, geometry, polar='VVVH', orbit='both', dist=10):
    
    """
    Get region info for a point geometry for S1 using filters
    
    Parameters
    ----------
    
    geometry: ee geometry (json like)
              lon for pont
              
    polar: string
            either VV, VH or both
            
    orbit: string or list of strings
            either 'ASCENDING', 'DESCENDING' or 'both'
            
    dist: int
          distance from point in metres (e.g. for 10m pixel it'd be 10)
          
    Returns
    -------
    
    a dataframe of S1 region info:
    
    'id', 'longitude', 'latitude', 'Date', 'VV', 'VH', 'angle'
    
    """
    
    # TODO - Needs tidied up conditional statements are a bit of a mess
    # the collection
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterDate(start_date,
                       end_date)
    
    # should we return both?
    if polar == "VVVH":
        pol_select = s1.filter(ee.Filter.eq('instrumentMode',
                                            'IW')).filterBounds(geometry)
        
        #s1f = s1.filter(ee.Filter.eq('instrumentMode', 'IW'))
    else:
        # select only one...
        s1f = s1.filter(ee.Filter.listContains('transmitterReceiverPolarisation',
                                               polar)).filter(ee.Filter.eq(
                                                       'instrumentMode', 'IW'))
        # emit/recieve characteristics
        pol_select = s1f.select(polar).filterBounds(geometry)

    # orbit filter
    if orbit != 'both':
         orbf = pol_select.filter(ee.Filter.eq('orbitProperties_pass', orbit))
    else:
        orbf = pol_select

    # get the point info 
    s1list = orbf.getRegion(geometry, dist).getInfo()
    
    cols = s1list[0]
    
    # stay consistent with S2 stuff
    cols[3]='Date'
    
    # the rest
    rem=s1list[1:len(s1list)]
    
    # now a big df to reduce somewhat
    df = pd.DataFrame(data=rem, columns=cols)

    # get a "proper" date - 
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')

 
    return df

def S1_ts(inshp, lats, lons, start_date='2016-01-01',
               end_date='2016-12-31', dist=20,  polar='VVVH',
               orbit='ASCENDING', stat='mean', outfile=None, month=True,
               para=True):
    
    
    """
    Sentinel 1 month time series from a point shapefile
    
    Parameters
    ----------
    
    inshp: string
            a shapefile to join the results to 
    
    lat: int
             lat for point
    
    lon: int
              lon for pont
             
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
                    
    dist: int
             the distance around point e.g. 20m
             
    polar: string
             send receive characteristic - VV, VH or VVVH
             
    orbit: string
             the orbit direction - either 'ASCENDING' or 'DESCENDING'
    
    month: bool
            aggregate to month
            
    outfile: string
               the output shapefile if required
             
    Returns
    -------
    
    geopandas dataframe
    
    Notes
    -----
    
    This spreads the point queries client side, meaning the bottleneck is the 
    of threads you have. This is maybe 'evened out' by returning the full dataframe 
    quicker than dowloading it all from the server side
    
    """
    gdf = gpd.read_file(inshp)
    
    if 'key_0' in gdf.columns:
        gdf = gdf.drop(columns=['key_0'])
    
    idx = np.arange(0, len(lons))
    
    wcld = Parallel(n_jobs=-1, verbose=2)(delayed(_s1_tseries)(lats[p], lons[p],
                    start_date=start_date,
                    end_date=end_date,
                    orbit=orbit, stat=stat, month=month, polar=polar, 
                    para=para) for p in idx) 
    
    finaldf = pd.DataFrame(wcld)
    
    finaldf.columns = finaldf.columns.strftime("%y-%m").to_list()
    
    finaldf.columns = [polar+'-'+c for c in finaldf.columns]

    newdf = pd.merge(gdf, finaldf, on=gdf.index)
    
    if outfile != None:
        newdf.to_file(outfile)
    
    return newdf

def gdf2ee(gdf):
    
    
    features = []
    
    for i in range(gdf.shape[0]):
        geom = gdf.iloc[i:i+1,:] 
        jsonDict = geom.to_json()
        geojsonDict = jsonDict['features'][0] 
        features.append(ee.Feature(geojsonDict)) 

    
# make a list of features
        
def _point2geefeat(points):
    
    """
    convert points to GEE feature collection
    """
    
    # there must be a better way - quick enough though
    feats = [ee.Feature(ee.Geometry.Point(p), {'id': str(idx)} ) for idx, p in enumerate(points)] 
      
    fcoll = ee.FeatureCollection(feats)
    
    return fcoll


def tseries_group(df, name, other_inds=None):
    
    """
    Extract time series of a particular variable e.g. rain
    
    Parameters
    ----------
    
    df: byte
        input pandas/geopandas df
    
    name: string
          the identifiable string e.g. rain, which can be part or all of 
          column name
                      
    year: string
            the year of interest
    
    other_inds: string
            other columns to be included 

    """
    # probably a more elegant way...but works
    ncols = [y for y in df.columns if name in y]
    
    # if we wish to include something else
    if other_inds != None:
        ncols = other_inds + ncols
        
    
    newdf = df[ncols]
    
    return newdf

# a notable alternative
#import ee
#ee.Initialize()
#

#using fcoll above
    
#points = list(zip(lons, lats))
#
#fcoll = _point2geefeat(points)
#
#collection = ee.ImageCollection(
#    'MODIS/006/MOD13Q1').filterDate('2017-01-01', '2017-05-01')
#
#def setProperty(image):
#    dict = image.reduceRegion(ee.Reducer.mean(), fcoll)
#    return image.set(dict)
#
#withMean = collection.map(setProperty)
#
#yip = withMean.aggregate_array('NDVI').getInfo()

#def zonal_tseries(inShp, collection, start_date, end_date, outfile,  
#                  bandnm='NDVI', attribute='id'):
#    
#    """
#    Zonal Time series for a feature collection 
#    
#    Parameters
#    ----------
#    
#    inShp: string
#                a shapefile in the WGS84 lat/lon proj
#              
#    collection: string
#                    the image collection  best if this is agg'd monthly or 
#                    something
#    
#    start_date: string
#                    start date of time series
#    
#    end_date: string
#                    end date of time series
#             
#    bandnm: string
#             the bandname of choice that exists or has been created in 
#             the image collection  e.g. B1 or NDVI
#            
#    attribute: string
#                the attribute for filtering (required for GEE)
#                
#    Returns
#    -------
#    
#    shapefile (saved) and pandas dataframe
#    
#    Notes
#    -----
#    
#    Unlike the other tseries functions here, this operates server side, meaning
#    the bottleneck is in the download/conversion to shapefile/geojson
#    """
#    
#    
#    shp = geemap.shp_to_ee(inShp)
#    
#    # name the img
#    def rename_band(img):
#        return img.select([0], [img.id()])
#    
#    
#    stacked_image = collection.map(rename_band).toBands()
#    
#    # get the img scale
#    scale = collection.first().projection().nominalScale()
#    
#    # the finished feat collection
#    ts = ee.Image(stacked_image).reduceRegions(collection=shp,
#                 reducer=ee.Reducer.mean(), scale=scale)
#    
#    geemap.ee_export_vector(ts, outfile)
#    
#    # TODO return a dataframe?
#    gdf = gpd.read_file(outfile)
#    
#    return gdf 
