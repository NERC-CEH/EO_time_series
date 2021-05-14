"""
A few functions for extracting of time series from S1/S2 data with GEE with a

view to comparing it with something else 

@author: Ciaran Robb

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
from osgeo import ogr, osr, gdal
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
    

    roi = extent2poly(geom, filetype='polygon')
    
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

def _feat2dict(lyr, idx, transform=None):
    """
    convert an ogr feat to a dict
    """
    feat = lyr.GetFeature(idx)
    geom = feat.GetGeometryRef()
    if transform != None:
        geom.Transform(transform)
    
    js = geom.ExportToJson()
    geoj = json.loads(js)
    
    # bloody GEE again infuriating
    # prefers lon, lat for points but the opposite for polygons
    # TODO - better fix is required....
    if geoj['type'] == "Point":
        new = [geoj["coordinates"][1], geoj["coordinates"][0]]
        geoj["coordinates"]=new
        
    return geoj

def poly2dictlist(inShp, wgs84=False):
    
    """
    Convert an ogr to a list of json like dicts
    
    Parameters
    ----------
    
    inShp: string
            input OGR compatible polygon
    
    """
    vds = ogr.Open(inShp)
    lyr = vds.GetLayer()
    
    if wgs84 == True:
        # Getting spatial reference of input 
        srs = lyr.GetSpatialRef()
    
        # make WGS84 projection reference3
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
    
        # OSR transform
        transform = osr.CoordinateTransformation(srs, wgs84)
        
    
    features = np.arange(lyr.GetFeatureCount()).tolist()
    
    # results in the repetiton of first one bug
    # feat = lyr.GetNextFeature() 

    oot = [_feat2dict(lyr, f, transform=transform) for f in features]
    
    return oot
    
    

def _raster_extent(inras):
    
    """
    Parameters
    ----------
    
    inras: string
        input gdal raster (already opened)
    
    """
    rds = gdal.Open(inras)
    rgt = rds.GetGeoTransform()
    minx = rgt[0]
    maxy = rgt[3]
    maxx = minx + rgt[1] * rds.RasterXSize
    miny = maxy + rgt[5] * rds.RasterYSize
    ext = (minx, miny, maxx, maxy)
    
    return ext


def extent2poly(infile, filetype='polygon', outfile=True, polytype="ESRI Shapefile", 
                   geecoord=False):
    
    """
    Get the coordinates of a files extent and return an ogr polygon ring with 
    the option to save the  
    
    
    Parameters
    ----------
    
    infile: string
            input ogr compatible geometry file or gdal raster
            
    filetype: string
            the path of the output file, if not specified, it will be input file
            with 'extent' added on before the file type
    
    outfile: string
            the path of the output file, if not specified, it will be input file
            with 'extent' added on before the file type
    
    polytype: string
            ogr comapatible file type (see gdal/ogr docs) default 'ESRI Shapefile'
            ensure your outfile string has the equiv. e.g. '.shp'
    
    geecoord: bool
           optionally convert to WGS84 lat,lon
           
    Returns
    -------
    
    a GEE polygon geometry
    
    """
    # ogr read in etc
    if filetype == 'raster':
        ext = _raster_extent(infile)
        
    else:
        # tis a vector
        vds = ogr.Open(infile)
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
        
        tproj = wgs84
    else:
        tproj = lyr.GetSpatialRef()
    
    # in case we wish to write it for later....    
    if outfile != None:
        outfile = infile[:-4]+'extent.shp'
        
        out_drv = ogr.GetDriverByName(polytype)
        
        # remove output shapefile if it already exists
        if os.path.exists(outfile):
            out_drv.DeleteDataSource(outfile)
        
        # create the output shapefile
        ootds = out_drv.CreateDataSource(outfile)
        ootlyr = ootds.CreateLayer("extent", tproj, geom_type=ogr.wkbPolygon)
        
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

def _conv_dateS1(entry):
    
    # TODO I'm sure there is a better way....
    # split based on the _ then take the first block of string
    dt = entry.split(sep='_')[5][0:8]
    
    # the dt entry
    oot = datetime.strptime(dt, "%Y%m%d")
    
    return oot  

def simplify(fc):
    def feature2dict(f):
            id = f['id']
            out = f['properties']
            out.update(id=id)
            return out
    out = [feature2dict(x) for x in fc['features']]
    return out
        
def _s2_tseries(geometry,  collection="COPERNICUS/S2", start_date='2016-01-01',
               end_date='2016-12-31', dist=20, cloud_mask=True, 
               stat='max', cloud_perc=100, para=False):
    
    """
    S2 Time series from a single coordinate using gee
    
    Parameters
    ----------
    
    geometry: list
                either [lon,lat] fot a point, or a set of coordinates for a 
                polygon


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
    
    # has to reside in here in order for para execution
    def _reduce_region(image):
        # cheers geeextract!   
        """Spatial aggregation function for a single image and a polygon feature"""
        stat_dict = image.reduceRegion(fun, geometry, 30);
        # FEature needs to be rebuilt because the backend doesn't accept to map
        # functions that return dictionaries
        return ee.Feature(None, stat_dict)
    
    S2 = ee.ImageCollection(collection).filterDate(start_date,
                       end_date).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',
                       cloud_perc))
    
    if geometry['type'] == 'Polygon':
        
        # cheers geeextract folks 
        if stat == 'mean':
            fun = ee.Reducer.mean()
        elif stat == 'median':
            fun = ee.Reducer.median()
        elif stat == 'max':
            fun = ee.Reducer.max()
        elif stat == 'min':
            fun = ee.Reducer.min()
        elif stat == 'perc':
            # for now as don't think there is equiv
            fun = ee.Reducer.mean()
        else:
            raise ValueError('Must be one of mean, median, max, or min')
        geomee = ee.Geometry.Polygon(geometry['coordinates'])
        
        s2List = S2.filterBounds(geomee).map(_reduce_region).getInfo()
        s2List = simplify(s2List)
        
        df = pd.DataFrame(s2List)
        
    elif geometry['type'] == 'Point':
    
        # a point
        geomee = ee.Geometry.Point(geometry['coordinates'])
        # give the point a distance to ensure we are on a pixel
        s2list = S2.filterBounds(geometry).getRegion(geometry, dist).getInfo()
    
        # the headings of the data
        cols = s2list[0]
        
        # the rest
        rem=s2list[1:len(s2list)]
        
        # now a big df to reduce somewhat
        df = pd.DataFrame(data=rem, columns=cols)
    else:
        raise ValueError('geom must be either Polygon or Point')

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
    if stat == 'mean':
        nd = nd.resample(rule='M').mean()
    if stat == 'median':
        nd = nd.resample(rule='M').max()
    elif stat == 'perc':
        # avoid potential outliers
        nd = nd.resample(rule='M').quantile(.95)
    
    # For entry to a shapefile must be this way up
    return nd.transpose()
    

# A quick/dirty answer but not an efficient one - this took almost 2mins....

def S2_ts(inshp, collection="COPERNICUS/S2", reproj=False,
          start_date='2016-01-01', end_date='2016-12-31', dist=20, cloud_mask=True, 
               stat='max', cloud_perc=100, para=False, outfile=None):
    
    
    """
    Monthly time series from a point shapefile 
    
    Parameters
    ----------
             
    inshp: string
                a shapefile to join the results to 
              
    collection: string
                    the S2 collection either.../S2 or .../S2_SR
    
    reproj: bool
                whether to reproject to wgs84, lat/lon
    
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
    quicker than dowloading it all from the server side and loading back into memory
    
    """
    # possible to access via gpd & the usual shapely fare....
    # geom = gdf['geometry'][0]
    # geom.exterior.coords.xy
    # but will lead to issues....
    
    geom = poly2dictlist(inshp, wgs84=reproj)
    
    idx = np.arange(0, len(geom))
    
    gdf = gpd.read_file(inshp)
    
    # silly gpd issue
    if 'key_0' in gdf.columns:
        gdf = gdf.drop(columns=['key_0'])
    
    datalist = Parallel(n_jobs=-1, verbose=2)(delayed(_s2_tseries)(
                    geom[p],
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

def plot_group(df, group, index, name, year=None):
    
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
    
    year: string
            the year to summarise e.g. '16' for 2016 (optional)

    
    """
    
    # Quick dirty time series plotting

    sqr = df[df[group]==index]
    
    yrcols = [y for y in sqr.columns if name in y]
    
    if year != None:
        if len(year) > 2:
            raise ValueError('year must be last 2 digits eg 16 for 2016') 
        yrcols = [y for y in yrcols if year in y]
        
    ndplotvals = sqr[yrcols]
    
    ndplotvals.transpose().plot.line()
    
    
def _S1_date(entry):
    
    # TODO I'm sure there is a better way....
    # split based on the _ then take the first block of string
    dt = entry.split(sep='_')[0][0:8]
    
    # the dt entry
    oot = datetime.strptime(dt, "%Y%m%d")
    
    return oot


def _s1_tseries(geometry, start_date='2016-01-01',
               end_date='2016-12-31', dist=20,  polar='VVVH',
               orbit='ASCENDING', stat='mean', month=True, para=True):
    
    """
    Time series from a single coordinate using gee
    
    Parameters
    ----------
    
    geometry: json like
             coords of point or polygon

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


def _get_s1_prop(start_date, end_date, geometry, polar='VVVH', orbit='both',
                 stat='mean', dist=10):
    
    """
    Get region info for a point geometry for S1 using filters
    
    Parameters
    ----------
    
    geometry: ee geometry (json like)
              
              
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
    
    def _reduce_region(image):
        # cheers geeextract!   
        """Spatial aggregation function for a single image and a polygon feature"""
        stat_dict = image.reduceRegion(fun, geometry, 30);
        # FEature needs to be rebuilt because the backend doesn't accept to map
        # functions that return dictionaries
        return ee.Feature(None, stat_dict)
    
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
        
    if geometry['type'] == 'Polygon':
        
        # cheers geeextract folks 
        if stat == 'mean':
            fun = ee.Reducer.mean()
        elif stat == 'median':
            fun = ee.Reducer.median()
        elif stat == 'max':
            fun = ee.Reducer.max()
        elif stat == 'min':
            fun = ee.Reducer.min()
        elif stat == 'perc':
            # for now as don't think there is equiv
            fun = ee.Reducer.mean()
        else:
            raise ValueError('Must be one of mean, median, max, or min')
        geomee = ee.Geometry.Polygon(geometry['coordinates'])
        
        s1List = orbf.filterBounds(geomee).map(_reduce_region).getInfo()
        s1List = simplify(s1List)
        
        df = pd.DataFrame(s1List)
        
        df['Date'] = df['id'].apply(_conv_dateS1)
    
    elif geometry['type'] == 'Point':
        geomee = ee.Geometry.Point(geometry['coordinates'])
        # a point
        geomee = ee.Geometry.Point(geomee)
        
        # get the point info 
        s1list = orbf.getRegion(geomee, dist).getInfo()
        
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

def S1_ts(inshp, start_date='2016-01-01', reproj=False, 
               end_date='2016-12-31', dist=20,  polar='VVVH',
               orbit='ASCENDING', stat='mean', outfile=None, month=True,
               para=True):
    
    
    """
    Sentinel 1 month time series from a point shapefile
    
    Parameters
    ----------
    
    inshp: string
            a shapefile to join the results to 
    
    reproj: bool
            whether to reproject to wgs84, lat/lon
    
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
    
    geom = poly2dictlist(inshp, wgs84=reproj)
    
    idx = np.arange(0, len(geom))
    
    if 'key_0' in gdf.columns:
        gdf = gdf.drop(columns=['key_0'])
      
    wcld = Parallel(n_jobs=-1, verbose=2)(delayed(_s1_tseries)(geom[p],
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
