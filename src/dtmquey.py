#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:25:47 2021

@author: ciaran
"""

from osgeo import ogr
from urllib.parse import urljoin
import geopandas as gpd
import os
from src.downloader import *


inShp = '/home/ciaran/OSGB_Grids/CSS_contained.shp'

gdf = gpd.read_file(inShp)

# What are the nuts and bolts we need to get this
base_url = 'https://dap.ceda.ac.uk/neodc/nextmap/by_tile/'

tile_info = 'hp/hp40/dtm/hp40dtm/'


# When will folk stop using bloody ESRI formats
'https://dap.ceda.ac.uk/neodc/nextmap/by_tile/hp/hp40/dtm/hp40dtm/'

'https://dap.ceda.ac.uk/neodc/nextmap/by_product/dtm/hp/hp40/hp40dtm'

# where we will iterate
tilenms = gdf["TILE_NAME"].tolist()


def dwnld_tile(base_url, tileid, para=True):
    
    """
    download a nextmap tile
    
    """
    esri_types = ['dblbnd.adf', 'hdr.adf', 'prj.adf',
              'sta.adf', 'w001001.adf', 'w001001x.adf']

    # this is a silly way to do this
    tile1 = urljoin(base_url, tileid[0:2].lower())   
    tile2 = urljoin(tile1, tileid)
    tile3 = urljoin(tile2, 'dtm')
    tile_fin = urljoin(tile3, tileid+'dtm')
    
    dwnpaths = [urljoin(tile_fin, e) for e in esri_types]
    
    if not os.path.isdir(tileid+'dtm'):
        os.mkdir(tileid+'dtm') 
    
    dloadbatch(dwnpaths, tileid+'dtm', para=para, nt=-1, method='urllib')
    
    
    