#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:25:47 2021

@author: Ciaran Robb
"""

from osgeo import ogr
from urllib.parse import urljoin
import geopandas as gpd
import os
from src.downloader import *
from osgeo import gdal
from tqdm import tqdm

inShp = '/home/ciaran/OSGB_Grids/CSS_contained.shp'

gdf = gpd.read_file(inShp)

# What are the nuts and bolts we need to get this
base_url = 'https://dap.ceda.ac.uk/neodc/nextmap/by_tile/'

#base_url = "ftp://ftp.ceda.ac.uk/neodc/nextmap/by_tile/"

tile_info = 'hp/hp40/dtm/hp40dtm/'


# When will folk stop using bloody ESRI formats
'https://dap.ceda.ac.uk/neodc/nextmap/by_tile/hp/hp40/dtm/hp40dtm/'

'https://dap.ceda.ac.uk/neodc/nextmap/by_product/dtm/hp/hp40/hp40dtm'

# where we will iterate
tilenms = gdf["TILE_NAME"].tolist()

# Should work but CEDA's dap route is not working for this data, only ftp
def dwnld_tile(base_url, tileid, para=True):
    
    """
    download a nextmap tile
    
    """
    esri_types = ['dblbnd.adf', 'hdr.adf', 'prj.adf',
              'sta.adf', 'w001001.adf', 'w001001x.adf']

    # this is a silly way to do this
    tile1 = urljoin(base_url, tileid[0:2].lower()+'/')   
    tile2 = urljoin(tile1, tileid+'/')
    tile3 = urljoin(tile2, 'dtm'+'/')
    tile_fin = urljoin(tile3, tileid+'dtm')
    
    dwnpaths = [urljoin(tile_fin, e) for e in esri_types]
    
    if not os.path.isdir(tileid+'dtm'):
        os.mkdir(tileid+'dtm') 
    
    dloadbatch(dwnpaths, tileid+'dtm', para=para, nt=-1, method='urllib')

# THE HACKY WAY ###############################################################
# its getting desp[erate]
    
def replace_str(template, t):
    out1 = template.replace('hp', t[0:2])
    out2 = out1.replace('40', t[2:4])
    return out2

template = "ftp://ftp.ceda.ac.uk/neodc/nextmap/by_tile/hp/hp40/dtm/hp40dtm/"
#wtf     = "ftp://ftp.ceda.ac.uk/neodc/nextmap/by_tile/sh60/sh60/dtm/sh60dtm/"

tilenms = [t.lower() for t in tilenms]

dwnurls = [replace_str(template, t) for t in tilenms]

cmd = ['wget', '-e', 'robots=off','--mirror', '--no-parent', 
       '-np', '-nH', '--cut-dirs', '6', '-r', '--user', 'ciarob01', '--password', 
       'k0himUpE81!']

for d in dwnurls:
    cmd.append(d)
    call(cmd)
    del cmd[14]

# Translate from stupid ESRI format

nmap = '/home/ciaran/SOC-D/NextMap'

dirlist = glob(os.path.join(nmap, '*dtm'))

inlist = [os.path.join(d, 'hdr.adf') for d in dirlist]

def batch_translate(inlist):
    
    for i in tqdm(inlist):
        hd, _ = os.path.split(i)
        ootpth = hd+".tif"
        srcds = gdal.Open(i)
        out = gdal.Translate(ootpth, srcds)
        out.FlushCache()
        out = None
        
batch_translate(inlist)

# cleanup
_ = [shutil.rmtree(d) for d in dirlist]

  
# pretty clumsy using ftp #####################################################
from ftplib import FTP

template = "ftp://ftp.ceda.ac.uk/neodc/nextmap/by_tile/hp/hp40/dtm/hp40dtm/"

tilenms = [t.lower() for t in tilenms]

dwnurls = [replace_str(template, t) for t in tilenms]

ftplist = [d.replace("ftp://ftp.ceda.ac.uk/", "") for d in dwnurls]

def dtmftp(user, passwd, ftp_path):
    
    """
    download the files for a nextmap dtm folder
    """
    # as it is parallel required ....
    ftp = FTP("ftp.ceda.ac.uk", "", "")
    ftp.login(user=user, passwd=passwd)
    # navigate to the dir
    ftp.cwd(path)
    # I hate ESRI
    esri_types = ['dblbnd.adf', 'hdr.adf', 'prj.adf',
              'sta.adf', 'w001001.adf', 'w001001x.adf']
    # outdir in which the dinosaur format goes 
    dirname = ftp_path.split(sep="/")[6]
    os.mkdir(dirname)
    # loop through the files and write to disk
    for e in esri_types:
        localfile = os.path.join(dirname, e)
        with open(localfile, "wb") as lf:
            ftp.retrbinary('RETR ' + e, lf.write, 1024)

Parallel(n_jobs=-1, verbose=2)(delayed(dtmftp)(user, passwd, f) for f in ftplist)

_ = [dtmftp(user, passwd, f, main_dir) for f in tqdm(ftplist)]    