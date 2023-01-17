#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------
Script to extract plant hydraulic traits
------------------------------------------------------------------------------
Description: This is a script to extract plant hydraulic traits from 
             Liu et al. 2022: Global ecosystem-scale plant hydraulic traits 
             retrieved using model–data fusion.
             DOI: https://doi.org/10.5194/hess-25-2399-2021. 
             
Note       : As the values are static, those values will be repeated for all 
             the dates.
        
Author     : Akash Koppa
Date       : 2022-08-31  
"""
#%% import required libraries
import xarray as xr
import pandas as pd
import os as os
import numpy as np

#%% user defined configuration (should be moved to configuration file later)
inpdir = "/media/akashkoppa/work/gleam_ml/data/input/raw/traits/hydraulic_traits"
ncfile = {"c"         : (os.path.join(inpdir, "MDF_C.nc")),
          "g1"        : (os.path.join(inpdir, "MDF_g1.nc")),
          "gpmax"     : (os.path.join(inpdir, "MDF_gpmax.nc")),
          "p50"       : (os.path.join(inpdir, "MDF_P50.nc")),
          "p50s_p50x" : (os.path.join(inpdir, "MDF_P50s_P50x.nc")),
          "vod_a"     : (os.path.join(inpdir, "MDF_VOD_a.nc")),
          "vod_b"     : (os.path.join(inpdir, "MDF_VOD_b.nc")),
          "vod_c"     : (os.path.join(inpdir, "MDF_VOD_c.nc")),
          }

ncvarn = {"c"         : "C_50",
          "g1"        : "g1_50",
          "gpmax"     : "gpmax_50",
          "p50"       : "P50_50",
          "p50s_p50x" : "P50s_P50x_50",
          "vod_a"     : "VOD_a",
          "vod_b"     : "VOD_b",
          "vod_c"     : "VOD_c",
          }

flxnet = {"stn": "/media/akashkoppa/work/gleam_ml/StressNet/stressnet_v1.0/" +
                 "input/tall_vegetation/sites_tall_vegetation.h5"}
#dafrst = "2003-01-01"
#dalast = "2020-12-31"

#%% main code
def main():
    """
    Main control script (subject to change until final version)
    NOTE: Change at your own risk

    Returns
    -------
    HDF5 file with input feature classes extracted from netcdf input files

    """
    #%% read in the fluxnet site locations
    flxsit = pd.read_hdf(path_or_buf = flxnet["stn"])
    
    #%% loop through the different variables and extract the values for the 
    #   required variables
    for i in ncfile.keys():
        print("variable under process: " + i)
        dstemp = xr.open_dataset(ncfile[i])
        outvar = nctoh5(inpdat = dstemp[ncvarn[i]],
                        inpvar = i,
                        sitinf = flxsit)
        outvar.to_hdf(path_or_buf = os.path.join(inpdir, "processed", i + ".h5"), 
                      key = i)

#%% functions
def nctoh5(inpdat, inpvar, sitinf):
    """
    function to extract data at specific locations from a netcdf file

    Parameters
    ----------
    inpdat : xarray dataarray
        a xarray dataarray containing the required variable
    inpvar : character
        name of the variable
    sitinf : pandas dataframe
        pandas dataframe with atleast the following information:
        1) Name of the stations as index of the pandas dataframe
        2) latitude with column name "lat"
        3) longitude with column name "lon"

    Returns
    -------
    A pandas dataframe with the 

    """
    
    datlst = []
    #%% loop through the stations and extract the data
    for i in sitinf.index:
        print("station under process: " + i)
        # extract the data for the station's lat and lon
        dattmp = inpdat.sel(lat = sitinf.loc[i, "lat"],
                            lon = sitinf.loc[i, "lon"],
                            method = "nearest").compute()
        dattmp = np.array(dattmp)
        # extract the time vector
        timtmp = np.array(inpdat.time)
        # create a pandas dataframe
        pdftmp = pd.DataFrame(data = dattmp, 
                              index = timtmp, 
                              columns = [i])
        datlst.append(pdftmp)
        
    # concatenate the pandas dataframes
    retn01 = pd.concat(objs = datlst, axis = 1)
    
    return retn01
        
#%% ----- run the main script -----
if __name__ == "__main__":
    main()

