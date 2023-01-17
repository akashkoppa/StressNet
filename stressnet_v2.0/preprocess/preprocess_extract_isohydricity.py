#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------
Script to extract isohydricity
------------------------------------------------------------------------------
Description: This is a script to extract isohydricity traits
             
Note       : As the values are static, those values will be repeated for all 
             the dates.
        
Author     : Akash Koppa
Date       : 2023-01-17 
"""
#%% import required libraries
import xarray as xr
import pandas as pd
import os as os
import numpy as np

#%% user defined configuration (should be moved to configuration file later)
inpdir = "/media/akashkoppa/work/gleam_ml/data/input/raw/traits/isohydricity"
outdir = "/media/akashkoppa/work/gleam_ml/data/input/processed/traits"
suffix = "short"
ncfile = {"iso"         : (os.path.join(inpdir, "isohydricityAMSRE_Global.nc")),
          }

ncvarn = {"iso"         : "isohydricity",
          }

flxnet = {"stn": "/media/akashkoppa/work/gleam_ml/StressNet/stressnet_v1.0/" +
                 "input/short_vegetation/sites_short_vegetation.h5"}
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
        dstemp = dstemp.assign_coords({"lon": dstemp.longitude,
                                      "lat": dstemp.latitude})
        outvar = nctoh5(inpdat = dstemp[ncvarn[i]],
                        inpvar = i,
                        sitinf = flxsit)
        outvar.to_hdf(path_or_buf = os.path.join(outdir, i + "_" + suffix + ".h5"), 
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
        #timtmp = np.array(inpdat.time)
        # create a pandas dataframe
        pdftmp = pd.Series(data = dattmp, 
                              index = [i], 
                              name = inpvar)
        datlst.append(pdftmp)
        
    #%% concatenate the pandas dataframes
    retn01 = pd.concat(objs = datlst, axis = 0)
    
    return retn01
        
#%% ----- run the main script -----
if __name__ == "__main__":
    main()

