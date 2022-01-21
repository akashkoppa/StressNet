#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Python Script for Figure 4
-------------------------------------------------------------------------------
Article Title: A Deep Learning-Based Model of Global Terrestrial Evaporation
Author: Akash Koppa 
Affiliation: Hydro-Climate Extremes Lab (H-CEL), Ghent University, Belgium
Contact:  akash.koppa@ugent.be
-------------------------------------------------------------------------------
"""
## import required libraries
import xarray as xr
import pandas as pd
import matplotlib as mp
import cartopy as ca
import os as os
import numpy as np
import scipy.io as sci

## user defined configuration
inpath = "<< Specify path to input data here >>"
inpdir = {"glm35b": os.path.join(inpath, "st_process_2015-2019_global.nc"),
          "glmhyb": os.path.join(inpath, "st_hybrid_2015-2019_global.nc"),
          "sifpar": os.path.join(inpath, "sifpar_2015-2019_global.nc")}
inpvar = {"glm35b": "St",
          "glmhyb": "St",
          "sifpar": "SIFPAR"}

## main code
# read in the data
glm35b = xr.open_dataset(inpdir["glm35b"])
glmhyb = xr.open_dataset(inpdir["glmhyb"])
sifpar = xr.open_dataset(inpdir["sifpar"])

# select the required variable
glm35b = glm35b[inpvar["glm35b"]].transpose("lat","lon","time")
glmhyb = glmhyb[inpvar["glmhyb"]].transpose("lat","lon","time")
sifpar = sifpar[inpvar["sifpar"]].transpose("lat","lon","time")

# calculate seasonal averages
sea35b = glm35b.groupby("time.season").mean()
seahyb = glmhyb.groupby("time.season").mean()
seasif = sifpar.groupby("time.season").mean()

# compute all of the seasonal estimates
sea35b = sea35b.compute()
seahyb = seahyb.compute()
seasif = seasif.compute()

# plot the spatial maps

#
cmapdm = sci.loadmat(os.path.join(outdir, "cmap_1.mat"))
cmapdm = cmapdm["cmap_1"]
cmapdm = mp.colors.ListedColormap(cmapdm)

#
mm = 0.0393701
figure = mp.pyplot.figure(figsize = [183*mm, 175*mm])

season = ["JJA", "DJF"]
for i in range(0, len(season)):
    # select the required season
    datlst = [seahyb.sel(season = season[i]),
              sea35b.sel(season = season[i]),
              seasif.sel(season = season[i])]
    
    for j in range(0,len(datlst)):
        figaxi = figure.add_subplot(len(datlst),
                                    len(season), 
                                    (1+j*2)+i, 
                                    projection = ca.crs.EqualEarth())
        figaxi.set_global()
        figaxi.add_feature(ca.feature.LAND, edgecolor='black', zorder = 2)
        figaxi.add_feature(ca.feature.OCEAN, zorder = 0)
        figaxi.coastlines(zorder = 0)
        figaxi.gridlines(crs=ca.crs.PlateCarree(), color = "black", 
                         linestyle = "--", zorder = 1)
        if j != 2:
            figtmp = datlst[j].plot(ax = figaxi,
                           transform = ca.crs.PlateCarree(),
                           add_colorbar = True,
                           cmap = cmapd1,
                           norm = cmapn1,
                           cbar_kwargs = {"orientation": "horizontal", 
                                          "fraction": 0.05,
                                          "pad": 0.04,
                                          "format": "%.3f",
                                          "label": None},
                           zorder = 3)
            axcbar = figtmp.colorbar
            axcbar.ax.set_xticklabels(axcbar.ax.get_xticklabels(), 
                                      rotation = 45,
                                      fontsize = 7)
            figaxi.set_title("")
        else:
            figtmp = datlst[j].plot(ax = figaxi,
                           transform = ca.crs.PlateCarree(),
                           add_colorbar = True,
                           cmap = cmapdm,
                           norm = cmapnm,
                           cbar_kwargs = {"orientation": "horizontal", 
                                          "fraction": 0.05,
                                          "pad": 0.04,
                                          "format": "%.3f",
                                          "label": None},
                           zorder = 3)
            axcbar = figtmp.colorbar
            axcbar.ax.set_xticklabels(axcbar.ax.get_xticklabels(), 
                                      rotation = 45,
                                      fontsize = 7)
            
            figaxi.set_title("")
        
mp.pyplot.savefig("<< Specify output path for the figure here >>", 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 1200)






