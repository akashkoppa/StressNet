#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Python Script for Figure 5
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
inpdir = {"glm35b": os.path.join(inpath, "e_process_2003-2015_global.nc"),
          "glmhyb": os.path.join(inpath, "e_hybrid_2003-2015_global.nc"),
          "flxcom": os.path.join(inpath, "e_fluxcom_2003-2015_global.nc")}
inpvar = {"glm35b": "E",
          "glmhyb": "E",
          "flxcom": "E"}

## main code
# read in the data
glm35b = xr.open_dataset(inpdir["glm35b"])
glmhyb = xr.open_dataset(inpdir["glmhyb"])
flxcom = xr.open_dataset(inpdir["flxcom"])

# select the required variable
glm35b = glm35b[inpvar["glm35b"]].transpose("lat","lon","time")
glmhyb = glmhyb[inpvar["glmhyb"]].transpose("lat","lon","time")
flxcom = flxcom[inpvar["flxcom"]].transpose("lat","lon","time")

# calculate seasonal averages
sea35b = glm35b.groupby("time.season").mean()
seahyb = glmhyb.groupby("time.season").mean()
seaflx = flxcom.groupby("time.season").mean()


# compute all of the seasonal estimates
sea35b = sea35b.compute()
seahyb = seahyb.compute()
seaflx = seaflx.compute()

# mask out the ocean pixels
sea35b = xr.where(np.isnan(seaflx), np.nan, sea35b)
seahyb = xr.where(np.isnan(seaflx), np.nan, seahyb)

# 
sea35b = xr.where(sea35b < 0, 0, sea35b)
sea35b = xr.where(sea35b > 150, 150, sea35b)

seahyb = xr.where(seahyb < 0, 0, seahyb)
seahyb = xr.where(seahyb > 150, 150, seahyb)

seaflx = xr.where(seaflx < 0, 0, seaflx)
seaflx = xr.where(seaflx > 150, 150, seaflx)

#
cmapdm = sci.loadmat(os.path.join(outdir, "cmap_2.mat"))
cmapdm = cmapdm["cmap_2"]
cmapdm = mp.colors.ListedColormap(cmapdm)

cmapnm = mp.colors.TwoSlopeNorm(vmin = 0.0,
                                vmax = 150.0,
                                vcenter = 75.0)


# plot the spatial maps
mm = 0.0393701
figure = mp.pyplot.figure(figsize = [183*mm, 175*mm])
season = ["JJA", "DJF"]
for i in range(0, len(season)):
    # select the required season
    datlst = [seahyb.sel(season = season[i]),
              sea35b.sel(season = season[i]),
              seaflx.sel(season = season[i])]
    
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
        figtmp = datlst[j].plot(ax = figaxi,
            transform = ca.crs.PlateCarree(),
            add_colorbar = True,
            cmap = cmapdm,
            norm = cmapnm,
            cbar_kwargs = {"orientation": "horizontal", 
                           "fraction": 0.05,
                           "pad": 0.04,
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





