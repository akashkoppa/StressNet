#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Python Script for Figure 6 (Transpiration Stress)
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

# calculate spatial correlations
spacor = {}
season = ["DJF", "JJA", "MAM", "SON"]
for i in season:
    dattmp = pd.DataFrame({"pm": np.array(sea35b.sel(season = i)).flatten(),
                           "dm": np.array(seaflx.sel(season = i)).flatten()})
    spacor["PM" + "-" + i] = dattmp["pm"].corr(dattmp["dm"])
    dattmp = pd.DataFrame({"hm": np.array(seahyb.sel(season = i)).flatten(),
                           "dm": np.array(seaflx.sel(season = i)).flatten()})    
    spacor["HM" + "-" + i] = dattmp["hm"].corr(dattmp["dm"])
    
# temporal correlation maps
glm35b = glm35b.compute()
glmhyb = glmhyb.compute()
flxcom = flxcom.compute()

#
glm35b = xr.where(glm35b < 0, 0, glm35b)
glmhyb = xr.where(glmhyb < 0, 0, glmhyb)
flxcom = xr.where(flxcom < 0, 0, flxcom)

# some preprocessing
flxcom["time"] = glm35b["time"]
glmhyb["time"] = glm35b["time"]
glm35b = xr.where(xr.ufuncs.isnan(flxcom), np.nan, glm35b)
glmhyb = xr.where(xr.ufuncs.isnan(flxcom), np.nan, glmhyb)

# calculate temporal correlations
cor35b = xr.corr(glm35b, flxcom, dim = "time")
corhyb = xr.corr(glmhyb, flxcom, dim = "time")
cordif = corhyb - cor35b

#
datlst = [corhyb, cor35b, cordif]

#
mm = 0.0393701
cmapnm = mp.colors.TwoSlopeNorm(vmin= -1.0, vmax = 1.0, vcenter=0.0)

figure = mp.pyplot.figure(figsize = [89*mm, 185*mm])
for j in range(0,len(datlst)):
    figaxi = figure.add_subplot(3, 
                                1, 
                                j+1, 
                                projection = ca.crs.EqualEarth())
    figaxi.set_global()

    figaxi.add_feature(ca.feature.LAND, edgecolor='black', zorder = 2)
    figaxi.add_feature(ca.feature.OCEAN, zorder = 0)
    figaxi.coastlines(zorder = 0)
    figaxi.gridlines(crs=ca.crs.PlateCarree(), color = "black", 
                     linestyle = "--", zorder = 1)
    if j!= 2:
        figtmp = datlst[j].plot(ax = figaxi,
                       transform = ca.crs.PlateCarree(),
                       add_colorbar = True,
                       cmap = "RdBu",
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
    else:
        figtmp = datlst[j].plot(ax = figaxi,
                       transform = ca.crs.PlateCarree(),
                       add_colorbar = True,
                       cmap = "RdBu",
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
mp.pyplot.savefig("<< Specify output path for the figure here >>"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 1200)










    

    



