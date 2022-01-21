#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Python Script for Figure 3 (Evaporation (E))
-------------------------------------------------------------------------------
Article Title: A Deep Learning-Based Model of Global Terrestrial Evaporation
Author: Akash Koppa 
Affiliation: Hydro-Climate Extremes Lab (H-CEL), Ghent University, Belgium
Contact:  akash.koppa@ugent.be
-------------------------------------------------------------------------------
"""
## import required libraries
import geopandas as gp
import pandas as pd
import pickle as pi
import matplotlib as mp
import cartopy as ca
import os as os
import numpy as np
import matplotlib.lines as li

## user defined configuration
inpdir = "<< Specify path to input data here >>"
reffil = {"ref": os.path.join(inpdir, "e_observed_sites.h5")}
modfil = {"mo1": os.path.join(inpdir, "e_process_sites.h5"),
          "mo2": os.path.join(inpdir, "e_process_sites.h5")}
modmap = {"ref": "FLUXNET",
          "mo1": "GLEAMv35b",
          "mo2": "GLEAMHybrid"}
figmap = {"ref": "FLUXNET",
          "mo1": "Process-Based Model",
          "mo2": "Hybrid Model"}
marmap = {"ref": "FluxMarker",
          "mo1": "GLEAMv35bMarker",
          "mo2": "GLEAMHybridMarker"}
sizmap = {"ref": "FluxSize",
          "mo1": "GLEAMv35bSize",
          "mo2": "GLEAMHybridSize"}
sitfil = {"siteda": os.path.join(inpdir, "sites.h5")}

## main code
# read in the site data
sitdat = pd.read_hdf(sitfil["siteda"])

# read in the reference FLUXNET data
refdat = pd.read_hdf(reffil["ref"])
refdat[refdat < 0] = np.nan

# subset sitdat
sitdat = sitdat.loc[refdat.keys()]

# loop through the models and calculate the correlation for every site
corall = sitdat
stdall = sitdat
rmsall = sitdat
kgeall = sitdat
for modtmp in modfil.keys():
    moddat = pd.read_hdf(modfil[modtmp])
    cortmp = []
    stdtmp = []
    rmstmp = []
    kgetmp = []
    # loop through the sites and calculate correlation and std 
    for sittmp in sitdat.index:

        refsit = refdat[sittmp]
        refsit.name = "ref"
        modsit = moddat[sittmp]
        modsit.name = modtmp

        datsit = pd.concat([refsit, modsit], axis = 1)
        datsit = datsit.dropna(how = "any")
        datcor = datsit["ref"].corr(datsit[modtmp])
        modstd = datsit[modtmp].std()
        datrms = ((datsit[modtmp] - datsit["ref"])**2).mean() ** 0.5
        # kge
        corrat = (datcor - 1)**2
        stdrat = ((modstd/datsit["ref"].std()) - 1)**2
        menrat = ((datsit[modtmp].mean()/datsit["ref"].mean()) - 1)**2
        kgeval = 1 - np.sqrt(corrat + stdrat + menrat)
        # append
        kgetmp.append(kgeval)
        cortmp.append(datcor)
        stdtmp.append(modstd)
        rmstmp.append(datrms)
        
    # create a pandas series from the correlation and standard deviation data
    cortm1 = pd.Series(cortmp, index = sitdat.index, name = modmap[modtmp])
    stdtm1 = pd.Series(stdtmp, index = sitdat.index, name = modmap[modtmp])
    rmstm1 = pd.Series(rmstmp, index = sitdat.index, name = modmap[modtmp])
    kgetm1 = pd.Series(kgetmp, index = sitdat.index, name = modmap[modtmp])
    
    # append the data to the final data frames
    corall = pd.concat([corall, cortm1], axis = 1)
    stdall = pd.concat([stdall, stdtm1], axis = 1)
    rmsall = pd.concat([rmsall, rmstm1], axis = 1)
    kgeall = pd.concat([kgeall, kgetm1], axis = 1)

# calculate the KGE difference between GLEAM-Hybrid and GLEAM v35b
kgeall["Difference"] = kgeall["GLEAMHybrid"] - kgeall["GLEAMv35b"]
    
# convert site data into a geopandas data frame
datplt = kgeall
datplt = gp.GeoDataFrame(datplt, geometry = gp.points_from_xy(datplt["lon"], 
                                                              datplt["lat"]))
datplt["marker"] = "o"
datplt.loc[datplt["svortv"] == "Tall", "marker"] = "v"
datplt["marker"] = datplt["marker"].astype(str)
cmapnm = mp.colors.TwoSlopeNorm(vmin= -1, vmax = 1, vcenter=0)

# US
mm = 0.0393701
figure = mp.pyplot.figure(figsize = (60*mm,89*mm))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.AlbersEqualArea(central_longitude = -103.0,
                                                                central_latitude = 44.5))
figaxi.add_feature(ca.feature.LAND, 
                   edgecolor='black',
                   linewidth = 0.25,
                   alpha = 1.0, 
                   zorder = 2)
figaxi.add_feature(ca.feature.OCEAN, zorder = 0)
figaxi.coastlines(zorder = 0,linewidth = 0.25)

subplt = datplt.loc[  (datplt["lat"] > 0) & (datplt["lat"] < 80) 
                    & (datplt["lon"] > -165) & (datplt["lon"] < -65)]

subplt.loc[subplt["svortv"] == "Short"].plot(ax = figaxi,
            transform = ca.crs.PlateCarree(),
            column = subplt.loc[subplt["svortv"] == "Short","Difference"], 
            categorical = False,
            legend = False,
            marker = "o",
            edgecolor = "black",
            linewidth = 0.25,
            markersize = 12,
            cmap = "RdBu",
            norm = cmapnm,
            legend_kwds= {"orientation": "horizontal",
                          "shrink": 0.9,
                          "pad": 0.01,
                          "anchor": [0.5, 1.0]},
            zorder = 3)

subplt.loc[subplt["svortv"] == "Tall"].plot(ax = figaxi,
            transform = ca.crs.PlateCarree(),
            column = subplt.loc[subplt["svortv"] == "Tall","Difference"], 
            categorical = False,
            legend = False,
            marker = "v",
            edgecolor = "black",
            linewidth = 0.25,
            markersize = 12,
            cmap = "RdBu",
            norm = cmapnm,
            legend_kwds= {"orientation": "horizontal",
                          "shrink": 0.9,
                          "pad": 0.01,
                          "anchor": [0.5, 1.0]},
            zorder = 3)

figure.patch.set_alpha(0)

mp.pyplot.savefig(os.path.join("<< Specify output path for the figure here >>"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 1200,
                  )
del figure
del figaxi

# EU

figure = mp.pyplot.figure(figsize = (30*mm,89*mm))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.AlbersEqualArea(central_longitude =0.0,
                                                                central_latitude = 60))
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder = 2,
                   linewidth = 0.25)
figaxi.add_feature(ca.feature.OCEAN)
figaxi.coastlines(linewidth = 0.25)

subplt = datplt.loc[  (datplt["lat"] > 20) & (datplt["lat"] < 90) 
                    & (datplt["lon"] > -30) & (datplt["lon"] < 30)]

subplt.loc[subplt["svortv"] == "Short"].plot(ax = figaxi,
            transform = ca.crs.PlateCarree(),
            column = subplt.loc[subplt["svortv"] == "Short","Difference"], 
            categorical = False,
            legend = False,
            marker = "o",
            edgecolor = "black",
            linewidth = 0.25,
            markersize = 12,
            cmap = "RdBu",
            norm = cmapnm,
            legend_kwds= {"orientation": "horizontal",
                          "shrink": 0.72,
                          "pad": 0.01,
                          "anchor": [0.5, 1.0]},
            zorder = 3)

subplt.loc[subplt["svortv"] == "Tall"].plot(ax = figaxi,
            transform = ca.crs.PlateCarree(),
            column = subplt.loc[subplt["svortv"] == "Tall","Difference"], 
            categorical = False,
            legend = False,
            marker = "v",
            edgecolor = "black",
            linewidth = 0.25,
            markersize = 12,
            cmap = "RdBu",
            norm = cmapnm,
            legend_kwds= {"orientation": "horizontal",
                          "shrink": 0.9,
                          "pad": 0.01,
                          "anchor": [0.5, 1.0]},
            zorder = 3)

figure.patch.set_alpha(0)

mp.pyplot.savefig("<< Specify output path for the figure here >>"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 1200,
                  )
del figure
del figaxi

# AS
mp.pyplot.rc('axes', axisbelow = True)
figure = mp.pyplot.figure(figsize = (30*mm,89*mm))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.AlbersEqualArea(central_longitude =100,
                                                                central_latitude = 45))
figaxi.add_feature(ca.feature.LAND, edgecolor='black', linewidth = 0.25,
                   alpha = 1.00, zorder = 2)
figaxi.add_feature(ca.feature.OCEAN, zorder = 0)
figaxi.coastlines(zorder = 0, linewidth = 0.25)

subplt = datplt.loc[  (datplt["lat"] > 0) & (datplt["lat"] < 90) 
                    & (datplt["lon"] > 60) & (datplt["lon"] < 180)]

subplt.loc[subplt["svortv"] == "Short"].plot(ax = figaxi,
            transform = ca.crs.PlateCarree(),
            column = subplt.loc[subplt["svortv"] == "Short","Difference"], 
            categorical = False,
            legend = False,
            marker = "o",
            edgecolor = "black",
            linewidth = 0.25,
            markersize = 12,
            cmap = "RdBu",
            norm = cmapnm,
            legend_kwds= {"orientation": "horizontal",
                          "shrink": 0.69,
                          "pad": 0.01,
                          "anchor": [0.5, 1.0]},
            zorder = 3)

subplt.loc[subplt["svortv"] == "Tall"].plot(ax = figaxi,
            transform = ca.crs.PlateCarree(),
            column = subplt.loc[subplt["svortv"] == "Tall","Difference"], 
            categorical = False,
            legend = False,
            marker = "v",
            edgecolor = "black",
            linewidth = 0.25,
            markersize = 12,
            cmap = "RdBu",
            norm = cmapnm,
            legend_kwds= {"orientation": "horizontal",
                          "shrink": 0.9,
                          "pad": 0.01,
                          "anchor": [0.5, 1.0]},
            zorder = 3)

figure.patch.set_alpha(0)

mp.pyplot.savefig("<< Specify output path for the figure here >>"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 1200,
                  )
del figure
del figaxi


# SH
figure = mp.pyplot.figure(figsize = (120*mm,44*mm))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Mercator())
figaxi.add_feature(ca.feature.LAND, edgecolor='black', 
                   linewidth = 0.25, alpha = 1.00, zorder = 2)
figaxi.add_feature(ca.feature.OCEAN, zorder = 0)
figaxi.coastlines(zorder = 0)

subplt = datplt.loc[  (datplt["lat"] < 15) & (datplt["lat"] > -90)]

subplt.loc[subplt["svortv"] == "Short"].plot(ax = figaxi,
            transform = ca.crs.PlateCarree(),
            column = subplt.loc[subplt["svortv"] == "Short","Difference"], 
            categorical = False,
            legend = False,
            marker = "o",
            edgecolor = "black",
            linewidth = 0.25,
            markersize = 12,
            cmap = "RdBu",
            norm = cmapnm,
            legend_kwds= {"orientation": "horizontal",
                          "shrink": 0.9,
                          "pad": 0.01,
                          "anchor": [0.5, 1.0]},
            zorder = 3)

subplt.loc[subplt["svortv"] == "Tall"].plot(ax = figaxi,
            transform = ca.crs.PlateCarree(),
            column = subplt.loc[subplt["svortv"] == "Tall","Difference"], 
            categorical = False,
            legend = False,
            marker = "v",
            edgecolor = "black",
            linewidth = 0.25,
            markersize = 12,
            cmap = "RdBu",
            norm = cmapnm,
            legend_kwds= {"orientation": "horizontal",
                          "shrink": 0.9,
                          "pad": 0.01,
                          "anchor": [0.5, 1.0]},
            zorder = 3)

figure.patch.set_alpha(0)
mp.pyplot.savefig("<< Specify output path for the figure here >>"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 1200,
                  )
del figure
del figaxi
