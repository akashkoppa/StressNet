#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Python Script for Figure 2b
-------------------------------------------------------------------------------
Article Title: A Deep Learning-Based Model of Global Terrestrial Evaporation
Author: Akash Koppa 
Affiliation: Hydro-Climate Extremes Lab (H-CEL), Ghent University, Belgium
Contact: Akash Koppa (akash.koppa@ugent.be)
-------------------------------------------------------------------------------
"""
## import required libraries
import pandas as pd
import os as os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.lines as li
import seaborn as sb

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
refdat[refdat < 0.0] = 0.0

# loop through the models and calculate the correlation for every site
corall = sitdat
stdall = sitdat
rmsall = sitdat
kgeall = sitdat
for modtmp in modfil.keys():
    moddat = pd.read_hdf(modfil[modtmp])
    moddat[moddat < 0.0] = 0.0
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
        datcor = datsit["ref"].corr(datsit[modtmp], method = "spearman")
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
    
# replace all infinite values with nan
stdall = stdall.replace(float('inf'), np.nan)
corall = corall.replace(float('inf'), np.nan)
rmsall = rmsall.replace(float('inf'), np.nan)

# melt all datasets
kgevio = kgeall[["svortv", "GLEAMHybrid","GLEAMv35b"]]
kgevio = kgevio.rename(columns = {"svortv": "Vegetation Type",
                        "GLEAMHybrid": "Hybrid Model",
                        "GLEAMv35b": "Process-Based Model"})
kgevio = kgevio.melt(id_vars = "Vegetation Type")
kgevio = kgevio.rename(columns = {"value": "Kling-Gupta Efficiency"})
kgevio.loc[kgevio["Kling-Gupta Efficiency"] < -1.5, "Kling-Gupta Efficiency"] = np.nan

# plot the violin plots
mm = 0.0393701
sb.set_theme(style = "darkgrid")
sb.set_style("ticks")
figure = mp.pyplot.figure(figsize = (89*mm, 89*mm))
figaxi = figure.add_subplot(1, 1, 1)
figaxi.set_title("Evaporation ($E$)", fontsize = 8)
figaxi = sb.violinplot(x = "Vegetation Type",
                   y = "Kling-Gupta Efficiency",
                   hue = "variable",
                   split = "true",
                   data = kgevio,
                   inner = "quartile",
                   palette = "Set2",
                   fontsize = 7,
                   linewidth = 1.0,
                   edgecolor = "black",
                   order = ["Short", "Tall"])
plt.legend(loc = "lower left", edgecolor = "black", fontsize = 7)
yticks = figaxi.get_yticks()
yticks[yticks == -0.5] =  -0.41
figaxi.set_yticks(yticks)
figaxi.set_ylim(-2.0)
figaxi.set_xlabel(figaxi.get_xlabel(), fontsize = 8)
figaxi.set_ylabel(figaxi.get_ylabel(), fontsize = 8)
figaxi.tick_params(axis='both', which='major', labelsize=7)
plt.axhline(-0.41, color = "red", 
            linestyle = "solid",
            linewidth = 1.0)
figure.tight_layout()
plt.savefig("<< Specify output path for the figure here >>")

