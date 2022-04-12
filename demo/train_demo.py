#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Python Script for Training the Deep Learning Model (StressNet) for 
Transpiration Stress (Tall Vegetation)
-------------------------------------------------------------------------------
Author: Akash Koppa 
Affiliation: Hydro-Climate Extremes Lab (H-CEL), Ghent University, Belgium
Contact:  akash.koppa@ugent.be
-------------------------------------------------------------------------------
Reference:
Koppa, A., Rains, D., Hulsman, P., Poyatos, R., and Miralles, D. G. 
A deep learning-based hybrid model of global terrestrial evaporation. 
Nature Communications 13, 1912 (2022). https://doi.org/10.1038/s41467-022-29543-7
-------------------------------------------------------------------------------
"""

#%% import libraries
import tensorflow as tf
#import tensorflow_recommenders as tfrs
import os as os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stressnet_functions as sf

#%% user defined configuration
inpdir = "<< Specify path to input data here >>" 

# station
flxnet = {"stn": (os.path.join(inpdir, "sites_demo.h5"))} # sites
# input features (absolute values)
absfil = {"ate": (os.path.join(inpdir, "ate_demo.h5")), # air temperature
          "co2": (os.path.join(inpdir, "co2_demo.h5")), # carbon di oxide
          "ssh": (os.path.join(inpdir, "ssh_demo.h5")), # plant available water
          "swi": (os.path.join(inpdir, "swi_demo.h5")), # incoming shortwave radiation
          "vod": (os.path.join(inpdir, "vod_demo.h5")), # vegetation optical depth
          "vpd": (os.path.join(inpdir, "vpd_demo.h5"))} # vapor pressure deficit

# input features (anomaly values)
anmfil = {"ate": (os.path.join(inpdir, "ate_anomaly_demo.h5")),
          "co2": (os.path.join(inpdir, "co2_anomaly_demo.h5")),
          "ssh": (os.path.join(inpdir, "ssh_anomaly_demo.h5")),
          "swi": (os.path.join(inpdir, "swi_anomaly_demo.h5")),
          "vod": (os.path.join(inpdir, "vod_anomaly_demo.h5")),
          "vpd": (os.path.join(inpdir, "vpd_anomaly_demo.h5"))}

# target variable
tarfil = {"str": (os.path.join(inpdir, "str_demo.h5"))} # transpiration stress

# output path for the final trained StressNet
outdir = "<< Specify path to input data here >>"
outfil = os.path.join(outdir, "stressnet_demo")

#%% main code
    
# read in the fluxnet site locations
flxsit = pd.read_hdf(path_or_buf = flxnet["stn"],   key="siteda")
flxsit = flxsit.dropna(how = "any")
    
## select the required cluster
sitreq = flxsit.index
    
## create a combined tensorflow dataset
trndat, tstdat, sclmin, sclmax = sf.h5totf(filabs = absfil, 
                                            filanm = anmfil, 
                                            filtar = tarfil,
                                            sitreq = sitreq,
                                            shufle = True,
                                            batchn = 100,
                                            trnper = 85)
    
## get the required model 
tstmod = sf.funmod(inpshp = 12, 
                   losobj = sf.kge, # specifiy the loss function (KGE in this case)
                   metric = sf.kge, # specify the validation metric (KGE in this case)
                   optmod = tf.keras.optimizers.Adam(learning_rate = 0.000142)) # specify the optimizer you want to use.  

## train the dataset
histst = tstmod.fit(trndat, 
                    epochs = 700, # specify the number of epochs or iterations
                    validation_data = tstdat)

## plot evolution
evolut = pd.DataFrame(histst.history)
evorms = evolut[["kge","val_kge"]]
evorms.plot()
    
# save the model
tstmod.save(outfil)
    

































