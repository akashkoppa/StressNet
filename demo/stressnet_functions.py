#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:11:56 2022

@author: akashkoppa
"""

#%% import libraries
import tensorflow as tf
#import tensorflow_recommenders as tfrs
import os as os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## functions
def h5totf(filabs, filanm, filtar, sitreq, shufle, batchn, trnper):
    """
    Script to convert the input hdf5 files into tensorflow datasets which act 
    as the primary input for the deep neural network

    Parameters
    ----------
    filabs : dictionary
        full paths to input feature data (absolute values)
    filanm : dictionary
        full path to input feature data (anomalies)
    filtar : dictionary
        full path to the target variable
    sitreq : list
        list of fluxnet sites
    shufle : Boolean
        True if data needs to be shuffled
    batchn : integer
        number of batches into which the data needs to divided into
    trnper : integer
        percentage between 0 and 100 based on which the data will be 
        divided into training and testing dataset

    Returns
    -------
    retn01 : A tf.dataset with the training input features and target variables
    retn02 : A tf.dataset with the testing input features and target variables
    retn03 : A vector with min values of the input features and target variables
    retn04 : A vector with max values of the input features and target variables

    """
    
    # loop through the different input variables and create final dictionary
    inpdat = {}
    for i in filabs.keys():
        print("var under process: " + i)
        # read in the absolute values
        tmpabs = pd.read_hdf(filabs[i])
        #  read in the anomaly values
        tmpanm = pd.read_hdf(filanm[i])
        
        # store the data in dictionary
        inpdat[i] = tmpabs
        inpdat[i+"anm"] = tmpanm
    # target variable    
    tardat = {}
    for i in filtar.keys():
        print("var under process: " + i)
        tmptar = pd.read_hdf(filtar[i])
        #tmptar[tmptar > 10] = np.nan
        tmptar[tmptar > 1] = 1.0
        tardat[i] = tmptar
    
    print(inpdat.keys())
        
    # loop through the stations and create a final dataset containing only 
    # the input variables
    datlst = []
    for i in sitreq:
        print("site under process: " + i)
        lsttmp = []
        for j in inpdat.keys():
            tmpvar = inpdat[j][i]
            tmpvar.name = j
            lsttmp.append(inpdat[j][i])
        # create a data frame of variables for each 
        # append the target stress
        tartmp = tardat[list(tardat.keys())[0]][i]
        tartmp.name = list(tardat.keys())[0]
        lsttmp.append(tartmp)
        dattmp = pd.concat(lsttmp, axis = 1)
        dattmp = dattmp.replace(to_replace = -9999.0, value = np.nan)
        dattmp = dattmp.replace(to_replace = -999.0, value = np.nan )
        dattmp = dattmp.dropna(axis = 0, how = "any")
        datlst.append(dattmp)
            
    del dattmp
    
    # create the final pandas data frame
    datfin = pd.concat(datlst, ignore_index = True)
    datfin = datfin.dropna(axis = 0, how = "any")
    print(datfin.shape)
    print(datfin.columns)
   
    # reset index 
    datfin = datfin.reset_index(drop = True)
    
    # shuffle the data
    if shufle == True:
        print(" >> shuffling data")
        datfin = datfin.sample(frac = 1)
    
    # normalize by max 
    print(" >> normalizing the data by max")
    tarval = datfin.pop(list(tardat.keys())[0])
    
    # calculate the max and min for backup
    datmin = datfin.quantile(0.05)
    datmax = datfin.quantile(1.00)
    datfin = datfin/datmax
    
    # convert the pandas data frame to a tf.dataset
    print(" >> Converting data frame into a tensorflow dataset")
    print(datfin.columns)
    print(datfin.shape)
    fullda = tf.data.Dataset.from_tensor_slices((datfin.values, 
                                                 tarval.values))
    
    # split into training and testing datasets
    trnsiz = int((trnper/100) * len(datfin))
    # training dataset
    retn01 = fullda.take(trnsiz)
    # testing dataset
    retn02 = fullda.skip(trnsiz)
        
    if shufle == True:
        retn01 = retn01.shuffle(trnsiz)
        tstsiz = len(datfin) - trnsiz
        retn02 = retn02.shuffle(tstsiz)
        
    retn01 = retn01.batch(batchn)
    retn02 = retn02.batch(batchn)
    retn03 = datmin
    retn04 = datmax
    
    return retn01, retn02, retn03, retn04

# kling gupta efficiency
def kge(actual, predct):
    """
    a custom loss function based on the Kling Gupta Efficiency
    formula: [1 - sqrt((r-1)**2 + ((stddev_sim/stddev_obs)-1)**2 + ((mean_sim/mean_obs) - 1)**2)]
    reference: Decomposition of the mean squared error and NSE performance criteria: 
               Implications for improving hydrological modelling. (2009). 
               Journal of Hydrology, 377(1–2), 80–91. 
               DOI: https://doi.org/10.1016/j.jhydrol.2009.08.003

    Parameters
    ----------
    actual : tensor
        ground truth data for the predictions to be compared against
    predct : tensor
        predicted data

    Returns
    -------
    (1-kge): scalar
        The loss function to be minimized

    """
    # >>> correlation
    acmean = tf.math.reduce_mean(actual)
    pdmean = tf.math.reduce_mean(predct)
    acmdev, pdmdev = actual - acmean, predct - pdmean
    cornum = tf.math.reduce_mean(tf.multiply(acmdev, pdmdev))        
    corden = tf.math.reduce_std(acmdev) * tf.math.reduce_std(pdmdev)
    corcof = cornum / corden
    cratio = (corcof - 1)**2
    
    # variability ratio
    actstd = tf.math.reduce_std(actual)
    prestd = tf.math.reduce_std(predct)
    stdrat = prestd / actstd
    vratio = (stdrat - 1)**2
    
    # bias ratio (Beta)
    menrat = pdmean / acmean
    bratio = (menrat - 1)**2
    
    kgeval = 1 - tf.math.sqrt(cratio + vratio + bratio)
    retn01 = 1 - kgeval
    
    return retn01

# deep learning model
def funmod(inpshp, losobj, metric, optmod):
    """
    Function to create a machine learning model using the Functional API module
    of tensorflow. This module can be used to create more powerful machine
    learning models compared to the sequential module

    Parameters
    ----------
    inpshp : integer
        The number of input variables
    losobj : tf.keras.losses 
        A tensorflow loss function 
    metric : tf.keras.metrics
        A tensorflow error metric function
    optmod : character
        An optimizer which is available in tensorflow

    Returns
    -------
    retn01 : tf.keras.Model
        A compiled Functional API model

    """
    print(" >>> creating a Functional API model")
    # define the input layers
    inplyr = tf.keras.Input(shape = (inpshp, ))
    
    crslyr = tf.keras.layers.Dense(512, activation=tf.nn.swish)(inplyr)
    crslyr = tf.keras.layers.Dropout(0.45)(crslyr)
    crslyr = tf.keras.layers.Dense(256, activation=tf.nn.swish)(crslyr)
    crslyr = tf.keras.layers.Dropout(0.3)(crslyr)
    
    seqlyr = tf.keras.layers.Dense(792, activation=tf.nn.swish)(inplyr)
    seqlyr = tf.keras.layers.Dropout(0.45)(seqlyr)
    seqlyr = tf.keras.layers.Dense(512, activation=tf.nn.gelu)(seqlyr)
    seqlyr = tf.keras.layers.Dropout(0.45)(seqlyr)
  
    concat = tf.keras.layers.concatenate([crslyr, seqlyr, inplyr])
    
    # construct the last part of the model
    outlyr = tf.keras.layers.Dense(768, activation=tf.nn.swish)(concat)
    outlyr = tf.keras.layers.Dropout(0.4)(outlyr)
    outlyr = tf.keras.layers.Dense(384, activation=tf.nn.swish)(outlyr)
    outlyr = tf.keras.layers.Dropout(0.4)(outlyr)
    outlyr = tf.keras.layers.Dense(256, activation=tf.nn.swish)(outlyr)
    outlyr = tf.keras.layers.Dropout(0.4)(outlyr)

    conca1 = tf.keras.layers.concatenate([outlyr, inplyr])
    outly1 = tf.keras.layers.Dense(256, activation=tf.nn.swish)(conca1)
    outly1 = tf.keras.layers.Dropout(0.3)(outly1)
    outly1 = tf.keras.layers.Dense(172, activation=tf.nn.swish)(outlyr)
    outly1 = tf.keras.layers.Dropout(0.35)(outlyr)
    
    conca2 = tf.keras.layers.concatenate([outly1, inplyr])
    outly2 = tf.keras.layers.Dense(128, activation=tf.nn.swish)(conca2)
    outly2 = tf.keras.layers.Dropout(0.3)(outly2)
    outly2 = tf.keras.layers.Dense(64, activation=tf.nn.gelu)(outly2)
    outmod = tf.keras.layers.Dense(6)(outly2)
    
    # combine the layers into a full model
    tmpmod = tf.keras.Model(inputs = inplyr, 
                            outputs = outmod, 
                            name = "glmfun")
    
    # define a loss function 
    losobj = losobj
    # define a metric function
    metric = [metric]
    # compile the model
    tmpmod.compile(optimizer = optmod, 
                   loss = losobj, 
                   metrics = metric)
    # return the compiled model
    retn01 = tmpmod
    
    return retn01