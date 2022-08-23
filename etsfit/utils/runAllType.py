#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 08:21:49 2022

@author: lindseygordon

File containing functions to run all of a given type of fit:
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from etsfit import etsMAIN
from astropy.time import Time
import gc




lightcurveFolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
bigInfoFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"
foldersave = "/Users/lindseygordon/research/urop/plotOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"


def run_all_fits(fitType, lightcurveFolder, foldersave, CBV_folder, 
                 quaternion_folder_raw, 
                 quaternion_folder_txt, bigInfoFile):
    """ 
    run of all a certain type of fit w/ otherwise default parameters
    """
    info = pd.read_csv(bigInfoFile)
    i = 0
    for root, dirs, files in os.walk(lightcurveFolder):
        for name in files:
            if name.endswith("-tessreduce"):
                holder = root + "/" + name
                #print(holder)
                print(i)
                loadedraw = pd.read_csv(holder)
                time = Time(loadedraw["time"], format='mjd').jd
                intensity = loadedraw["flux"].to_numpy()
                error = loadedraw["flux_err"].to_numpy()
                #p
                fulllabel = holder.split("/")[-1].split("-")[0]
                targetlabel = fulllabel[0:7]
                if targetlabel[-1].isdigit():
                    targetlabel=targetlabel[0:6]
                sector = fulllabel[-4:-2]
                camera = fulllabel[-2]
                ccd = fulllabel[-1]
                print(targetlabel, sector, camera, ccd)
                #get discovery time
                d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                #print(discoverytime)
                #run it
                trlc = etsMAIN(foldersave, bigInfoFile)
                
                trlc.load_single_lc(time, intensity, error, discoverytime, 
                                   targetlabel, sector, camera, ccd, lygosbg=None)
                
                if fitType in (2,4,5):
                    trlc.use_quaternions_cbvs(CBV_folder, quaternion_folder_raw, 
                                              quaternion_folder_txt)
                
                filterMade = trlc.window_rms_filt()
                trlc.pre_run_clean(fitType, cutIndices=filterMade, 
                                   binYesNo = False, fraction = None)
                trlc.run_MCMC(n1=10000, n2=60000, thinParams = None,
                             saveBIC=False, args=None, logProbFunc = None, 
                             plotFit = None,
                             filesavetag=None,
                             labels=None, init_values=None)
                del(loadedraw)
                del(trlc)
                gc.collect()
                i+=1
    return

run_all_fits(1, lightcurveFolder, foldersave, CBV_folder, 
                  quaternion_folder_raw, 
                  quaternion_folder_txt, bigInfoFile)

def run_allGP(lightcurveFolder, foldersave, CBV_folder, 
                 quaternion_folder_raw, 
                 quaternion_folder_txt, bigInfoFile):
    """ 
    run of all a certain type of fit w/ otherwise default parameters
    """
    info = pd.read_csv(bigInfoFile)
    i = 0
    for root, dirs, files in os.walk(lightcurveFolder):
        for name in files:
            if name.endswith("-tessreduce"):
                holder = root + "/" + name
                print(i)
                loadedraw = pd.read_csv(holder)
                time = Time(loadedraw["time"], format='mjd').jd
                intensity = loadedraw["flux"].to_numpy()
                error = loadedraw["flux_err"].to_numpy()
                #p
                fulllabel = holder.split("/")[-1].split("-")[0]
                targetlabel = fulllabel[0:7]
                sector = fulllabel[-4:-2]
                camera = fulllabel[-2]
                ccd = fulllabel[-1]
                if targetlabel[-1].isdigit():
                    targetlabel=targetlabel[0:6]
                print(targetlabel, sector, camera, ccd)
                #get discovery time
                d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                #print(discoverytime)
                #run it
                trlc = etsMAIN(foldersave, bigInfoFile)
                
                trlc.load_single_lc(time, intensity, error, discoverytime, 
                                   targetlabel, sector, camera, ccd, lygosbg=None)
                
                filterMade = trlc.window_rms_filt()
                trlc.run_GP_fit(filterMade, binYesNo=False, fraction=None, 
                               n1=10000, n2=40000, filesavetag=None,
                               customSigmaRho = None, thinParams=None)
                del(loadedraw)
                del(trlc)
                gc.collect()
                i+=1
    return

# run_allGP(lightcurveFolder, foldersave, CBV_folder, 
#                   quaternion_folder_raw, 
#                   quaternion_folder_txt, bigInfoFile)