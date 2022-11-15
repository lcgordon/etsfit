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

from astropy.time import Time
import gc
from etsfit import etsMAIN
import etsfit.utils.utilities as ut
#import etsfit


lightcurveFolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
bigInfoFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"
foldersave = "/Users/lindseygordon/research/urop/paperOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"


def run_all_fits(fitType, lightcurveFolder, foldersave, CBV_folder, 
                 quaternion_folder_raw, 
                 quaternion_folder_txt, bigInfoFile, fraction=None,
                 goodList = None):
    """ 
    run of all a certain type of fit w/ otherwise default parameters
    """
    info = pd.read_csv(bigInfoFile)
    i = 0
    for root, dirs, files in os.walk(lightcurveFolder):
        for name in files:
            if name.endswith("-tessreduce") and i==0:
                holder = root + "/" + name
                #print(holder)
                #print(i)
                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder)
                #print(targetlabel, sector)
                if goodList is not None and targetlabel not in goodList:
                    #print("skipping")
                    continue
                #get discovery time
                d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                #run it
                trlc = etsMAIN(foldersave, bigInfoFile)
                
                trlc.load_single_lc(time, intensity, error, discoverytime, 
                                   targetlabel, sector, camera, ccd, lygosbg=None)
                
                if fitType in (2,4,5):
                    trlc.use_quaternions_cbvs(CBV_folder, quaternion_folder_raw, 
                                              quaternion_folder_txt)
                
                filterMade = trlc.window_rms_filt(plot=False)
                
                if "2018fhw" in targetlabel:
                    filterMade[1040:1080] = 0.0
                
                trlc.pre_run_clean(fitType, cutIndices=filterMade, 
                                   binYesNo = False, fraction = fraction)
                trlc.run_MCMC(n1=10000, n2=60000, thinParams = None,
                             saveBIC=False, args=None, logProbFunc = None, 
                             plotFit = None,
                             filesavetag=None,
                             labels=None, init_values=None)
                #del(loadedraw)
                del(trlc)
                gc.collect()
                i+=1
    return

fraction = 0.6
gList = ["2018exc", "2018fhw", "2018fub", "2020tld", "2020zbo", "2018hzh", "2020hvq", 
         "2020hdw", "2020bj", "2019gqv"]
# run_all_fits(1, lightcurveFolder, foldersave, CBV_folder, 
#                   quaternion_folder_raw, 
#                   quaternion_folder_txt, bigInfoFile, fraction=fraction, goodList = gList)

def run_allGP_celerite(lightcurveFolder, foldersave, CBV_folder, 
                 quaternion_folder_raw, 
                 quaternion_folder_txt, bigInfoFile, 
                 goodList = None):
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
                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder)

                if goodList is not None and targetlabel not in goodList:
                        continue
                #get discovery time
                d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                #run it
                trlc = etsMAIN(foldersave, bigInfoFile)
                
                trlc.load_single_lc(time, intensity, error, discoverytime, 
                                   targetlabel, sector, camera, ccd, lygosbg=None)
                
                filterMade = trlc.window_rms_filt(plot=False)
                if "2018fhw" in targetlabel:
                    filterMade[1040:1080] = 0.0
                trlc.run_GP_fit(filterMade, binYesNo=False, fraction=fraction, 
                                      n1=10000, n2=25000, gpUSE = "celerite",
                                      thinParams=None, customSigmaRho=None, 
                                      filesavetag=None, bounds=True)
                #del(loadedraw)
                del(trlc)
                gc.collect()
                i+=1
    return

# run_allGP_celerite(lightcurveFolder, foldersave, CBV_folder, 
#                   quaternion_folder_raw, 
#                   quaternion_folder_txt, bigInfoFile, gList)

def run_allGP_tinygp(lightcurveFolder, foldersave, CBV_folder, 
                 quaternion_folder_raw, 
                 quaternion_folder_txt, bigInfoFile, 
                 goodList = None):
    """ 
    run of all a certain type of fit w/ otherwise default parameters
    """
    info = pd.read_csv(bigInfoFile)
    i = 0
    for root, dirs, files in os.walk(lightcurveFolder):
        for name in files:
            if name.endswith("-tessreduce") and i==0:
                holder = root + "/" + name
                print(i)
                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder)

                if goodList is not None and targetlabel not in goodList:
                        continue
                #get discovery time
                d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                #run it
                trlc = etsMAIN(foldersave, bigInfoFile)
                trlc.load_single_lc(time, intensity, error, discoverytime, 
                                   targetlabel, sector, camera, ccd, lygosbg=None)
                filterMade = trlc.window_rms_filt(plot=False)
                if "2018fhw" in targetlabel:
                    filterMade[1040:1080] = 0.0
                # trlc.pre_run_clean(1, cutIndices=filterMade, 
                #                    binYesNo = False, fraction = fraction)
                trlc.run_GP_fit(filterMade, binYesNo=False, fraction=fraction, 
                               n1=10000, n2=25000, gpUSE="matern32",
                               thinParams=None)

                gc.collect()
                i+=1
    return

run_allGP_tinygp(lightcurveFolder, foldersave, CBV_folder, 
                  quaternion_folder_raw, 
                  quaternion_folder_txt, bigInfoFile, gList)


def run_all_materncomp(lightcurveFolder, foldersave, CBV_folder, 
                 quaternion_folder_raw, 
                 quaternion_folder_txt, bigInfoFile, fraction, bounds,
                 goodList = None):
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
                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder)

                if goodList is not None and targetlabel not in goodList:
                        continue
                #get discovery time
                d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                #run it
                trlc = etsMAIN(foldersave, bigInfoFile)
                
                trlc.load_single_lc(time, intensity, error, discoverytime, 
                                   targetlabel, sector, camera, ccd, lygosbg=None)
                
                filterMade = trlc.window_rms_filt(plot=False)
                if "2018fhw" in targetlabel:
                    filterMade[1040:1080] = 0.0
                    
                trlc.run_both_matern32(filterMade, binYesNo=False, fraction=fraction,
                                       bounds=bounds)
                
                gc.collect()
                i+=1
    return trlc

# trlc = run_all_materncomp(lightcurveFolder, foldersave, CBV_folder, 
#                   quaternion_folder_raw, 
#                   quaternion_folder_txt, bigInfoFile, fraction=fraction,
#                   bounds=True, goodList = gList)

