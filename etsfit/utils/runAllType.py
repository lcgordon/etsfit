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
foldersave = "/Users/lindseygordon/research/urop/plotOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"

#trlc = etsMAIN(foldersave, bigInfoFile)
#trlc.test(opt="six", optional=True)


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
            if name.endswith("-tessreduce"):
                holder = root + "/" + name
                #print(holder)
                print(i)
                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder)
                if goodList is not None and targetlabel not in goodList:
                        continue
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
                
                filterMade = trlc.window_rms_filt(plot=False)
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

fraction = None
gList = ["2018exc", "2018fhw", "2018fub", "2020tld", "2020zbo", "2020xyw", "2020hvq", 
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
                trlc.pre_celerite_setup()
                trlc.run_GP_fit_celerite(filterMade, binYesNo=False, fraction=None, 
                               n1=7000, n2=20000, thinParams=None)
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
                trlc.pre_run_clean(1, cutIndices=filterMade, 
                                   binYesNo = False, fraction = fraction)
                trlc.run_GP_fit_tinygp(filterMade, binYesNo=False, fraction=fraction, 
                               n1=7000, n2=20000, gpUSE="expsinsqr",
                               thinParams=None)

                gc.collect()
                i+=1
    return

# run_allGP_tinygp(lightcurveFolder, foldersave, CBV_folder, 
#                   quaternion_folder_raw, 
#                   quaternion_folder_txt, bigInfoFile, gList)


def run_all_materncomp(lightcurveFolder, foldersave, CBV_folder, 
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
                trlc.pre_run_clean(1, cutIndices=filterMade, 
                                   binYesNo = False, fraction = fraction)
                trlc.run_both_matern32(filterMade, binYesNo=False, fraction=None)
                
                gc.collect()
                i+=1
    return trlc

trlc = run_all_materncomp(lightcurveFolder, foldersave, CBV_folder, 
                  quaternion_folder_raw, 
                  quaternion_folder_txt, bigInfoFile, gList)

#%%
print("celerite log sigma, log rho: ", trlc.best_mcmc[0][4:])
print("celerite sigma^2, rho: ", np.exp(trlc.best_mcmc[0][4])**2, np.exp(trlc.best_mcmc[0][5]))


print("tinygp output params: ", trlc.tinygp_soln)
print("tinygp output converted: ", np.exp(trlc.tinygp_soln["log_amps"]*2),
      np.exp(trlc.tinygp_soln["log_scales"]))

def convert_gp_params(best_mcmc, tinygp_soln):
    cel_sigma_sq = np.exp(best_mcmc[0][4])**2
    cel_rho = np.exp(trlc.best_mcmc[0][5])
    return cel_sigma_sq, cel_rho