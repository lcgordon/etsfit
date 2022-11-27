#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 08:21:49 2022
Last updated: Nov 27 2022
@author: lindseygordon

File containing functions to run all of a given type of fit:
"""
import os
import gc
from etsfit import etsMAIN
import etsfit.utils.utilities as ut


# lightcurveFolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
# CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
# TNSFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"
# foldersave = "/Users/lindseygordon/research/urop/paperOutput/"
# quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
# quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"
# gList = ["2018exc", "2018fhw", "2018fub", "2020tld", "2020zbo", "2018hzh", "2020hvq", 
#          "2020hdw", "2020bj", "2019gqv"]


def run_all_fits(fitType, lightcurveFolder, foldersave, TNSFile,
                 filekey = "-tessreduce",
                 goodList=None, CBV_folder=None, quaternion_folder_raw=None,
                 quaternion_folder_txt=None, 
                 fraction=None, binning=False, n1=10000, n2=40000):
    """ 
    Run a certain fit type on all light curves in a given folder
    Only runs on 1-5, 7
    ----------------------------
    Params:
        - fitType (int, 1-7) ID of fit to be run
        - lightcurveFolder (str) path to directory holding all data
            *** assumes files are formatted in the given tessreduce manner
        - foldersave (str) path to directory to put all outputs into
        - TNSfile (str) path to file containing TNS target information
        - filekey (str) end-of-file identifier for which data to use
            program will not attempt to open files without this appendix
        *
        *
        - goodList (array, optional) names of just the files to be run on
            (targets in folder not on list will be skipped)
        - CBV_folder (str, optional) folder of CBV files if needed
        - quaternion_folder_raw (str, optional) folder of quat raw files
        - quaternion_folder_txt (str, optional) folder of quat txt files
        - fraction (float 0-1, optional) percent to crop data to
        - binning (bool) whether or not to bin to 8 hours
        - n1 (int) steps for burn in (default 10k)
        - n2 (int) steps for production (default 40k)

    """
    for root, dirs, files in os.walk(lightcurveFolder):
        for name in files:
            if name.endswith(filekey):
                holder = root + "/" + name
                #load
                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder)
                if goodList is not None and targetlabel not in goodList:
                    continue
                
                #get discovery time
                discoverytime = ut.get_disctime(TNSFile, targetlabel)
                
                #run it
                trlc = etsMAIN(foldersave, TNSFile)
                
                trlc.load_single_lc(time, intensity, error, discoverytime, 
                                   targetlabel, sector, camera, ccd)
                
                if fitType in (2,4,5):
                    trlc.use_quaternions_cbvs(CBV_folder, quaternion_folder_raw, 
                                              quaternion_folder_txt)
                
                filterMade = trlc.window_rms_filt(plot=False)
                
                if "2018fhw" in targetlabel:
                    filterMade[1040:1080] = 0.0
                
                trlc.pre_run_clean(fitType, cutIndices=filterMade, 
                                   binning = binning, fraction = fraction)
                
                trlc.run_MCMC(n1, n2)
                del(trlc)
                gc.collect()
    return


def run_all_GP(GPtype, lightcurveFolder, foldersave, TNSFile,
               filekey = "-tessreduce", goodList=None, fraction=None, 
               binning=False, n1=10000, n2=40000):
    """ 
    Run a certain GP fit all light curves in a given folder
    ----------------------------
    Params:
        - fitType (str) ie, 'matern32', 'expsinsqr', 'expsqr', 'celerite'
        - lightcurveFolder (str) path to directory holding all data
            *** assumes files are formatted in the given tessreduce manner
        - foldersave (str) path to directory to put all outputs into
        - TNSfile (str) path to file containing TNS target information
        - filekey (str) end-of-file identifier for which data to use
            program will not attempt to open files without this appendix
        *
        *
        - goodList (array, optional) names of just the files to be run on
            (targets in folder not on list will be skipped)
        - fraction (float 0-1, optional) percent to crop data to
        - binning (bool) whether or not to bin to 8 hours
        - n1 (int) steps for burn in (default 10k)
        - n2 (int) steps for production (default 40k)
    """
    for root, dirs, files in os.walk(lightcurveFolder):
        for name in files:
            if name.endswith(filekey):
                holder = root + "/" + name
                #get stuff
                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder)

                if goodList is not None and targetlabel not in goodList:
                        continue
                    
                #get discovery time
                discoverytime = ut.get_disctime(TNSFile, targetlabel)
                #run it
                trlc = etsMAIN(foldersave, TNSFile)
                trlc.load_single_lc(time, intensity, error, discoverytime, 
                                   targetlabel, sector, camera, ccd)
                
                filterMade = trlc.window_rms_filt(plot=False)
                
                if "2018fhw" in targetlabel:
                    filterMade[1040:1080] = 0.0
                    
                trlc.run_GP_fit(filterMade, binning=binning, fraction=fraction, 
                               n1=n1, n2=n2, gpUSE=GPtype)

                gc.collect()

    return

def run_all_matern32comp(lightcurveFolder, foldersave, TNSFile,
                         filekey = "-tessreduce", goodList=None, fraction=None, 
                         binning=False, n1=10000, n2=40000, bounds=True):
    """ 
    Run the matern-3/2 comparison for all light curves in a given folder
    ----------------------------
    Params:
        - fitType (str) ie, 'matern32', 'expsinsqr', 'expsqr', 'celerite'
        - lightcurveFolder (str) path to directory holding all data
            *** assumes files are formatted in the given tessreduce manner
        - foldersave (str) path to directory to put all outputs into
        - TNSfile (str) path to file containing TNS target information
        - filekey (str) end-of-file identifier for which data to use
            program will not attempt to open files without this appendix
        *
        *
        - goodList (array, optional) names of just the files to be run on
            (targets in folder not on list will be skipped)
        - fraction (float 0-1, optional) percent to crop data to
        - binning (bool) whether or not to bin to 8 hours
        - n1 (int) steps for burn in (default 10k)
        - n2 (int) steps for production (default 40k)
        - bounds (bool) whether or not to bound the GP values
    """

    for root, dirs, files in os.walk(lightcurveFolder):
        for name in files:
            if name.endswith(filekey):
                holder = root + "/" + name

                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder)

                if goodList is not None and targetlabel not in goodList:
                        continue
                #get discovery time
                discoverytime = ut.get_disctime(TNSFile, targetlabel)
                #run it
                trlc = etsMAIN(foldersave, TNSFile)
                
                trlc.load_single_lc(time, intensity, error, discoverytime, 
                                   targetlabel, sector, camera, ccd, lygosbg=None)
                
                filterMade = trlc.window_rms_filt(plot=False)
                
                if "2018fhw" in targetlabel:
                    filterMade[1040:1080] = 0.0
                    
                trlc.run_both_matern32(filterMade, binning=binning, fraction=fraction,
                                       bounds=bounds)
                
                gc.collect()

    return trlc
