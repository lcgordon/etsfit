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


def run_all_fits(fitType, data_dir, save_dir, TNSFile,
                 filekey = "-tessreduce",
                 goodList=None, cbv_dir=None, quaternion_raw_dir=None,
                 quaternion_txt_dir=None, 
                 fraction=None, binning=False, n1=10000, n2=40000):
    """ 
    Run a certain fit type on all light curves in a given folder
    Only runs on 1-5, 7
    ----------------------------
    Params:
        - fitType (int, 1-7) ID of fit to be run
        - data_dir (str) path to directory holding all data
            *** assumes files are formatted in the given tessreduce manner
        - save_dir (str) path to directory to put all outputs into
        - TNSfile (str) path to file containing TNS target information
        - filekey (str) end-of-file identifier for which data to use
            program will not attempt to open files without this appendix
        *
        *
        - goodList (array, optional) names of just the files to be run on
            (targets in folder not on list will be skipped)
        - cbv_dir (str, optional) folder of CBV files if needed
        - quaternion_raw_dir (str, optional) folder of quat raw files
        - quaternion_txt_dir (str, optional) folder of quat txt files
        - fraction (float 0-1, optional) percent to crop data to
        - binning (bool) whether or not to bin to 8 hours
        - n1 (int) steps for burn in (default 10k)
        - n2 (int) steps for production (default 40k)
        
    
    # trlc = run_all_fits(1, data_dir, save_dir, TNSFile,
    #                  filekey = "-tessreduce",
    #                  goodList=gList, cbv_dir=None, quaternion_raw_dir=None,
    #                  quaternion_txt_dir=None, 
    #                      fraction=0.6, binning=False, n1=10000, n2=40000)

    """
    i = 0
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith(filekey):
                fname = root + "/" + name
                #load
                (time, flux, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(fname)
                if goodList is not None and targetlabel not in goodList:
                    continue
                
                #get discovery time
                discoverytime = ut.get_disctime(TNSFile, targetlabel)
                
                #run it
                trlc = etsMAIN(save_dir, TNSFile)
                
                trlc.load_single_lc(time, flux, error, discoverytime, 
                                   targetlabel, sector, camera, ccd)
                
                if fitType in (2,4,5):
                    trlc.use_quaternions_cbvs(cbv_dir, quaternion_raw_dir, 
                                              quaternion_txt_dir)
                
                winfilter = trlc.window_rms_filt()
                
                if "2018fhw" in targetlabel:
                    winfilter[1040:1080] = 0.0
                if "2020hdw" in targetlabel:
                    winfilter[0:45] = 0.0
                    winfilter[610:685] = 0.0
                
                trlc.pre_run_clean(fitType, flux_mask=winfilter, 
                                   binning = binning, fraction = fraction)
                #trlc.test_plot()
                trlc.run_MCMC(n1, n2, quiet=True)
                #del(trlc)
                gc.collect()
                i=i+1
    return trlc


data_dir = "/Users/lindseygordon/research/urop/tessreduce_lc/"
cbv_dir = "/Users/lindseygordon/research/urop/eleanor_cbv/"
TNSFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"
save_dir = "/Users/lindseygordon/research/urop/paperOutput/"
quaternion_raw_dir = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_txt_dir = "/Users/lindseygordon/research/urop/quaternions-txt/"
gList = ["2018exc", "2018fhw", "2018fub", "2020tld", "2020zbo", "2018hzh", "2020hvq", 
          "2020hdw", "2020bj", "2019gqv"]

run_all_fits(1, data_dir, save_dir, TNSFile,
                  filekey = "-tessreduce",
                  goodList=[gList[0]], cbv_dir=None, quaternion_raw_dir=None,
                  quaternion_txt_dir=None, 
                  fraction=0.8, binning=False, n1=5_000, n2=40_000)


def run_all_GP(GPtype, data_dir, save_dir, TNSFile,
               filekey = "-tessreduce", goodList=None, fraction=None, 
               binning=False, n1=10000, n2=40000, usebounds=True,
               cbounds=None):
    """ 
    Run a certain GP fit all light curves in a given folder
    ----------------------------
    Params:
        - fitType (str) ie, 'matern32', 'expsinsqr', 'expsqr', 'celerite_residual',
        'celerite_mean'
        - data_dir (str) path to directory holding all data
            *** assumes files are formatted in the given tessreduce manner
        - save_dir (str) path to directory to put all outputs into
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
        - bounds (bool) t/f using tight bounds
        - cbounds (dict/none) custom bounds dict, containing entries for:
            - log_sigma
            - log_rho
            - boundlabel (string)
            
    ex: 
        import numpy as np
        rho_bounds = np.log((0.25, 10)) #0, 2.302
        sigma_bounds = np.log( np.sqrt((0.1, 20)) ) #sigma range 0.316 to 4.47, take log
        bounds_dict = dict(log_sigma=sigma_bounds, log_rho=rho_bounds, boundlabel="-0-25day")

        trlc = run_all_GP('celerite_residual', data_dir, save_dir, TNSFile,
                          filekey = "-tessreduce",
                          goodList=gList, 
                          fraction=0.6, binning=False, n1=5000, n2=25000, bounds=True,
                          cbounds=bounds_dict)
    """
    i = 0
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith(filekey):
                fname = root + "/" + name
                #get stuff
                (time, flux, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(fname)

                if goodList is not None and targetlabel not in goodList:
                        continue
                    
                #get discovery time
                discoverytime = ut.get_disctime(TNSFile, targetlabel)
                #run it
                trlc = etsMAIN(save_dir, TNSFile)
                trlc.load_single_lc(time, flux, error, discoverytime, 
                                   targetlabel, sector, camera, ccd)
                
                winfilter = trlc.window_rms_filt(plot=False)
                
                if "2018fhw" in targetlabel:
                    winfilter[1040:1080] = 0.0
                if "2020hdw" in targetlabel:
                    winfilter[0:45] = 0.0
                    winfilter[610:685] = 0.0
                    
                trlc.pre_run_clean(11, flux_mask=winfilter, 
                                   binning = binning, fraction = fraction)
                
                trlc.run_GP_fit(n1=n1, n2=n2, gpUSE=GPtype, usebounds=bounds, 
                               cbounds=cbounds)

                print(trlc.filesavetag, trlc.best_mcmc)
                gc.collect()
                i=i+1

    return trlc

# import numpy as np
# # rho_bounds = np.log((0.25, 10)) #0, 2.302
# # sigma_bounds = np.log( np.sqrt((0.1, 20)) ) #sigma range 0.316 to 4.47, take log
# # bounds_dict = dict(log_sigma=sigma_bounds, log_rho=rho_bounds, boundlabel="-0-25day")

# trlc = run_all_GP('celerite_residual', data_dir, save_dir, TNSFile,
#                   filekey = "-tessreduce",
#                   goodList=gList, 
#                   fraction=0.6, binning=False, n1=1000, n2=15000, bounds=True,
#                   cbounds=None)



def run_all_matern32comp(data_dir, save_dir, TNSFile,
                         filekey = "-tessreduce", goodList=None, fraction=None, 
                         binning=False, n1=10000, n2=40000, bounds=True):
    """ 
    Run the matern-3/2 comparison for all light curves in a given folder
    ----------------------------
    Params:
        - fitType (str) ie, 'matern32', 'expsinsqr', 'expsqr', 'celerite'
        - data_dir (str) path to directory holding all data
            *** assumes files are formatted in the given tessreduce manner
        - save_dir (str) path to directory to put all outputs into
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
    i=0
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith(filekey) and i==0:
                fname = root + "/" + name

                (time, flux, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(fname)

                if goodList is not None and targetlabel not in goodList:
                        continue
                #get discovery time
                discoverytime = ut.get_disctime(TNSFile, targetlabel)
                #run it
                trlc = etsMAIN(save_dir, TNSFile)
                
                trlc.load_single_lc(time, flux, error, discoverytime, 
                                   targetlabel, sector, camera, ccd)
                
                winfilter = trlc.window_rms_filt(plot=False)
                
                if "2018fhw" in targetlabel:
                    winfilter[1040:1080] = 0.0
                if "2020hdw" in targetlabel:
                    winfilter[0:45] = 0.0
                    winfilter[610:685] = 0.0
                    
                trlc.run_both_matern32(winfilter, binning=binning, fraction=fraction,
                                       bounds=bounds)
            
                gc.collect()
                i=i+1

    return trlc

