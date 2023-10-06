#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 13:43:06 2023

@author: lindseygordon

gp background for injections
"""
import numpy as np
import matplotlib.pyplot as plt
from etsfit import etsMAIN
import etsfit.utils.utilities as ut
from pylab import rcParams
import os
import pandas as pd
from astropy.time import Time
from astropy.stats import SigmaClip
from scipy.stats import truncnorm
rcParams['figure.figsize'] = 8,3
rcParams['font.family'] = 'serif'


# load in light curve
datafolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
file = f"{datafolder}2020tld2921/2020tld2921-tessreduce" 
TNSFile = "/Users/lindseygordon/research/paper_outputs/TNS_10_paper.csv"
print(f"loading {file}")

(time, flux, error, targetlabel, 
 sector, camera, ccd) = ut.tr_load_lc(file)



# plt.scatter(time, flux, color='black')

# dt = Time(time[1], format='jd') - Time(time[0], format='jd')

# if dt.sec > 1700:
#     print(dt.sec, "in the okay cadence! ")
# else: 
#     print(dt.sec, "need to bin this ")
#     n_pts = int(np.rint(1800/dt.sec)) #3 ten minutes = 30 minutes
#     t_ = []
#     f_ = []
#     e_ = []
    
#     n = int(0)
#     m = int(n_pts+1)
    
#     while m<len(time):
#         t_.append(time[n])
#         range_flux = flux[n:m][~np.ma.masked_invalid(flux[n:m]).mask]
#         range_error = error[n:m][~np.ma.masked_invalid(error[n:m]).mask]
#         if len(flux) == 0:
#             f_.append(np.nan)
#         else: 
#             f_.append(np.nanmean(range_flux))
#             e_.append(np.nanmean(range_error))
    
#         n+= n_pts
#         m+= n_pts  
 
#     time = np.asarray(t_)
#     flux = np.asarray(f_)
    
# plt.scatter(time, flux, color='green')    
# flux -= np.mean(flux)

# sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
# mask = np.ma.getmask(sigclip(flux))
# flux = flux[~mask]
# time = time[~mask]
# error = error[~mask]
# plt.scatter(time,flux,  s=2, color='hotpink')
# plt.show()


# fit gp model to it (constrained)
TNSinfo = pd.read_csv(TNSFile)
discoverytime = ut.get_disctime(TNSFile, targetlabel)

#make an etsfit instance
folderSave = "./research/paper_outputs/gp_injections/"
ets = etsMAIN(folderSave, TNSFile)
             
#load the data in                           
ets.load_single_lc(time, flux, error, discoverytime, 
                   targetlabel, sector, camera, ccd)

filt = ets.window_rms_filt()

#set the fit type to 11 for gp usage
ets.pre_run_clean(11, flux_mask=filt, 
                    binning = False, fraction = None)
                
ets.test_plot()

gpUSE = "celerite_residual"
rho_bounds = np.log((0.25, 10)) #0, 2.302
sigma_bounds = np.log( np.sqrt((0.1, 20)) ) #sigma range 0.316 to 4.47, take log
bounds_dict = dict(log_sigma=sigma_bounds, log_rho=rho_bounds, boundlabel="-0-25day")

ets.run_GP_fit(n1=10000, n2=30000, gpUSE=gpUSE, usebounds=True, 
                               custom_bounds=bounds_dict)



# sample from it
# use as injection background (recycle code)