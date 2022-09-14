#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:29:49 2022

@author: lindseygordon

CDPP plotting:
    
    combined differential photometric precision
    
    - host & peak magnitudes for every objects. what is standard deviation of 
    photometry data points in a given amount of time. take mean in an hour and 
    std over whole LC, CDPP - binning the photometry and compare to std over longer 
    time scales. characterize photometric reproducibility over whole target. idea is 
    peak mag is still so faint be sure we recover SN we think we should recover

ground disc vs CDPP and color them green/red depending on if you can visually see anything
"""

import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import os
from etsfit import etsMAIN
from astropy.time import Time, TimeDelta
import gc
from pylab import rcParams
import etsfit.utils.snPlotting as sp
import etsfit.utils.utilities as ut
from pylab import rcParams
rcParams['figure.figsize'] = 10,10


datafolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
foldersave = "/Users/lindseygordon/research/urop/plotOutput/"
bigFile = "/Users/lindseygordon/research/urop/Ia18thmag.csv"
info = pd.read_csv(bigFile)

# #load in each
# #take hour binned means
# #compare with std
# #take mean of comps
# #plot mean vs mag
# cleaned = True
# gList = ["2018exc", "2018fhw", "2018fub", "2020tld", "2020zbo", "2020xyw", "2020hvq", 
#              "2020hdw", "2020bj", "2019gqv"]


# for root,dirs,files in os.walk(datafolder):
#     for f in files:
#         if f.endswith("tessreduce"):
#             (time, intensity, error, 
#              targetlabel, sector, camera, ccd) = ut.tr_load_lc(root + "/" + f)
#             d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
#             discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
#             tmin = time[0]
#             time -= tmin
#             discoverytime -= tmin
#             discmag = info[info["Name"].str.contains(targetlabel)]["Discovery Mag/Flux"].iloc[0]
            
#             if cleaned: 
#                 rms_filt = ut.window_rms(time, intensity, innerfilt = None, outerfilt = None,
#                                     plot=False)
#                 rms_filt_plot = np.nonzero(rms_filt)
#                 time=time[rms_filt_plot]
#                 intensity = intensity[rms_filt_plot]
            
#             standardDev = np.std(intensity)
#             dt = TimeDelta(time[1], format='jd').sec
#             #print(dt)
#             if dt < 601: #ten minute cadence
#                 #go in groups of 6 for hour binning
#                 binnedDiff = np.zeros(math.ceil(len(time)/6))
#                 n=0
#                 m=6
#                 h=0
#                 while n < len(time):
#                     mean = np.mean(intensity[n:m])
#                     binnedDiff[h] = np.abs(standardDev - mean)
#                     h+=1
#                     n+=6
#                     m+=6
                    
#             else: #thirty minute cadence
#                 #groups of 2
#                 binnedDiff = np.zeros(math.ceil(len(time)/2))
#                 #print(math.ceil(len(time)/2))
#                 n=0
#                 m=2
#                 h=0
#                 while n < len(time):
#                     mean = np.mean(intensity[n:m])
#                     #print(n, m)
#                     binnedDiff[h] = np.abs(standardDev - mean)
#                     h+=1
#                     n+=2
#                     m+=2
            
#             #take mean of differences? 
#             meany = np.mean(binnedDiff)
#             if meany > 40:
#                 print(targetlabel)
#             if targetlabel in gList:
#                 color = 'green'
#             else:
#                 color = 'red'
            
#             plt.scatter(meany, discmag, color=color)
            
# plt.xlabel("Avg. diff. between 1hr binned means and overall std")
# plt.ylabel("Discovery mag.")       
# plt.savefig(foldersave + "cdpp-mag-cleaned-plot.png")   

gList = ["2018exc", "2018fhw", "2018fub", "2020tld", "2020zbo", "2020xyw", "2020hvq", 
             "2020hdw", "2020bj", "2019gqv"]

normalize = True
i=0
from astropy.stats import SigmaClip

for root,dirs,files in os.walk(datafolder):
    for f in files:
        if f.endswith("tessreduce"): #and i==0:
            #load in 
            (time, intensity, error, 
             targetlabel, sector, camera, ccd) = ut.tr_load_lc(root + "/" + f)
            d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
            discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
            tmin = time[0]
            time -= tmin
            discoverytime -= tmin
            discmag = info[info["Name"].str.contains(targetlabel)]["Discovery Mag/Flux"].iloc[0]
        
            ogrange = max(intensity) - min(intensity)
            #print(ogrange)
                
            if normalize:
                time, intensity, error, bg = ut.normalize_sigmaclip(time, intensity,error, None, axis=0)
                #plt.scatter(time, output)
                #i+=1
            # determine and calculate the rms filter:
            dt = TimeDelta(time[1], format='jd').sec
            #print(dt)
            if dt < 601: #ten minute cadence, groups of 6
                innerfilt = 6
                outerfilt = 18
            else: # thirty minute cadence, groups of 2
                innerfilt = 2
                outerfilt = 6
            rms_filt = ut.window_rms(time, intensity, innerfilt = innerfilt, outerfilt = outerfilt,
                          plot=False)
            rms_filt_plot = np.nonzero(rms_filt)
            time=time[rms_filt_plot]
            intensity = intensity[rms_filt_plot]
            
            
            #take rolling rms
            df = pd.DataFrame({'A': intensity})
            rollingrms = df["A"].pow(2).rolling(innerfilt).apply(lambda x: np.sqrt(x.mean()))
            
            
            sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
            clipped_inds = np.nonzero(np.ma.getmask(sigclip(rollingrms)))
            #print(clipped_inds[0])
            rollingrms = np.delete(rollingrms.to_numpy(), clipped_inds)
            rollingmean = np.mean(rollingrms)
            
            
            if targetlabel in gList:
                color = 'green'
                lab = "Good"
            else:
                color = 'red'
                lab = "Bad"
                
            if ogrange > 100:
                shape = 'x'
            elif ogrange > 50:
                shape = 'o'
            else:
                shape = 'v'
            
            plt.scatter(discmag, rollingmean, color=color, marker=shape)
            
plt.ylabel("Mean rolling RMS of cleaned light curve")
plt.xlabel("Discovery mag.")    
plt.title("x=range>100, o=range>50, v=range<50. green=good, red=bad")  
plt.savefig(foldersave + "rollingRMS-mag-cleaned-normalized-plot.png")   