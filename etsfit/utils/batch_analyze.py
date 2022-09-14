#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:54:28 2022

@author: lindseygordon

batch analyze aggregate info on parameters
"""

#load parameters from files
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from etsfit import etsMAIN
from astropy.time import Time
import gc
from pylab import rcParams
import etsfit.utils.snPlotting as sp
import etsfit.utils.utilities as ut

datafolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
foldersave = "/Users/lindseygordon/research/urop/plotOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"

#cmd + 1 to comment
t0all = []
Aall= []
betaall = []
Ball = []
for root, dirs, files in os.walk(foldersave):
    for name in files:
        if name.endswith("singlepower-output-params.txt"):
            filepath = root + "/" + name
            #print(filepath)
            filerow1 = np.loadtxt(filepath, skiprows=0, dtype=str, max_rows=1)
            bicrow = np.loadtxt(filepath, skiprows=4, dtype=str, max_rows=1)
            #print(bicrow)
            conv = np.loadtxt(filepath, skiprows=5, dtype=str, max_rows=1)
            #print(conv)
            if "False" in str(conv):
                
                continue
            else:
                print(filepath)
                print(bicrow)
            #filerow2 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
            #filerow3 = np.loadtxt(filepath, skiprows=3, dtype=str, max_rows=1)
            #print(filerow1)
            if filerow1[0] == "[": #first string is just [
                
                t0= float(filerow1[1])
                A=float(filerow1[2])
                beta=float(filerow1[3])
                B=float(filerow1[4][:-1])
            
            else: #first string contains [
                #print(filerow1, filerow1[0][1:],filerow1[3][:-1])
                t0=float(filerow1[0][1:])
                A=float(filerow1[1])
                beta=float(filerow1[2])
                B=float(filerow1[3][:-1])
                
            t0all.append(t0)
            Aall.append(A)
            betaall.append(beta)
            Ball.append(B)

            
sp.plot_histogram(np.asarray(betaall), 32, "beta", "/Users/lindseygordon/research/urop/plotOutput/beta-histogram-converged.png")
#%%
def extract_singlepowerparams_from_file(filepath):
    
    filerow1 = np.loadtxt(filepath, skiprows=0, dtype=str, max_rows=1)
    if filerow1[0] == "[": #first string is just [
        
        t0= float(filerow1[1])
        A=float(filerow1[2])
        beta=float(filerow1[3])
        B=float(filerow1[4][:-1])
    
    else: #first string contains [
        #print(filerow1, filerow1[0][1:],filerow1[3][:-1])
        t0=float(filerow1[0][1:])
        A=float(filerow1[1])
        beta=float(filerow1[2])
        B=float(filerow1[3][:-1])
    return t0,A,beta,B


#load in and plot 3 things
rerunspartials = 
nrows = 2
ncols = 3

fig, ax = plt.subplots(nrows, ncols, sharex=True,
                           figsize=(8*ncols * 2, 3*nrows * 2))
#write down the name sof the ones you want and then give it the folders to skim the actual files from
lc_to_load = ["2020tld", "2018hkx", "2018fhw"]
datafolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
outputfolder = "/Users/lindseygordon/research/urop/plotOutput/"
bigFile = "/Users/lindseygordon/research/urop/Ia18thmag.csv"
info = pd.read_csv(bigFile)

for i in range(len(lc_to_load)):
    for root,dirs,files in os.walk(datafolder):
        for f in files:
            if lc_to_load[i] in f and f.endswith("tessreduce"):
                (time, intensity, error, 
                 targetlabel, sector, camera, ccd) = ut.tr_load_lc(root + "/" + f)
                d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                tmin = time[0]
                time -= tmin
                discoverytime -= tmin
                
                rms_filt = ut.window_rms(time, intensity, innerfilt = None, outerfilt = None,
                                    plot=False)
                
                ax[0,i].scatter(time, intensity, s=1, color='red', label = "TR raw")
                rms_filt_plot = np.nonzero(rms_filt)
                ax[0,i].scatter(time[rms_filt_plot], intensity[rms_filt_plot], s=5, color='black', label = "TR filtered")
                ax[0,i].axvline(discoverytime, color="green", linestyle="dashed", label = "disc. time")
                ax[1,i].axvline(discoverytime, color="green", linestyle="dashed", label = "disc. time")
                
                #load in parameters:
                for root1, dirs1, files1 in os.walk(outputfolder):
                    for f1 in files1:
                        if lc_to_load[i] in f1 and f1.endswith("singlepower-output-params.txt"):
                            filepath = root1 + "/" + f1
                            #print(filepath)
                            t0,A,beta,B = extract_singlepowerparams_from_file(filepath)
                            break   
                        
                #build model, plot model
                t1 = time - t0
                model = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + B
                ax[0,i].plot(time, model, color="blue", label="model")
                ax[0,i].axvline(t0, linestyle="dashed", color="purple", label="t0")
                ax[1,i].axvline(t0, linestyle="dashed", color="purple", label="t0")
                ax[0,i].axhline(B, linestyle="dashed", label = "B")
                
                ax[1,i].scatter(time[rms_filt_plot], (intensity-model)[rms_filt_plot], color = "black", s=5, label="residuals")
                ax[1,i].axhline(0, color='darkgreen', linestyle='dashed')
                ax[0,i].set_ylabel("Raw TR Flux")
                ax[1,i].set_ylabel("Residual Flux")
                ax[0,i].set_title(targetlabel)
                ax[1,i].set_xlabel("BJD - {timestart:.3f}".format(timestart=tmin))
                
                ax[0,i].legend(loc="upper left", fontsize=12)
                ax[1,i].legend(loc="lower left", fontsize=12)
                
#fig.suptitle("Good, Average, Ugly")
plt.tight_layout()
#plt.savefig("/Users/lindseygordon/research/urop/plotOutput/triplePlotTest.png")
