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

def retrieve_disctimes(datafolder, info):
    disctimeall = []
    for root, dirs, files in os.walk(datafolder):
        for name in files:
            if name.endswith("-tessreduce"):
                targ = name.split("-")[0][:-4]
                #print(targ)
                if targ not in gList:
                    continue
                holder = root + "/" + name
                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder)

                #get discovery time
                d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                tmin = time[0]
                time = time - tmin
                disctime = discoverytime-tmin
                disctimeall.append(disctime)
    return disctimeall 

datafolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
foldersave = "/Users/lindseygordon/research/urop/plotOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"
bigInfoFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"

gList = ["2018exc", "2018fhw", "2018fub", 
         "2020tld", "2020hvq", 
         "2020hdw", "2019gqv"]

info = pd.read_csv(bigInfoFile)
#cmd + 1 to comment
discall = retrieve_disctimes(datafolder, info)
t0all = []
Aall= []
betaall = []
Ball = []
for root, dirs, files in os.walk(foldersave):
    for name in files:
        if name.endswith("singlepower-output-params.txt"):
            targ = name.split("-")[0]
            print(targ)
            #print(targ[:-4])
            if targ[:-4] not in gList:
                continue

            
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

          
sp.plot_histogram(np.asarray(betaall), 32, "beta", 
                  "/Users/lindseygordon/research/urop/plotOutput/good_betahisto_only.png")

time_between = np.asarray(discall)-np.asarray(t0all)
plt.scatter(np.asarray(betaall), time_between)
#%%
   

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


#%%
bigInfoFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"
datafolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
foldersave = "/Users/lindseygordon/research/urop/plotOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"

gList = ["2018exc", "2018fhw", "2018fub", 
         "2020tld", "2020hvq", 
         "2020hdw", "2019gqv"]

nrows = 4
ncols = 2

fig, ax = plt.subplots(nrows, ncols, sharex=False,
                           figsize=(8*ncols, 3*nrows))

#plotting all on gList
info = pd.read_csv(bigInfoFile)
i = 0
m = 0
n = 0
for root, dirs, files in os.walk(datafolder):
    for name in files:
        if name.endswith("-tessreduce"):
            targ = name.split("-")[0][:-4]
            #print(targ)
            if targ not in gList:
                continue
            holder = root + "/" + name
            (time, intensity, error, targetlabel, 
             sector, camera, ccd) = ut.tr_load_lc(holder)

            #get discovery time
            d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
            discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
            tmin = time[0]
            time = time - tmin
            disctime = discoverytime-tmin
            #print(discoverytime)
            #run it
            #m row n col
            ax[m,n].scatter(time, intensity, color='black', s=2, label="Raw")
            ax[m,n].set_title(targetlabel, fontsize=14)
            ax[m,n].axvline(disctime, label="Disc. time")
            ax[m,n].set_xlabel("BJD - {timestart:.2f}".format(timestart=tmin), fontsize=10)
            ax[m,n].set_ylabel("flux (e-/s)", fontsize=10)
            ax[m,n].legend(loc="upper left", fontsize=8)
            
            if n<(ncols-1):
                n=n+1
            else:
                m=m+1
                n=0
                
fig.tight_layout()


#%%

class fake(object):
    """Make one of these for the light curve you're going to fit!"""
    
    def __init__(self):
        self.n = 1
        self.args = (10,20, self.n)
        
    def testy(self):
        for i in range(40):
            if i%5 == 0:
                self.n = self.n+1
            print(self.args)
            
h = fake()
h.testy()