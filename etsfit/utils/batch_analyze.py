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
#%%
def extract_singlepowerparams_from_file(filepath):
    filerow1 = np.loadtxt(filepath, skiprows=0, dtype=str, max_rows=1)
    #filerow2 = np.loadtxt(filepath, skiprows=1, dtype=str, max_rows=1)
    #print(filerow2)
    bicrow = np.loadtxt(filepath, skiprows=3, dtype=str, max_rows=1)
    conv = np.loadtxt(filepath, skiprows=4, dtype=str, max_rows=1)
    if filerow1[0] == "[": #first string is just [
        t0= float(filerow1[1])
        A=float(filerow1[2])
        beta=float(filerow1[3])
        B=float(filerow1[4][:-1])
    else: #first string contains [
        #print(filerow1, filerow1[0][1:],filerow1[3][:-1])
        t0=float(filerow1[0])
        A=float(filerow1[1])
        beta=float(filerow1[2])
        B=float(filerow1[3][:-1])
    return t0,A,beta,B, bicrow, conv

def get_upper_e(filepath):
    filerow1 = np.loadtxt(filepath, skiprows=1, dtype=str, max_rows=1)
    t0=float(filerow1[0])
    A=float(filerow1[1])
    beta=float(filerow1[2])
    B=float(filerow1[3][:-1])
    return t0,A,beta,B

def get_lower_e(filepath):
    filerow1 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
    t0=float(filerow1[0])
    A=float(filerow1[1])
    beta=float(filerow1[2])
    B=float(filerow1[3][:-1])
    return t0,A,beta,B


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
foldersave = "/Users/lindseygordon/research/urop/paperOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"
bigInfoFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"

gList = ["2018exc", "2018fhw", "2018fub", "2020tld", 
         "2020zbo", "2020hvq", "2018hzh",
         "2020hdw", "2020bj", "2019gqv"]

info = pd.read_csv(bigInfoFile)
#cmd + 1 to comment
discall = retrieve_disctimes(datafolder, info)
t0all = []
Aall= []
betaall = []
Ball = []
convy = []
upper_all = []
lower_all = []

#retrieve an d print out the things: 
for root, dirs, files in os.walk(foldersave):
    for name in files:
        if name.endswith("singlepower-0.6-output-params.txt"):
            targ = name.split("-")[0]
            #print(targ[0])
            #print(targ[:-4])
            if targ[:-4] not in gList:
                continue
            
            filepath = root + "/" + name
            #print(filepath)
            
            
            t0,A,beta,B, bicrow, conv = extract_singlepowerparams_from_file(filepath)
            upper_all.append(get_upper_e(filepath))
            lower_all.append(get_lower_e(filepath))
            print("{t} & {t0:.2f}  &{A:.2f} & {b:.2f} & {B:.2f} \\".format(t=targ[:-4],
                                                            t0 = t0,
                                                            A=A,
                                                            b=beta,
                                                            B=B,))
            #print(t0, A, beta, B)
            #print(bicrow)
            #print(conv)

            t0all.append(t0)
            Aall.append(A)
            betaall.append(beta)
            Ball.append(B)
            
            if "True" in str(conv):
                convy.append("True")
            else:
                convy.append("False")
        

#%%

rcParams['figure.figsize'] = 10,10
fig, ax1 = plt.subplots()
n_in, bins, patches = ax1.hist(np.asarray(betaall), 9, color='black', 
                               alpha=0.5, label="Unconverged")
ax1.hist(np.asarray(betaall)[np.where(np.asarray(convy) == "True")], bins, 
         color='purple', alpha=0.3, label="Converged")

#y_range = np.abs(n_in.max() - n_in.min())
#x_range = np.abs(data.max() - data.min())
ax1.set_ylabel('Number of light curves')
ax1.set_xlabel(r"$\beta$")
plt.title(r"Histogram of Retrieved $\beta$ Values")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/lindseygordon/research/urop/paperOutput/good_betahisto_only.png")
plt.show()
plt.close()
#rcParams['figure.figsize'] = 16,6

             #%%   

#put error bars on here! 
time_between = np.asarray(discall)-np.asarray(t0all)
plt.errorbar(np.asarray(betaall), time_between, xerr = np.asarray(upper_all)[:,2],
              yerr = np.asarray(upper_all)[:,0], fmt='o',
              label = "Unconverged", color='red')
plt.errorbar(np.asarray(betaall)[np.where(np.asarray(convy) == "True")], 
            time_between[np.where(np.asarray(convy) == "True")], 
            xerr = np.asarray(upper_all)[np.where(np.asarray(convy) == "True")][:,2],
            yerr = np.asarray(upper_all)[np.where(np.asarray(convy) == "True")][:,0], fmt='o',label = "Converged",
            color="Blue")
plt.xlabel(r"$\beta$")
plt.ylabel(r"Disc time. - $t_0$ (JD)")
plt.title(r"Time between $t_0$ and discovery time versus $\beta$")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/lindseygordon/research/urop/paperOutput/disc-t0-beta.png")


#%%
   

#%%
import tessreduce as tr
obs = tr.sn_lookup("2020xyw")
cdir = "/Users/lindseygordon/.lightkurve-cache/tesscut/"
tess = tr.tessreduce(obs_list=obs[16],plot=False,reduce=True)

#%%
bigInfoFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"
datafolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
foldersave = "/Users/lindseygordon/research/urop/plotOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"

gList = ["2018exc", "2018fhw", "2018fub", "2020tld", 
         "2020zbo", "2020hvq", "2018hzh","2020hdw", "2020bj", "2019gqv"]

nrows = 5
ncols = 2

fig, ax = plt.subplots(nrows, ncols, sharex=False,
                       figsize=(8*ncols, 3*nrows))


info = pd.read_csv(bigInfoFile)
i = 0
m = 0
n = 0
rcParams['figure.figsize'] = 8,3
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
            
            trlc = etsMAIN(foldersave, bigInfoFile)
            
            trlc.load_single_lc(time, intensity, error, discoverytime, 
                               targetlabel, sector, camera, ccd, lygosbg=None)
            
            
            filterMade = trlc.window_rms_filt(plot=False)
            if "2018fhw" in targetlabel:
                filterMade[1040:1080] = 0.0
            trlc.pre_run_clean(1, cutIndices=filterMade, 
                               binYesNo = False, fraction = 0.6)
            
            filepath = ('/Users/lindseygordon/research/urop/paperOutput/'+
                        trlc.targetlabel + trlc.sector + trlc.camera + trlc.ccd + 
                        '/singlepower-0.6/'+
                        trlc.targetlabel + trlc.sector + trlc.camera + trlc.ccd + 
                        '-singlepower-0.6-output-params.txt')
            
            t0,A,beta,B, bicrow, conv = extract_singlepowerparams_from_file(filepath)
            
            t1 = trlc.time - t0
            model = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + B
            
            # plt.scatter(trlc.time, trlc.intensity)
            # plt.show()
            # plt.close()
            
            ax[m,n].scatter(trlc.time, trlc.intensity, color='black', s=2, label="data")
            ax[m,n].plot(trlc.time, model, color='red', label='model')
            ax[m][n].set_title(trlc.targetlabel, fontsize=14)
            ax[m][n].axvline(trlc.disctime-tmin, color="brown", linestyle = "dotted",
                             label="Disc. time")
            ax[m][n].axvline(t0, label="t0", color="green", linestyle="dashed")
            ax[m][n].set_xlabel("BJD - {timestart:.2f}".format(timestart=tmin), fontsize=10)
            ax[m][n].set_ylabel("flux (e-/s)", fontsize=20)
            ax[m][n].tick_params('x', labelsize=10)
            ax[m][n].legend(loc="upper left", fontsize=8)
            
            if n<(ncols-1):
                n=n+1
            else:
                m=m+1
                n=0
fig.suptitle("Standard Power Law, Flat Background, 60% Peak Flux")
fig.tight_layout()
fig.show()   
plt.savefig("/Users/lindseygordon/research/urop/paperOutput/all10-fit.png")            
# fig.tight_layout()
#%%

    

