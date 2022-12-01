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
import etsfit.utils.utilities as ut

datafolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
foldersave = "/Users/lindseygordon/research/urop/paperOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"
bigInfoFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"

#gList = ["2018exc", "2018fhw", "2018fub", "2020tld", 
 #         "2020zbo", "2020hvq", "2018hzh",
  #        "2020hdw", "2020bj", "2019gqv"]
gList = ["2020tld"]
#filepath = "/Users/lindseygordon/research/urop/paperOutput/2020tld2921/singlepower-0.6/2020tld2921-singlepower-0.6-output-params.txt"



def retrieve_disctimes(datafolder, info, gList):
    disctimeall = {}
    for root, dirs, files in os.walk(datafolder):
        for name in files:
            if name.endswith("-tessreduce"):
                targ = name.split("-")[0][:-4]
                #print(targ)
                if targ not in gList:
                    continue
                holder = root + "/" + name
                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder, printname=False)

                #get discovery time
                d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                tmin = time[0]
                time = time - tmin
                disctime = discoverytime-tmin
                disctimeall[targetlabel] = disctime
                #disctimeall.append(disctime)
    return disctimeall 

def retrieve_all_singlepower06(bigInfoFile, datafolder, foldersave, gList):
    info = pd.read_csv(bigInfoFile)
    disc_all = retrieve_disctimes(datafolder, info, gList)
    params_all = {}
    converged_all = {}
    upper_all = {}
    lower_all = {}
    
    #retrieve an d print out the things: 
    for root, dirs, files in os.walk(foldersave):
        for name in files:
            if name.endswith("singlepower-0.6-output-params.txt"):
                targ = name.split("-")[0][:-4]
                #print(targ)
                if targ not in gList:
                    continue
                
                filepath = root + "/" + name
                #print(filepath)
                
                (params,  upper_e, 
                 lower_e,  converg) = extract_singlepower_all(filepath)
                
                params_all[targ] = params
                upper_all[targ] = upper_e
                lower_all[targ] = lower_e
                converged_all[targ] = converg
                
                
    return (disc_all, params_all, converged_all, upper_all, lower_all)



def extract_singlepower_all(filepath):
    
    #main params:
    filerow1 = np.loadtxt(filepath, skiprows=0, dtype=str, max_rows=1)
    if filerow1[0] == "[": #first string is just [
        params = (float(filerow1[1]), float(filerow1[2]), 
                  float(filerow1[3]), float(filerow1[4][:-1]))
    else: #first string contains [
        params = (float(filerow1[0][1:]), float(filerow1[1]), 
                  float(filerow1[2]), float(filerow1[3][:-1]))
    
    #upper error:
    filerow3 = np.loadtxt(filepath, skiprows=1, dtype=str, max_rows=1)
    if filerow3[0] == "[": #first string is just [
        upper_e = (float(filerow3[1]), float(filerow3[2]), 
                  float(filerow3[3]), float(filerow3[4][:-1]))
    else: #first string contains [
        upper_e = (float(filerow3[0][1:]), float(filerow3[1]), 
                  float(filerow3[2]), float(filerow3[3][:-1]))
    
    filerow4 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
    if filerow3[0] == "[": #first string is just [
        lower_e = (float(filerow4[1]), float(filerow4[2]), 
                  float(filerow4[3]), float(filerow4[4][:-1]))
    else: #first string contains [
        lower_e = (float(filerow4[0][1:]), float(filerow4[1]), 
                  float(filerow4[2]), float(filerow4[3][:-1]))
    
    #get convergence
    filerow9 = np.loadtxt(filepath, skiprows=3, dtype=str, max_rows=1)
    if "True" in filerow9:
        converg = True
    else:
        converg = False
    return params, upper_e, lower_e, converg

#retrieve_all_singlepower06(bigInfoFile, datafolder, foldersave, gList)


def retrieve_all_singlepower06celerite(bigInfoFile, datafolder, foldersave, gList):
    info = pd.read_csv(bigInfoFile)
    disc_all = retrieve_disctimes(datafolder, info, gList)
    params_all = {}
    converged_all = {}
    upper_all = {}
    lower_all = {}
    
    #retrieve an d print out the things: 
    for root, dirs, files in os.walk(foldersave):
        for name in files:
            if name.endswith("celerite-matern32-0.6-output-params.txt"):
                targ = name.split("-")[0][:-4]
                print(targ)
                if targ not in gList:
                    continue
                
                filepath = root + "/" + name
                #print(filepath)
                
                (params,  upper_e, 
                 lower_e,  converg) = extract_celerite_all(filepath)
                
                params_all[targ] = params
                upper_all[targ] = upper_e
                lower_all[targ] = lower_e
                converged_all[targ] = converg
                
                
    return (disc_all, params_all, converged_all, upper_all, lower_all)

def extract_celerite_all(filepath):
    
    # target label row 0
    # bic row 1
    # convg row 2
    filerow1 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
    print(filerow1)
    if "True" in filerow1:
        converg = True
    else:
        converg = False
    
    #params row 1: 
    filerow1 = np.loadtxt(filepath, skiprows=3, dtype=str, max_rows=1)
    
    sigsq, rho, t0, A = (np.exp(2*float(filerow1[0])), 
                         np.exp(float(filerow1[1])), 
                         float(filerow1[2]), 
                         float(filerow1[3]))
    
    filerow2 = np.loadtxt(filepath, skiprows=4, dtype=str, max_rows=1)    
    beta, B = (float(filerow2[0]), float(filerow2[1]))                                    

    params = (sigsq, rho, t0, A, beta, B)
    
    #upper error
    filerow1 = np.loadtxt(filepath, skiprows=5, dtype=str, max_rows=1)
    sigsq, rho, t0, A = (float(filerow1[0]), 
                         float(filerow1[1]), 
                         float(filerow1[2]), 
                         float(filerow1[3]))
    filerow2 = np.loadtxt(filepath, skiprows=6, dtype=str, max_rows=1)    
    beta, B = (float(filerow2[0]), float(filerow2[1]))                                    

    upper_e = (sigsq, rho, t0, A, beta, B)
    
    #lower error
    filerow1 = np.loadtxt(filepath, skiprows=7, dtype=str, max_rows=1)
    sigsq, rho, t0, A = (float(filerow1[0]), 
                         float(filerow1[1]), 
                         float(filerow1[2]), 
                         float(filerow1[3]))
    filerow2 = np.loadtxt(filepath, skiprows=8, dtype=str, max_rows=1)    
    beta, B = (float(filerow2[0]), float(filerow2[1]))                                    

    lower_e = (sigsq, rho, t0, A, beta, B)
    
    return params, upper_e, lower_e, converg

(disc_all, params_all, 
 converged_all, upper_all, 
 lower_all) = retrieve_all_singlepower06celerite(bigInfoFile, datafolder, 
                                                 foldersave, gList)


