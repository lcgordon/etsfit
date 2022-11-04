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

def get_upper_e_single(filepath):
    filerow1 = np.loadtxt(filepath, skiprows=1, dtype=str, max_rows=1)
    t0=float(filerow1[0])
    A=float(filerow1[1])
    beta=float(filerow1[2])
    B=float(filerow1[3][:-1])
    return t0,A,beta,B

def get_lower_e_single(filepath):
    filerow1 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
    t0=float(filerow1[0])
    A=float(filerow1[1])
    beta=float(filerow1[2])
    B=float(filerow1[3][:-1])
    return t0,A,beta,B


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
                 sector, camera, ccd) = ut.tr_load_lc(holder)

                #get discovery time
                d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                tmin = time[0]
                time = time - tmin
                disctime = discoverytime-tmin
                disctimeall[targetlabel] = disctime
                #disctimeall.append(disctime)
    return disctimeall 

def retrieve_all_singlepower(bigInfoFile, datafolder, foldersave, gList):
    info = pd.read_csv(bigInfoFile)
    #cmd + 1 to comment
    discall = retrieve_disctimes(datafolder, info, gList)
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
                upper_all.append(get_upper_e_single(filepath))
                lower_all.append(get_lower_e_single(filepath))
                print("{t} & {t0:.2f}  &{A:.2f} & {b:.2f} & {B:.2f} \\".format(t=targ[:-4],
                                                                t0 = t0,
                                                                A=A,
                                                                b=beta,
                                                                B=B,))
    
    
                t0all.append(t0)
                Aall.append(A)
                betaall.append(beta)
                Ball.append(B)
                
                if "True" in str(conv):
                    convy.append("True")
                else:
                    convy.append("False")
    return t0all, Aall, betaall, Ball, upper_all, lower_all, discall, convy


#%%
datafolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
foldersave = "/Users/lindseygordon/research/urop/paperOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"
bigInfoFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"

gList = ["2018exc", "2018fhw", "2018fub", "2020tld", 
         "2020zbo", "2020hvq", "2018hzh",
         "2020hdw", "2020bj", "2019gqv"]

filepath = "/Users/lindseygordon/research/urop/paperOutput/2018exc0111/celerite-tinygp-matern32/2018exc0111-celerite-tinygp-matern32-output-params.txt"

def extract_tgpc_all(filepath):
    
    #main params:
    filerow1 = np.loadtxt(filepath, skiprows=0, dtype=str, max_rows=1)
    if filerow1[0] == "[": #first string is just [
        params = (float(filerow1[1]), float(filerow1[2]), 
                  float(filerow1[3]), float(filerow1[4]))
    else: #first string contains [
        params = (float(filerow1[0][1:]), float(filerow1[1]), 
                  float(filerow1[2]), float(filerow1[3]))
    
    #celerite params: (in-place conversion)
    filerow2 = np.loadtxt(filepath, skiprows=1, dtype=str, max_rows=1)
    celerite_params = (float(filerow2[0]), float(filerow2[1][:-1])) #log sigma, log rho
    celerite_params = (np.exp(celerite_params[0]*2), np.exp(celerite_params[1]))
    
    
    #upper error:
    filerow3 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
    if filerow3[0] == "[": #first string is just [
        upper_e = (float(filerow3[1]), float(filerow3[2]), 
                  float(filerow3[3]), float(filerow3[4]), 
                  float(filerow3[5]), float(filerow3[6][:-1]))
    else: #first string contains [
        upper_e = (float(filerow3[0][1:]), float(filerow3[1]), 
                  float(filerow3[2]), float(filerow3[3]), 
                  float(filerow3[4]), float(filerow3[5][:-1]))
    
    filerow4 = np.loadtxt(filepath, skiprows=3, dtype=str, max_rows=1)
    if filerow3[0] == "[": #first string is just [
        lower_e = (float(filerow4[1]), float(filerow4[2]), 
                  float(filerow4[3]), float(filerow4[4]), 
                  float(filerow4[5]), float(filerow4[6][:-1]))
    else: #first string contains [
        lower_e = (float(filerow4[0][1:]), float(filerow4[1]), 
                  float(filerow4[2]), float(filerow4[3]), 
                  float(filerow4[4]), float(filerow4[5][:-1]))
        
    #skip a row, get tinygp params: 
    filerow6 = np.loadtxt(filepath, skiprows=5, dtype=str, max_rows=1, delimiter=",")
    tinygp_params = (float(filerow6[0]), float(filerow6[1])) #log sigma, log rho
    tinygp_params = (np.exp(tinygp_params[0]*2), np.exp(tinygp_params[1]))
    
    #get convergence
    filerow9 = np.loadtxt(filepath, skiprows=8, dtype=str, max_rows=1)
    if "True" in filerow9:
        converg = True
    else:
        converg = False
    return params, celerite_params, upper_e, lower_e, tinygp_params, converg

#%%




def retrieve_all_tinygp_celerite(bigInfoFile, datafolder, foldersave, gList):
    info = pd.read_csv(bigInfoFile)
    #cmd + 1 to comment
    disc_all = retrieve_disctimes(datafolder, info, gList)
    params_all = {}
    celerite_all = {}
    tinygp_all = {}
    converged_all = {}
    upper_all = {}
    lower_all = {}
    
    #retrieve an d print out the things: 
    for root, dirs, files in os.walk(foldersave):
        for name in files:
            if name.endswith("matern32-noBounds-output-params.txt"):
                targ = name.split("-")[0]
                print(targ)
                if targ[:-4] not in gList:
                    continue
                
                filepath = root + "/" + name
                print(filepath)
                
                
                (params, celerite_params, upper_e, 
                 lower_e, tinygp_params, converg) = extract_tgpc_all(filepath)
                
                params_all[targ] = params
                celerite_all[targ] = celerite_params
                upper_all[targ] = upper_e
                lower_all[targ] = lower_e
                converged_all[targ] = converg
                tinygp_all[targ] = tinygp_params
                
    
    
                
    return (disc_all, params_all, celerite_all, tinygp_all, 
            converged_all, upper_all, lower_all)

(disc_all, params_all, 
 celerite_all, tinygp_all, 
 converged_all, upper_all, 
 lower_all) = retrieve_all_tinygp_celerite(bigInfoFile, datafolder, foldersave, gList)

#%%
for k in params_all.keys(): 
    print("{k}&{t0:.2f}&{A:.2f}&{beta:.2f}&{B:.2f}".format(k=k[:-4],
                                                           t0 = params_all[k][0],
                                                           A = params_all[k][1],
                                                           beta = params_all[k][2],
                                                           B=params_all[k][3]))
#%%
for k in celerite_all.keys(): 
    print("{h} ({sigsq:.3f},{rho:.3f})".format(h=k, 
                                               sigsq=celerite_all[k][0], 
                                               rho=celerite_all[k][1] ))
    
    
    
for k in tinygp_all.keys(): 
    print("{h} ({sigsq:.3f},{rho:.3f})".format(h=k, 
                                               sigsq=tinygp_all[k][0], 
                                               rho=tinygp_all[k][1] ))