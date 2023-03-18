#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:54:28 2022

@author: lindseygordon

Batch retrieval methods for parameters in output files in the same parent directory
"""

#load parameters from files
import pandas as pd
import numpy as np
import os
from astropy.time import Time
import etsfit.utils.utilities as ut

# data_dir = "/Users/lindseygordon/research/urop/tessreduce_lc/"
# CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
# params_dir = "/Users/lindseygordon/research/urop/paperOutput/"
# quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
# quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"
# TNSFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"

# gList = ["2018exc", "2018fhw", "2018fub", "2020tld", 
#           "2020zbo", "2020hvq", "2018hzh",
#           "2020hdw", "2020bj", "2019gqv"]


def retrieve_disctimes(data_dir, tns_info, gList, datatag):
    disctimeall = {}
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith(datatag):
                targ = name.split("-")[0][:-4]
                #print(targ)
                if targ not in gList:
                    continue
                holder = root + "/" + name
                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder, printname=False)

                #get discovery time
                d = tns_info[tns_info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                tmin = time[0]
                time = time - tmin
                disctime = discoverytime-tmin
                disctimeall[targetlabel] = disctime
    return disctimeall 

def retrieve_all_singlepower(TNSFile, data_dir, params_dir, gList, 
                               datatag="-tessreduce", paramstag="singlepower-0.6"):
    tns_info = pd.read_csv(TNSFile)
    disc_all = retrieve_disctimes(data_dir, tns_info, gList, datatag)
    params_all = {}
    converged_all = {}
    upper_all = {}
    lower_all = {}
    
    #retrieve an d print out the things: 
    for root, dirs, files in os.walk(params_dir):
        for name in files:
            if (name.endswith(paramstag+"-output-params.txt") and
                'celerite' not in name and 'tinygp' not in name):
                targ = name.split("-")[0][:-4]
                if targ not in gList:
                    continue
                filepath = root + "/" + name
                (params,  upper_e, 
                 lower_e,  converg) = extract_singlepower_all(filepath)
                
                params_all[targ] = params
                upper_all[targ] = upper_e
                lower_all[targ] = lower_e
                converged_all[targ] = converg
    return (disc_all, params_all, converged_all, upper_all, lower_all)

def extract_singlepower_all(filepath):
    
    # target label row 0
    # bic row 1
    # convg row 2
    filerow1 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
    if "True" in filerow1:
        converg = True
    else:
        converg = False
        
    #main params:
    filerow1 = np.loadtxt(filepath, skiprows=3, dtype=str, max_rows=1)
    params = (float(filerow1[0]), float(filerow1[1]), 
              float(filerow1[2]), float(filerow1[3]))
    filerow1 = np.loadtxt(filepath, skiprows=5, dtype=str, max_rows=1)
    upper_e = (float(filerow1[0]), float(filerow1[1]), 
               float(filerow1[2]), float(filerow1[3]))
    
    filerow1 = np.loadtxt(filepath, skiprows=7, dtype=str, max_rows=1)
    lower_e = (float(filerow1[0]), float(filerow1[1]), 
               float(filerow1[2]), float(filerow1[3]))
    return params, upper_e, lower_e, converg


def extract_doublepower_all(filepath):
    """ 
    Function to pull double power fitted (type 3) parameters from the output files
    Params:
        - filepath (str)
    """
    
    # target label row 0, bic row 1
    
    # convg row 2:
    filerow1 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
    if "True" in filerow1:
        converg = True
    else:
        converg = False
        
    #main params:
    p3 = np.loadtxt(filepath, skiprows=3, dtype=str, max_rows=1)
    p4 = np.loadtxt(filepath, skiprows=4, dtype=str, max_rows=1)
    params = [float(p3[0]),float(p3[1]),float(p3[2]),float(p3[3]),
              float(p4[0]),float(p4[1]),float(p4[2])]
    
    #upper errrors: 
    p3 = np.loadtxt(filepath, skiprows=5, dtype=str, max_rows=1)
    p4 = np.loadtxt(filepath, skiprows=6, dtype=str, max_rows=1)
    upper_e = [float(p3[0]),float(p3[1]),float(p3[2]),float(p3[3]),
              float(p4[0]),float(p4[1]),float(p4[2])]
    
    #upper errrors: 
    p3 = np.loadtxt(filepath, skiprows=7, dtype=str, max_rows=1)
    p4 = np.loadtxt(filepath, skiprows=8, dtype=str, max_rows=1)
    lower_e = [float(p3[0]),float(p3[1]),float(p3[2]),float(p3[3]),
              float(p4[0]),float(p4[1]),float(p4[2])]
    
    return params, upper_e, lower_e, converg
    
    
    
    
    
def retrieve_all_celerite(TNSFile, data_dir, params_dir, gList,
                          datatag="-tessreduce", 
                          paramstag="celerite-matern32-residual-0.6-bounded"):
    """ 
    TNSFile = big csv file
    data_dir = path to data
    params_dir = where are all of the output files saved
    gList = list of targets to get
    datatag = what is the target labelled as (ie, '-tessreduce')
    paramstag = the celerite tag being used, ie, "celerite-matern32-residual-0.6-bounded"
    """
    tns_info = pd.read_csv(TNSFile)
    disc_all = retrieve_disctimes(data_dir, tns_info, gList, datatag)
    params_all = {}
    converged_all = {}
    upper_all = {}
    lower_all = {}
    
    #retrieve an d print out the things: 
    for root, dirs, files in os.walk(params_dir):
        for name in files:
            if name.endswith(paramstag+"-output-params.txt"):
                targ = name.split("-")[0][:-4]

                if targ not in gList:
                    continue
                
                filepath = root + "/" + name
                
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
    #print("conv", filerow1)
    if "True" in filerow1:
        converg = True
    else:
        converg = False
    
    #params row 1: 
    filerow1 = np.loadtxt(filepath, skiprows=3, dtype=str, max_rows=1)
    #print("params1", filerow1)
    sigsq, rho, t0, A = (float(filerow1[0]), 
                         float(filerow1[1]), 
                         float(filerow1[2]), 
                         float(filerow1[3]))
    
    filerow2 = np.loadtxt(filepath,  skiprows=4, dtype=str, max_rows=1)  
    #print("params2", filerow2)
    beta, B = (float(filerow2[0]), float(filerow2[1]))                                    

    params = (sigsq, rho, t0, A, beta, B)
    
    #upper error
    filerow1 = np.loadtxt(filepath, skiprows=5, dtype=str, max_rows=1)
    #print(filerow1)
    sigsq, rho, t0, A = (float(filerow1[0]), 
                         float(filerow1[1]), 
                         float(filerow1[2]), 
                         float(filerow1[3]))
    filerow2 = np.loadtxt(filepath, skiprows=6, dtype=str, max_rows=1)    
    #print(filerow2)
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

# (disc_all2, params_all2, 
#  converged_all2, upper_all2, 
#  lower_all2) = retrieve_all_celerite(TNSFile, data_dir, params_dir, gList,
#                            datatag="-tessreduce", 
#                            paramstag="residual-0.6")

def table_singlepower(params_all):
    for k in params_all.keys():
        t0, A, beta, b = params_all[k]
        print("{k} & {t0:.2f} & {A:.2f} & {be:.2f} & {b:.2f}".format(k=k,t0=t0, 
                                                                A=A, be=beta, b=b))

def table_differences(params_all1, params_all2):
    #print them out in table form:
    for k in params_all2.keys():
        t0 = np.abs(params_all1[k][0] - params_all2[k][0])
        A = np.abs(params_all1[k][1] - params_all2[k][1])
        beta = np.abs(params_all1[k][2] - params_all2[k][2])
        b = np.abs(params_all1[k][3] - params_all2[k][3])
        print("{k} & {t0:.2f} & {A:.2f} & {be:.2f} & {b:.2f}".format(k=k,t0=t0, 
                                                                    A=A, be=beta, b=b))

def table_celerite(params_all):
    for k in params_all.keys():   
        #print(k, params_all2[k])
        t0, A, beta, b, sig, rho = params_all[k]
        print("{k} & {t0:.2f} & {A:.2f} & {be:.2f} & {b:.2f}".format(k=k,t0=t0, 
                                                                    A=A, be=beta, b=b))
        print(" & {sig:.2f} & {r:.2f} ".format(sig = np.exp(2*sig), r=np.exp(rho)))

#table_differences(params_all1, params_all2)