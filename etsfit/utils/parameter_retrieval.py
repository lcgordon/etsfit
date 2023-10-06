#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:54:28 2022
Updated Oct 6 2023

Batch retrieval methods for parameters in output files in the same parent directory
This is not complete by any means. 
"""

#load parameters from files
import pandas as pd
import numpy as np
import os
from astropy.time import Time
import etsfit.utils.utilities as ut


def retrieve_disctimes(data_dir, csv_file, searchlist, datatag, colheader="Discovery Date (UT)"):
    """ 
    For a list of targets in a data directory, pull their discovery times from a corresponding csv file,
    default column is the correct one for a TNS output. 
    
    :param data_dir: where is the data
    :param csv_file: csv formatted file of info on targets
    :param searchlist: list/array of names of targets to retrieve
    :param datatag: how is data named in folder
    :param colheader: which column has discovery date in it - uses tns default
    """
    disctimeall = {}
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith(datatag):
                targ = name.split("-")[0][:-4]
                #print(targ)
                if targ not in searchlist:
                    continue
                holder = root + "/" + name
                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder, printname=False)

                #get discovery time
                d = csv_file[csv_file["Name"].str.contains(targetlabel)][colheader]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                tmin = time[0]
                time = time - tmin
                disctime = discoverytime-tmin
                disctimeall[targetlabel] = disctime
    return disctimeall 

def retrieve_all_singlepower_noCBV(data_dir, csv_file, searchlist, params_dir,
                               datatag="-tessreduce", paramstag="singlepower-0.6"):
    """ 
    Pull all parameters out of output files for a given set of targets, all coming from a 
    single power law with no CBV output. 

    :param data_dir: where is your data
    :param csv_file: csv of data info
    :param searchlist: list of targets to retrieve
    :param params_dir: where is the outputs from running etsfit
    :param datatag: data file tag used
    :param paramstag: parameter output file tag used
    """
    tns_info = pd.read_csv(csv_file)
    disc_all = retrieve_disctimes(data_dir, tns_info, searchlist, datatag)
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
                if targ not in searchlist:
                    continue
                filepath = root + "/" + name
                (params,  upper_e, 
                 lower_e,  converg) = extract_singlepower_params(filepath)
                
                params_all[targ] = params
                upper_all[targ] = upper_e
                lower_all[targ] = lower_e
                converged_all[targ] = converg
    return (disc_all, params_all, converged_all, upper_all, lower_all)

def extract_singlepower_params(filepath):
    """ 
    Pulls parameters out of a singlepower etsfit run output file

    :param filepath: file to pull from
    """
    # target label row 0, bic row 1, covergence row 2
    filerow1 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
    if "True" in str(filerow1):
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

def extract_flat_params(filepath): 
    """ 
    Pulls parameters out of a flat background etsfit run output file

    :param filepath: file to pull from
    :return: converged (bool), B (float), upper error, lower error
    """
    filerow1 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
    #print(filerow1)
    if "True" in str(filerow1):
        converg = True
    else:
        converg = False
    fr2 = np.loadtxt(filepath, skiprows=3, dtype=str, max_rows=1)
    #print(fr2)
    fr3 = np.loadtxt(filepath, skiprows=4, dtype=str, max_rows=1)
    #print(fr3)
    fr4 = np.loadtxt(filepath, skiprows=5, dtype=str, max_rows=1)
    #print(fr4)
    return  float(fr2), float(fr3), float(fr4), converg

def extract_doublepower_params(filepath):
    """ 
    Pulls parameters out of a double power no CBV etsfit run output file

    :param filepath: file to pull from
    :return: params, upper error, lower error, convergence
    """
    
    # target label row 0, bic row 1, convg row 2:
    filerow1 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
    if "True" in str(filerow1):
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
    
def retrieve_all_celerite(data_dir, csv_file, searchlist, params_dir, 
                          datatag="-tessreduce", 
                          paramstag="celerite-matern32-residual-0.6-bounded"):
    """ 
    Pull all parameters out of output files for a given set of targets, all coming from a 
    celerite run.

    :param data_dir: where is your data
    :param csv_file: csv of data info
    :param searchlist: list of targets to retrieve
    :param params_dir: where is the outputs from running etsfit
    :param datatag: data file tag used
    :param paramstag: parameter output file tag used
    """
    tns_info = pd.read_csv(csv_file)
    disc_all = retrieve_disctimes(data_dir, tns_info, searchlist, datatag)
    params_all = {}
    converged_all = {}
    upper_all = {}
    lower_all = {}
    
    #retrieve an d print out the things: 
    for root, dirs, files in os.walk(params_dir):
        for name in files:
            if name.endswith(paramstag+"-output-params.txt"):
                targ = name.split("-")[0][:-4]

                if targ not in searchlist:
                    continue
                
                filepath = root + "/" + name
                
                (params,  upper_e, 
                 lower_e,  converg) = extract_celerite_all(filepath)
                
                params_all[targ] = params
                upper_all[targ] = upper_e
                lower_all[targ] = lower_e
                converged_all[targ] = converg    
    return disc_all, params_all, converged_all, upper_all, lower_all

def extract_celerite_params(filepath):
    """ 
    Pulls parameters out of a celerite etsfit run output file

    :param filepath: file to pull from
    """
    # target label row 0, bic row 1, convg row 2
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