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

datafolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
foldersave = "/Users/lindseygordon/research/urop/plotOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"

#cmd + 1 to comment
t0 = []
A = []
beta = []
B = []
for root, dirs, files in os.walk(foldersave):
    for name in files:
        if name.endswith("singlepower-output-params.txt"):
            filepath = root + "/" + name
            #print(filepath)
            filerow1 = np.loadtxt(filepath, skiprows=0, dtype=str, max_rows=1)
            #filerow2 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
            #filerow3 = np.loadtxt(filepath, skiprows=3, dtype=str, max_rows=1)
            #print(filerow1)
            if filerow1[0] == "[": #first string is just [
                
                t0.append(float(filerow1[1]))
                A.append(float(filerow1[2]))
                beta.append(float(filerow1[3]))
                B.append(float(filerow1[4][:-1]))
            
            else: #first string contains [
                #print(filerow1, filerow1[0][1:],filerow1[3][:-1])
                t0.append(float(filerow1[0][1:]))
                A.append(float(filerow1[1]))
                beta.append(float(filerow1[2]))
                B.append(float(filerow1[3][:-1]))

            
def plot_histogram(data, bins, x_label, filename):
    """ 
    Plot a histogram with one light curve from each bin plotted on top
    * Data is the histogram data
    * Bins is bins for the histogram
    * x_label for the x-axis of the histogram
    * filename is the exact place you want it saved
    """
    rcParams['figure.figsize'] = 10,10
    fig, ax1 = plt.subplots()
    n_in, bins, patches = ax1.hist(data, bins)
    
    y_range = np.abs(n_in.max() - n_in.min())
    x_range = np.abs(data.max() - data.min())
    ax1.set_ylabel('Number of light curves')
    ax1.set_xlabel(x_label)
    #ax1.set_xticks(fontsize=10)
    
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
    plt.show()
    #plt.close()
    rcParams['figure.figsize'] = 16,6
    return 

plot_histogram(np.asarray(beta), 32, "beta", "/Users/lindseygordon/research/urop/plotOutput/beta-all.png")


# filepath = "/Users/lindseygordon/research/urop/plotOutput/2018exc0111/singlepower/2018exc0111-singlepower-output-params.txt"
# filerow1 = np.loadtxt(filepath, skiprows=0, dtype=str, max_rows=1)
# filerow2 = np.loadtxt(filepath, skiprows=2, dtype=str, max_rows=1)
# filerow3 = np.loadtxt(filepath, skiprows=3, dtype=str, max_rows=1)