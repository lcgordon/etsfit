# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:14:13 2022

@author: lcgordon
"""
import utils.utilities as ut
import utils.snPlotting as sp
import utils.MCMC as mc

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 16,6
rcParams["font.size"] = 20

from celerite.modeling import Model
from scipy.optimize import minimize
import celerite
from celerite import terms

import os
import emcee
import time as timeModule
import pandas as pd





class etsMAIN(object):
    """Make one of these for ONE light curve you're going to fit """
    
    def __init__(self, folderSAVE, bigInfoFile, CBV_folder,
                 quaternion_folder_raw, quaternion_folder_txt):
        """Load these bad bois in """
        self.info = pd.read_csv(bigInfoFile)
        self.bigInfoFile = bigInfoFile
        self.folderSAVE = folderSAVE
        self.CBV_folder = CBV_folder
        self.quaternion_folder_raw = quaternion_folder_raw
        self.quaternion_folder_txt = quaternion_folder_txt
        
    def load_data_lygos_single(self, fileToLoad, override=False):
        """Given a SPECIFIC filepath, load in data + information
        and when i say SPECIFIC i mean like, 
        "D:/18th1aAll/SN2018eod/lygos/data/rflxtarg_SN2018eod_0114_30mn_n005_d4.0_of11.csv"
        """
        pieces = fileToLoad.split("_")
        #print(pieces)
        #look up sector of discovery in big file
        self.sector = self.info[self.info["Name"].str.contains(pieces[1][2:])]["Sector"].iloc[0]
        #load in
        if self.sector < 10:
            self.sector = "0" + str(self.sector)
        if pieces[2].startswith(str(self.sector)) or override==True:
            self.targetlabel = pieces[1]
            self.sector = pieces[2][0:2]
            self.camera=pieces[2][2]
            self.ccd = pieces[2][3]
        
            print("LOADING IN:", self.targetlabel, self.sector, self.camera, self.ccd)
            
            (self.time, self.intensity, 
             self.error, self.lygosbg) = ut.load_lygos_csv(fileToLoad)
            
            self.disctime = ut.get_disctime(self.bigInfoFile, self.targetlabel[2:])
            
            (self.time, self.intensity, 
            self.error, self.lygosbg) =  ut.normalize_sigmaclip(self.time, self.intensity, 
                                                                self.error, self.lygosbg) 
            self.tmin = self.time[0]
            self.time -= self.tmin
            self.disctime -= self.tmin
            self.bic_all = []
            self.params_all = []
            
            return
        else: 
            print("Not discovery sector data! Not loading anything in.")
            print("If you want to load in anyways, pass override=True.")
            return
        
    def run_MCMC(self, fitType, binYesNo, fraction = None, n1=10000, n2=40000,
                 saveBIC=False):
        """Run one MCMC instance
        fitType is an integer 1-6
        binYesNo is true/false do you want it binned to 8 hours
        fraction = None if not trimming, 0.6 for 60% or whatever otherwise
        n1 is burnin steps n2 is production upper limit
        """
        best, bic, QCBVs = mc.mcmc_outer_structure(self.folderSAVE, self.targetlabel,
                                                   self.time, self.intensity, self.error,
                                                   self.lygosbg, self.tmin, self.sector, 
                                                   self.camera, self.ccd, self.disctime,
                                                   fitType=fitType, fract=fraction,
                                                   Bin8Hr = binYesNo, n1=n1, n2=n2,
                                                   CBV_folder = self.CBV_folder,
                                                   qfolder = self.quaternion_folder_txt)
        if saveBIC:
            self.bic_all.append(bic)
            self.params_all.append(best)
            
        return best, bic, QCBVs
    
    def run_multiple_MCMC_from_folder(self, folderToLoadFrom, fitType, binYesNo, 
                                      fraction = None, n1=1000, n2=40000, 
                                      saveBIC=False):
        """Load all in the rflxtarg's within the folder"""
        
        
        for root, dirs, files in os.walk(folderToLoadFrom):
            for f in files:
                if "rflxtarg" in f:
                    self.load_data_lygos_single(os.path.join(root,f))
                    self.run_MCMC(fitType,binYesNo, fraction,n1,n2,saveBIC)
                    
        
                    
#%%
folderLOAD = "D:/18thIaAll/"
folderSAVE = "D:/packagetesting/"
testSN = "D:/18th1aAll/SN2018eod/lygos/data/rflxtarg_SN2018eod_0114_30mn_n005_d4.0_of11.csv"
CBV_folder = "C:/Users/conta/.eleanor/metadata/"
quaternion_folder_raw = "D:/quaternions-raw/"
quaternion_folder_txt = "D:/quaternions-txt/"
bigInfoFile = "D:/18thmag_Ia.csv"

etstest = etsMAIN(folderSAVE, bigInfoFile, CBV_folder,
                 quaternion_folder_raw, quaternion_folder_txt)


folderToLoadFrom = "D:/specialBabies/"
etstest.run_multiple_MCMC_from_folder(folderToLoadFrom, 1, False)

#etstest.load_data_lygos_single(testSN)

#etstest.run_MCMC(1,False)