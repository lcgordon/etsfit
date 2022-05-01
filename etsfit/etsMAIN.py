# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:14:13 2022

@author: lcgordon

todo list:
    - load in priors and ability to set own priors!
    - put GP code fitting in
    - test all of the diff fits like this
    - what if you need to make the quat .txt files
    - enforce lygosbg = None not allowing that one to work
"""
import utils.utilities as ut
import utils.snPlotting as sp
import utils.MCMC as mc
import time as timeModule
import pandas as pd
# from scipy.optimize import minimize
import numpy as np
#import matplotlib.pyplot as plt
import os
import emcee
import datetime 
from pylab import rcParams
rcParams['figure.figsize'] = 16, 6
rcParams["font.size"] = 20

# from celerite.modeling import Model
# import celerite
# from celerite import terms







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
        return
    
    def make_quatsTxt(self):
        """Only run me if you don't have the quat.txt files yet 
        should skip generation if they already exist in the txt folder"""
        ut.make_quat_txtfiles(self.quaternion_folder_raw, self.quaternion_folder_txt)
        return
    
    def load_data_lygos_single(self, fileToLoad, override=False):
        """Given a SPECIFIC filepath, load in data + information
        and when i say SPECIFIC i mean like, 
        "D:/18th1aAll/SN2018eod/lygos/data/rflxtarg_SN2018eod_0114_30mn_n005_d4.0_of11.csv"
        """
        pieces = fileToLoad.split("_")
        # look up sector of discovery in big file
        self.sector = self.info[self.info["Name"].str.contains(pieces[1][2:])]["Sector"].iloc[0]
        # load in
        if (self.sector < 10):
            self.sector = "0" + str(self.sector)
        if (pieces[2].startswith(str(self.sector)) or override == True):
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
        
    def load_your_lc_in(self, time, intensity, error , disctime,
                        sector, camera, ccd):
        """load in your own light cuve. 
        assumes time & disctime has not be tmin subtracted
        lygosbg = None in this case (cannot run that one - )
        seems to be working fine 
        
        """
        
        self.time = time
        self.intensity = intensity
        self.error = error
        self.lygosbg = None
        self.disctime = disctime
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.tmin = time[0]
        self.time-=tmin
        self.disctime-=tmin
        return
    
    def custom_mask_it(self, cutIndices):
        """remove certain indices from your light curve.
        cutIndices should be an array of size len(time), 0 = re,ove, 1=keep
        this should NOT be used if using CBVs - that's not set up yet!!
        """
        if hasattr(self, time):
            a = np.nonzero(cutIndices) # which ones you are keeping
            print(a)
            self.time = self.time[a]
            self.intensity = self.intensity[a]
            self.error = self.error[a]
            if self.lygosbg is not None:
                self.lygosbg = self.lygosbg[a]
        else:
            print("No time loaded in yet!! Run again once light curve is loaded")
        return
        
    def run_MCMC(self, fitType, binYesNo, fraction = None, n1=1000, n2=10000,
                 saveBIC=False):
        """Run one MCMC instance
        fitType is an integer 1-6
        binYesNo is true/false do you want it binned to 8 hours
        fraction = None if not trimming, 0.6 for 60% or whatever otherwise
        n1 is burnin steps n2 is production upper limit
        """
        
        # ####################
        # handle CBVs if necessary:
        # ####################
        if fitType in (2,4,5): 
            (self.time, self.intensity, self.error,
             self.quatTime, self.quatsIntensity, self.CBV1, self.CBV2,
             self.CBV3) = ut.generate_clip_quats_cbvs(self.sector, self.time,
                                                      self.intensity,self.error, 
                                                      self.tmin, self.camera, self.ccd,
                                                      self.CBV_folder, 
                                                      self.quaternion_folder_txt)
            self.quatsandcbvs = [self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3]
        else:
            self.quatsandcbvs = None #has to initiate as None or it'll freak
            
        # ########################
        # check for 8hr bin BEFORE trimming to percentages
        # ########################
        if binYesNo: #if need to bin
            (self.time, self.intensity, 
             self.error, self.lygosBG,
             self.quatsandcbvs) = ut.bin_8_hours(self.time, self.intensity, self.error, 
                                                 self.lygosBG, QCBVALL=self.quatsandcbvs) 
                                                 
        # if doing percent of max fitting
        if fraction is not None:
            (self.time, self.intensity, self.error, self.lygosBG, 
             self.quatsandcbvs) = ut.fractionalfit(self.time, self.intensity, 
                                                   self.error, self.lygosBG, 
                                                   fraction, self.quatsandcbvs)
                               
                                                
        # load parameters by fit type                                           
        self.__setup_fittype_params(fitType, binYesNo, fraction)
        self.__gen_output_folder() 
                                                        
        # run it
        best, bic = self.__mcmc_outer_structure(fitType, n1, n2)
        if saveBIC:
            self.bic_all.append(bic)
            self.params_all.append(best)
            
        return best, bic
    
    def run_multiple_MCMC_from_folder(self, folderToLoadFrom, fitType, binYesNo, 
                                      fraction = None, n1=1000, n2=40000, 
                                      saveBIC=False):
        """Load all in the rflxtarg's within the folder"""
        
        
        for root, dirs, files in os.walk(folderToLoadFrom):
            for f in files:
                if "rflxtarg" in f:
                    self.load_data_lygos_single(os.path.join(root,f))
                    self.run_MCMC(fitType,binYesNo, fraction,n1,n2,saveBIC)
                    
    def __setup_fittype_params(self, fitType, binYesNo, fraction, args=None, logProbFunc = None,
                               filesavetag=None, labels=None, init_values=None):
        """
        1 = single without
        2 = single with cbv
        3 = doubel without
        4 = double with
        5 = just cbv
        6 = single lygos bg
        0 = custom inputs
        any other number = will exit with an error
        """
        if fitType == 1: # single without
            self.args = (self.time, self.intensity, self.error, self.disctime)
            self.logProbFunc = mc.log_probability_singlepower_noCBV
            self.filesavetag = "-singlepower"
            self.labels = ["t0", "A", "beta",  "b"]
            self.init_values = np.array((self.disctime-3, 0.1, 1.8, 1))
            
            
        elif fitType == 2: # single with
            self.args = (self.time, self.intensity, self.error, 
                         self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3, 
                         self.disctime)
            self.logProbFunc = mc.log_probability_singlepower_withCBV
            self.filesavetag = "-singlepower-CBV"
            self.labels = ["t0", "A", "beta", "B", "cQ", "c1", "c2", "c3"]
            self.init_values = np.array((self.disctime-3, 0.1, 1.8, 0, 0,0,0,0))
        elif fitType == 3: # double without
            self.args = (self.time, self.intensity, self.error, self.disctime)
            self.logProbFunc = mc.log_probability_doublepower_noCBV
            self.filesavetag = "-doublepower"
            self.labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  "b"]
            self.init_values = np.array((self.disctime-8, self.disctime-2, 0.1, 0.1, 1.8, 1.8, 1))
        elif fitType ==4: # double with
            self.args = (self.time, self.intensity, self.error,
                         self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3, 
                         self.disctime)
            self.logProbFunc = mc.log_probability_doublepower_withCBV
            self.filesavetag = "-doublepower-CBV"
            self.labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  
                      "cQ", "c1", "c2", "c3"]
            self.init_values = np.array((self.disctime-8, self.disctime-2, 0.1, 0.1, 
                                    1.8, 1.8, 0,0,0,0))
        elif fitType == 5: # just CBVs
            self.args = (self.time, self.intensity, self.error, 
                         self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3, 
                         self.disctime)
            self.logProbFunc = mc.log_probability_justCBV
            self.filesavetag = "-CBV"
            self.labels = ["b", "cQ", "c1", "c2", "c3"]
            self.init_values = np.array((1, 0,0,0,0))
        elif fitType == 6: # detrending lygos BG
            self.args = (self.time, self.intensity, self.error, self.lygosBG, 
                         self.disctime)
            self.logProbFunc = mc.log_probability_singlePower_LBG
            self.filesavetag = "-singlepower-lygosBG"
            self.labels = ["t0", "A", "beta",  "b", "LBG"]
            self.init_values = np.array((self.disctime-3, 0.1, 1.8, 1, 1))
        elif fitType == 0: # diy your stuff
            self.args = args
            self.logProbFunc = logProbFunc
            self.filesavetag = filesavetag
            self.labels = labels
            self.init_values = init_values 
        else:
            print("THAT IS NOT AN ALLOWED FIT TYPE, EXITING")
            raise ValueError("not an allowed fit type")
            
        if binYesNo: # it has to go in this order - need to load, then set args, then set this
            self.filesavetag = self.filesavetag + "-8HourBin"
    
        if fraction is not None:
            self.filesavetag = self.filesavetag + "-{fraction}".format(fraction=fraction)
        return
            
            
    def __gen_output_folder(self):
        """set up output folder & files """
        # check for an output folder's existence, if not, put it in. 
        newfolderpath = (self.folderSAVE + self.targetlabel + 
                         str(self.sector) + str(self.camera) + str(self.ccd))
        if not os.path.exists(newfolderpath):
            os.mkdir(newfolderpath)
        # make subfolder for this run
        subfolderpath = newfolderpath + "/" + self.filesavetag[1:]
        if not os.path.exists(subfolderpath):
            os.mkdir(subfolderpath)
        self.folderSAVE = subfolderpath + "/"
        self.parameterSaveFile = self.folderSAVE + "output_params.txt"
                    
    def __mcmc_outer_structure(self, fitType, n1, n2):
        """Fitting things that are NOT GP based
        fitType may be getting moved into another thingy
        """
        
        print("***")
        print("***")
        print("***")
        print("***")
        print("Beginning MCMC run")
         
        timeModule.sleep(3) # this keeps things running orderly

        # ### MCMC setup
        np.random.seed(42)
        
        nwalkers = 100
        ndim = len(self.labels) # labels are provided when you run it
        p0 = np.zeros((nwalkers, ndim)) # init positions
        for n in range(len(p0)): # add a little spice - YYY gaussian??
            p0[n] = self.init_values + (np.ones(ndim) - 0.9) * np.random.rand(ndim) 
        
        # ### Initial run
        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                        self.logProbFunc,args=self.args) # setup
        sampler.run_mcmc(p0, n1, progress=True) # run it
        sp.plot_chain_logpost(self.folderSAVE, self.targetlabel, self.filesavetag, 
                              sampler, self.labels, ndim, appendix = "-burnin")
        
        discardy = int(n1/2)
        flat_samples = sampler.get_chain(discard=discardy, flat=True, thin=15)
        # get intermediate best
        best_mcmc_inter = np.zeros((1,ndim))
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            best_mcmc_inter[0][i] = mcmc[1]
            
        # ### Main run
        np.random.seed(50)
        p0 = np.zeros((nwalkers, ndim))
        for i in range(nwalkers): # reinitialize the walkers around prev. best
            p0[i] = best_mcmc_inter[0] + 0.1 * np.random.rand(1, ndim)
           
        sampler.reset()
        
        # ### CORRELATION FUNCTION 
        index = 0 # number of checks
        autocorr = np.empty(n2) # total possible checks
        old_tau = np.inf
        autoStep = 1000 # how often to check
        autocorr_all = np.empty((int(n2/autoStep) + 2,len(self.labels))) # save all autocorr times
        
        # sample up to n2 steps
        for sample in sampler.sample(p0, iterations=n2, progress=True):
            # Only check convergence every 100 steps
            if sampler.iteration % autoStep:
                continue
            # Compute the autocorrelation time so far
            tau = sampler.get_autocorr_time(tol=0) # tol=0 always get estimate
            if np.any(tau == np.nan) or np.any(tau == np.inf) or np.any(tau == -np.inf):
                print("autocorr is nan or inf")
                print(tau)
            # this pops out with len(tau) = ndims - need all to converge to be conv
            autocorr[index] = np.mean(tau) # save mean autocorr time
            autocorr_all[index] = tau # save all autocorr times for plotting
            index += 1 # how many times have you saved it
        
            # Check convergence
            # this first condition is absolutely where it's failing
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01) # normally 0.01
            if converged:
                print("Converged, ending chain")
                break
            old_tau = tau
        
        # #plot autocorr things
        sp.plot_autocorr_mean(self.folderSAVE, self.targetlabel, index, 
                              autocorr, converged, 
                              autoStep, self.filesavetag)
        
        sp.plot_autocorr_individual(self.folderSAVE, self.targetlabel, index,
                                    autocorr_all, autoStep, self.labels, 
                                    self.filesavetag)
        
        #thin and burn out dump
        tau = sampler.get_autocorr_time(tol=0)
        if (np.max(tau) < (sampler.iteration/50)):
            burnin = int(2 * np.max(tau))
            thin = int(0.5 * np.min(tau))
        else:
            burnin = 5000
            thin = 15
        flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        
        # this will be separate - plotting p(parameter)
        sp.plot_paramIndividuals(flat_samples, self.labels, self.folderSAVE, 
                                 self.targetlabel, self.filesavetag)
        
        sp.plot_chain_logpost(self.folderSAVE, self.targetlabel, self.filesavetag,
                              sampler, self.labels, ndim, appendix = "-production")
        
        print(len(flat_samples), "samples post second run")
    
        # ### BEST FIT PARAMS
        best_mcmc = np.zeros((1,ndim))
        upper_error = np.zeros((1,ndim))
        lower_error = np.zeros((1,ndim))
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            print(self.labels[i], mcmc[1], -1 * q[0], q[1] )
            best_mcmc[0][i] = mcmc[1]
            upper_error[0][i] = q[1]
            lower_error[0][i] = q[0]
     
        logprob, blob = sampler.compute_log_prob(best_mcmc)
        # print(logprob)
        # ### BIC
        BIC = ndim * np.log(len(self.time)) - 2 * np.log(logprob)
        print("BAYESIAN INF CRIT: ", BIC)
        if np.isnan(np.float64(BIC[0])): # if it's a nan
            BIC = 50000
        else:
            BIC = BIC[0]
            
        
        sp.plot_mcmc(self.folderSAVE, self.time, self.intensity, self.targetlabel, 
                     self.disctime, best_mcmc[0], 
                     flat_samples, self.labels, fitType, self.filesavetag, 
                     self.tmin, self.lygosbg,
                     self.quatsandcbvs)
        
        with open(self.parameterSaveFile, 'w') as file:
            file.write(self.filesavetag + "-" + str(datetime.datetime.now()))
            file.write("\n {best} \n {upper} \n {lower} \n".format(best=best_mcmc,
                                                                   upper=upper_error,
                                                                   lower=lower_error))
            file.write("BIC:{bicy:.3f} Converged:{conv} \n".format(bicy=BIC, 
                                                                conv=converged))
        
        return best_mcmc, BIC




        
                    
# %%
folderLOAD = "D:/18thIaAll/"
folderSAVE = "D:/packagetesting/"
CBV_folder = "C:/Users/conta/.eleanor/metadata/"
quaternion_folder_raw = "D:/quaternions-raw/"
quaternion_folder_txt = "D:/quaternions-txt/"
bigInfoFile = "D:/18thmag_Ia.csv"
#testSN = "D:/18th1aAll/SN2018eod/lygos/data/rflxtarg_SN2018eod_0114_30mn_n005_d4.0_of11.csv"
testSN = "D:/specialBabies/SN2018hzh/lygos/data/rflxtarg_SN2018hzh_0431_30mn_n005_d4.0_of11.csv"


etstest = etsMAIN(folderSAVE, bigInfoFile, CBV_folder,
                 quaternion_folder_raw, quaternion_folder_txt)


# folderToLoadFrom = "D:/specialBabies/"
# etstest.run_multiple_MCMC_from_folder(folderToLoadFrom, 1, False)

etstest.load_data_lygos_single(testSN)

etstest.run_MCMC(2,False)

