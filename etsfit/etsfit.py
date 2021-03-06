# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:14:13 2022

@author: lcgordon

todo list:
    - double check GP stuff is working [seems fine as of 05092022]
    - save a plot of trimmed regions for when doing custom masking. [done 05092022]
    - set up GP parameter scan function
    - TEST GP CUSTOM PRIOR/INPUTS 
"""

import etsfit.utils.utilities as ut
import etsfit.utils.snPlotting as sp
import etsfit.utils.MCMC as mc
#import utils.utilities as ut
#import utils.snPlotting as sp
#import utils.MCMC as mc

import time as timeModule
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import emcee
import datetime 
from pylab import rcParams
rcParams['figure.figsize'] = 16, 6
rcParams["font.size"] = 20

from celerite.modeling import Model
import celerite
from celerite import terms

import warnings
warnings.filterwarnings("ignore")


class etsMAIN(object):
    """Make one of these for the light curve you're going to fit!"""
    
    def __init__(self, folderSAVE, bigInfoFile):
        """
        Initialize an etsfit object
        
        This sets up a save path, loads in a csv of targets if you want,
        and sets default values of the data not being trimmed, binned, and
        there being no CBVs/quaternions accessible.
        
        -----------------------------------------------
        Parameters:
            
            - folderSAVE (str) the path to where you want everything to be saved
            should end with a /, it will self-correct if not
            
            - bigInfoFile (str) the path to a big pandas-readable CSV file
            containing all the targets you could possibly care about
        
        """
        
        if not os.path.exists(folderSAVE):
            raise ValueError("Not a valid save path!")
        else:
            if folderSAVE[-1] != "/": #if not ending with /, 
                folderSAVE += "/"
                self.folderSAVE = folderSAVE
                self.foldersaveperm = folderSAVE #keep a copy in case
        
        
        if os.path.exists(bigInfoFile):
            self.info = pd.read_csv(bigInfoFile)
        self.bigInfoFile = bigInfoFile
        
        
        self.cbvsquats = False #no cbvs/quats in
        self.fractiontrimmed = False #not trimmed
        self.binned = False #not binned
        return
    
    def use_quaternions_cbvs(self, CBV_folder,quaternion_folder_raw, 
                             quaternion_folder_txt):
        """ 
        Function to set up access to CBVs and quaternions. 
        If you do not have text file versions of the quaternions, they will
        be generated for you.
        
        ----------------------------------------------------
        Parameters: 
            
            - CBV_folder (str) a path to the folder holding all CBVs. 
            this is probably within the eleanor directory
            the directory structure should be such that there are files like
            folder/s001/cbv_components_s0001_0001_0001.txt
            
            - quaternion_folder_raw (str) is the path to the folder holding
            all of the .fits file versions of the quaternions. this can be
            some random path if you already have produced the txt versions
            
            - quaternion_folder_txt (str) is the path to the folder either
            HOLDING the .txt file versions of the quaternions OR the path to
            the EMPTY folder that all the txt versions are about 
            to be generated into
        """
        
        if not os.path.exists(CBV_folder):
            raise ValueError("Not a valid CBV path")
        else:
            self.CBV_folder = CBV_folder
            
        if not :
            raise ValueError("Not  a valid raw quat path")
        else:
            self.quaternion_folder_raw = quaternion_folder_raw
        
        if not os.path.exists(quaternion_folder_txt):
            raise ValueError("Not a valid txt quat path")
        else:
            self.quaternion_folder_txt = quaternion_folder_txt
            if (len(os.listdir(self.quaternion_folder_txt))==0 &&
                len(os.listdir(self.quaternion_folder_raw))!=0):
                #empty directory, make txt files
                ut.make_quat_txtfiles(self.quaternion_folder_raw, self.quaternion_folder_txt)
                
        self.cbvsquats = True
        return
    
    def window_rms_filt(self, plot=True):
        """ 
        Runs an RMS filter over the light curve and returns an array of 
        0's (bad) and 1's (good) that can be used in the custom masking
        argument of other functions. 
        
        This DOES NOT save the filter output inside the object. 
        
        Defaults the inner window as len(self.time)*0.005 and the outer as
        inner*20. Custom arguments are on the to-do list, nudge Lindsey if 
        you need them.
        
        -------------------------------
        Parameters:
            
            - plot (bool) defaults as True, plots light curve w/ mask
        
        """
        innersize = int(len(self.time)*0.005)
        outersize = innersize * 20
        print(innersize, outersize)
        n = len(self.time)
        rms_filt = np.ones(n)
        for i in range(n):
            outer_lower = max(0, i-outersize) #outer window, lower bound
            outer_upper = min(n, i+outersize) #outer window, upper bound
            inner_lower = max(0, i-innersize) #inner window, lower bound
            inner_upper = min(n, i+innersize) #inner window, upper bound
            
            outer_window = self.intensity[outer_lower:outer_upper]
            inner_window = self.intensity[inner_lower:inner_upper]
            
            std_outer = np.std(outer_window)
            
            rms_outer = np.sqrt(sum([s**2 for s in outer_window])/len(outer_window))
            rms_inner = np.sqrt(sum([s**2 for s in inner_window])/len(inner_window))
            
            if ((rms_inner > (rms_outer + std_outer)) 
                or (rms_inner < (rms_outer - std_outer))):
                rms_filt[inner_lower:inner_upper] = 0 #bad point, discard
                print(rms_inner, rms_outer, std_outer)
        
        if plot:
            rms_filt_plot = np.nonzero(rms_filt)
            plt.scatter(self.time, self.intensity, color='green', label='bad')
            plt.scatter(self.time[rms_filt_plot], self.intensity[rms_filt_plot], color='blue', label='good')
            plt.legend()
        return rms_filt
        
    
    
    def load_data_lygos_single(self, fileToLoad, disctime=None, override=False):
        """
        Given a SPECIFIC filepath to a lygos lightcurve, load in the data
        And I do mean SPECIFIC path.
        
        --------------------------------------------
        Parameters:
            
            - fileToLoad (str), 
            ie "D:/18th1aAll/SN2018eod/rflxtarg_SN2018eod_0114_30mn_n005_d4.0_of11.csv"
            
            - disctime (double, defaults to NONE) if no big CSV file is loaded, 
            provide the discovery time (or custom disctime)
            
            - override (bool, defaults to FALSE) ignore it if loading data
            from a sector that is NOT the discovery sector
        
        """
        pieces = fileToLoad.split("_")
        # look up sector of discovery in big file
        self.sector = self.info[self.info["Name"].str.contains(pieces[1][2:])]["Sector"].iloc[0]
        # load in
        if (self.sector < 10):
            self.sector = "0" + str(self.sector)
        if (pieces[2].startswith(str(self.sector)) or override == True):
            
            time, intensity, error, lygosbg = ut.load_lygos_csv(fileToLoad)
            if self.bigInfoFile is None and disctime is None:
                raise Exception("No big info file given AND no disctime was provided")
            elif disctime is None:
                disctime = ut.get_disctime(self.bigInfoFile, pieces[1][2:])
            
            self.load_custom_lc(time, intensity, error, lygosbg, disctime, pieces[1],
                        pieces[2][0:2], pieces[2][2], pieces[2][3])
            
        
            print("LOADING IN:", self.targetlabel, "SECTOR: ", self.sector, "CAMERA: ",
			self.camera, "CCD: ", self.ccd)
            
            (self.time, self.intensity, 
            self.error, self.lygosbg) =  ut.normalize_sigmaclip(self.time, self.intensity, 
                                                                self.error, self.lygosbg) 
            self.tmin = self.time[0]
            self.time -= self.tmin
            self.disctime -= self.tmin
            self.bic_all = []
            self.params_all = []
            self.xlabel = "BJD - {timestart:.3f}".format(timestart=self.tmin)
            self.ylabel = "Rel. Flux"
            
            return
        else: 
            raise ValueError("Not discovery sector data  \n" + 
                             "If you want to load in anyways, pass override=True")
        return

    
    def load_single_lc(self, time, intensity, error, discoverytime, 
                       targetlabel, sector, camera, ccd, lygosbg=None):
        """ 
        Load in one light curve from information you supply
        
        ----------------------------------
        Parameters:
            
            - time (array) time axis for the lc. this will get the 0th
            index subtracted off. 
            
            - intensity (array) flux array for the lc 
            [yyy handle pre-cleaned data?]
            
            - error (array) error on flux array
            
            - discoverytime (double) when ground telescopes found it
            
            - targetlabel (str) no spaces name, will be used on files
            
            - sector (str) this needs to be a two-character string (ie, sector
                                                                    2 is "02")
            
            - camera (str) needs to be a 1 char string
            
            - ccd (str) needs to be a 1 char string
            
            - lygosbg (defaults to NONE) can put in lygos background array if 
            doing that one specific fit. should also work to float other 
            background arrays.
        """
        
        if (len(time) != len(intensity) or len(intensity) != len(error) or
            len(time) != len(error)):
            print("Time length:", len(time))
            print("Intensity length:", len(intensity))
            print("Error length:", len(error))
            raise ValueError("Mismatched sizes on time, intensity, and error!")
        elif (time is None or intensity is None or 
              error is None or discoverytime is None or 
              targetlabel is None or sector is None or 
              camera is None or ccd is None):
            raise ValueError("Inputs all have to be SOMETHING you can't give any None's here")
        elif type(sector) != str:
            raise ValueError("Sector must be a string, see help()")
        elif type(camera) != str:
            raise ValueError("camera must be a string, see help()")
        elif type(ccd) != str:
            raise ValueError("ccd must be a string, see help()")
        
        self.time = time
        self.intensity = intensity
        self.error = error
        self.lygosbg = lygosbg
        self.disctime = discoverytime
        self.targetlabel = targetlabel
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.tmin = time[0]
        self.time-=self.tmin
        self.disctime-=self.tmin
        self.bic_all = []
        self.params_all = []
        self.xlabel = "BJD - {timestart:.3f}".format(timestart=self.tmin)
        self.ylabel = "Rel. Flux"
        return

    
    
    def run_MCMC(self, fitType, cutIndices=None, binYesNo = False, fraction = None, 
                 n1=1000, n2=10000, thinParams = None,
                 saveBIC=False, args=None, logProbFunc = None, plotFit = None,
                 filesavetag=None,
                 labels=None, init_values=None):
        """
        Run one MCMC instance - non GP fits
        
        Order of data cleaning:
            1) load in CBVs/quats
            2) handle custom masking of points
            3) bin 
            4) fractional cutoff applied to flux
                                                                
        
        ---------------------------------------------------
        Parameters:
            
            - fitType (int)
                1-6 are default runs
                0 are custom arguments (see below)
            
            - cutIndices (array of ints) true/false array of points ot trim
                default is NONE
                
            - binYesNo (bool, default NONE) bins to 8 hours if true
            
            - fraction (default NONE, 0-1 for percent) trims intensity to 
            percent of max. 0.4 is 40% of max, etc.
            
            - n1 (int def 1000) burn in first chain length
            - n2 (int def 10000) production chain length
            
            
            - thinParams, NONE to use defaults (1/4 first run discard, 15% trime) 
                            or [first run discard, percent thin]
                            
            - saveBIC (bool) do you want to save the BIC value that comes out
            
            - args, NONE for 1-6, use for 0 if you want it (see below)
            
            - logProbFunc, NONE for 1-6, use for 0
            
            - plotFit = NONE unless doing a custom fit - needs to match up with the 
                logprobfunc being used
                
            - filesavetag = NONE, custom string if you want it
            
            - labels, NONE unless doing custom bullshit, then array of str
            
            - init_values, NONE unless doing custom bullshit
        
        If you are doing custom priors:
            - don't
            - run under fitType = 0
            - last items in args must be your priors array -- all probability functions
                come with a positional argument priors = None that this should override
            - hopefully the tutorial level shows how to set this up right
    

        """

        # handle CBVs+quats
        if cbvquats and fitType in (2,4,5):
            (self.time, self.intensity, self.error,
             self.quatTime, self.quatsIntensity, self.CBV1, self.CBV2,
             self.CBV3) = ut.generate_clip_quats_cbvs(self.sector, self.time,
                                                      self.intensity,self.error, 
                                                      self.tmin, self.camera, self.ccd,
                                                      self.CBV_folder, 
                                                      self.quaternion_folder_txt)
                                                    
            self.quatsandcbvs = [self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3]
        elif not cbvquats and fitType in (2,4,5):
            raise ValueError("Cannot run the requested fit type, you need to provide quaternion locations via use_quaternions_cbvs()")
        else:
            self.quatsandcbvs = None # has to initiate as None or it'll freak
            
            
        ### THEN DO CUSTOM MASKING if both not already cut and indices are given
        if not hasattr(self, 'cutindexes') and cutIndices is not None:
            self.__custom_mask_it(cutIndices, saveplot = None)
        
        # 8hr binning
        if binYesNo and self.binned == False: #if need to bin
            (self.time, self.intensity, 
             self.error, self.lygosbg,
             self.quatsandcbvs) = ut.bin_8_hours(self.time, self.intensity, self.error, 
                                                 self.lygosbg, QCBVALL=self.quatsandcbvs)                                    
            self.binned = True # make sure you can't bin it more than once
                                                 
        # percent of max fitting
        if fraction is not None and self.fractiontrimmed==False:
            (self.time, self.intensity, self.error, self.lygosbg, 
             self.quatsandcbvs) = ut.fractionalfit(self.time, self.intensity, 
                                                   self.error, self.lygosbg, 
                                                   fraction, self.quatsandcbvs)
            self.fractiontrimmed=True #make sure you can't trim it more than once
        
        # this is to fix the quats and cbv inputs after trimming                       
        if fitType in (2,4,5):
            self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3 = self.quatsandcbvs 
            

        # load parameters by fit type                                           
        self.__setup_fittype_params(fitType, binYesNo, fraction, args,
                                    logProbFunc, plotFit, filesavetag, 
                                    labels, init_values)
        # set up the output folder
        self.__gen_output_folder() 
                                                        
        # run it
        (best, upperError, 
         lowerError, bic) = self.__mcmc_outer_structure(n1, n2, thinParams)
        
        if saveBIC:
            self.bic_all.append(bic)
            self.params_all.append(best)
            
        return best, upperError, lowerError, bic
      
    def __custom_mask_it(self, cutIndices, saveplot = None):
        """remove certain indices from your light curve.
        cutIndices should be an array of size len(time), 0 = remove, 1=keep
        
        *****this should NOT be used if using CBVs - that's not set up yet!!
        """
        if hasattr(self, 'cutindexes'): #if already did a trim
            print("*******")
            print("ALREADY TRIMMED - RELOAD AND TRY AGAIN")
            print("*******")
            return
        elif hasattr(self, 'time'): #if something loaded in and going to trim
            plt.scatter(self.time, self.intensity, color='red', s=2)
                
            self.cutindexes = np.nonzero(cutIndices) # which ones you are keeping
            self.time = self.time[self.cutindexes]
            self.intensity = self.intensity[self.cutindexes]
            self.error = self.error[self.cutindexes]
            if self.lygosbg is not None:
                self.lygosbg = self.lygosbg[self.cutindexes]
                
            if hasattr(self, 'quatsIntensity'): #if cbvs, trim them
                self.quatsIntensity = self.quatsIntensity[self.cutindexes]
                self.CBV1 = self.CBV1[self.cutindexes]
                self.CBV2 = self.CBV2[self.cutindexes]
                self.CBV3 = self.CBV3[self.cutindexes]
                self.quatsandcbvs = [self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3]
                
            #self.custommasked = True
            plt.scatter(self.time, self.intensity, color='blue', s=2)
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.show()
            if saveplot is not None:
                plt.savefig(saveplot)
            return
        else:
            print("No data loaded in yet!! Run again once light curve is loaded")
            return     
         
    def __gen_output_folder(self):
        """
        Internal function, sets up output folder for files
        
        """
        # check for an output folder's existence, if not, put it in. 
        if (self.folderSAVE is None or 
            self.targetlabel is None or 
            self.sector is None
            or self.camera is None or 
            self.ccd is None):
            raise ValueError("Cannot generate output folders, one of the parameters is None")
        
        newfolderpath = (self.folderSAVE + self.targetlabel + 
                         str(self.sector) + str(self.camera) + str(self.ccd))
        if not os.path.exists(newfolderpath):
            os.mkdir(newfolderpath)
        # make subfolder for this run
        print(self.filesavetag)
        subfolderpath = newfolderpath + "/" + self.filesavetag[1:]
        if not os.path.exists(subfolderpath):
            os.mkdir(subfolderpath)
        self.folderSAVE = subfolderpath + "/"
        self.parameterSaveFile = self.folderSAVE + "output_params.txt"
        return
    
    def __setup_fittype_params(self, fitType, binYesNo, fraction, args=None, 
                               logProbFunc = None, plotFit = None,
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
        
        IF YOU ARE DOING CUSTOM PRIORS:
            - run under fitType=0
            - last item in args must be your priors array -- all probability functions
            come with a positional argument priors=None that this should override
            
            
        """
        if fitType == 1: # single without
            self.args = (self.time, self.intensity, self.error, self.disctime)
            self.logProbFunc = mc.log_probability_singlepower_noCBV
            self.filesavetag = "-singlepower"
            self.labels = ["t0", "A", "beta",  "b"]
            self.init_values = np.array((self.disctime-3, 0.1, 1.8, 1))
            self.plotFit = fitType
            
        elif fitType == 2: # single with
            self.args = (self.time, self.intensity, self.error, 
                         self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3, 
                         self.disctime)
            self.logProbFunc = mc.log_probability_singlepower_withCBV
            self.filesavetag = "-singlepower-CBV"
            self.labels = ["t0", "A", "beta", "B", "cQ", "c1", "c2", "c3"]
            self.init_values = np.array((self.disctime-3, 0.1, 1.8, 0, 0,0,0,0))
            self.plotFit = fitType
        elif fitType == 3: # double without
            self.args = (self.time, self.intensity, self.error, self.disctime)
            self.logProbFunc = mc.log_probability_doublepower_noCBV
            self.filesavetag = "-doublepower"
            self.labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  "b"]
            self.init_values = np.array((self.disctime-8, self.disctime-4, 0.1, 0.1, 1.8, 1.8, 1))
            #print(self.init_values)
            self.plotFit = fitType
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
            self.plotFit = fitType
        elif fitType == 5: # just CBVs
            self.args = (self.time, self.intensity, self.error, 
                         self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3, 
                         self.disctime)
            self.logProbFunc = mc.log_probability_justCBV
            self.filesavetag = "-CBV"
            self.labels = ["b", "cQ", "c1", "c2", "c3"]
            self.init_values = np.array((1, 0,0,0,0))
            self.plotFit = fitType
        elif fitType == 6: # detrending lygos BG
            if self.lygosbg is None:
                print("NO LYGOS BG LOADED IN - CANNOT RUN THIS FIT")
                raise AttributeError("Missing lygosbg!!")
                return
            else:
                self.args = (self.time, self.intensity, self.error, self.lygosbg, 
                             self.disctime)
                self.logProbFunc = mc.log_probability_singlePower_LBG
                self.filesavetag = "-singlepower-lygosBG"
                self.labels = ["t0", "A", "beta",  "b", "LBG"]
                self.init_values = np.array((self.disctime-3, 0.1, 1.8, 1, 1))
                self.plotFit = fitType
        elif fitType == 0: # diy your stuff
            self.args = args # LAST ONE MUST BE PRIORS IF USING CUSTOMS
            self.logProbFunc = logProbFunc
            self.filesavetag = filesavetag
            self.labels = labels
            self.init_values = init_values 
            #print(self.args)
            self.plotFit = plotFit
        else:
            print("THAT IS NOT AN ALLOWED FIT TYPE, EXITING")
            raise ValueError("not an allowed fit type")
            
        if binYesNo: # it has to go in this order - need to load, then set args, then set this
            self.filesavetag = self.filesavetag + "-8HourBin"
    
        if fraction is not None:
            self.filesavetag = self.filesavetag + "-{fraction}".format(fraction=fraction)
        return
            
            

    def __mcmc_outer_structure(self, n1, n2, thinParams):
        """Fitting things that are NOT GP based
        Params:
            - n1 is an integer number of steps for the first chain
            - n2 is an integer number of steps for the second chain
            - thinParams is EITHER NONE (default thinning is used, 1/4 for the first run,
                                         15% thinning) or [int to discard, thinning percent]
        """
        
        print("***")
        print("***")
        print("***")
        print("***")
        print("Beginning MCMC run")
         
        timeModule.sleep(3) # this keeps things running orderly
        
        if thinParams is None:
            discard1 = int(n1/4)
            thinning = 15
        else:
            discard1, thinning = thinParams
                              

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
    
        flat_samples = sampler.get_chain(discard=discard1, flat=True, thin=thinning)
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
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01) # normally 0.01
            if converged:
                print("Converged, ending chain")
                break
            old_tau = tau
        
        # #plot autocorr things
        if self.plotFit != 10:
            sp.plot_autocorr_mean(self.folderSAVE, self.targetlabel, index, 
                                  autocorr, converged, 
                                  autoStep, self.filesavetag)
            
            sp.plot_autocorr_individual(self.folderSAVE, self.targetlabel, index,
                                        autocorr_all, autoStep, self.labels, 
                                        self.filesavetag)
        else:
            sp.plot_autocorr_mean(self.folderSAVE, self.targetlabel, index, 
                                  autocorr, converged, 
                                  autoStep, self.filesavetag)
            
            sp.plot_autocorr_individual(self.folderSAVE, self.targetlabel, index,
                                        autocorr_all, autoStep, self.filelabels, 
                                        self.filesavetag)
        
        #thin and burn out dump
        tau = sampler.get_autocorr_time(tol=0)
        if (np.max(tau) < (sampler.iteration/50)):
            burnin = int(2 * np.max(tau))
            thinning = int(0.5 * np.min(tau))
        else:
            burnin = int(n2/4)

        flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thinning)
        
        # this will be separate - plotting p(parameter)
        if self.plotFit != 10:
            sp.plot_chain_logpost(self.folderSAVE, self.targetlabel, self.filesavetag,
                                  sampler, self.labels, ndim, appendix = "-production")
        else:
            sp.plot_chain_logpost(self.folderSAVE, self.targetlabel, self.filesavetag,
                                  sampler, self.filelabels, ndim, appendix = "-production")
        
        sp.plot_paramTogether(flat_samples, self.labels, self.folderSAVE, 
                                  self.targetlabel, self.filesavetag)
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

        # ### BIC
        BIC = ndim * np.log(len(self.time)) - 2 * np.log(logprob)
        print("BAYESIAN INF CRIT: ", BIC)
        if np.isnan(np.float64(BIC[0])): # if it's a nan
            BIC = 50000
        else:
            BIC = BIC[0]
            
        if self.plotFit != 10:
            sp.plot_mcmc(self.folderSAVE, self.time, self.intensity, self.targetlabel, 
                         self.disctime, best_mcmc[0], 
                         flat_samples, self.labels, self.plotFit, self.filesavetag, 
                         self.tmin, self.lygosbg,
                         self.quatsandcbvs)
        else:
            sp.plot_mcmc_GP(self.folderSAVE, self.time, self.intensity, self.error, 
                            best_mcmc, self.gp, self.disctime, self.tmin,
                 self.targetlabel, self.filesavetag, plotComponents=False)
        
        with open(self.parameterSaveFile, 'w') as file:
            file.write(self.filesavetag + "-" + str(datetime.datetime.now()))
            file.write("\n {best} \n {upper} \n {lower} \n".format(best=best_mcmc,
                                                                   upper=upper_error,
                                                                   lower=lower_error))
            file.write("BIC:{bicy:.3f} Converged:{conv} \n".format(bicy=BIC, 
                                                                conv=converged))
        
        return best_mcmc, upper_error, lower_error, BIC
    
    
    def test_plot(self):
        """Quick little thing to spit out the current light curve that's loaded in """
        plt.errorbar(self.time, self.intensity, yerr=self.error, fmt='.', color='black')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.targetlabel)
        plt.show()
        plt.close()
        return
    
    def run_GP_fit(self, cutIndices, binYesNo, fraction=None, 
                   n1=1000, n2=10000, filesavetag=None,
                   customSigmaRho = None):
        """Run the GP fitting 
        
        customSigmaRho must unpack as: [sigma start, rho start, sigma lower, sigma upper,
                                        rho lower, rho upper, sigma frozen, rho frozen]
        the default run of this is [0.01, 1.2, 0.0001, 0.3, 1, 2, 0, 0]
        
        """
        if filesavetag is None:
            self.filesavetag = "-GP-matern32-fit"
        else:
            self.filesavetag = filesavetag
            
        ### THEN DO CUSTOM MASKING if both not already cut and indices are given
        if not hasattr(self, 'cutindexes') and cutIndices is not None:
            self.__custom_mask_it(cutIndices, saveplot = None)
            
        # check for 8hr bin BEFORE trimming to percentages
        if binYesNo: #if need to bin
            (self.time, self.intensity, 
             self.error, self.lygosbg,
             self.quatsandcbvs) = ut.bin_8_hours(self.time, self.intensity, self.error, 
                                                 self.lygosbg, QCBVALL=None) 
                                                 
        # if doing percent of max fitting
        if fraction is not None:
            (self.time, self.intensity, self.error, self.lygosbg, 
             self.quatsandcbvs) = ut.fractionalfit(self.time, self.intensity, 
                                                   self.error, self.lygosbg, 
                                                   fraction, self.quatsandcbvs)
        #make folders to save into
        self.__gen_output_folder()   


        #set up kernel
        #### SET UP NEW MATERN-32 GP
        if customSigmaRho is None:
            sigma = 0.01 #amplitude
            rho = 1.2 #timescale
            sigma_bounds = (0.0001,0.3)
            rho_bounds = (1,2)
            bounds_dict = dict(log_sigma=np.log(sigma_bounds), log_rho=np.log(rho_bounds))
            kernel = terms.Matern32Term(log_sigma=np.log(sigma), log_rho=np.log(rho), 
                                        bounds=bounds_dict)
            self.init_values = np.array((self.disctime-3, 0.1, 1.8, 0,np.log(sigma), np.log(rho)))
        else:
            sigma = customSigmaRho[0]
            rho = customSigmaRho[1]
            sigma_bounds = (customSigmaRho[2], customSigmaRho[3])
            rho_bounds = (customSigmaRho[4], customSigmaRho[5])
            bounds_dict = dict(log_sigma=np.log(sigma_bounds), log_rho=np.log(rho_bounds))
            kernel = terms.Matern32Term(log_sigma=np.log(sigma), log_rho=np.log(rho), 
                                        bounds=bounds_dict)
            self.init_values = np.array((self.disctime-3, 0.1, 1.8, 0))
            
            if customSigmaRho[6]: #if frozen (1)
                kernel.freeze_parameter("log sigma")
            else: # if not frozen (0)
                initsigma = np.array((np.log(sigma))) # if not frozen, add to init
                self.init_values = np.concatenate((self.init_values, initsigma))
                
            if customSigmaRho[7]: # if frozen true
                kernel.freeze_parameter("log rho")
            else:
                initrho = np.array((np.log(rho)))
                self.init_values = np.concatenate((self.init_values, initrho))
            
            
        self.gp = celerite.GP(kernel, mean=1.0)
        self.gp.compute(self.time, self.error)
        print("Initial log-likelihood: {0}".format(self.gp.log_likelihood(self.intensity)))
        
        
        # set up arguments etc.
        self.args = (self.time,self.intensity, self.error, self.disctime, self.gp)
        self.logProbFunc = mc.log_probability_GP
        self.labels = ["t0", "A", "beta",  "b", r"$log\sigma$",r"$log\rho$"] 
        self.filelabels = ["t0", "A", "beta",  "b",  "logsigma", "logrho"]
        
        fitType = 10
        self.plotFit = 10
        self.__mcmc_outer_structure(n1, n2)
        return
                    
