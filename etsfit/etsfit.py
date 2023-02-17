# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:14:13 2022

@author: lcgordon
"""

from etsfit.utils import utilities as ut
from etsfit.utils import snPlotting as sp
from etsfit.utils import MCMC as mc

import time as timeModule
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import emcee
from pylab import rcParams
rcParams['figure.figsize'] = 16, 6
rcParams["font.size"] = 20

from celerite.modeling import Model
import celerite
from celerite import terms

import warnings
warnings.filterwarnings("ignore")

from tinygp import kernels, GaussianProcess
import jax
import jax.numpy as jnp
import jaxopt
jax.config.update("jax_enable_x64", True)


class etsMAIN(object):
    """Make one of these for the light curve you're going to fit!"""
    
    def __init__(self, save_dir, TNSFile, plot=True):
        """
        Initialize an etsfit object
        
        This sets up a save path, loads in a csv of targets if you want,
        and sets default values of the data not being trimmed, binned, and
        there being no CBVs/quaternions accessible.
        
        -----------------------------------------------
        Parameters:
            
            - save_dir (str) the path to where you want everything to be saved
            should end with a /, it will self-correct if not
            
            - TNSFile (str) the path to a big pandas-readable TNS CSV file
            containing all the targets you could possibly care about
        
        """
        
        if not os.path.exists(save_dir): #if the folder that the subfolders are going into
        #is not real
            raise ValueError("Outermost folder to save into must be a real path!")
        else:
            if save_dir[-1] != "/": #if not ending with /, 
                save_dir += "/"
            self.save_dir = save_dir
            self.save_dir_perm = save_dir #keep a copy in case
        
        
        if os.path.exists(TNSFile):
            self.info = pd.read_csv(TNSFile)
        self.TNSFile = TNSFile
        
        
        self.cbvquat_access = False #cannot currently access cbvs/quaternions
        self.fractiontrimmed = False #not trimmed
        self.binned = False #not binned
        self.quats_cbvs = None
        self.using_GP = False
        self.cleaningdone = False
        self.plot = plot
        return
    
    def reset(self):
        """
        Removes all settings in the object that may have been overwritten
        by an MCMC run. 
        ---------------------------------------------------
        """
        print("WARNING")
        print("reset() will wipe out anything you have loaded into the object! ")
        self.save_dir = self.save_dir_perm
        self.cbvquat_access = False #cannot currently access cbvs/quaternions
        self.fractiontrimmed = False #not trimmed
        self.binned = False #not binned
        self.quats_cbvs = None
        self.using_GP = False
        self.cleaningdone = False
        if hasattr(self, 'time'):
            del self.time
            del self.flux
            del self.error
            del self.BGdata
            del self.disctime
            del self.targetlabel
            del self.sector
            del self.camera
            del self.ccd
            del self.tmin
            del self.bic_all 
            del self.params_all 
            del self.xlabel 
            del self.ylabel
        if hasattr(self, 'fitType'):
            del self.fitType 
            del self.plotFit
            del self.args
            del self.logProbFunc 
            del self.filesavetag 
            del self.labels 
            del self.filelabels
            del self.init_values 
        if hasattr(self, 'flux_mask'):  
            del self.flux_mask 
        if hasattr(self, 'fract'):
            del self.fract
        if hasattr(self, 'parameterSaveFile'):
            del self.parameterSaveFile
        if hasattr(self, 'filesavetag'):
            del self.filesavetag
        if hasattr(self, 'sampler'):
            del self.sampler 
            del self.best_mcmc
        
        return
    
    
    def use_quaternions_cbvs(self, cbv_dir, quaternion_raw_dir, 
                             quaternion_txt_dir):
        """ 
        Function to set up access to CBVs and quaternions. 
        If you do not have text file versions of the quaternions, they will
        be generated for you.
        
        ----------------------------------------------------
        Parameters: 
            - cbv_dir (str) a path to the folder holding all CBVs. 
            this is probably within the eleanor directory
            the directory structure should be such that there are files like
            folder/s001/cbv_components_s0001_0001_0001.txt
            - quaternion_raw_dir (str) is the path to the folder holding
            all of the .fits file versions of the quaternions. this can be
            some random path if you already have produced the txt versions
            - quaternion_txt_dir (str) is the path to the folder either
            HOLDING the .txt file versions of the quaternions OR the path to
            the EMPTY folder that all the txt versions are about 
            to be generated into
        """
        
        if (not os.path.exists(cbv_dir) or not os.path.exists(quaternion_raw_dir)
            or not os.path.exists(quaternion_txt_dir)):
            raise ValueError("At least one or your paths are bad!")
        else:
            self.cbv_dir = cbv_dir
            self.quaternion_raw_dir = quaternion_raw_dir
            self.quaternion_txt_dir = quaternion_txt_dir
            #empty directory, make txt files
            if (len(os.listdir(self.quaternion_txt_dir))==0 and
                len(os.listdir(self.quaternion_raw_dir))!=0):
                
                ut.make_quat_txtfiles(self.quaternion_raw_dir, self.quaternion_txt_dir)
        self.cbvquat_access = True
        return

    
    def window_rms_filt(self, innerfilt = None, outerfilt = None):
        """ 
        Runs an RMS filter over the light curve and returns an array of 
        0's (bad) and 1's (good) that can be used in the custom masking
        argument of other functions. 
        
        This DOES NOT save the filter output inside the object. 
        
        Defaults the inner window as len(self.time)*0.0q and the outer as
        inner*10. 
        
        -------------------------------
        Parameters:
            
            - innerfilt = None by default, can set to an int for the inner
            window of compariosn
            - outerfilt = None by default, can set to an int for the outer
            window of comparison
            - plot (bool) defaults as True, plots light curve w/ mask
        
        """
        return ut.window_rms(self.time, self.flux, innerfilt = innerfilt, 
                        outerfilt = outerfilt, plot=self.plot)
    
    def load_single_lc(self, time, flux, error, discoverytime, 
                       targetlabel, sector, camera, ccd, BGdata=None):
        """ 
        Load in one light curve from information you supply
        
        ----------------------------------
        Parameters:
            - time (array) time axis for the lc. this will get the 0th
            index subtracted off. 
            - flux (array) flux array for the lc 
            - error (array) error on flux array
            - discoverytime (double) when ground telescopes found it
            - targetlabel (str) no spaces name, will be used on files
            - sector (str) this needs to be a two-character string (ie, sector
                                                                    2 is "02")
            - camera (str) needs to be a 1 char string
            - ccd (str) needs to be a 1 char string
            - BGdata (defaults to NONE) can put in background array if 
            doing that one specific fit. should also work to float other 
            background arrays.
        """
        
        if (len(time) != len(flux) or len(flux) != len(error) or
            len(time) != len(error)):
            print("Time length:", len(time))
            print("flux length:", len(flux))
            print("Error length:", len(error))
            raise ValueError("Mismatched sizes on time, flux, and error!")
        elif (time is None or flux is None or 
              error is None or discoverytime is None or 
              targetlabel is None or sector is None or 
              camera is None or ccd is None):
            raise ValueError("Inputs all have to be SOMETHING you can't give any None's here")
        elif (type(sector) != str or type(camera) != str or type(ccd) != str) :
            raise ValueError("Sector, camera, ccd must be strings, see help()")
        
        self.time = time
        self.flux = flux
        self.error = error
        self.BGdata = BGdata
        self.disctime = discoverytime
        self.targetlabel = targetlabel
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.tmin = time[0]
        self.time -= self.tmin
        self.disctime -= self.tmin
        self.bic_all = []
        self.params_all = []
        self.xlabel = "Time [BJD - 2457000]"
        self.ylabel = "Flux (e-/s)"
        
        return
    
    def test_plot(self):
        """
        Quick little thing to spit out the current light curve that's loaded in 
        """
        if not hasattr(self, 'time'):
            return ValueError("No light curve loaded in to plot!")
        plt.errorbar(self.time, self.flux, yerr=self.error, fmt='.', color='black')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.targetlabel)
        if hasattr(self, 'disctime'):
            plt.axvline(self.disctime)
        plt.show()
        plt.close()
        return

    
    def pre_run_clean(self, fitType, flux_mask=None, binning = False, 
                      fraction = None):
        """
        A function to be run before running run_MCMC()
        This function MUST be run before running run_MCMC() or it will
        give u an error, even if you have no cleaning to be done. 
        
        ---------------------------------
        Parameters:
            
            - fitType (int)
                1-6 are default runs
                0 are custom arguments (see below)
            
            - flux_mask (array of ints) true/false array of points ot trim
                default is NONE
                
            - binning (bool, default NONE) bins to 8 hours if true
            
            - fraction (default NONE, 0-1 for percent) trims flux to 
            percent of max. 0.4 is 40% of max, etc.
        """
        
        self.fitType = fitType
        self.flux_mask = flux_mask
        
        # handle CBVs+quats
        if self.cbvquat_access and fitType in (2,4,5):
            print("Loading in quaternions and CBVs")
            (self.time, self.flux, self.error,
             self.quatTime, self.quaternions, 
             self.CBV1, self.CBV2, 
             self.CBV3) = ut.generate_clip_quats_cbvs(self.sector, 
                                                      self.time,
                                                      self.flux,
                                                      self.error, 
                                                      self.tmin, 
                                                      self.camera, 
                                                      self.ccd,
                                                      self.cbv_dir, 
                                                      self.quaternion_txt_dir)
            self.quats_cbvs = [self.quaternions, self.CBV1, self.CBV2, self.CBV3]
        elif not self.cbvquat_access and fitType in (2,4,5):
            print("You need to provide quaternion paths via use_quaternions_cbvs()")
            raise ValueError("Cannot run the requested fit type")
 
        ### THEN DO CUSTOM MASKING if both not already cut and indices are given
        if flux_mask is not None:
            self.__custom_mask_it()
            
        # 8hr binning
        if binning: #if need to bin
            self.__8hrbinning()
                                                 
        # percent of max fitting
        if fraction is not None and not self.fractiontrimmed:
            #fractional fit code (fraction can be None)                                  
            self.__fract_fit(fraction)
            
        # this is to fix the quats and cbv inputs after trimming                       
        if fitType in (2,4,5):
            self.quaternions, self.CBV1, self.CBV2, self.CBV3 = self.quats_cbvs
         
        #once  you are done with this stuff:
        self.cleaningdone = True 
        return
    
    def __custom_mask_it(self):
        """
        remove certain indices from your light curve.
        flux_mask should be an array of size len(time), 0 = remove, 1=keep
        
        """
        if hasattr(self, 'masked'): #if already did a trim
            print("ALREADY TRIMMED - RELOAD AND TRY AGAIN")
            return
        elif hasattr(self, 'time') and hasattr(self, 'flux_mask'): #if something loaded in and going to trim
            ut.data_masking(self)
            self.masked = True
            return
        else:
            raise ValueError("No data loaded or no flux mask given")
            return   
        
    def __fract_fit(self, fraction):
        """ 
        Internal function to do a fractional fit: 
        """
        if fraction is not None:
            (self.time, self.flux, self.error, self.BGdata, 
             self.quats_cbvs) = ut.fractionalfit(self.time, self.flux, 
                                                   self.error, self.BGdata, 
                                                   fraction, self.quats_cbvs)
        self.fractiontrimmed=True #make sure you can't trim it more than once
        self.fract = fraction                                        
        return
    
    def __8hrbinning(self):
        """ 
        Internal function to do 8 hour binning
        """
        if self.binned: 
            print("Data already binned! ")
            return
        else:     
            (self.time, self.flux, 
             self.error, self.BGdata,
             self.quats_cbvs) = ut.bin_8_hours(self.time, self.flux, self.error, 
                                                 self.BGdata, QCBVALL=None) 
            self.binned = True                                    
            return
   
    def __gen_output_folder(self, filesavetag):
        """
        Internal function, sets up output folder for files
        
        """
        ut.gen_output_folder(self, filesavetag)
        return 
    
    def __fileparamsave(self):
        """ 
        Internal function to save parameters into txt file 
        """
        ut.param_save(self)
        return
   
    
    def __filetag_update(self, usebounds=False):
        if self.binned: # it has to go in this order - need to load, then set args, then set this
            self.filesavetag = self.filesavetag + "-8HourBin"
    
        if self.fractiontrimmed and self.fract is not None:
            self.filesavetag = self.filesavetag + "-{fraction}".format(fraction=self.fract)
            
        if usebounds and self.using_GP:
            self.filesavetag = self.filesavetag + "-bounded"
        
        if usebounds is False and self.using_GP:
            self.filesavetag = self.filesavetag + "-noBounds"
        return
    
 
    def __setup_fittype_params(self, args=None, 
                               logProbFunc = None, plotFit = None,
                               filesavetag=None, labels=None, init_values=None,
                               mu = 2, sigma = 1):
        """
        
        Internal function to set up self.plotFit, logProbFunc, filesavetag, labels
        filelabels, init_values array, args array
        
        1 = single without
        2 = single with cbv
        3 = doubel without
        4 = double with
        5 = just cbv
        6 = single lygos bg
        7 = gaussian prior on beta
        0 = custom inputs
        any other number = will exit with an error
        
        IF YOU ARE DOING CUSTOM PRIORS:
            - run under fitType=0
            - last item in args must be your priors array -- all probability functions
            come with a positional argument priors=None that this should override
            
            
        """
        self.plotFit = self.fitType
        start_t = min(self.disctime-3, self.time[-1]-2)
        
        if self.fitType == 1: # single without
            self.args = (self.time, self.flux, self.error)
            self.logProbFunc = mc.log_probability_singlepower_noCBV
            self.filesavetag = "-singlepower"
            self.labels = [r"$t_0$", "A", r"$\beta$",  "b"]
            self.filelabels = ["t0", "A", "beta",  "b"]
            
            self.init_values = np.array((start_t, 0.1, 1.8, 1))
            
        elif self.fitType == 2: # single with
            self.args = (self.time, self.flux, self.error, 
                         self.quaternions, self.CBV1, self.CBV2, self.CBV3)
            self.logProbFunc = mc.log_probability_singlepower_withCBV
            self.filesavetag = "-singlepower-CBV"
            self.labels = ["t0", "A", "beta", "B", "cQ", "c1", "c2", "c3"]
            self.filelabels = self.labels
            self.init_values = np.array((start_t, 0.1, 1.8, 0, 0,0,0,0))
            
        elif self.fitType == 3: # double without
            self.args = (self.time, self.flux, self.error, self.disctime)
            self.logProbFunc = mc.log_probability_doublepower_noCBV
            self.filesavetag = "-doublepower"
            self.labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  "b"]
            self.filelabels = self.labels
            self.init_values = np.array((start_t, start_t+1, 0.1, 0.1, 1.8, 1.8, 1))

        elif self.fitType ==4: # double with
            self.args = (self.time, self.flux, self.error, 
                         self.quaternions, self.CBV1, self.CBV2, self.CBV3, 
                         self.disctime)
            self.logProbFunc = mc.log_probability_doublepower_withCBV
            self.filesavetag = "-doublepower-CBV"
            self.labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  
                      "cQ", "c1", "c2", "c3"]
            self.filelabels = self.labels
            self.init_values = np.array((start_t, start_t+1, 0.1, 0.1, 
                                    1.8, 1.8, 0,0,0,0))
        elif self.fitType == 5: # just CBVs
            self.args = (self.time, self.flux, self.error, 
                         self.quaternions, self.CBV1, self.CBV2, self.CBV3)
            self.logProbFunc = mc.log_probability_justCBV
            self.filesavetag = "-CBV"
            self.labels = ["b", "cQ", "c1", "c2", "c3"]
            self.filelabels = self.labels
            self.init_values = np.array((1, 0,0,0,0))
        elif self.fitType == 6: # detrending lygos BG
            if self.BGdata is None:
                raise AttributeError("NO BG LOADED IN - CANNOT RUN THIS FIT")
                return
            else:
                self.args = (self.time, self.flux, self.error, self.BGdata)
                self.logProbFunc = mc.log_probability_singlePower_BG
                self.filesavetag = "-singlepower-BGdata"
                self.labels = ["t0", "A", "beta",  "b", "LBG"]
                self.filelabels = self.labels
                self.init_values = np.array((start_t, 0.1, 1.8, 1, 1))
         
        elif self.fitType == 7: # gaussian beta
            self.args = (self.time, self.flux, self.error, mu, sigma)
            self.logProbFunc = mc.log_probability_singlepower_gaussianbeta
            self.filesavetag = "-singlepower-GBeta"
            self.labels = ["t0", "A", "beta",  "b"]
            self.filelabels = self.labels
            self.init_values = np.array((start_t, 0.1, 1.8, 1))
        
        elif self.fitType == 0: # diy your stuff
            self.args = args # LAST ONE MUST BE PRIORS IF USING CUSTOMS
            self.logProbFunc = logProbFunc
            self.filesavetag = filesavetag
            self.labels = labels
            self.filelabels = self.labels
            self.init_values = init_values 
            self.plotFit = plotFit #overwrites the default fitType that was set to plotfit
        else:
            print("THAT IS NOT AN ALLOWED DEFAULT FIT TYPE, EXITING")
            raise ValueError("not an allowed fit type")
            
        if args is not None: #just overwrite it at the end if custom
            self.args = args
            
        if filesavetag is not None:
            self.filesavetag = filesavetag
          
        self.__filetag_update(usebounds=False)    
          
        return
   
    
    def run_MCMC(self, n1=1000, n2=10000, thinParams = None,
                 saveBIC=False, args=None, logProbFunc = None, plotFit = None,
                 filesavetag=None,
                 labels=None, init_values=None, mu=2, sigma=1, local_dir=False,
                 quiet=False):
        """
        Run one MCMC instance - non GP fits
        
        Order of data cleaning:
            1) load in CBVs/quats
            2) handle custom masking of points
            3) bin 
            4) fractional cutoff applied to flux                             
        ---------------------------------------------------
        Parameters:
            
            - n1 (int def 1000) burn in first chain length
            - n2 (int def 10000) production chain length
            - thinParams, NONE to use defaults (1/4 first run discard, 15% trime) 
                            or [first run discard, percent thin]  
            - saveBIC (bool) do you want to save the BIC value that comes out
            - args, NONE for 1-6, use for 0 if you want it (see below)
            - logProbFunc, NONE for 1-6, use for 0
            - plotFit = NONE unless doing a custom fit
            - filesavetag = NONE, custom string if you want it
            - labels, NONE unless doing custom bullshit, then array of str
            - init_values, NONE unless doing custom bullshit
            - local_dir (FALSE) - saves into the local folderSave location, rather 
                than producing a new folder
        
        If you are doing custom priors:
            - run under fitType = 0
            - last items in args must be your priors array -- all probability functions
                come with a positional argument priors = None that this should override
        """

        if not self.cleaningdone:
            raise Exception("You must run self.pre_run_clean() first!")
        # load parameters by fit type                                           
        self.__setup_fittype_params(args,logProbFunc, plotFit, filesavetag, 
                                    labels, init_values, mu, sigma)
        if not local_dir:
            # set up the output folder
            self.__gen_output_folder(self.filesavetag) 
                                                        
        # run it
        self.__mcmc_inner_structure(n1, n2, thinParams, quiet=quiet)
        
        if saveBIC:
            self.bic_all.append(self.BIC[0])
            self.params_all.append(self.best_mcmc)
            
        return
      
        
    def __mcmc_inner_structure(self, n1, n2, thinParams, quiet=False):
        """
        Fitting things that are NOT GP based
        Params:
            - n1 is an integer number of steps for the first chain
            - n2 is an integer number of steps for the second chain
            - thinParams is EITHER NONE (default thinning is used, 1/4 for the first run,
                                         15% thinning) or [int to discard, thinning percent]
        """
        if not quiet:
            print(" *** \n *** \n *** \n ***")
            print("Beginning MCMC run")
         
        timeModule.sleep(3) # this keeps things running orderly
        
        if thinParams is None:
            discard_ = int(n1/4)
            thinning = 15
        else:
            discard_, thinning = thinParams
                              

        # ### MCMC setup
        np.random.seed(42)
        
        nwalkers = 100
        self.ndim = len(self.labels) # labels are provided when you run it
        p0 = (np.ones((nwalkers, self.ndim)) * self.init_values + 
              np.random.uniform(0, 0.1, (nwalkers, self.ndim)))
        
        if not quiet:
            print("Starting burnin chain")
        # ### Initial run
        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, 
                                        self.logProbFunc,args=self.args) # setup
        self.sampler.run_mcmc(p0, n1, progress=True) # run it
        
        #plot burn in chain
        if self.plotFit < 10 and self.plot:
            sp.plot_chain_logpost(self, appendix = "burnin")
        elif self.plotFit == 10 and self.plot:
            sp.plot_chain(self, appendix = "burnin")
            
    
        self.flat_samples = self.sampler.get_chain(discard=discard_, flat=True, thin=thinning)
        
        # get intermediate best
        best_mcmc_inter = np.zeros((1,self.ndim))
        for i in range(self.ndim):
            mcmc = np.percentile(self.flat_samples[:, i], [16, 50, 84])
            best_mcmc_inter[0][i] = mcmc[1]
          
        for i in best_mcmc_inter[0]:
            if i > 50000:
                print("Something has gone terribly wrong")
                print(best_mcmc_inter[0])
                return
            
            
        # ### Main run
        p0 = (np.ones((nwalkers, self.ndim)) * best_mcmc_inter[0] + 
              np.random.uniform(0, 0.1, (nwalkers, self.ndim)))
           
        self.sampler.reset()
        
        # ### CORRELATION FUNCTION 
        self.index = 0 # number of checks
        self.autocorr = np.empty(n2) # total possible checks
        old_tau = np.inf
        self.autoStep = 100 # how often to check
        self.autocorr_all = np.empty((int(n2/self.autoStep) + 2,len(self.labels))) # save all autocorr times
        
        # sample up to n2 steps
        for sample in self.sampler.sample(p0, iterations=n2, progress=True):
            # Only check convergence every 100 steps
            if self.sampler.iteration % self.autoStep:
                continue
            # Compute the autocorrelation time so far
            tau = self.sampler.get_autocorr_time(tol=0) # tol=0 always get estimate
            if np.any(tau == np.nan) or np.any(tau == np.inf) or np.any(tau == -np.inf):
                
                print("autocorr is nan or inf")
                print(tau)
            self.autocorr[self.index] = np.mean(tau) # save mean autocorr time
            self.autocorr_all[self.index] = tau # save all autocorr times for plotting
            self.index += 1 # how many times have you saved it
        
            # Check convergence
            self.converged = np.all((tau * 100) < self.sampler.iteration)
            self.converged &= np.all((np.abs(old_tau - tau) / tau) < 0.01) # normally 0.01
            if self.converged:
                if not quiet:
                    print("Converged, ending chain")
                break
            old_tau = tau
        
        # ######
        #plot autocorr things
        ########
        if self.plot:
            sp.plot_autocorr_all(self)
        
            
        #thin and burn out dump
        tau = self.sampler.get_autocorr_time(tol=0)
        if (np.max(tau) < (self.sampler.iteration/50)):
            burnin = int(2 * np.max(tau))
            thinning = int(0.5 * np.min(tau))
        else:
            burnin = int(n2/4)

        self.flat_samples = self.sampler.get_chain(discard=burnin, flat=True, thin=thinning)
        if self.plot:
            sp.plot_param_samples_all(self)
        if not quiet:
            print(len(self.flat_samples), "samples post second run")
    
        # ### BEST FIT PARAMS
        self.best_mcmc = np.zeros((1,self.ndim))
        self.upper_error = np.zeros((1,self.ndim))
        self.lower_error = np.zeros((1,self.ndim))
        for i in range(self.ndim):
            mcmc = np.percentile(self.flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            if not quiet:
                print(self.labels[i], mcmc[1], -1 * q[0], q[1] )
            self.best_mcmc[0][i] = mcmc[1]
            self.upper_error[0][i] = q[1]
            self.lower_error[0][i] = q[0]
     

        logprob, blob = self.sampler.compute_log_prob(self.best_mcmc)

        # ### BIC
        #print(np.log(len(self.time)))
        
        self.BIC = (self.ndim * np.log(len(self.time)) - 2 * (logprob * -1.0))[0]
        if not quiet:
            print("log prob:", logprob)
            print("BAYESIAN INF CRIT: ", self.BIC)
        if np.isnan(np.float64(self.BIC)): # if it's a nan
            self.BIC = 500000
             
        
        if self.plotFit < 10 and self.plot:
            sp.plot_chain_logpost(self, appendix = "production")
            sp.plot_mcmc(self)
        elif self.plotFit == 10 and self.plot:
            sp.plot_chain(self, appendix = "production")
            if 'mean' in self.gpUSE:
                # Plot the data.
                sp.plot_mcmc_GP_celerite_mean(self)
                sp.celerite_post_pred(self)
            else:
                sp.plot_mcmc_GP_celerite_residual(self)
        
        #save output parameters
        self.__fileparamsave()
        
        return 

    
    def run_GP_fit(self, n1=1000, n2=10000, gpUSE = "expsqr",
                          thinParams=None, usebounds = True, 
                          custom_bounds=None, quiet=False):
        """
        Run the GP fitting 
        allowed runs: ['celerite_mean', 'celerite_residual', 'expsqr', 'expsinsqr', 'matern32']
        -------------------------
        Params:
            - n1 (int) burn in steps
            - n2 (int) production steps
            - gpUSE (str) ['celerite_mean', 'celerite_residual', 'expsqr', 'expsinsqr', 'matern32']
            - thinParams (arr) either None or [int to discard, thinning percent]
            - usebounds (bool) t/f using tight bounds
            - custom_bounds (dict/none) custom bounds dict, containing entries for:
                - log_sigma
                - log_rho
                - boundlabel (string)
            - quiet = bool yes/no printing
        
        """
        self.using_GP = True
        if not self.cleaningdone:
            raise ValueError("Need to run pre_run_clean() first!")
            return
        
        
        allowed = ['celerite_mean', 'celerite_residual', 'expsqr', 'expsinsqr', 'matern32']
        self.gpUSE = gpUSE
        if self.gpUSE not in allowed:
            return ValueError("Not a valid gpUSE input!")
        
        if 'celerite_mean' in self.gpUSE:
            self.__run_GP_fit_celerite_mean(n1, n2, thinParams, bounds=usebounds,
                                            custom_bounds=custom_bounds, quiet=quiet)
            
        elif 'celerite_residual' in self.gpUSE:
            self.__run_GP_fit_celerite_residual(n1, n2, thinParams, bounds=usebounds,
                                                custom_bounds=custom_bounds, quiet=quiet)
            
        else: #tinygp options
           
            #set up gpUSE settings
            self.__tinygp_setup(gpUSE=gpUSE, bounds=usebounds)                                          
            #make folders to save into
            
            self.__gen_output_folder(self.filesavetag)   
            #print("entering mcmc + gp concurrent fitting")
            self.__mcmc_concurrent_gp(n1, n2, thinParams, quiet)
        return
   
    def __tinygp_setup(self, gpUSE='expsqr', usebounds=True):
        """ 
        Internal function to set up the tinygp run
        """
        
        self.plotFit = 1
        self.fitType = 1
        self.args = (self.time, self.flux, self.error)
        self.logProbFunc = mc.log_probability_singlepower_noCBV
        self.labels = ["t0", "A", "beta",  "b"]
        self.filelabels = self.labels
        start_t = min(self.time[-1]-2, self.disctime-3)
        self.init_values = np.array((start_t, 0.1, 1.8, 1))
        
        if gpUSE == 'expsqr':
            self.filesavetag = "-tinygp-expsqr"
            self.theta = {
                "log_sigma": np.log(2),
                "log_rho": np.log(1),
            }
            self.build_gp = self.__build_tinygp_expsqr #no quotes on it
            self.update_theta = self.__update_theta_ampsscale
            if usebounds: #sigma, rho
                self.tinygp_bounds = np.asarray([[np.log(1.1), np.log(1)], 
                                             [np.log(21.2), np.log(3)]])
            else: 
                self.tinygp_bounds = None
            
        elif gpUSE == 'matern32':
            self.filesavetag = "-tinygp-matern32"
            self.theta = {
                "log_sigma": np.log(2),
                "log_rho": np.log(1),
            }
            self.build_gp = self.__build_tinygp_matern32 #no quotes on it
            self.update_theta = self.__update_theta_ampsscale
            if usebounds: 
                self.tinygp_bounds = np.asarray([[np.log(1.1), np.log(1)], 
                                             [np.log(21.2), np.log(3)]])
            else: 
                self.tinygp_bounds = None
            
        elif gpUSE == 'expsinsqr':
            self.filesavetag = "-tinygp-expsinsqr"
            self.theta = {
                "log_sigma": np.log(2),
                "log_rho": np.log(1),
                "log_gamma": np.log(1),
            }
            self.build_gp = self.__build_tinygp_expsinsqr #no quotes on it
            self.update_theta = self.__update_theta_ampsscalegamma
            if usebounds: #sigma, rho
                self.tinygp_bounds = np.asarray([[np.log(1.1), np.log(1), np.log(1)], 
                                             [np.log(21.2), np.log(3), np.log(3)]])
            else: 
                self.tinygp_bounds = None
        
        self.__filetag_update(usebounds)
            
        return
    

    
    def __run_GP_fit_celerite_residual(self, n1=1000, n2=10000, 
                                       thinParams=None, usebounds=True, 
                                       custom_bounds=None, quiet=True):
        """
        Run the GP fitting w/ celerite fitting to the residual
        -------------------------
        Params:
            - n1 (int) burn in steps
            - n2 (int) production steps
            - thinParams (arr) either None or [int to discard, thinning percent]
            - usebounds (bool) t/f using tight bounds
            - custom_bounds (dict/none) custom bounds dict, containing entries for:
                - log_sigma
                - log_rho
                - boundlabel (string)
            - quiet bool print
        
        """
        self.filesavetag = "-celerite-matern32-residual"

        self.__filetag_update(usebounds)
        
        
        
        #set up kernel
        start_t = min(self.disctime-3, self.time[-1]-2)
        # set up gp:
        rho = 2 # init value
        sigma = 1
        if usebounds and custom_bounds is None:
           
            sigma_bounds = np.log( np.sqrt((0.1,20  )) ) #sigma range 0.316 to 4.47, take log
            rho_bounds = np.log((1, 10)) #0, 2.302
            bounds_dict = dict(log_sigma=sigma_bounds, log_rho=rho_bounds)
            
        elif usebounds and custom_bounds is not None: 
            bounds_dict = custom_bounds.copy()
            if not quiet:
                print(bounds_dict)
            self.filesavetag = self.filesavetag + bounds_dict["boundlabel"]
            bounds_dict.pop('boundlabel')
            
        else: #functionally unbounded
            
            sigma_bounds = np.log( np.sqrt((1e-10,1e4  )) ) #sigma range 0.316 to 4.47, take log
            rho_bounds = np.log((1e-10, 1e4)) #0, 2.302
            bounds_dict = dict(log_sigma=sigma_bounds, log_rho=rho_bounds)
            
        kernel = terms.Matern32Term(log_sigma=np.log(sigma), log_rho=np.log(rho), 
                                    bounds=bounds_dict)
        
        #make folders to save into
        self.__gen_output_folder(self.filesavetag)  
        
        self.init_values = np.array((start_t, 0.1, 1.8, 0,np.log(sigma), np.log(rho)))
        self.gp = celerite.GP(kernel, mean=0.0)
        self.gp.compute(self.time, self.error)
        if not quiet:
            print("Initial log-likelihood: {0}".format(self.gp.log_likelihood(self.flux)))
        # set up arguments etc.
        self.args = (self.time,self.flux, self.error, self.gp)
        self.logProbFunc = mc.log_probability_celerite_residual
        self.labels = ["t0", "A", "beta",  "b", r"$log\sigma$",r"$log\rho$"] 
        self.filelabels = ["t0", "A", "beta",  "b",  "logsigma", "logrho"]
        self.plotFit = 10
        
        #run it
        self.__mcmc_inner_structure(n1, n2, thinParams, quiet=quiet)
        return
    
    def __run_GP_fit_celerite_mean(self, n1=1000, n2=10000, 
                                   thinParams=None, usebounds=False, 
                                   custom_bounds = None, quiet=False):
        """
        Run the GP fitting w/ celerite and a mean model 
        -------------------------
        Params:
            - n1 (int) burn in steps
            - n2 (int) production steps
            - thinParams (arr) either None or [int to discard, thinning percent]
            - usebounds (bool) t/f using tight bounds
            - custom_bounds (dict/none) custom bounds dict, containing entries for:
                - log_sigma
                - log_rho
                - boundlabel (string)
        """
        
        self.filesavetag = "-celerite-matern32-mean-model"
        self.__filetag_update(usebounds)
        #make folders to save into
        self.__gen_output_folder(self.filesavetag)  
        
        from celerite.modeling import Model
        from scipy.optimize import minimize
        import celerite
        from celerite import terms
        
        class MeanModel(Model):
            parameter_names = ("t0", "A", "beta", "b")

            def get_value(self, t):
                t1 = t-self.t0
                mod = np.heaviside((t1), 1) * self.A *np.nan_to_num((t1**self.beta), copy=False)
                return mod + self.b

            
            def compute_gradient(self, t):
                t1 = t-self.t0
                dt = np.heaviside((t1), 1) * -self.A * self.t0 * (t1)**(self.beta-1)
                dt[np.isnan(dt)] = 0
                dA = np.heaviside((t1), 1) * t1**self.beta
                dA[np.isnan(dA)] = 0
                dbeta = np.heaviside((t1), 1) * self.A * np.log(t1)*(t1)**self.beta
                dbeta[np.isnan(dbeta)] = 0
                dB = np.heaviside((t1), 1) * np.ones((len(t),)) #np.heaviside((t1), 1) * 
                return np.array([dt, dA, dbeta, dB])
        
        #set up power law model
        bounds_model_dict = {"t0":(0, self.time[-1]),
                             "A": (0.001, 20),
                             "beta":(0.5,6.0),
                             "b":(-50, 50)}
        
        start_t = min(self.disctime-3, self.time[-1]-2)
        t0, A, beta, b = (start_t, 5, 1.8, 5)
        
        mean_model = MeanModel(t0=t0, A=A, beta=beta, b=b,
                               bounds = bounds_model_dict)

        # Set up the GP model
        #presently unbounded
        if usebounds and custom_bounds is None:
            bounds_dict = {'log_sigma':np.log(np.sqrt((0.1,20  ))), 
                       'log_rho':np.log((1,10))}
            kernel = terms.Matern32Term(log_rho=np.log(2), log_sigma=np.log(1),
                                        bounds=bounds_dict)
        elif usebounds and custom_bounds is not None:
            bounds_dict = custom_bounds.copy()
            self.filesavetag = self.filesavetag + bounds_dict["boundlabel"]
            bounds_dict.pop('boundlabel')
            kernel = terms.Matern32Term(log_rho=np.log(2), log_sigma=np.log(1),
                                        bounds=bounds_dict)
        else:
            kernel = terms.Matern32Term(log_rho=np.log(2), log_sigma=np.log(1))
            
        self.gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
        self.gp.compute(self.time, self.error)
        if not quiet:
            print("Initial log-likelihood: {0}".format(self.gp.log_likelihood(self.flux)))

        # Define a cost function
        def neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.log_likelihood(y)

        def grad_neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.grad_log_likelihood(y)[1]

        # Fit for the maximum likelihood parameters
        initial_params = self.gp.get_parameter_vector()
        bounds = self.gp.get_parameter_bounds()
        if not quiet:
            print("Running scipy maximizer")
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                        method="L-BFGS-B", bounds=bounds, args=(self.flux, self.gp))

        self.gp.set_parameter_vector(soln.x)
        # Make the maximum likelihood prediction
        self.t = np.linspace(0, self.time[-1], 500)
        self.mu, var = self.gp.predict(self.flux, self.t, return_var=True)
        self.std = np.sqrt(var)

        # Plot the data + scipy prediction
        if self.plot:
            sp.plot_scipy_max(self)

        self.init_values = soln.x
        self.args = (self.flux, self.gp)
        self.logProbFunc = mc.log_probability_celerite_mean
        self.labels = [r"$log\sigma$",r"$log\rho$", "t0", "A", "beta",  "b"] 
        self.filelabels = ["logsigma", "logrho", "t0", "A", "beta",  "b"]
        self.plotFit = 10
    
        #emcee section
        #run it
        self.__mcmc_inner_structure(n1, n2, thinParams, quiet=quiet)
           
        return
    
    def __build_tinygp_matern32(self, theta, X):
        """
        Make the matern3-2 kernel 
        log_sigma is defined the SAME as log sigma in celerite
        jnp required here for data type reasons
        """
        k1 = jnp.exp(theta["log_sigma"]*2) * kernels.Matern32(jnp.exp(theta["log_rho"]))
        return GaussianProcess(k1, X, mean=0.0, diag=self.error)
    
    def __build_tinygp_expsinsqr(self, theta, X):
        """Make the expssinqr kernel """
        k1 = jnp.exp(theta["log_sigma"]) * kernels.ExpSineSquared(jnp.exp(theta["log_rho"]),
                                                           gamma = jnp.exp(theta["log_gamma"]))
        return GaussianProcess(k1, X, mean=0.0, diag=self.error)
    
    def __build_tinygp_expsqr(self, theta, X):
        """Make the expsqr kernel """
        k1 = jnp.exp(theta["log_sigma"]) * kernels.ExpSquared(jnp.exp(theta["log_rho"]))
        return GaussianProcess(k1, X, mean=0.0, diag=self.error)
    
    def __update_theta_ampsscale(self, solnparams):
        self.theta["log_sigma"] = solnparams["log_sigma"]
        self.theta["log_rho"] = solnparams["log_rho"]
        return
    
    def __update_theta_ampsscalegamma(self, solnparams):
        self.theta["log_sigma"] = solnparams["log_sigma"]
        self.theta["log_rho"] = solnparams["log_rho"]
        self.theta['log_gamma'] = solnparams['log_gamma']
        return
    
    
    def __mcmc_concurrent_gp(self, n1, n2, thinParams, quiet=False):
        """Fitting things that are NOT GP based
        Params:
            - n1 is an integer number of steps for the first chain
            - n2 is an integer number of steps for the second chain
            - thinParams is EITHER NONE (default thinning is used, 1/4 for the first run,
                                         15% thinning) or [int to discard, thinning percent]
        """
        def make_residual(x, y, best_mcmc):
            t0, A,beta,B = best_mcmc
            t1 = x - t0
            sl = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + B
            return y - sl
        
        def neg_log_likelihood(theta, X, y):
            gp = self.build_gp(theta, X)
            return -gp.log_probability(y)
        
        if not quiet:
            print(" *** \n *** \n *** \n ***")
            print("Beginning MCMC + GP run")
         
        timeModule.sleep(3) # this keeps things running orderly
        
        if thinParams is None:
            discard_ = int(n1/4)
            thinning = 15
        else:
            discard_, thinning = thinParams
                              

        # ### MCMC setup
        np.random.seed(42)
        
        nwalkers = 100
        self.ndim = len(self.labels) # labels are provided when you run it
        p0 = (np.ones((nwalkers, self.ndim)) * self.init_values + 
              np.random.uniform(0, 0.1, (nwalkers, self.ndim)))
        
        if not quiet:
            print("Starting burn-in chain")
        # ### Initial run
        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.logProbFunc,args=self.args) # setup
        self.sampler.run_mcmc(p0, n1, progress=True) # run it
        
        #plot burn in chain
        if self.plot:
            sp.plot_chain_logpost(self, appendix = "burnin")
    
        self.flat_samples = self.sampler.get_chain(discard=discard_, flat=True, thin=thinning)
        
        # get intermediate best
        best_mcmc_inter = np.zeros((1,self.ndim))
        for i in range(self.ndim):
            best_mcmc_inter[0][i] = np.percentile(self.flat_samples[:, i], [16, 50, 84])[1]
            
        # ### Main run
        p0 = (np.ones((nwalkers, self.ndim)) * best_mcmc_inter[0] + 
              np.random.uniform(0, 0.1, (nwalkers, self.ndim)))
           
        self.sampler.reset()
        
        # ### CORRELATION FUNCTION/GP FIT
        self.index = 0 # number of checks
        self.autocorr = np.empty(n2) # total possible checks
        old_tau = np.inf
        self.autoStep = 100 # how often to check
        self.autocorr_all = np.empty((int(n2/self.autoStep) + 2,len(self.labels))) # save all autocorr times
        
        # GP setup
        #calculate residual from intermediate best:
        res = make_residual(self.time, self.flux, best_mcmc_inter[0])
        #print("created residual")
        
        obj = jax.jit(jax.value_and_grad(neg_log_likelihood))
        if not quiet:
            print(f"Initial negative log likelihood: {obj(self.theta, self.time, res)[0]}")

        solver = jaxopt.ScipyBoundedMinimize(fun=neg_log_likelihood)
        soln = solver.run(self.theta, self.tinygp_bounds, X=self.time, y=res)
        if not quiet:
            print(f"Final negative log likelihood: {soln.state.fun_val}")
        #set theta to new values
        self.update_theta(soln.params)
        self.GP_LL_all = [soln.state.fun_val]
        
        # sample up to n2 steps
        for sample in self.sampler.sample(p0, iterations=n2, progress=True):
            
            #refit the GP every 1000 steps
            if self.sampler.iteration % 1000 == 0:
                #new residual:
                self.flat_samples = self.sampler.get_chain(flat=True)
                best_mcmc_inter = np.zeros((1,self.ndim))
                for i in range(self.ndim):
                    best_mcmc_inter[0][i] = np.percentile(self.flat_samples[:, i], [16, 50, 84])[1]
                    
                res = make_residual(self.time, self.flux, best_mcmc_inter[0])
                #solver = jaxopt.ScipyMinimize(fun=neg_log_likelihood)
                soln = solver.run(self.theta, self.tinygp_bounds, X=self.time, y=res)
                self.GP_LL_all.append(soln.state.fun_val)
                self.update_theta(soln.params)
            
            # Only check convergence every 100 steps
            if self.sampler.iteration % self.autoStep:
                continue
            # Compute the autocorrelation time so far
            tau = self.sampler.get_autocorr_time(tol=0) # tol=0 always get estimate
            if np.any(tau == np.nan) or np.any(tau == np.inf) or np.any(tau == -np.inf):
                print("autocorr is nan or inf")
                print(tau)
            self.autocorr[self.index] = np.mean(tau) # save mean autocorr time
            self.autocorr_all[self.index] = tau # save all autocorr times for plotting
            self.index += 1 # how many times have you saved it
        
            # Check convergence
            self.converged = np.all((tau * 100) < self.sampler.iteration)
            self.converged &= np.all((np.abs(old_tau - tau) / tau) < 0.01) # normally 0.01
            if self.converged and not quiet:
                print("Converged, ending chain")
                break
            old_tau = tau
        
        
        #save output gp params:
        self.update_theta(soln.params)
        #plot gp log likelihood over steps
        if self.plot:
            sp.plot_tinygp_ll(self)
        
            #plot autocorrelation times 
            sp.plot_autocorr_all(self)
            
        #thin and burn out dump
        tau = self.sampler.get_autocorr_time(tol=0)
        if (np.max(tau) < (self.sampler.iteration/50)):
            burnin = int(2 * np.max(tau))
            thinning = int(0.5 * np.min(tau))
        else:
            burnin = int(n2/4)

        self.flat_samples = self.sampler.get_chain(discard=burnin, flat=True, thin=thinning)
        
        # plot chains, parameters
        if self.plot:
            sp.plot_chain_logpost(self, appendix = "production")
            
            sp.plot_param_samples_all(self)
        if not quiet:
            print(len(self.flat_samples), "samples post second run")
    
        # ### BEST FIT PARAMS
        self.best_mcmc = np.zeros((1,self.ndim))
        self.upper_error = np.zeros((1,self.ndim))
        self.lower_error = np.zeros((1,self.ndim))
        for i in range(self.ndim):
            mcmc = np.percentile(self.flat_samples[:, i], [16, 50, 84])
            if not quiet:
                print(self.labels[i], mcmc[1], -1 * np.diff(mcmc)[0], np.diff(mcmc)[1] )
            self.best_mcmc[0][i] = mcmc[1]
            self.upper_error[0][i] = np.diff(mcmc)[1]
            self.lower_error[0][i] = np.diff(mcmc)[0]
     
        logprob, blob = self.sampler.compute_log_prob(self.best_mcmc)

        # ### BIC
        #so have logprob from the self.sampler (just model)
        # want to add the log prob from the tinygp model
        
        negll = -1.0 * float(self.GP_LL_all[-1]) #neg ll
        logprob = -1.0 * (logprob+negll)
        
        if not quiet:
            print("negative log prob, no GP: ", logprob) #this spits out a negative value
            print("negative log like,  GP: ", negll)
        
        self.BIC = (self.ndim * np.log(len(self.time)) - 2 * logprob)[0]
        if not quiet:
            print("BAYESIAN INF CRIT: ", self.BIC)
        if np.isnan(np.float64(self.BIC)): # if it's a nan
            self.BIC = 500000

         
        self.gp = self.build_gp(self.theta, self.time)
        if self.plot:
            sp.plot_mcmc_GP_tinygp(self)

        self.__fileparamsave()
        
        return self.best_mcmc, self.upper_error, self.lower_error, self.BIC
    

   