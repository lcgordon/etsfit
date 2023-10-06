# -*- coding: utf-8 -*-
"""
Created on Mon Sept 11 2023

@author: lcgordon
"""

from etsfit.utils import utilities as ut
from etsfit.utils import default_plots as sp
from etsfit.utils import MCMC as mc
from etsfit.utils import gp_plots as spgp

import time as timeModule
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import emcee
from astropy.time import Time
from pylab import rcParams
rcParams['figure.figsize'] = 16, 6
rcParams["font.size"] = 20

from celerite.modeling import Model
import celerite
from celerite import terms
from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")

from tinygp import kernels, GaussianProcess
import jax
import jax.numpy as jnp
import jaxopt
jax.config.update("jax_enable_x64", True)

class etsfit(object): 
    """ 
    The etsfit object that loads in data and runs modeling.
    
    :param save_dir: directory to save everything into, should end in / but will self correct if not
    :type save_dir: str 
    :param plot: if you want output plots
    :type plot: bool
    """
    def __init__(self, save_dir, plot=True): 
        """
        constructor
        """
        # handle folder: 
        if not os.path.exists(save_dir):
            raise ValueError("Outermost folder to save into must be a real path!")
        else:
            if save_dir[-1] != "/": save_dir += "/"
            self.save_dir = save_dir
            self.save_dir_perm = save_dir #keep a copy in case

        
        self.cbvquat_access = False # cannot currently access cbvs/quaternions
        self.quats_cbvs_array = None # empty 
        self.using_GP = False # not using GP
        self.preprocess_done = False # no cleaning applied
        self.plot = plot # plotting on/off
        
        print("etsfit object initialized")
        return
    
    def load_single_lc(self, time, flux, error, targetlabel,  **kwargs):
        """ 
        Load in one light curve from information you supply
        
        ----------------------------------
        Parameters:
            - time (array) time axis for the lc. this will get the 0th
                index subtracted off. 
            - flux (array) flux array for the lc 
            - error (array) error on flux array
            - targetlabel (str) no spaces name, will be used on files
            KWARGS OPTIONS:
            - discovery_time (double/None) when ground telescopes found it. 
                same units as time array. can be None otherwise.
            - sector (int) TESS sector of data, can set to 0 if DNE
            - camera (int) TESS camera of data, can set to 0 if DNE
            - ccd (int) TESS ccd of data, can set to 0 if DNE
            - background (array, defaults to NONE) if fitting with a background (annulus?) array
                can provide this information here
            - xlabel (str) custom xlabel
            - ylabel (str) custom ylabel
            - time_unit (str) astropy-readable time unit, default is JD
        """
        # validate
        if (len(time) != len(flux) or len(flux) != len(error) or
            len(time) != len(error)):
            print("Time length:", len(time))
            print("Flux length:", len(flux))
            print("Error length:", len(error))
            raise ValueError("Mismatched sizes on time, flux, and error!")
        elif (time is None or flux is None or 
              error is None  or targetlabel is None):
            raise ValueError("Inputs all have to be SOMETHING you can't give any None's here")

        
        self.time = time
        self.flux = flux
        self.error = error
        self.targetlabel = targetlabel
        self.tmin = time[0]
        self.time_unit = kwargs.get("time_unit", "jd") 
        # tessreduce loader converts into JD - important if using binning only

        # load kwargs
        self.background = kwargs.get("background", None)
        self.xlabel = kwargs.get("xlabel", "Time [BJD - 2457000]")
        self.ylabel = kwargs.get("ylabel", "Flux (e-/s)") #get(key, default)
        self.sector = kwargs.get('sector', 0)
        self.camera = kwargs.get('camera', 0)
        self.ccd = kwargs.get('ccd', 0)
        
        # if discovery time is given, use it
        # if not, set to end of dataset - it has to be set to SOMETHING
        self.discovery_time = kwargs.get("discovery_time", time[-1])
        
        # reset the time axis
        self.time -= self.tmin 
        self.discovery_time -= self.tmin
        
        # handle file
        self.targetfile = kwargs.get("targetfile", None)
        if self.targetfile is not None and os.path.isfile(self.targetfile): #file, real
            self.info = pd.read_csv(self.targetfile)
        elif self.targetfile is not None and not os.path.isfile(self.targetfile): #file, not real
            raise ValueError("That is not a valid path to a CSV file!")

        print("Data load successful")
        return
    
    def quaternion_cbv_access(self, cbv_dir, quaternion_raw_dir, 
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
            raise ValueError("At least one of your dirs is bad!")
        else:
            self.cbv_dir = cbv_dir
            self.quaternion_raw_dir = quaternion_raw_dir
            self.quaternion_txt_dir = quaternion_txt_dir
            #empty directory, make txt files
            if (len(os.listdir(self.quaternion_txt_dir))==0 and
                len(os.listdir(self.quaternion_raw_dir))!=0):
                # make text file versions for all .fits files in the directory
                ut.all_quat_textfiles(self.quaternion_raw_dir, self.quaternion_txt_dir)
        
        #can access and use them now        
        self.cbvquat_access = True 
        return
    
    def test_plot(self):
        """
        Quick little fxn to spit out the current light curve that's loaded in 
        """
        if not hasattr(self, 'time'):
            return ValueError("No light curve loaded in to plot!")
        fig, ax = plt.subplots(1, figsize=(6, 2))
        plt.scatter(self.time, self.flux, color='k', s=2)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.targetlabel)
        if self.discovery_time is not None:
            plt.axvline(self.discovery_time)
        plt.show()
        plt.close()
        return
    
    def data_preprocess(self, **kwargs):
        """ 
        This function preprocesses the loaded data given various keywords. 
        Even if no cleaning is to be done, this has to be run before running 
        MCMC or you'll kick up an error. 
        
        Kwargs:
            - qcbv_fit (array of strings) if using a fit type that requires use of 
                the cbvs/quats. args get passed to quaternion_cbv_access(), 
                so the array in order is [cbv_dir, quaternion_raw_dir, 
                                          quaternion_txt_dir]
            - flux_mask (bool) true/false run the masking
            - binning_data (bool) currently TURNED OFF, sorry
            - fraction (float) float on 0-1 that trims flux to that percent 
                of the max
        
        """
        if self.preprocess_done:
            print("Preprocessing already done on this data - reload and try again?")
            return 
        
        self.qcbv_fit = kwargs.get("qcbv_fit", None)
        self.use_flux_mask = kwargs.get("use_flux_mask", False)
        self.fraction_trim = kwargs.get("fraction_trim", None)
        self.binning_data = kwargs.get("binning_data", False)
        
        self.preprocessing_tag = ""
        
        if self.qcbv_fit is not None: #if you do need to load them
            if self.sector is None or self.camera is None or self.ccd is None: 
                raise ValueError("Can only use quaternions/cbvs if  \
                                 sector/cam/ccd info was provided on loading!")
        
            # run the access fxn
            self.quaternion_cbv_access(self.qcbv_fit[0], self.qcbv_fit[1], 
                                      self.qcbv_fit[2])
            # run the loader
            print("Loading quaternions + CBVs")
            ut.generate_align_quats_cbvs(self, **kwargs)
            self.quats_cbvs = [self.Qall, self.CBV1, self.CBV2, self.CBV3]

        if self.use_flux_mask is True:
            # produce the mask
            print("Producing window RMS data mask")
            innerfilt = kwargs.get("innerfilt", None)
            outerfilt = kwargs.get("outerfilt", None)
            self.mask = ut.window_rms(self.time, self.flux, 
                                      innerfilt = innerfilt, 
                                      outerfilt = outerfilt, plot=self.plot)
            # apply the mask:
            ut.data_masking(self)
        
        if self.binning_data:
            print("Binning! ")
            def_time = Time("1999-01-01T08:00:00", format='isot') \
                - Time("1999-01-01T00:00:00", format='isot')
            self.binning_dt = kwargs.get("binning_dt", def_time.jd)
            ut.time_binning(self, self.binning_dt, self.time_unit)
            self.preprocessing_tag += "_binned"
                  
        if self.fraction_trim is not None:
            print("Trimming to fraction of peak flux")
            ut.fractional_trim(self)
            self.preprocessing_tag += f"_frac{int(self.fraction_trim*100)}"
            
        self.preprocess_done = True
        return

    
    def mcmc_setup(self, **kwargs): 
        """
        Final initialization work for MCMC - setting the fit type, initial
        conditions, folder setup, etc. 
        
        Kwargs: 
            - default_fit_type (int 1-8) initializes one of the default options
            - custom_fit (true) initializes a custom set of inputs
            - custom_filesavetag (str) should start with a hyphen or underscore!!
        """
        self.default_fit = kwargs.get("default_fit", False)
        self.custom_fit = kwargs.get("custom_fit", False)
        
        self.custom_filesavetag = kwargs.get("custom_filesavetag", None)
        
        # set up fit params:
        if self.default_fit > self.custom_fit: #if using a default set
            self.use_fit = self.default_fit
            self.__default_fit_setup()
            
        elif self.custom_fit > self.default_fit: # if using a custom set
            self.use_fit = 0 
            self.__custom_fit_setup(**kwargs)
        else:
            raise ValueError("You must either use a default fit type or \
                             provide the args for custom fitting!")  
        
        # generate output folder:
        if self.sector is not None: 
            internal = f"{self.targetlabel}{self.sector:02}{self.camera}{self.ccd}"
            newfolder = f"{self.save_dir}{internal}/"
        else:
            internal = self.targetlabel
            newfolder = f"{self.save_dir}{self.targetlabel}/"
        if not os.path.exists(newfolder): os.mkdir(newfolder) #make parent
        
        self.save_dir = f"{newfolder}{self.filesavetag[1:]}/"
        if not os.path.exists(self.save_dir): os.mkdir(self.save_dir) #make subfolder
        print(f"New save directory: {self.save_dir}")
        # gen parameter saving file:
        self.parameter_save_file = f"{self.save_dir}{internal}{self.filesavetag}-output_params.txt"
        return
    
    def __default_fit_setup(self):
        """
        
        Internal function to set up initial conditions for stuff
        
        1 = single 
        2 = single with cbv
        3 = double 
        4 = double with
        5 = just cbv
        6 = single + amplitude modelling a background array
        7 = gaussian prior on beta
        8 = a literal flat line y=B
  
        """
        self.plotFit = self.use_fit
        start_t = min(self.discovery_time-3, self.time[-1]-2)
        # this sets the plotting function to be called in mcmc
        self.MCMC_PLOTTER = sp.plot_mcmc
        
        if self.use_fit == 1: # single without
            self.logProbArgs = (self.time, self.flux, self.error)
            self.logProbFunc = mc.log_probability_singlepower_noCBV
            self.filesavetag = "-singlepower"
            self.labels = [r"$t_0$", "A", r"$\beta$",  "b"]
            self.filelabels = ["t0", "A", "beta",  "b"]
            self.init_values = np.array((start_t, 0.1, 1.8, 1))
            
        elif self.use_fit == 2: # single with
            self.logProbArgs = (self.time, self.flux, self.error, 
                          self.Qall, self.CBV1, self.CBV2, self.CBV3)
            self.logProbFunc = mc.log_probability_singlepower_withCBV
            self.filesavetag = "-singlepower-CBV"
            self.labels = [r"$t_0$", "A", r"$\beta$", "B", "cQ", "c1", "c2", "c3"]
            self.filelabels = ["t0", "A", "beta", "B", "cQ", "c1", "c2", "c3"]
            self.init_values = np.array((start_t, 0.1, 1.8, 0, 0,0,0,0))
            
        elif self.use_fit == 3: # double without
            self.logProbArgs = (self.time, self.flux, self.error, self.discovery_time)
            self.logProbFunc = mc.log_probability_doublepower_noCBV
            self.filesavetag = "-doublepower"
            self.labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  "b"]
            self.filelabels = self.labels
            self.init_values = np.array((start_t, start_t+1, 0.1, 0.1, 1.8, 1.8, 1))

        elif self.use_fit ==4: # double with
            self.logProbArgs = (self.time, self.flux, self.error, 
                          self.Qall, self.CBV1, self.CBV2, self.CBV3, 
                          self.discovery_time)
            self.logProbFunc = mc.log_probability_doublepower_withCBV
            self.filesavetag = "-doublepower-CBV"
            self.labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  
                      "cQ", "c1", "c2", "c3"]
            self.filelabels = self.labels
            self.init_values = np.array((start_t, start_t+1, 0.1, 0.1, 
                                    1.8, 1.8, 0,0,0,0))
        elif self.use_fit == 5: # just CBVs
            self.logProbArgs = (self.time, self.flux, self.error, 
                          self.Qall, self.CBV1, self.CBV2, self.CBV3)
            self.logProbFunc = mc.log_probability_justCBV
            self.filesavetag = "-CBV"
            self.labels = ["b", "cQ", "c1", "c2", "c3"]
            self.filelabels = self.labels
            self.init_values = np.array((1, 0,0,0,0))
        elif self.use_fit == 6: # detrending annulus BG
            if self.background is None:
                raise AttributeError("NO BACKGROUND LOADED IN - CANNOT RUN THIS FIT")
                return
            else:
                self.logProbArgs = (self.time, self.flux, self.error, self.background)
                self.logProbFunc = mc.log_probability_singlePower_BG
                self.filesavetag = "-singlepower-background"
                self.labels = ["t0", "A", "beta",  "b", "BG"]
                self.filelabels = self.labels
                self.init_values = np.array((start_t, 0.1, 1.8, 1, 1))
         
        elif self.use_fit == 7: # gaussian beta
            self.logProbArgs = (self.time, self.flux, self.error, 2, 1)
            self.logProbFunc = mc.log_probability_singlepower_gaussianbeta
            self.filesavetag = "-singlepower-GBeta"
            self.labels = ["t0", "A", "beta",  "b"]
            self.filelabels = self.labels
            self.init_values = np.array((start_t, 0.1, 1.8, 1))
        
        elif self.use_fit == 8: #FLAT
            self.logProbArgs = (self.time, self.flux, self.error)
            self.logProbFunc = mc.log_probability_flat
            self.filesavetag = "-flat"
            self.labels = ["B"]
            self.filelabels = ["B"]
            self.init_values = np.array((1))
        else:
            print("THAT IS NOT AN ALLOWED DEFAULT FIT TYPE, EXITING")
            raise ValueError("not an allowed fit type")
            
        if self.custom_filesavetag is not None: #overwrite save tag
            self.filesavetag = self.custom_filesavetag
         
        # update filetag if preprocessing requires it
        self.filesavetag += self.preprocessing_tag
          
        return
    
    def __custom_fit_setup(self, **kwargs): 
        """ 
        Initialize a custom fit from kwargs
        Required for this to work: 
            - args
            - logProbFunc
            - filesavetag
            - labels
            - filelabels (can be same as labels)
            - init_values
            - MCMC_PLOTTER - function to call to plot the output
                    fit with the data, equiv to sp.plot_mcmc
        """
        self.logProbArgs = kwargs.pop("args")
        self.logProbFunc = kwargs.pop("logProbFunc")
        self.filesavetag = kwargs.pop("filesavetag")
        self.labels = kwargs.pop("labels")
        self.filelabels = kwargs.get("filelabels", self.labels)
        self.init_values = kwargs.pop("init_values")
        self.MCMC_PLOTTER = kwargs.pop("MCMC_PLOTTER")
        
        return
    
    def run_mcmc(self, n1=5_000, n2=10_000, **kwargs):
        """ 
        Actually execute the mcmc run! 
        parameters:
            - n1 (int) first chain
            - n2 (int) second chain
            
        kwargs: 
            - quiet (bool) turn print statements off (def: false)
            - nwalkers (int) how many walkers (def: 100)
            - discard (int) default 0 (initial chain discard)
            - thinning (int) default 1 (uses every sample, a setting of 15 would
                                        use every 15th, etc.)
            - seed (int) random seed initializer (def: 42)
            - nwalkers (int) default: 100
        """
        quiet = kwargs.get("quiet", False)
        
        if not quiet:
            print(" *** \n *** \n *** \n ***")
            print("Beginning MCMC run")
            
        timeModule.sleep(3) # this keeps things running orderly
        
        discard_samples = kwargs.get("discard", 0)
        thinning = kwargs.get("thinning", 1)
            
        # init
        np.random.seed(kwargs.get("seed", 42)) 
        nwalkers = kwargs.get("nwalkers", 100)
        self.ndim = len(self.labels) #labels should've already been set up
        pos_0 = np.full((nwalkers, self.ndim), self.init_values) + \
                np.random.uniform(0, 0.1, (nwalkers, self.ndim))
        #         p0 = (np.ones((nwalkers, self.ndim)) * self.init_values + 
        #               np.random.uniform(0, 0.1, (nwalkers, self.ndim)))
        
        #### First chain
        if not quiet: print("Starting burn-in chain...")
        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, 
                                             self.logProbFunc, 
                                             args=self.logProbArgs)
        self.sampler.run_mcmc(pos_0, n1, progress=True)
        
        # plot the burn in chain (if plotting): 
        if self.plot: sp.plot_chain_withlogpost(self, appendix="burnin")
        
        #get intermediate best to initialize next chain: 
        self.flat_samples = self.sampler.get_chain(discard=discard_samples, 
                                                   flat=True, thin=thinning)
        best_inter = np.zeros(self.ndim)
        for i in range(self.ndim):
            best_inter[i] = np.percentile(self.flat_samples[:, i], [16, 50, 84])[1]
            
        #### Main Chain
        if not quiet: print("\nStarting production chain...")
        pos_1 = np.full((nwalkers, self.ndim), best_inter) + \
                np.random.uniform(0, 0.1, (nwalkers, self.ndim))
        self.sampler.reset()
        
        # Correlation function
        self.index = 0
        self.stepsize = 1000 # how often to check
        old_tau = np.inf # prev autocorr time 
        # empty array to save all checked autocorrs into - one mean, one all
        self.autocorr_means = np.empty(int(np.ceil(n2/self.stepsize))+1)
        self.autocorr_all = np.empty((int(np.ceil(n2/self.stepsize))+1, self.ndim))
        
        # run sampler
        for sample in self.sampler.sample(pos_1, iterations=n2, progress=True):
            # only check conv. every stepsize: 
            if self.sampler.iteration % self.stepsize: 
                continue
            # compute autocorr time
            tau = self.sampler.get_autocorr_time(tol=0) # tol=0 always get estimate
            if np.any(tau == np.nan) or np.any(tau == np.inf) \
                or np.any(tau == -np.inf):
                print("autocorr is nan or inf")
                print(tau)
            self.autocorr_means[self.index] = np.mean(tau) #mean
            self.autocorr_all[self.index] = tau #per parameter
            self.index += 1
            
            # check convergence
            self.converged = np.all((tau * 100) < self.sampler.iteration)
            self.converged &= np.all((np.abs(old_tau - tau) / tau) < 0.01) # normally 0.01
            if self.converged:
                if not quiet: print("Converged, ending chain")
                break
            old_tau = tau # update old tau
        
        # plot autocorr once out of loop:
        if self.plot: sp.plot_autocorr_all(self)
        
        # thin and burn out
        tau = self.sampler.get_autocorr_time(tol=0)
        if (np.max(tau) < (self.sampler.iteration/100)):
            burnin = int(2 * np.max(tau))
        else:
            burnin = int(n2/10)
        thinning = 1
         
        self.flat_samples = self.sampler.get_chain(discard=burnin, flat=True, thin=thinning)
        if self.plot: sp.plot_param_samples_all(self)
        if not quiet: print(f"{len(self.flat_samples)} samples post second run")
        
        #### Best Fit Params:
        self.best_mcmc = np.empty(self.ndim)
        self.upper_error = np.empty(self.ndim)
        self.lower_error = np.empty(self.ndim)
        for i in range(self.ndim):
            mcmc = np.percentile(self.flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            if not quiet: print(rf"{self.labels[i]}", mcmc[1], -1 * q[0], q[1] )
            self.best_mcmc[i] = mcmc[1]
            self.upper_error[i] = q[1]
            self.lower_error[i] = q[0]
            
        # Apply GR corrections
        self.__calc_gelmanrubin()
        self.upper_error = np.sqrt(self.upper_error**2 * self.GR)
        self.lower_error = np.sqrt(self.lower_error**2 * self.GR)
        print("Applied Gelman-Rubin Corrections")
        
        #### Estimate BIC
        logprob, blob = self.sampler.compute_log_prob(self.best_mcmc.reshape(1,self.ndim))
        self.BIC = (self.ndim * np.log(len(self.time)) - 2 * (logprob * -1.0))[0]
        if not quiet: print(f"BIC: {self.BIC:.2f}; Log Prob: {logprob[0]:.2f}")
        if np.isnan(np.float64(self.BIC)): self.BIC = np.inf
        
        # Final plotting: 
        if self.plot: 
            # normal
            if not hasattr(self, 'self.GP_type'): 
                sp.plot_chain_withlogpost(self, appendix="production")
                self.MCMC_PLOTTER(self) #sp.plot_mcmc(self)
            # celerite mean plotting
            elif hasattr(self, 'self.GP_type') and self.GP_type == 'celerite_mean':
                spgp.plot_mcmc_GP_celerite_mean(self)
                spgp.celerite_post_pred(self)
            # celerite residual plotting
            elif hasattr(self, 'self.GP_type') and self.GP_type == 'celerite_residual':
                spgp.plot_mcmc_GP_celerite_residual(self)
        
        # Save Output Parameters
        ut.param_save(self)
        print("MCMC Run Completed!")
        return


    def __calc_gelmanrubin(self):
        """ 
        Internal use (only?) function to calculate the gelman rubin statistic
        """

        tau = self.sampler.get_autocorr_time(tol=0)
        if np.any(np.isnan(tau)) or np.any(np.isinf(tau)):
            burnin = 1
            print(f"Autocorr time is a nan or inf - rerun {self.targetlabel} light curve! ")
            print("Setting burnin to 1")
        else:    
            burnin = int(2 * np.max(tau))
        samples = self.sampler.get_chain(discard=burnin, flat=False)
        
        N = samples.shape[0]
        M = samples.shape[1] #nchains
        variances = np.var(samples, axis=0) #shape (100, 4)
        chain_means = np.mean(samples, axis=0) #shape 100 x 4
        total_means = np.mean(chain_means, axis=0) #shape 4
        
        W = (1/M) * np.sum(variances, axis=0) #shape 4
        B = (N / (M-1)) * np.sum((chain_means-total_means), axis=0)
        v_ = ( 1 - (1/N))* W + (B/N)
        self.GR = np.sqrt(v_/W)
        return
    
    def mcmc_gp_setup(self, **kwargs): 
        """ 
        Sets up the parameters for a GP run. 
        This should be executed *after* preprocessing has been run
        --------------------
        kwargs:
            - default_GP or custom_GP (bool) one of these needs to be turned on
            - GP_type from allowed list: 
                allowed = ['celerite_mean', 'celerite_residual', 
                           'expsqr', 'expsinsqr', 'matern32']
            - custom_filesavetag (str) to change the default file tag
            - quiet (bool)
        """
        if not self.preprocess_done:
            raise ValueError("Need to run data_preprocess() first!")
            return 
        
        self.allowed = ['celerite_mean', 'celerite_residual', 
                   'expsqr', 'expsinsqr', 'matern32']
        self.default_GP = kwargs.get("default_GP", False)
        self.custom_GP = kwargs.get("custom_GP", False)
        self.custom_filesavetag = kwargs.get("custom_filesavetag", None)
        self.quiet = kwargs.get("quiet", False)
        
        if self.default_GP > self.custom_GP: # using a default setup
            self.GP_type = kwargs.get("GP_type", "celerite_mean")
            if self.GP_type not in self.allowed:
                print(self.allowed)
                return ValueError("Please provide a correct default type^^^")
            # if good:
            print(f"Using a default GP setup {self.GP_type}")
            # then split out into the setup fxn
            self.__default_gp_setup(**kwargs)
            
        elif self.custom_GP > self.default_GP: # using a custom setup
            print("Setting up custom GP run...")
            self.__custom_gp_setup(**kwargs)
        else: 
            raise ValueError("Have to activate ONE of default or custom!")
            return
        
        # generate output folder:
        if self.sector is not None: 
            internal = f"{self.targetlabel}{self.sector:02}{self.camera}{self.ccd}"
            newfolder = f"{self.save_dir}{internal}/"
        else:
            internal = self.targetlabel
            newfolder = f"{self.save_dir}{self.targetlabel}/"
        if not os.path.exists(newfolder): os.mkdir(newfolder) #make parent
        
        self.save_dir = f"{newfolder}{self.filesavetag[1:]}/"
        if not os.path.exists(self.save_dir): os.mkdir(self.save_dir) #make subfolder
        print(f"New save directory: {self.save_dir}")
        # gen parameter saving file:
        self.parameter_save_file = f"{self.save_dir}{internal}{self.filesavetag}-output_params.txt"
        
        self.using_GP = True
        return
    
    def __custom_gp_setup(self, **kwargs):
        """ 
        Internal function to initialize custom gp runs
        """
        self.GP_type = "custom"
        # gp bounds
        self.use_gp_bounds = kwargs.get("use_gp_bounds", False)
        self.gp_bounds = kwargs.get("gp_bounds", None)
        
        self.MCMC_PLOTTER = kwargs.pop("MCMC_PLOTTER")
        self.logProbArgs = kwargs.pop("logProbArgs")
        self.logProbFunc = kwargs.pop("logProbFunc")
        self.labels = kwargs.pop("labels")
        self.filelabels = kwargs.pop("filelabels")
        self.init_values = kwargs.pop("init_values")
        self.filesavetag = kwargs.pop("filesavetag")
        self.theta = kwargs.pop("theta")
        # if an iterative scheme....
        self.build_gp = kwargs.get("build_gp", None)
        self.update_theta = kwargs.get("update_theta", None)
        
        # set up default bounds if needed: 
        if self.gp_bounds is None and self.use_gp_bounds is True:
            print("Using default tinygp bounds of ")
            self.gp_bounds = np.asarray([[np.log(1.1), np.log(1)], 
                                      [np.log(21.2), np.log(3)]])
        
        return
    
    
    def __default_gp_setup(self, **kwargs):
        """ 
        Internal function to get things initialized for GP runs. 
        Options are: 
            allowed = ['celerite_mean', 'celerite_residual', 
                       'expsqr', 'expsinsqr', 'matern32']
            
            kwargs:
                - use_gp_bounds (bool)
                - gp_bounds (array) - should be as [[lower, upper sigma], 
                                                    [lower, upper rho]] 
        """
        self.use_gp_bounds = kwargs.get("use_gp_bounds", False)
        self.gp_bounds = kwargs.get("gp_bounds", None)
        
        if self.GP_type in self.allowed[2:]: 
            print("tinygp initialization:")
            self.plotFit = 1
            self.use_fit = 1
            self.logProbArgs = (self.time, self.flux, self.error)    
            self.logProbFunc = mc.log_probability_singlepower_noCBV
            self.labels = [r"$t_0$", "A", r"$\beta$",  "B"]
            self.filelabels = ["t0", "A", "beta",  "b"]
            start_t = min(self.time[-1]-2, self.discovery_time-3)
            self.init_values = np.array((start_t, 0.1, 1.8, 1))
            self.MCMC_PLOTTER = spgp.plot_mcmc_GP_tinygp
            
            # set up default bounds if needed: 
            if self.gp_bounds is None and self.use_gp_bounds is True:
                print("Using default tinygp bounds of ")
                self.gp_bounds = np.asarray([[np.log(1.1), np.log(1)], 
                                         [np.log(21.2), np.log(3)]])
            
            # expsqr
            if self.GP_type == 'expsqr':
                self.filesavetag = "-tinygp-expsqr"
                self.theta = {"log_sigma": np.log(2), 
                              "log_rho": np.log(1), } #nat logs
                # set up fxn to build kernel iteratively during mcmc
                self.build_gp = self.__build_tinygp_expsqr
                self.update_theta = self.__update_theta_sig_rho
                
            
            # matern 3/2
            elif self.GP_type == 'matern32':
                self.filesavetag = "-tinygp-matern32"
                self.theta = {"log_sigma", np.log(2),
                              "log_rho", np.log(1), }
                self.build_gp = self.__build_tinygp_matern32
                self.update_theta = self.__update_theta_sig_rho
            
            # expsinsqr
            elif self.GP_type == 'expsinsqr':
                self.filesavetag = "-tinygp-expsinsqr"
                self.theta = {
                    "log_sigma": np.log(2),
                    "log_rho": np.log(1),
                    "log_gamma": np.log(1),
                }
                self.build_gp = self.__build_tinygp_expsinsqr
                self.update_theta = self.__update_theta_sig_rho_gamma
            print("****** \n tinygp Initialized - Can \
                  Call run_gp_mcmc() from here! \n *******")
        elif self.GP_type == 'celerite_mean':
            self.__init_celerite_meanmodel(**kwargs)
                
        elif self.GP_type == 'celerite_residual':
            self.__init_celerite_residual(**kwargs)
            
        if self.gp_bounds is not None: 
            self.filesavetag += "-bounded"
        return
    
    def __init_celerite_residual(self, **kwargs):
        """ 
        Initializer for celerite residual model setup
        """
        self.filesavetag = "-celerite-M32-residual"
        # init kernel
        start_t = min(self.discovery_time-3, self.time[-1]-2)
        rho = 2
        sigma = 1
        # if using default bounds:
        if self.use_gp_bounds and self.gp_bounds is None:
            sigma_bounds = np.log( np.sqrt((0.1,20  )) ) #sigma range 0.316 to 4.47, take log
            rho_bounds = np.log((1, 10)) #0, 2.302
            bounds_dict = dict(log_sigma=sigma_bounds, log_rho=rho_bounds)
        elif not self.use_gp_bounds:
            # still have to lightly bound this one or it will freak out
            sigma_bounds = np.log( np.sqrt((1e-10,1e4  )) )
            rho_bounds = np.log((1e-10, 1e4))
            bounds_dict = dict(log_sigma=sigma_bounds, log_rho=rho_bounds)
       
        self.kernel = terms.Matern32Term(log_sigma=np.log(sigma), log_rho=np.log(rho), 
                                    bounds=bounds_dict)
                
        self.init_values = np.array((start_t, 0.1, 1.8, 0, np.log(sigma), np.log(rho)))
        self.gp = celerite.GP(self.kernel, mean=0.0)
        self.gp.compute(self.time, self.error)
        if not self.quiet: print(f"Initial log-likelihood: {self.gp.log_likelihood(self.flux)}")
        # set up arguments etc.
        self.logProbArgs = (self.time,self.flux, self.error, self.gp)
        self.logProbFunc = mc.log_probability_celerite_residual
        self.labels = ["t0", "A", "beta",  "b", r"$log\sigma$",r"$log\rho$"] 
        self.filelabels = ["t0", "A", "beta",  "b",  "logsigma", "logrho"]
        self.plotFit = 10
        
        print("****** \n Celerite Residuals Initialized - Can \
              Call Regular run_mcmc() from here! \n *******")
        return
    
    def __init_celerite_meanmodel(self, **kwargs):
        """ 
        Initializer for celerite mean model setup 
        """
        class MeanModel(Model):
            """ simplest celerite matern 3/2 mean model"""
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
        # end class, begin initialization
        self.filesavetag = "-celerite-M32-mean"
        # power law setup: 
        self.bounds_model_dict = {"t0":(0, self.time[-1]),
                                  "A": (0.001, 20),
                                  "beta":(0.5,6.0),
                                  "b":(-50, 50)}
        start_t = min(self.discovery_time-3, self.time[-1]-2)
        t0, A, beta, b = (start_t, 5, 1.8, 5)
        self.mean_model = MeanModel(t0=t0, A=A, beta=beta, b=b,
                                    bounds = self.bounds_model_dict)
        # gp setup:
        if self.gp_bounds is None and self.use_gp_bounds is True:
            print("Using default celerite mean model bounds of ")
            # this used to be called bounds_dict
            self.gp_bounds = {'log_sigma':np.log(np.sqrt((0.1,20))), 
                                    'log_rho':np.log((1,10))}
        self.kernel = terms.Matern32Term(log_rho=np.log(2), log_sigma=np.log(1),
                                         bounds=self.gp_bounds)
        self.gp = celerite.GP(self.kernel, mean=self.mean_model, fit_mean=True)
        self.gp.compute(self.time, self.error)
        if not self.quiet:
            print(f"Initial log-likelihood: {self.gp.log_likelihood(self.flux)}")
            
        # define cost fxn and its gradient
        def neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.log_likelihood(y)
        
        def grad_neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.grad_log_likelihood(y)[1]
    
        # Fit for the maximum likelihood parameters
        initial_params = self.gp.get_parameter_vector()
        bounds = self.gp.get_parameter_bounds()
        if not self.quiet: print("Running scipy maximizer")
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                        method="L-BFGS-B", bounds=bounds, args=(self.flux, self.gp))

        self.gp.set_parameter_vector(soln.x)
        # Make the maximum likelihood prediction
        self.t = np.linspace(0, self.time[-1], 500)
        self.mu, var = self.gp.predict(self.flux, self.t, return_var=True)
        self.std = np.sqrt(var)

        # Plot the data + scipy prediction
        if self.plot: sp.plot_scipy_max(self)

        self.init_values = soln.x
        self.logProbArgs = (self.flux, self.gp)
        self.logProbFunc = mc.log_probability_celerite_mean
        self.labels = [r"$log\sigma$",r"$log\rho$", "t0", "A", "beta",  "b"] 
        self.filelabels = ["logsigma", "logrho", "t0", "A", "beta",  "b"]
        self.plotFit = 10
        
        print("****** \n Mean Model Initialized - Can \
              Call Regular run_mcmc() from here! \n *******")
        return 
        
    def __build_tinygp_expsqr(self, theta, X):
        """ Make the expsqr kernel for the current theta, X values"""
        k1 = jnp.exp(theta["log_sigma"]) * kernels.ExpSquared(jnp.exp(theta["log_rho"]))
        return GaussianProcess(k1, X, mean=0.0, diag=self.error)
    
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
         
    def __update_theta_sig_rho_gamma(self, solnparams):
        """ Updates dict of values """
        self.theta["log_sigma"] = solnparams["log_sigma"]
        self.theta["log_rho"] = solnparams["log_rho"]
        self.theta['log_gamma'] = solnparams['log_gamma']
        return
        
    def __update_theta_sig_rho(self, solnparams):
        """ Updates dict of values """
        self.theta["log_sigma"] = solnparams["log_sigma"]
        self.theta["log_rho"] = solnparams["log_rho"]
        return

    
    def run_gp_mcmc(self, n1=5_000, n2=10_000, **kwargs):
        """ 
        Execute the mcmc chain with concurrent GP fitting
        parameters:
            - n1 (int) first chain
            - n2 (int) second chain
            
        kwargs: 
            - quiet (bool) turn print statements off (def: false)
            - nwalkers (int) how many walkers (def: 100)
            - discard (int) default 0 (initial chain discard)
            - thinning (int) default 1 (uses every sample, a setting of 15 would
                                        use every 15th, etc.)
            - seed (int) random seed initializer (def: 42)
            - nwalkers (int) default: 100
        """
        if not self.using_GP:
            raise ValueError("Must run mcmc_gp_setup() first!")
            return 
        
        def make_residual(x, y, best_mcmc):
            t0, A,beta,B = best_mcmc
            t1 = x - t0
            sl = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + B
            return y - sl
        
        def neg_log_likelihood(theta, X, y):
            gp = self.build_gp(theta, X)
            return -gp.log_probability(y)
        
        self.quiet = kwargs.get("quiet", False)
        
        if not self.quiet:
            print(" *** \n *** \n *** \n ***")
            print("Beginning MCMC + tinyGP run")
            
        timeModule.sleep(3) # this keeps things running orderly
        
        discard_samples = kwargs.get("discard", 0)
        thinning = kwargs.get("thinning", 1)
            
        # init
        np.random.seed(kwargs.get("seed", 42)) 
        nwalkers = kwargs.get("nwalkers", 100)
        self.ndim = len(self.labels) #labels should've already been set up
        pos_0 = np.full((nwalkers, self.ndim), self.init_values) + \
                np.random.uniform(0, 0.1, (nwalkers, self.ndim))
        #         p0 = (np.ones((nwalkers, self.ndim)) * self.init_values + 
        #               np.random.uniform(0, 0.1, (nwalkers, self.ndim)))
        
        #### First chain
        if not self.quiet: print("Starting burn-in chain...")
        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, 
                                             self.logProbFunc, 
                                             args=self.logProbArgs)
        self.sampler.run_mcmc(pos_0, n1, progress=True)
        
        # plot the burn in chain (if plotting): 
        if self.plot: sp.plot_chain_withlogpost(self, appendix="burnin")
        
        #get intermediate best to initialize next chain: 
        self.flat_samples = self.sampler.get_chain(discard=discard_samples, 
                                                   flat=True, thin=thinning)
        best_inter = np.zeros(self.ndim)
        for i in range(self.ndim):
            best_inter[i] = np.percentile(self.flat_samples[:, i], [16, 50, 84])[1]
             #save just the 50th percentile as best
            
        #### Main Chain
        if not self.quiet: print("\nStarting production chain...")
        pos_1 = np.full((nwalkers, self.ndim), best_inter) + \
                np.random.uniform(0, 0.1, (nwalkers, self.ndim))
        self.sampler.reset()
        
        # Correlation function / Iterative GP Fitting
        self.index = 0
        self.stepsize = 100 # how often to check
        self.GP_stepsize = 1000 # how often to refit
        old_tau = np.inf # prev autocorr time 
        # empty array to save all checked autocorrs into - one mean, one all
        self.autocorr_means = np.empty(int(np.ceil(n2/self.stepsize))+1)
        self.autocorr_all = np.empty((int(np.ceil(n2/self.stepsize))+1, self.ndim))
        
        # GP setup for this part: calc residual from intermediate best
        res = make_residual(self.time, self.flux, best_inter)
        # then jax the neg log likelihood
        NLL = jax.jit(jax.value_and_grad(neg_log_likelihood))
        if not self.quiet: print(f"Initial negative log likelihood: \
                                 {NLL(self.theta, self.time, res)[0]}")
        solver = jaxopt.ScipyBoundedMinimize(fun=neg_log_likelihood)
        soln = solver.run(self.theta, self.gp_bounds, X=self.time, y=res)
        if not self.quiet: print(f"Final negative log likelihood: {soln.state.fun_val}")
        # set theta to new values
        self.update_theta(soln.params)
        self.GP_LL_all = [soln.state.fun_val]
        
        
        # run sampler
        for sample in self.sampler.sample(pos_1, iterations=n2, progress=True):
            
            #refit the GP every 1000 steps
            if self.sampler.iteration % 1000 == 0:
                #new residual:
                self.flat_samples = self.sampler.get_chain(flat=True)
                best_inter = np.zeros(self.ndim)
                for i in range(self.ndim):
                    best_inter[i] = np.percentile(self.flat_samples[:, i], [16, 50, 84])[1]
                    
                res = make_residual(self.time, self.flux, best_inter)
                soln = solver.run(self.theta, self.gp_bounds, X=self.time, y=res)
                self.GP_LL_all.append(soln.state.fun_val)
                self.update_theta(soln.params)
                
            # only check conv. every stepsize: 
            if self.sampler.iteration % self.stepsize: 
                continue
            # compute autocorr time
            tau = self.sampler.get_autocorr_time(tol=0) # tol=0 always get estimate
            if np.any(tau == np.nan) or np.any(tau == np.inf) \
                or np.any(tau == -np.inf):
                print("autocorr is nan or inf")
                print(tau)
            self.autocorr_means[self.index] = np.mean(tau) #mean
            self.autocorr_all[self.index] = tau #per parameter
            self.index += 1
            
            # check convergence
            self.converged = np.all((tau * 100) < self.sampler.iteration)
            self.converged &= np.all((np.abs(old_tau - tau) / tau) < 0.01) # normally 0.01
            if self.converged:
                if not self.quiet: print("Converged, ending chain")
                break
            old_tau = tau # update old tau
        
        # save output gp params: 
        self.update_theta(soln.params)
        # plot gp LL over steps and autocorr times
        if self.plot: 
            spgp.plot_tinygp_ll(self)
            sp.plot_autocorr_all(self)
            
        # thin and burn out
        tau = self.sampler.get_autocorr_time(tol=0)
        if (np.max(tau) < (self.sampler.iteration/100)):
            burnin = int(2 * np.max(tau))
        else:
            burnin = int(n2/10)
        thinning = 1
         
        self.flat_samples = self.sampler.get_chain(discard=burnin, flat=True, thin=thinning)
        if self.plot: sp.plot_param_samples_all(self)
        if not self.quiet: print(f"{len(self.flat_samples)} samples post second run")
        
        #### Best Fit Params:
        self.best_mcmc = np.empty(self.ndim)
        self.upper_error = np.empty(self.ndim)
        self.lower_error = np.empty(self.ndim)
        for i in range(self.ndim):
            mcmc = np.percentile(self.flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            if not self.quiet: print(r"{self.labels[i]}", mcmc[1], -1 * q[0], q[1] )
            self.best_mcmc[i] = mcmc[1]
            self.upper_error[i] = q[1]
            self.lower_error[i] = q[0]
            
        # Apply GR corrections
        self.__calc_gelmanrubin()
        self.upper_error = np.sqrt(self.upper_error**2 * self.GR)
        self.lower_error = np.sqrt(self.lower_error**2 * self.GR)
        print("Applied Gelman-Rubin Corrections")
        
        #### Estimate BIC - have to add the model + the gp log probs together
        negll = -1.0 * float(self.GP_LL_all[-1])
        logprob_, blob = self.sampler.compute_log_prob(self.best_mcmc.reshape(1,self.ndim))
        logprob = -1.0 * (logprob_+negll)
        
        if not self.quiet:
            print("negative log prob, no GP: ", logprob_)
            print("negative log like,  GP: ", negll)
        
        self.BIC = (self.ndim * np.log(len(self.time)) - 2 * (logprob * -1.0))[0]
        if not self.quiet: print(f"BIC: {self.BIC:.2f}; Log Prob: {logprob[0]:.2f}")
        if np.isnan(np.float64(self.BIC)): self.BIC = np.inf
        
        # Final plotting: 
        if self.plot: 
            sp.plot_chain_withlogpost(self, appendix="production")
            self.gp = self.build_gp(self.theta, self.time)
            self.MCMC_PLOTTER(self)
            
        
        # Save Output Parameters
        ut.param_save(self)
        print("MCMC Run Completed!")
        
        return 


    

# ############## workflow for a normal predefined non-GP fit
# save_dir = "/Users/lindseygordon/research/"
# ets = etsfit(save_dir, plot=True)

# TNSFile = "/Users/lindseygordon/research/etsfit/tutorials/tutorial_data/2018hzh_TNS.csv"
# TNSinfo = pd.read_csv(TNSFile)
# #data file:
# dataFile = "/Users/lindseygordon/research/etsfit/tutorials/tutorial_data/2018hzh0431-tessreduce"

# #load data:
# (time, flux, error, targetlabel, 
#                  sector, camera, ccd) = ut.tr_load_lc(dataFile)
# #get the discovery time from the TNSFile
# discovery_time = ut.get_disctime(TNSFile, targetlabel)

# ets.load_single_lc(time, flux, error, targetlabel, 
#                    sector=int(sector), camera=int(camera), ccd=int(ccd), 
#                    discovery_time=discovery_time, targetfile=TNSFile)
    
# ets.test_plot()  

# ets.data_preprocess(fraction_trim=0.5, use_flux_mask=True)
# ets.mcmc_setup(default_fit=1)
# ets.run_mcmc(n1=5_000, n2=10_000)
    
################## workflow for a normal predefined GP fit.
# save_dir = "/Users/lindseygordon/research/"
# ets = etsfit(save_dir, plot=False)

# TNSFile = "/Users/lindseygordon/research/etsfit/tutorials/tutorial_data/2018hzh_TNS.csv"
# TNSinfo = pd.read_csv(TNSFile)
# #data file:
# dataFile = "/Users/lindseygordon/research/etsfit/tutorials/tutorial_data/2018hzh0431-tessreduce"

# #load data:
# (time, flux, error, targetlabel, 
#                  sector, camera, ccd) = ut.tr_load_lc(dataFile)
# #get the discovery time from the TNSFile
# discovery_time = ut.get_disctime(TNSFile, targetlabel)

# ets.load_single_lc(time, flux, error, targetlabel, 
#                    sector=int(sector), camera=int(camera), ccd=int(ccd), 
#                    discovery_time=discovery_time, targetfile=TNSFile)
    
# ets.test_plot()  

# ets.data_preprocess(fraction_trim=0.5, use_flux_mask=True)
# ets.mcmc_gp_setup()
# ets.run_gp_mcmc()

   

    


    

    

 