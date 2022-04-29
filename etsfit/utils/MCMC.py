# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:50:24 2021

@author: lindsey gordon

all 6 mcmc running functions for fitting to Ia SNe

different powerlaws to fit:
    - single power law no cbvs (log_probability_singlepower_noCBV)
    - double power law no cbvs (log_probability_doublepower_noCBV)
    - single power law with cbvs
    - double power law with cbvs
    - ONLY fitting cbv background model
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
import emcee
import datetime 
import time as timeModule
from pylab import rcParams
rcParams['figure.figsize'] = 16,6


# from celerite.modeling import Model
# from scipy.optimize import minimize
# import celerite
# from celerite import terms


def log_probability_singlepower_noCBV(theta, x, y, yerr, disctime):
        """ calculates log probabilty for model w/ single power law and no cbv fitting
        labels = ["t0", "A", "beta",  "b"]
        init_values = np.array((disctime-3, 0.1, 1.8, 1))
        returns loglike, logprior"""
        
        t0, A, beta, b = theta
        # handle log priors
        
        if (x[0] < t0 < x[-1] and 0.5 < beta < 6.0 
            and 0.0 < A < 5.0 and -5 < b < 5):
            lp = 0.0
        else:
            lp = -np.inf
        
        # if not allowed values
        if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
            return -np.inf, lp
        else: # if allowed, calculate the log likelihood
            t1 = x - t0
            model = (np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta))) + 1 + b
            
            yerr2 = yerr**2.0
            return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp
        
def log_probability_singlepower_withCBV(theta, x, y, yerr, 
                                        Qall, CBV1, CBV2, CBV3, disctime):
        """ calculates log probabilty for single power law with cbv fitting
        labels = ["t0", "A", "beta", "B","cQ", "c1", "c2", "c3"]
        init_values = np.array((disctime-3, 0.1, 1.8, 0, 0,0,0,0))
        
        """
        
        t0, A, beta, B, cQ, c1, c2, c3 = theta
        # handle log priors
        if (x[0] < t0 < x[-1] and 0.5 < beta < 6.0 
            and 0.0 < A < 5.0):
            lp = 0.0
        else:
            lp = -np.inf
        
        # if not allowed values
        if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
            return -np.inf, lp
        
        else: # if allowed, calculate the log likelihood
            t1 = x - t0
            model = (np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta)) + 
                 cQ * Qall + c1 * CBV1 + c2 * CBV2 + c3 * CBV3 + 1 + B)
            
            yerr2 = yerr**2.0
            return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp
        

def log_probability_doublepower_noCBV(theta, x, y, yerr, disctime):
        """ calculates log probabilty
        labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  "b"]
        init_values = np.array((disctime-6, disctime, 0.1, 0.1, 1.8, 1.8, 1))"""
        def func1(x, t1, t2, a1, a2, B1, B2):
            return B1 *(x-t1)**a1
        def func2(x, t1, t2, a1, a2, B1, B2):
            return B1 * (x-t1)**a1 + B2 * (x-t2)**a2
        
        
        t1, t2, a1,a2, beta1, beta2, b = theta
        # handle log priors
        if (x[0] < t1 < disctime
            and t1 < t2 < x[-1]
            and 0.5 < beta1 < 6.0 and 0.5 < beta2 < 6.0
            and 0.0 < a1 < 3.0 and 0.0 < a2 < 3.0 
            and -5 < b < 5):
            lp = 0.0
        else:
            lp = -np.inf
        
        # if not allowed values
        if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
            return -np.inf, lp
        else: # if allowed, calculate the log likelihood
            model = np.piecewise(x, [(t1 <= x)*(x < t2), t2 <= x], 
                             [func1, func2],
                             t1, t2, a1, a2, beta1, beta2) + 1 + b
            
            yerr2 = yerr**2.0
            return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp
        
def log_probability_doublepower_withCBV(theta, x, y, yerr, 
                                        Qall, CBV1, CBV2, CBV3, disctime):
        """ calculates log probabilty
        labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",\
                  "cQ", "c1", "c2", "c3"]
        init_values = np.array((disctime-3, disctime, 0.1, 0.1, 1.8, 1.8, 0,0,0,0))"""
        def func1(x, t1, t2, a1, a2, B1, B2):
            return B1 *(x-t1)**a1
        def func2(x, t1, t2, a1, a2, B1, B2):
            return B1 * (x-t1)**a1 + B2 * (x-t2)**a2
        
        
        t1, t2, a1,a2, beta1, beta2, cQ, c1, c2, c3 = theta
        # handle log priors
        if (x[0] < t0 < x[-1]
            and (t1) < t2 < x[-1]
            and 0.5 < beta1 < 6.0 and 0.5 < beta2 < 6.0
            and 0.0 < a1 < 3.0 and 0.0 < a2 < 3.0 
            and -1.0 < cQ < 1.0 
            and -1.0 < c1 < 1.0 and -1.0 < c2 < 1.0 
            and -1.0 < c3 < 1.0):
            lp = 0.0
        else:
            lp = -np.inf
        
        # if not allowed values
        if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
            return -np.inf, -np.inf
        else: # if allowed, calculate the log likelihood
            model = (np.piecewise(x, [(t1 <= x)*(x < t2), t2 <= x], 
                                  [func1, func2], t1, t2, a1, a2, beta1, beta2)
                     + cQ * Qall + c1 * CBV1 + c2 * CBV2 + c3 * CBV3 + 1)
            
            yerr2 = yerr**2.0
            return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp

def log_probability_justCBV(theta, x, y, yerr, Qall, CBV1, CBV2, CBV3, disctime):
        """ calculates log probabilty for single power law with cbv fitting
        labels = ["b", "cQ", "c1", "c2", "c3"]
        init_values = np.array((1, 0,0,0,0))
        
        ** does this need to have Qall, cbv's included as passed params?"""
        
        b, cQ, c1, c2, c3 = theta
        # handle log priors
        if (-200.0 < cQ < 200.0 and -200.0 < c1 < 200.0 and -200.0 < c2 < 200.0 
            and -200.0 < c3 < 200.0 and -200.0 < b < 200.0):
            lp = 0.0
        else:
            lp = -np.inf
        
        # if not allowed values
        if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
            return -np.inf, -np.inf
        
        else: # if allowed, calculate the log likelihood
            model = b + (cQ * Qall) + (c1 * CBV1) + (c2 * CBV2) + (c3 * CBV3) + 1
            
            yerr2 = yerr**2.0
            return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp
        
def log_probability_singlePower_LBG(theta, x, y, yerr, lygosBG, disctime):
        """ calculates log probabilty for single power law with LYGOS BACKGROUND
        labels = ["t0", "A", "beta",  "b", "LBG"]
        init_values = np.array((disctime-3, 0.1, 1.8, 1, 1))
        
        """
        
        t0, A, beta, b, LBG = theta
        # handle log priors
        
        if (x[0] < t0 < x[-1] and 0.5 < beta < 6.0 
            and 0.0 < A < 5.0 and -5 < b < 5 and -5 < LBG < 5):
            lp = 0.0
        else:
            lp = -np.inf
        
        # if not allowed values
        if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
            return -np.inf, -np.inf
        else: # if allowed, calculate the log likelihood
            t1 = x - t0
            model = ((np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta))) + 
                     1 + b + lygosBG * LBG)
            
            yerr2 = yerr**2.0
            return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp
        


# =============================================================================
# def mcmc_outer_structure(path, targetlabel, time, intensity, error, 
#                          lygosBG, tmin, sector, camera, ccd, disctime, fitType,
#                          fract = None, Bin8Hr = False, n1=10000, n2=40000,
#                          CBV_folder = "C:/Users/conta/.eleanor/metadata/",
#                          qfolder = "D:/quaternions-txt/"):
#     """ 
#     MCMC fitting - outline for all 5 different fittings
#     Params: 
#         - path to save everything into
#         - str targetlabel
#         - time axis x
#         - intensity data
#         - error data
#         - lygosBG data
#         - tmin of time data
#         - sector of discovery (int), camera, ccd
#         - disctime - discovery_time in BJD (must match up with x)
#         - fitType 1-6 of what's being fit
#         - fract - None if taking whole light curve, 0.4 for 40%, etc.
#         - 8hrBin - default is False, set to true if you want to bin it
#         - CBV folder where CBVs are 
#         - qfolder where quaternion TEXT files are.
#         
#     
#     """
#     import emcee
#     import os
#     import utils.snPlotting as sp
#     import utils.utilities as ut
#     
#     
#     #### CBVS LOADING
#     if fitType in (2,4,5): #if you need CBVs, load them
#         (time, intensity, error, 
#          quatTime, Qall, CBV1, CBV2, 
#          CBV3) = ut.generate_clip_quats_cbvs(sector, time,intensity,
#                                                 error, tmin, camera, ccd,
#                                                 CBV_folder, qfolder)
#         quatsandcbvsforplotting = [Qall, CBV1, CBV2, CBV3]
#     else:
#         quatsandcbvsforplotting = None #this has to be initialized as None
#         
#     timeModule.sleep(3) #this keeps things running orderly
# 
#     print("RUNNING MCMC: time at start: ",time[0])
#     print("RUNNING MCMC: discovery time: ", disctime)
#     
#     #check for 8hr bin BEFORE trimming to percentages
#     if Bin8Hr:
#         (time, intensity, 
#          error, lygosBG,
#          quatsandcbvsforplotting) = ut.bin_8_hours(time, intensity, error, 
#                                                    lygosBG, 
#                                                    QCBVALL=quatsandcbvsforplotting)
#     #print(len(time))
#     #if quatsandcbvsforplotting is not None:
#      #   print(len(quatsandcbvsforplotting[0]))
#     #if doing percent of max fitting
#     if fract is not None:
#         (time, intensity, error, lygosBG, 
#          quatsandcbvsforplotting) = ut.fractionalfit(time, intensity, 
#                                                      error, lygosBG, 
#                                                      fract, 
#                                                      quatsandcbvsforplotting)
#     
#     #####fitType sets up all the details
#     if fitType == 1: #single without
#         args = (time, intensity, error, disctime)
#         logProbFunc = log_probability_singlepower_noCBV
#         filesavetag = "-singlepower"
#         labels = ["t0", "A", "beta",  "b"]
#         init_values = np.array((disctime-3, 0.1, 1.8, 1))
#     elif fitType == 2: #single with
#         Qall, CBV1, CBV2, CBV3 = quatsandcbvsforplotting
#         args = (time, intensity, error, Qall, CBV1, CBV2, CBV3, disctime)
#         logProbFunc = log_probability_singlepower_withCBV
#         filesavetag = "-singlepower-CBV"
#         labels = ["t0", "A", "beta", "B", "cQ", "c1", "c2", "c3"]
#         init_values = np.array((disctime-3, 0.1, 1.8, 0, 0,0,0,0))
#     elif fitType == 3: #double without
#         args = (time, intensity, error, disctime)
#         logProbFunc = log_probability_doublepower_noCBV
#         filesavetag = "-doublepower"
#         labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  "b"]
#         init_values = np.array((disctime-8, disctime-2, 0.1, 0.1, 1.8, 1.8, 1))
#     elif fitType ==4: #double with
#         Qall, CBV1, CBV2, CBV3 = quatsandcbvsforplotting
#         args = (time, intensity, error, Qall, CBV1, CBV2, CBV3, disctime)
#         logProbFunc = log_probability_doublepower_withCBV
#         filesavetag = "-doublepower-CBV"
#         labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  
#                   "cQ", "c1", "c2", "c3"]
#         init_values = np.array((disctime-8, disctime-2, 0.1, 0.1, 
#                                 1.8, 1.8, 0,0,0,0))
#     elif fitType == 5: #just CBVs
#         Qall, CBV1, CBV2, CBV3 = quatsandcbvsforplotting
#         args = (time, intensity, error, Qall, CBV1, CBV2, CBV3, disctime)
#         logProbFunc = log_probability_justCBV
#         filesavetag = "-CBV"
#         labels = ["b", "cQ", "c1", "c2", "c3"]
#         init_values = np.array((1, 0,0,0,0))
#     elif fitType == 6: #detrending lygos BG
#         args = (time, intensity, error, lygosBG, disctime)
#         logProbFunc = log_probability_singlePower_LBG
#         filesavetag = "-singlepower-lygosBG"
#         labels = ["t0", "A", "beta",  "b", "LBG"]
#         init_values = np.array((disctime-3, 0.1, 1.8, 1, 1))
#     else:
#         #raise ValueError("not an allowed fit type")
#         print("THAT IS NOT AN ALLOWED FIT TYPE, EXITING")
#         return time, intensity, error
#     
#     if Bin8Hr:
#         filesavetag = filesavetag + "-8HourBin"
#     
#     if fract is not None:
#         filesavetag = filesavetag + "-{fraction}".format(fraction=fract)
#     
#         
# 
#     #### set up output folder + files
#     #check for an output folder's existence, if not, put it in. 
#     newfolderpath = path + targetlabel + str(sector) + str(camera) + str(ccd)
#     if not os.path.exists(newfolderpath):
#         os.mkdir(newfolderpath)
#     #make subfolder for this run
#     subfolderpath = newfolderpath + "/" + filesavetag[1:]
#     if not os.path.exists(subfolderpath):
#         os.mkdir(subfolderpath)
#     path = subfolderpath + "/"
#     parameterSaveFile = path + "output_params.txt"
#     
#     print(path)
#     print("***")
#     print("***")
#     print("***")
#     print("***")
#      
#     timeModule.sleep(3) #this keeps things running orderly
# 
#     #### MCMC setup
#     np.random.seed(42)
#     nwalkers = 100
#     ndim = len(labels) #labels are provided when you run it
#     p0 = np.zeros((nwalkers, ndim)) #init positions
#     for n in range(len(p0)): #add a little spice - YYY gaussian??
#         p0[n] = init_values + (np.ones(ndim) - 0.9) * np.random.rand(ndim) 
#     
#     #### Initial run
#     sampler = emcee.EnsembleSampler(nwalkers, ndim, logProbFunc,args=args) #setup
#     sampler.run_mcmc(p0, n1, progress=True) #run it
#     sp.plot_chain_logpost(path, targetlabel, filesavetag, sampler, labels, 
#                        ndim, appendix = "-burnin")
#     
#     discardy = int(n1/2)
#     flat_samples = sampler.get_chain(discard=discardy, flat=True, thin=15)
#     #get intermediate best
#     best_mcmc_inter = np.zeros((1,ndim))
#     for i in range(ndim):
#         mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
#         best_mcmc_inter[0][i] = mcmc[1]
#         
#     #### Main run
#     np.random.seed(50)
#     p0 = np.zeros((nwalkers, ndim))
#     for i in range(nwalkers): #reinitialize the walkers around prev. best
#         p0[i] = best_mcmc_inter[0] + 0.1 * np.random.rand(1, ndim)
#        
#     sampler.reset()
#     
#     #### CORRELATION FUNCTION 
#     index = 0 #number of checks
#     autocorr = np.empty(n2) #total possible checks
#     old_tau = np.inf
#     autoStep = 1000 #how often to check
#     autocorr_all = np.empty((int(n2/autoStep) + 2,len(labels))) #save all autocorr times
#     
#     #sample up to n2 steps
#     for sample in sampler.sample(p0, iterations=n2, progress=True):
#         # Only check convergence every 100 steps
#         if sampler.iteration % autoStep:
#             continue
#         # Compute the autocorrelation time so far
#         tau = sampler.get_autocorr_time(tol=0) #tol=0 always get estimate
#         if np.any(tau == np.nan) or np.any(tau == np.inf) or np.any(tau == -np.inf):
#             print("autocorr is nan or inf")
#             print(tau)
#         #this pops out with len(tau) = ndims - need all to converge to be conv
#         autocorr[index] = np.mean(tau) #save mean autocorr time
#         autocorr_all[index] = tau #save all autocorr times for plotting
#         index += 1 #how many times have you saved it
#     
#         # Check convergence
#         #this first condition is absolutely where it's failing
#         converged = np.all(tau * 100 < sampler.iteration)
#         converged &= np.all(np.abs(old_tau - tau) / tau < 0.01) #normally 0.01
#         if converged:
#             print("Converged, ending chain")
#             break
#         old_tau = tau
#     
#     ##plot autocorr things
#     sp.plot_autocorr_mean(path, targetlabel, index, autocorr, converged, 
#                           autoStep, filesavetag)
#     sp.plot_autocorr_individual(path, targetlabel, index, autocorr_all,
#                              autoStep, labels, filesavetag)
#     
#     #thin and burn out dump
#     tau = sampler.get_autocorr_time(tol=0)
#     if (np.max(tau) < (sampler.iteration/50)):
#         burnin = int(2 * np.max(tau))
#         thin = int(0.5 * np.min(tau))
#     else:
#         burnin = 5000
#         thin = 15
#     flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
#     
#     #this will be separate - plotting p(parameter)
#     sp.plot_paramIndividuals(flat_samples, labels, path, 
#                              targetlabel, filesavetag)
#     
#     sp.plot_chain_logpost(path, targetlabel, filesavetag, sampler, labels, 
#                        ndim, appendix = "-production")
#     
#     print(len(flat_samples), "samples post second run")
# 
#     #### BEST FIT PARAMS
#     best_mcmc = np.zeros((1,ndim))
#     upper_error = np.zeros((1,ndim))
#     lower_error = np.zeros((1,ndim))
#     for i in range(ndim):
#         mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
#         q = np.diff(mcmc)
#         print(labels[i], mcmc[1], -1 * q[0], q[1] )
#         best_mcmc[0][i] = mcmc[1]
#         upper_error[0][i] = q[1]
#         lower_error[0][i] = q[0]
#  
#     logprob, blob = sampler.compute_log_prob(best_mcmc)
#     #print(logprob)
#     #### BIC
#     BIC = ndim * np.log(len(time)) - 2 * np.log(logprob)
#     print("BAYESIAN INF CRIT: ", BIC)
#     if np.isnan(np.float64(BIC[0])): #if it's a nan
#         BIC = 50000
#     else:
#         BIC = BIC[0]
#         
#     
#     sp.plot_mcmc(path, time, intensity, targetlabel, disctime, best_mcmc[0], 
#                  flat_samples,labels, fitType, filesavetag, tmin, lygosBG,
#                  quatsandcbvsforplotting)
#     
#     with open(parameterSaveFile, 'w') as file:
#         file.write(filesavetag + "-" + str(datetime.datetime.now()))
#         file.write("\n {best} \n {upper} \n {lower} \n".format(best=best_mcmc,
#                                                                upper=upper_error,
#                                                                lower=lower_error))
#         file.write("BIC:{bicy:.3f} Converged:{conv} \n".format(bicy=BIC, 
#                                                             conv=converged))
#     
#     return best_mcmc, BIC, quatsandcbvsforplotting
# 
# 
# 
# 
# =============================================================================
