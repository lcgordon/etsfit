# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:56:22 2022

@author: lindsey gordon

GP fitting code bits
"""

import utilities as ut
import numpy as np
import matplotlib.pyplot as plt
import sn_plotting as sp

from celerite.modeling import Model
from scipy.optimize import minimize
import celerite
from celerite import terms

from pylab import rcParams
rcParams['figure.figsize'] = 16,6
rcParams["font.size"] = 20
import os

import emcee



        

def mcmc_GP(pathSave, time, intensity, error, lygosBG, tmin, targetlabel, 
            sector, camera, ccd, disctime,):
        
    """ Run the mcmc fit of the GP included model"""


        
   
    ndim, nwalkers = len(initial), 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_GP,
                                    args = (time,intensity, error, disctime, gp))
        
        
    print("Running burn-in...")
    p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
    p0, lp, _, other = sampler.run_mcmc(p0, 500, progress=True)
    print("Plotting first chain...")
    sp.plot_chain_logpost(pathSave, targetlabel, filesavetag, sampler, labels, 
                          ndim,appendix = "-burnin")
        
    
        
    print("Running production...")
    sampler.reset()
    sampler.run_mcmc(p0, 5000, progress=True)
    print("Plotting production chain...")
    sp.plot_chain_logpost(pathSave, targetlabel, filesavetag, sampler, labels, 
                          ndim, appendix = "-production")
        
    flat_samples = sampler.get_chain(discard=1000, flat=True, thin=10)
    
    sp.plot_corner(flat_samples, labels, pathSave, targetlabel, filesavetag)
    sp.plot_paramIndividuals(flat_samples, filelabels, pathSave, 
                             targetlabel, filesavetag)
    
    #### Get best fit parameters
    #### BEST FIT PARAMS
    best_mcmc = np.zeros((1,ndim))
    upper_error = np.zeros((1,ndim))
    lower_error = np.zeros((1,ndim))
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(labels[i], mcmc[1], -1 * q[0], q[1] )
        best_mcmc[0][i] = mcmc[1]
        upper_error[0][i] = q[1]
        lower_error[0][i] = q[0]
        
    #### plot best fit parameters
    plot_mcmc_GP(pathSave, time, intensity, error, best_mcmc, gp, disctime, tmin,
                 targetlabel, filesavetag)
    #### BIC
    logprob, blob = sampler.compute_log_prob(best_mcmc)
    BIC = ndim * np.log(len(time)) - 2 * np.log(logprob)
    if np.isnan(np.float64(BIC[0])): #if it's a nan
        BIC = 50000
    else:
        BIC = BIC[0]
    print("BAYESIAN INF CRIT: ", BIC)
    return 

