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
import emcee
import datetime 
import time as timeModule
from pylab import rcParams
rcParams['figure.figsize'] = 16,6


def check_priors(priors, theta):
    """ 
    Function to check that the current walker position is within the allowed
    prior values. This allows for override of the defaults by the user. 
    
    --------------------------------------------------
    Parameters:
        
        - priors (array of doubles): ie, for fit type 1, 
            priors = [x[0], x[-1], 0.5, 6.0, 0.0, 5.0, -5, 5]
            it is SO important that these are in the correct order for
            the parameters (see each fit type for what order they're in)
            
          priors should be ordred as [lower limit, upper limit] per pair                  
                            
        - theta (array of doubles): the parameters from MCMC being compared
        in order by default.
        
        
    it WILL check them in order - if you only put priors on the first 4 
    it will check those
    if you don't need a prior on one value then use +/- inf 
    for those if you have priors to check after that one in theta!!'
    (ie, params A,B,C, B has no limits on priors but C does, so set B to +/- inf )
            
    """
    
    if len(priors) % 2 != 0:
        #if the priors do not have the right number of terms (ie, odd number)
        raise Exception("size of prior range array is odd - fix this!")
    
    for p in range(int(len(priors)/2)):
        #array indexes as 2p, 2p+1
        #if the parameter is NOT in the range, return lp = -np.inf
        #else return lp = 0
        if not (priors[2*p] < theta[p] < priors[(2*p)+1]):
            return -np.inf
        
    return 0.0 #everything checked out
        
    

def log_probability_singlepower_noCBV(theta, x, y, yerr, disctime, priors=None):
        """ 
        
        Calculates the log probability for the model with a single power law
        and a flat background. 
        
        Associated labels: ["t0", "A", "beta",  "b"]
        init_values in MCMC: np.array((disctime-3, 0.1, 1.8, 1))
        
        Parameters:
            - theta (params in MCMC)
            - x (time index)
            - y (flux)
            - yerr (error)
            - disctime (discovery time)
            - priors (defaults to NONE, can be custom-set)
        

        returns loglike, logprior
        
        """
        
        t0, A, beta, b = theta
        # handle log priors
        if priors is None: #if you didn't feed it something else
            priors = [x[0], x[-1], 0.0, 10.0, 0.5, 6.0, -30, 30]
        lp = check_priors(priors, theta)
    
        # if not allowed values
        if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
            return -np.inf, lp
        else: # if allowed, calculate the log likelihood
            t1 = x - t0
            model = (np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta))) + 1 + b
            
            yerr2 = yerr**2.0
            return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp
        
def log_probability_singlepower_withCBV(theta, x, y, yerr, 
                                        Qall, CBV1, CBV2, CBV3, disctime, 
                                        priors=None):
        """ calculates log probabilty for single power law with cbv fitting
        labels = ["t0", "A", "beta", "B","cQ", "c1", "c2", "c3"]
        init_values = np.array((disctime-3, 0.1, 1.8, 0, 0,0,0,0))
        
        """
        
        t0, A, beta, B, cQ, c1, c2, c3 = theta
        # handle log priors
        
        if priors is None: #if you didn't feed it something else
            priors = [x[0], x[-1], 0.0, 10.0, 0.5, 6.0, -30,30, -30,30, -30,30, -30,30, -30,30] 
        lp = check_priors(priors, theta)
        
        # if not allowed values
        if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
            return -np.inf, lp
        
        else: # if allowed, calculate the log likelihood
            t1 = x - t0
            model = (np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta)) + 
                 cQ * Qall + c1 * CBV1 + c2 * CBV2 + c3 * CBV3 + 1 + B)
            
            yerr2 = yerr**2.0
            return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp
        

def log_probability_doublepower_noCBV(theta, x, y, yerr, disctime, priors=None):
        """ calculates log probabilty
        labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  "b"]
        init_values = np.array((disctime-6, disctime, 0.1, 0.1, 1.8, 1.8, 1))"""
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        
        
        t0, t1, A1, A2, beta1, beta2, B = theta
        #handle priors:
        if priors is None: #if you didn't feed it something else
            priors = [x[0], disctime, t0, x[-1], 0, 10, 0, 10, 0.5, 6, 0.5, 6, -30, 30] 
            #t0, t1, a1, a2, beta1, beta2,b
        lp = check_priors(priors, theta)
        
        # if not allowed values
        if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
            return -np.inf, lp
        else: # if allowed, calculate the log likelihood
            model = np.piecewise(x, [(t0 <= x)*(x < t1), t1 <= x], 
                             [func1, func2],
                             t0, t1, A1, A2, beta1, beta2) + 1 + B
            
            yerr2 = yerr**2.0
            return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp
        
def log_probability_doublepower_withCBV(theta, x, y, yerr, 
                                        Qall, CBV1, CBV2, CBV3, disctime,
                                        priors=None):
        """ calculates log probabilty
        labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",\
                  "cQ", "c1", "c2", "c3"]
        init_values = np.array((disctime-3, disctime, 0.1, 0.1, 1.8, 1.8, 0,0,0,0))"""
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        
        t0, t1, A1, A2, beta1, beta2, cQ, c1, c2, c3 = theta
        
        # handle log priors
        if priors is None: #if you didn't feed it something else
            priors = [x[0], disctime, t1,x[-1], 0, 5, 0, 5, 0.5, 6, 0.5, 6,
                      -1,1,-1,1,-1,1,-1,1]
            
        lp = check_priors(priors, theta)
        
        # if not allowed values
        if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
            return -np.inf, -np.inf
        else: # if allowed, calculate the log likelihood
            model = (np.piecewise(x, [(t0 <= x)*(x < t1), t1 <= x], 
                             [func1, func2],
                             t0, t1, A1, A2, beta1, beta2)
                     + cQ * Qall + c1 * CBV1 + c2 * CBV2 + c3 * CBV3 + 1)
            
            yerr2 = yerr**2.0
            return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp

def log_probability_justCBV(theta, x, y, yerr, Qall, CBV1, CBV2, CBV3, disctime, 
                            priors=None):
        """ calculates log probabilty for single power law with cbv fitting
        labels = ["b", "cQ", "c1", "c2", "c3"]
        init_values = np.array((1, 0,0,0,0))
        
        ** does this need to have Qall, cbv's included as passed params?"""
        
        b, cQ, c1, c2, c3 = theta
        # handle log priors
        if priors is None: #if you didn't feed it something else
            priors = [-1,1,-1,1,-1,1,-1,1,-1,1]
            
        lp = check_priors(priors, theta)
        
        # if not allowed values
        if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
            return -np.inf, -np.inf
        
        else: # if allowed, calculate the log likelihood
            model = b + (cQ * Qall) + (c1 * CBV1) + (c2 * CBV2) + (c3 * CBV3) + 1
            
            yerr2 = yerr**2.0
            return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp
        
def log_probability_singlePower_LBG(theta, x, y, yerr, lygosBG, disctime, priors=None):
        """ calculates log probabilty for single power law with LYGOS BACKGROUND
        labels = ["t0", "A", "beta",  "b", "LBG"]
        init_values = np.array((disctime-3, 0.1, 1.8, 1, 1))
        
        """
        
        t0, A, beta, b, LBG = theta
        # handle log priors
        if priors is None: #if you didn't feed it something else
            priors = [x[0], x[-1], 0, 5, 0.5, 6, -5, 5, -5, 5]
        lp = check_priors(priors, theta)

        # if not allowed values
        if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
            return -np.inf, -np.inf
        else: # if allowed, calculate the log likelihood
            t1 = x - t0
            model = ((np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta))) + 
                     1 + b + lygosBG * LBG)
            
            yerr2 = yerr**2.0
            return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp
        

def log_probability_GP(theta, x, y, yerr, disctime, gp, priors=None):
        """GP log probability calculator"""
        
        t0, A, beta, b = theta[:4]
        GPparams = theta[4:]
        #first handle the issues with the gp stuff
        gp.set_parameter_vector(GPparams) #gp params set to current params
        #get lp of GP
        lp = gp.log_prior() #gp log prior
        #add lp of actual
        
        if priors is None: #if you didn't feed it something else
            priors = [x[0], disctime, 0.001, 2, 0.5, 6, -5, 5]
            
        lp += check_priors(priors, theta[:4]) #ADD to gp prior function
        
        #if lp is no good   
        if not np.isfinite(lp):
            return -np.inf, -np.inf #ll, lp

        t1 = x - t0
        model = ((np.heaviside((t1), 1) * A 
                  *np.nan_to_num((t1**beta))) + 1 + b)

        model += gp.predict(y, x, return_cov=False)
        
        yerr2 = yerr**2.0
        ll = -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2))
        #if ll is no good
        if not np.isfinite(ll):
            return lp, -np.inf
        
        return ll+lp, lp