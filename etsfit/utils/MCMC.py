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
        

def log_probability_GP(theta, x, y, yerr, disctime, gp):
        """GP log probability calculator"""
        t0, A, beta, b = theta[:4]
        GPparams = theta[4:]
        #first handle the issues with the gp stuff
        gp.set_parameter_vector(GPparams) #gp params set to current params
        #get lp of GP
        lp = gp.log_prior() #gp log prior
        #add lp of actual
        if (x[0] < t0 < disctime and 0.5 < beta < 6.0 
            and 0.001 < A < 2.0 and -5 < b < 5):
            lp += 0.0
        else:
            lp += -np.inf
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