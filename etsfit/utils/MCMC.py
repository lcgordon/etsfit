# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:50:24 2021
Updated Oct 6 2023 - Fixing docstrings for sphinx

Default set of MCMC log prob functions for power law modeling

"""
import numpy as np


def check_priors(priors, theta):
    """ 
    Function to check that the current walker position is within the allowed
    prior values. This allows for override of the defaults by the user. 
    Only provides uniform prior compact support. 
    Format for priors is as [x_0_low, x_0_high, x_1_low, x_1_high,..., x_n_low, x_n_high]

    :param priors: priors in parameter order, ordered as [lower limit, upper limit] per parameter
    :type priors: array
    :param theta: current state, array in parameter order
    :type theta: array of floats
    
    :return: 0 if theta within priors, -inf if not

    """
    
    if len(priors) % 2 != 0:
        #if the priors do not have the right number of terms (ie, odd number)
        raise Exception("Size of prior range array is odd")
    
    for p in range(int(len(priors)/2)):
        #array indexes as 2p, 2p+1
        #if the parameter is NOT in the range, return lp = -np.inf
        #else return lp = 0
        if not (priors[2*p] < theta[p] < priors[(2*p)+1]):
            return -np.inf
        
    return 0.0 #everything checked out

def log_probability_singlepower_noCBV(theta, x, y, yerr, priors=None):
    """ 
    Calculates the log probability for the model with a single power law
    and a flat background. 
    
    Associated labels: ["t0", "A", "beta",  "b"]
    init_values in MCMC: np.array((min(disctime-3, x[-1]-2), 0.1, 1.8, 1))
    args = (x,y,yerr, (optionally) priors)

    :param theta: current state in parameter order
    :type theta: array of floats
    :param x: x-array of data (time)
    :param y: y-array of data (flux)
    :param yerr: error on y of data (err)
    :param priors: custom priors array, defaults to None

    :return: loglikelihood, logprior
    
    """
        
    t0, A, beta, b = theta
    # handle log priors
    if priors is None: #if you didn't feed it something else
        priors = [x[0], x[-1], 0.0, 30.0, 0.5, 6.0]
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
                                        Qall, CBV1, CBV2, CBV3, 
                                        priors=None):
    """ 
    Calculates the log probability for the model with a single power law and a complex CBV background. 
    
    Associated labels: ["t0", "A", "beta", "B","cQ", "c1", "c2", "c3"]
    init_values in MCMC: np.array((min(disctime-3, x[-1]-2), 0.1, 1.8, 0, 0,0,0,0))
    args = (x,y,yerr, Qall, CBV1, CBV2, CBV3, (optionally) priors)
    
    :param theta: current state in parameter order
    :type theta: array of floats
    :param x: x-array of data (time)
    :param y: y-array of data (flux)
    :param yerr: error on y of data (err)
    :param Qall: array of quaternion y-data
    :param CBV1, CBV2, CBV3: arrays of CBV data
    :param priors: custom priors array, defaults to None

    :return: loglikelihood, logprior
    
    """
    
    t0, A, beta, B, cQ, c1, c2, c3 = theta
    # handle log priors
    
    if priors is None: #if you didn't feed it something else
        priors = [x[0], x[-1], 0.0, 30.0, 0.5, 6.0, -30, 30, 
                  -30,30, -30,30, -30,30, -30,30] 
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
    """ 
    Calculates the log probability for the model with a double power law and a flat background. 
    
    Associated labels: ["t1", "t2", "a1", "a2", "beta1", "beta2",  "b"]
    init_values in MCMC: np.array((min(disctime-3, x[-1]-2), disctime, 0.1, 0.1, 1.8, 1.8, 1))
    args = (x,y,yerr, disctime, (optionally) priors)

    :param theta: current state in parameter order
    :type theta: array of floats
    :param x: x-array of data (time)
    :param y: y-array of data (flux)
    :param yerr: error on y of data (err)
    :param disctime: discovery time 
    :param priors: custom priors array, defaults to None

    :return: loglikelihood, logprior

    """
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
    """ 
    Calculates the log probability for the model with a double power law and a complex CBV background. 
    
    Associated labels: ["t1", "t2", "a1", "a2", "beta1", "beta2", "cQ", "c1", "c2", "c3"]
    init_values in MCMC: np.array((min(disctime-3, x[-1]-2), 0.1, 0.1, 1.8, 1.8, 0, 0,0,0,0))
    args = (x,y,yerr, Qall, CBV1, CBV2, CBV3, disctime, (optionally) priors)

    :param theta: current state in parameter order
    :type theta: array of floats
    :param x: x-array of data (time)
    :param y: y-array of data (flux)
    :param yerr: error on y of data (err)
    :param disctime: discovery time 
    :param priors: custom priors array, defaults to None

    :return: loglikelihood, logprior
    
    """
    def func1(x, t0, t1, A1, A2, beta1, beta2):
        return A1 *(x-t0)**beta1
    def func2(x, t0, t1, A1, A2, beta1, beta2):
        return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
    
    t0, t1, A1, A2, beta1, beta2, cQ, c1, c2, c3 = theta
    
    # handle log priors
    if priors is None: #if you didn't feed it something else
        priors = [x[0], disctime, t1, x[-1], 0, 5, 0, 5, 0.5, 6, 0.5, 6,
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

def log_probability_justCBV(theta, x, y, yerr, Qall, CBV1, CBV2, CBV3,
                            priors=None):
    """ 
    Calculates the log probability for a model that is just complex background
    
    Associated labels: ["b", "cQ", "c1", "c2", "c3"]
    init_values in MCMC: np.array((1, 0,0,0,0))
    args = (x,y,yerr, Qall, CBV1, CBV2, CBV3, (optionally) priors)
    
    :param theta: current state in parameter order
    :type theta: array of floats
    :param x: x-array of data (time)
    :param y: y-array of data (flux)
    :param yerr: error on y of data (err)
    :param Qall: array of quaternion y-data
    :param CBV1, CBV2, CBV3: arrays of CBV data
    :param disctime: discovery time 
    :param priors: custom priors array, defaults to None

    :return: loglikelihood, logprior

    """
    
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
        
def log_probability_singlePower_BG(theta, x, y, yerr, BGdata, priors=None):
    """ 
    Calculates the log probability for the model with a single power law
    and a background model taken from the data. Originally designed for
    fitting lygos-retrieved light curve backgrounds.
    
    Associated labels: [["t0", "A", "beta",  "b", "LBG"]
    init_values in MCMC: np.array((min(disctime-3, x[-1]-2), 0.1, 1.8, 1, 1))
    args = (x,y,yerr, BGdata, (optionally) priors)
    
    :param theta: current state in parameter order
    :type theta: array of floats
    :param x: x-array of data (time)
    :param y: y-array of data (flux)
    :param yerr: error on y of data (err)
    :param BGData: annulus data for background
    :param priors: custom priors array, defaults to None

    :return: loglikelihood, logprior
    
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
                 1 + b + BGdata * LBG)
        
        yerr2 = yerr**2.0
        return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp
  
def log_probability_singlepower_gaussianbeta(theta, x, y, yerr, mu,
                                             sigma, priors=None):
    """ 
    Calculates the log probability for the model with a single power law
    and a flat background with gaussian priors on beta
    
    Associated labels: ["t0", "A", "beta",  "b"]
    init_values in MCMC: np.array((min(disctime-3, x[-1]-2), 0.1, 1.8, 1))
    args = (x,y,yerr, (optionally) priors)

    :param theta: current state in parameter order
    :type theta: array of floats
    :param x: x-array of data (time)
    :param y: y-array of data (flux)
    :param yerr: error on y of data (err)
    :param mu: mu for gaussian
    :param sigma: sigma for gaussian
    :param priors: custom priors array, defaults to None

    :return: loglikelihood, logprior
    
    """
    
    t0, A, beta, b = theta
    newtheta = [t0,A]
    # handle log priors
    if priors is None: #if you didn't feed it something else
        priors = [x[0], x[-1], 0.0, 10.0]
    lp = check_priors(priors, newtheta)
    
    lp = lp + np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(beta-mu)**2/sigma**2

    # if not allowed values
    if not np.isfinite(lp) or np.isnan(lp): # if lp is not 0.0
        return -np.inf, lp
    else: # if allowed, calculate the log likelihood
        t1 = x - t0
        model = (np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta))) + 1 + b
        
        yerr2 = yerr**2.0
        return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp

def log_probability_celerite_mean(theta, y, gp):
    """ 
    Calculates the log probability for the celerite model
    
    Associated labels: ["sigma", "rho", "t0", "A", "beta",  "b"]
    args = (y, gp)

    :param theta: current state in parameter order
    :type theta: array of floats
    :param y: array of y data (flux)
    :param gp: gp object from celerite

    :return: loglikelihood, logprior

    """
    gp.set_parameter_vector(theta)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    return gp.log_likelihood(y, quiet=True) + lp

def log_probability_celerite_residual(theta, x, y, yerr, gp, priors=None):
    """
    Calculates the log probability for the model with a single power law
    and a celerite background. 
    
    Associated labels: ["t0", "A", "beta",  "b",  "logsigma", "logrho"]
    init_values in MCMC:  np.array((start_t, 0.1, 1.8, 0,np.log(1), np.log(2)))
    args = (x,y,yerr, gp, (optionally) priors)

    :param theta: current state in parameter order
    :type theta: array of floats
    :param x: time axis of data
    :param y: array of y data (flux)
    :param yerr: error on y of data (err)
    :param gp: gp object from celerite
    :param priors: custom priors array, defaults to None

    :return: loglikelihood, logprior

    """
    
    t0, A, beta, b = theta[:4]
    GPparams = theta[4:]
    #set up gp
    gp.set_parameter_vector(GPparams) #gp params set to current params
    lp = gp.log_prior() #gp log prior
    
    if priors is None: #if you didn't feed it something else
        priors = [x[0], x[-1], 0.001, 20, 0.5, 6]
    lp += check_priors(priors, theta[:4]) #ADD to gp prior function
    
    #if lp is no good   
    if not np.isfinite(lp):
        return -np.inf, -np.inf #ll, lp

    t1 = x - t0
    model = ((np.heaviside((t1), 1) * A * np.nan_to_num((t1**beta))) + 1 + b)
    
    residual = y - model

    ll = gp.log_likelihood(residual, quiet=True) #fit the GP to JUST the residual
    
    yerr2 = yerr**2.0
    ll += -0.5 * np.nansum((residual) ** 2 / yerr2 + np.log(yerr2))
    #if ll is no good
    if not np.isfinite(ll):
        return lp, -np.inf
    
    return ll+lp, lp

def log_probability_flat(theta, x, y, yerr):
    """ 
    Calculates the log probability for FLAT
    
    Associated labels: ["B"]
    init_values in MCMC: np.array(1)
    args = (x,y,yerr)
    
    :param theta: current state in parameter order
    :type theta: array of floats
    :param x: time axis of data
    :param y: array of y data (flux)
    :param yerr: error on y of data (err)
    :return: loglikelihood, logprior

    """
        
    B = theta
    model = np.zeros_like(x) + B
    yerr2 = yerr**2.0
    lp = 0
    return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2)), lp
