#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:23:28 2022

@author: lindseygordon
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from astropy.time import Time
import gc
#import tessreduce as tr
#import time
# from astroquery.mast import Tesscut
# from astropy.coordinates import SkyCoord
# from astropy import units as u
# import etsfit

from etsfit import etsMAIN
import etsfit.utils.utilities as ut


data_dir = "/Users/lindseygordon/research/urop/tessreduce_lc/"
cbv_dir = "/Users/lindseygordon/research/urop/eleanor_cbv/"
TNSFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"
save_dir = "/Users/lindseygordon/research/urop/paperOutput/"
quaternion_raw_dir = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_txt_dir = "/Users/lindseygordon/research/urop/quaternions-txt/"
gList = ["2018exc", "2018fhw", "2018fub", "2020tld", "2020zbo", "2018hzh", "2020hvq", 
          "2020hdw", "2020bj", "2019gqv"]



i=0
for root, dirs, files in os.walk(data_dir):
    for name in files:
        if name.endswith("-tessreduce") and i==0:
            fname = root + "/" + name

            (time, flux, error, targetlabel, 
             sector, camera, ccd) = ut.tr_load_lc(fname)

            if gList is not None and targetlabel not in gList:
                    continue
            #get discovery time
            discoverytime = ut.get_disctime(TNSFile, targetlabel)
            #run it
            trlc = etsMAIN(save_dir, TNSFile)
            
            trlc.load_single_lc(time, flux, error, discoverytime, 
                               targetlabel, sector, camera, ccd)
            
            winfilter = trlc.window_rms_filt(plot=False)
            
            if "2018fhw" in targetlabel:
                winfilter[1040:1080] = 0.0
            if "2020hdw" in targetlabel:
                winfilter[0:45] = 0.0
                winfilter[610:685] = 0.0
            
            
            
            
            i=i+1
         
from celerite.modeling import Model
#15.52351115  4.28574937  1.24650957  6.99656486
from scipy.optimize import minimize
import corner
import celerite
from celerite import terms
import emcee

x = time
y = flux
yerr = error
# Define the model
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
        dB = np.heaviside((t1), 1) * np.ones((3571,)) #np.heaviside((t1), 1) * 
        return np.array([dt, dA, dbeta, dB])
        
bdict = {'t0':(0,x[-1]), 'A':(0.001,20), 'beta':(0.5,6.0), 'b':(-50,50)}
mean_model = MeanModel(t0=15.5, A=4.2, beta=1.24, b=6.99, 
                       bounds=bdict)


# Set up the GP model

kbounds = {'log_sigma':np.log(np.sqrt((0.1,20  ))), 'log_rho':np.log((1,10))}

kernel = terms.Matern32Term(log_rho=np.log(2), log_sigma=np.log(1))#,bounds=kbounds)
gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
gp.compute(time, error)
print("Initial log-likelihood: {0}".format(gp.log_likelihood(flux)))

# Define a cost function
def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]

# Fit for the maximum likelihood parameters
initial_params = gp.get_parameter_vector()
bounds = gp.get_parameter_bounds()

print("running soln")
soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                method="L-BFGS-B", bounds=bounds, args=(y, gp))

gp.set_parameter_vector(soln.x)
# Make the maximum likelihood prediction
t = np.linspace(0, x[-1], 500)
mu, var = gp.predict(y, t, return_var=True)
std = np.sqrt(var)

# Plot the data
color = "#ff7f0e"
plt.errorbar(x, y, yerr=yerr, fmt=".k", alpha=0.2,capsize=0)

plt.plot(t, mu, lw=4, color=color)
plt.fill_between(t, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.title("maximum likelihood prediction")
plt.show()
plt.close()

def log_probability(params):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    return gp.log_likelihood(y, quiet=True) + lp



initial = np.array(soln.x)
ndim, nwalkers = len(initial), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

print("Running burn-in...")
p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 1000)

print("Running production...")
sampler.reset()
sampler.run_mcmc(p0, 10000);

# Plot the data.
plt.errorbar(x, y, yerr=yerr, fmt=".k",alpha=0.3, capsize=0)

# Plot 24 posterior samples.
samples = sampler.flatchain
for s in samples[np.random.randint(len(samples), size=24)]:
    gp.set_parameter_vector(s)
    mu = gp.predict(y, t, return_cov=False)
    plt.plot(t, mu, color=color, alpha=0.5)

plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
#plt.xlim(-5, 5)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.title("posterior predictions")
plt.show()
plt.close()

#%%
import etsfit.utils.snPlotting as sp
labels = ["sig", "rho", "t0", 'A', 'beta', 'b']
filesavetag = 'celeritetest'
save_dir = "/Users/lindseygordon/research/urop/paperOutput/"
sp.plot_corner(sampler.flatchain, labels, save_dir, targetlabel, filesavetag)
# names = gp.get_parameter_names()
# #cols = mean_model.get_parameter_names()
# #inds = np.array([names.index("mean:"+k) for k in cols])
# inds = np.arange(0,6,1)
# corner.corner(sampler.flatchain[:, inds],quantiles = [0.16, 0.5, 0.84],
# show_titles=True, title_fmt = ".3f", labels=["sig", "rho", "t0", 'A', 'beta', 'b'])

