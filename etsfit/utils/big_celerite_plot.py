#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:59:16 2023

@author: lindseygordon
"""

import numpy as np
import matplotlib.pyplot as plt
from etsfit import etsMAIN
import etsfit.utils.utilities as ut
from pylab import rcParams
import os
import pandas as pd
from astropy.time import Time
from astropy.stats import SigmaClip
from scipy.stats import truncnorm
rcParams['figure.figsize'] = 8,3
rcParams['font.family'] = 'serif'



#run four sets of celerite possibilities and plot them. 
#2020tld w/ 0.6% max. 
file = "/Users/lindseygordon/research/urop/tessreduce_lc/2020tld2921/2020tld2921-tessreduce"
TNSFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"
save_dir = "./research/paper_outputs/celerite_plot/"

fig, ax = plt.subplots(3,4, figsize=(30,10), sharex=True)

# load stuff
(time, flux, error, targetlabel, 
 sector, camera, ccd) = ut.tr_load_lc(file)

(time, flux, error, bg, Q) = ut.fractionalfit(time, flux, error, bg=None, fraction=0.6, QCBVALL=None)

discoverytime = ut.get_disctime(TNSFile, targetlabel)

time -= 2457000
disctime = discoverytime - 2457000

#normal stuff
for i in range(4):
    ax[-1][i].set_xlabel("Time [BJD - 2457000]")
    ax[0][i].scatter(time, flux, s=1, c='k', label='Data', alpha=0.5)
    ax[0][i].axvline(disctime, linestyle='dashed', color='grey')
    ax[1][i].axvline(disctime, linestyle='dashed', color='grey')
    ax[2][i].axvline(disctime, linestyle='dashed', color='grey')

for i in range(3):
    ax[i][0].set_ylabel("Rel. Flux")
    
    
COLORS = ["c", "m"]


########################3
# set of plots 1: mean model w/ no bounds: 
############################3
from celerite.modeling import Model
from scipy.optimize import minimize
import celerite
from celerite import terms
import etsfit.utils.batch_analyze as ba

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
bounds_model_dict = {"t0":(0, time[-1]),
                      "A": (0.001, 20),
                      "beta":(0.5,6.0),
                      "b":(-50, 50)}



f1 = "./research/paper_outputs/celerite_plot/old_tld/celerite-matern32-mean-model-0.6/2020tld2921-celerite-matern32-mean-model-0.6-output-params.txt"
params, upper_e, lower_e, converg = ba.extract_celerite_all(f1)

# generate from params: 
ax[0][0].set_title("Celerite - Mean Model - Unbounded")

t0, A, beta, b = params[2:]

mean_model = MeanModel(t0=t0, A=A, beta=beta, b=b,
                        bounds = bounds_model_dict)

 
t1 = time - time[0] - t0
justmod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + b

kernel = terms.Matern32Term(log_rho=params[1], log_sigma=params[0])
gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
gp.set_parameter_vector(params)
gp.compute(time-time[0], error)
model = gp.predict(flux, time-time[0], return_cov=False)

ax[0][0].plot(time, model, lw=1, color=COLORS[0], label="Celerite")
ax[0][0].plot(time, justmod, lw=4, color=COLORS[1], label="Model")
ax[0][0].legend(fontsize=16)

ax[1][0].scatter(time, flux-justmod, s=1, c=COLORS[1], label='Model Residual')
ax[1][0].legend(fontsize=16)
ax[2][0].scatter(time, flux-model, s=1, c=COLORS[0], label='GP Residual')
ax[2][0].legend(fontsize=16)

########################3
# set of plots 2: mean model WITH bounds 
###########################
f2 = "/Users/lindseygordon/research/paper_outputs/celerite_plot/old_tld/celerite-matern32-mean-model-0.6-bounded/2020tld2921-celerite-matern32-mean-model-0.6-bounded-output-params.txt"
params, upper_e, lower_e, converg = ba.extract_celerite_all(f2)

ax[0][1].set_title("Celerite - Mean Model - Bounded")

t0, A, beta, b = params[2:]

mean_model = MeanModel(t0=t0, A=A, beta=beta, b=b,
                        bounds = bounds_model_dict)

justmod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + b


bounds_dict = {'log_sigma':np.log(np.sqrt((0.1,25  ))), 
                'log_rho':np.log((1,10))}
kernel = terms.Matern32Term(log_rho=params[1], log_sigma=params[0],
                            bounds=bounds_dict)

gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
gp.set_parameter_vector(params)
gp.compute(time-time[0], error)
model = gp.predict(flux, time-time[0], return_cov=False)

ax[0][1].plot(time, model, lw=4, color=COLORS[0], label="Celerite")
ax[0][1].plot(time, justmod, lw=4, color=COLORS[1], label="Model")
ax[0][1].legend(fontsize=16)

ax[1][1].scatter(time, flux-justmod, s=1, c=COLORS[1], label='Model Residual')
ax[1][1].legend(fontsize=16)
ax[2][1].scatter(time, flux-model, s=1, c=COLORS[0], label='GP Residual')
ax[2][1].legend(fontsize=16)


########################3
# set of plots 3: residual modeling withOUT bounds
###########################
f3 = "/Users/lindseygordon/research/paper_outputs/celerite_plot/old_tld/celerite-matern32-residual-0.6/2020tld2921-celerite-matern32-residual-0.6-output-params.txt"
params, upper_e, lower_e, converg = ba.extract_celerite_all(f3)

ax[0][2].set_title("Celerite - Residual - Unbounded")

t0, A, beta, b = params[0:4]
justmod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + b
residual = flux - justmod
#plot base model
ax[0][2].plot(time, justmod, lw=4, color=COLORS[1], label="Model")
ax[0][2].legend(fontsize=16)

#model residual
ax[1][2].scatter(time, residual, s=1, c=COLORS[1], label='Model Residual')



#GP stuff
kernel = terms.Matern32Term(log_sigma=params[4], log_rho=params[5])
gp = celerite.GP(kernel, mean=0.0)
gp.set_parameter_vector(params[4:])
gp.compute(time-time[0], error)
model = gp.predict(residual, time-time[0], return_cov=False)

ax[1][2].plot(time, model, lw=1, c=COLORS[0], label='GP Fit')
ax[1][2].legend(fontsize=16)
ax[2][2].scatter(time, residual-model, s=1, c=COLORS[0], label='GP Residual')
ax[2][2].legend(fontsize=16)

########################3
# set of plots 4: residual modeling with bounds
###########################
f4 = "/Users/lindseygordon/research/paper_outputs/celerite_plot/old_tld/celerite-matern32-residual-0.6-bounded/2020tld2921-celerite-matern32-residual-0.6-bounded-output-params.txt"
params, upper_e, lower_e, converg = ba.extract_celerite_all(f4)

ax[0][3].set_title("Celerite - Residual - Bounded")

t0, A, beta, b = params[0:4]
justmod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + b
residual = flux - justmod
#plot base model
ax[0][3].plot(time, justmod, lw=4, color=COLORS[1], label="Model")
ax[0][3].legend(fontsize=16)

#model residual
ax[1][3].scatter(time, residual, s=1, c=COLORS[1], label='Model Residual')



#GP stuff
sigma_bounds = np.log(np.sqrt((0.1,25))) #sigma range 0.316 to 4.47, take log
rho_bounds = np.log((1, 10)) #0, 2.302
bounds_dict = dict(log_sigma=sigma_bounds, log_rho=rho_bounds)
kernel = terms.Matern32Term(log_sigma=params[4], log_rho=params[5],
                            bounds = bounds_dict)
gp = celerite.GP(kernel, mean=0.0)
gp.set_parameter_vector(params[4:])
gp.compute(time-time[0], error)
model = gp.predict(residual, time-time[0], return_cov=False)

ax[1][3].plot(time, model, lw=4, c=COLORS[0], label='GP Fit')
ax[1][3].legend(fontsize=16)
ax[2][3].scatter(time, residual-model, s=1, c=COLORS[0], label='GP Residual')
ax[2][3].legend(fontsize=16)



plt.suptitle("2020tld - 60% Peak Flux", fontsize=30)
plt.tight_layout()
plt.savefig("/Users/lindseygordon/research/paper_outputs/celerite_plot/celerite_all.png")