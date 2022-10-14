#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:51:19 2022

@author: lindseygordon
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
#from etsfit import etsMAIN
from astropy.time import Time
import gc
import etsfit.utils.utilities as ut
import etsfit


lightcurveFolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
bigInfoFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"
foldersave = "/Users/lindseygordon/research/urop/plotOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"


info = pd.read_csv(bigInfoFile)

holder = "/Users/lindseygordon/research/urop/tessreduce_lc/2020tld2311/2020tld2311-tessreduce"
#print(holder)

(time, intensity, error, targetlabel, 
 sector, camera, ccd) = ut.tr_load_lc(holder)

#get discovery time
d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
#print(discoverytime)

def make_residual(x, y, best_mcmc):
    t0, A,beta,B = best_mcmc
    t1 = x - t0
    sl = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False)
    bg = np.ones(len(x)) + B
    return y - sl - bg
t = time - time[0]

#calculate residual from intermediate best:
y = make_residual(t, intensity, [15.31875494,3.57272544,1.32415633,6.94628453] )
plt.scatter(t, y)
plt.show()
plt.close()
print("created residual")



# now do gp

from tinygp import kernels, GaussianProcess
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)




def build_gp(theta, X):
    amps = jnp.exp(theta["log_amps"])
    scales = jnp.exp(theta["log_scales"])

    # Construct the kernel by multiplying and adding `Kernel` objects
    k1 = amps * kernels.ExpSquared(scales)
    

    return GaussianProcess(
        k1, X, 
        #diag=jnp.exp(theta["log_diag"]), m
        mean=theta["mean"]
    )


def neg_log_likelihood(theta, X, y):
    lp = 0
    #this is going to need priors somehow??
    gp = build_gp(theta, X)
    return -gp.log_probability(y) + lp


theta_init = {
    "mean": np.float64(0.0),
    #"log_diag": np.log(0.19),
    "log_amps": np.log(2),
    "log_scales": np.log(1),
}

# `jax` can be used to differentiate functions, and also note that we're calling
# `jax.jit` for the best performance.
obj = jax.jit(jax.value_and_grad(neg_log_likelihood))

print(f"Initial negative log likelihood: {obj(theta_init, t, y)[0]}")

print(
    f"Gradient of the negative log likelihood, wrt the parameters:\n{obj(theta_init, t, y)[1]}"
)

import jaxopt

solver = jaxopt.ScipyMinimize(fun=neg_log_likelihood)
soln = solver.run(theta_init, X=t, y=y)
print(f"Final negative log likelihood: {soln.state.fun_val}")

theta_init["log_amps"] = soln.params["log_amps"]
theta_init["log_scales"] = soln.params["log_scales"]

#then update theta params to this
#run another 1000 steps
#re-run the optimization and update them again