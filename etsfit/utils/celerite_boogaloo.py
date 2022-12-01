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
         

# tinygp boogaloo
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)



def mean_function(params, X):
    t1 = X - params['t0']
    mod = (
        jnp.heaviside((t1), 1) * params['A'] *jnp.nan_to_num((t1**params['beta']), copy=False)
        ) + params['b']
    return mod

mean_params = {
    't0': 15.0,
    'A': 5.0,
    'beta': 1.8,
    'b': 5.0    
}


random = np.random.default_rng(42)
X = time
y = flux

model = jax.vmap(partial(mean_function, mean_params))(X)

from tinygp import kernels, GaussianProcess


def build_gp(params, X, error):
    k1 = jnp.exp(params["log_sigma"]*2) * kernels.Matern32(jnp.exp(params["log_rho"]))
    return GaussianProcess(k1,X,diag=error, 
        mean=partial(mean_function, params))


@jax.jit
def loss(params, X, y, error):
    gp = build_gp(params, X, error)
    return -gp.log_probability(y)


params = dict(
    log_sigma=np.log(0.1),
    log_rho=np.log(3.0),
    #log_gp_diag=jnp.log(error),
    **mean_params
)
loss(params, X, y, error)

import jaxopt

solver = jaxopt.ScipyMinimize(fun=loss)
soln = solver.run(jax.tree_util.tree_map(jnp.asarray, params), bounds=None, 
                  X = X, y=y, error=error)
print(f"Final negative log likelihood: {soln.state.fun_val}")


gp = build_gp(soln.params, X, error)
_, cond = gp.condition(y, X)

mu = cond.loc
std = np.sqrt(cond.variance)

plt.plot(X, y, ".k", label="data")
plt.plot(X, mu, label="model")
plt.fill_between(X, mu + std, mu - std, color="C0", alpha=0.3)

plt.xlim(X.min(), X.max())
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
plt.close()

