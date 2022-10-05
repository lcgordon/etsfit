#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 07:52:26 2022

@author: lindseygordon
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time

file = "/Users/lindseygordon/research/urop/tessreduce_lc/2020tld2311/2020tld2311-tessreduce"

loadedraw = pd.read_csv(file)
time = Time(loadedraw["time"], format='mjd').jd
intensity = loadedraw["flux"].to_numpy()

t = time - time[0]
y = intensity


plt.plot(t, y, ".k")
plt.xlim(t.min(), t.max())
plt.xlabel("JD")
_ = plt.ylabel("rel flux")


import jax
import jax.numpy as jnp

from tinygp import kernels, GaussianProcess


jax.config.update("jax_enable_x64", True)


def build_gp(theta, X):

    # We want most of our parameters to be positive so we take the `exp` here
    # Note that we're using `jnp` instead of `np`
    amps = jnp.exp(theta["log_amps"])
    scales = jnp.exp(theta["log_scales"])

    # Construct the kernel by multiplying and adding `Kernel` objects
    k1 = amps[0] * kernels.ExpSquared(scales[0])
    
    kernel = k1

    return GaussianProcess(
        kernel, X, diag=jnp.exp(theta["log_diag"]), mean=theta["mean"]
    )


def neg_log_likelihood(theta, X, y):
    gp = build_gp(theta, X)
    return -gp.log_probability(y)


theta_init = {
    "mean": np.float64(0.0),
    "log_diag": np.log(0.19),
    "log_amps": np.log([26.0]),
    "log_scales": np.log([5.0]),
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

print(soln.params)