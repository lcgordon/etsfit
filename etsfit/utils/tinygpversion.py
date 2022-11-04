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
rcParams['figure.figsize'] = 8,3

file = "/Users/lindseygordon/research/urop/tessreduce_lc/2020tld2921/2020tld2921-tessreduce"

loadedraw = pd.read_csv(file)
time = Time(loadedraw["time"], format='mjd').jd
intensity = loadedraw["flux"].to_numpy()
error  = loadedraw["flux_err"].to_numpy()

t = time - time[0]
y = intensity


plt.scatter(t, y, s=2)
plt.xlim(t.min(), t.max())
plt.xlabel("JD")
plt.ylabel("rel flux")

t0 = 15.58
A = 4.55
beta = 1.22
B = 6.99

t1 = t - t0
model = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + B

plt.plot(t, model)
from celerite.modeling import Model
import celerite
from celerite import terms

log_sigma = 1.4975
log_rho = 0.0

kernel = terms.Matern32Term(log_sigma=log_sigma, log_rho=log_rho,
                            eps=1e-6)
gp = celerite.GP(kernel, mean=0.0)
gp.compute(t, error)
cel_bg, cel_var = gp.predict(intensity-model, t, return_var = True)

plt.plot(t, model+cel_bg, label="celerite")

#test the output on tinygp kernel w/ same params: 
import jax
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
jax.config.update("jax_enable_x64", True)
#import tinygp.kernels.distance.Distance

k = np.exp(log_sigma)**2 * kernels.Matern32(scale=1, 
                                            distance= kernels.distance.L2Distance())
gptiny = GaussianProcess(k, time, mean=0.0, diag = error)
#tinygp_bg = gptiny.predict(intensity-model, time, return_cov=False)
cond_gp = gptiny.condition(intensity-model, time).gp
mu, var = cond_gp.loc, cond_gp.variance

plt.plot(t, model+mu, label="tinygp")

plt.legend()

#%%


    
    





#%%


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

    gp = GaussianProcess(kernel, X, mean=theta["mean"])
    
    return gp


def neg_log_likelihood(theta, X, y):

    gp = build_gp(theta, X)
    return -gp.log_probability(y)


theta_init = {
    "mean": np.float64(0.0),
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

bounds = np.asarray([[np.log(1.1), np.log(1), -20], 
                             [np.log(21.2), np.log(3), 20]])

soln = solver.run(theta_init, bounds, X=t, y=y)
print(f"Final negative log likelihood: {soln.state.fun_val}")

print(soln.params)


#%%
#kernel function plotting

import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
rcParams['figure.figsize'] = 10,10

def matern_32_rho(path):
    x = np.arange(0, 10, 0.1)
    l = [0.01, 0.1, 1, 2]
    
    for n in range(len(l)):
        k = (1+np.sqrt(3)*x/l[n])*np.exp(-np.sqrt(3)*x/l[n])
        plt.plot(x, k, label=l[n])
        
    plt.legend()
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"K($\tau$)")
    plt.title(r"Matern 3/2 with varying $\rho$")
    plt.tight_layout()
    plt.savefig("{p}Matern-32-rho.png".format(p=path))
    
# repeat for variance in sigma squared:
def matern_32_sigsq(path):
    x = np.arange(0, 10, 0.1)
    l = 1
    sig = [0.5, 1, 2, 5, 10]
    
    for n in range(len(sig)):
        k = sig[n] * (1+np.sqrt(3)*x/l)*np.exp(-np.sqrt(3)*x/l)
        plt.plot(x, k, label=sig[n])
        
    plt.legend()
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"K($\tau$)")
    plt.title(r"Matern 3/2 with varying $\sigma^2$")
    plt.tight_layout()
    plt.savefig("{p}Matern-32-sigsq.png".format(p=path))

# repeat for variance in both
def matern_32_bothvary(path):
    x = np.arange(0, 10, 0.1)
    l = [0.01, 0.1, 1, 2]
    sig = [0.5, 1, 2]
    
    for n in range(len(sig)):
        for i in range(len(l)):
            k = sig[n] * (1+np.sqrt(3)*x/l[i])*np.exp(-np.sqrt(3)*x/l[i])
            plt.plot(x, k, label=r"$\sigma^2$:{s} $\rho$:{r}".format(s=sig[n],
                                                                 r = l[i]))
        
    plt.legend()
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"K($\tau$)")
    plt.title(r"Matern 3/2 with varying $\sigma^2, \rho$ ")
    plt.tight_layout()
    plt.savefig("{p}Matern-32-bothvary.png".format(p=path))

matern_32_bothvary("/Users/lindseygordon/research/urop/plotOutput/")