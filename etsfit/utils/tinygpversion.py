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
import scipy
from scipy.optimize import minimize
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


solver2 = scipy.optimize.minimize(neg_log_likelihood, theta_init)


#%%
import scipy
from scipy.optimize import minimize, Bounds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from tinygp import kernels, GaussianProcess

file = "/Users/lindseygordon/research/urop/tessreduce_lc/2020tld2311/2020tld2311-tessreduce"

loadedraw = pd.read_csv(file)
time = Time(loadedraw["time"], format='mjd').jd
intensity = loadedraw["flux"].to_numpy()

t = time - time[0]
y = intensity

#write a function NOT dependent on a dictionary
def build_gp_nondict(theta, X):
    logamps, logscales, mean = theta
    k1 = np.exp(logamps) * kernels.Matern32(np.exp(logscales))
    return GaussianProcess(k1, X, mean=mean)

def fun(theta, X, y):
    """ theta = [amps, scales, mean]"""
    logamps, logscales, mean = theta
    lp = 0
    if mean > 20 or mean < -20:
        lp = np.inf
    gp = build_gp_nondict(theta, X)
    return -gp.log_probability(y) + lp

#x0 is array of initial parameters:
x0 = [np.log(2), np.log(1), 0.0]
#for celerite, bounds are logs of :
    #sigma_bounds = (0.01,0.35) (amps = sigma^2)
    # these bounds correspond to a sigma squared range of (21.2, 1.1)
    # so amps should be on that same range
    #rho_bounds = (1,3) (same as scales)
#bnds = ((np.log(1.1), np.log(21.2)), (np.log(1), np.log(3)), (-20, 20))
bounds = np.asarray([[np.log(1.1), np.log(1), -20], 
                             [np.log(21.2), np.log(3), 20]])
bnds = scipy.optimize.Bounds(lb=bounds[0], 
                             ub=bounds[1])

res = scipy.optimize.minimize(fun, x0, args=(t, y), bounds=bnds)

gp = build_gp_nondict(res.x, t)

tinygp_bg = gp.predict(y, t, return_cov=False)

plt.scatter(t, y, s = 2, color='black')
plt.plot(t, tinygp_bg, color = 'purple', alpha=0.2)



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