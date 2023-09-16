#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 19:33:12 2023

gp plotting fxns

@author: lindseygordon
"""

import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
rcParams['figure.figsize'] = 16,6
rcParams["font.size"] = 20
rcParams["xtick.labelsize"] = 18
rcParams["ytick.labelsize"] = 18

from etsfit.utils import default_plots as sp

def plot_tinygp_ll(ets):
    """ 
    Plot the tinygp log likelihood per re-calculation as run concurrently 
    with the MCMC modelling
    ----------------------------------------
    Params:
        - ets (etsfit obj)
    """
    fig, ax1 = plt.subplots(figsize=(10,10))
    x = np.arange(0, len(ets.GP_LL_all), 1) * 1000 #x axis
    ax1.scatter(x, ets.GP_LL_all)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("GP Neg. Log Likelihood")
    ax1.set_title(ets.targetlabel + "  GP log likelihood over MCMC steps")
    plt.tight_layout()
    plt.savefig('{s}{t}{f}-GP-loglike-steps.png'.format(s=ets.save_dir,
                                                      t=ets.targetlabel,
                                                      f=ets.filesavetag))

    plt.close()
    return

def plot_mcmc_GP_celerite_mean(ets):
    """
    Plots the best fit model from MCMC with a celerite background
    Calls plot_2panel_model() and gp_plots()
    ---------------------------------------------
    Params:
        - ets (etsfit obj)
    """
    
    t0, A,beta,B = ets.best_mcmc[2:]
    t1 = ets.time - t0
    justmod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B
    
    ets.gp.set_parameter_vector(ets.best_mcmc)
    #gp.compute(time, error)
    model = ets.gp.predict(ets.flux, ets.time, return_cov=False)

    bg = model - justmod
    #fix time axis
    tplot = ets.time + ets.tmin - 2457000
    dplot = ets.disctime + ets.tmin - 2457000
    t0plot = t0 + ets.tmin - 2457000
    
    sp.plot_corner(ets)
    
    sp.plot_2panel_model(ets, tplot, model, dplot, t0plot)
    
    gp_plots(ets, tplot, justmod, dplot, t0plot, bg, 
             gpfiletag="MCMC-celerite-mean-TriplePlotResiduals")
    return

def plot_mcmc_GP_celerite_residual(ets):
    """
    Plots the best fit model from MCMC with a celerite background
    Calls plot_2panel_model() and gp_plots()
    ---------------------------------------------
    Params:
        - ets (etsfit object)
    """
    
    t0, A,beta,B = ets.best_mcmc[:4]
    t1 = ets.time - t0
    mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B
    
    ets.gp.set_parameter_vector(ets.best_mcmc[4:])
    bg = ets.gp.predict(ets.flux-mod, ets.time, return_cov=False)

    model = mod + bg
    #model = mod + bg
    #fix time axis
    tplot = ets.time + ets.tmin - 2457000
    dplot = ets.disctime + ets.tmin - 2457000
    t0plot = t0 + ets.tmin - 2457000
    
    sp.plot_corner(ets)
    
    sp.plot_2panel_model(ets, tplot, model, dplot, t0plot)
    
    gp_plots(ets, tplot, mod, dplot, t0plot, bg, 
             gpfiletag = "MCMC-celerite-residual-TriplePlotResiduals")
    return

def plot_mcmc_GP_tinygp(ets):
    """
    Plots the best fit model from MCMC with a tinygp background
    Calls plot_2panel_model() and gp_plots()
    ---------------------------------------------
    Params:
        - ets (etsfit obj)
        
    """

    t0, A,beta,B = ets.best_mcmc[:4]
    t1 = ets.time - t0
    mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B

    _, cond = ets.gp.condition(ets.flux-mod, ets.time)
    bg = cond.loc #mu
    #std = np.sqrt(cond.variance)

    model = mod + bg
    
    #fix time axis
    tplot = ets.time + ets.tmin - 2457000
    dplot = ets.disctime + ets.tmin - 2457000
    t0plot = t0 + ets.tmin - 2457000
    
    sp.plot_2panel_model(ets, tplot, model, dplot, t0plot)
    gp_plots(ets, tplot, model, dplot, t0plot, bg, 
             gpfiletag = "MCMC-tinyGP-TriplePlotResiduals")
             
    return

def gp_plots(ets, tplot, model, dplot, t0plot, bg, gpfiletag):
    """ 
    Produces three panel plot of output model including GP
    Top panel gives the data + power law, middle panel gives power law residual,
    bottom panel gives GP residual. 
    
    Also produces a histogram of the GP residual fluxes 
    ---------------------------------------------
    Params:
        - ets (etsfit obj)
        - tplot (time axis to plot)
        - model (power law model)
        - dplot (disc. time to plot)
        - t0plot (t0 to plot)
        - bg (separate background)
        - gpfiletag (str) tag added on to the end to indicate which gp model
            was used (ie, "MCMC-tinyGP-TriplePlotResiduals")
    """
    
    #second plot: 
    nrows = 3
    ncols = 1
    fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                   figsize=(8*ncols * 2, 3*nrows * 2))
    
    #top row: data, model ONLY
    ax[0].scatter(tplot, ets.flux, label = "Data", s = 3, color = 'black')
    ax[0].plot(tplot, model, label="Best Fit Model", lw=3,color = 'red')
    
    
    #middle row: residual, GP fit to residual
    residual1 = ets.flux - model
    ax[1].set_title("Model Residual")
    ax[1].scatter(tplot, residual1, label = "Model  Residual", s = 3, color = 'black')
    ax[1].plot(tplot, bg, label="GP", color="blue",lw=3, alpha=0.5)
    
    
    #bottom row: GP residual
    residual2 = residual1 - bg
    ax[2].set_title("GP Residual")
    ax[2].scatter(tplot, residual2, label = "GP Residual", s = 3, color = 'black')
    #all plots: t_0, disc, rel flux label 
    for n in range(nrows):
        ax[n].axvline(t0plot, color = 'green',lw=2, linestyle = 'dotted',
                          label=r"$t_0$")
        ax[n].axvline(dplot, color = 'grey', lw=2,linestyle = 'dotted', 
                      label="Ground Disc.")
        ax[n].set_ylabel("Flux (e-/s)", fontsize=20)
        ax[n].tick_params('y', labelsize=18)
        
    #plot labelling
    ax[0].set_title(ets.targetlabel + ets.filesavetag)
    ax[nrows-1].set_xlabel(ets.xlabel, fontsize=20)
    ax[nrows-1].tick_params('x', labelsize=18)
    
    ax[0].legend(fontsize=18, loc="upper left")
    ax[1].legend(fontsize=18)
    ax[2].legend(fontsize=18)
    
    plt.tight_layout()
    plt.savefig('{p}{t}{f}-{gpf}.png'.format(p=ets.save_dir,t=ets.targetlabel,
                                             f=ets.filesavetag,
                                             gpf=gpfiletag))

    plt.close()
    
    # residual histo
    fig, ax1 = plt.subplots(figsize=(10,10))
    n_in, bins, patches = ax1.hist(residual2, int(len(residual2)/25))
    ax1.set_xlabel("Rel. Flux")
    ax1.set_title("Histogram of GP Residual Flux")
    plt.tight_layout()
    plt.savefig('{p}{t}{f}-residual-histogram.png'.format(p=ets.save_dir,t=ets.targetlabel,
                                             f=ets.filesavetag))

    plt.close()
    return


def plot_celerite_tinygp_comp(ets):
    """ 
    Produces 2x3 plot comparing the no-GP, tinygp, and celerite fittings + 
    their residuals
    ---------------------------------------------
    Params:
        - ets
    """
    
    t0, A,beta,B = ets.best_mcmc[:4]
    t1 = ets.time - t0
    mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B
    
    ets.gpcelerite.set_parameter_vector(ets.best_mcmc[4:])
    celerite_bg, celerite_var = ets.gpcelerite.predict(ets.flux-mod, ets.time, 
                                                   return_var=True)
    #cel_std = np.sqrt(celerite_var)
    
    tinygp_bg = ets.gptinygp.predict(ets.flux-mod, ets.time, return_cov=False)

    #fix time axis stuff
    time = ets.time + ets.tmin - 2457000
    disctime = ets.disctime + ets.tmin - 2457000
    t0 = t0 + ets.tmin - 2457000
    
    #set up
    nrows = 3
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                                   figsize=(8*ncols, 3*nrows))
    
    #fill the data into all rows, disctimes, tmin, axis labels:
    ax[0][0].set_title("Flat Background", fontsize=20)
    ax[0][1].set_title("Residuals", fontsize=20)
    ax[1][0].set_title("celerite", fontsize=20)
    ax[2][0].set_title("tinygp", fontsize=20)
    
    
    ax[0][0].plot(time, mod, lw=3, label="Model", color = 'red')
    ax[0][1].scatter(time, ets.flux-mod, label="Residual", color='black', s=3)
    
    ax[1][0].plot(time, mod+celerite_bg, lw=3,label="Model + celerite", color='red')
    #ax[1][0].fill_between(time, mod+celerite_bg+cel_std, 
     #                     mod+celerite_bg-cel_std, color='pink', 
      #                    alpha=0.3,edgecolor="none", label="1 sigma")
    ax[1][1].scatter(time, ets.flux-mod-celerite_bg, label="Residual", color='black', s=3)
    
    ax[2][0].plot(time, mod+tinygp_bg, lw=3,label="Model + tinygp", color='red')
    ax[2][1].scatter(time, ets.flux-mod-tinygp_bg, label="Residual", color='black', s=3)
    
    ax[0][0].set_ylabel("Flux (e-/s)", fontsize=20)
    ax[1][0].set_ylabel("Flux (e-/s)", fontsize=20)
    ax[2][0].set_ylabel("Flux (e-/s)", fontsize=20)
    
    for n in range(ncols):
        #ax[1][n].axhline(0, color="orange", lw=2, label="Zero", linestyle='dotted')
        ax[nrows-1][n].set_xlabel(ets.xlabel, fontsize=20)
        
        for i in range(nrows):
            ax[i][0].scatter(time, ets.flux, label = "Data", s = 3, color = 'black')
            ax[i][n].axvline(t0, color = 'green', lw=2,linestyle = 'dotted',
                              label=r"$t_0$")
            ax[i][n].axvline(disctime, color = 'grey', lw=2,linestyle = 'dotted', 
                          label="Ground Disc.")
            ax[i][n].legend(fontsize=12, loc='upper left')
            ax[i][n].tick_params('y', labelsize=18)
    
    ax[nrows-1][0].tick_params('x', labelsize=18)
    ax[nrows-1][1].tick_params('x', labelsize=18)
    plt.tight_layout()
    plt.savefig('{p}{t}{f}-comparison-plot.png'.format(p=ets.save_dir,
                                                      t=ets.targetlabel,
                                                      f=ets.filesavetag))
    plt.close()
    
    fig, ax1 = plt.subplots(figsize=(10,10))
    n_in, bins, patches = ax1.hist(ets.flux-mod-tinygp_bg, int(len(ets.flux-mod-tinygp_bg)/25))
    ax1.set_xlabel("Rel. Flux")
    ax1.set_title("Histogram of GP Residual Flux")
    plt.tight_layout()
    plt.savefig('{p}{t}{f}-residual-histogram.png'.format(p=ets.save_dir,t=ets.targetlabel,
                                             f=ets.filesavetag))

    plt.close()
    return

def plot_scipy_max(ets):
    """ 
    Plots the maximum celerite fit using scipy maximization 
    ---------------------------------------------
    Params:
        - ets
        
    """
    x = ets.time + ets.tmin - 2457000
    disctime = ets.disctime + ets.tmin - 2457000
    t = ets.t + ets.tmin - 2457000
    y = ets.flux
    yerr = ets.error
    
    plt.errorbar(x, y, yerr=yerr, fmt=".k", alpha=0.2,capsize=0, label="Data")
    plt.plot(t, ets.mu, lw=4, color='green', label="Prediction")
    plt.fill_between(t, ets.mu+ets.std, ets.mu-ets.std, color='green', alpha=0.3, edgecolor="none")
    plt.axvline(disctime, color = 'grey', lw=2,linestyle = 'dotted', 
                  label="Ground Disc.")
    plt.ylabel("Flux")
    plt.xlabel("Time [BJD-2457000]")
    plt.title("{t}: Scipy max. likelihood prediction".format(t=ets.targetlabel))
    plt.legend()
    plt.tight_layout()
    plt.savefig('{s}{t}{f}-scipy-prediction.png'.format(s=ets.save_dir,
                                                        t=ets.targetlabel,
                                                        f=ets.filesavetag))
    #plt.show()
    plt.close()
    return


    
def celerite_post_pred(ets):
    """ 
    Plots the posterior predictions for celerite  mean model concurrent 
    ---------------------------------------------
    Params:
        - ets
        
    """
    x = ets.time + ets.tmin - 2457000
    disctime = ets.disctime + ets.tmin - 2457000
    tplot = ets.t + ets.tmin - 2457000
    
    plt.errorbar(x, ets.flux, yerr=ets.error, fmt=".k",alpha=0.3, capsize=0)
    plt.axvline(disctime, color = 'grey', lw=2,linestyle = 'dotted', 
                  label="Ground Disc.")
    # Plot 24 posterior samples.
    for s in ets.flat_samples[np.random.randint(len(ets.flat_samples), size=24)]:
        ets.gp.set_parameter_vector(s)
        mu = ets.gp.predict(ets.flux, tplot, return_cov=False)
        plt.plot(tplot, mu, color='green', alpha=0.5)

    plt.ylabel("Flux")
    plt.xlabel("Time [BJD-2457000]")
    plt.title("{t} posterior predictions".format(t=ets.targetlabel))
    plt.legend()
    plt.savefig('{s}{t}{f}-celerite-post-pred.png'.format(s=ets.save_dir,
                                                        t=ets.targetlabel,
                                                        f=ets.filesavetag))
    #plt.show()
    plt.close()
    return