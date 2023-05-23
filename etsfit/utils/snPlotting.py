# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:47:01 2021
Updated: Nov 21 2022

@author: Lindsey Gordon

Plotting functions for use in etsfit

Access docstrings by help(fxn_name)

"""
import matplotlib.pyplot as plt
import numpy as np
import corner
from pylab import rcParams
rcParams['figure.figsize'] = 16,6
rcParams["font.size"] = 20
rcParams["xtick.labelsize"] = 18
rcParams["ytick.labelsize"] = 18

def plot_autocorr_all(ets):
    """ 
    Produces two batches of autocorrelation plots by calling 
    plot_autocorr_individual()
    and 
    plot_autocorr_mean()
    -----------------------------------------
    Params:
        - ets (etsfit object)
    
    """
    plot_autocorr_mean(ets)
    plot_autocorr_individual(ets)
    return


def plot_autocorr_mean(ets):
    """ 
    Plots the mean autocorrelation time versus the number of steps
    Convergence threshold N/100 line is also plotted.
    ----------------------------------
    Params:
        - ets (etsfit object)
        
    """
    rcParams['figure.figsize'] = 8, 8
    n = ets.autoStep * np.arange(1, ets.index + 1) #x axis = total number of steps
    plotAutocorr = ets.autocorr[:ets.index]
    plt.plot(n, n / 100, "--k", label = "N/100 threshold") #plots the N vs N/100=tau threshold 
    #this determines length of chain vs autocorrelation time
    plt.plot(n, plotAutocorr, label="Autocorrelation")
    plt.xlim(0, n.max())
    plt.xlabel("Number of Steps", fontsize=20)
    plt.ylabel(r"$\hat{\tau}$", fontsize=20)
    plt.title(ets.targetlabel + r" $\hat{\tau}$" + " Converged={c}".format(c=ets.converged))
    plt.legend(loc="lower right")
    plt.tick_params('x', labelsize=18)
    plt.tick_params('y', labelsize=18)
    plt.tight_layout()
    plt.savefig(f"{ets.save_dir}{ets.targetlabel}{ets.filesavetag}-autocorr-mean.png")
    plt.close()
    rcParams['figure.figsize'] = 16,6
    return
    
def plot_autocorr_individual(ets):
    """
    Plot each parameter's autocorrelation time per step in an individual file
    -------------------------------------
    Params:
        - ets (etsfit object)
    """
    n = ets.autoStep * np.arange(1, ets.index + 1) #x axis - number of steps
    for i in range(len(ets.labels)):
        plotAutocorr = ets.autocorr_all[:ets.index,i]
        plt.plot(n, n / 100, "--k", label="N/100 Threshold") #plots the N vs N/100=tau threshold 
        #this determines length of chain vs autocorrelation time
        plt.plot(n, plotAutocorr, label="Autocorrelation")
        plt.xlim(0, n.max())
        plt.xlabel("Number of Steps", fontsize=20)
        plt.ylabel(r"$\hat{\tau}$ for " + ets.labels[i], fontsize=20)
        plt.title("{t} Autocorrelation Time for {l}".format(t=ets.targetlabel,
                                                   l = ets.labels[i]), fontsize=22)
        plt.tick_params('x', labelsize=18)
        plt.tick_params('y', labelsize=18)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{ets.save_dir}{ets.targetlabel}{ets.filesavetag}-autocorr-{ets.filelabels[i]}.png")
        plt.close()
        



def plot_corner(ets):
    """ 
    Produces corner plot using corner.py 
    --------------------------------------
    Params:
        - ets (etsfit object)
    """
    fig = corner.corner(ets.flat_samples, 
                        labels=ets.labels,
                        quantiles = [0.16, 0.5, 0.84],
                        show_titles=True, title_fmt = ".3f", 
                        title_kwargs={"fontsize": 20},
                        label_kwargs={'size':18})

    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=18)
        ax.yaxis.set_label_coords(-.5, .5)
        ax.xaxis.set_label_coords(0.5, -0.5)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)

    plt.tight_layout()
    fig.savefig(f"{ets.save_dir}{ets.targetlabel}{ets.filesavetag}-corner-plot-params.png")
    #plt.show()
    plt.close()
    return


    
def plot_param_samples_all(ets):
    """
    Plot all the parameter sampling per parameter from the chains in one big plot
    --------------------------------------
    Params:
        - ets (etsfit obj)
    """
    if ets.fitType == 20: 
        fig, ax = plt.subplots(1, figsize=(5,5))
        ax.hist(ets.flat_samples, 100, color='k', histtype='step')
        ax.set_xlabel(ets.labels[0], fontsize=20)
        ax.set_ylabel("p({pl})".format(pl=ets.labels[0]))
        ax.tick_params('y', labelsize=18)
        ax.tick_params('x', labelsize=18)
        
    else: 
        axN = len(ets.labels)
        if axN % 2 != 0:
            axN+=1 #make it even (two columns)
        nrows = int(axN/2)
        ncols = 2
        fig, ax = plt.subplots(nrows, ncols, sharex=False,
                               figsize=(ncols*4, nrows*4))
        p = 0
        for n in range(2):
            for m in range(nrows):
                if p < len(ets.labels):
                    ax[m, n].hist(ets.flat_samples[:, p], 100, color="k", histtype="step")
                    ax[m, n].set_xlabel(ets.labels[p], fontsize=20)
                    ax[m, n].set_ylabel("p({pl})".format(pl=ets.labels[p]))
                    ax[m,n].tick_params('y', labelsize=18)
                    ax[m,n].tick_params('x', labelsize=18)
                    p+=1
        
    #fig.suptitle("Chain Sampling By Parameter", fontsize=22, y=0.95)
    plt.tight_layout()
    plt.savefig(f"{ets.save_dir}{ets.targetlabel}{ets.filesavetag}-chain-samples-histograms.png")
    plt.close()
    return


def fitTypeModel(ets):
    """
    Produces the plotting model for a given fit type (1-7) 
    ------------------------------------------
    Params:
        - ets (etsfit obj)
    ------------------------------------------
    Returns:
        - mod (power law model)
        - bg (background model)
    """
    x = ets.time
    
    if ets.plotFit in (1,7):
        t0, A,beta,B = ets.best_mcmc[0]
        t1 = x - t0
        mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False)
        bg = np.ones(len(x)) + B
    elif ets.plotFit == 2: 
        t0, A, beta, B, cQ, cbv1, cbv2, cbv3 = ets.best_mcmc[0]
        t1 = x - t0
        Qall, CBV1, CBV2, CBV3 = ets.quats_cbvs
        mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta))
        bg = cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3 + 1 + B
        
    elif ets.plotFit == 3:
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        t0, t1, A1, A2, beta1, beta2, B = ets.best_mcmc[0]
        mod = np.piecewise(x, [(t0 <= x)*(x < t1), t1 <= x], 
                             [func1, func2],
                             t0, t1, A1, A2, beta1, beta2) 
        bg = np.ones(len(x)) + B
    elif ets.plotFit ==4:
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        
        Qall, CBV1, CBV2, CBV3 = ets.quats_cbvs
        t0, t1, A1, A2, beta1, beta2, cQ, cbv1, cbv2, cbv3 = ets.best_mcmc[0]
        mod = np.piecewise(x, [(t0 <= x)*(x < t1), t1 <= x], 
                             [func1, func2],
                             t0, t1, A1, A2, beta1, beta2)
        bg = cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3 + 1
        
    elif ets.plotFit ==5:
        Qall, CBV1, CBV2, CBV3 = ets.quats_cbvs
        b, cQ, cbv1, cbv2, cbv3 = ets.best_mcmc[0]
        mod = np.zeros(len(x))
        bg = (b + np.ones(len(x)) + cQ * Qall + 
              cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3)
        print("background starts with", bg[0:5])
    
    elif ets.plotFit == 6:
        t0, A,beta,B, LBG = ets.best_mcmc[0]
        t1 = x - t0
        mod = (np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta)))
        bg = 1 + B + ets.BGdata * LBG
    else:
        raise ValueError("No valid plot Fit value given")
    
    return mod, bg

def plot_chain_logpost(ets, appendix=""):
    """
    Plots MCMC chain trace plots for all parameters and the log posterior
    --------------------------------------
    Params:
        - ets (etsfit object)
        - appendix (str) tail end string for the filename - usually "burnin" 
                or "production"
    
    """
    fig, axes = plt.subplots(ets.ndim+1, figsize=(10, 7), sharex=True)
    samples = ets.sampler.get_chain()
    logprobs = ets.sampler.get_log_prob()
    logprior = ets.sampler.get_blobs()
    logpost = logprobs+logprior
    xaxis = np.linspace(1,len(logpost), len(logpost[:,0]))
    
    for h in range(len(logpost[0])):
        ax = axes[0]
        ax.scatter(xaxis, logpost[:,0], alpha=0.3, color='black', s=2)
        ax.set_ylabel("Log Post.", fontsize=16)
        ax.set_title("MCMC Chain Traces", fontsize=22)
        ax.tick_params('y', labelsize=18)
    
    for i in range(ets.ndim):
        ax = axes[i+1]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(ets.labels[i], fontsize=20)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.tick_params('y', labelsize=18)
    
    axes[-1].set_xlabel("Step Number", fontsize=20)
    axes[-1].tick_params(axis='x', labelsize=18)
    plt.tight_layout()
    plt.savefig(f"{ets.save_dir}{ets.targetlabel}{ets.filesavetag}-chain-logpost-{appendix}.png")
    #plt.show()
    plt.close()
    return

def plot_chain(ets, appendix=""):
    """
    Plots MCMC chain trace plots for all parameters and the log posterior
    --------------------------------------
    Params:
        - ets (etsfit obj)
        - appendix (str) tail end string for the filename - usually "burnin" 
                or "production"
    
    """
    fig, axes = plt.subplots(ets.ndim+1, figsize=(10, 7), sharex=True)
    samples = ets.sampler.get_chain()
    logprobs = ets.sampler.get_log_prob()
    #logprior = sampler.get_blobs()
    #logpost = logprobs+logprior
    xaxis = np.linspace(1,len(logprobs), len(logprobs[:,0]))
    
    for h in range(len(logprobs[0])):
        ax = axes[0]
        ax.scatter(xaxis, logprobs[:,0], alpha=0.3, color='black', s=2)
        ax.set_ylabel("Log Prob.", fontsize=16)
        ax.set_title("MCMC Chain Traces", fontsize=22)
        ax.tick_params('y', labelsize=18)
    
    for i in range(ets.ndim):
        ax = axes[i+1]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(ets.labels[i], fontsize=20)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.tick_params('y', labelsize=18)
    
    axes[-1].set_xlabel("Step Number", fontsize=20)
    axes[-1].tick_params(axis='x', labelsize=18)
    plt.tight_layout()
    plt.savefig('{s}{t}{f}-chain-{a}.png'.format(s=ets.save_dir,
                                                 t=ets.targetlabel,
                                                 f=ets.filesavetag,
                                                 a=appendix))
    #plt.show()
    plt.close()
    return


def plot_histogram(data, bins, x_label, y_label, filename):
    """ 
    Plot a simple histogram
    ----------------------------------------------
    Params:
        - data (array) histogram info
        - bins (int) how many bins
        - x_label (str)
        - y_label (str)
        - filename (str or None) full path to save into
    """
    fig, ax1 = plt.subplots(figsize=(10,10))
    n_in, bins, patches = ax1.hist(data, bins)
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)
    
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)

    plt.close()
    return 

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

def plot_mcmc(ets):
    """
    Main plotting function: produces the corner plot and MCMC best fit model
    plot for a given fitting run
    ----------------------------------------------
    Params:
        - ets (etsfit object)
    """
    #set up model = sl + bg but can be plot separately.
    t0 = ets.best_mcmc[0][0]
    sl, bg = fitTypeModel(ets)
    model = sl + bg   
    #fix time axis for plotting
    tplot = ets.time + ets.tmin - 2457000
    dplot = ets.disctime + ets.tmin - 2457000
    t0plot = t0 + ets.tmin - 2457000
    
    
    #plot corner
    plot_corner(ets)
    #plot model
    plot_2panel_model(ets, tplot, model, dplot, t0plot)
    
    return

def plot_mcmc_GP_celerite_mean(ets):
    """
    Plots the best fit model from MCMC with a celerite background
    Calls plot_2panel_model() and gp_plots()
    ---------------------------------------------
    Params:
        - ets (etsfit obj)
    """
    
    t0, A,beta,B = ets.best_mcmc[0][2:]
    t1 = ets.time - t0
    justmod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B
    
    ets.gp.set_parameter_vector(ets.best_mcmc[0])
    #gp.compute(time, error)
    model = ets.gp.predict(ets.flux, ets.time, return_cov=False)

    bg = model - justmod
    #fix time axis
    tplot = ets.time + ets.tmin - 2457000
    dplot = ets.disctime + ets.tmin - 2457000
    t0plot = t0 + ets.tmin - 2457000
    
    plot_corner(ets)
    
    plot_2panel_model(ets, tplot, model, dplot, t0plot)
    
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
    
    t0, A,beta,B = ets.best_mcmc[0][:4]
    t1 = ets.time - t0
    mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B
    
    ets.gp.set_parameter_vector(ets.best_mcmc[0][4:])
    bg = ets.gp.predict(ets.flux-mod, ets.time, return_cov=False)

    model = mod + bg
    #model = mod + bg
    #fix time axis
    tplot = ets.time + ets.tmin - 2457000
    dplot = ets.disctime + ets.tmin - 2457000
    t0plot = t0 + ets.tmin - 2457000
    
    plot_corner(ets)
    
    plot_2panel_model(ets, tplot, model, dplot, t0plot)
    
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

    t0, A,beta,B = ets.best_mcmc[0][:4]
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
    
    plot_2panel_model(ets, tplot, model, dplot, t0plot)
    gp_plots(ets, tplot, model, dplot, t0plot, bg, 
             gpfiletag = "MCMC-tinyGP-TriplePlotResiduals")
             
    return

def plot_2panel_model(ets, tplot, model, dplot, t0plot):
    """ 
    Produces two panel plot of output model. 
    Top panel gives the data + model, bottom panel gives the residual. 
    ---------------------------------------------
    Params:
        - ets (etsfit obj)
        - tplot (time axis to plot)
        - model (complete model)
        - dplot (disc. time to plot)
        - t0plot (t0 to plot)
    """
    nrows = 2
    ncols = 1
    fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                   figsize=(8*ncols * 2, 3*nrows * 2))
    
    #plot model, data
    ax[0].plot(tplot, model, label="Best Fit Model", lw=3, color = 'red')
    ax[0].scatter(tplot, ets.flux, label = "Data", s = 3, color = 'black')
    
    for n in range(nrows):
        axy = ax[n]
        axy.axvline(t0plot, color = 'green', lw=2, linestyle = 'dotted',
                          label=r"$t_0$")
        axy.axvline(dplot, color = 'grey', lw=2, linestyle = 'dotted', 
                      label="Ground Disc.")
        axy.set_ylabel("Flux (e-/s)", fontsize=20)
        axy.tick_params('y', labelsize=18)
        
    #main
    ax[0].set_title(ets.targetlabel + ets.filesavetag)
    ax[0].legend(fontsize=18, loc="upper left")
    ax[nrows-1].set_xlabel("Time [BJD - 2457000]")
    ax[nrows-1].tick_params('x', labelsize=18)
    
    #residuals
    ax[1].set_title("Residual")
    residuals = ets.flux - model
    ax[1].scatter(tplot, residuals, s=3, color = 'black', label='Residual, All')
    ax[1].legend(fontsize=18)
    
    plt.tight_layout()
    plt.savefig('{p}{t}{f}-MCMCmodel-bestFit.png'.format(p=ets.save_dir,
                                                      t=ets.targetlabel,
                                                      f=ets.filesavetag))
    plt.close()
    
    fig, ax1 = plt.subplots(figsize=(10,10))
    n_in, bins, patches = ax1.hist(residuals, int(len(residuals)/25))
    ax1.set_xlabel("Rel. Flux")
    ax1.set_title("Histogram of Residual Flux")
    plt.tight_layout()
    plt.savefig('{p}{t}{f}-residual-histogram.png'.format(p=ets.save_dir,
                                                          t=ets.targetlabel,
                                                          f=ets.filesavetag))

    plt.close()
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
    
    t0, A,beta,B = ets.best_mcmc[0][:4]
    t1 = ets.time - t0
    mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B
    
    ets.gpcelerite.set_parameter_vector(ets.best_mcmc[0][4:])
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

def plot_all_in_folder(data_dir, file_TNS):
    """ 
    Plots all tessreduce light curves saved in a given directory w/
    their discovery time
    """
    import os
    import etsfit.util.utilities as ut
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith("-tessreduce"):
                holder = root + "/" + name
                #print(holder)
                (time, flux, error, 
                  targetlabel, sector, 
                  camera, ccd) = ut.tr_load_lc(holder)
                fig, ax = plt.subplots(1, figsize=(4, 1.5))
                ax.scatter(time, flux, s=2, color='black')
                dd_ = ut.get_disctime(file_TNS, targetlabel)
                ax.axvline(dd_, color='red')
                ax.set_title(targetlabel)
                plt.tight_layout()
    return