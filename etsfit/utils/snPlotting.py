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



def plot_autocorr_mean(save_dir, targetlabel, index, autocorr, converged,
                       autoStep, filesavetag):
    """ 
    Plots the mean autocorrelation time versus the number of steps
    Convergence threshold N/100 line is also plotted.
    ----------------------------------
    Params:
        - save_dir (str) folder to save plot into
        - targetlabel (str) target ID
        - index (int) number of autocorreleation checks that the chain performed
        - autocorr (array) mean autocorrelation at each autoStep convergence check
        - converged (bool) if the chain converged
        - autoStep (int) number of chain steps between convergence checks
        - filesavetag (str) filename used for this fit
        
    """
    rcParams['figure.figsize'] = 8, 8
    n = autoStep * np.arange(1, index + 1) #x axis = total number of steps
    plotAutocorr = autocorr[:index]
    plt.plot(n, n / 100, "--k", label = "N/100 threshold") #plots the N vs N/100=tau threshold 
    #this determines length of chain vs autocorrelation time
    plt.plot(n, plotAutocorr, label="Autocorrelation")
    plt.xlim(0, n.max())
    plt.xlabel("Number of Steps", fontsize=20)
    plt.ylabel(r"$\hat{\tau}$", fontsize=20)
    plt.title(targetlabel + r" $\hat{\tau}$" + " Converged={c}".format(c=converged))
    plt.legend(loc="lower right")
    plt.tick_params('x', labelsize=18)
    plt.tick_params('y', labelsize=18)
    plt.tight_layout()
    plt.savefig("{s}{t}{f}-autocorr-mean.png".format(s=save_dir,
                                                      t=targetlabel,
                                                      f = filesavetag))
    plt.close()
    rcParams['figure.figsize'] = 16,6
    return
    
def plot_autocorr_individual(save_dir, targetlabel, index, autocorr_all,
                             autoStep, labels, filelabels, filesavetag):
    """
    Plot each parameter's autocorrelation time per step in an individual file
    -------------------------------------
    Params:
        - save_dir (str) folder to save plot into
        - targetlabel (str) target ID
        - index (int) number of autocorreleation checks that the chain performed
        - autocorr_all (array of size (params,index)) contains all 
            parameters' autocorrelation times
        - autoStep (int) number of chain steps between convergence checks
        - labels (array of strings) param names, may use latex formatting
        - filelabels (array of strings) param names, formatted for use in file naming
        - filesavetag (string) filename used for this fit
    """
    n = autoStep * np.arange(1, index + 1) #x axis - number of steps
    for i in range(len(labels)):
        plotAutocorr = autocorr_all[:index,i]
        plt.plot(n, n / 100, "--k", label="N/100 Threshold") #plots the N vs N/100=tau threshold 
        #this determines length of chain vs autocorrelation time
        plt.plot(n, plotAutocorr, label="Autocorrelation")
        plt.xlim(0, n.max())
        plt.xlabel("Number of Steps", fontsize=20)
        plt.ylabel(r"$\hat{\tau}$ for " + labels[i], fontsize=20)
        plt.title("{t} Autocorrelation Time for {l}".format(t=targetlabel,
                                                   l = labels[i]), fontsize=22)
        plt.tick_params('x', labelsize=18)
        plt.tick_params('y', labelsize=18)
        plt.legend()
        plt.tight_layout()
        plt.savefig("{s}{t}{f}-autocorr-{fl}.png".format(s=save_dir,
                                                          t=targetlabel,
                                                          f = filesavetag,
                                                          fl = filelabels[i]))
        plt.close()
        
def plot_autocorr_all(save_dir, targetlabel, index, autocorr, 
                      autocorr_all, converged,
                      autoStep, labels, filelabels, filesavetag):
    """ 
    Produces two batches of autocorrelation plots by calling 
    plot_autocorr_individual()
    and 
    plot_autocorr_mean()
    -----------------------------------------
    Params:
        - save_dir (str) folder to save plot into
        - targetlabel (str) target ID
        - index (int) number of autocorreleation checks that the chain performed
        - autocorr (array) mean autocorrelation at each autoStep convergence check
        - autocorr_all (array of size (params,index)) contains all 
            parameters' autocorrelation times
        - converged (bool) if the chain converged
        - autoStep (int) number of chain steps between convergence checks
        - labels (array of strings) param names, may use latex formatting
        - filelabels (array of strings) param names, formatted for use in file naming
        - filesavetag (string) filename used for this fit
    
    """
    plot_autocorr_mean(save_dir, targetlabel, index, autocorr, converged,
                           autoStep, filesavetag)
    plot_autocorr_individual(save_dir, targetlabel, index, autocorr_all,
                                 autoStep, labels, filelabels, filesavetag)
    return


def plot_corner(flat_samples, labels, save_dir, targetlabel, filesavetag):
    """ 
    Produces corner plot using corner.py 
    --------------------------------------
    Params:
        - flat_samples (array) returned from emcee sampler
        - labels (array of strings) param names, may use latex formatting
        - save_dir (str) folder to save plot into
        - targetlabel (str) target ID
        - filesavetag (str) filename used for this fit
    """
    fig = corner.corner(flat_samples, 
                        labels=labels,
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
    fig.savefig('{s}{t}{f}-corner-plot-params.png'.format(s=save_dir,
                                                      t=targetlabel,
                                                      f=filesavetag))
    #plt.show()
    plt.close()
    return



def plot_param_samples_all(flat_samples, labels, save_dir, targetlabel, filesavetag):
    """
    Plot all the parameter sampling per parameter from the chains in one big plot
    --------------------------------------
    Params:
        - flat_samples (array) returned from emcee sampler
        - labels (array of strings) param names, may use latex formatting
        - save_dir (str) folder to save plot into
        - targetlabel (str) target ID
        - filesavetag (str) filename used for this fit
    """
    
    axN = len(labels)
    if axN % 2 != 0:
        axN+=1 #make it even (two columns)
    nrows = int(axN/2)
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols, sharex=False,
                           figsize=(ncols*4, nrows*4))
    p = 0
    for n in range(2):
        for m in range(nrows):
            if p < len(labels):
                ax[m, n].hist(flat_samples[:, p], 100, color="k", histtype="step")
                ax[m, n].set_xlabel(labels[p], fontsize=20)
                ax[m, n].set_ylabel("p({pl})".format(pl=labels[p]))
                ax[m,n].tick_params('y', labelsize=18)
                ax[m,n].tick_params('x', labelsize=18)
                p+=1
    
    #fig.suptitle("Chain Sampling By Parameter", fontsize=22, y=0.95)
    plt.tight_layout()
    plt.savefig('{s}{t}{f}-chainHisto-all.png'.format(s=save_dir,
                                                      t=targetlabel,
                                                      f=filesavetag))
    plt.close()
    return
    


def fitTypeModel(fitType, x, best_mcmc, QCBVs = None, lygosBG = None):
    """
    Produces the plotting model for a given fit type (1-7) 
    ------------------------------------------
    Params:
        - fitType (int) 1-7, which fit is being produced
        - x (array) x-axis, should start at 0
        - best_mcmc (array) params for best fit model
        - QCBVs (either an array [Qall, CBV1, CBV2, CBV3] or just None)
                only for fit types 2,4,5
        - lygosBG (array of size x, or None) 
                only for fit type 6
    ------------------------------------------
    Returns:
        - mod (power law model)
        - bg (background model)
    """
    if fitType in (2,4,5) and QCBVs is None:
        return ValueError("QCBVs must be loaded in for fit types 2,4,5!")
    
    if fitType in (1,7):
        t0, A,beta,B = best_mcmc
        t1 = x - t0
        mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False)
        bg = np.ones(len(x)) + B
    elif fitType == 2: 
        t0, A, beta, B, cQ, cbv1, cbv2, cbv3 = best_mcmc
        t1 = x - t0
        Qall, CBV1, CBV2, CBV3 = QCBVs
        mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta))
        bg = cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3 + 1 + B
        
    elif fitType == 3:
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        t0, t1, A1, A2, beta1, beta2, B = best_mcmc
        mod = np.piecewise(x, [(t0 <= x)*(x < t1), t1 <= x], 
                             [func1, func2],
                             t0, t1, A1, A2, beta1, beta2) 
        bg = np.ones(len(x)) + B
    elif fitType ==4:
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        
        Qall, CBV1, CBV2, CBV3 = QCBVs
        t0, t1, A1, A2, beta1, beta2, cQ, cbv1, cbv2, cbv3 = best_mcmc#[0]
        mod = np.piecewise(x, [(t0 <= x)*(x < t1), t1 <= x], 
                             [func1, func2],
                             t0, t1, A1, A2, beta1, beta2)
        bg = cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3 + 1
        
    elif fitType ==5:
        Qall, CBV1, CBV2, CBV3 = QCBVs
        b, cQ, cbv1, cbv2, cbv3 = best_mcmc#[0]
        mod = np.zeros(len(x))
        bg = (b + np.ones(len(x)) + cQ * Qall + 
              cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3)
        print("background starts with", bg[0:5])
    
    elif fitType == 6:
        t0, A,beta,B, LBG = best_mcmc
        t1 = x - t0
        mod = (np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta)))
        bg = 1 + B + lygosBG * LBG
    
    return mod, bg

def plot_chain_logpost(save_dir, targetlabel, filesavetag, sampler, labels, ndim,
                       appendix=""):
    """
    Plots MCMC chain trace plots for all parameters and the log posterior
    --------------------------------------
    Params:
        - save_dir (str) folder to save plot into
        - targetlabel (str) target ID
        - filesavetag (str) filename used for this fit
        - sampler (emcee obj) the sampler object produced by emcee
        - labels (array of strings) param names, may use latex formatting
        - ndim (int) number of parameters
        - appendix (str) tail end string for the filename - usually "burnin" 
                or "production"
    
    """
    fig, axes = plt.subplots(ndim+1, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    logprobs = sampler.get_log_prob()
    logprior = sampler.get_blobs()
    logpost = logprobs+logprior
    xaxis = np.linspace(1,len(logpost), len(logpost[:,0]))
    
    for h in range(len(logpost[0])):
        ax = axes[0]
        ax.scatter(xaxis, logpost[:,0], alpha=0.3, color='black', s=2)
        ax.set_ylabel("Log Post.", fontsize=16)
        ax.set_title("MCMC Chain Traces", fontsize=22)
        ax.tick_params('y', labelsize=18)
    
    for i in range(ndim):
        ax = axes[i+1]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i], fontsize=20)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.tick_params('y', labelsize=18)
    
    axes[-1].set_xlabel("Step Number", fontsize=20)
    axes[-1].tick_params(axis='x', labelsize=18)
    plt.tight_layout()
    plt.savefig('{s}{t}{f}-chain-logpost-{a}.png'.format(s=save_dir,
                                                      t=targetlabel,
                                                      f=filesavetag,
                                                      a=appendix))
    #plt.show()
    plt.close()
    return

def plot_chain(save_dir, targetlabel, filesavetag, sampler, labels, ndim,
                       appendix=""):
    """
    Plots MCMC chain trace plots for all parameters and the log posterior
    --------------------------------------
    Params:
        - save_dir (str) folder to save plot into
        - targetlabel (str) target ID
        - filesavetag (str) filename used for this fit
        - sampler (emcee obj) the sampler object produced by emcee
        - labels (array of strings) param names, may use latex formatting
        - ndim (int) number of parameters
        - appendix (str) tail end string for the filename - usually "burnin" 
                or "production"
    
    """
    fig, axes = plt.subplots(ndim+1, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    logprobs = sampler.get_log_prob()
    #logprior = sampler.get_blobs()
    #logpost = logprobs+logprior
    xaxis = np.linspace(1,len(logprobs), len(logprobs[:,0]))
    
    for h in range(len(logprobs[0])):
        ax = axes[0]
        ax.scatter(xaxis, logprobs[:,0], alpha=0.3, color='black', s=2)
        ax.set_ylabel("Log Prob.", fontsize=16)
        ax.set_title("MCMC Chain Traces", fontsize=22)
        ax.tick_params('y', labelsize=18)
    
    for i in range(ndim):
        ax = axes[i+1]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i], fontsize=20)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.tick_params('y', labelsize=18)
    
    axes[-1].set_xlabel("Step Number", fontsize=20)
    axes[-1].tick_params(axis='x', labelsize=18)
    plt.tight_layout()
    plt.savefig('{s}{t}{f}-chain-{a}.png'.format(s=save_dir,
                                                      t=targetlabel,
                                                      f=filesavetag,
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

def plot_tinygp_ll(save_dir, gpll, targetlabel, filesavetag):
    """ 
    Plot the tinygp log likelihood per re-calculation as run concurrently 
    with the MCMC modelling
    ----------------------------------------
    Params:
        - save_dir (str) folder to save plot into
        - gpll (array) each log likelihood estimate in time
        - targetlabel (str) target ID
        - filesavetag (str) filename used for this fit
    """
    fig, ax1 = plt.subplots(figsize=(10,10))
    x = np.arange(0, len(gpll), 1) * 1000 #x axis
    ax1.scatter(x, gpll)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("GP Neg. Log Likelihood")
    ax1.set_title(targetlabel + "  GP log likelihood over MCMC steps")
    plt.tight_layout()
    plt.savefig('{s}{t}{f}-GP-loglike-steps.png'.format(s=save_dir,
                                                      t=targetlabel,
                                                      f=filesavetag))

    plt.close()
    return

def plot_mcmc(save_dir, time, flux, error, 
              targetlabel, disctime, best_mcmc, flat_samples,
              labels, fitType, filesavetag, xlabel, tmin, lygosBG, QCBVs = None):
    """
    Main plotting function: produces the corner plot and MCMC best fit model
    plot for a given fitting run
    ----------------------------------------------
    Params:
        - save_dir (str) folder to save plot into
        - x  (array) time axis of data, starts at 0
        - y  (array) relative flux data
        - targetlabel (str) target ID
        - disctime (float) ground discovery time, with the sector start time
                subtracted off
        - best_mcmc (array) best fit output parameters
        - flat_samples (array) returned from emcee sampler
        - labels (array of strings) param names, may use latex formatting
        - fitType (int) 1-7, which fit is being produced
        - filesavetag (str) filename used for this fit
        - xlabel (str) generated by etsMAIN, of the style "Time [BJD-2457000]"
        - tmin (float) start time for sector
        - lygosBG (array or None) used in fit type 6
        - QCBVs (arrays or None) used in fit types 2,4,5
        
    fitType options:
        - 1 = single no
        - 2 = single with
        - 3 = double no
        - 4 = double with
        - 5 = flat
        - 6 = lygos bg
        - 7 = gaussian beta (same as 1)
    """
    #set up model = sl + bg but can be plot separately.
    t0 = best_mcmc[0]
    sl, bg = fitTypeModel(fitType, time, best_mcmc, QCBVs, lygosBG)
    model = sl + bg   
    #fix time axis for plotting
    time = time + tmin - 2457000
    disctime = disctime + tmin - 2457000
    t0 = t0 + tmin - 2457000
    
    plot_corner(flat_samples, labels, save_dir, targetlabel, filesavetag)
    
    plot_mcmc_model(save_dir, model, time, flux, error,
                     disctime, t0, xlabel,targetlabel, filesavetag)
    
    return

def plot_mcmc_GP_celerite(save_dir, time, flux, error, best_mcmc, gp, 
                          disctime, xlabel, tmin, targetlabel, 
                          filesavetag):
    """
    Plots the best fit model from MCMC with a celerite background
    Calls plot_mcmc_model() and gp_plots()
    ---------------------------------------------
    Params:
        - save_dir (str) folder to save plot into
        - time (array) time axis of data, starts at 0
        - flux (array) relative flux data
        - error (array) error on the flux data
        - best_mcmc (array) best fit output parameters
        - gp (celerite object)
        - disctime (float) ground discovery time, with the sector start time
                subtracted off
        - xlabel (str) generated by etsMAIN, of the style "Time [BJD-2457000]"
        - tmin (float) start time for sector
        - targetlabel (str) target ID
        - filesavetag (str) filename used for this fit
    """
    
    #t0, A,beta,B = best_mcmc[0][2:]
    #t1 = time - t0
    #mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B
    
    gp.set_parameter_vector(best_mcmc[0])
    #gp.compute(time, error)
    model = gp.predict(flux, time, return_cov=False)

    #model = mod + bg
    #fix time axis
    time = time + tmin - 2457000
    disctime = disctime + tmin - 2457000
    t0 = best_mcmc[0][2]
    t0 = t0 + tmin - 2457000
    
    plot_mcmc_model(save_dir, model, time, flux, error,
                     disctime, t0, xlabel,targetlabel, filesavetag)
    return

def plot_mcmc_GP_tinygp(save_dir, time, flux, error, best_mcmc,
                        gp, disctime, xlabel, tmin, 
                        targetlabel, filesavetag):
    """
    Plots the best fit model from MCMC with a tinygp background
    Calls plot_mcmc_model() and gp_plots()
    ---------------------------------------------
    Params:
        - save_dir (str) folder to save plot into
        - time (array) time axis of data, starts at 0
        - flux (array) relative flux data
        - error (array) error on the flux data
        - best_mcmc (array) best fit output parameters
        - gp (tinygp object)
        - disctime (float) ground discovery time, with the sector start time
                subtracted off
        - xlabel (str) generated by etsMAIN, of the style "Time [BJD-2457000]"
        - tmin (float) start time for sector
        - targetlabel (str) target ID
        - filesavetag (str) filename used for this fit 
    """

    t0, A,beta,B = best_mcmc[:4]
    t1 = time - t0
    mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B

    bg = gp.predict(flux-mod, time, return_cov=False)

    model = mod + bg
    
    #fix time axis
    time = time + tmin - 2457000
    disctime = disctime + tmin - 2457000
    t0 = t0 + tmin - 2457000
    
    plot_mcmc_model(save_dir, model, time, flux, error,
                     disctime, t0, xlabel,targetlabel, filesavetag)
    

    gp_plots(save_dir, mod, bg, time, flux, error,
                     disctime, t0, xlabel,targetlabel, filesavetag, 
                     gpfiletag = "MCMC-tinyGP-TriplePlotResiduals")
    return

def plot_mcmc_model(save_dir, model, time, flux, error,
                    disctime, t0, xlabel, targetlabel, filesavetag):
    """ 
    Produces two panel plot of output model. 
    Top panel gives the data + model, bottom panel gives the residual. 
    ---------------------------------------------
    Params:
        - save_dir (str) folder to save plot into
        - model (array) best fit model for data
        - time (array) time axis of data, starts at 0
        - flux (array) relative flux data
        - error (array) error on the flux data
        - disctime (float) ground discovery time, time axis corrected
        - t0 (float) the t0 output parameter
        - xlabel (str) generated by etsMAIN, of the style "Time [BJD-2457000]"
        - targetlabel (str) target ID
        - filesavetag (str) filename used for this fit 
    """
    nrows = 2
    ncols = 1
    fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                   figsize=(8*ncols * 2, 3*nrows * 2))
    
    #plot model, data
    ax[0].plot(time, model, label="Best Fit Model", lw=3, color = 'red')
    ax[0].scatter(time, flux, label = "Data", s = 3, color = 'black')
    
    for n in range(nrows):
        axy = ax[n]
        axy.axvline(t0, color = 'green', lw=2, linestyle = 'dotted',
                          label=r"$t_0$")
        axy.axvline(disctime, color = 'grey', lw=2, linestyle = 'dotted', 
                      label="Ground Disc.")
        axy.set_ylabel("Flux (e-/s)", fontsize=20)
        axy.tick_params('y', labelsize=18)
        
    #main
    ax[0].set_title(targetlabel + filesavetag)
    ax[0].legend(fontsize=18, loc="upper left")
    ax[nrows-1].set_xlabel(xlabel)
    ax[nrows-1].tick_params('x', labelsize=18)
    
    #residuals
    ax[1].set_title("Residual")
    residuals = flux - model
    ax[1].scatter(time,residuals, s=3, color = 'black', label='Residual, All')
    #ax[1].axhline(0, color='orange', linestyle = 'dashed', label="Zero")
    ax[1].legend(fontsize=18)
    
    plt.tight_layout()
    plt.savefig('{p}{t}{f}-MCMCmodel-bestFit.png'.format(p=save_dir,
                                                      t=targetlabel,
                                                      f=filesavetag))
    plt.close()
    return

def gp_plots(save_dir, model, bg, time, flux, error,
             disctime, t0, xlabel, targetlabel, filesavetag, gpfiletag):
    """ 
    Produces three panel plot of output model including GP
    Top panel gives the data + power law, middle panel gives power law residual,
    bottom panel gives GP residual. 
    ---------------------------------------------
    Params:
        - save_dir (str) folder to save plot into
        - model (array) best fit power law model for data
        - bg (array) best fit gp background model for residual
        - time (array) time axis of data, starts at 0
        - flux (array) relative flux data
        - error (array) error on the flux data
        - disctime (float) ground discovery time, time axis corrected
        - t0 (float) the t0 output parameter
        - xlabel (str) generated by etsMAIN, of the style "Time [BJD-2457000]"
        - targetlabel (str) target ID
        - filesavetag (str) filename used for this fit 
        - gpfiletag (str) tag added on to the end to indicate which gp model
            was used (ie, "MCMC-tinyGP-TriplePlotResiduals")
    """
    
    #second plot: 
    nrows = 3
    ncols = 1
    fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                   figsize=(8*ncols * 2, 3*nrows * 2))
    
    #top row: data, model ONLY
    ax[0].scatter(time, flux, label = "Data", s = 3, color = 'black')
    ax[0].plot(time, model, label="Best Fit Model", lw=3,color = 'red')
    
    
    #middle row: residual, GP fit to residual
    residual1 = flux - model
    ax[1].set_title("Model Residual")
    #bg, var = gp.predict(residual1, time, return_var=True)
    #err_bg = np.sqrt(var)
    ax[1].scatter(time, residual1, label = "Model  Residual", s = 3, color = 'black')
    ax[1].plot(time, bg, label="GP", color="blue",lw=3, alpha=0.5)
    #ax[1].axhline(0, color='orange', linestyle = 'dashed', label="zero")
    
    
    #bottom row: GP residual
    residual2 = residual1 - bg
    ax[2].set_title("GP Residual")
    ax[2].scatter(time, residual2, label = "GP Residual", s = 3, color = 'black')
    ax[2].axhline(0, color='orange', linestyle = 'dashed', lw=2,label="zero")
    
    #all plots: t_0, disc, rel flux label 
    for n in range(nrows):
        ax[n].axvline(t0, color = 'green',lw=2, linestyle = 'dotted',
                          label=r"$t_0$")
        ax[n].axvline(disctime, color = 'grey', lw=2,linestyle = 'dotted', 
                      label="Ground Disc.")
        ax[n].set_ylabel("Flux (e-/s)", fontsize=20)
        ax[n].tick_params('y', labelsize=18)
        
    #plot labelling
    ax[0].set_title(targetlabel + filesavetag)
    ax[nrows-1].set_xlabel(xlabel, fontsize=20)
    ax[nrows-1].tick_params('x', labelsize=18)
    
    ax[0].legend(fontsize=18, loc="upper left")
    ax[1].legend(fontsize=18)
    ax[2].legend(fontsize=18)
    
    plt.tight_layout()
    plt.savefig('{p}{t}{f}-{gpf}.png'.format(p=save_dir,t=targetlabel,
                                             f=filesavetag,
                                             gpf=gpfiletag))

    plt.close()
    return


def plot_celerite_tinygp_comp(save_dir, time, flux, targetlabel, 
                              filesavetag, best_mcmc, gpcelerite, gptinygp, 
                              disctime, xlabel, tmin):
    """ 
    Produces 2x3 plot comparing the no-GP, tinygp, and celerite fittings + 
    their residuals
    ---------------------------------------------
    Params:
        - save_dir (str) folder to save plot into
        - time (array) time axis of data, starts at 0
        - flux (array) relative flux data
        - targetlabel (str) target ID
        - filesavetag (str) filename used for this fit 
        - best_mcmc (array) best fit output parameters
        - gpcelerite (celerite gp object)
        - gptinygp (tinygp gp object)
        - disctime (float) ground discovery time
        - xlabel (str) generated by etsMAIN, of the style "Time [BJD-2457000]"
        - tmin (float) start time for sector
    """
    
    t0, A,beta,B = best_mcmc[0][:4]
    t1 = time - t0
    mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B
    
    gpcelerite.set_parameter_vector(best_mcmc[0][4:])
    celerite_bg, celerite_var = gpcelerite.predict(flux-mod, time, 
                                                   return_var=True)
    cel_std = np.sqrt(celerite_var)
    
    tinygp_bg = gptinygp.predict(flux-mod, time, return_cov=False)

    #fix time axis stuff
    time = time + tmin - 2457000
    disctime = disctime + tmin - 2457000
    t0 = t0 + tmin - 2457000
    
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
    # ax[0][0].set_title("{t} {f} Model Only".format(t=targetlabel,
    #                                                f=filesavetag), fontsize=20)
    # ax[0][1].set_title("{t} {f} celerite".format(t=targetlabel,
    #                                                f=filesavetag), fontsize=20)
    # ax[0][2].set_title("{t} {f} tinygp".format(t=targetlabel,
    #                                                f=filesavetag), fontsize=20)
    
    
    ax[0][0].plot(time, mod, lw=3, label="Model", color = 'red')
    ax[0][1].scatter(time, flux-mod, label="Residual", color='black', s=3)
    
    ax[1][0].plot(time, mod+celerite_bg, lw=3,label="Model + celerite", color='red')
    #ax[1][0].fill_between(time, mod+celerite_bg+cel_std, 
     #                     mod+celerite_bg-cel_std, color='pink', 
      #                    alpha=0.3,edgecolor="none", label="1 sigma")
    ax[1][1].scatter(time, flux-mod-celerite_bg, label="Residual", color='black', s=3)
    
    ax[2][0].plot(time, mod+tinygp_bg, lw=3,label="Model + tinygp", color='red')
    ax[2][1].scatter(time, flux-mod-tinygp_bg, label="Residual", color='black', s=3)
    
    ax[0][0].set_ylabel("Flux (e-/s)", fontsize=20)
    ax[1][0].set_ylabel("Flux (e-/s)", fontsize=20)
    ax[2][0].set_ylabel("Flux (e-/s)", fontsize=20)
    
    for n in range(ncols):
        #ax[1][n].axhline(0, color="orange", lw=2, label="Zero", linestyle='dotted')
        ax[nrows-1][n].set_xlabel(xlabel, fontsize=20)
        
        for i in range(nrows):
            ax[i][0].scatter(time, flux, label = "Data", s = 3, color = 'black')
            ax[i][n].axvline(t0, color = 'green', lw=2,linestyle = 'dotted',
                              label=r"$t_0$")
            ax[i][n].axvline(disctime, color = 'grey', lw=2,linestyle = 'dotted', 
                          label="Ground Disc.")
            ax[i][n].legend(fontsize=12, loc='upper left')
            ax[i][n].tick_params('y', labelsize=18)
    
    ax[nrows-1][0].tick_params('x', labelsize=18)
    ax[nrows-1][1].tick_params('x', labelsize=18)
    plt.tight_layout()
    plt.savefig('{p}{t}{f}-comparison-plot.png'.format(p=save_dir,
                                                      t=targetlabel,
                                                      f=filesavetag))
    plt.close()
    return

def plot_scipy_max(save_dir, filesavetag, targetlabel, x, y, yerr, 
                   t, mu, std, tmin, disctime):
    x = x + tmin - 2457000
    disctime = disctime + tmin - 2457000
    t = t + tmin - 2457000
    
    plt.errorbar(x, y, yerr=yerr, fmt=".k", alpha=0.2,capsize=0, label="Data")
    plt.plot(t, mu, lw=4, color='green', label="Prediction")
    plt.fill_between(t, mu+std, mu-std, color='green', alpha=0.3, edgecolor="none")
    plt.axvline(disctime, color = 'grey', lw=2,linestyle = 'dotted', 
                  label="Ground Disc.")
    plt.ylabel("Flux")
    plt.xlabel("Time [BJD-2457000]")
    plt.title("{t}: Scipy max. likelihood prediction".format(t=targetlabel))
    plt.legend()
    plt.tight_layout()
    plt.savefig('{s}{t}{f}-scipy-prediction.png'.format(s=save_dir,
                                                        t=targetlabel,
                                                        f=filesavetag))
    plt.show()
    plt.close()
    
def celerite_post_pred(save_dir, filesavetag, targetlabel, x, y, yerr,
                       t, flat_samples, gp, tmin, disctime):
    x = x + tmin - 2457000
    disctime = disctime + tmin - 2457000
    tplot = t + tmin - 2457000
    
    plt.errorbar(x, y, yerr=yerr, fmt=".k",alpha=0.3, capsize=0)
    plt.axvline(disctime, color = 'grey', lw=2,linestyle = 'dotted', 
                  label="Ground Disc.")
    # Plot 24 posterior samples.
    for s in flat_samples[np.random.randint(len(flat_samples), size=24)]:
        gp.set_parameter_vector(s)
        mu = gp.predict(y, t, return_cov=False)
        plt.plot(tplot, mu, color='green', alpha=0.5)

    plt.ylabel("Flux")
    plt.xlabel("Time [BJD-2457000]")
    plt.title("{t} posterior predictions".format(t=targetlabel))
    plt.legend()
    plt.savefig('{s}{t}{f}-celerite-post-pred.png'.format(s=save_dir,
                                                        t=targetlabel,
                                                        f=filesavetag))
    plt.show()
    plt.close()