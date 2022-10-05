# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:47:01 2021

@author: conta

SN Plotting
"""
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
rcParams['figure.figsize'] = 16,6
rcParams["font.size"] = 20
import corner
import os



def plot_autocorr_mean(savepath, targetlabel, index, autocorr, converged,
                       autoStep, filesavetag):
    """ Plot autocorrelation time vs number of steps
    Params:
        - path (str) to save into
        - targetlabel (str) name of target
        - index (int) number of autocorr tests
        - autocorr (array) autocorr test mean output
        - converged (bool) did the chain converge in the end or not
        - autoStep (int) how many steps
        - filesavetag (str) name for file
        
    """
    n = autoStep * np.arange(1, index + 1) #x axis = total number of steps
    plotAutocorr = autocorr[:index]
    plt.plot(n, n / 100, "--k", label = "N/100 threshold") #plots the N vs N/100=tau threshold 
    #this determines length of chain vs autocorrelation time
    plt.plot(n, plotAutocorr)
    plt.xlim(0, n.max())
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$")
    plt.title(targetlabel + ", mean autocorr. time, converged = " + str(converged))
    plt.savefig(savepath+targetlabel+ "-" + filesavetag + "-autocorr-mean-plot.png")
    plt.close()
    
def plot_autocorr_individual(savepath, targetlabel, index, autocorr_all,
                             autoStep, labels, filelabels, filesavetag):
    """Plot each autocorrleation time function from an array of all of them 
    Params:
        - savepath (str) to put files into
        - targetlabel (str) name of target
        - index (int) number of autocorr tests
        - autocorr_all (array of size (params,index)) with all 
            parameter autocorrelation times
        - autoStep (int) how many steps
        - labels (array of strings) what each param is called
        - filelabels (array of strings) what each is called formatted for filenames
        - filesavetag (string) to name file with
    """
    n = autoStep * np.arange(1, index + 1) #x axis - number of steps
    for i in range(len(labels)):
        
        plotAutocorr = autocorr_all[:index,i]
        plt.plot(n, n / 100, "--k") #plots the N vs N/100=tau threshold 
        #this determines length of chain vs autocorrelation time
        plt.plot(n, plotAutocorr)
        plt.xlim(0, n.max())
        #plt.ylim(0, plotAutocorr.max() + 0.1 * (plotAutocorr.max() - plotAutocorr.min()))
        plt.xlabel("number of steps")
        plt.ylabel(r"$\hat{\tau} for $" + labels[i])
        plt.title(targetlabel + "  autocorr time for " + labels[i])
        plt.savefig(savepath+targetlabel+ "-" + filesavetag + 
                    "-autocorr-" + filelabels[i] +".png")
        plt.close()


def plot_corner(flat_samples, labels, path, targetlabel, filesavetag):
    """ Produces corner plot of all parameters against one another
    Params:
        - flat_samples (array) returned from emcee
        - labels (array of strings) what each parameter is called
        - path (string) to save into
        - targetlabel (string) name of target
        - filesavetag (string) what to label file with
    """
    fig = corner.corner(
        flat_samples, labels=labels,
        quantiles = [0.16, 0.5, 0.84],
                       show_titles=True, title_fmt = ".4f", 
                       title_kwargs={"fontsize": 18});
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.tight_layout()
    fig.savefig(path + targetlabel + filesavetag + '-corner-plot-params.png')
    plt.show()
    plt.close()
    return

def plot_paramIndividuals(flat_samples, labels, path, targetlabel, filesavetag):
    """ Plots parameters vs p(parameter) histograms individually"""
    
    for p in range(len(labels)):
        plt.hist(flat_samples[:, p], 100, color="k", histtype="step")
        plt.xlabel(labels[p])
        plt.ylabel("p("+labels[p]+")")
        plt.gca().set_yticks([]);
        plt.savefig(path + targetlabel + "-" + filesavetag + 
                    "-chainHisto-" + labels[p] + ".png")
        plt.close()
    return

def plot_paramTogether(flat_samples, labels, path, targetlabel, filesavetag):
    """Plot all the parameter vs p(param) histograms together """
    
    axN = len(labels)
    if axN % 2 != 0:
        axN+=1 #make it even (two columns)
    nrows = int(axN/2)
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols, sharex=False,
                               figsize=(8*ncols * 2, 3*nrows * 2))
    p = 0
    for n in range(2):
        for m in range(nrows):
            if p < len(labels):
                ax[m, n].hist(flat_samples[:, p], 100, color="k", histtype="step")
                ax[m, n].set_xlabel(labels[p])
                ax[m, n].set_ylabel("p("+labels[p]+")")
                p+=1
    
    fig.suptitle("Parameter Plots")
    plt.savefig(path + targetlabel + "-" + filesavetag + "-chainHisto-all.png.png")
    plt.close()
    return
    
def plot_log_post(path, targetlabel,filesavetag, sampler):
    '''
    plot the log posteriors
    unclear if this gets used anywhere?
    '''
    logprobs = sampler.get_log_prob()
    logprior = sampler.get_blobs()
    logpost = logprobs+logprior
    xaxis = np.linspace(1,len(logpost), len(logpost[:,0]))
    for h in range(len(logpost[0])):
        plt.scatter(xaxis, logpost[:,0])
    plt.xlabel("steps")
    plt.ylabel("log posterior")
    plt.savefig(path + "/" + targetlabel + "-" + filesavetag + "-log-post.png")
    plt.close()
    return



def fitTypeModel(fitType, x, best_mcmc, QCBVs = None, lygosBG = None):
    """
    Produce plotting model for a given fit type (1-6) 
    Params:
        - fitType (int, which fit are you doing)
        - x (array, x-axis that starts at 0)
        - best_mcmc (array, params for best fit model)
        - QCBVs (either an array [Qall, CBV1, CBV2, CBV3] or just None)
    Returns:
        - sl (actual model)
        - bg (background part, either a single run of B or the complex linear combo)
    """
    print("starting model creation time axis at: ", x[0])
    if fitType == 1 or fitType == 7:
        t0, A,beta,B = best_mcmc
        t1 = x - t0
        sl = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False)
        bg = np.ones(len(x)) + B
    elif fitType == 2: 
        t0, A, beta, B, cQ, cbv1, cbv2, cbv3 = best_mcmc
        t1 = x - t0
        Qall, CBV1, CBV2, CBV3 = QCBVs
        sl = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta))
        bg = cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3 + 1 + B
        
    elif fitType == 3:
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        t0, t1, A1, A2, beta1, beta2, B = best_mcmc
        sl = np.piecewise(x, [(t0 <= x)*(x < t1), t1 <= x], 
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
        sl = np.piecewise(x, [(t0 <= x)*(x < t1), t1 <= x], 
                             [func1, func2],
                             t0, t1, A1, A2, beta1, beta2)
        bg = cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3 + 1
        
    elif fitType ==5:
        Qall, CBV1, CBV2, CBV3 = QCBVs
        b, cQ, cbv1, cbv2, cbv3 = best_mcmc#[0]
        sl = np.zeros(len(x))
        bg = (b + np.ones(len(x)) + cQ * Qall + 
              cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3)
        print("background starts with", bg[0:5])
    
    elif fitType == 6:
        t0, A,beta,B, LBG = best_mcmc
        t1 = x - t0
        sl = (np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta)))
        bg = 1 + B + lygosBG * LBG
    
    return sl, bg

def plot_chain_logpost(path, targetlabel, filesavetag, sampler, labels, 
                       ndim, appendix = ""):
    """
    plot mcmc chain by parameter AND log posterior at top 
    """
    rcParams['figure.figsize'] = 30,30
    rcParams['ytick.labelsize'] = 10
    fig, axes = plt.subplots(ndim+1, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    logprobs = sampler.get_log_prob()
    logprior = sampler.get_blobs()
    logpost = logprobs+logprior
    xaxis = np.linspace(1,len(logpost), len(logpost[:,0]))
    
    for h in range(len(logpost[0])):
        ax = axes[0]
        ax.scatter(xaxis, logpost[:,0], alpha=0.3, color='black', s=2)
        ax.set_ylabel("Log \n Post.")
    
    for i in range(ndim):
        ax = axes[i+1]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number");
    plt.savefig(path + "/" + targetlabel + "-" + filesavetag + appendix 
                + "-chain-logpost.png")
    plt.show()
    plt.close()
    rcParams['figure.figsize'] = 16,6
    return

def plot_chain(path, targetlabel, plotlabel, samples, labels, ndim):
    """
    Plots mcmc chains - each parameter is a subpanel stacked vertically
    """
    rcParams['figure.figsize'] = 30,30
    rcParams['ytick.labelsize'] = 10
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    labels = labels
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number");
    plt.savefig(path + targetlabel+ plotlabel)
    plt.show()
    plt.close()
    rcParams['figure.figsize'] = 16,6
    return

def plot_histogram(data, bins, x_label, filename):
    """ 
    Plot a histogram with one light curve from each bin plotted on top
    * Data is the histogram data
    * Bins is bins for the histogram
    * x_label for the x-axis of the histogram
    * filename is the exact place you want it saved
    """
    rcParams['figure.figsize'] = 10,10
    fig, ax1 = plt.subplots()
    n_in, bins, patches = ax1.hist(data, bins)
    
    y_range = np.abs(n_in.max() - n_in.min())
    x_range = np.abs(data.max() - data.min())
    ax1.set_ylabel('Number of light curves')
    ax1.set_xlabel(x_label)
    
    if filename is not None:
             plt.tight_layout()
             plt.savefig(filename)
    plt.show()
    plt.close()
    rcParams['figure.figsize'] = 16,6
    return 

def plot_mcmc(pathSave, time, intensity, targetlabel, disctime, best_mcmc, flat_samples,
              labels, fitType, filesavetag, tmin, lygosBG, QCBVs = None):
    ""
    """main plotting function for mcmc 
    params:
        - path (string, to save into)
        - x  (array, time axis STARTING AT 0)
        - y  (array, actual intensity data)
        - targetlabel (string, SN name for various things. no spaces.)
        - disctime (float, start date from target, should be from 0 to 30!)
        - best_mcmc (array of best fit parameters)
        - flat_samples (??) idk this comes out of emcee sampler
        - labels (array of str) what parameters were floated
        - fitType (int) indicates which of the models you're using
        - filesavetag (str) label for output
        - tmin (float) start time for sector, rounded 
        
    fitType options:
        - 1 = single no
        - 2 = single with
        - 3 = double no
        - 4 = double with
        - 5 = flat
    """
    #set up model = sl + bg but can be plot separately.
    sl, bg = fitTypeModel(fitType, time, best_mcmc, QCBVs, lygosBG)
    model = sl + bg    
    plot_corner(flat_samples, labels, path, targetlabel, filesavetag)
    
    plot_mcmc_model(pathSave, sl, bg, model, time, intensity, error,
                     disctime, t0, tmin,targetlabel, filesavetag)
    
    return

def plot_mcmc_GP_celerite(pathSave, time, intensity, error, best_mcmc, gp, 
                          disctime, tmin, targetlabel, 
                          filesavetag, plotComponents=False):
    """Plot the best fit model from the mcmc run w/ GP on """
    
    t0, A,beta,B = best_mcmc[0][:4]
    t1 = time - t0
    sl = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B
    gp.set_parameter_vector(best_mcmc[0][4:])
    bg = gp.predict(intensity, time, return_cov=False)

    model = sl + bg
    
    plot_mcmc_model(pathSave, sl, bg, model, time, intensity, error,
                     disctime, t0, tmin,targetlabel, filesavetag)
    
    gp_plots(pathSave, sl, bg, model, time, intensity, error,
                 disctime, t0, tmin, targetlabel, filesavetag, 
                 gpfiletag = "-MCMC-celeriteGP-TriplePlotResiduals.png")
    return

def plot_mcmc_GP_tinygp(pathSave, time, intensity, error, best_mcmc,
                        disctime, t0, tmin,targetlabel, filesavetag, 
                        plotComponents=False):
    """Plot the best fit model from the mcmc run w/ tinyGP on """
    
    t0, A,beta,B = best_mcmc[0][:4]
    t1 = time - t0
    sl = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B
    
    kernel = best_mcmc[0][4] * kernels.ExpSquared(best_mcmc[0][5])
    gp = GaussianProcess(kernel, time, mean=0.0)
    
    bg = gp.predict(intensity-sl, time, return_cov=False)

    model = sl + bg
    
    plot_mcmc_model(pathSave, sl, bg, model, time, intensity, error,
                     disctime, t0, tmin,targetlabel, filesavetag)
    
    gp_plots(pathSave, sl, bg, model, time, intensity, error,
                 disctime, tmin, targetlabel, filesavetag, 
                 gpfiletag = "-MCMC-tinyGP-TriplePlotResiduals.png")
    
    return

def plot_mcmc_model(pathSave, sl, bg, model, time, intensity, error,
                 disctime, t0, tmin,targetlabel, filesavetag):
    """ 
    Plots the main 2 panel mcmc model + residual
    """
    nrows = 2
    ncols = 1
    fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                   figsize=(8*ncols * 2, 3*nrows * 2))
    
    ax[0].plot(time, model, label="Best Fit Model", color = 'red')
    #if plotComponents:
     #   ax[0].plot(time, bg, label="Just GP", color="green", alpha=0.2)
      #  ax[0].plot(time, sl, label="Just Model", color="blue", alpha=0.2)
        
    ax[0].scatter(time, intensity, label = "Data", s = 5, color = 'black')
    
    for n in range(nrows):
        ax[n].axvline(t0, color = 'saddlebrown', linestyle = 'dashed',
                          label=r"$t_0$")
        ax[n].axvline(disctime, color = 'grey', linestyle = 'dotted', 
                      label="Ground Disc.")
        ax[n].set_ylabel("Flux (e-/s)", fontsize=12)
        
    #main
    ax[0].set_title(targetlabel + filesavetag)
    ax[0].legend(fontsize=10, loc="upper left")
    ax[nrows-1].set_xlabel("BJD - {timestart:.3f}".format(timestart=tmin))
    
    #residuals
    ax[1].set_title("Residual")
    residuals = intensity - model
    ax[1].scatter(time,residuals, s=5, color = 'black', label='Residual, All')
    ax[1].axhline(0,color='purple', linestyle = 'dashed', label="zero")
    ax[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(pathSave + targetlabel + filesavetag + "-MCMCmodel-bestFit.png")
    plt.close()
    return

def gp_plots(pathSave, sl, bg, model, time, intensity, error,
                 disctime,t0, tmin,targetlabel, filesavetag, gpfiletag):
                 #plotComponents=False):
    """ 
    Plots the gp residual plots (3 panels, 1 model, 2 residuals)
    """
    
    #second plot: 
    nrows = 3
    ncols = 1
    fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                   figsize=(8*ncols * 2, 3*nrows * 2))
    
    #top row: data, model ONLY
    ax[0].scatter(time, intensity, label = "Data", s = 5, color = 'black')
    ax[0].plot(time, sl, label="Best Fit Model", color = 'red')
    
    # if plotComponents:
    #     ax[0].plot(time, bg, label="Just GP", color="green", alpha=0.2)
    #     ax[0].plot(time, sl, label="Just Model", color="blue", alpha=0.2)
    
    #middle row: residual, GP fit to residual
    residual1 = intensity - sl
    ax[1].set_title("Model Residual")
    #bg, var = gp.predict(residual1, time, return_var=True)
    #err_bg = np.sqrt(var)
    ax[1].scatter(time, residual1, label = "Model  Residual", s = 5, color = 'black')
    ax[1].plot(time, bg, label="GP", color="green")
    ax[1].axhline(0,color='purple', linestyle = 'dashed', label="zero")
    
    
    #bottom row: GP residual
    residual2 = residual1 - bg
    ax[2].set_title("GP Residual")
    ax[2].scatter(time, residual2, label = "GP Residual", s = 5, color = 'black')
    
    #all plots: t_0, disc, rel flux label 
    for n in range(nrows):
        ax[n].axvline(t0, color = 'saddlebrown', linestyle = 'dashed',
                          label=r"$t_0$")
        ax[n].axvline(disctime, color = 'grey', linestyle = 'dotted', 
                      label="Ground Disc.")
        ax[n].set_ylabel("Flux (e-/s)", fontsize=12)
        
    #plot labelling
    ax[0].set_title(targetlabel + filesavetag)
    ax[nrows-1].set_xlabel("BJD - {timestart:.3f}".format(timestart=tmin))
    
    ax[0].legend(fontsize=10, loc="upper left")
    ax[1].legend(fontsize=10)
    ax[2].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(pathSave + targetlabel + filesavetag + gpfiletag)
    plt.show()
    plt.close()
    return
