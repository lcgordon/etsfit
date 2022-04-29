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


def plot_bic_ranked(path, x,y, QCBVs, lygosBG,
                    targetlabel, disctime, 
                    tmin, best_mcmc, bic_list):
    """
    Plot all five plots + residuals for their BIC values, ranked by best 
    (lowest bic) to worst (highest bic)
    """
    nrows = len(bic_list) #only make rows for as many as there are fits
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                  figsize=(8*ncols * 2, 3*nrows * 2))
    
    ####sort bic_list - want sorted INDEXES based on the values inside of bic_list
    if (len(bic_list) == 3):
        for n in range(len(bic_list)):
            print(bic_list[n])
            if bic_list[n] == 2:
                bic_list[n] = 5 # if you're only plotting the three of them lol
    bic_sort = np.argsort(np.asarray(bic_list))
    print(bic_sort)
    titletags = ["Single-No", "Single-WithCBV", "Double-No", "Double-WithCBV", "QCBV-Only"]
    
    ####for each fit type
    for j in range(len(bic_sort)):
        fitType = bic_sort[j] + 1 #get which of them you're going to plot
        indexOfFitType = bic_sort[j]
        print(fitType, titletags[indexOfFitType])
        print("params:", best_mcmc[indexOfFitType])
        sl, bg = fitTypeModel(fitType, x, best_mcmc[indexOfFitType], QCBVs = QCBVs,
                              lygosBG = lygosBG)
        #model
        model = sl + bg
        ax[j][0].plot(x, model, label="Best Fit Model", color = 'red')
        ax[j][0].set_title(titletags[indexOfFitType] + 
                           "   BIC = {numbo:.2f}".format(numbo=(bic_list[j])))
        ax[j][0].scatter(x, y, label = "Data", s = 5, color = 'black')
        
        if fitType != 5:
            ax[j][0].plot(x, sl, label="Power Law", color = 'blue')
            ax[j][0].plot(x, bg, label="Background", color = 'green')
         
        #ax[j][0].legend(fontsize=18, loc = "upper left")
        #residuals
        ax[j][1].set_title("Residual (data-model)")
        residuals = y - model
        ax[j][1].scatter(x,residuals, s=5, color = 'black', label='Residual')
        ax[j][1].axhline(0,color='purple', linestyle = 'dashed', label="zero")
        #misc.
        ax[j][0].axvline(disctime, color = 'grey', linestyle = 'dotted', 
                  label="Ground Disc.")
        ax[j][1].axvline(disctime, color = 'grey', linestyle = 'dotted', 
                  label="Ground Disc.")
        
        ax[j][0].set_ylabel("Rel. Flux", fontsize=12)
        ax[j][1].set_ylabel("Rel. Flux", fontsize=12)
        if fitType != 5:
            ax[j][0].axvline(best_mcmc[fitType][0], color = "saddlebrown",
                             linestyle = 'dashed', label = r"$t_0$")
            ax[j][1].axvline(best_mcmc[fitType][0], color = "saddlebrown",
                             linestyle = 'dashed', label = r"$t_0$")
        if fitType in (3,4):
            ax[j][0].axvline(best_mcmc[fitType][1], color = 'saddlebrown',
                             linestyle = 'dashed', label = r"$t_1$")
            ax[j][1].axvline(best_mcmc[fitType][1], color = 'saddlebrown',
                             linestyle = 'dashed', label = r"$t_1$")
        ax[j][1].legend(fontsize=18, loc='upper left')
        ax[j][0].legend(fontsize = 18, loc = 'upper left')
    
    fig.suptitle(targetlabel)
    ax[nrows-1][0].set_xlabel("BJD - {timestart:.3f}".format(timestart=tmin))
    ax[nrows-1][1].set_xlabel("BJD - {timestart:.3f}".format(timestart=tmin))
    plt.tight_layout()
    plt.savefig(path + targetlabel + "-bic-sorted-all.png")
    return
    
def plot_autocorr_mean(savepath, targetlabel, index, autocorr, converged,
                  autoStep, filesavetag):
    """ Plot autocorrelation time vs number of steps
    Params:
        - path (str) to save into
        - targetlabel (str) name of target
        - index (int) number of autocorr tests
        - autocorr (array) test output
        - converged (bool) did it converge or no
        
    """
    n = autoStep * np.arange(1, index + 1) #x axis - number of steps
    plotAutocorr = autocorr[:index]
    plt.plot(n, n / 100, "--k") #plots the N vs N/100=tau threshold 
    #this determines length of chain vs autocorrelation time
    plt.plot(n, plotAutocorr)
    plt.xlim(0, n.max())
    #plt.ylim(0, plotAutocorr.max() + 0.1 * (plotAutocorr.max() - plotAutocorr.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$")
    plt.title(targetlabel + "  converged=" + str(converged))
    plt.savefig(savepath+targetlabel+ "-" + filesavetag + "-autocorr-mean.png")
    plt.close()
    
def plot_autocorr_individual(savepath, targetlabel, index, autocorr_all,
                             autoStep, labels, filesavetag):
    """Plot each autocorrleation time function from an array of all of them """
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
                    "-autocorr-" + labels[i] +".png")
        plt.close()


def plot_corner(flat_samples, labels, path, targetlabel, filesavetag):
    """ produce corner plot of params """
    fig = corner.corner(
        flat_samples, labels=labels,
        quantiles = [0.16, 0.5, 0.84],
                       show_titles=True, title_fmt = ".4f", 
                       title_kwargs={"fontsize": 18}
    );
    plt.yticks(fontsize=6)
    plt.xticks(fontsize =6)
    plt.tight_layout()
    fig.savefig(path + targetlabel + filesavetag + '-corner-plot-params.png')
    plt.show()
    plt.close()
    return

def plot_paramIndividuals(flat_samples, labels, path, targetlabel, filesavetag):
    """ plots param vs p(param) histograms """
    for p in range(len(labels)):
        plt.hist(flat_samples[:, p], 100, color="k", histtype="step")
        plt.xlabel(labels[p])
        plt.ylabel("p("+labels[p]+")")
        plt.gca().set_yticks([]);
        plt.savefig(path + targetlabel + "-" + filesavetag + 
                    "-chainHisto-" + labels[p] + ".png")
        plt.close()

def plot_log_post(path, targetlabel,filesavetag, sampler):
    '''plot that sweet sweet log posterior'''
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

def plot_chain_logpost(path, targetlabel, filesavetag, sampler, labels, 
                       ndim, appendix = ""):
    """plot mcmc chain by parameter AND log posterior at top """
    rcParams['figure.figsize'] = 30,30
    rcParams['ytick.labelsize'] = 10
    fig, axes = plt.subplots(ndim+1, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = labels
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

def fitTypeModel(fitType, x, best_mcmc, QCBVs = None, lygosBG = None):
    """
    Produce plotting model for a given fit type (1-5) 
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
    if fitType == 1:
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
        def func1(x, t1, t2, a1, a2, B1, B2):
            return B1 *(x-t1)**a1
        def func2(x, t1, t2, a1, a2, B1, B2):
            return B1 * (x-t1)**a1 + B2 * (x-t2)**a2
        t1, t2, a1,a2, beta1, beta2, b = best_mcmc
        sl = np.piecewise(x, [(t1 <= x)*(x < t2), t2 <= x], 
                             [func1, func2],
                             t1, t2, a1, a2, beta1, beta2) 
        bg = np.ones(len(x)) + b
    elif fitType ==4:
        def func1(x, t1, t2, a1, a2, B1, B2):
            return B1 *(x-t1)**a1
        def func2(x, t1, t2, a1, a2, B1, B2):
            return B1 * (x-t1)**a1 + B2 * (x-t2)**a2
        
        Qall, CBV1, CBV2, CBV3 = QCBVs
        t1, t2, a1,a2, beta1, beta2, cQ, cbv1, cbv2, cbv3 = best_mcmc#[0]
        sl = np.piecewise(x, [(t1 <= x)*(x < t2), t2 <= x], 
                                  [func1, func2], t1, t2, a1, a2, beta1, beta2)
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

def plot_mcmc(path, time, intensity, targetlabel, disctime, best_mcmc, flat_samples,
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
    
    nrows = 2
    ncols = 1
    fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                   figsize=(8*ncols * 2, 3*nrows * 2))
    
    ax[0].plot(time, model, label="Best Fit Model", color = 'red')
    ax[0].scatter(time, intensity, label = "Data", s = 5, color = 'black')
    if fitType != 5:
        ax[0].plot(time, sl+1, label="Source", color = 'blue')
        if fitType % 2 == 0:
            #evens are w/ cbvs, odds are without
            ax[0].plot(time, bg, label="CBV fit", color = 'green')
        else:
            ax[0].plot(time, bg, label="Offset", color = 'green')
    
    for n in range(nrows):
        if fitType != 5:
            ax[n].axvline(best_mcmc[0], color = 'saddlebrown', linestyle = 'dashed',
                          label=r"$t_0$")
        ax[n].axvline(disctime, color = 'grey', linestyle = 'dotted', 
                      label="Ground Disc.")
        ax[n].set_ylabel("Rel. Flux", fontsize=12)
        
    #main
    ax[0].set_title(targetlabel + filesavetag)
    ax[0].legend(fontsize=18, loc="upper left")
    ax[nrows-1].set_xlabel("BJD - {timestart:.3f}".format(timestart=tmin))
    
    #residuals
    ax[1].set_title("Residual")
    residuals = intensity - model
    ax[1].scatter(time,residuals, s=5, color = 'black', label='Residual')
    ax[1].axhline(0,color='purple', linestyle = 'dashed', label="zero")
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(path + targetlabel + filesavetag + "-MCMCmodel-bestFit.png")
    return

def plot_chain(path, targetlabel, plotlabel, samples, labels, ndim):
    """Plots mcmc chain by parameter """
    rcParams['figure.figsize'] = 30,30
    rcParams['ytick.labelsize'] = 10
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    #samples = sampler.get_chain()
    labels = labels
    #plt.yticks(fontsize=6)
    #plt.xticks(fontsize =6)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number");
    plt.savefig(path + targetlabel+ plotlabel)
    plt.show()
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
    
    plt.savefig(filename)
    plt.close()
    rcParams['figure.figsize'] = 16,6
    return 
def plot_beta_redshift(savepath, info, sn_names, bestparams):
    for n in range(len(bestparams)):
        target = sn_names['ID'][n][:-4]
        #get z where the name matches in info??
        beta = bestparams['beta'][n]
        df1 = info[info['ID'].str.contains(target)]
        df1.reset_index(inplace=True)
        for i in range(len(df1)): #AHHHHH so there are sometimes multiple thingies w/ the same key
            if df1["ID"][i] == target:
                redshift = df1['Z'][i]
                
        plt.scatter(redshift, beta)
        
    
    plt.xlabel('redshift')
    plt.ylabel('beta value')
    plt.title("Plotting " +  r'$\beta$' + " versus redshift for Ia SNe")   
    plt.savefig(savepath + "redshift-beta.png") 
    
def plot_absmag(t,i, xlabel='',ylabel='', title='',savepath=None):
    fig, ax =plt.subplots()
    ax.scatter(t,i)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if savepath is not None:
        plt.savefig(savepath)
        
def quicklook_plotall(path, all_t, all_i, all_labels, discovery_dictionary):
    """Plot all in the list and save plots into dedicated folder
    allows for quick flip thru them all to get rid of gunk. """
    from pylab import rcParams
    rcParams['figure.figsize'] = 8,3
    for n in range(len(all_labels)):
        key = all_labels[n]
        if -3 <= discovery_dictionary[key] <= 30:
            plt.scatter(all_t[n], all_i[n])
            plt.axvline(discovery_dictionary[key])
            plt.title(all_labels[n])
            plt.savefig(path + all_labels[n] + "-.png")
            plt.close()
            
def print_table_formatting(best,upper,lower):
    for n in range(len(best[0])):
        print("param ${:.4f}".format(best[0][n]), "^{:.4f}".format(upper[0][n]),
              "_{:.4f}$".format(lower[0][n]))
        
def plot_SN_LCs(path, t,i,e,label,sector,galmag,extinction, z, 
                discdate, badIndexes):

    import sn_functions as sn
    
    nrows = 4
    ncols = 1
    fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                  figsize=(8*ncols * 2, 3*nrows * 2))
    
    #row 0: raw lygos light curve
    if badIndexes is None:
        cPlot = 'green'
        ecoll = 'springgreen'
        binT, binI, binE = sn.bin_8_hours_TIE(t,i,e) #bin to 8 hours
    else:
        cPlot = 'yellow'
        ecoll = 'yellow'
        cT, cI, cE = sn.clip_TIE(badIndexes, t,i,e) #clip out designated indices
        ax[0].errorbar(cT, cI, cE, fmt = 'o', label = "Lygos (Clipped)", 
                       color = 'green',
                       ecolor = 'springgreen', zorder=2)
        binT, binI, binE = sn.bin_8_hours_TIE(cT, cI, cE) #bin to 8 hours
        
    ax[0].errorbar(t,i,yerr=e, fmt = 'o', label = "Lygos (Raw)", color = cPlot,
                   ecolor=ecoll, zorder=1)
    
    #ax[0].axhline(1, color='orchid', label='Lygos Background')
    ax[0].set_ylabel("Rel. Flux", fontsize=16)
    ax[0].set_title(label, fontsize=18)
    
    #row 1: binned and cleaned up flux.
    ax[1].set_title("Binned Flux", fontsize=16)
    ax[1].errorbar(binT, binI, yerr=binE, fmt = 'o', label = "Binned and Cleaned",
                   color = 'blue', ecolor = "blue", markersize = 5)
    ax[1].set_ylabel("Rel. Flux", fontsize=16)
    
    #row 2: apparent TESS magnitude
    (absT, absI, absE, absGalmag,
     d, apparentM, apparentE) = sn.conv_to_abs_mag(binT, binI, binE , galmag, z,
                                                       extinction = extinction)
    
    
    ax[2].errorbar(absT, apparentM, yerr=apparentE, fmt = 'o', 
                   color = 'darkslateblue', ecolor='slateblue', markersize=5)
    ax[2].set_title("Apparent TESS Magnitude", fontsize=16)
    ax[2].set_ylabel("Apparent Mag.", fontsize=16)
    ax[2].invert_yaxis()
    
    #row 3: absolute magntiude conversion
    
    ax[3].errorbar(absT, absI, yerr = absE, 
                   fmt = 'o',label="abs mag",  color = 'purple',
                   ecolor='lavender', markersize=5)
    #ax[3].axhline(absGalmag, color = 'orchid',label="background mag." )
    ax[3].invert_yaxis()
    ax[3].set_title("Absolute Magnitude Converted", fontsize=16)
    ax[3].set_ylabel("Abs. Magnitude", fontsize=16)
    
    for i in range(nrows):
        ax[i].axvline(discdate, color = 'black', 
                      label="discovery time")
        
        ax[i].legend(loc="upper left", fontsize=12)
        
        
    ax[nrows-1].set_xlabel("BJD-2457000", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(path + label + "flux-plot.png")
    plt.show()
    #plt.close()
    return binT, binI, binE, absT, absI, absE