#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 08:44:40 2022

@author: lindseygordon

paper plots
"""
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
rcParams['figure.figsize'] = 16,6
rcParams["font.size"] = 20
import os
import pandas as pd
from etsfit import etsMAIN
from astropy.time import Time

def beta_histo(foldersave, betaall, convy ):

    fig, ax1 = plt.subplots(figsize=(10,10))
    n_in, bins, patches = ax1.hist(np.asarray(betaall), 9, color='black', 
                                   alpha=0.5, label="Unconverged")
    ax1.hist(np.asarray(betaall)[np.where(np.asarray(convy) == "True")], bins, 
             color='purple', alpha=0.3, label="Converged")
    
    ax1.set_ylabel('Number of light curves')
    ax1.set_xlabel(r"$\beta$")
    plt.title(r"Histogram of Retrieved $\beta$ Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig("{f}/beta-histogram.png".format(f=foldersave))
    plt.show()
    plt.close()

def plot_t0_disc_beta(foldersave, discall, t0all, betaall, upper_all,
                      convy):

    time_between = np.asarray(discall)-np.asarray(t0all)
    plt.errorbar(np.asarray(betaall), time_between, xerr = np.asarray(upper_all)[:,2],
                  yerr = np.asarray(upper_all)[:,0], fmt='o',
                  label = "Unconverged", color='red')
    plt.errorbar(np.asarray(betaall)[np.where(np.asarray(convy) == "True")], 
                time_between[np.where(np.asarray(convy) == "True")], 
                xerr = np.asarray(upper_all)[np.where(np.asarray(convy) == "True")][:,2],
                yerr = np.asarray(upper_all)[np.where(np.asarray(convy) == "True")][:,0], fmt='o',label = "Converged",
                color="Blue")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"Disc time. - $t_0$ (JD)")
    plt.title(r"Time between $t_0$ and discovery time versus $\beta$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("{f}/disc-t0-beta.png".format(f=foldersave))
    return

def big_plot_singlepower(bigInfoFile, datafolder, foldersave, targetlist,
                         filetag, 
                         fraction = None, binning=False):
    """ 
    Plot a bunch of single powerlaws together
    """
    import etsfit.utils.batch_analyze as ba
    import etsfit.utils.utilities as ut
    ncols = 2
    nrows = int(len(targetlist)/2)
    fig, ax = plt.subplots(nrows, ncols, sharex=False,
                           figsize=(8*ncols, 3*nrows))

    info = pd.read_csv(bigInfoFile)
    #i = 0
    m = 0
    n = 0
    
    for root, dirs, files in os.walk(datafolder):
        for name in files:
            if name.endswith("-tessreduce"):
                targ = name.split("-")[0][:-4]
                if targ not in targetlist:
                    continue
                holder = root + "/" + name
                (time, intensity, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder)
    
                #get discovery time
                d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                tmin = time[0]
                time = time - tmin
                #disctime = discoverytime-tmin
                
                trlc = etsMAIN(foldersave, bigInfoFile)
                
                trlc.load_single_lc(time, intensity, error, discoverytime, 
                                   targetlabel, sector, camera, ccd, lygosbg=None)
                
                
                filterMade = trlc.window_rms_filt(plot=False)
                if "2018fhw" in targetlabel:
                    filterMade[1040:1080] = 0.0
                    
                trlc.pre_run_clean(1, cutIndices=filterMade, 
                                   binYesNo = binning, fraction = fraction)
                
                internal = trlc.targetlabel + trlc.sector + trlc.camera + trlc.ccd
                
                filepath = "{f}{i}/{t}/{i}-{t}-output-params.txt".format(f=foldersave, 
                                                                         i=internal,
                                                                         t=filetag)
 
    
                t0,A,beta,B, bicrow, conv = ba.extract_singlepowerparams_from_file(filepath)
                
                t1 = trlc.time - t0
                model = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + B
    
                
                ax[m,n].scatter(trlc.time, trlc.intensity, color='black', s=2, label="data")
                ax[m,n].plot(trlc.time, model, color='red', label='model')
                ax[m][n].set_title(trlc.targetlabel, fontsize=14)
                ax[m][n].axvline(trlc.disctime-tmin, color="brown", linestyle = "dotted",
                                 label="Disc. time")
                ax[m][n].axvline(t0, label="t0", color="green", linestyle="dashed")
                ax[m][n].set_xlabel("BJD - {timestart:.2f}".format(timestart=tmin), fontsize=10)
                ax[m][n].set_ylabel("flux (e-/s)", fontsize=20)
                ax[m][n].tick_params('x', labelsize=10)
                ax[m][n].legend(loc="upper left", fontsize=8)
                
                if n<(ncols-1):
                    n=n+1
                else:
                    m=m+1
                    n=0
    fig.suptitle("Collated Power Law Fits")
    fig.tight_layout()
    fig.show()   
    plt.savefig("{f}/collated-single-powerlaws.png".format(f=foldersave))            
    return