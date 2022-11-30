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
    #plt.legend()
    plt.tight_layout()
    plt.savefig("{f}/beta-histogram.png".format(f=foldersave))
    plt.show()
    plt.close()

def plot_t0_disc_beta(foldersave, disc_all, params_all, upper_e, lower_e):
    rcParams['figure.figsize'] = 8,8
    for k in params_all.keys():
        t_b = disc_all[k] - params_all[k][0]
        print(k, t_b)
        beta = params_all[k][2]
        u_e_t = upper_e[k][0]
        l_e_t = lower_e[k][0]
        u_e_b = upper_e[k][2]
        l_e_b = lower_e[k][2]
        plt.errorbar(beta, t_b, xerr=np.asarray([[l_e_b],[u_e_b]]),
                     yerr=np.asarray([[l_e_t],[u_e_t]]), fmt='o',
                     color='blue')

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"Disc. time. - $t_0$ (JD)")
    plt.title(r"Time between $t_0$ and Disc. time versus $\beta$")
    #plt.legend()
    plt.tight_layout()
    plt.savefig("{f}/disc-t0-beta.png".format(f=foldersave))
    rcParams['figure.figsize'] = 16,6
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
    (disc_all, params_all, 
     converged_all, 
     upper_all, lower_all) = ba.retrieve_all_singlepower06(bigInfoFile, datafolder, 
                                                           foldersave, gList)
    
    for root, dirs, files in os.walk(datafolder):
        for name in files:
            if name.endswith("-tessreduce"):
                targ = name.split("-")[0][:-4]
                if targ not in targetlist:
                    continue
                holder = root + "/" + name
                (time, flux, error, targetlabel, 
                 sector, camera, ccd) = ut.tr_load_lc(holder)
    
                #get discovery time
                d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                # tmin = time[0]
                # time = time - tmin
                #disctime = discoverytime-tmin
                
                trlc = etsMAIN(foldersave, bigInfoFile)
                
                trlc.load_single_lc(time, flux, error, discoverytime, 
                                   targetlabel, sector, camera, ccd)
                
                
                filterMade = trlc.window_rms_filt(plot=False)
                if "2018fhw" in targetlabel:
                    filterMade[1040:1080] = 0.0
                if "2020hdw" in targetlabel:
                    filterMade[0:45] = 0.0
                    filterMade[610:685] = 0.0
                    
                trlc.pre_run_clean(1, flux_mask=filterMade, 
                                   binning = binning, fraction = fraction)
                
                internal = trlc.targetlabel + trlc.sector + trlc.camera + trlc.ccd
                
                filepath = "{f}{i}/{t}/{i}-{t}-output-params.txt".format(f=foldersave, 
                                                                         i=internal,
                                                                         t=filetag)
 
                t0, A, beta, B = params_all[trlc.targetlabel]
                
                t1 = trlc.time - t0
                model = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + B
                
                tplot = trlc.time + trlc.tmin - 2457000
                
                #print(trlc.tmin)
    
                
                ax[m,n].scatter(tplot, trlc.flux, color='black', s=2, label=trlc.targetlabel)
                ax[m,n].plot(tplot, model, color='red', label='Model')
                #ax[m][n].set_title(trlc.targetlabel, fontsize=14)
                ax[m][n].axvline(trlc.disctime+trlc.tmin-2457000, color="brown", linestyle = "dotted",
                                 label="Disc. time")
                ax[m][n].axvline(t0+trlc.tmin - 2457000, label="t0", color="green", linestyle="dashed")
                ax[m][n].set_xlabel("Time [BJD-2457000]", fontsize=16)
                ax[m][0].set_ylabel("flux (e-/s)", fontsize=16)
                ax[m][n].tick_params('x', labelsize=14)
                ax[m][n].tick_params('y', labelsize=14)
                ax[m][n].legend(fontsize=12)
                
                if n<(ncols-1):
                    n=n+1
                else:
                    m=m+1
                    n=0
    #fig.suptitle("Collated Power Law Fits")
    fig.tight_layout()
    fig.show()   
    plt.savefig("{f}/collated-single-powerlaws.png".format(f=foldersave))            
    return trlc

datafolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
foldersave = "/Users/lindseygordon/research/urop/paperOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"
bigInfoFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"

gList = ["2018exc", "2018fhw", "2018fub", "2020tld", 
         "2020zbo", "2020hvq", "2018hzh",
         "2020hdw", "2020bj", "2019gqv"]

import etsfit.utils.batch_analyze as ba

#disc_all, params_all, converged_all, upper_all, lower_all = ba.retrieve_all_singlepower06(bigInfoFile, datafolder, foldersave, gList)

# big_plot_singlepower(bigInfoFile, datafolder, foldersave, gList,
#                          "-singlepower-0.6", 
#                          fraction = 0.6, binning=False)
