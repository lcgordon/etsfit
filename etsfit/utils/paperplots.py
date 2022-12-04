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

def beta_histo(save_dir, betaall, convy ):

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
    plt.savefig("{f}/beta-histogram.png".format(f=save_dir))
    plt.show()
    plt.close()

def plot_t0_disc_beta(save_dir, disc_all, params_all, upper_e, lower_e, 
                      params_all2, upper_e2, lower_e2):
    rcParams['figure.figsize'] = 8,8
    i=1
    for k in params_all.keys():
        t_b = disc_all[k] - params_all[k][0]
        print(k, t_b)
        beta = params_all[k][2]
        u_e_t = upper_e[k][0]
        l_e_t = lower_e[k][0]
        u_e_b = upper_e[k][2]
        l_e_b = lower_e[k][2]
        if i==1:
            plt.errorbar(beta, t_b, xerr=np.asarray([[l_e_b],[u_e_b]]),
                         yerr=np.asarray([[l_e_t],[u_e_t]]), fmt='o',
                         color='blue', label="No GP")
            i=2
        else:
            plt.errorbar(beta, t_b, xerr=np.asarray([[l_e_b],[u_e_b]]),
                         yerr=np.asarray([[l_e_t],[u_e_t]]), fmt='o',
                         color='blue')
    i = 1
    for k in params_all2.keys():
        t_b = disc_all[k] - params_all2[k][2]
        print(k, t_b)
        beta = params_all2[k][4]
        u_e_t = upper_e2[k][2]
        l_e_t = lower_e2[k][2]
        u_e_b = upper_e2[k][4]
        l_e_b = lower_e2[k][4]
        if i==1:
            plt.errorbar(beta, t_b, xerr=np.asarray([[l_e_b],[u_e_b]]),
                         yerr=np.asarray([[l_e_t],[u_e_t]]), fmt='o',
                         color='green', label="With GP")
            i=2
        else:
            plt.errorbar(beta, t_b, xerr=np.asarray([[l_e_b],[u_e_b]]),
                         yerr=np.asarray([[l_e_t],[u_e_t]]), fmt='o',
                         color='green')

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"Disc. time. - $t_0$ (JD)")
    plt.title(r"Time between $t_0$ and Disc. time versus $\beta$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("{f}/disc-t0-beta.png".format(f=save_dir))
    rcParams['figure.figsize'] = 16,6
    return

def big_plot_singlepower(TNSFile, data_dir, save_dir, targetlist,
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

    info = pd.read_csv(TNSFile)
    #i = 0
    m = 0
    n = 0
    (disc_all, params_all, 
     converged_all, 
     upper_all, lower_all) = ba.retrieve_all_singlepower06(TNSFile, data_dir, 
                                                           save_dir, gList)
    
    for root, dirs, files in os.walk(data_dir):
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
                
                trlc = etsMAIN(save_dir, TNSFile)
                
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
                
                filepath = "{f}{i}/{t}/{i}-{t}-output-params.txt".format(f=save_dir, 
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
    plt.savefig("{f}/collated-single-powerlaws.png".format(f=save_dir))            
    return trlc
from celerite.modeling import Model
from scipy.optimize import minimize
import celerite
from celerite import terms

def big_plot_celerite(TNSFile, data_dir, save_dir, targetlist,
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

    info = pd.read_csv(TNSFile)
    #i = 0
    m = 0
    n = 0
    (disc_all, params_all, 
     converged_all, 
     upper_all, lower_all) = ba.retrieve_all_singlepower06celerite(TNSFile, data_dir, 
                                                           save_dir, gList)
    
    for root, dirs, files in os.walk(data_dir):
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
                
                trlc = etsMAIN(save_dir, TNSFile)
                
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
                
                # filepath = "{f}{i}/{t}/{i}-{t}-output-params.txt".format(f=save_dir, 
                #                                                          i=internal,
                #                                                          t=filetag)
 
                t0, A, beta, B, sig, rho = params_all[trlc.targetlabel]

                
                t1 = trlc.time - t0
                model = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + B
                kernel = terms.Matern32Term(log_rho=rho, log_sigma=sig)
                gp = celerite.GP(kernel, mean=0.0)
                gp.compute(trlc.time, trlc.error)
                bg = gp.predict(trlc.flux-mod, trlc.time, return_cov=False)
                model = model + bg
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
    plt.savefig("{f}/collated-celerite.png".format(f=save_dir))            
    return trlc
#%%
data_dir = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
save_dir = "/Users/lindseygordon/research/urop/paperOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"
TNSFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"

gList = ["2018exc", "2018fhw", "2018fub", "2020tld", 
          "2020zbo", "2020hvq", "2018hzh",
          "2020hdw", "2020bj", "2019gqv"]
big_plot_celerite(TNSFile, data_dir, save_dir, gList,
                  " ", fraction = 0.6, binning=False)

#%%
gList = ["2020tld"]

p1 = "-celerite-matern32-mean-model-0.6-output-params.txt"
params1 = [ 15.8931,  5.0456, 1.1874,  8.8434, 1.6000,  -5.2059 ]
p2 = "-celerite-matern32-mean-model-0.6-bounded-output-params.txt"
params2 = [ 14.9917,  1.8097, 1.0108,  -12.8727, 1.5498,  0.0117 ]
p3 = "-celerite-matern32-residual-0.6-output-params.txt"
params3 = [15.5260,  4.2888,  1.2463,  7.0039, 1.6053,  -5.1970 ]
p4 = "-celerite-matern32-residual-0.6-bounded-output-params.txt"
params4 = [15.5231,  4.2843,  1.2485,  6.9995,  1.4976,  0.0004 ]
    
pall = [params1, params2, params3, params4]                
           
ncols = 4
nrows = 2
fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                       figsize=(8*ncols, 3*nrows))

for i in range(4):
    best_mcmc = pall[i]

    tplot = trlc.time + trlc.tmin - 2457000
    dplot = trlc.disctime + trlc.tmin - 2457000
    t0plot = t0 + trlc.tmin - 2457000


    t0, A,beta,B = ets.best_mcmc[0][:4]
    t1 = ets.time - t0
    mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B
    
    ets.gp.set_parameter_vector(ets.best_mcmc[0][4:])
    bg = ets.gp.predict(ets.flux-mod, ets.time, return_cov=False)

