#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 08:44:40 2022

@author: lindseygordon

paper plot generator functions
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
import celerite
import etsfit.utils.utilities as ut
from celerite import terms
import etsfit.utils.parameter_retrieval as ba
import math
rcParams['font.family'] = 'serif'

# data_dir = "/Users/lindseygordon/research/urop/tessreduce_lc/"
# CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
# save_dir = "/Users/lindseygordon/research/urop/paperOutput/"
# quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
# quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"
# TNSFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"

# gList = ["2018exc", "2018fhw", "2018fub", "2018hzh","2019gqv",
#          "2020bj","2020hdw","2020hvq","2020tld", "2020zbo"]



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


def plot_2param_comp(save_dir, disc_all, params_all, upper_e,
                     index_x, l_x, index_y, l_y):
    
    rcParams['figure.figsize'] = 8,8
    i=0
    color_arr = ['pink','red', 'orange', 'yellow', 'lime', 
                 'darkgreen', 'cyan', 'blue', 'purple', 'black']
    
    for k in params_all[0].keys(): #for each SN
        plotx = []
        plotx_e = []
        ploty = []
        ploty_e = []
        for j in range(len(params_all)):
            pall = params_all[j]
            uall = upper_e[j]
            if index_x == 0: #time
                plotx.append(disc_all[k] - pall[k][index_x])
            else:
                plotx.append(pall[k][index_x])
            plotx_e.append(uall[k][index_x])
            
            if index_y == 0:
                ploty.append(disc_all[k] - pall[k][index_y])
            else:
                ploty.append(pall[k][index_y])
            ploty_e.append(uall[k][index_y])


        # plot the arrays in one color (only one label)
        plt.errorbar(plotx, ploty, xerr=np.asarray(plotx_e), 
                     yerr=np.asarray(ploty_e),  fmt='o', color=color_arr[i])
        i = i + 1


    plt.ylabel(l_y)
    plt.xlabel(l_x)
    plt.title("{} and {}".format(l_x, l_y))
    plt.legend()
    plt.tight_layout()
    plt.savefig("{f}/{a}-{b}-gp-param-comparison.png".format(f=save_dir, 
                                                             a=l_x, b=l_y))
    rcParams['figure.figsize'] = 16,6
    return

def load_all(save_dir, data_dir, TNSFile, gList):
    """ load all paramsets"""

    
    (disc_all1, params_all1, 
      converged_all1, upper_all1, 
      lower_all1) = ba.retrieve_all_singlepower(TNSFile, data_dir, save_dir, gList, 
                                    datatag="-tessreduce", paramstag="singlepower-0.6")
    
    (disc_all2, params_all2, 
     converged_all2, upper_all2, 
     lower_all2) = ba.retrieve_all_celerite(TNSFile, data_dir, save_dir, gList,
                               datatag="-tessreduce", 
                               paramstag="residual-0.6")
                                            
    (disc_all3, params_all3, 
     converged_all3, upper_all3, 
     lower_all3) = ba.retrieve_all_celerite(TNSFile, data_dir, save_dir, gList,
                               datatag="-tessreduce", 
                               paramstag="residual-0.6-bounded-0-25day")     
                                            
    (disc_all4, params_all4, 
     converged_all4, upper_all4, 
     lower_all4) = ba.retrieve_all_celerite(TNSFile, data_dir, save_dir, gList,
                               datatag="-tessreduce", 
                               paramstag="residual-0.6-bounded-0-5day") 
                                            
    (disc_all5, params_all5, 
     converged_all5, upper_all5, 
     lower_all5) = ba.retrieve_all_celerite(TNSFile, data_dir, save_dir, gList,
                               datatag="-tessreduce", 
                               paramstag="residual-0.6-bounded")                                        
             
    params_all = {"singlepower-0.6":params_all1,
                  "residual-0.6":params_all2,
                  "residual-0.6-bounded-0-25day":params_all3,
                  "residual-0.6-bounded-0-5day":params_all4,
                  "residual-0.6-bounded":params_all5}
    upper_e = {"singlepower-0.6":upper_all1,
                  "residual-0.6":upper_all2,
                  "residual-0.6-bounded-0-25day":upper_all3,
                  "residual-0.6-bounded-0-5day":upper_all4,
                  "residual-0.6-bounded":upper_all5}
    return params_all, upper_e, disc_all1
    


def big_plot_singlepower(TNSFile, data_dir, save_dir, targetlist,
                         filetag="singlepower-0.6", 
                         fraction = None, binning=False):
    """ 
    Plot a bunch of single powerlaws together
    """
    import etsfit.utils.parameter_retrieval as ba
    import etsfit.utils.utilities as ut
    
    
    ncols = 3
    nrows_max = len(targetlist)
    nrows = 5
    
    nfigs = math.ceil(nrows_max/nrows)
    info = pd.read_csv(TNSFile)
    
    (disc_all, params_all, 
     converged_all, 
     upper_all, lower_all) = ba.retrieve_all_singlepower(TNSFile, data_dir, save_dir, 
                                                         targetlist, 
                                   datatag="-tessreduce", paramstag=filetag)
    
    
    for f in range(nfigs):
        #make figure:
        fig, ax = plt.subplots(nrows, ncols, sharex=False, #sharey=True,
                               figsize=(8*ncols, 3*nrows), 
                               gridspec_kw={'width_ratios': [3, 3, 1]})
    
        m=0 #row
    
        for root, dirs, files in os.walk(data_dir):
            for name in files:
                if name.endswith("-tessreduce"):
                    targ = name.split("-")[0][:-4]
                    #print(f*5+m, targ, targetlist[f*5 + m])
                    
                    if targ not in targetlist[f*5:f*5+5]:
                        continue
                    
                    for i in range(len(targetlist)): #plots them in order
                        if targ in targetlist[i]:
                            m=i - f*5
                            break
                        
                    
                    holder = root + "/" + name
                    (time, flux, error, targetlabel, 
                     sector, camera, ccd) = ut.tr_load_lc(holder)
        
                    #get discovery time
                    d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
                    discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
                    
                    trlc = etsMAIN(save_dir, TNSFile, plot=False)
                    
                    trlc.load_single_lc(time, flux, error, discoverytime, 
                                       targetlabel, sector, camera, ccd)
                    
                    
                    filterMade = trlc.window_rms_filt()
                    if "2018fhw" in targetlabel:
                        filterMade[1040:1080] = 0.0
                    if "2020hdw" in targetlabel:
                        filterMade[0:45] = 0.0
                        filterMade[610:685] = 0.0
                        
                    trlc.pre_run_clean(1, flux_mask=filterMade, 
                                       binning = binning, fraction = fraction)
                    
                    internal = trlc.targetlabel + trlc.sector + trlc.camera + trlc.ccd
                    
                    filepath = f"{save_dir}{internal}/{filetag}/{internal}-{filetag}-output-params.txt"
     
                    t0, A, beta, B = params_all[trlc.targetlabel]
                    tplot = trlc.time + trlc.tmin - 2457000
                    
                    t1 = trlc.time - t0
                    mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False)
                    bg = np.ones(len(tplot)) * B
                    model =  mod + bg
                    
                    
        
                    
                    ax[m,0].scatter(tplot, trlc.flux, color='black', s=2, 
                                    label="Data")
                    
                    ax[m,0].plot(tplot, bg, color='blue', label='Baseline Model')
                    ax[m,0].plot(tplot, mod, color='green', label='SN Model')
                    
                    ax[m,0].plot(tplot, model, color='red', label='Full Model')
                    #ax[m][n].set_title(trlc.targetlabel, fontsize=14)
                    ax[m][0].axvline(trlc.disctime+trlc.tmin-2457000, 
                                     color="brown", linestyle = "dotted",
                                     label="Disc. Time")
                    ax[m][0].axvline(t0+trlc.tmin - 2457000, label="t0", 
                                     color="navy", linestyle="dashed")
                    
                    ax[m,1].scatter(tplot, trlc.flux-model, color='black', s=2, 
                                    label="Residual")
                    ax[m,1].axhline(0, linestyle='dotted', color='blue', 
                                    label="Zero")
                    
                    #labels!
                    if (m==nrows-1):
                        ax[m][0].set_xlabel("Time [BJD-2457000]", fontsize=18)
                        ax[m][1].set_xlabel("Time [BJD-2457000]", fontsize=18)
                        
                    ax[m][0].set_ylabel("Rel. Flux", fontsize=18)
                    ax[m][0].tick_params('x', labelsize=14)
                    ax[m][0].tick_params('y', labelsize=14)
                    ax[m][1].tick_params('x', labelsize=14)
                    ax[m][1].tick_params('y', labelsize=14)
                    ax[m][2].tick_params('x', labelsize=14)
                    ax[m][2].tick_params('y', labelsize=14)
                    ax[m][0].set_title(trlc.targetlabel + " Sector " + trlc.sector, fontsize=20)
                    ax[m][1].set_title(trlc.targetlabel + " Residual", fontsize=20)
                    if (m==nrows-1):
                        ax[m][0].legend(fontsize=16, loc='right')
                        ax[m][1].legend(fontsize=16)
                        
                    ax[m, 1].set_ylim(ax[m,0].get_ylim())
                    
                    ax[m,2].hist(trlc.flux-model, color='black', density=True, 
                                 alpha=0.8)
                    ax[m,2].set_title("Residual Hist.", fontsize=20)
                    
                    m += 1
        
        fig.suptitle("Collated Power Law Fits (Part {f})".format(f=f+1))
        plt.tight_layout() 
        plt.savefig("{f}/collated-single-powerlaws-withhist-{i}.png".format(f=save_dir, i=f+1))    
        plt.show()
        plt.close()        
    return


def big_plot_celerite(TNSFile, data_dir, save_dir, targetlist,
                         filetag="celerite-matern32-residual-0.6", 
                         fraction = None, binning=False):
    """ 
    Plot a bunch of single powerlaws together
    """
    import etsfit.utils.parameter_retrieval as ba
    import etsfit.utils.utilities as ut
    
    
    ncols = 3
    nrows_max = len(targetlist)
    nrows = 5
    
    nfigs = math.ceil(nrows_max/nrows)
    info = pd.read_csv(TNSFile)
    
    (disc_all, params_all, 
     converged_all, 
     upper_all, lower_all) = ba.retrieve_all_celerite(TNSFile, data_dir, save_dir, 
                                                      targetlist, datatag="-tessreduce", 
                                                      paramstag=filetag)
    #print(params_all)
    
    for f in range(nfigs):
        #make figure:
        fig, ax = plt.subplots(nrows, ncols, sharex=False, #sharey=True,
                               figsize=(8*ncols, 3*nrows), 
                               gridspec_kw={'width_ratios': [3, 3, 1]})
    
        m=0 #row
    
        for root, dirs, files in os.walk(data_dir):
            for name in files:
                if name.endswith("-tessreduce"):
                    targ = name.split("-")[0][:-4]
                    #print(f*5+m, targ, targetlist[f*5 + m])
                    
                    if targ not in targetlist[f*5:f*5+5]:
                        continue
                    
                    for i in range(len(targetlist)): #plots them in order
                        if targ in targetlist[i]:
                            m=i - f*5
                            break
                        
                    
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
     
        
                    t0, A, beta, B, sig, rho = params_all[trlc.targetlabel]

                    
                    t1 = trlc.time - t0
                    mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + B
                    kernel = terms.Matern32Term(log_rho=rho, log_sigma=sig)
                    gp = celerite.GP(kernel, mean=0.0)
                    gp.compute(trlc.time, trlc.error)
                    bg = gp.predict(trlc.flux-mod, trlc.time, return_cov=False)
                    model = mod + bg
                    tplot = trlc.time + trlc.tmin - 2457000
                    
                    
                    ax[m,0].scatter(tplot, trlc.flux, color='black', s=2, 
                                    label="Data")
                    
                    ax[m,0].plot(tplot, bg, color='blue', label='Baseline Model')
                    ax[m,0].plot(tplot, mod, color='green', label='SN Model')
                    
                    ax[m,0].plot(tplot, model, color='red', label='Full Model')
                    ax[m][0].axvline(trlc.disctime+trlc.tmin-2457000, 
                                     color="brown", linestyle = "dotted",
                                     label="Disc. Time")
                    ax[m][0].axvline(t0+trlc.tmin - 2457000, label="t0", 
                                     color="navy", linestyle="dashed")
                    
                    ax[m,1].scatter(tplot, trlc.flux-model, color='black', s=2, 
                                    label="Residual")
                    ax[m,1].axhline(0, linestyle='dotted', color='blue', 
                                    label="Zero")
                    
                    #labels!
                    if (m==nrows-1):
                        ax[m][0].set_xlabel("Time [BJD-2457000]", fontsize=18)
                        ax[m][1].set_xlabel("Time [BJD-2457000]", fontsize=18)
                        
                    ax[m][0].set_ylabel("Rel. Flux", fontsize=18)
                    ax[m][0].tick_params('x', labelsize=14)
                    ax[m][0].tick_params('y', labelsize=14)
                    ax[m][1].tick_params('x', labelsize=14)
                    ax[m][1].tick_params('y', labelsize=14)
                    ax[m][2].tick_params('x', labelsize=14)
                    ax[m][2].tick_params('y', labelsize=14)
                    ax[m][0].set_title(trlc.targetlabel + " Sector " + trlc.sector, fontsize=20)
                    ax[m][1].set_title(trlc.targetlabel + " Residual", fontsize=20)
                    if (m==0):
                        ax[m][0].legend(fontsize=16, loc='upper left')
                        ax[m][1].legend(fontsize=16)
                        
                    ax[m, 1].set_ylim(ax[m,0].get_ylim())
                    
                    ax[m,2].hist(trlc.flux-model, color='black', density=True, 
                                 alpha=0.8)
                    ax[m,2].set_title("Residual Hist.", fontsize=20)
                    
                    m += 1
        
        fig.suptitle("Collated Celerite + Power Law Fits (Part {f})".format(f=f+1))
        plt.tight_layout()
        plt.savefig("{f}/collated-celerite-withhist-{i}.png".format(f=save_dir, i=f+1))    
        plt.close()        
    return

# big_plot_celerite(TNSFile, data_dir, save_dir, gList,
#                           filetag="celerite-matern32-residual-0.6", 
#                           fraction = 0.6, binning=False)


def res_cel_histos(save_dir, data_dir, gList, TNSFile):

    rcParams['figure.figsize'] = 8,8 
    params_all, upper_e, disc_all = load_all(save_dir, data_dir, TNSFile, gList)
    # these stupid histograms: 
    for i in range(len(gList)): # for each target
        print(gList[i])
        # load in data
        for root, dirs, files in os.walk(data_dir):
            for name in files:
                if name.endswith("-tessreduce"):
                    targ = name.split("-")[0][:-4]
                    #print(targ)
                    if targ != gList[i]:
                        continue
                    holder = root + "/" + name
    
                    (time, flux, 
                     error, targetlabel, 
                     sector, camera, ccd) = ut.tr_load_lc(holder, printname=True)
                    
                    disctime = ut.get_disctime(TNSFile, targetlabel)
                    continue
            continue
        #plt.plot(time, flux, label=targetlabel)
        tmin = time[0]
        disctime = disctime - tmin
        t = time - tmin
    
        for k in params_all.keys(): #for each fit style
            #generate model, get residual
            par = params_all[k][gList[i]]
            #print(k, par)
            
            t0, A,beta,B = par[:4]
            t1 = t - t0
            mod = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + 1 + B
            bg = 0
            if len(par) > 4:
                kernel = terms.Matern32Term(log_sigma=par[4], 
                                            log_rho=par[5])
                
                gp = celerite.GP(kernel, mean=0.0)
                gp.compute(t, error)
                bg = gp.predict(flux-mod, t, return_cov=False)
                
            model = mod + bg
            residual = flux - model
            #make histo
            plt.hist(residual, bins = 20, alpha=0.2, label=k)
        
        #format + save
        plt.title(targetlabel + " residual histogram")
        plt.legend(fontsize=12)
        plt.ylabel("#")
        plt.xlabel("Residual Flux")
        plt.tight_layout()
        plt.savefig("{f}/{t}-residual-hist-celerites.png".format(f=save_dir,
                                                                 t=targetlabel))
        plt.show()
        plt.close()
        
    return
            
                   
            
            
        
