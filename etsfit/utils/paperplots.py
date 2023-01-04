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

def plot_A_beta(save_dir, disc_all, params_all, upper_e):
    """ 
    give params_all, upper_e as arrays of dictionaries
    """
    rcParams['figure.figsize'] = 8,8
    i=0
    color_arr = ['pink','red', 'orange', 'yellow', 'lime', 
                 'darkgreen', 'cyan', 'blue', 'purple', 'black']
    for k in params_all[0].keys(): #for each SN
        tplot = []
        tplote = []
        bplot = []
        bplote = []
        for j in range(len(params_all)): #for each input set of parameters
            # get set of params
            pall = params_all[j]
            uall = upper_e[j]
            # calc the values
            # put into the arrays
            #tplot.append( disc_all[k] - pall[k][0] )
            tplot.append(pall[k][1])
            tplote.append( uall[k][1])
            bplot.append(pall[k][2])
            bplote.append(uall[k][2])

        # plot the arrays in one color (only one label)
        plt.errorbar(tplot, bplot, xerr=np.asarray(tplote), 
                     yerr=np.asarray(bplote),  fmt='o', color=color_arr[i])
        i = i + 1


    plt.ylabel(r"$\beta$")
    plt.xlabel(r"A")
    plt.title(r"A versus $\beta$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("{f}/A-beta.png".format(f=save_dir))
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
     upper_all, lower_all) = ba.retrieve_all_singlepower(TNSFile, data_dir, save_dir, 
                                                         targetlist, 
                                   datatag="-tessreduce", paramstag="singlepower-0.6")
    
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



def big_plot_celerite(TNSFile, data_dir, save_dir, targetlist,
                         filetag, 
                         fraction = None, binning=False):
    """ 
    Plot a bunch of single powerlaws together
    """
    import etsfit.utils.batch_analyze as ba
    import etsfit.utils.utilities as ut
    import celerite
    from celerite import terms
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
     upper_all, lower_all) = ba.retrieve_all_celerite(TNSFile, data_dir, save_dir, 
                                                         targetlist, 
                                   datatag="-tessreduce", paramstag="singlepower-0.6")
    
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
                bg = gp.predict(trlc.flux-model, trlc.time, return_cov=False)
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
# big_plot_celerite(TNSFile, data_dir, save_dir, gList,
#                   " ", fraction = 0.6, binning=False)
import etsfit.utils.batch_analyze as ba

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
         
pall = [params_all1, params_all2, params_all3, params_all4, params_all5]    
uall = [upper_all1, upper_all2, upper_all3, upper_all4, upper_all5]                          
plot_t0_disc_beta(save_dir, disc_all1, pall, uall)

