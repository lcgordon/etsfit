#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:28:17 2023

@author: lindseygordon
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
import etsfit.utils.batch_analyze as ba
import math
rcParams['font.family'] = 'serif'
import etsfit.utils.batch_analyze as ba
import etsfit.utils.utilities as ut

TNSFile = "/Users/lindseygordon/research/paper_outputs/TNS_10_paper.csv"
data_dir = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
save_dir = "/Users/lindseygordon/research/paper_outputs/4-1-60-run/"

targetlist = gList = ["2018exc", "2018fhw", "2018fub", "2018hzh","2019gqv",
          "2020bj","2020hdw","2020hvq","2020tld", "2020zbo"]

filetag="singlepower-0.6"
fraction = 0.6
binning=False

ncols = 3
nrows_max = len(targetlist)
nrows = 10

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
                
                if targ not in targetlist:
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
                
                ax[m,0].plot(tplot, bg, color='blue', label='Baseline Model', lw=3)
                ax[m,0].plot(tplot, mod, color='green', label='SN Model', lw=3)
                
                ax[m,0].plot(tplot, model, color='red', label='Full Model', lw=3)
                #ax[m][n].set_title(trlc.targetlabel, fontsize=14)
                ax[m][0].axvline(trlc.disctime+trlc.tmin-2457000, 
                                 color="brown", linestyle = "dotted",
                                 label="Disc. Time")
                ax[m][0].axvline(t0+trlc.tmin - 2457000, label="t0", 
                                 color="navy", linestyle="dashed")
                
                ax[m,1].scatter(tplot, trlc.flux-model, color='black', s=2, 
                                label="Residual")
                ax[m,1].axhline(0, linestyle='dotted', color='blue', 
                                label="Zero", lw=3)
                
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
    
    fig.suptitle("Collated Power Law Fits")
    plt.tight_layout() 
    plt.savefig("{f}/collated-single-powerlaws-withhist-all.png".format(f=save_dir))    
    plt.show()
    plt.close()        