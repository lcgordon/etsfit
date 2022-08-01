# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:48:30 2022

@author: conta
"""
#running tessreduce
import tessreduce as tr
obs = tr.sn_lookup('sn2018fub')
tess = tr.tessreduce(obs_list=obs,plot=False,reduce=True)
#tess.to_flux()
tess.plotter()
filesave= "D:/SNPROJ/tessreduce_lc/sn2018fub.csv"
tess.save_lc(filesave)

#%%
#plt.scatter(tess.lc[0],tess.lc[1])


import pandas as pd
import numpy as np
import tessreduce as tr
import time
import os

#Ia18thFile = "D:/SNPROJ/18thmag_Ia.csv"

Ia18thFile = "C:/Users/conta/UROP/18thmag_Ia.csv"
info = pd.read_csv(Ia18thFile)
filesavestub = "C:/Users/conta/UROP/"
for i in (3,4):
    try:
        time.sleep(10)
        print(info['Name'].iloc[i])
        targ = info['Name'].iloc[i]
        obs = tr.sn_lookup(targ)
        #cdir = "D:/SNPROJ/downloaddir/"
        tess = tr.tessreduce(obs_list=obs,plot=False,reduce=True)#, 
                             #cache_dir=cdir)
        cdir ="C:/Users/conta/.lightkurve-cache/tesscut/"
        holder = ""
        for root, dirs, files in os.walk(cdir):
            for name in files:
                print(name)
                holder = root + "/" + name
                filenamepieces = name.split("-")
                sector = str( filenamepieces[1][3:])
                camera = str( filenamepieces[2])
                ccd = str(filenamepieces[3][0])
        print(sector)
        print(camera)
        print(ccd)

        filesave = filesavestub + targ[3:] + sector + camera + ccd + "-tessreduce.csv"
        tess.save_lc(filesave)
        tess.to_flux()
        filesave = filesavestub + targ[3:] + sector + camera+ccd + "-tessreduce-fluxconverted.csv"
        tess.save_lc(filesave)
        
        del(obs)
        del(tess)
        #WHY cant i delete the stupid file

        


    except ValueError:
        print("value error - something is wrong with vizier as fucking always")
        continue
    except IndexError:
        print("index error - tesscut can't find it?")
        continue
 
    #%%


# =============================================================================
#     
#     
#     
#     
# =============================================================================
    
    
#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
raw = "C:/Users/conta/UROP/2018exc0111-tessreduce"
loadedraw = pd.read_csv(raw)
plt.scatter(loadedraw["time"], loadedraw["flux"])
plt.show()
plt.close()

flux = "C:/Users/conta/UROP/2018exc0111-tessreduce-fluxconverted"
loadedflux = pd.read_csv(flux)
plt.scatter(loadedflux["time"], loadedflux["flux"])
#%%
from etsfit import etsMAIN
from astropy.time import Time
folderSAVE = "C:/Users/conta/UROP/2018exc_tessreducetest/"
Ia18thFile = "C:/Users/conta/UROP/18thmag_Ia.csv"
info = pd.read_csv(Ia18thFile)
trlc = etsMAIN(folderSAVE, Ia18thFile)

time = loadedraw["time"].to_numpy()
intensity = loadedraw["flux"].to_numpy()
error = loadedraw["flux_err"].to_numpy()

d = info[info["Name"].str.contains("2018exc")]["Discovery Date (UT)"]
d = Time(d.iloc[0], format = 'iso', scale='utc')
discoverytime = d.jd
#tessreduce doesn't save its camera/ccd values which is SO annoying 
targetlabel = raw.split("/")[-1].split("-")[0]
sector = targetlabel[-4:-2]
camera = targetlabel[-2]
ccd = targetlabel[-1]
#%%
trlc.load_single_lc(time, intensity, error, discoverytime, 
                   targetlabel, sector, camera, ccd, lygosbg=None)
#%%
filterMade = trlc.window_rms_filt()

#%%

trlc.pre_run_clean(1, cutIndices=filterMade, binYesNo = False, fraction = None)

    #%%
#how to clean up this light curve -> bc i want to make it all positive?
#possibly i can just extend the boundaries on the priors to fit like this?

