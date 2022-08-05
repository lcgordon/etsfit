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
filesave= "/Users/lindseygordon/research/urop/sn2018fub.csv"
tess.save_lc(filesave)

#%%


import pandas as pd
import numpy as np
import tessreduce as tr
import time
import os


Ia18thFile = "/Users/lindseygordon/research/urop/18thmag_Ia.csv"
info = pd.read_csv(Ia18thFile)
filesavestub = "/Users/lindseygordon/research/urop/tessreduce_lc/"
failures = []


for i in range(50,len(info)):
    try:
        print(i)
        time.sleep(40)
        print(info['Name'].iloc[i])
        targ = info['Name'].iloc[i][3:]
        obs = tr.sn_lookup(targ)
        cdir = "/Users/lindseygordon/.lightkurve-cache/tesscut/"
        tess = tr.tessreduce(obs_list=obs,plot=False,reduce=True)

        holder = ""
        for root, dirs, files in os.walk(cdir):
            for name in files:
                holder = root + "/" + name
                print(holder)
                try:
                    filenamepieces = name.split("-")
                    sector = str( filenamepieces[1][3:])
                    camera = str( filenamepieces[2])
                    ccd = str(filenamepieces[3][0])
                    os.remove(holder)
                    break
                except IndexError:
                    print("eek")
                    os.remove(holder)
                    continue
        print(sector)
        print(camera)
        print(ccd)
        
        #make subfolder to save into 
        targlabel = targ[3:] + sector + camera + ccd 
        newfolder = filesavestub + targlabel + "/"
        if not os.path.exists(newfolder):
            os.mkdir(newfolder)
            filesave = newfolder + targlabel + "-tessreduce.csv"
            tess.save_lc(filesave)
            tess.to_flux()
            filesave = newfolder + targlabel + "-tessreduce-fluxconverted.csv"
            tess.save_lc(filesave)
            
            del(obs)
            del(tess)
        else:
            print("Folder already exists, exiting")
            continue
        
    except ValueError:
        print("value error - something is wrong with vizier as fucking always")
        print("failed??")
        failures.append(i)
        continue
    except IndexError:
        print("index error - tesscut can't find it?")
        continue
    except ConnectionResetError:
        print("failed??")
        failures.append(i)
        continue
    except TimeoutError:
        print("failed??")
        failures.append(i)
        continue
 
    #%%

# =============================================================================
#     
#     
#     
#     
# =============================================================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from etsfit import etsMAIN
from astropy.time import Time
import gc

topfolder = "/Users/lindseygordon/research/urop/tessreduce_lc/"
CBV_folder = "/Users/lindseygordon/research/urop/eleanor_cbv/"
Ia18thFile = "/Users/lindseygordon/research/urop/18thmag_Ia.csv"
foldersave = "/Users/lindseygordon/research/urop/plotOutput/"
quaternion_folder_raw = "/Users/lindseygordon/research/urop/quaternions-raw/"
quaternion_folder_txt = "/Users/lindseygordon/research/urop/quaternions-txt/"
info = pd.read_csv(Ia18thFile)
for root, dirs, files in os.walk(topfolder):
memet    for name in files:
        if name.endswith("-tessreduce"):
            holder = root + "/" + name
            #print(holder)
            loadedraw = pd.read_csv(holder)
            time = Time(loadedraw["time"], format='mjd').jd
            intensity = loadedraw["flux"].to_numpy()
            error = loadedraw["flux_err"].to_numpy()
            #p
            fulllabel = holder.split("/")[-1].split("-")[0]
            targetlabel = fulllabel[0:7]
            sector = fulllabel[-4:-2]
            camera = fulllabel[-2]
            ccd = fulllabel[-1]
            print(targetlabel, sector, camera, ccd)
            #get discovery time
            d = info[info["Name"].str.contains(targetlabel)]["Discovery Date (UT)"]
            discoverytime = Time(d.iloc[0], format = 'iso', scale='utc').jd
            print(discoverytime)

            trlc = etsMAIN(foldersave, Ia18thFile)
            
            trlc.load_single_lc(time, intensity, error, discoverytime, 
                               targetlabel, sector, camera, ccd, lygosbg=None)
            
            trlc.use_quaternions_cbvs(CBV_folder, quaternion_folder_raw, 
                                      quaternion_folder_txt)
            
            filterMade = trlc.window_rms_filt()
            trlc.pre_run_clean(2, cutIndices=filterMade, 
                               binYesNo = False, fraction = None)
            trlc.run_MCMC(n1=10000, n2=30000, thinParams = None,
                         saveBIC=False, args=None, logProbFunc = None, 
                         plotFit = None,
                         filesavetag=None,
                         labels=None, init_values=None)
            del(loadedraw)
            del(trlc)
            gc.collect()
    
#%%
