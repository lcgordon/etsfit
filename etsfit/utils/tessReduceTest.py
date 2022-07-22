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


Ia18thFile = "D:/SNPROJ/18thmag_Ia.csv"
info = pd.read_csv(Ia18thFile)
filesavestub = "D:/SNPROJ/tessreduce_lc/"
for i in (3,4):
    try:
        time.sleep(10)
        print(info['Name'].iloc[i])
        targ = info['Name'].iloc[i]
        obs = tr.sn_lookup(targ)
        cdir = "D:/SNPROJ/downloaddir/"
        tess = tr.tessreduce(obs_list=obs,plot=False,reduce=True, 
                             cache_dir=cdir)
        #tess.to_flux()
        filesave = filesavestub + targ[3:] + "-tessreduce-flux.csv"
        tess.save_lc(filesave)
    except ValueError:
        print("value error - something is wrong with vizier as fucking always")
        continue
    except IndexError:
        print("index error - tesscut can't find it?")
        continue
    
    
#%%
import matplotlib.pyplot as plt
loaded = pd.read_csv("D:/SNPROJ/tessreduce_lc/2018exc-tessreduce-flux")
plt.scatter(loaded["time"], loaded["flux"])
    
