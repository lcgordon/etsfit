# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:59:01 2021
Updated Nov 21 2022

@author: Lindsey Gordon

Utility functions, mostly to do with data manipulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.time import Time
import tessreduce as tr
import time

def load_lygos_csv(file):
    """
    Load data from a lygos rflxtarg csv file 
    Assumes 3 columns time/intensity/error AND that there is an rflx file
    in the same folder containing the background data
    ---------------------------------------
    Params:
        - file (str) path to an rflxtarg file to be loaded
    ---------------------------------------
    Returns:
        - time, intensity, error, background (arrays)
        
    """
    data = pd.read_csv(file, sep = ',', header = 0)
    t = np.asarray(data[data.columns[0]])
    ints = np.asarray(data[data.columns[1]])
    error = np.asarray(data[data.columns[2]])
    
    data = pd.read_csv(file.replace("rflxtarg", "rflx"), sep = ',', header = 0)
    bg = np.asarray(data[data.columns[-2]])
    
    return t, ints, error, bg
 
def get_disctime(file, name):
    """
    Helper function to retrieve discovery time from a big TNS file
    ---------------------------------------
    Params:
        - file (str) path to the TNS file
        - name (str) SN to retrieve, ie, '2020abc'
    ---------------------------------------
    Returns:
        - discovery time in JD
    """
    f = pd.read_csv(file)
    d = f[f["Name"].str.contains(name)]["Discovery Date (UT)"]
    d = Time(d.iloc[0], format = 'iso', scale='utc')
    return d.jd
    
def window_rms(time, intensity, innerfilt = None, outerfilt = None,
                    plot=True):
    """ 
    Runs an RMS filter over the light curve and returns an array of 
    0's (bad) and 1's (good) that can be used in the custom masking
    argument of other functions. 
    
    This DOES NOT save the filter output inside the object. 
    
    Defaults the inner window as len(self.time)*0.01 and the outer as
    inner*10.
    
    -------------------------------
    Parameters:
        
        - innerfilt = None by default, can set to an int for the inner
        window of compariosn
        - outerfilt = None by default, can set to an int for the outer
        window of comparison
        - plot (bool) defaults as True, plots light curve w/ mask
    
    """
    if innerfilt is None:
        innersize = int(len(time)*0.01)
    else:
        innersize = innerfilt
    if outerfilt is None:
        outersize = innersize * 10
    else:
        outersize=outerfilt
    #print("window sizes: ", innersize, outersize)
    n = len(time)
    rms_filt = np.ones(n)
    for i in range(n):
        outer_lower = max(0, i-outersize) #outer window, lower bound
        outer_upper = min(n, i+outersize) #outer window, upper bound
        inner_lower = max(0, i-innersize) #inner window, lower bound
        inner_upper = min(n, i+innersize) #inner window, upper bound
        
        outer_window = intensity[outer_lower:outer_upper]
        inner_window = intensity[inner_lower:inner_upper]
        
        std_outer = np.std(outer_window)
        
        rms_outer = np.sqrt(sum([s**2 for s in outer_window])/len(outer_window))
        rms_inner = np.sqrt(sum([s**2 for s in inner_window])/len(inner_window))
        
        if ((rms_inner > (rms_outer + std_outer)) 
            or (rms_inner < (rms_outer - std_outer))):
            rms_filt[inner_lower:inner_upper] = 0 #bad point, discard
            #print(rms_inner, rms_outer, std_outer)
    
    if plot:
        rms_filt_plot = np.nonzero(rms_filt)
        plt.scatter(time, intensity, color='green', label='bad', s=2)
        plt.scatter(time[rms_filt_plot], intensity[rms_filt_plot], 
                    color='blue', s=2, label='good')
        plt.legend()
        plt.show()
        plt.close()
    return rms_filt


def sigmaclip(time, flux,error, bg, axis=0):
    """ 
    5 sigma clip data
    ---------------------------------------
    Params:
        - time, flux, error, bg (arrays)
        - axis = 0 for individual 1D arrays
    ---------------------------------------
    Returns:
        - sigma clipped time, flux, error, bg
    
    """

    from astropy.stats import SigmaClip
    sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
    clipped_inds = np.nonzero(np.ma.getmask(sigclip(flux)))
    time = np.delete(time, clipped_inds)
    flux = np.delete(flux, clipped_inds)
    error = np.delete(error, clipped_inds)
    if bg is not None:
        bg = np.delete(bg, clipped_inds)
        bg = bg / np.median(bg, axis = axis, keepdims=True)
        
    return time,flux,error, bg

def fractionalfit(time, flux, error, bg, fraction, QCBVALL):
    """
    Trims light curve to a chosen fraction of the peak flux
    Considers "above" to be when 10 indices in a row are all brighter
    (Actually looks from the top)
    ---------------------------------------
    Params:
        - time, flux, error, bg (arrays)
        - fraction (0 to 1.0)
        - QCBVALL (array of arrays or None) containing the qcbvs
    ---------------------------------------
    Returns:
        - time, flux, error, bg, QCBVALL
    """
    #fractionalBright = ((flux.max()-1) * fraction) + 1
    #find range, take percent of range, add to min
    fractionalBright = flux.max() - (((flux.max() - flux.min())) * (1-fraction))
    
    p = 0 # if you hit 10 in a row brighter than cutoff, you can stop there
    for n in range(len(flux)):
        if flux[n] >= fractionalBright:
            p+=1
            if p == 10:
                cutoffindex = n-10
                break
        else: # if you haven't hit 10 in a row above fractional position, try again
            p=0
            
    if QCBVALL is not None:
        print("TRIMMING FRACTIONALLY CBVS")
        Qall, CBV1, CBV2, CBV3 = QCBVALL
        Qall = Qall[:cutoffindex]
        CBV1 = CBV1[:cutoffindex]
        CBV2 = CBV2[:cutoffindex]
        CBV3 = CBV3[:cutoffindex]
        QCBVALL = [Qall, CBV1, CBV2, CBV3]
        print('quall', len(Qall))
    if bg is not None:
        bg = bg[:cutoffindex]

    return (time[:cutoffindex], flux[:cutoffindex], 
            error[:cutoffindex], bg, QCBVALL)

def bin_8_hours(time, flux, error, bg, QCBVALL=None):
    """
    Bin light curve to 8 hours
    QCBVALL should unpack as Qall, CBV1, CBV2, CBV3 OR be None 
    ---------------------------------------
    Params:
        - time, flux, error, bg (arrays)
        - QCBVALL (array of arrays or None) containing the qcbvs
    ---------------------------------------
    Returns:
        - time, flux, error, bg, QCBVALL
    """

    n_points = 16
    binned_time = []
    binned_flux = []
    binned_error = []
    binned_bg = []
    if QCBVALL is not None:
        Qall, CBV1, CBV2, CBV3 = QCBVALL
        binned_quat = []
        b_cbv1 = []
        b_cbv2 = []
        b_cbv3 = []
    
    n = 0
    m = n_points
        
    while m <= len(time):
        # get the midpoint of this data as the point to plot at
        bin_t = time[n + 8] 
        binned_time.append(bin_t) # put into new array
        binned_flux.append(np.nanmean(flux[n:m])) # put into new array
        # error propagates as sqrt(sum of squares of error)
        binned_error.append((np.sqrt(np.sum(error[n:m]**2)) / n_points ))
        binned_bg.append(np.nanmean(bg[n:m]))
        if QCBVALL is not None:
            binned_quat.append(np.nanmean(Qall[n:m]))
            b_cbv1.append(np.nanmean(CBV1[n:m]))
            b_cbv2.append(np.nanmean(CBV2[n:m]))
            b_cbv3.append(np.nanmean(CBV3[n:m]))
        
        n+= n_points
        m+= n_points
     
    if QCBVALL is not None:
        QCBVALL = [np.asarray(binned_quat), np.asarray(b_cbv1), 
                   np.asarray(b_cbv2), np.asarray(b_cbv3)]
    return (np.asarray(binned_time), np.asarray(binned_flux),
            np.asarray(binned_error), np.asarray(binned_bg), QCBVALL)

    
    
 
def quat_txtfile_production(file, fileoutput):
    """
    Produce Quaternion files for easy opening/handling and smaller storage size
    This only needs to be run once for each sector at the start of ALL 
    the work, thank god. 
    ------------------------------
    Params:
        - file (str) input quaternion fits file that needs to be compressed into txt
        - fileoutput (str) filename of output txt file
    """
    from astropy.io import fits
    if os.path.exists(fileoutput):
        print("quat.txt file already exists!")
        return
    
    f = fits.open(file, memmap=False)
    
    t = f[1].data['TIME']
    Q1 = f[1].data['C1_Q1']
    Q2 = f[1].data['C1_Q2']
    Q3 = f[1].data['C1_Q3']
    f.close()
    
    big_quat_array = np.asarray((t, Q1, Q2, Q3))
    np.savetxt(fileoutput, big_quat_array)
    return
    
def make_quat_txtfiles(inFolder, outFolder):
    """
    Produces ALL quaternion text files for a given folder of quaternion fits files
    --------------------------------
    Params:
        - inFolder (str) where all the fits files currently are
        - outFolder (str) where all the txt files should go
    """
    for root, dirs, files in os.walk(inFolder):
        for name in files:
            import re
            s = re.split(r"_|-", name)
            fileoutput = outFolder + "quats-" + s[1] + ".txt"
            quat_txtfile_production(root+name, fileoutput)
    return

def quaternion_binning(quaternion_t, quat_data, tmin):
    """
    Bin the quaternions
    ---------------------------
    Params:
        - quaternion_t (array, quat time axis)
        - quat_data (array, actual quat y axis)
        - tmin (float, 0 of the time axis for loading and aligning purposes)
    """
    bins = 900 #30 min times sixty seconds/2 second cadence
    def find_nearest_values_index(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    binning_start = find_nearest_values_index(quaternion_t, tmin)
    n = binning_start
    m = n + bins
    binned_Q = []
    binned_t = []
            
    while m <= len(quaternion_t): #take avgs in bins and save
        bin_t = quaternion_t[n]
        binned_t.append(bin_t)
        bin_q = np.mean(quat_data[n:m])
        binned_Q.append(bin_q)
        n += 900
        m += 900
            
    standard_dev = np.std(np.asarray(binned_Q))
    mean_Q = np.mean(binned_Q)
    outlier_indexes = []
            
    for n in range(len(binned_Q)): # remove any 5 stdev outliers
        if (binned_Q[n] >= mean_Q + 5*standard_dev or 
            binned_Q[n] <= mean_Q - 5*standard_dev):
            outlier_indexes.append(n)
            
                  
    return np.asarray(binned_t), np.asarray(binned_Q), outlier_indexes
  
def speed_load_quats_from_fastloadfile(file):
    """ 
    Load in quaternions from text file
    --------------------
    Params:
        - file (str) path to file to load
    ----------------------
    Returns:
        - tQ, Q1, Q2, Q3
    """
    c = np.genfromtxt(file) #
    tQ = c[0]
    Q1 = c[1]
    Q2 = c[2]
    Q3 = c[3] 
    return tQ, Q1, Q2, Q3
          
def metafile_load_smooth_quaternions(sector, tmin,
                                     quaternion_folder):
    """
    Helper function to get quaternions out of their text files 
    and bin them. Has option to save into a faster loading
    pre-binned file.
    --------------------------------------
    Params:
        - sector (str, yes i know it's a string and that's weird, okay? it is 
                  what it is and is how the file path gets opened)
        - tmin (float, time[0] of original sector time axis)
        - quaternion_folder (str, path to where the quats are saved)
    """
    import os
    # if we've been here before:
    shortcut_file = quaternion_folder + "quats-sector" + sector + "FASTLOAD.txt"
    if os.path.exists(shortcut_file):
        c = np.genfromtxt(shortcut_file) #
        tQ = c[0]
        Q1 = c[1]
        Q2 = c[2]
        Q3 = c[3]   
        outlier_indexes = np.ones(0)
        return tQ, Q1, Q2, Q3, outlier_indexes         
        
    # if first time, generate raw + save for sector
    from scipy.signal import medfilt
    
    filepath = quaternion_folder + "quats-sector" + sector + ".txt"
    c = np.genfromtxt(filepath) # this takes a million years to load - need to save better
    tQ = c[0]
    Q1 = c[1]
    Q2 = c[2]
    Q3 = c[3]   

    tQ += 2457000  # these come pre-trimmed down?     

    q = [Q1, Q2, Q3]

    for n in range(3):
        smoothed = medfilt(q[n], kernel_size = 31)
        if n == 0:
            Q1 = smoothed
            tQ_, Q1, Q1_outliers = quaternion_binning(tQ, Q1, tmin)
        elif n == 1:
            Q2 = smoothed
            tQ_, Q2, Q2_outliers = quaternion_binning(tQ, Q2, tmin)
        elif n == 2:
            Q3 = smoothed
            tQ_, Q3, Q3_outliers = quaternion_binning(tQ, Q3, tmin)
    
    outlier_indexes = np.unique(np.concatenate((Q1_outliers, Q2_outliers, Q3_outliers)))
    
    # save into quickload!
    big_quat_array = np.asarray((tQ_, Q1, Q2, Q3))
    np.savetxt(shortcut_file, big_quat_array)
    return tQ_, Q1, Q2, Q3, outlier_indexes  

def generate_clip_quats_cbvs(sector, x, y, yerr, tmin, camera, ccd, 
                             CBV_folder, quaternion_folder):
    """
    Load in cbv and quaternions and match them up to the x values given 
    ---------------------------------
    Params:
        - sector (str, for generating file name)
        - x (array, axis)
        - y (array, intensity)
        - yerr (array)
        - camera (str)
        - ccd (str)
        - CBV_folder (str, path to where CBV files are held)
        - quaternion_folder (str, path to where quat files are held)
    """
    print("Loading quaternions")
    tQ, Q1, Q2, Q3, outliers = metafile_load_smooth_quaternions(sector, tmin,
                                                                quaternion_folder)
    Qall = Q1 + Q2 + Q3
    print("quaternion load complete - loading cbvs")
    # load CBVs
    cbv_file = (CBV_folder + 
                "s00{sector}/cbv_components_s00{sector}_000{camera}_000{ccd}.txt".format(sector = sector,
                                                                                          camera = camera,
                                                                                          ccd = ccd))
    print("cbv load completed")
    cbvs = np.genfromtxt(cbv_file)
    CBV1 = cbvs[:,0]
    CBV2 = cbvs[:,1]
    CBV3 = cbvs[:,2]
    # correct length differences:
    lengths = np.array((len(x), len(tQ), len(CBV1)))
    length_corr = lengths.min()
    x = x[:length_corr]
    y = y[:length_corr]
    yerr = yerr[:length_corr]
    tQ = tQ[:length_corr]
    Qall = Qall[:length_corr]
    CBV1 = CBV1[:length_corr]
    CBV2 = CBV2[:length_corr]
    CBV3 = CBV3[:length_corr]
    return x,y,yerr, tQ, Qall, CBV1, CBV2, CBV3

def tr_downloader(fileOfTargets, fileSavePath, cdir):
    """ 
    Download the tessreduce lc for your list
    --------------------------
    Params:
        - fileOftargets (str) directory link to a pandas readable csv file of 
            all targets to retrieve data for
        - fileSavePath (str) directory link to where to save the data
        - cdir (str) directory link to where tesscut downloads. 
            *** Lindsey your cdir is "/Users/lindseygordon/.lightkurve-cache/tesscut/"
    """
    info = pd.read_csv(fileOfTargets)
    failures = []
    for i in range(0,len(info)):
        sec = int(info["Sector"].iloc[i])
        print(i)
        time.sleep(40)
        print(info['Name'].iloc[i][3:])
        targ = info['Name'].iloc[i][3:]
        try:
            obs = tr.sn_lookup(targ)
            lookup = obs[np.where(np.asarray(obs)[:,2] == sec)[0][0]]
            tess = tr.tessreduce(obs_list=lookup,plot=False,reduce=True)
            
        except ValueError:   
            print("value error - something is wrong with vizier or no target in pixels")
            failures.append(i)
            continue
        except IndexError:
            print("index error - tesscut thinks it wasn't observed")
            continue
        except ConnectionResetError:
            print("vizier problems again")
            failures.append(i)
            continue
        except TimeoutError:
            print("vizier problems")
            failures.append(i)
            continue
        except ConnectionError:
            print("more! vizier! problems!")
            failures.append(i)
            continue
    
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
        targlabel = targ + sector + camera + ccd 
        newfolder = fileSavePath + targlabel + "/"
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
    return failures

def tr_load_lc(file, printname=True):
    """
    Given a filename, load in the data. 
    Assumes filenames formatted as in tr_downloader()
    ---------------------
    Params:
        - file (str) link to file to load
        - printname (default True) print the target's name 
    """
    loadedraw = pd.read_csv(file)
    time = Time(loadedraw["time"], format='mjd').jd
    intensity = loadedraw["flux"].to_numpy()
    error = loadedraw["flux_err"].to_numpy()
    #
    fulllabel = file.split("/")[-1].split("-")[0]
    targetlabel = fulllabel[0:7]
    if targetlabel[-1].isdigit():
        targetlabel=targetlabel[0:6]
    sector = fulllabel[-4:-2]
    camera = fulllabel[-2]
    ccd = fulllabel[-1]
    if targetlabel[-1].isdigit():
        targetlabel=targetlabel[0:6]
    
    if printname:
        print(targetlabel, sector, camera, ccd)
        
    time, intensity, error, bg = normalize_sigmaclip(time, intensity, error, None, axis=0)
        
    return time, intensity, error, targetlabel, sector, camera, ccd