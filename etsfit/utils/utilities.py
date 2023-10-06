# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:59:01 2021
Updated Nov 21 2022
Updated Oct 6 2023 - docstrings

Utility functions, mostly to do with data manipulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.time import Time
import tessreduce as tr 
import time
from scipy.signal import medfilt

def get_sn_sublist(csvfile, searchlist, outFile):
    """ 
    Get list of only the SN on your searchlist
    :param csvfile: all sn in csv
    :param searchlist: names of sn to keep
    :param outFile: output csv file
    """
    info = pd.read_csv(csvfile)
    for i in range(len(info)):
        sn = info["Name"][i][3:]
        print(sn)
        if sn not in searchlist:
            info.drop(i, inplace=True)
    
    info.reset_index(inplace=True)
    del info["Unnamed: 0"]
    info.to_csv(outFile)
    return

def load_lygos_csv(file):
    """
    Load data from a lygos rflxtarg csv file 
    Assumes 3 columns time/flux/error AND that there is an rflx file
    in the same folder containing the background data

    :param file: (str) path to an rflxtarg file to be loaded
    
    :return: time, flux, error, background (arrays)
        
    """
    data = pd.read_csv(file, sep = ',', header = 0)
    t = np.asarray(data[data.columns[0]])
    flux = np.asarray(data[data.columns[1]])
    error = np.asarray(data[data.columns[2]])
    
    data = pd.read_csv(file.replace("rflxtarg", "rflx"), sep = ',', header = 0)
    bg = np.asarray(data[data.columns[-2]])
    
    return t, flux, error, bg
 
def get_disctime(file, name):
    """
    Helper function to retrieve discovery time from a big TNS file
    
    :param file: (str) path to the TNS file
    :param name: (str) SN to retrieve, ie, '2020abc'
    
    :return: discovery time in JD
    """
    f = pd.read_csv(file)
    d = f[f["Name"].str.contains(name)]["Discovery Date (UT)"]
    d = Time(d.iloc[0], format = 'iso', scale='utc') # tns files are in UTC
    return d.jd
    
def window_rms(time, flux, innerfilt = None, outerfilt = None,
               plot=True):
    """ 
    Runs an RMS filter over the light curve and returns an array of 
    0's (bad) and 1's (good) that can be used in the custom masking
    argument of other functions. 
    
    Defaults the inner window as len(self.time)*0.005 and the outer as
    inner*10.
    
    :param time: time array
    :param flux: flux array
    :param innerfilt: None by default, can set to an int for the inner
                    window of compariosn
    :param outerfilt: None by default, can set to an int for the outer
        window of comparison
    :param plot: (bool) defaults as True, plots light curve w/ mask
    
    """
    if innerfilt is None:
        innersize = int(len(time)*0.005)
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
        
        outer_window = flux[outer_lower:outer_upper]
        inner_window = flux[inner_lower:inner_upper]
        
        std_outer = np.std(outer_window)
        
        rms_outer = np.sqrt(sum([s**2 for s in outer_window])/len(outer_window))
        rms_inner = np.sqrt(sum([s**2 for s in inner_window])/len(inner_window))
        
        if ((rms_inner > (rms_outer + std_outer)) 
            or (rms_inner < (rms_outer - std_outer))):
            rms_filt[inner_lower:inner_upper] = 0 #bad point, discard
            #print(rms_inner, rms_outer, std_outer)
    
    if plot:
        rms_filt_plot = np.nonzero(rms_filt)
        fig, ax = plt.subplots(1, figsize=(6,2))
        ax.scatter(time, flux, color='green', label='bad', s=2)
        ax.scatter(time[rms_filt_plot], flux[rms_filt_plot], 
                    color='blue', s=2, label='good')
        ax.legend(fontsize=16)
        ax.set_title("window rms output")
        plt.show()
        plt.close()
    return rms_filt

def sigmaclip(time, flux,error, bg, axis=0):
    """ 
    5 sigma clip data

    :param time: 
    :param flux:
    :param error:
    :param bg: can be None if no background to handle
    :param axis: default 0 for 1D arrays
    
    :returns: time, flux, error, bg
    
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

def data_masking(obj):
    """ 
    Mask the data using the object's flux_mask attribute
   
    :param obj: etsfit object
    """
    mask = np.nonzero(obj.mask) # which ones you are keeping
    obj.time = obj.time[mask]
    obj.flux = obj.flux[mask]
    obj.error = obj.error[mask]
    if obj.background is not None:
        obj.background = obj.background[mask]
        
    if obj.qcbv_fit is not None:#if cbvs, trim them
        obj.Qall = obj.Qall[mask]
        obj.CBV1 = obj.CBV1[mask]
        obj.CBV2 = obj.CBV2[mask]
        obj.CBV3 = obj.CBV3[mask]
        obj.quats_cbvs = [obj.Qall, obj.CBV1, obj.CBV2, obj.CBV3]
    return 

def param_save(obj):
    """ 
    Save output parameter files

    :param obj: etsfit object
    """
    if hasattr(obj, 'BIC_celerite'):
        BICstring = f"BIC celerite:{obj.BIC_celerite[0]:.3f}\nBIC tinygp{obj.BIC_tinygp[0]:.3f}\n"
    else:
        BICstring = f"BIC:{obj.BIC:.3f}\n"
    
    CONVstring = f"Converged:{obj.converged} \n"
    BPstring = ""
    UEstring = ""
    LEstring = ""
    THstring = ""
    #save parameters in rows of 4:
    for n in range(obj.ndim):
        BPstring = f"{BPstring} {obj.best_mcmc[n]:.4f}"
        UEstring = f"{UEstring} {obj.upper_error[n]:.4f}"
        LEstring = f"{LEstring} {obj.lower_error[n]:.4f}"
        if not (n+1) % 4: #every 4, make a new line
            BPstring = f"{BPstring}\n"
            UEstring = f"{UEstring}\n"
            LEstring = f"{LEstring}\n"
            
    #then add another gap after that
    BPstring = f"{BPstring}\n"
    UEstring = f"{UEstring}\n"
    LEstring = f"{LEstring}\n"
    
    #save theta values for tinygp
    if hasattr(obj, 'theta'):
        for k in obj.theta.keys():
            THstring = f"{THstring}{k} \n {obj.theta[k]:.4f} \n"
    
    #write into file:
    with open(obj.parameter_save_file, 'w') as file:
        file.write(f"{obj.targetlabel}\n")
        file.write(BICstring)
        file.write(CONVstring)
        file.write(BPstring)
        file.write(UEstring)
        file.write(LEstring)
        file.write(THstring)
    
    return
    
def fractional_trim(obj):
    """
    Trims light curve to a chosen fraction of the peak flux
    Considers "above" to be when 10 indices in a row are all brighter
    (Actually looks from the top)
    
    :param obj: etsfit object
    """
    #fractionalBright = ((flux.max()-1) * fraction) + 1
    #find range, take percent of range, add to min
    fractionalBright = obj.flux.max() - ((obj.flux.max() - obj.flux.min()) * (1-obj.fraction_trim))
    
    p = 0 # if you hit 10 in a row brighter than cutoff, you can stop there
    cutoffindex = len(obj.flux)
    for n in range(len(obj.flux)):
        if obj.flux[n] >= fractionalBright:
            p+=1
            if p == 10:
                cutoffindex = n-10
                break
        else: # if you haven't hit 10 in a row above fractional position, try again
            p=0
            
    if obj.qcbv_fit is not None:
        print("Trimming length of qcbvs")
        for i in range(4):
            obj.quats_cbvs[i] = obj.quats_cbvs[i][:cutoffindex]
    if obj.background is not None:
        obj.background = obj.background[:cutoffindex]

    return

def time_binning(obj, goal_dt, time_unit):
    """
    Bin light curve to goal_dt timeframe

    :param obj: etsfit object
    :param goal_dt: float, how long each timeframe should be 
    :param time_unit: str, what unit the time array is in
    """
    # make sure everything is in JD
    time = Time(obj.time, format=time_unit).jd
    goal_dt = Time(goal_dt, format=time_unit).jd
        
    # then need to calculate how many points per bin
    n_points = int(np.ceil((time[1] - time[0])/goal_dt))
    n_len = int(np.ceil(len(time)/n_points)) #how long outputs will be
    
    rows = 8
    if hasattr(obj, "Qall"): rows += 4 # make space for cbvs
    if hasattr(obj, "background"): rows += 1 # make space for background
    
    binned = np.zeros((rows, n_len)) # put solution in here
    # rows: t, f, e, bg, qall, cbv1, cbv2, cbv3
    
    for i in range(n_len): # for each in output
        j = i * n_points # left index
        k = min ( j+n_points, len(time)) # right index (or end of array)
        
        binned[0][i] = time[j] # leftmost time per bin = bin time
        binned[1][i] = np.nanmean( obj.flux[j:k]) #bin mean
        # error propagates as sqrt sum of squares of errors / npoints
        binned[2][i] = np.sqrt(np.sum(obj.error[j:k]**2)) / n_points  
        if hasattr(obj, "background"): 
            binned[3][i] = np.nanmean(obj.background[j:k]) #background
        if hasattr(obj, "Qall"):
            binned[4][i] = np.nanmean(obj.Qall[j:k])
            binned[5][i] = np.nanmean(obj.CBV1[j:k])
            binned[6][i] = np.nanmean(obj.CBV2[j:k])
            binned[7][i] = np.nanmean(obj.CBV3[j:k])
    
    # put them back into obj:
    while binned[0][-1] == 0: # if there were too many here oopsie
        binned = binned[:, :-1] #chop last column
    obj.time = binned[0]
    obj.time_unit = 'JD'
    obj.flux = binned[1]
    obj.error = binned[2]
    if hasattr(obj, "background"):
        obj.background = binned[3]
    if hasattr(obj, "Qall"):
        obj.Qall = binned[4]
        obj.CBV1 = binned[5]
        obj.CBV2 = binned[6]
        obj.CBV3 = binned[7]
    
    return

def tr_downloader(file, data_dir, cdir, start=0):
    """ 
    Download the tessreduce lc for your list
    *** Lindsey your cdir is "/Users/lindseygordon/.lightkurve-cache/tesscut/"

    :params file: (str) directory link to a pandas readable csv file of all targets to retrieve data for
    :params data_dir: (str) directory link to where to save the data
    :params cdir: (str) directory link to where tesscut downloads. 
    :params start: where in list to start looking
            
    """
    info = pd.read_csv(file)
    failures = []
    for i in range(start,len(info)):
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
        newfolder = data_dir + targlabel + "/"
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
    
    :param file: (str) link to file to load
    :param printname: (default True) print the target's name 
    """
    loadedraw = pd.read_csv(file)
    time = Time(loadedraw["time"], format='mjd').jd
    flux = loadedraw["flux"].to_numpy()
    error = loadedraw["flux_err"].to_numpy()
    #
    fh = file.split("/")[-1].split("-")[0]
    
    ccd = int(fh[-1])
    camera = int(fh[-2])
    sector = int(fh[-4:-2])
    targetlabel = fh[:-4] #this fixes issues of varying name length
    
    if printname: print(targetlabel, sector, camera, ccd)
        
    time, flux, error, bg = sigmaclip(time, flux, error, None, axis=0)
    return time, flux, error, targetlabel, sector, camera, ccd

### quaternion handling section  

def generate_align_quats_cbvs(obj, **kwargs):
    """
    Load in cbv and quaternions and match them up to the x values given 
    
    :param obj: etsfit object
    """
    # Load quaternions from files:
    print("Loading quaternions...")
    tQ, Qall = load_quaternions(obj.quaternion_text_dir, obj.sector, obj.time,
                                manual_redo=kwargs.get('realign_quats', False))
    # Load CBVs from files:
    print("Loading CBVs...")
    CBV1, CBV2, CBV3 = load_cbvs(obj.cbv_dir, obj.time, obj.sector, 
                                 obj.camera, obj.ccd, 
                                 realign_cbvs=kwargs.get("realign_cbvs", False))
    # There are no length differences at this point. Load these into the obj. 
    obj.tQ = tQ # this is really time at this point...
    obj.Qall = Qall 
    obj.CBV1 = CBV1
    obj.CBV2 = CBV2
    obj.CBV3 = CBV3
    return   

def load_cbvs(cbv_dir, time, sector, camera, ccd, realign_cbvs):
    """ 
    cbv loader - requires working internet to get tess cutout. 
    Loosely based on eleanor/update.py to get the FFI timestamps 

    :param cbv_dir: directory for cbvs
    :param time: time axis of data
    :param sector: int 
    :param camera: int
    :param ccd: int
    :param realign_cbvs: this was a kwarg, if the alignment already happened set to False
    """
    cbv_file = f"{cbv_dir}s{int(sector):04}/cbv_components_s{int(sector):04}_{int(camera):04}_{int(ccd):04}.txt"
    print(cbv_file)
    if not os.path.isfile(cbv_file):
        raise ValueError("You need to have a legit cbv file - probably didn't download?")
    # load the cbvs
    cbvs = np.genfromtxt(cbv_file)
    CBV1 = cbvs[:,0]
    CBV2 = cbvs[:,1]
    CBV3 = cbvs[:,2]   
    
    # now comes the mess - you either have the time alignments saved or need to
    # redo them now. 
    alignments = f"{cbv_dir}s{int(sector):04}/cbv_components_s{int(sector):04}_{int(camera):04}_{int(ccd):04}_alignment.txt"
    if os.path.isfile(alignments) or realign_cbvs:
        print('alignment file exists, loading in')
        arg_lineups = np.genfromtxt(alignments, dtype=int)
        CBV1 = CBV1[arg_lineups]
        CBV2 = CBV2[arg_lineups]
        CBV3 = CBV3[arg_lineups]
        return CBV1, CBV2, CBV3
        
    print("aligning time axis...")
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astroquery.mast import Tesscut
    from astropy.io import fits
    
    # set up coordinates (based on eleanor package, coords from MAST)
    
    # northern ecliptic pole (NEP) -> "18:00:00.000 +66:33:38.55"
    # cycle 2 (sectors 14 - 26), cycle 4 (40-53)
    if (sector in np.arange(14, 26+1, 1)) or (sector in np.arange(40, 53+1, 1)):
        use_coords = SkyCoord('18:00:00.000 +66:33:38.55',
                              unit=(u.hourangle, u.deg))
        
    # southern ecliptic pole (SEP) -> "6:00:00.000 -66.33.38.55"
    # cycle 1 (sectors 1-13) cycle 3 (27 - 39)
    elif (sector in np.arange(1, 13+1, 1)) or (sector in np.arange(27, 39+1, 1)):
        use_coords = SkyCoord('6:00:00.000 -66.33.38.55',
                              unit=(u.hourangle, u.deg))
    
    try:
        manifest = Tesscut.download_cutouts(coordinates=use_coords, size=31, 
                                            sector=sector)
    except:
        print("This sector isn't available yet or some other shit went wrong")
        return

    cutout = fits.open(manifest['Local Path'][0], memmap=False)
    time_cbv = cutout[1].data['TIME'] - cutout[1].data['TIMECORR']
    
    #okay how to line these up now. 
    arg_lineups = np.zeros(len(time), dtype=int)
    if time[0] > 2457000: 
        time -= 2457000 # time correction if necessary
    # for each in time, get the closest item in time_cbv's index
    for i in range(len(time)):
        arg_lineups[i] = np.argmin(np.abs(time_cbv-time[i]))
        
    # then correct the cbv time index -> same will apply to the actual cbv indexes
    # can save the lineup array for later reuse
    CBV1 = CBV1[arg_lineups]
    CBV2 = CBV2[arg_lineups]
    CBV3 = CBV3[arg_lineups]
    
    # save alignment into file: 
    print("saving alignment into file...")
    np.savetxt(alignments, arg_lineups, fmt="%d")
    
    return CBV1, CBV2, CBV3
    

def load_quaternions(quat_folder, sector, time, **kwargs):
    """
    Helper function to load in (and bin if needed) the quaternions
    Has an option that loads from a shortcut if you're not having it totally
    redo the binning (kwarg)

    :param quat_folder: folder of quaternions
    :param sector: int
    :param time: time axis
    :param manual_rebin: bool to force rebinning of quaternions
    
    """
    redo = kwargs.get("manual_rebin", False)
    # if we've been here before or are redoing it:
    shortcut_file = f"{quat_folder}quats-sector-{int(sector):02}-FASTLOAD.txt"
    if os.path.isfile(shortcut_file) and redo:
        c = np.genfromtxt(shortcut_file) #
        tQ = c[0]
        Qall = c[1]
        return tQ, Qall 
    
    # generate raw version and save for sector (otherwise just going to quickload)
    filepath = f"{quat_folder}quats-sector-{int(sector):02}.txt"
    c = np.genfromtxt(filepath) # this takes a reeeeeeally long time to load in
    tQ = c[0]
    Q1 = c[1]
    Q2 = c[2]
    Q3 = c[3]  
    
    # if pre-trimmed-down, this will reset to match your other axis (probably)
    if tQ[0] < 2457000 and time[0] > 2457000:
        tQ += 2457000
            
    # determine if the quaternion is on 2 second cadence. i don't know why it would
    # not be, but just in case:
    two_sec_size_jd = 2 / (60*60*24)
    h = np.abs(tQ[1] - tQ[0] - two_sec_size_jd)
    if h < 1e-5:
        print(f"Within tolerance [1e-5] to be 2 second cadence (Actual: {h}")
        print("bin size: 900 per")
        bin_size = 900
    else: 
        print("NOT two second cadence quaternions - probably you should \
              double check what the hell this thing is accessing.")
              
    # you're just going to add them together eventually anyways? do that now. 
    q = Q1+Q2+Q3 # and if not it will be easier to unpack later anyways. 
    
    # smoothing median filter, size=30 (1 minute smoothing)
    from scipy import ndimage
    q = ndimage.median_filter(q, size=30)
    
    # binning based off of the time array of the real data being used
    # the array is always going to be the same length as the fucking time array. 
    binned = np.empty((2, len(time)))
    
    for i in range(len(time)):
        # where in time
        start_t = time[i]
        # where is that in the array
        start_idx = (np.abs(tQ - start_t)).argmin()
        # get the next bin_size of points' mean
        binned[0][i] = time[i]
        binned[1][i] = np.nanmean(q[start_idx:start_idx+bin_size])    
    
    # save into quickload
    print("Binning complete, saving output file...")
    np.savetxt(shortcut_file, binned)
    return binned[0], binned[1]

def single_quat_textfile(file_in, file_out):
    """
    Produce Quaternion files for easy opening/handling and smaller storage size
    This only needs to be run once for each sector at the start to make the new files

    :param file_in: input quaternion fits file that needs to be compressed into txt
    :param file_out: (str) filename of output txt file
    """
    from astropy.io import fits
    if os.path.isfile(file_out):
        print(".txt file version already exists!")
        return
    
    f = fits.open(file_in, memmap=False)
    
    t = f[1].data['TIME']
    Q1 = f[1].data['C1_Q1']
    Q2 = f[1].data['C1_Q2']
    Q3 = f[1].data['C1_Q3']
    f.close()
    
    big_quat_array = np.asarray((t, Q1, Q2, Q3))
    np.savetxt(file_out, big_quat_array)
    return
    
def all_quat_textfiles(inFolder, outFolder):
    """
    Produces ALL quaternion text files for a given folder of quaternion .fits files
    
    :param inFolder: (str) where all the fits files currently are
    :param outFolder: (str) where all the txt files should go
    """
    for root, dirs, files in os.walk(inFolder):
        for name in files:
            if name.endswith(".txt"): continue
            import re
            s = re.split(r"_|-", name)
            file_out = f"{outFolder}quats-sector-{s[1][-2:]}.txt"
            single_quat_textfile(root+name, file_out)
            print(f"Created {file_out}")
    return


