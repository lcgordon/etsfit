# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:14:13 2022

@author: lcgordon

todo list:
    - double check GP stuff is working [seems fine as of 05092022]
    - save a plot of trimmed regions for when doing custom masking. [done 05092022]
    - set up GP parameter scan function
    - TEST GP CUSTOM PRIOR/INPUTS 
"""

#import etsfit.utils.utilities as ut
#import etsfit.utils.snPlotting as sp
#import etsfit.utils.MCMC as mc
#import utils.utilities as ut
#import utils.snPlotting as sp
#import utils.MCMC as mc

from etsfit.utils import utilities as ut
from etsfit.utils import snPlotting as sp
from etsfit.utils import MCMC as mc

import time as timeModule
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import emcee
import datetime 
from pylab import rcParams
rcParams['figure.figsize'] = 16, 6
rcParams["font.size"] = 20

from celerite.modeling import Model
import celerite
from celerite import terms

import warnings
warnings.filterwarnings("ignore")


class etsMAIN(object):
    """Make one of these for the light curve you're going to fit!"""
    
    def __init__(self, folderSAVE, bigInfoFile):
        """
        Initialize an etsfit object
        
        This sets up a save path, loads in a csv of targets if you want,
        and sets default values of the data not being trimmed, binned, and
        there being no CBVs/quaternions accessible.
        
        -----------------------------------------------
        Parameters:
            
            - folderSAVE (str) the path to where you want everything to be saved
            should end with a /, it will self-correct if not
            
            - bigInfoFile (str) the path to a big pandas-readable CSV file
            containing all the targets you could possibly care about
        
        """
        
        if not os.path.exists(folderSAVE): #if the folder that the subfolders are going into
        #is not real
            raise ValueError("Outermost folder to save into must be a real path!")
        else:
            if folderSAVE[-1] != "/": #if not ending with /, 
                folderSAVE += "/"
            self.folderSAVE = folderSAVE
            self.foldersaveperm = folderSAVE #keep a copy in case
        
        
        if os.path.exists(bigInfoFile):
            self.info = pd.read_csv(bigInfoFile)
        self.bigInfoFile = bigInfoFile
        
        
        self.cbvsquatsActive = False #no cbvs/quats in
        self.fractiontrimmed = False #not trimmed
        self.binned = False #not binned
        return
    
    def use_quaternions_cbvs(self, CBV_folder, quaternion_folder_raw, 
                             quaternion_folder_txt):
        """ 
        Function to set up access to CBVs and quaternions. 
        If you do not have text file versions of the quaternions, they will
        be generated for you.
        
        ----------------------------------------------------
        Parameters: 
            
            - CBV_folder (str) a path to the folder holding all CBVs. 
            this is probably within the eleanor directory
            the directory structure should be such that there are files like
            folder/s001/cbv_components_s0001_0001_0001.txt
            
            - quaternion_folder_raw (str) is the path to the folder holding
            all of the .fits file versions of the quaternions. this can be
            some random path if you already have produced the txt versions
            
            - quaternion_folder_txt (str) is the path to the folder either
            HOLDING the .txt file versions of the quaternions OR the path to
            the EMPTY folder that all the txt versions are about 
            to be generated into
        """
        
        if not os.path.exists(CBV_folder):
            raise ValueError("Not a valid CBV path")
        else:
            self.CBV_folder = CBV_folder
            
        if not os.path.exists(quaternion_folder_raw):
            raise ValueError("Not  a valid raw quat path")
        else:
            self.quaternion_folder_raw = quaternion_folder_raw
        
        if not os.path.exists(quaternion_folder_txt):
            raise ValueError("Not a valid txt quat path")
        else:
            self.quaternion_folder_txt = quaternion_folder_txt
            if (len(os.listdir(self.quaternion_folder_txt))==0 and
                len(os.listdir(self.quaternion_folder_raw))!=0):
                #empty directory, make txt files
                ut.make_quat_txtfiles(self.quaternion_folder_raw, self.quaternion_folder_txt)
                
        self.cbvsquatsActive = True
        return
    
    def run_tessreduce_retrieval_csvlist(self, fileToPullFrom, folderToPutIn,
                                         cdir, failurererun=None):
        """
        Given a file with a column of names to pull, retrieve all tessreduce lc
        returns indices of the light curves that should exist but vizier crapped out on
        
        -------------------------------
        Parameters:
            - fileToPullFrom is a csv file containing a column labeled "Name"
            - folderToPut in is a path to the top level folder to save all retreived data into
            - cdir is the local cache directory that tessreduce saves you into. 
        
        """
        import tessreduce as tr
        import gc
        
        biglist = pd.read_csv(fileToPullFrom)
        failures = []
        
        for i in range(0,len(biglist)):
            try:
                print(i)
                #if given a rerun list and i is not on that list, move on
                if failurererun is not None and  not (i in failurererun):
                    continue
                time.sleep(40)
                print(biglist['Name'].iloc[i])
                targ = biglist['Name'].iloc[i]
                if targ.startswith("SN "):
                    targ = targ[3:]
                    
                obs = tr.sn_lookup(targ)
                tess = tr.tessreduce(obs_list=obs,plot=False,reduce=True)

                holder = ""
                for root, dirs, files in os.walk(cdir):
                    for name in files:
                        holder = root + "/" + name
                        print(holder)
                        try:
                            filenamepieces = name.split("-")
                            sector = str(filenamepieces[1][3:])
                            camera = str(filenamepieces[2])
                            ccd = str(filenamepieces[3][0])
                            os.remove(holder)
                            break
                        except IndexError:
                            print("deleting extraneous file in folder")
                            os.remove(holder)
                            continue
                print(sector)
                print(camera)
                print(ccd)
                
                #make subfolder to save into 
                targlabel = targ + sector + camera + ccd 
                newfolder = folderToPutIn + targlabel + "/"
                if not os.path.exists(newfolder):
                    os.mkdir(newfolder)
                    filesave = newfolder + targlabel + "-tessreduce.csv"
                    tess.save_lc(filesave)
                    tess.to_flux()
                    filesave = newfolder + targlabel + "-tessreduce-fluxconverted.csv"
                    tess.save_lc(filesave)
                    
                    del(obs)
                    del(tess)
                    gc.collect()
                else:
                    print("Folder already exists (lc already downloaded), exiting")
                    continue
                
            except ValueError:
                print("value error - something is wrong with vizier as always")
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
        return failures

    
    def window_rms_filt(self, innerfilt = None, outerfilt = None,
                        plot=True):
        """ 
        Runs an RMS filter over the light curve and returns an array of 
        0's (bad) and 1's (good) that can be used in the custom masking
        argument of other functions. 
        
        This DOES NOT save the filter output inside the object. 
        
        Defaults the inner window as len(self.time)*0.005 and the outer as
        inner*20. Custom arguments are on the to-do list, nudge Lindsey if 
        you need them.
        
        -------------------------------
        Parameters:
            
            - innerfilt = None by default, can set to an int for the inner
            window of compariosn
            - outerfilt = None by default, can set to an int for the outer
            window of comparison
            - plot (bool) defaults as True, plots light curve w/ mask
        
        """
        return ut.window_rms(self.time, self.intensity, innerfilt = innerfilt, 
                        outerfilt = outerfilt,plot=plot)
        
    
    
    def load_data_lygos_single(self, fileToLoad, disctime=None, override=False):
        """
        Given a SPECIFIC filepath to a lygos lightcurve, load in the data
        And I do mean SPECIFIC path.
        
        --------------------------------------------
        Parameters:
            
            - fileToLoad (str), 
            ie "D:/18th1aAll/SN2018eod/rflxtarg_SN2018eod_0114_30mn_n005_d4.0_of11.csv"
            
            - disctime (double, defaults to NONE) if no big CSV file is loaded, 
            provide the discovery time (or custom disctime)
            
            - override (bool, defaults to FALSE) ignore it if loading data
            from a sector that is NOT the discovery sector
        
        """
        pieces = fileToLoad.split("_")
        # look up sector of discovery in big file
        self.sector = self.info[self.info["Name"].str.contains(pieces[1][2:])]["Sector"].iloc[0]
        # load in
        if (self.sector < 10):
            self.sector = "0" + str(self.sector)
        if (pieces[2].startswith(str(self.sector)) or override == True):
            
            time, intensity, error, lygosbg = ut.load_lygos_csv(fileToLoad)
            if self.bigInfoFile is None and disctime is None:
                raise Exception("No big info file given AND no disctime was provided")
            elif disctime is None:
                disctime = ut.get_disctime(self.bigInfoFile, pieces[1][2:])
            
            self.load_custom_lc(time, intensity, error, lygosbg, disctime, pieces[1],
                        pieces[2][0:2], pieces[2][2], pieces[2][3])
            
        
            print("LOADING IN:", self.targetlabel, "SECTOR: ", self.sector, "CAMERA: ",
            self.camera, "CCD: ", self.ccd)
            
            (self.time, self.intensity, 
            self.error, self.lygosbg) =  ut.normalize_sigmaclip(self.time, self.intensity, 
                                                                self.error, self.lygosbg) 
            self.tmin = self.time[0]
            self.time -= self.tmin
            self.disctime -= self.tmin
            self.bic_all = []
            self.params_all = []
            self.xlabel = "BJD - {timestart:.3f}".format(timestart=self.tmin)
            self.ylabel = "Rel. Flux"
            self.cleaningdone = False
            
            return
        else: 
            raise ValueError("Not discovery sector data  \n" + 
                             "If you want to load in anyways, pass override=True")
        return

    
    def load_single_lc(self, time, intensity, error, discoverytime, 
                       targetlabel, sector, camera, ccd, lygosbg=None):
        """ 
        Load in one light curve from information you supply
        
        ----------------------------------
        Parameters:
            
            - time (array) time axis for the lc. this will get the 0th
            index subtracted off. 
            
            - intensity (array) flux array for the lc 
            [yyy handle pre-cleaned data?]
            
            - error (array) error on flux array
            
            - discoverytime (double) when ground telescopes found it
            
            - targetlabel (str) no spaces name, will be used on files
            
            - sector (str) this needs to be a two-character string (ie, sector
                                                                    2 is "02")
            
            - camera (str) needs to be a 1 char string
            
            - ccd (str) needs to be a 1 char string
            
            - lygosbg (defaults to NONE) can put in lygos background array if 
            doing that one specific fit. should also work to float other 
            background arrays.
        """
        
        if (len(time) != len(intensity) or len(intensity) != len(error) or
            len(time) != len(error)):
            print("Time length:", len(time))
            print("Intensity length:", len(intensity))
            print("Error length:", len(error))
            raise ValueError("Mismatched sizes on time, intensity, and error!")
        elif (time is None or intensity is None or 
              error is None or discoverytime is None or 
              targetlabel is None or sector is None or 
              camera is None or ccd is None):
            raise ValueError("Inputs all have to be SOMETHING you can't give any None's here")
        elif type(sector) != str:
            raise ValueError("Sector must be a string, see help()")
        elif type(camera) != str:
            raise ValueError("camera must be a string, see help()")
        elif type(ccd) != str:
            raise ValueError("ccd must be a string, see help()")
        
        self.time = time
        self.intensity = intensity
        self.error = error
        self.lygosbg = lygosbg
        self.disctime = discoverytime
        self.targetlabel = targetlabel
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.tmin = time[0]
        self.time-=self.tmin
        self.disctime-=self.tmin
        self.bic_all = []
        self.params_all = []
        self.xlabel = "BJD - {timestart:.3f}".format(timestart=self.tmin)
        self.ylabel = "Rel. Flux"
        self.cleaningdone = False
        return
    
    def test_plot(self):
        """Quick little thing to spit out the current light curve that's loaded in """
        plt.errorbar(self.time, self.intensity, yerr=self.error, fmt='.', color='black')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.targetlabel)
        if hasattr(self, 'disctime'):
            plt.axvline(self.disctime)
        plt.show()
        plt.close()
        return

    
    def pre_run_clean(self, fitType, cutIndices=None, binYesNo = False, 
                      fraction = None):
        """
        A function to be run before running run_MCMC()
        This function MUST be run before running run_MCMC() or it will
        give u an error, even if you have no cleaning to be done. 
        
        ---------------------------------
        Parameters:
            
            - fitType (int)
                1-6 are default runs
                0 are custom arguments (see below)
            
            - cutIndices (array of ints) true/false array of points ot trim
                default is NONE
                
            - binYesNo (bool, default NONE) bins to 8 hours if true
            
            - fraction (default NONE, 0-1 for percent) trims intensity to 
            percent of max. 0.4 is 40% of max, etc.
        """
        # handle CBVs+quats
        if self.cbvsquatsActive and fitType in (2,4,5):
            print("Loading in quaternions and CBVs")
            (self.time, 
             self.intensity, 
             self.error,
             self.quatTime, 
             self.quatsIntensity, 
             self.CBV1, 
             self.CBV2, 
             self.CBV3) = ut.generate_clip_quats_cbvs(self.sector, 
                                                      self.time,
                                                      self.intensity,
                                                      self.error, 
                                                      self.tmin, 
                                                      self.camera, 
                                                      self.ccd,
                                                      self.CBV_folder, 
                                                      self.quaternion_folder_txt)
            self.quatsandcbvs = [self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3]
        elif not self.cbvsquatsActive and fitType in (2,4,5):
            print("You need to provide quaternion locations via use_quaternions_cbvs()")
            raise ValueError("Cannot run the requested fit type")
        else:
            self.quatsandcbvs = None # has to initiate as None or it'll freak
 
        ### THEN DO CUSTOM MASKING if both not already cut and indices are given
        if not hasattr(self, 'cutindexes') and cutIndices is not None:
            self.__custom_mask_it(cutIndices, saveplot = None)
       
        # 8hr binning
        if binYesNo and self.binned == False: #if need to bin
            (self.time, self.intensity, 
             self.error, self.lygosbg,
             self.quatsandcbvs) = ut.bin_8_hours(self.time, self.intensity, self.error, 
                                                 self.lygosbg, QCBVALL=self.quatsandcbvs)                                    
            self.binned = True # make sure you can't bin it more than once
                                                 
        # percent of max fitting
        if fraction is not None and self.fractiontrimmed==False:
            (self.time, self.intensity, self.error, self.lygosbg, 
             self.quatsandcbvs) = ut.fractionalfit(self.time, self.intensity, 
                                                   self.error, self.lygosbg, 
                                                   fraction, self.quatsandcbvs)
            self.fractiontrimmed=True #make sure you can't trim it more than once
            self.fract = fraction
            
        
        # this is to fix the quats and cbv inputs after trimming                       
        if fitType in (2,4,5):
            self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3 = self.quatsandcbvs
         
        #once  you are done with this stuff:
        self.cleaningdone = True 
        self.fitType = fitType
        return
    
    def __custom_mask_it(self, cutIndices, saveplot = None):
        """remove certain indices from your light curve.
        cutIndices should be an array of size len(time), 0 = remove, 1=keep
        
        *****this should NOT be used if using CBVs - that's not set up yet!!
        """
        if hasattr(self, 'cutindexes'): #if already did a trim
            print("*******")
            print("ALREADY TRIMMED - RELOAD AND TRY AGAIN")
            print("*******")
            return
        elif hasattr(self, 'time'): #if something loaded in and going to trim
            #plt.scatter(self.time, self.intensity, color='red', s=2)
                
            self.cutindexes = np.nonzero(cutIndices) # which ones you are keeping
            self.time = self.time[self.cutindexes]
            self.intensity = self.intensity[self.cutindexes]
            self.error = self.error[self.cutindexes]
            if self.lygosbg is not None:
                self.lygosbg = self.lygosbg[self.cutindexes]
                
            if hasattr(self, 'quatsIntensity'): #if cbvs, trim them
                self.quatsIntensity = self.quatsIntensity[self.cutindexes]
                self.CBV1 = self.CBV1[self.cutindexes]
                self.CBV2 = self.CBV2[self.cutindexes]
                self.CBV3 = self.CBV3[self.cutindexes]
                self.quatsandcbvs = [self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3]
                
            #self.custommasked = True
            #plt.scatter(self.time, self.intensity, color='blue', s=2)
            #plt.xlabel(self.xlabel)
            #plt.ylabel(self.ylabel)
            #plt.show()
            #if saveplot is not None:
            #    plt.savefig(saveplot)
            #plt.close()
            return
        else:
            print("No data loaded in yet!! Run again once light curve is loaded")
            return     
   
    def __gen_output_folder(self):
        """
        Internal function, sets up output folder for files
        
        """
        # check for an output folder's existence, if not, put it in. 
        if (self.folderSAVE is None or 
            self.targetlabel is None or 
            self.sector is None
            or self.camera is None or 
            self.ccd is None):
            raise ValueError("Cannot generate output folders, one of the parameters is None")
        
        internaluse = self.targetlabel + str(self.sector) + str(self.camera) + str(self.ccd)
        newfolderpath = (self.folderSAVE + internaluse)
        if not os.path.exists(newfolderpath):
            os.mkdir(newfolderpath)
        # make subfolder for this run
        #print(internaluse)
        print("files willl be tagged: ", self.filesavetag)
        subfolderpath = newfolderpath + "/" + self.filesavetag[1:]
        if not os.path.exists(subfolderpath):
            os.mkdir(subfolderpath)
        self.folderSAVE = subfolderpath + "/"
        self.parameterSaveFile = self.folderSAVE + internaluse + self.filesavetag + "-output-params.txt"
        print("saving into folder: ",self.folderSAVE)
        return 
   
    def __setup_fittype_params(self, fitType, args=None, 
                               logProbFunc = None, plotFit = None,
                               filesavetag=None, labels=None, init_values=None,
                               mu = 2, sigma = 1):
        """
        
        Internal function to set up self.plotFit, logProbFunc, filesavetag, labels
        filelabels, init_values array, args array
        
        1 = single without
        2 = single with cbv
        3 = doubel without
        4 = double with
        5 = just cbv
        6 = single lygos bg
        7 = gaussian prior on beta
        0 = custom inputs
        any other number = will exit with an error
        
        IF YOU ARE DOING CUSTOM PRIORS:
            - run under fitType=0
            - last item in args must be your priors array -- all probability functions
            come with a positional argument priors=None that this should override
            
            
        """
        self.plotFit = fitType
        
        
        if fitType == 1: # single without
            if args is None:
                self.args = (self.time, self.intensity, self.error, self.disctime)
            else:
                self.args = args
            self.logProbFunc = mc.log_probability_singlepower_noCBV
            self.filesavetag = "-singlepower"
            self.labels = ["t0", "A", "beta",  "b"]
            self.filelabels = self.labels
            self.init_values = np.array((self.disctime-3, 0.1, 1.8, 1))
            
        elif fitType == 2: # single with
            if args is None:
                self.args = (self.time, self.intensity, self.error, 
                             self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3, 
                             self.disctime)
            else:
                self.args = args
            self.logProbFunc = mc.log_probability_singlepower_withCBV
            self.filesavetag = "-singlepower-CBV"
            self.labels = ["t0", "A", "beta", "B", "cQ", "c1", "c2", "c3"]
            self.filelabels = self.labels
            self.init_values = np.array((self.disctime-3, 0.1, 1.8, 0, 0,0,0,0))
            
        elif fitType == 3: # double without
            if args is None:
                self.args = (self.time, self.intensity, self.error, self.disctime)
            else:
                self.args = args
                
            self.logProbFunc = mc.log_probability_doublepower_noCBV
            self.filesavetag = "-doublepower"
            self.labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  "b"]
            self.filelabels = self.labels
            self.init_values = np.array((self.disctime-8, self.disctime-4, 0.1, 0.1, 1.8, 1.8, 1))

        elif fitType ==4: # double with
            if args is None:
                self.args = (self.time, self.intensity, self.error, 
                             self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3, 
                             self.disctime)
            else:
                self.args = args
            self.logProbFunc = mc.log_probability_doublepower_withCBV
            self.filesavetag = "-doublepower-CBV"
            self.labels = ["t1", "t2", "a1", "a2", "beta1", "beta2",  
                      "cQ", "c1", "c2", "c3"]
            self.filelabels = self.labels
            self.init_values = np.array((self.disctime-8, self.disctime-2, 0.1, 0.1, 
                                    1.8, 1.8, 0,0,0,0))
        elif fitType == 5: # just CBVs
            if args is None:
                self.args = (self.time, self.intensity, self.error, 
                             self.quatsIntensity, self.CBV1, self.CBV2, self.CBV3, 
                             self.disctime)
            else:
                self.args = args
            self.logProbFunc = mc.log_probability_justCBV
            self.filesavetag = "-CBV"
            self.labels = ["b", "cQ", "c1", "c2", "c3"]
            self.filelabels = self.labels
            self.init_values = np.array((1, 0,0,0,0))
        elif fitType == 6: # detrending lygos BG
            if self.lygosbg is None:
                print("NO LYGOS BG LOADED IN - CANNOT RUN THIS FIT")
                raise AttributeError("Missing lygosbg!!")
                return
            else:
                self.args = (self.time, self.intensity, self.error, self.lygosbg, 
                             self.disctime)
                self.logProbFunc = mc.log_probability_singlePower_LBG
                self.filesavetag = "-singlepower-lygosBG"
                self.labels = ["t0", "A", "beta",  "b", "LBG"]
                self.filelabels = self.labels
                self.init_values = np.array((self.disctime-3, 0.1, 1.8, 1, 1))
         
        elif fitType == 7: # gaussian beta
            if args is None:
                self.args = (self.time, self.intensity, self.error, self.disctime,
                             mu, sigma)
            else:
                self.args = args
            self.logProbFunc = mc.log_probability_singlepower_gaussianbeta
            self.filesavetag = "-singlepower-GBeta"
            self.labels = ["t0", "A", "beta",  "b"]
            self.filelabels = self.labels
            self.init_values = np.array((self.disctime-3, 0.1, 1.8, 1))
        
        elif fitType == 0: # diy your stuff
            self.args = args # LAST ONE MUST BE PRIORS IF USING CUSTOMS
            self.logProbFunc = logProbFunc
            self.filesavetag = filesavetag
            self.labels = labels
            self.filelabels = self.labels
            self.init_values = init_values 
            #print(self.args)
            self.plotFit = plotFit #overwrites the default fitType that was set to plotfit
        else:
            print("THAT IS NOT AN ALLOWED DEFAULT FIT TYPE, EXITING")
            raise ValueError("not an allowed fit type")
            
        if self.binned: # it has to go in this order - need to load, then set args, then set this
            self.filesavetag = self.filesavetag + "-8HourBin"
    
        if self.fractiontrimmed:
            self.filesavetag = self.filesavetag + "-{fraction}".format(fraction=self.fract)
        
        
        if filesavetag is not None:
            self.filesavetag = filesavetag
        
        return
   
    
    def run_MCMC(self, n1=1000, n2=10000, thinParams = None,
                 saveBIC=False, args=None, logProbFunc = None, plotFit = None,
                 filesavetag=None,
                 labels=None, init_values=None, mu=2, sigma=1):
        """
        Run one MCMC instance - non GP fits
        
        Order of data cleaning:
            1) load in CBVs/quats
            2) handle custom masking of points
            3) bin 
            4) fractional cutoff applied to flux
                                                                
        
        ---------------------------------------------------
        Parameters:
            
            - n1 (int def 1000) burn in first chain length
            - n2 (int def 10000) production chain length
            
            
            - thinParams, NONE to use defaults (1/4 first run discard, 15% trime) 
                            or [first run discard, percent thin]
                            
            - saveBIC (bool) do you want to save the BIC value that comes out
            
            - args, NONE for 1-6, use for 0 if you want it (see below)
            
            - logProbFunc, NONE for 1-6, use for 0
            
            - plotFit = NONE unless doing a custom fit - needs to match up with the 
                logprobfunc being used
                
            - filesavetag = NONE, custom string if you want it
            
            - labels, NONE unless doing custom bullshit, then array of str
            
            - init_values, NONE unless doing custom bullshit
        
        If you are doing custom priors:
            - don't
            - run under fitType = 0
            - last items in args must be your priors array -- all probability functions
                come with a positional argument priors = None that this should override
            - hopefully the tutorial level shows how to set this up right
    

        """

        if not self.cleaningdone:
            raise Exception("You must run self.pre_run_clean() first!")
        # load parameters by fit type                                           
        self.__setup_fittype_params(self.fitType, args,
                                    logProbFunc, plotFit, filesavetag, 
                                    labels, init_values, mu, sigma)
        # set up the output folder
        self.__gen_output_folder() 
                                                        
        # run it
        (self.best, self.upperError, 
         self.lowerError, self.bic) = self.__mcmc_outer_structure(n1, n2, thinParams)
        
        if saveBIC:
            self.bic_all.append(self.bic)
            self.params_all.append(self.best)
            
        return (self.best, self.upperError, self.lowerError, self.bic) 
      
        
    def __mcmc_outer_structure(self, n1, n2, thinParams):
        """Fitting things that are NOT GP based
        Params:
            - n1 is an integer number of steps for the first chain
            - n2 is an integer number of steps for the second chain
            - thinParams is EITHER NONE (default thinning is used, 1/4 for the first run,
                                         15% thinning) or [int to discard, thinning percent]
        """
        
        print("***")
        print("***")
        print("***")
        print("***")
        print("Beginning MCMC run")
         
        timeModule.sleep(3) # this keeps things running orderly
        
        if thinParams is None:
            discard1 = int(n1/4)
            thinning = 15
        else:
            discard1, thinning = thinParams
                              

        # ### MCMC setup
        np.random.seed(42)
        
        nwalkers = 100
        ndim = len(self.labels) # labels are provided when you run it
        p0 = np.zeros((nwalkers, ndim)) # init positions
        for n in range(len(p0)): # add a little spice - YYY gaussian??
            p0[n] = self.init_values + (np.ones(ndim) - 0.9) * np.random.rand(ndim) 
        
        print("Starting burnin chain")
        # ### Initial run
        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                        self.logProbFunc,args=self.args) # setup
        sampler.run_mcmc(p0, n1, progress=True) # run it
        
        sp.plot_chain_logpost(self.folderSAVE, self.targetlabel, self.filesavetag, 
                              sampler, self.labels, ndim, appendix = "-burnin")
    
        flat_samples = sampler.get_chain(discard=discard1, flat=True, thin=thinning)
        # get intermediate best
        best_mcmc_inter = np.zeros((1,ndim))
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            best_mcmc_inter[0][i] = mcmc[1]
            
        # ### Main run
        np.random.seed(50)
        p0 = np.zeros((nwalkers, ndim))
        for i in range(nwalkers): # reinitialize the walkers around prev. best
            p0[i] = best_mcmc_inter[0] + 0.1 * np.random.rand(1, ndim)
           
        sampler.reset()
        
        # ### CORRELATION FUNCTION 
        index = 0 # number of checks
        autocorr = np.empty(n2) # total possible checks
        old_tau = np.inf
        autoStep = 100 # how often to check
        autocorr_all = np.empty((int(n2/autoStep) + 2,len(self.labels))) # save all autocorr times
        
        # sample up to n2 steps
        for sample in sampler.sample(p0, iterations=n2, progress=True):
            # Only check convergence every 100 steps
            if sampler.iteration % autoStep:
                continue
            # Compute the autocorrelation time so far
            tau = sampler.get_autocorr_time(tol=0) # tol=0 always get estimate
            if np.any(tau == np.nan) or np.any(tau == np.inf) or np.any(tau == -np.inf):
                print("autocorr is nan or inf")
                print(tau)
            autocorr[index] = np.mean(tau) # save mean autocorr time
            autocorr_all[index] = tau # save all autocorr times for plotting
            index += 1 # how many times have you saved it
        
            # Check convergence
            converged = np.all((tau * 100) < sampler.iteration)
            converged &= np.all((np.abs(old_tau - tau) / tau) < 0.01) # normally 0.01
            if converged:
                print("Converged, ending chain")
                break
            old_tau = tau
        
        # ######
        #plot autocorr things
        ########
        sp.plot_autocorr_mean(self.folderSAVE, self.targetlabel, index, 
                              autocorr, converged, 
                              autoStep, self.filesavetag)
        sp.plot_autocorr_individual(self.folderSAVE, self.targetlabel, index,
                                    autocorr_all, autoStep, self.labels,
                                    self.filelabels, 
                                    self.filesavetag)
        
            
        #thin and burn out dump
        tau = sampler.get_autocorr_time(tol=0)
        if (np.max(tau) < (sampler.iteration/50)):
            burnin = int(2 * np.max(tau))
            thinning = int(0.5 * np.min(tau))
        else:
            burnin = int(n2/4)

        flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thinning)
        
        # plot chains, parameters
        sp.plot_chain_logpost(self.folderSAVE, self.targetlabel, self.filesavetag,
                              sampler, self.labels, ndim, appendix = "-production")
        
        sp.plot_paramTogether(flat_samples, self.labels, self.folderSAVE, 
                                  self.targetlabel, self.filesavetag)
        
        print(len(flat_samples), "samples post second run")
    
        # ### BEST FIT PARAMS
        best_mcmc = np.zeros((1,ndim))
        upper_error = np.zeros((1,ndim))
        lower_error = np.zeros((1,ndim))
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            print(self.labels[i], mcmc[1], -1 * q[0], q[1] )
            best_mcmc[0][i] = mcmc[1]
            upper_error[0][i] = q[1]
            lower_error[0][i] = q[0]
     
        logprob, blob = sampler.compute_log_prob(best_mcmc)

        # ### BIC
        #print(np.log(len(self.time)))
        print("log prob:", logprob)
        BIC = ndim * np.log(len(self.time)) - 2 * logprob
        print("BAYESIAN INF CRIT: ", BIC)
        if np.isnan(np.float64(BIC[0])): # if it's a nan
            BIC = 50000
        else:
            BIC = BIC[0]
             
        
        if self.plotFit < 10:
            sp.plot_mcmc(self.folderSAVE, self.time, self.intensity, self.error,
                         self.targetlabel, 
                         self.disctime, best_mcmc[0], 
                         flat_samples, self.labels, self.plotFit, self.filesavetag, 
                         self.tmin, self.lygosbg,
                         self.quatsandcbvs)
        elif self.plotFit == 10:
            sp.plot_mcmc_GP_celerite(self.folderSAVE, self.time, self.intensity, 
                            self.error, best_mcmc, self.gp, self.disctime, 
                            self.tmin,self.targetlabel, self.filesavetag, 
                            plotComponents=False)
        elif self.plotfit == 11:
            sp.plot_mcmc_GP_tinygp(self.folderSAVE, self.time, self.intensity, 
                            self.error, best_mcmc, self.disctime, 
                            self.tmin,self.targetlabel, self.filesavetag, 
                            plotComponents=False)
        
        with open(self.parameterSaveFile, 'w') as file:
            #file.write(self.filesavetag + "-" + str(datetime.datetime.now()))
            file.write("\n {best} \n {upper} \n {lower} \n".format(best=best_mcmc[0],
                                                                   upper=upper_error[0],
                                                                   lower=lower_error[0]))
            file.write("BIC:{bicy:.3f} \n Converged:{conv} \n".format(bicy=BIC, 
                                                                conv=converged))
        
        return best_mcmc, upper_error, lower_error, BIC
    
    def pre_celerite_setup(self, customSigmaRho=None, filesavetag=None):
        """ 
        Set up celerite matern 3-2 kernel (either to default or custom params)
        customSigmaRho must unpack as: [sigma start, rho start, 
                                        sigma lower, sigma upper,
                                        rho lower, rho upper, 
                                        sigma frozen (bool), rho frozen (bool)]
        the default run of this is [0.01, 1.2, 
                                    0.0001, 0.3, 
                                    1, 2, 
                                    0, 0]
        """
        if filesavetag is None:
            self.filesavetag = "-matern32-celerite"
        else:
            self.filesavetag = filesavetag
        #set up kernel
        # SET UP NEW MATERN-32 GP
        if customSigmaRho is None:
            sigma = 0.01 #amplitude
            rho = 1.2 #timescale
            sigma_bounds = (0.0001,0.3)
            rho_bounds = (1,2)
            bounds_dict = dict(log_sigma=np.log(sigma_bounds), log_rho=np.log(rho_bounds))
            kernel = terms.Matern32Term(log_sigma=np.log(sigma), log_rho=np.log(rho), 
                                        bounds=bounds_dict)
            self.init_values = np.array((self.disctime-3, 0.1, 1.8, 0,np.log(sigma), np.log(rho)))
        else:
            sigma = customSigmaRho[0]
            rho = customSigmaRho[1]
            sigma_bounds = (customSigmaRho[2], customSigmaRho[3])
            rho_bounds = (customSigmaRho[4], customSigmaRho[5])
            bounds_dict = dict(log_sigma=np.log(sigma_bounds), log_rho=np.log(rho_bounds))
            kernel = terms.Matern32Term(log_sigma=np.log(sigma), log_rho=np.log(rho), 
                                        bounds=bounds_dict)
            self.init_values = np.array((self.disctime-3, 0.1, 1.8, 0))
            
            if customSigmaRho[6]: #if frozen (1)
                kernel.freeze_parameter("log sigma")
            else: # if not frozen (0)
                initsigma = np.array((np.log(sigma))) # if not frozen, add to init
                self.init_values = np.concatenate((self.init_values, initsigma))
                
            if customSigmaRho[7]: # if frozen true
                kernel.freeze_parameter("log rho")
            else:
                initrho = np.array((np.log(rho)))
                self.init_values = np.concatenate((self.init_values, initrho))
            
            
        self.gp = celerite.GP(kernel, mean=0.0)
        self.gp.compute(self.time, self.error)
        print("Initial log-likelihood: {0}".format(self.gp.log_likelihood(self.intensity)))
        # set up arguments etc.
        self.args = (self.time,self.intensity, self.error, self.disctime, self.gp)
        self.logProbFunc = mc.log_probability_celerite
        self.labels = ["t0", "A", "beta",  "b", r"$log\sigma$",r"$log\rho$"] 
        self.filelabels = ["t0", "A", "beta",  "b",  "logsigma", "logrho"]
        
        self.celerite_setup = True
        
        return
    
    
    def run_GP_fit_celerite(self, cutIndices, binYesNo, fraction=None, 
                   n1=1000, n2=10000, 
                   customSigmaRho = None, thinParams=None):
        """Run the GP fitting 
        
        *** must first run pre_celerite_setup()
        
        
        """
        if not hasattr(self, 'celerite_setup'):
            #if you have not yet setup the run params
            raise AttributeError("You have to run pre_celerite_setup() first!")
            return
        
        ### THEN DO CUSTOM MASKING if both not already cut and indices are given
        if not hasattr(self, 'cutindexes') and cutIndices is not None:
            self.__custom_mask_it(cutIndices, saveplot = None)
            
        # check for 8hr bin BEFORE trimming to percentages
        if binYesNo: #if need to bin
            (self.time, self.intensity, 
             self.error, self.lygosbg,
             self.quatsandcbvs) = ut.bin_8_hours(self.time, self.intensity, self.error, 
                                                 self.lygosbg, QCBVALL=None) 
                                                 
        # if doing percent of max fitting
        if fraction is not None:
            (self.time, self.intensity, self.error, self.lygosbg, 
             self.quatsandcbvs) = ut.fractionalfit(self.time, self.intensity, 
                                                   self.error, self.lygosbg, 
                                                   fraction, self.quatsandcbvs)
        #make folders to save into
        self.__gen_output_folder()   
        
        fitType = 10
        self.plotFit = 10
        self.__mcmc_outer_structure(n1, n2, thinParams)
        return
    
    def run_GP_fit_tinygp(self, cutIndices, binYesNo, fraction=None, 
                          n1=1000, n2=10000, filesavetag=None,
                          thinParams=None):
        """
        GP fitting using tinygp's expsqr stuff
        Update 10-7-22 - trying to do a GP fit every 100 or 1000 steps?
        
        """
        from tinygp import kernels, GaussianProcess
        import jax
        import jax.numpy as jnp
        
        if filesavetag is None:
            self.filesavetag = "-tinygp-expsqr"
        else:
            self.filesavetag = filesavetag
            
        ### THEN DO CUSTOM MASKING if both not already cut and indices are given
        if not hasattr(self, 'cutindexes') and cutIndices is not None:
            self.__custom_mask_it(cutIndices, saveplot = None)
            
        # check for 8hr bin BEFORE trimming to percentages
        if binYesNo: #if need to bin
            (self.time, self.intensity, 
             self.error, self.lygosbg,
             self.quatsandcbvs) = ut.bin_8_hours(self.time, self.intensity, self.error, 
                                                 self.lygosbg, QCBVALL=None) 
                                                 
        # if doing percent of max fitting
        if fraction is not None:
            (self.time, self.intensity, self.error, self.lygosbg, 
             self.quatsandcbvs) = ut.fractionalfit(self.time, self.intensity, 
                                                   self.error, self.lygosbg, 
                                                   fraction, self.quatsandcbvs)
        #make folders to save into
        self.__gen_output_folder()   


        # set up arguments etc.
        self.args = (self.time,self.intensity, self.error, self.disctime)
        self.init_values = np.array((self.disctime-3, 0.1, 1.8, 0.0, np.log(0.3), np.log(1.1)))
   
        self.logProbFunc = mc.log_probability_tinygp
        self.labels = ["t0", "A", "beta",  "b", r"$\alpha$","length"] 
        self.filelabels = ["t0", "A", "beta",  "b",  "alpha", "length"]
        
        fitType = 11
        self.plotFit = 11
        print("entering mcmc outer")
        self.__mcmc_outer_structure(n1, n2, thinParams)
        return
    
    def run_postfit_tinygp_fit(self,cutIndices, binYesNo, fraction=None, 
                   n1=1000, n2=10000, filesavetag=None,
                   args=None, thinParams=None, saveBIC=False, 
                   logProbFunc = None, plotFit = None,
                   labels=None, init_values=None):
        """
        Fit a single power law, then fit the tinygp to the residual
        """
        self.run_MCMC(n1, n2, thinParams,
                     saveBIC, args, logProbFunc, plotFit,
                     filesavetag,
                     labels, init_values, mu=None, sigma=None)
        
        t0, A, beta, B = self.best[0]
        t1 = self.time - t0
        sl = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False)
        bg = np.ones(len(self.time)) + B
        
        residual = self.intensity - sl - bg
        
        #fit  tinygp to the residual
        
        import jax
        import jax.numpy as jnp
        from tinygp import kernels, GaussianProcess
        jax.config.update("jax_enable_x64", True)


        def build_gp(theta, X):
            amps = jnp.exp(theta["log_amps"])
            scales = jnp.exp(theta["log_scales"])

            k1 = amps[0] * kernels.ExpSquared(scales[0])
            return GaussianProcess(
                k1, X, diag=jnp.exp(theta["log_diag"]), mean=theta["mean"]
            )

        def neg_log_likelihood(theta, X, y):
            #this needs to have posteriors
            
            lp = 0
            # if (jnp.exp(theta["log_amps"]) < -10 or 
            #     jnp.exp(theta["log_amps"]) > 10 or
            #     jnp.exp(theta["log_scales"]) < -10 or
            #     jnp.exp(theta["log_scales"]) > 10):
                
            #     #if not allowed ->we're minimizing this func output
            #     #so punish bad values w/ high lp?
            #     lp = jnp.inf
            
            gp = build_gp(theta, X)
            return -gp.log_probability(y) + lp


        theta_init = {
            "mean": np.float64(0.0),
            "log_diag": np.log(0.19),
            "log_amps": np.log([10.0]),
            "log_scales": np.log([5.0]),
        }

        # `jax` can be used to differentiate functions, and also note that we're calling
        # `jax.jit` for the best performance.
        obj = jax.jit(jax.value_and_grad(neg_log_likelihood))
        t = self.time
        y = residual

        print(f"Initial negative log likelihood: {obj(theta_init, t, y)[0]}")
        print(
            f"Gradient of the negative log likelihood, wrt the parameters:\n{obj(theta_init, t, y)[1]}"
        )
        import jaxopt

        solver = jaxopt.ScipyMinimize(fun=neg_log_likelihood)
        soln = solver.run(theta_init, X=t, y=y)
        print(f"Final negative log likelihood: {soln.state.fun_val}")

        print(soln.params)
        amps = np.exp(soln.params["log_amps"][0])
        scales = np.exp(soln.params["log_scales"][0])
        
        
        
        #plot the residual stuff
        best_mcmc = [t0, A, beta, B, amps, scales]
        
        
        #needs to contain the four first and then the gp params
        sp.plot_mcmc_GP_tinygp(self.folderSAVE, self.time, 
                               self.intensity, self.error, best_mcmc,
                                self.disctime, t0, self.tmin,
                                self.targetlabel, self.filesavetag, 
                                plotComponents=False)
        
        return
        