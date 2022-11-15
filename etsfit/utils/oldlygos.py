#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 12:40:54 2022

@author: lindseygordon

old lygos code
"""

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

 # def __pre_celerite_setup(self, customSigmaRho=None, filesavetag=None):
 #     """ 
 #     Set up celerite matern 3-2 kernel (either to default or custom params)
 #     customSigmaRho must unpack as: [sigma start, rho start, 
 #                                     sigma lower, sigma upper,
 #                                     rho lower, rho upper, 
 #                                     sigma frozen (bool), rho frozen (bool)]
 #     the default run of this is [0.01, 1.2, 
 #                                 0.0001, 0.3, 
 #                                 1, 2, 
 #                                 0, 0]
 #     """
 #     if filesavetag is None:
 #         self.filesavetag = "-celerite-matern32"
 #     else:
 #         self.filesavetag = filesavetag
 #     #set up kernel
 #     start_t = min(self.disctime-3, self.time[-1]-2)
 #     # SET UP NEW MATERN-32 GP
 #     if customSigmaRho is None:
 #         rho = 2 # init value
 #         sigma = 1
 #         rho_bounds = np.log((1, 10)) #0, 2.302
 #         sigma_bounds = np.log( np.sqrt((0.1,20  )) ) #sigma range 0.316 to 4.47, take log
 #         bounds_dict = dict(log_sigma=sigma_bounds, log_rho=rho_bounds)
 #         kernel = terms.Matern32Term(log_sigma=np.log(sigma), log_rho=np.log(rho), 
 #                                     bounds=bounds_dict)
         
 #         self.init_values = np.array((start_t, 0.1, 1.8, 0,np.log(sigma), np.log(rho)))
 #     else:
 #         sigma = customSigmaRho[0]
 #         rho = customSigmaRho[1]
 #         sigma_bounds = (customSigmaRho[2], customSigmaRho[3])
 #         rho_bounds = (customSigmaRho[4], customSigmaRho[5])
 #         bounds_dict = dict(log_sigma=np.log(sigma_bounds), log_rho=np.log(rho_bounds))
 #         kernel = terms.Matern32Term(log_sigma=np.log(sigma), log_rho=np.log(rho), 
 #                                     bounds=bounds_dict)
 #         self.init_values = np.array((start_t, 0.1, 1.8, 0))
         
 #         if customSigmaRho[6]: #if frozen (1)
 #             kernel.freeze_parameter("log sigma")
 #         else: # if not frozen (0)
 #             initsigma = np.array((np.log(sigma))) # if not frozen, add to init
 #             self.init_values = np.concatenate((self.init_values, initsigma))
             
 #         if customSigmaRho[7]: # if frozen true
 #             kernel.freeze_parameter("log rho")
 #         else:
 #             initrho = np.array((np.log(rho)))
 #             self.init_values = np.concatenate((self.init_values, initrho))
         
         
 #     self.gp = celerite.GP(kernel, mean=0.0)
 #     self.gp.compute(self.time, self.error)
 #     print("Initial log-likelihood: {0}".format(self.gp.log_likelihood(self.intensity)))
 #     # set up arguments etc.
 #     self.args = (self.time,self.intensity, self.error, self.disctime, self.gp)
 #     self.logProbFunc = mc.log_probability_celerite
 #     self.labels = ["t0", "A", "beta",  "b", r"$log\sigma$",r"$log\rho$"] 
 #     self.filelabels = ["t0", "A", "beta",  "b",  "logsigma", "logrho"]
 #     self.plotFit = 10
     
 #     return