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