#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 09:24:31 2022

@author: lindseygordon

yet another crossmatch file

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from astropy.time import Time
import gc
import tessreduce as tr
import time
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord
from astropy import units as u


def reduce_list(csvall, filesave, typeKey, fluxLim):
    """
    Given a path to a file containing a TNS output of SN, save a new csv file
    containing all of the targets that are classified type Ias and also brighter than
    18th magnitude
    
    -----------------------------------------
    Parameters:
        - csvall, (str) direct path to full TNS file
        - filesave, (str) direct path to file you want to save output into
        - typeKey, (str), will seatch Obj. Type by contains this (ie, "SN Ia")
        - fluxLim, (int/double), will search discovery mags for < this cutoff
    """
    biglist = pd.read_csv(csvall)
    #typekey reduce
    biglist = biglist[biglist["Obj. Type"].str.contains(typeKey)]
    #discovery magnitude list
    biglist = biglist[biglist["Discovery Mag/Flux"]<fluxLim].reset_index()
    biglist.to_csv(filesave)
    return

def prep_WTV_file(tnsfile, outputfile):
    """This converts all RA and DEC in an input pandas-readable CSV file
    from whatever format it currently is in into degrees and saves it into 
    its own new csv file."""
    #converting to decimal degrees
    import pandas as pd
    from astropy.coordinates import Angle
    import astropy.units as u
    df = pd.read_csv(tnsfile)
    print (df)
    
    for n in range(len(df)):
        a = Angle(df['RA'][n], u.degree)
        a = a.degree
        df['RA'][n] = a
        b = Angle(df['DEC'][n], u.degree)
        b = b.degree
        df['DEC'][n] = b
    
    new = df[['RA', 'DEC']].copy()
    print(new)
    new.to_csv(outputfile, index = False)
    return

def process_WTV_results(TNS_file, WTV_file, output_file):
    """
    Go through the WTV output file and compare with the TNS sector to ID those
    that WTV thinks TESS should have found
    ----------------------------------------
    Parameters:
        - TNS_file (str) points to file holding all TNS targets
        - WTV_file (str) points to file holding WTV results
        - output_file (str) points to where to save results
    
    """
    all_tns = pd.read_csv(TNS_file)
    all_wtv = pd.read_csv(WTV_file, skiprows=61)

    just_sectors = all_tns["Sector"]
    #counter = 0
    WTV_confirmed =  pd.DataFrame(columns = all_tns.columns)
    for n in range(len(all_wtv)-1):
        correct_sector = all_tns["Sector"][n]
        columnname = "S" + str(correct_sector)
        if all_wtv[columnname][n] != 0.0: 
            WTV_confirmed = WTV_confirmed.append(all_tns.iloc[[n]])
            
    WTV_confirmed.reset_index(inplace = True, drop=True)    
    WTV_confirmed.to_csv(output_file, index=False)
    return WTV_confirmed



def add_sector(file, sector):
    """ 
    for a given sector's csv file, add a column to all rows with the sector
    """
    listy = pd.read_csv(file)
    sectorcolumn = np.zeros(len(listy)) + sector
    listy["Sector"] = sectorcolumn
    outfile = file[:-4] + "-sectorAdded.csv"
    listy.to_csv(outfile)
    return

def compile_csvs(folder, suffix, savefilename = None):
    """
    Compile all CSV's in a folder 
    Parameters:
        - folder: containing all csv files
        - suffix: ending to search for on end of all files (probably -tesscut.csv)
        - savefilename: if not none, will save output of this into this file
        
    Returns: concatenated pandas data frame containing all info from the csv files"""
    all_info = pd.DataFrame()
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith((suffix)):
                filepath = root  + f
                p = pd.read_csv(filepath)
                
                if all_info.empty:
                    all_info = p #first load
                else:
                    all_info = pd.concat((all_info, p)) #other ones
                    
    if savefilename is not None:
        print(folder + savefilename + ".csv")
        del all_info["Unnamed: 0"]
        all_info.to_csv(folder + savefilename + ".csv", index=False)
        
    return all_info.reset_index()
 
 
def sectorfile_tesscut_match(TNSfile, savefile, sector):
    """
    For a given CSV file (TNS format) with sectors attached to each target,
    run tesscut on coordinates to see if it was observed in that sector
    """
    clist = pd.read_csv(TNSfile)
    del clist["Unnamed: 0"]
    goodIndices = []

    #for each, get ra/dec, run thru tesscut
    for n in range(len(clist)):
        cutout_coord = SkyCoord(clist["RA"].iloc[n], clist["DEC"].iloc[n], 
                                unit=(u.hourangle, u.deg))
        #print(cutout_coord)
        sector_table = Tesscut.get_sectors(coordinates=cutout_coord)
        #print(sector_table)
        #traverse the table
        for i in range(len(sector_table)):
            if sector_table[i][1] == sector:
                goodIndices.append(n)
        
        time.sleep(3)
    print(goodIndices)
    #retrieve clipped clist, save into savefile
    clist_clipped = clist.iloc[goodIndices]
    clist_clipped.to_csv(savefile)
    return

def run_all_tesscut_matches(folderIn, folderOut, startsat=None):
    """ 
    For a folder full of cycle-sorted sectorized csv files, run tesscut mtching and 
    save the results
    """
    k = 0
    for i in (1,2,3):
        folderIn2 = folderIn + "CYCLE{num}/".format(num=i)
        print(folderIn2)
        for n in range(1,14):
            if startsat is not None and (n+k < startsat):
                continue
            folderIn3 = folderIn2 + "Sector{numb}-sectored.csv".format(numb=n+k)
            print(folderIn3)
            print(n+k)
            fileout = folderOut + "sector{numb}-tesscut.csv".format(numb=n+k)
            sectorfile_tesscut_match(folderIn3, fileout, n+k)
            
        k += 13
    return


def get_not_in_common_entries(file1, file2):
    """
    For 2 TNS valued csvs, find the entries that only exist in one list
    the left file will be the file1, file2 is right (for merge)
    """

    file1 = pd.read_csv(tesscut_all)
    file2 = pd.read_csv(wtv_all)
    
    df_all_left = file1.merge(file2.drop_duplicates(), on=["Name", "Name"], how="left", indicator=True)
    df_ind_left = df_all_left[df_all_left["_merge"] != "both"]
    df_all_right = file1.merge(file2.drop_duplicates(), on=["Name", "Name"], how="right", indicator=True)
    df_ind_right = df_all_right[df_all_right["_merge"] != "both"]
    df_ind = pd.concat((df_ind_left, df_ind_right))
    return df_ind

tesscut_all = "/Users/lindseygordon/research/urop/august2022crossmatch/all-tesscut-matches.csv"
wtv_all = "/Users/lindseygordon/research/urop/august2022crossmatch/WTV-matched-all.csv"
df_ind = get_not_in_common_entries(wtv_all,tesscut_all)
df_ind.to_csv("/Users/lindseygordon/research/urop/august2022crossmatch/uniqueToOneList.csv")


compile_csvs("/Users/lindseygordon/research/urop/august2022crossmatch/cycle3/", 
             "tesscut.csv", savefilename = "cycle3-tesscut-all")

reduce_list(tesscut_all, "/Users/lindseygordon/research/urop/august2022crossmatch/Ia205mag.csv", 
            "Ia", 20.5)