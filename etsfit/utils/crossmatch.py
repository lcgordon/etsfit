#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 09:24:31 2022
Updated: Oct 6 2023 - new docstring format
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import gc
import time
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord
from astropy import units as u


def TNS_URL_gen(date_start = None, date_end = None, discovered_within = None, 
            discovered_within_units = None, unclassified_at = False, 
            classified_sne = True):
    """ 
    Produces a URL for searching TNS - annoyingly this has to be copypasted in now,
    and also only does 50 results per page. 

    :param date_start: format: "2020-12-11"
    :param data_end: format: "2020-12-11"
    :param discovered_within: discovered within is "2" 
    :param discovered_within_units: "days" "months" or "years"
    :param unclassified_at: bool
    :param classified_sne: bool 

    :return: url string
    """
    
    query = ""
    if date_start is not None and date_end is not None:
        query = query + "&date_start%5Bdate%5D=" + date_start
        query = query + "&date_end%5Bdate%5D=" + date_end
    
    if discovered_within is not None and discovered_within_units is not None:
        query = query + "&discovered_period_value=" + discovered_within
        query = query + "&discovered_period_units=" + discovered_within_units
    
    if unclassified_at: #include them
        query = query + "&unclassified_at=1"
            
    if classified_sne: #only classified supernovae
        query = query + "&classified_sne=1"
    
    url = ("https://www.wis-tns.org/search?" + query + 
           "&num_page=500&format=csv")
    
    return url

def append_sector(file, folder, sector):
    """ 
    For a given csv file, add a column with the sector integer to all rows. 

    :param file: csv file
    :param folder: directory of output
    :param sector: integer of sector to append in column
    """
    df = pd.read_csv(file)
    sectorcolumn = np.zeros(len(df)) + int(sector)
    df["Sector"] = sectorcolumn
    df.to_csv(f"{folder}sector{sector}_tns.csv")
    return

def compile_csvs(folder, csvstring, savefilename = None):
    """
    Compile all CSV's in a folder into one megacsv

    :param folder: folder with csvs and save folder
    :param csvstring: suffix of all relevant csv files (probably 'tesscut.csv')
    :param savefilename: if you want to save it, put the file name here
    
    :return: pandas data frame containing all csvs read in
    """
    all_info = pd.DataFrame()
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith((csvstring)):
                filepath = root  + f
                p = pd.read_csv(filepath)
                if all_info.empty:
                    all_info = p #first load
                else:
                    all_info = pd.concat((all_info, p)) #other ones
    del all_info["Unnamed: 0"] #remove extra column
    if savefilename is not None:
        print(f"New file: {folder}{savefilename}.csv")
        all_info.to_csv(f"{folder}{savefilename}.csv", index=False)
    return all_info.reset_index()

def prep_WTV_file(csv_file, outputfile):
    """
    This converts all RA and DEC in an input pandas-readable CSV file
    into degrees (req. for WTV) and saves it into its own new csv file.

    :param tnsfile: file (path) from tns
    :param outputfile: file (path) to save into
    """
    #converting to decimal degrees
    from astropy.coordinates import Angle
    df = pd.read_csv(csv_file)

    for n in range(len(df)):
        df['RA'][n] = Angle(df['RA'][n], u.degree).degree
        df['DEC'][n] = Angle(df['DEC'][n], u.degree).degree
    
    new = df[['RA', 'DEC']].copy()
    new.to_csv(outputfile, index = False)
    return

def process_WTV_results(TNS_file, WTV_file, output_file, skiprows=76):
    """
    Go through the WTV output file and compare with the TNS sector to ID those
    that WTV thinks TESS should have found

    :param TNS_file: tns original file with sectors attached per target
    :param WTV_file: file from wtv run
    :param output_file: where to save results
    :param skiprows: how many rows at the top of the wtv output that are just junk (will incr. with time)
    """
    all_tns = pd.read_csv(TNS_file)
    all_wtv = pd.read_csv(WTV_file, skiprows=skiprows)
    all_wtv.dropna(inplace=True)   #remove nan row if exists
    all_wtv.reset_index(inplace=True)

    just_sectors = all_tns["Sector"]
    WTV_confirmed =  pd.DataFrame(columns = all_tns.columns)
    for n in range(len(all_wtv)-1):
        correct_sector = all_tns["Sector"][n]
        columnname = "S" + str(int(correct_sector))
        if all_wtv[columnname][n] != 0.0: 
            WTV_confirmed = WTV_confirmed.append(all_tns.iloc[[n]])
            
    WTV_confirmed.reset_index(inplace = True, drop=True)    
    WTV_confirmed.to_csv(output_file, index=False)
    return WTV_confirmed

def run_all_tesscut_matches(tns_folder, output_folder, start=0, verbose=False):
    """ 
    For a folder full of TNS files with sectors attached, run tesscut on each

    :param tns_folder: folder holding sectorwise tns files
    :param output_folder: folder to save tesscut outputs
    :param start: which sector to start at 
    :param verbose: boolean, prints filenames

    """
    for i in range(start, 55):
        file = f"{tns_folder}sector{i}_tns.csv"
        if verbose:
            print(file)
        if os.path.exists(file):
            fileout = f"{output_folder}sector{i}_tesscut.csv"
            sectorfile_tesscut_match(file, fileout, i)
        else:
            if verbose:
                print(f"{file} does not exist")
    return

 
def sectorfile_tesscut_match(csv_file, savefile, sector):
    """
    For a given CSV file (TNS format) with sectors attached to each target,
    run tesscut on each set of coordinates to see if it was observed in that sector
    This would ideally be extended to previous sector as well. 

    :param csv_file: file of targets
    :param savefile: file of output targets that were observed
    :param sector: int sector number
    """
    df = pd.read_csv(csv_file)
    del df["Unnamed: 0"]
    goodIndices = []
    #for each, get ra/dec, run thru tesscut
    for n in range(len(df)):
        cutout_coord = SkyCoord(df["RA"].iloc[n], df["DEC"].iloc[n], 
                                unit=(u.hourangle, u.deg))
        sector_table = Tesscut.get_sectors(coordinates=cutout_coord)
        #traverse the table
        for i in range(len(sector_table)):
            if sector_table[i][1] == sector:
                goodIndices.append(n)
        
        time.sleep(3)
    print(goodIndices)
    #clip df, save
    df = df.iloc[goodIndices]
    df.to_csv(savefile)
    return

def get_not_in_common_entries(file1, file2):
    """
    For 2 TNS formatted csvs, find the entries that only exist in one list
    the left file will be the file1, file2 is right (for merge)
    """

    file1 = pd.read_csv(file1)
    file2 = pd.read_csv(file2)
    
    df_all_left = file1.merge(file2.drop_duplicates(), on=["Name", "Name"], how="left", indicator=True)
    df_ind_left = df_all_left[df_all_left["_merge"] != "both"]
    df_all_right = file1.merge(file2.drop_duplicates(), on=["Name", "Name"], how="right", indicator=True)
    df_ind_right = df_all_right[df_all_right["_merge"] != "both"]
    df_ind = pd.concat((df_ind_left, df_ind_right))
    return df_ind

def number_Ias(file):
    """ 
    Returns how many type Ia sn are in a given csv file. 
    """
    f = pd.read_csv(file)
    return len(f[f["Obj. Type"] == "SN Ia"])




