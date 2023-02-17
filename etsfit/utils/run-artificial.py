#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:58:03 2023

@author: lindseygordon

run artificial data gen + fitting
"""

save_dir = "./research/urop/fake_data/"
# fit flow (single powers)
lc = artificial_lc(save_dir)
TNSFile = "/Users/lindseygordon/research/urop/august2022crossmatch/all-tesscut-matches.csv"  
tr1 = "/Users/lindseygordon/research/urop/tessreduce_lc/2018fvi0233/2018fvi0233-tessreduce"        
# come up with the 100 LC params
# produce + fit the 100 w/out the noise model
lc.generate_singlerise_fakes(100, tr1, TNSFile, 
                             tess_background=False, load_params=False)
lc.fit_fakes_1(n1=5_000, n2=50_000)
# produce + fit 100 with the noise model AND the same params (x3) 

lc.generate_singlerise_fakes(100, tr1, TNSFile, 
                             tess_background=True, load_params=True)
lc.fit_fakes_1(n1=5_000, n2=50_000)