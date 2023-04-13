#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:05:58 2023

@author: lindseygordon


Generate parameter sets + models 

"""

import numpy as np
import matplotlib.pyplot as plt
from etsfit import etsMAIN
import etsfit.utils.utilities as ut
from pylab import rcParams
import os
import pandas as pd
rcParams['figure.figsize'] = 8,3
rcParams['font.family'] = 'serif'


class signal_generator(object):
    """ 
    Class of functions to make SNe signal parameters + actual curves
    """
    
    def __init__(self, save_dir, n, rise_=1, x = None):
        """ 
        Set things up
        Params: 
            save_dir (str) path
            n (int) # LC to make
            rise_ (int) 1 or 2 
        """
        self.save_dir = save_dir
        self.n = n
        self.rise_ = rise_
        
        if self.rise_ not in (1,2):
            print('NOT A VALID RISE_')
            return 
        
        if self.rise_ == 1:
            self.ndim = 4
            self.labels = [r'$t_0$', 'A', r'$\beta$', 'B']
        elif self.rise_ == 2:
            self.ndim = 7
            self.labels = [r'$t_0$', r'$t_1$', r'$A_1$', r'$A_2$', 
                           r'$\beta_1$', r'$\beta_2$', 'B']
        
        
        if not os.path.exists(self.save_dir):
            print("Making new save folder")
            os.mkdir(self.save_dir)
            
        if x is None: #if none, need to generate x
            self.__gen_x()
        else:
            self.x = x
            self.l = len(x)
        return
    
    def __gen_x(self):
        """ 
        Make x-axis with orbit gap
        """
        self.l = 1500
        # x axis with a fake orbit gap of size 1/10th array
        tenth = int(self.l / 10)
        start_ = int(self.l / 2) - int(tenth/2)
        end_ = start_ + tenth
        self.x = np.linspace(0, 28, (self.l+tenth))
        
        self.orbit_gap = [self.x[start_], self.x[end_]]
        
        mask = np.ones(self.l+tenth)
        mask[start_:end_] = 0
        mask = np.nonzero(mask) # which ones you are keeping
        self.x = self.x[mask]
        return
    
    def gen_params(self):
        """ 
        Produce the true parameter set + save
        """
        
        print("Generating parameters")
        self.true_param_file = "{s}true-params.csv".format(s=self.save_dir) 
        if os.path.exists(self.true_param_file):
            print('params already exist, loading them in')
            h = pd.read_csv(self.true_param_file)
            self.true_params = h.to_numpy()[:,1:-1]
            self.dtimes = h.to_numpy()[:,-1]
            self.disctimes = {}
            labels = list(range(self.n))
            for i in range(self.n):
                self.disctimes[labels[i]] = self.dtimes[i]
       
        else: 
            print('making params from scatch')
            from scipy.stats import truncnorm
            myclip_a, myclip_b = 0.5, 4
            loc, scale = 2, 1
            a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
        
            if self.rise_== 1:
                self.true_params = np.zeros((self.n, self.dim)) ##t0 A beta B
                self.true_params[:,0] = np.random.uniform(5, 20, self.n) #t0
                self.true_params[:,1] = np.random.uniform(0.001, 1.5, self.n) #A1
                # beta is pulled from a unif distro on the arctans
                self.true_params[:,2] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n)
                self.true_params[:,3] = np.random.uniform(1, 20, self.n) *  np.random.choice((-1,1), self.n)
                #B can't get to close to 0 or summary stats look like trash
                
                self.disctimes = {}
                labels = list(range(self.n))
                self.dtimes = self.true_params[:,0] + np.random.uniform(0.5, 6, self.n) + 2457000
                for i in range(self.n):
                    self.disctimes[labels[i]] = self.dtimes[i]
                
            elif self.rise_== 2:
                self.true_params = np.zeros((self.n, self.dim)) ##t0 A beta B
                self.true_params[:,0] = np.random.uniform(1, 20, self.n) #t0
                self.true_params[:,1] = np.random.uniform(self.true_params[:,0], 25, self.n) #t1 defined by t0
                self.true_params[:,2] = np.random.uniform(0.001, 1.5, self.n) #A1
                self.true_params[:,3] = np.random.uniform(0.001, 1.5, self.n) #A2
                self.true_params[:,6] = np.random.uniform(1, 20, self.n) *  np.random.choice((-1,1), self.n) #B
                # beta is pulled from a unif distro on the arctans
                self.true_params[:,4] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n)
                self.true_params[:,5] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n)
                
                self.disctimes = {}
                labels = list(range(self.n))
                self.dtimes = self.true_params[:,0] + np.random.uniform(0.5, 6, self.n) + 2457000
                for i in range(self.n):
                    self.disctimes[labels[i]] = self.dtimes[i]
    
                
            else: 
                print("not a valid rise number (1 or 2)")
            self.__save_true_params()
            
        return
    
    def __save_true_params(self):
        """ 
        Put the real values into a file for access later
        """
        if self.rise_ == 1:
            di = {'t0':self.true_params[:,0], 
                  'A':self.true_params[:,1],
                  'beta':self.true_params[:,2],
                  'B':self.true_params[:,3], 
                  'disc':self.dtimes}
        elif self.rise_ == 2:
            di = {'t0':self.true_params[:,0], 
                  't1':self.true_params[:,1],
                  'A1':self.true_params[:,2],
                  'A2':self.true_params[:,3],
                  'beta1':self.true_params[:,4],
                  'beta2':self.true_params[:,5],
                  'B':self.true_params[:,6], 
                  'disc':self.dtimes}
        
        df = pd.DataFrame(di)
        df.to_csv(self.true_param_file)
        return
    
    def gen_lc(self):
        """ 
        Make LC using given bg + subfolder + save LC into file
        NO backgrounds are applied here
        """

        self.fake_flux = np.zeros((self.n, self.l)) # n l-length arrays
        self.fake_error = np.zeros((self.n, self.l))
        
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2

        for i in range(self.n):
            if self.rise_ == 1:
                t0, A, beta, B = self.true_params[i]
                t_ = self.x - t0
                self.fake_flux[i] = (np.heaviside((t_), 1) * A *np.nan_to_num((t_**beta))) + 1 + B
            
            elif self.rise_ == 2:
                t0, t1, A1, A2, beta1, beta2, B = self.true_params[i]
                self.fake_flux[i] = np.piecewise(self.x, [(t0 <= self.x)*(self.x < t1), t1 <= self.x], 
                                                 [func1, func2],
                                                 t0, t1, A1, A2, beta1, beta2) + 1 + B
            
            
            #attach noise: 
            self.fake_flux[i] = self.fake_flux[i] + self.noise_model
            # set error = expectation value of noise model
            self.fake_error[i] += np.mean(self.noise_model)
            
                    
        self.x += 2457000 #reset time axis because it will get subtracted
        return
    