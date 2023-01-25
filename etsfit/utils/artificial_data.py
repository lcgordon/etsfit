# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 12:39:41 2022

@author: conta

Generating artificial light curves for testing

Upated: 1/25/23 LG


"""
import numpy as np
import matplotlib.pyplot as plt
from etsfit import etsMAIN
from pylab import rcParams
rcParams['figure.figsize'] = 8,3



class artificial_lc(object):
    
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        
        
    def gen_fakes(self, n):
        print("making parameter vectors")
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        
        self.n = n
        #t0, t1, A0, A1, beta0, beta1, B
        self.params_all = np.zeros((self.n, 7))
        self.l = 10_000
        self.x = np.linspace(0, 28, self.l)
        self.flux_fake = np.ones((self.n, self.l))
        
        self.params_all[:,0] = np.random.uniform(1, 28, self.n) #t0
        self.params_all[:,2] = np.random.uniform(0.01, 5, self.n) #A1
        
        self.params_all[:,4] = np.random.uniform(0.5, 3, self.n) #beta1
        
        self.params_all[:,6] = np.random.uniform(-30, 30, self.n) #B
        
        for i in range(self.n):
            if not i%4: #do like a quarter of them
                self.params_all[i][1] = np.random.uniform(self.params_all[i][0], 28, 1)
                self.params_all[i][3] = np.random.uniform(0.01, 5, 1) #A2
                self.params_all[i][5] = np.random.uniform(0.5, 3, 1) #beta2
                t0, t1, A1, A2, beta1, beta2, B = self.params_all[i]
                self.flux_fake[i] = np.piecewise(self.x, [(t0 <= self.x)*(self.x < t1), t1 <= self.x], 
                                     [func1, func2], t0, t1, A1, A2, beta1, beta2) + 1 + B
                
            else:
                t0, t1, A1, A2, beta1, beta2, B = self.params_all[i]
                t_ = self.x - t0
                self.flux_fake[i] = (np.heaviside((t_), 1) * A1 *np.nan_to_num((t_**beta1))) + 1 + B
        return
    def plot_fake(self, index):
        
        plt.scatter(self.x, self.flux_fake[index], color='k', s=2)
        plt.axvline(self.params_all[index][0], color='r')
        plt.axvline(self.params_all[index][1], color='r')
        plt.xlabel("time")
        plt.ylabel('fake flux')
        plt.title("index:{}, 2 components:{}".format(index, not bool(index%4)))
        plt.show()
            
            
            
        
 
lc = artificial_lc(".")
lc.gen_fakes(10)
lc.plot_fake(1)
        
