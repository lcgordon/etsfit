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
        """ 
        
        Note to self: for paper purposes, expand out the limits on the 
        params to go past the priors
        
        """
        print("making parameter vectors")
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        
        self.n = n
        #t0, t1, A0, A1, beta0, beta1, B
        self.params_true = np.zeros((self.n, 7))
        self.l = 1_000
        self.x = np.linspace(0, 28, self.l)
        self.flux_fake = np.ones((self.n, (self.l-100)))
        self.error_fake = np.ones((self.n, (self.l-100)))
        
        self.params_true[:,0] = np.random.uniform(1, 28, self.n) #t0
        self.params_true[:,2] = np.random.uniform(0.01, 5, self.n) #A1
        
        self.params_true[:,4] = np.random.uniform(0.5, 3, self.n) #beta1
        
        self.params_true[:,6] = np.random.uniform(-30, 30, self.n) #B
        
        #remove the middles like it's an orbit gap
        #indexes 475 (13.31) to 575 (16.11)
        mask = np.ones(self.l)
        mask[475:575] = 0
        mask = np.nonzero(mask) # which ones you are keeping
        #mask x as well
        self.x = self.x[mask]
        
        for i in range(self.n):
            if not i%4: #do like a quarter of them
                self.params_true[i][1] = np.random.uniform(self.params_true[i][0], 28, 1)
                self.params_true[i][3] = np.random.uniform(0.01, 5, 1) #A2
                self.params_true[i][5] = np.random.uniform(0.5, 3, 1) #beta2
                t0, t1, A1, A2, beta1, beta2, B = self.params_true[i]
                self.flux_fake[i] = np.piecewise(self.x, [(t0 <= self.x)*(self.x < t1), t1 <= self.x], 
                                     [func1, func2], t0, t1, A1, A2, beta1, beta2) + 1 + B
                
            else:
                t0, t1, A1, A2, beta1, beta2, B = self.params_true[i]
                t_ = self.x - t0
                self.flux_fake[i] = (np.heaviside((t_), 1) * A1 *np.nan_to_num((t_**beta1))) + 1 + B
        
            #put some noise on those guys:
            #gaussian white noise + uniform white noise
            #uniform noise: 5% max
            unif = np.random.uniform(0,0.05*max(self.flux_fake[i]), self.l-100)
            # gaussian noise: gaussian * 5% of max
            gauss = np.random.normal(loc=0,
                                     scale = 1, size=self.l-100) * 0.05*max(self.flux_fake[i])
            #error? 
            err = unif + gauss
            expectation_val = np.mean(err)
            self.flux_fake[i] = self.flux_fake[i] + err
            self.error_fake[i] = expectation_val
            
            #end for

        
        #we also assign an arbitrary discovery time as being 0.5-6 days post-t0. 
        #dict: 
        self.disctimes = {}
        #labels
        labels = list(range(self.n))
        dtimes = self.params_true[:,0] + np.random.uniform(0.5, 6, self.n)
        for i in range(self.n):
            self.disctimes[labels[i]] = dtimes[i]
        
        
        return
    
    
    def plot_fake(self, index):
        
        plt.scatter(self.x, self.flux_fake[index], color='k', s=0.5)
        plt.errorbar(self.x, self.flux_fake[index], 
                     self.error_fake[index], color='k', markersize=0.5)
        plt.axvline(self.params_true[index][0], color='r')
        plt.axvline(self.params_true[index][1], color='r')
        plt.axvline(self.disctimes[index], color='g')
        plt.xlabel("time")
        plt.ylabel('fake flux')
        plt.title("index:{}, 2 components:{}".format(index, not bool(index%4)))
        plt.show()
        return

    def fit_fakes_1(self, start=0, n1=500, n2=5000):  
        
        self.output_p_1 = np.zeros((self.n, 4)) #always going to be t0, a , beta, b
        self.output_u_1 = np.zeros((self.n, 4)) #upper error
        self.output_l_1 = np.zeros((self.n, 4)) #lower error
        self.bic_1 = np.zeros((self.n, 1))
        for i in range(start, self.n):
            dt = self.disctimes[i]
            trlc = etsMAIN(self.save_dir, 'nofile')
        
            trlc.load_single_lc(self.x, self.flux_fake[i], self.error_fake[i], 
                                dt, 
                                "index{}".format(i), "00", "0", "0")
        
        
        
            trlc.pre_run_clean(fitType=1)
            #trlc.test_plot()
            trlc.run_MCMC(n1, n2, quiet=True)
            
            self.output_p_1[i] = trlc.best_mcmc[0]
            self.output_u_1[i] = trlc.upper_error[0]
            self.output_l_1[i] = trlc.lower_error[0]
            print(trlc.BIC)
            self.bic_1[0] = trlc.BIC
        return
    
    def fit_fakes_3(self, start=0, n1=500, n2=5000):  
        
        self.output_p_3 = np.zeros((self.n, 7)) #always going to be 7 params
        self.output_u_3 = np.zeros((self.n, 7)) #upper error
        self.output_l_3 = np.zeros((self.n, 7)) #lower error
        self.bic_3 = np.zeros((self.n, 1))
        for i in range(start, self.n):
            dt = self.disctimes[i]
            trlc = etsMAIN(self.save_dir, 'nofile')
        
            trlc.load_single_lc(self.x, self.flux_fake[i], self.error_fake[i], 
                                dt, 
                                "index{}".format(i), "00", "0", "0")
        
        
        
            trlc.pre_run_clean(fitType=3)
            #trlc.test_plot()
            trlc.run_MCMC(n1, n2, quiet=True)
            
            self.output_p_3[i] = trlc.best_mcmc[0]
            self.output_u_3[i] = trlc.upper_error[0]
            self.output_l_3[i] = trlc.lower_error[0]
            self.bic_3[0] = trlc.BIC
        return
        
   
        
            
        
 
lc = artificial_lc("./research/urop/fake_data/")
lc.gen_fakes(10)
# lc.plot_fake(1)
lc.fit_fakes_1()
lc.fit_fakes_3()
        
