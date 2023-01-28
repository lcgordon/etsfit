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
import etsfit.utils.utilities as ut
from pylab import rcParams
rcParams['figure.figsize'] = 8,3



class artificial_lc(object):
    
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        
        
    def gen_fakes(self, n, tessreducefile, TNSFile):
        """ 
        
        Note to self: for paper purposes, expand out the limits on the 
        params to go past the priors
        n = # to make
        tessreducefile = Ia whose noise model ur gonna generate
        TNSFile = u know
        
        """
        print("making parameter vectors")
        
        self.n = n
        #t0 A beta B
        self.params_true = np.zeros((self.n, 4)) #7
        # make the noise model
        self.__tess_noise(tessreducefile, TNSFile)
        
        # x axis with a fake orbit gap
        self.x = np.linspace(0, 28, self.l+100)
        #indexes 475 (13.31) to 575 (16.11)
        mask = np.ones(self.l)
        mask[475:575] = 0
        mask = np.nonzero(mask) # which ones you are keeping
        self.x = self.x[mask]
        
        self.flux_fake = np.ones((self.n, (self.l))) # n l-length arrays
        self.error_fake = np.ones((self.n, (self.l)))
        
        self.params_true[:,0] = np.random.uniform(1, 28, self.n) #t0
        self.params_true[:,1] = np.random.uniform(0.001, 5, self.n) #A1
        self.params_true[:,3] = np.random.uniform(-20, 20, self.n) #B
        # beta is pulled from a unif distro on the arctans
        self.params_true[:,2] = np.tan(np.random.uniform(np.arctan(0.5), np.arctan(6), self.n))
        #self.params_true[:,2] = np.random.uniform(0.5, 3, self.n) #beta1
        
        
        
        
        for i in range(self.n):
            t0, A, beta, B = self.params_true[i]
            t_ = self.x - t0
            self.flux_fake[i] = (np.heaviside((t_), 1) * A *np.nan_to_num((t_**beta))) + 1 + B
            
            #attach noise: 
            self.flux_fake[i] = self.flux_fake[i] + self.noise_model
            # set error = expectation value of noise model
            self.error_fake[i] += np.mean(self.noise_model)
        
            # #put some noise on those guys:
            # #gaussian white noise + uniform white noise
            # #uniform noise: 5% max
            # unif = np.random.uniform(0,0.05*max(self.flux_fake[i]), self.l-100)
            # # gaussian noise: gaussian * 5% of max
            # gauss = np.random.normal(loc=0,
            #                          scale = 1, size=self.l-100) * 0.05*max(self.flux_fake[i])
            # #error? 
            # err = unif + gauss
            # expectation_val = np.mean(err)
            
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
    
    def __tess_noise(self, tessreducefile, TNSFile):
        """ 
        Generate a noise model to use for each target that you have
        """
        print("making noise model from data")
        #load tess data
        (time, flux, error, targetlabel, 
             sector, camera, ccd) = ut.tr_load_lc(tessreducefile)
        discoverytime = ut.get_disctime(TNSFile, targetlabel)
        #run it once with a type 1
        self.trlc = etsMAIN(self.save_dir, TNSFile)
        
        self.trlc.load_single_lc(time, flux, error, discoverytime, 
                           targetlabel, sector, camera, ccd)

        winfilter = self.trlc.window_rms_filt(plot=False)
        self.trlc.pre_run_clean(1, flux_mask=winfilter)
        #trlc.test_plot()
        self.trlc.run_MCMC(5000, 25000, quiet=True)
        #cut to just data prior to the t0 
        print("t0 is: ", self.trlc.best_mcmc[0][0])
        self.t_lim = np.nonzero(np.where(self.trlc.time <= self.trlc.best_mcmc[0][0], 1, 0))

        self.cut_time = self.trlc.time[self.t_lim]
        self.cut_flux = self.trlc.flux[self.t_lim]
        #plt.scatter(self.cut_time, self.cut_flux)
        #relocate to mean=0
        m = np.mean(self.cut_flux)
        self.cut_flux -= m
        #draw samples from that to fill in a full fake light curve
        self.l = len(time)
        self.noise_model = np.random.choice(self.cut_flux, self.l)
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
        self.isright_1 = np.zeros((self.n, 4))
        
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
            self.bic_1[i] = trlc.BIC
            
            #comp with true
            for j in range(4):
                low = self.output_p_1[i][j]-self.output_l_1[i][j]
                high = self.output_p_1[i][j]+self.output_u_1[i][j]
                tru = self.params_true[i][j*2]
                if (tru <= high) and (tru >= low) :
                    self.isright_1[i][j] = 1
        return
    
    def fit_fakes_3(self, start=0, n1=500, n2=5000):  
        
        self.output_p_3 = np.zeros((self.n, 7)) #always going to be 7 params
        self.output_u_3 = np.zeros((self.n, 7)) #upper error
        self.output_l_3 = np.zeros((self.n, 7)) #lower error
        self.bic_3 = np.zeros((self.n, 1))
        self.isright_3 = np.zeros((self.n, 7))
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
            self.bic_3[i] = trlc.BIC
            
            #comp with true
            for j in range(4):
                low = self.output_p_3[i][j]-self.output_l_3[i][j]
                high = self.output_p_3[i][j]+self.output_u_3[i][j]
                tru = self.params_true[i][j]
                if (tru <= high) and (tru >= low) :
                    self.isright_3[i][j] = 1
        return
    
    

        
        
    
        
   
        
TNSFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"           
tessreducefile = "/Users/lindseygordon/research/urop/tessreduce_lc/2020tld2921/2020tld2921-tessreduce"        
 
lc = artificial_lc("./research/urop/fake_data/")
lc.gen_fakes(10, tessreducefile, TNSFile)
# lc.plot_fake(1)
#lc.fit_fakes_1()
#lc.fit_fakes_3()
        
