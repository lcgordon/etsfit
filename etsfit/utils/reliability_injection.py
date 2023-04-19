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


class reliability_injection(object): 
    
    def __init__(self, save_dir, index_of_tess, n_signals_to_inject, rise): 
        """ 
        Initialize
        
        """
        if rise not in (1,2):
            print('NOT A VALID rise')
            return 
        
        self.n_signals_to_inject = n_signals_to_inject
        self.rise = rise
        self.n_tess = 1
        self.tess_idx = index_of_tess
        
        self.save_dir = f"{save_dir}tessidx-{index_of_tess}-ninj-{n_signals_to_inject}/"
        if not os.path.exists(self.save_dir):
            print("Making new save folder")
            os.mkdir(self.save_dir)
        
        if self.rise == 1:
            self.ndim = 4
            self.labels = [r'$t_0$', 'A', r'$\beta$', 'B']
        elif self.rise == 2:
            self.ndim = 7
            self.labels = [r'$t_0$', r'$t_1$', r'$A_1$', r'$A_2$', 
                           r'$\beta_1$', r'$\beta_2$', 'B']
        
        self.true_param_file = "{s}true-params.csv".format(s=self.save_dir)
        
        self.TOTAL = self.n_signals_to_inject + 1 #how many total (including noninjection)
        self.targetlabel = f"tessidx_{self.tess_idx}_ninj_{self.n_signals_to_inject}"
        return
    
    def gen_params(self):
        """ 
        Produce the true parameter set + save
        """
        
        print("Generating parameters:", end="  ")
        if os.path.exists(self.true_param_file):
            print('Params already exist, loading them in: ', end="  ")
            h = pd.read_csv(self.true_param_file)
            self.true_params = h.to_numpy()[:,1:-1]
            self.dtimes = h.to_numpy()[:,-1]
            self.disctimes = {}
            labels = list(range(self.TOTAL))
            for i in range(self.TOTAL):
                self.disctimes[labels[i]] = self.dtimes[i]
            print("done")
        else: 
            print('Making params from scatch:' , end="  ")
            from scipy.stats import truncnorm
            myclip_a, myclip_b = 0.5, 4
            loc, scale = 2, 1
            a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
            self.true_params = np.zeros((self.TOTAL, self.ndim)) ##t0 A beta B
        
            if self.rise== 1:
                self.true_params[:,0] = np.random.uniform(0, 20, self.TOTAL) #t0
                self.true_params[:,1] = np.random.uniform(0.001, 1, self.TOTAL) #A1
                # beta is pulled from a unif distro on the arctans
                self.true_params[:,2] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.TOTAL)
                self.true_params[:,3] = np.random.uniform(0, 5, self.TOTAL) *  np.random.choice((-1,1), self.TOTAL)
                #B can't get to close to 0 or summary stats look like trash
                
                self.disctimes = {}
                labels = list(range(self.TOTAL))
                self.dtimes = self.true_params[:,0] + np.random.uniform(0.5, 6, self.TOTAL) + 2457000
                for i in range(self.TOTAL):
                    self.disctimes[labels[i]] = self.dtimes[i]
                
            elif self.rise== 2:
                self.true_params = np.zeros((self.TOTAL, self.ndim)) ##t0 A beta B
                self.true_params[:,0] = np.random.uniform(0, 20, self.TOTAL) #t0
                self.true_params[:,1] = np.random.uniform(self.true_params[:,0], 25, self.TOTAL) #t1 defined by t0
                self.true_params[:,2] = np.random.uniform(0.001, 1, self.TOTAL) #A1
                self.true_params[:,3] = np.random.uniform(0.001, 1, self.TOTAL) #A2
                self.true_params[:,6] = np.random.uniform(1, 20, self.TOTAL) *  np.random.choice((-1,1), self.TOTAL) #B
                # beta is pulled from a unif distro on the arctans
                self.true_params[:,4] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.TOTAL)
                self.true_params[:,5] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.TOTAL)
                
                self.disctimes = {}
                labels = list(range(self.TOTAL))
                self.dtimes = self.true_params[:,0] + np.random.uniform(0.5, 6, self.TOTAL) + 2457000
                for i in range(self.TOTAL):
                    self.disctimes[labels[i]] = self.dtimes[i]
                    
            # reset row 0: 
            self.true_params[0] = np.zeros(self.ndim)
            self.dtimes[0] = 2457015
            self.disctimes[0] = 2457015
                    
            print("done")
            self.__save_true_params()
            
        return
    
    def __save_true_params(self):
        """ 
        Put the real values into a file for access later
        """
        if self.rise == 1:
            di = {'t0':self.true_params[:,0], 
                  'A':self.true_params[:,1],
                  'beta':self.true_params[:,2],
                  'B':self.true_params[:,3], 
                  'disc':self.dtimes}
        elif self.rise == 2:
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
        print("Saved true injection params!")
        return
    
    def real_data_load(self, folder): 
        """ 
        Given a folder of csv files, load the data into an array
        """
        print("Loading tess data!")
        i = 0
        for root, dirs, files in os.walk(folder):
            for name in files:
                if i != self.tess_idx: 
                    i+= 1
                    continue
                fname = root + "/" + name
                print(fname)
                data = pd.read_csv(fname)
                #print(len(data))
                 
                #make x, make full data array, full error array + fill
                self.x = data['time'].to_numpy()
                self.l = len(self.x)
                print(self.l)
                    
                self.true_flux = data['flux'].to_numpy() 
                self.true_flux -= np.nanmean(self.true_flux)
                self.true_error = data['error'].to_numpy()
                self.true_labels = name.split("_")[1].split(".")[0]
                
                #renorm
                dr = self.true_flux
                e = self.true_error
                drn = (dr - np.nanmin(dr))/ (np.nanmax(dr) - np.nanmin(dr))
                en = (e - np.nanmin(dr))/ (np.nanmax(dr) - np.nanmin(dr))
                self.true_flux = drn
                self.true_error = en
                
                i += 1
                
        return 
    
    def plot_all_real(self): 
        """ 
        Plot all the real datasets in one big plot. 
        """
        # n_tess is how many to plot
        print("Plotting real tess data!")
        fig, axs = plt.subplots(1, figsize=(8, 3))
        
        axs.errorbar(self.x, self.true_flux, self.true_error, 
                     mfc='black', mec='black', ms=1, fmt=".", ecolor='black')
        axs.set_title(f"TIC {int(self.true_labels)}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}real_data.png")
        return
    
    def gen_injecting_signals(self):
        """ 
        Make signals to inject into the real data.  
        """
        print("Generating injection signals: ", end="")
        self.fake_signals = np.zeros((self.TOTAL, self.l)) # n l-length arrays
        self.x -= self.x[0] #time axis gets 0'd out
        
        self.fake_signals[0] = np.zeros(self.l) #blank 0th 
        
        for i in range(1, self.TOTAL):
            if self.rise == 1:
                self.fake_signals[i] = self.__gen_singlerise(self.true_params[i])
            
            elif self.rise == 2:
                self.fake_signals[i] = self.__gen_doublerise(self.true_params[i])
                
        self.x += 2457000 #reset time axis because it will get subtracted
        print("done")
        return
    
    def __gen_singlerise(self, params):
        t0, A, beta, B = params
        t_ = self.x - t0
        model = (np.heaviside((t_), 1) * 1 *np.nan_to_num((t_**beta))) + 1 + B
        dr = model
        drn = ((dr - np.nanmin(dr))/ (np.nanmax(dr) - np.nanmin(dr))) * A  # RESCALE TO A
        model = drn
        return model
    
    def __gen_doublerise(self, params):
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        
        t0, t1, A1, A2, beta1, beta2, B = params
        model = np.piecewise(self.x, [(t0 <= self.x)*(self.x < t1), t1 <= self.x], 
                                         [func1, func2],
                                         t0, t1, A1, A2, beta1, beta2) + 1 + B
        dr = model
        drn = (dr - np.nanmin(dr))/ (np.nanmax(dr) - np.nanmin(dr)) #also norm to 1
        model = drn
        return model
        
    
    def inject_real_data(self):
        """ 
        Generate randomly injected sources
        """
        print("Injecting signals: ") #, end=""
        
        #injected fluxes are going to be total datasets, length l
        self.injected_flux = np.zeros((self.TOTAL, self.l)) 
        self.injected_error = np.zeros((self.TOTAL, self.l)) 
        
        #injected indexes: for each in n_tess, number_to_inject indexes
        for j in range(self.TOTAL): #for each real 
            #0th row makes this no problem
            self.injected_flux[j] = self.true_flux + self.fake_signals[j]
            self.injected_error[j] = self.true_error
        
        print("done")
        return
    
    def plot_all_injections(self):
        """ 
        plot the fake data  + retrieved models
        
        - every background tess gets their own plot 
        """
        
        cols = 2
        rows = int(np.ceil(self.TOTAL/cols))
        print(f'cols: {cols}, rows: {rows}')
        
        #reset axis: 
        if self.x[0] != 0:
            x_ = self.x - 2457000
        else:
            x_ = self.x
        
        #k = 0 #array indexer
        c = ['black', 'darkgreen']
        fig, ax = plt.subplots(rows, cols, figsize=(8*cols, 3*rows))
        for j in range(self.TOTAL):
            axy = ax[int(j/2),j%2]
            #nanmask and downsample: 
            mask = np.isnan(self.injected_flux[j])
            use_x = x_[~mask][0:np.sum(~mask):10]
            use_y = self.injected_flux[j][~mask][0:np.sum(~mask):10]
            #use_e = self.injected_error[k][~mask][0:np.sum(~mask):10]
                
            axy.scatter(use_x, use_y, s=2, color=c[0 if j==0 else 1])
            
            if j != 0: 
                axy.axvline(self.dtimes[j]-2457000, color='blue', label='disc. time', linestyle='dashed')
                truemodel = self.fake_signals[j]
                axy.plot(x_, truemodel, lw=2, color='red', linestyle='dashed')
                axy.axvline(self.true_params[j][0], color='blue', label='t0', linestyle='dashed')
                
        plt.suptitle(f"Injection Plot: TIC {self.true_labels}")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/injection-plot.png")
        plt.show()
        plt.close()
                
        return
            
    def fit_fakes(self, start=0, n1=500, n2=5000):
        """ 
        Fit models to fake data starting at start index
        """
        
        #check to see if this has already been run: 
        final_file = f"{self.save_dir}/index{self.TOTAL-1}/"
        print(final_file)
        if os.path.exists(final_file):
            print("Fitting has already been run")
            return
        
        if self.rise == 1: 
            self.__fit_fakes_singlerise(start, n1, n2)
        elif self.rise == 2:
            self.__fit_fakes_doublerise(start, n1, n2)
        return 
        
        
    def __fit_fakes_singlerise(self, start=0, n1=500, n2=5000):  
        
        self.s_3 = np.zeros((self.TOTAL, self.ndim))
        
        for i in range(self.TOTAL):
            if i < start: continue
            print(f"index: {i}")
            dt = self.disctimes[i]
        
            self.trlc = etsMAIN(self.save_dir, 'nofile', plot=True)
            
            #nanmask and downsample!! 
            mask = np.isnan(self.injected_flux[i])
            use_x = self.x[~mask][0:np.sum(~mask):10]
            use_y = self.injected_flux[i][~mask][0:np.sum(~mask):10]
            use_e = self.injected_error[i][~mask][0:np.sum(~mask):10]
        
            self.trlc.load_single_lc(use_x, use_y, use_e, 
                                     dt, f"index{i}", "", "", "")

            self.trlc.pre_run_clean(fitType=1)
            self.trlc.test_plot()
            self.trlc.run_MCMC(n1, n2, quiet=True)
            self.s_3[i] = self.trlc.sampler.get_autocorr_time(tol=0)
            
        #save file:
        di = {'act_t':self.s_3[:,0],
              'act_a':self.s_3[:,1],
              'act_beta':self.s_3[:,2],
              'act_B':self.s_3[:,3]}
        df = pd.DataFrame(di)
        self.autocorr_file = f"{self.save_dir}allfluxes-{self.targetlabel}-{self.rise}-autocorr.csv"
        df.to_csv(self.autocorr_file)
            
        return
    
    # def __fit_fakes_doublerise(self, start=0, n1=500, n2=5000):  
    #     """ fit double powers """ 
        
                
    #         #save S3 file:
    #         di = {'act_t0':self.s_3[:,0],
    #               'act_t1':self.s_3[:,1],
    #               'act_A1':self.s_3[:,2],
    #               'act_A2':self.s_3[:,3],
    #               'act_beta1':self.s_3[:,4],
    #               'act_beta2':self.s_3[:,5],
    #               'act_B':self.s_3[:,6]}
    #         df = pd.DataFrame(di)
    #         self.autocorr_file = f"{self.save_dir}allfluxes-{self.targetlabel}-{self.rise}-autocorr.csv"
    #         df.to_csv(self.autocorr_file)
    #         return
    
    def retrieve_params(self, bg=0):
        """ 
        Load in the saved true params and the output params for comparison purposes
        
        """
        #load in true: 
        true_p = pd.read_csv(self.true_param_file)
        
        # load in calculated params
        import etsfit.utils.batch_analyze as ba
        params_all = {}
        converged_all = {}
        upper_all = {}
        lower_all = {}

        
        for root, dirs, files in os.walk(self.save_dir):
            for name in files:
                if (name.endswith("-output-params.txt")):
                    targ = name.split("-")[0]
                    if targ[0] != "i":
                        continue #oops hit a noise model
                    
                    filepath = root + "/" + name
                    if self.rise == 1: 
                        (params,  upper_e, 
                         lower_e,  converg) = ba.extract_singlepower_all(filepath)
                    elif self.rise == 2:
                        (params,  upper_e, 
                         lower_e,  converg) = ba.extract_doublepower_all(filepath)
                    
                    params_all[targ] = params
                    upper_all[targ] = upper_e
                    lower_all[targ] = lower_e
                    converged_all[targ] = converg
                    #print(converg)
        
        #print(params_all)
        # dicts into arrays: 
        p_ =  len(params_all)
        self.converged_all = converged_all
        
        self.output_params = np.zeros((p_, self.ndim))
        self.upper_error = np.zeros((p_, self.ndim))
        self.lower_error = np.zeros((p_, self.ndim))
        self.converged_retrieved = np.zeros(p_)
        for i in range(p_):
            st_ = 'index{}'.format(i)
            self.output_params[i] = params_all[st_]
            self.upper_error[i] = upper_all[st_]
            self.lower_error[i] = lower_all[st_]
            self.converged_retrieved[i] = converged_all[st_]
        
        self.converged_retrieved = self.converged_retrieved.astype(int)
        self.params_true = true_p.to_numpy()[:,1:self.ndim+1]
        
        print("Convergence Rate:", np.sum(self.converged_retrieved)/len(self.converged_retrieved))

        return 
    
    def plot_true_and_fit(self):
        """ 
        plot the true parameter stuff and the fit values 
        """
        cols = 2
        rows = int(np.ceil(self.TOTAL/cols))
        print(f'cols: {cols}, rows: {rows}')
        
        #reset axis: 
        if self.x[0] != 0:
            x_ = self.x - 2457000
        else:
            x_ = self.x
        
        #k = 0 #array indexer
        #c = ['black', 'darkgreen']
        fig, ax = plt.subplots(rows, cols, figsize=(8*cols, 3*rows))
        for j in range(self.TOTAL):
            axy = ax[int(j/2),j%2]
            #nanmask and downsample: 
            mask = np.isnan(self.injected_flux[j])
            use_x = x_[~mask][0:np.sum(~mask):10]
            use_y = self.injected_flux[j][~mask][0:np.sum(~mask):10]
            #use_e = self.injected_error[k][~mask][0:np.sum(~mask):10]
                
            axy.scatter(use_x, use_y, s=2, color='k')
            
            if j != 0: 
                axy.axvline(self.dtimes[j]-2457000, color='blue', label='disc. time', linestyle='dashed')
                truemodel = self.fake_signals[j]
                axy.plot(x_, truemodel, lw=2, color='red', linestyle='dashed')
                axy.axvline(self.true_params[j][0], color='blue', label='t0', linestyle='dashed')
             
            t0, A, beta, B = self.output_params[j]
            t_ = use_x - t0
            model = (np.heaviside((t_), 1) * A *np.nan_to_num((t_**beta))) + 1 + B
            axy.plot(use_x, model, color='cyan', label='model', lw=3)
            
                
        plt.suptitle(f"Model Plots: TIC {self.true_labels}")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/big-all-models-plot.png")
        plt.show()
        plt.close()
        
        return 
        
    
    def chi_sq(self):
        """ 
        calc chisquared values
        """
        self.chi_squared = np.zeros(self.TOTAL)
        x_ = self.x
        if self.x[0] != 0:
            x_ -= 2457000
        if self.rise == 1:
            for i in range(self.TOTAL): 
                mask = np.isnan(self.injected_flux[i])
                use_x = x_[~mask][0:np.sum(~mask):10]
                use_y = self.injected_flux[i][~mask][0:np.sum(~mask):10]
                t0, A, beta, B = self.output_params[i]
                t_ = use_x - t0
                model = (np.heaviside((t_), 1) * A *np.nan_to_num((t_**beta))) + 1 + B
                self.chi_squared[i] = np.nansum( ( model - use_y )**2 / use_y)
                
        # elif self.rise == 2: 
        #     for i in range(p): 
        #         model = self.__gen_doublerise(self.output_params[i])
        #         self.chi_squared[i] = np.nansum( ( model - self.injected_flux[i] )**2 / self.injected_flux[i])
        
        # s_1: 
        max_err = np.maximum(self.upper_error.round(3), self.lower_error.round(3))
        #self.s_1 = np.abs(self.output_params.round(3) / max_err)
        self.s_1 = np.abs(max_err / self.output_params.round(3) )
        
        print(f"chisq: {self.chi_squared.round(2)}, s_1: {self.s_1}")
        
        return
        
  
        
#%%
data_folder = "/Users/lindseygordon/research/urop/reliability_testing/chosen_data/"
save_dir = "/Users/lindseygordon/research/urop/reliability_testing/output/"
n_injections = 6
n_tess = 1

#init
ri = reliability_injection(save_dir, index_of_tess=0, n_signals_to_inject=6, rise=1)

# real tess data: 
ri.real_data_load(data_folder)
ri.plot_all_real()

# # fake signals:
ri.gen_params()
ri.gen_injecting_signals()

# signal injection
ri.inject_real_data()
ri.plot_all_injections()

# # # #signal fitting: 
ri.fit_fakes(start=6, n1=3_000, n2=30_000)
ri.retrieve_params()
ri.plot_true_and_fit()
# ri.s_stats()
ri.chi_sq()
#%%
    
    
    