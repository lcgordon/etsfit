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
    """ 
    Class of functions to make SNe signal parameters + actual curves
    """
    
    def __init__(self, save_dir, n_injections, n_tess, rise, fraction=0.6):
        """ 
        Set things up
        Params: 
            save_dir (str) path
            n (int) # signals to make 
            rise (int) 1 or 2 
        """
        self.n_injections = n_injections
        self.rise = rise
        self.n_tess = n_tess
        self.save_dir = f"{save_dir}rise-{rise}-ninj-{self.n_injections}-ntess-{self.n_tess}/"
        
        if not os.path.exists(self.save_dir):
            print("Making new save folder")
            os.mkdir(self.save_dir)
        
        if self.rise not in (1,2):
            print('NOT A VALID rise')
            return 
        
        if self.rise == 1:
            self.ndim = 4
            self.labels = [r'$t_0$', 'A', r'$\beta$', 'B']
        elif self.rise == 2:
            self.ndim = 7
            self.labels = [r'$t_0$', r'$t_1$', r'$A_1$', r'$A_2$', 
                           r'$\beta_1$', r'$\beta_2$', 'B']
        
        self.injected_index_file = f"{self.save_dir}injection_index_list.csv"
        self.true_param_file = "{s}true-params.csv".format(s=self.save_dir)
        
        self.fraction_to_inject = int(self.n_injections * fraction)
        self.n_LC = self.fraction_to_inject + 1 #how many total (including noninjection)
        self.targetlabel = f"ntess_{self.n_tess}_ninjections_{self.fraction_to_inject}"
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
            labels = list(range(self.n_injections))
            for i in range(self.n_injections):
                self.disctimes[labels[i]] = self.dtimes[i]
            print("done")
        else: 
            print('Making params from scatch:' , end="  ")
            from scipy.stats import truncnorm
            myclip_a, myclip_b = 0.5, 4
            loc, scale = 2, 1
            a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
        
            if self.rise== 1:
                self.true_params = np.zeros((self.n_injections, self.ndim)) ##t0 A beta B
                self.true_params[:,0] = np.random.uniform(0, 20, self.n_injections) #t0
                self.true_params[:,1] = np.random.uniform(0.001, 1, self.n_injections) #A1
                # beta is pulled from a unif distro on the arctans
                self.true_params[:,2] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n_injections)
                self.true_params[:,3] = np.random.uniform(1, 20, self.n_injections) *  np.random.choice((-1,1), self.n_injections)
                #B can't get to close to 0 or summary stats look like trash
                
                self.disctimes = {}
                labels = list(range(self.n_injections))
                self.dtimes = self.true_params[:,0] + np.random.uniform(0.5, 6, self.n_injections) + 2457000
                for i in range(self.n_injections):
                    self.disctimes[labels[i]] = self.dtimes[i]
                
            elif self.rise== 2:
                self.true_params = np.zeros((self.n_injections, self.ndim)) ##t0 A beta B
                self.true_params[:,0] = np.random.uniform(0, 20, self.n_injections) #t0
                self.true_params[:,1] = np.random.uniform(self.true_params[:,0], 25, self.n_injections) #t1 defined by t0
                self.true_params[:,2] = np.random.uniform(0.001, 1, self.n_injections) #A1
                self.true_params[:,3] = np.random.uniform(0.001, 1, self.n_injections) #A2
                self.true_params[:,6] = np.random.uniform(1, 20, self.n_injections) *  np.random.choice((-1,1), self.n_injections) #B
                # beta is pulled from a unif distro on the arctans
                self.true_params[:,4] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n_injections)
                self.true_params[:,5] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n_injections)
                
                self.disctimes = {}
                labels = list(range(self.n_injections))
                self.dtimes = self.true_params[:,0] + np.random.uniform(0.5, 6, self.n_injections) + 2457000
                for i in range(self.n_injections):
                    self.disctimes[labels[i]] = self.dtimes[i]
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
                if i >= self.n_tess: 
                    continue
                fname = root + "/" + name
                print(fname)
                data = pd.read_csv(fname)
                #print(len(data))
                if i==0: 
                    #make x, make full data array, full error array + fill
                    self.x = data['time'].to_numpy()
                    self.l = len(self.x)
                    #print(self.l)
                    self.true_flux = np.zeros((self.n_tess, self.l))
                    self.true_error = np.zeros((self.n_tess, self.l))
                    self.true_labels = np.zeros(self.n_tess)
                    
                self.true_flux[i] = data['flux'].to_numpy() 
                self.true_flux[i] -= np.nanmean(self.true_flux[i])
                self.true_error[i] = data['error'].to_numpy()
                self.true_labels[i] = name.split("_")[1].split(".")[0]
                
                #renorm
                dr = self.true_flux[i]
                e = self.true_error[i]
                drn = (dr - np.nanmin(dr))/ (np.nanmax(dr) - np.nanmin(dr))
                en = (e - np.nanmin(dr))/ (np.nanmax(dr) - np.nanmin(dr))
                self.true_flux[i] = drn
                self.true_error[i] = en
                
                i += 1
                
        return 
    
    def plot_all_real(self): 
        """ 
        Plot all the real datasets
        """
        # n_tess is how many to plot
        print("Plotting all real tess data!")
        cols = 2 if self.n_tess > 2 else 1
        rows = int(np.ceil(self.n_tess/cols)) #
        
        fig, axs = plt.subplots(rows, cols, figsize=(cols*8, rows*3))
        
        if self.n_tess == 1:
            t = self.x
            flux = self.true_flux[0] 
            err = self.true_error[0]
            axs.errorbar(t, flux, err, mfc='black', mec='black', ms=1, fmt=".", ecolor='black')
            axs.set_title(f"TIC {int(self.true_labels[0])}", fontsize=14)

        elif self.n_tess == 2:
            for i in range(2): 
                axy = axs[i]
                t = self.x
                flux = self.true_flux[i] 
                err = self.true_error[i]
        
                axy.errorbar(t, flux, err, mfc='black', mec='black', ms=1, fmt=".", ecolor='black')
                axy.set_title(f"TIC {int(self.true_labels[i])}", fontsize=14)
                
        else: #multiple rows 
            for j in range(cols):
                for i in range(rows):
                    axy = axs[i][j]

                    t = self.x
                    flux = self.true_flux[j*10 + i] 
                    err = self.true_error[j*10 + i]
            
                    axy.errorbar(t, flux, err, mfc='black', mec='black', ms=1, fmt=".", ecolor='black')
                    axy.set_title(f"TIC {int(self.true_labels[j*10 + i])}", fontsize=14)
             
        plt.suptitle("Real Light Curve(s)")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}injected_data.png")
        return
    
    def gen_signals(self):
        """ 
        Make signals 
        """
        print("Generating injection signals: ", end="")
        self.fake_flux = np.zeros((self.n_injections, self.l)) # n l-length arrays
        self.fake_error = np.zeros((self.n_injections, self.l))
        self.x -= self.x[0] #time axis gets 0'd out
        
        for i in range(self.n_injections):
            if self.rise == 1:
                self.fake_flux[i] = self.__gen_singlerise(self.true_params[i])
            
            elif self.rise == 2:
                self.fake_flux[i] = self.__gen_doublerise(self.true_params[i])
                
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
        
    
    def gen_injections(self):
        """ 
        Generate randomly injected sources
        """
        print("Injecting signals: ", end="")
        
        #injected fluxes are going to be n_LC, length l
        self.injected_flux = np.zeros((self.n_LC, self.l)) 
        self.injected_error = np.zeros((self.n_LC, self.l))
        self.injected_indexes = np.zeros((self.n_tess, self.fraction_to_inject)) 
        
        #Needs to be repeatable - have to save which indexes got injected!! 
        if os.path.exists(self.injected_index_file):
            print("Injections already chosen, loading them in: ", end="")
            #write me! 
            self.injected_indexes = pd.read_csv(self.injected_index_file).to_numpy()
            
            
        else: 
            print("injections need to be generated:", end="")
            rng = np.random.default_rng() #init random generator
            
            #generate
            for j in range(self.n_tess): #for each real 
                self.injected_indexes[j] = rng.choice(self.n_injections, self.fraction_to_inject, replace=False)
                
            di = {} #save cols: 
            for i in range(self.fraction_to_inject): 
                di[f"col_{i}"] = self.injected_indexes[:,i]
            
            df = pd.DataFrame(di)
            df.to_csv(self.injected_index_file)
        
        
        #injected indexes: for each in n_tess, fraction_to_inject indexes
        
        
        
        
        for j in range(self.n_tess): #for each real 
            
            #0th in that range will be true
            self.injected_flux[j*self.n_tess] = self.true_flux[j]
            self.injected_error[j*self.n_tess] = self.true_error[j] #just use the real data errors
            #inject the rest: 
            for i in range(1, self.n_LC): 
                 inj_ = int(self.injected_indexes[j][i-1]) #index in injection array
                 #print(inj_)
                 self.injected_flux[int(j*self.n_tess + i)] = self.fake_flux[inj_] + self.true_flux[j]
                 self.injected_error[int(j*self.n_tess + i)] = self.true_error[j]
        
        
        #save injections: 
        
        print("done")
        return
    
    def plot_all_injections(self):
        """ 
        plot the fake data  + retrieved models
        
        - every background tess gets their own plot 
        """
        
        n_plots = self.n_tess
        cols = 2
        rows = int(np.ceil(self.n_LC/cols))
        print(f'plots:{n_plots}, cols: {cols}, rows: {rows}')
        
        #reset axis: 
        if self.x[0] != 0:
            x_ = self.x - 2457000
        else:
            x_ = self.x
        
        k = 0 #array indexer
        c = ['black', 'darkgreen']
        
        for i in range(n_plots): 
            fig, ax = plt.subplots(rows, cols, figsize=(8*cols, 3*rows))
            inj_row = self.injected_indexes[i].astype(int)
            #print(inj_row)
            #now fill subplots: 
            for j in range(self.n_LC):
                axy = ax[int(j/2),j%2]
                axy.scatter(x_, self.injected_flux[k], s=2, color=c[0 if j==0 else 1])
                
                if j != 0: 
                    inj_index = int(inj_row[j-1])
                    axy.axvline(self.dtimes[inj_index]-2457000, color='blue', label='disc. time', linestyle='dashed')
                    truemodel = self.fake_flux[inj_index]
                    axy.plot(x_, truemodel, lw=2, color='red', linestyle='dashed')
                    axy.axvline(self.true_params[inj_index][0], color='blue')
                            
                k+=1
                
            plt.suptitle(f"Injection Plot {i}: TIC {self.true_labels[i]}")
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/injection-plot-{i}.png")
                
        return
            
    def fit_fakes(self, start=0, n1=500, n2=5000):
        """ 
        Fit models to fake data starting at start index
        """
        
        #check to see if this has already been run: 
        final_file = f"{self.save_dir}/index{self.n_LC-1}/"
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
        
        self.s_3 = np.zeros((self.n_injections*self.n_tess, self.ndim))
        
        for i in range(self.n_tess): #for each tess
            for j in range(self.n_LC): #for each injection created
                
                ind = i*self.n_tess + j
                if ind < start: 
                    continue
                print("index: ", ind)
                dt = self.disctimes[j]
                self.trlc = etsMAIN(self.save_dir, 'nofile', plot=False)
            
                self.trlc.load_single_lc(self.x, self.injected_flux[ind], self.injected_error[ind], 
                                    dt, "index{}".format(ind), "", "", "")
    
                self.trlc.pre_run_clean(fitType=1)
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
    
    def __fit_fakes_doublerise(self, start=0, n1=500, n2=5000):  
        
        self.s_3 = np.zeros((self.n_injections*self.n_tess, self.ndim))
        for i in range(start, self.n_LC):
            dt = self.disctimes[i]
            self.trlc = etsMAIN(self.subfolder, 'nofile', plot=False)
        
            self.trlc.load_single_lc(self.x, self.flux_fake[i], self.error_fake[i], 
                                dt, "index{}".format(i), "", "", "")

            self.trlc.pre_run_clean(fitType=3)
            self.trlc.run_MCMC(n1, n2, quiet=True)
            self.s_3[i] = self.trlc.sampler.get_autocorr_time(tol=0)
            
        #save S3 file:
        di = {'act_t0':self.s_3[:,0],
              'act_t1':self.s_3[:,1],
              'act_A1':self.s_3[:,2],
              'act_A2':self.s_3[:,3],
              'act_beta1':self.s_3[:,4],
              'act_beta2':self.s_3[:,5],
              'act_B':self.s_3[:,6]}
        df = pd.DataFrame(di)
        self.autocorr_file = f"{self.save_dir}allfluxes-{self.targetlabel}-{self.rise}-autocorr.csv"
        df.to_csv(self.autocorr_file)
        return
    
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
    
    def s_stats(self):
        """ 
        Calculating s-stats as given in the paper
        """
        # init arrays
        self.s_1 = np.zeros((len(self.output_params), self.ndim))
        self.s_2 = np.zeros((len(self.output_params), self.ndim))
        # set 0th in each set to nan (no true params exist)
        self.s_1[0:len(self.output_params):self.n_LC] = np.nan
        self.s_2[0:len(self.output_params):self.n_LC] = np.nan
        
        #in loops through the intermediate values
        for i in range(self.n_tess): #for each set
            ind_start = i * self.n_LC + 1
            inj_ind = self.injected_indexes[i] #which ones are they
            print(inj_ind)
            for j in range(len(inj_ind)): #for each injected index
                h = ind_start+j
                #print(h)
                ii = inj_ind[j].astype(int)
                #print(ii)
                max_err = np.maximum(self.upper_error[h].round(3), self.lower_error[h].round(3))
                self.s_1[h] = np.abs(self.output_params[h].round(3) / max_err)
                ptrue = self.true_params[ii]
                self.s_2[h] = np.abs(np.abs(self.output_params[h].round(3) - ptrue.round(3))/ptrue.round(3))
                
        return 
        
    
    def chi_sq(self):
        """ 
        calc chisquared values
        """
        p = self.n_LC * self.n_tess
        self.chi_squared = np.zeros(p) #n values
        x_ = self.x
        if self.x[0] != 0:
            x_ -= 2457000
        if self.rise == 1:
            for i in range(p): 
                model = self.__gen_singlerise(self.output_params[i])
                self.chi_squared[i] = np.nansum( ( model - self.injected_flux[i] )**2 / self.injected_flux[i])
                
        elif self.rise == 2: 
            for i in range(p): 
                model = self.__gen_doublerise(self.output_params[i])
                self.chi_squared[i] = np.nansum( ( model - self.injected_flux[i] )**2 / self.injected_flux[i])
        
        return
        
  
        
#%%
data_folder = "/Users/lindseygordon/research/urop/reliability_testing/chosen_data/"
save_dir = "/Users/lindseygordon/research/urop/reliability_testing/output/"
n_injections = 10
n_tess = 1

#init
ri = reliability_injection(save_dir, n_injections, n_tess, rise=1)

# real tess data: 
ri.real_data_load(data_folder)
ri.plot_all_real()

# fake signals:
ri.gen_params()
ri.gen_signals()

# signal injection
ri.gen_injections()
ri.plot_all_injections()

# #signal fitting: 
ri.fit_fakes(start=0, n1=5_000, n2=50_000)
# ri.retrieve_params()
# ri.s_stats()
# ri.chi_sq()
#%%



