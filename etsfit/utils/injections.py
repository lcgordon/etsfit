#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 08:41:41 2023
update oct 6 2023 - docstring updates

Produces and injects artificial signals into fake tess light curves for validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from etsfit import etsfit
import etsfit.utils.utilities as ut
from pylab import rcParams
import os
import pandas as pd
from astropy.time import Time
from astropy.stats import SigmaClip
from scipy.stats import truncnorm
rcParams['figure.figsize'] = 8,3
rcParams['font.family'] = 'serif'



class artificial_injections(object): 
    """ 
    Run artificial signal injections to test precision/recall 

    :param save_dir: (str) - outermost dir to save everything into
    :param total_lightcurves: (int) - how many total LC to run on
    :param percent_injected: what fraction of n_total_lightcurves to put a fake signal into. must be < 1, will round UP on calc.
    :param PL_rise:  1 or 2, how many power laws to inject
    :param folderprefix: (str) wherever in your directories your data is hanging out
    """
    
    def __init__(self, save_dir, total_lightcurves, percent_injected, 
                 PL_rise=1, folderprefix="/Users/lindseygordon/research/urop/tessreduce_lc/"):
        """ 
        obj constructor
        """
        self.total_lightcurves = total_lightcurves
        self.datafolder = folderprefix
        
        if percent_injected > 1: 
            return ValueError("Injection percent must be < 1!")
        if PL_rise not in (1,2):
            return ValueError("Power law must have 1 or 2 rises! ")
        
        # calc how many injections: 
        self.percent_injected = percent_injected
        self.n_injections = int(np.ceil(self.total_lightcurves * self.percent_injected))
        self.n_null = self.total_lightcurves - self.n_injections
        # array of truths
        self.injection_truths = np.ones(self.total_lightcurves)
        self.injection_truths[0:self.n_null] = 0
        
        # power law rises
        self.PL_rise = PL_rise
        if self.PL_rise == 1:
            self.dim = 4
            self.labels = [r'$t_0$', 'A', r'$\beta$', 'B']
        elif self.PL_rise == 2:
            self.dim = 7
            self.labels = [r'$t_0$', r'$t_1$', r'$A_1$', r'$A_2$', 
                            r'$\beta_1$', r'$\beta_2$', 'B']
        
        # save dir:
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print(f"Generating save directory: {save_dir}")
        
        return 
        
    def load_background(self, background_type="noise", background=0):
        """ 
        Load in the background to use
        
        :param background_type: (str) - "noise" or "badIa"
        :param background: (int) - 0,1,2 for each
        """
        if background not in (0,1,2):
            return ValueError(f"{background} is not a valid background - 0,1,2!")
        if background_type not in ("noise", "badIa"):
            return ValueError(f"{background_type} is not a valid type - 'noise' or 'badIa' ")
        
        self.bg_type = background_type
        print(f"Background type: {self.bg_type}")
        self.bg = background
        print(f"Background number: {self.bg}")
        self.noise_bg = ["2020tld2921", "2018fhw0141", "2020vem3012"]
        self.real_bg = ['2018fwi','2019sqj', '2020azn']
        
        if background_type == "noise":
            self.targetlabel = self.noise_bg[self.bg]
            # make x axis: 
            self.__gen_x()
            # generate all backgrounds: 
            self.generate_all_noisemodels()
            # create new folder to save into: 
            new_dir = f"{self.save_dir}tess_noise_{self.total_lightcurves}_{self.n_injections}_{self.PL_rise}/"
            if not os.path.exists(new_dir):
                print(f"Making new save folder: {new_dir}")
                os.mkdir(new_dir)
            self.save_dir = new_dir
            
            self.__background_LC_maker()
            
            
        else: # bad Ia backgrounds: 
            self.targetlabel = self.real_bg[self.bg]
            self.real_data_load()
            # create new folder to save into: 
            new_dir = f"{self.save_dir}tess_real_{self.total_lightcurves}_{self.n_injections}_{self.PL_rise}/"
            if not os.path.exists(new_dir):
                print(f"Making new save folder: {new_dir}")
                os.mkdir(new_dir)
            self.save_dir = new_dir
        
        
        return 
    
    def __gen_x(self):
        """ 
        Make x-axis with orbit gap for noise model running
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
    
    def generate_all_noisemodels(self): 
        """ 
        Make the global files containing the TESS noise to draw from
        Should go in the top directory for all fake data
        """
        self.noise_f = "{s}/tess_noise_models.csv".format(s=self.save_dir)
        
        if os.path.exists(self.noise_f):
            print("Background models to draw from already exist!")
            return
        
        bg1 = f"{self.datafolder}2020tld2921/2020tld2921-tessreduce" 
        bg2 = f"{self.datafolder}2018fhw0141/2018fhw0141-tessreduce" 
        bg3 = f"{self.datafolder}2020vem3012/2020vem3012-tessreduce" 
        bgs = [bg1, bg2, bg3]
        t_cuts = [2100.507, 1337.969, 2128.120] #tld, fhw, vem
        
        di = {}
        
        #generate models
        for i in range(3): 
            tessreducefile = bgs[i]
            # targetlabel = tessreducefile.split("/")[-1].split("-")[0]
            # print(targetlabel)
            # self.targetlabel = targetlabel
            #load tess data
            (time, flux, error, targetlabel, 
             sector, camera, ccd) = ut.tr_load_lc(tessreducefile)
            t_c = t_cuts[i] #which to use
            plt.scatter(time, flux, color='black')
            index_ = np.abs(time - t_c - 2457000).argmin()
            plt.axvline(time[index_], color='red')
            cut_time = time[0:index_]
            cut_flux = flux[0:index_]
            plt.scatter(cut_time, cut_flux, color='blue')
            
            dt = Time(cut_time[1], format='jd') - Time(cut_time[0], format='jd')
            
            if dt.sec > 1700:
                print(dt.sec, "in the okay cadence! ")
            else: 
                print(dt.sec, "need to bin this ")
                n_pts = int(np.rint(1800/dt.sec)) #3 ten minutes = 30 minutes
                t_ = []
                f_ = []
                
                n = int(0)
                m = int(n_pts+1)
                
                while m<len(cut_time):
                    t_.append(cut_time[n])
                    rang = cut_flux[n:m][~np.ma.masked_invalid(cut_flux[n:m]).mask]
                    if len(cut_flux) == 0:
                        f_.append(np.nan)
                    else: 
                        f_.append(np.nanmean(rang))
                
                    n+= n_pts
                    m+= n_pts  
             
                cut_time = np.asarray(t_)
                cut_flux = np.asarray(f_)
            plt.scatter(cut_time, cut_flux, color='green')    
            cut_flux -= np.mean(cut_flux)
            
            sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
            mask = np.ma.getmask(sigclip(cut_flux))
            cut_flux = cut_flux[~mask]
            cut_time = cut_time[~mask]
            plt.scatter(cut_time,cut_flux,  s=2, color='hotpink')
            plt.show()
            
            h = np.full(700, np.nan)
            h[0:len(cut_flux)] = cut_flux
            
            di[self.noise_bg[i]] = h
            #put models into dict
        #put dict into csv
        df = pd.DataFrame(di)
        df.to_csv(self.noise_f)
        
        return
    
    def __background_LC_maker(self): 
        """ 
        Sample the bg to produce a fake light curve
        """
        # make new subfolder for this run
        self.subfolder = f"{self.save_dir}{self.targetlabel}/"
        
        if not os.path.exists(self.subfolder):
            os.mkdir(self.subfolder)
            print(f"New subfolder: {self.subfolder}")
        else:
            print(f"SUBFOLDER {self.subfolder} ALREADY EXISTS - YOU MAY BE OVERWRITING RESULTS")
            
        # file to load if it exists: 
        f_noise_run = f"{self.subfolder}{self.targetlabel}-background.csv"
        if os.path.exists(f_noise_run):
            #load it
            self.__retrieve_noisemodel()
        else: 
            #generate it
            print("Generating noise model: ", end="")
            # load in the TESS bg file: 
            noise_all = pd.read_csv(self.noise_f)
            noise_use = noise_all[self.targetlabel]
            #mask nans: 
            noise_use = noise_use[~np.ma.masked_invalid(noise_use).mask]
            self.true_flux = np.random.choice(noise_use, self.l)
            self.true_error = np.full(self.l, 0.1*np.mean(self.true_flux))
            #print("noise shape", self.noise_model.shape)
            #save
            di = {'noise_model':self.true_flux}
            df = pd.DataFrame(di)
            df.to_csv(f_noise_run)
            print("done!")
        
        return 
    
    def __retrieve_noisemodel(self):
        f_noise_run = f"{self.subfolder}{self.targetlabel}-background.csv"
        #load it
        print("Loading existing noise model: ", end="")
        self.true_flux = pd.read_csv(f_noise_run)['noise_model']
        self.true_error = np.full(self.l, 0.1*np.mean(self.true_flux))
        #print("noise shape", self.noise_model.shape)
        print("done!")
        return 
    
    def real_data_load(self): 
        """ 
        Given a folder of csv files, load the data into an array
        """
        print("Loading TESS data!")
        bg0 = f"{self.datafolder}2018fwi0211/2018fwi0211-tessreduce"
        bg1 = f"{self.datafolder}2019sqj1714/2019sqj1714-tessreduce"
        bg2 = f"{self.datafolder}2020azn2112/2020azn2112-tessreduce"
        
        all_bg = [bg0, bg1, bg2]
        data = pd.read_csv(all_bg[self.bg])
        
        #make x, make full data array, full error array + fill
        self.x = data['time'].to_numpy()
        self.true_flux = data['flux'].to_numpy() 
        self.true_error = data['flux_err'].to_numpy()
        print(f"Data starts at: {self.x[0].round(2)}")
        self.x -= self.x[0]
        
    
        #sigma clip: 
        sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
        sc = sigclip(self.true_flux).mask
        self.x = self.x[~sc]
        self.true_flux = self.true_flux[~sc]
        self.true_error = self.true_error[~sc]
        
        #remove yucky regions in 2019sqj
        if self.bg == 1:
            mask = np.ones(len(self.x)).astype(bool)
            mask[400:530] = False
            mask[1020:] = False
            print(f"Data trunc on 2019sqj: {self.x[400].round(2)}, {self.x[530].round(2)},\
                  {self.x[1020].round(2)}, {self.x[-1].round(2)}")
            self.x = self.x[mask]
            self.true_flux = self.true_flux[mask]
            self.true_error = self.true_error[mask]
    
        self.l = len(self.x)
        print(f"x length: {self.l}")
        self.subfolder = f"{self.save_dir}{self.targetlabel}/"
        if not os.path.exists(self.subfolder):
            os.mkdir(self.subfolder)
            print(f"New subfolder: {self.subfolder}")
        else:
            print(f"SUBFOLDER {self.subfolder} ALREADY EXISTS - YOU MAY BE OVERWRITING RESULTS")
                
        return 
    
    def gen_injections(self, plot=True): 
        """ 
        Produce the set of injections for testing by calling gen_params() and
        __inject(). 

        :param plot: bool to plot injections
        """
        # generate parameters 
        self.gen_params()
        # inject all 
        self.__inject()
        # plot if you want
        if plot:
            self.plot_injections()
        return 
    
    def gen_params(self):
        """ 
        Produce the true parameter set + save
        """
        self.true_param_file = "{s}true-params.csv".format(s=self.save_dir) 
        if os.path.exists(self.true_param_file):
            print('Params already exist, loading them in')
            h = pd.read_csv(self.true_param_file)
            self.params_true = h.to_numpy()[:,1:-1]
            self.dtimes = h.to_numpy()[:,-1]
            self.disctimes = {}
            labels = list(range(self.total_lightcurves))
            for i in range(self.total_lightcurves):
                self.disctimes[labels[i]] = self.dtimes[i]
       
        else: 
            print('Making new paramset from scatch')
            myclip_a, myclip_b = 0.5, 4
            loc, scale = 2, 1
            a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
        
            if self.PL_rise == 1:
                self.params_true = np.zeros((self.total_lightcurves, self.dim)) ##t0 A beta B
                self.params_true[self.n_null:,0] = np.random.uniform(5, 20, self.n_injections) #t0
                self.params_true[self.n_null:,1] = np.random.uniform(0.1, 5, self.n_injections) #A1
                # beta is pulled from a unif distro on the arctans
                self.params_true[self.n_null:,2] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n_injections)
                self.params_true[self.n_null:,3] = np.random.uniform(1, 20, self.n_injections) *  \
                                                    np.random.choice((-1,1), self.n_injections)
                
            elif self.PL_rise == 2:
                self.params_true = np.zeros((self.total_lightcurves, self.dim)) ##t0 A beta B
                self.params_true[self.n_null:,0] = np.random.uniform(1, 20, self.n_injections) #t0
                self.params_true[self.n_null:,1] = np.random.uniform(self.params_true[:,0], 25, self.n_injections) #t1 defined by t0
                self.params_true[self.n_null:,2] = np.random.uniform(0.1, 5, self.n_injections) #A1
                self.params_true[self.n_null:,3] = np.random.uniform(0.1, 5, self.n_injections) #A2
                self.params_true[self.n_null:,4] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n_injections)
                self.params_true[self.n_null:,5] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n_injections)
                self.params_true[self.n_null:,6] = np.random.uniform(1, 20, self.n_injections) * \
                    np.random.choice((-1,1), self.n_injections) #B
                
             
            #gen disc times
            self.disctimes = {}
            labels = list(range(self.total_lightcurves))
            self.dtimes = self.params_true[:,0] + np.random.uniform(0.5, 6, self.total_lightcurves) + 2457000
            for i in range(self.total_lightcurves):
                self.disctimes[labels[i]] = self.dtimes[i]
                
            # save params
            if self.PL_rise == 1:
                di = {'t0':self.params_true[:,0], 
                      'A':self.params_true[:,1],
                      'beta':self.params_true[:,2],
                      'B':self.params_true[:,3], 
                      'disc':self.dtimes}
            elif self.PL_rise == 2:
                di = {'t0':self.params_true[:,0], 
                      't1':self.params_true[:,1],
                      'A1':self.params_true[:,2],
                      'A2':self.params_true[:,3],
                      'beta1':self.params_true[:,4],
                      'beta2':self.params_true[:,5],
                      'B':self.params_true[:,6], 
                      'disc':self.dtimes}
            
            df = pd.DataFrame(di)
            df.to_csv(self.true_param_file)
            
        return
    
    def __inject(self): 
        """ 
        Inject signals into background.
        Don't need to save produced LC because always the same when generated
        """
        
        self.generated_flux = np.zeros((self.total_lightcurves, self.l)) # n l-length arrays
        self.generated_error = np.zeros((self.total_lightcurves, self.l))
        print("Injecting fluxes: ", end="")
        for i in range(self.total_lightcurves):
            if self.PL_rise == 1:
                model = self.gen_singlerise(self.params_true[i])
            elif self.PL_rise == 2:
                model = self.gen_doublerise(self.params_true[i])
            
            #attach noise: 
            self.generated_flux[i] = model + self.true_flux
            # set error = expectation value of noise model
            self.generated_error[i] += self.true_error
            
                    
        self.x += 2457000 #reset time axis because it will get subtracted
        print("done")
        return 
    
    def gen_singlerise(self, params):
        t0, A, beta, B = params
        if self.x[0] != 0:
            t_ = self.x - self.x[0] - t0
        else:
            t_ = self.x - t0
        
        model = (np.heaviside((t_), 1) * A * np.nan_to_num((t_**beta))) + 1 + B
        return model
    
    def gen_doublerise(self, params):
        
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        
        t0, t1, A1, A2, beta1, beta2, B = params
        if self.x[0] != 0:
            t_ = self.x - self.x[0] - t0
        else:
            t_ = self.x - t0
        model = np.piecewise(t_, [(t0 <= t_)*(t_ < t1), t1 <= t_], 
                                         [func1, func2],
                                         t0, t1, A1, A2, beta1, beta2) + 1 + B
        return model
    
    def plot_injections(self): 
        """ 
        Plot all of the injections in the completenes sample: 
        """
        cols = 5
        rows = int(np.ceil(self.total_lightcurves/cols))
        print(f'cols: {cols}, rows: {rows}')
        
        #reset axis: 
        if self.x[0] != 0:
            x_ = self.x - 2457000
        else:
            x_ = self.x
        
        fig, ax = plt.subplots(rows, cols, figsize=(4*cols, 1.5*rows))
        for j in range(self.total_lightcurves):
            axy = ax[int(j/5),j%5]
            
            axy.scatter(x_, self.generated_flux[j], s=2, color='black')
            #axy.axvline(self.dtimes[j]-2457000, color='grey', label='disc. time', linestyle='dashed')
            axy.axvline(self.params_true[j][0], color='blue', label='t0', linestyle='dashed')
                
        plt.suptitle(f"Injection Plot: {self.targetlabel}")
        plt.tight_layout()
        plt.savefig(f"{self.subfolder}/injection-plot.png")
        plt.show()
        plt.close()
        return 
      
    def run_PL_fitting(self, start=0, n1=1_000, n2=15_000): 
        """ 
        Perform completeness fitting on the generated sample
        
        :param start: which injection to start fitting on
        :param n1: mcmc n1
        :param n2: mcmc n2
        """
        tag = ['x', 'singlepower', 'doublepower']
        final_file = f"{self.subfolder}/index{self.total_lightcurves-1}/{tag[self.PL_rise]}/"
        print(f"The final file is: {final_file}, ", end="")
        if os.path.exists(final_file):
            print("which already exits, exiting PL fitting.")
            return
        print("which does not exist, running PL fitting now. ")
        
        types_ = [None, 1, 3] #which fit type to run 
        
        for i in range(start, self.total_lightcurves):
            print(f"index: {i}")
            dt = self.disctimes[i]
        
            self.trlc = etsMAIN(self.subfolder, 'nofile', plot=False)
        
            self.trlc.load_single_lc(self.x, self.generated_flux[i], 
                                     self.generated_error[i], 
                                     dt, f"index{i}", "", "", "")

            self.trlc.pre_run_clean(fitType=types_[self.PL_rise])
            self.trlc.run_MCMC(n1, n2, quiet=True)   
            
            if (self.x[0] ==0 ):
                self.x += 2457000
        
        return 
    
    def run_flat_fitting(self, start=0, n1=100, n2=1000):
        """ 
        Run fitting for the flat background 
        
        :param start: which injection to start fitting on
        :param n1: mcmc n1
        :param n2: mcmc n2

        """
        final_file = f"{self.subfolder}/index{self.total_lightcurves-1}/flat"
        print(f"The final file is: {final_file}, ", end="")
        if os.path.exists(final_file):
            print("which already exits, exiting flat fitting.")
            return
        print("which does not exist, running flat fitting now. ")
        
        for i in range(start, self.total_lightcurves):
            print(f"index: {i}")
        
            self.trlc = etsMAIN(self.subfolder, 'nofile', plot=False)
        
            self.trlc.load_single_lc(self.x, self.generated_flux[i], 
                                     self.generated_error[i], 
                                     self.disctimes[i], f"index{i}", "", "", "")

            self.trlc.pre_run_clean(fitType=20)
            self.trlc.run_MCMC(n1, n2, quiet=True)   
        return
    
    def retrieve_singlepower_params(self):
        """ 
        Load params in from output files
        """
        true_p = pd.read_csv(self.true_param_file)
        
        # load in calculated params
        import etsfit.utils.parameter_retrieval as ba
        params_all = {}
        converged_all = {}
        upper_all = {}
        lower_all = {}

        
        for root, dirs, files in os.walk(self.subfolder):
            for name in files:
                if (name.endswith("-singlepower-output-params.txt")):
                    targ = name.split("-")[0]
                    if targ[0] != "i":
                        continue #oops hit a noise model
                    
                    filepath = root + "/" + name
                    if self.PL_rise == 1: 
                        (params,  upper_e, 
                         lower_e,  converg) = ba.extract_singlepower_params(filepath)
                    elif self.PL_rise == 2:
                        (params,  upper_e, 
                         lower_e,  converg) = ba.extract_doublepower_params(filepath)
                    
                    params_all[targ] = params
                    upper_all[targ] = upper_e
                    lower_all[targ] = lower_e
                    converged_all[targ] = converg
                    
        # dicts into arrays: 
        p_ =  len(params_all)
        self.converged_all = converged_all
        
        self.output_params = np.zeros((p_, self.dim))
        self.upper_error = np.zeros((p_, self.dim))
        self.lower_error = np.zeros((p_, self.dim))
        self.converged_retrieved = np.zeros(p_)
        for i in range(p_):
            st_ = 'index{}'.format(i)
            self.output_params[i] = params_all[st_]
            self.upper_error[i] = upper_all[st_]
            self.lower_error[i] = lower_all[st_]
            self.converged_retrieved[i] = converged_all[st_]
        
        self.converged_retrieved = self.converged_retrieved.astype(int)
        self.params_true = true_p.to_numpy()[:,1:self.dim+1]
        print("Loaded single power law params.")
        return 
    
    def retrieve_flat_params(self):
        """ 
        Load in flat param best fits
        """
        # load in calculated params
        import etsfit.utils.parameter_retrieval as ba
        params_all = {}
        converged_all = {}
        upper_all = {}
        lower_all = {}

        for root, dirs, files in os.walk(self.subfolder):
            for name in files:
                if (name.endswith("-flat-output-params.txt")):
                    targ = name.split("-")[0]
                    if targ[0] != "i":
                        continue #oops hit a noise model
                    
                    filepath = root + "/" + name
                    (params,  upper_e, 
                     lower_e,  converg) = ba.extract_flat_params(filepath)
                    
                    params_all[targ] = params
                    upper_all[targ] = upper_e
                    lower_all[targ] = lower_e
                    converged_all[targ] = converg
                    
        # dicts into arrays: 
        p_ =  len(params_all)
        
        self.flat_params = np.zeros((p_, 1))
        self.flat_upper = np.zeros((p_, 1))
        self.flat_lower = np.zeros((p_, 1))
        for i in range(p_):
            st_ = 'index{}'.format(i)
            self.flat_params[i] = params_all[st_]
            self.flat_upper[i] = upper_all[st_]
            self.flat_lower[i] = lower_all[st_]
       
        print("Loaded flat params")
        return 
    
    def plot_powerlaw_predictions(self): 
        """ 
        Plot all of the injections in the sample: 
        """
        cols = 5
        rows = int(np.ceil(self.total_lightcurves/cols))
        print(f'cols: {cols}, rows: {rows}')
        
        #reset axis: 
        if self.x[0] != 0:
            x_ = self.x - 2457000
        else:
            x_ = self.x
        
        fig, ax = plt.subplots(rows, cols, figsize=(4*cols, 1.5*rows))
        for j in range(self.total_lightcurves):
            axy = ax[int(j/cols),j%cols]
            
            axy.scatter(x_, self.generated_flux[j], s=2, color='black')
            if self.PL_rise==1:
                model = self.gen_singlerise(self.output_params[j])
            else:
                model = self.gen_doublerise(self.output_params[j])
            axy.plot(x_, model, lw=2, color='red')
            
            #axy.axvline(self.dtimes[j]-2457000, color='blue', label='disc. time', linestyle='dashed')
            axy.axvline(self.params_true[j][0], color='blue', label='t0', linestyle='dashed')
            axy.axvline(self.output_params[j][0], color='red', label='t0', linestyle='dashed')
                
        plt.suptitle(f"Output models: {self.targetlabel}")
        plt.tight_layout()
        plt.savefig(f"{self.subfolder}/injection-output-models-plot.png")
        plt.show()
        plt.close()
        return 
    
    def plot_flat_predictions(self): 
        """ 
        Plot all of the injections in the sample: 
        """
        cols = 5
        rows = int(np.ceil(self.total_lightcurves/cols))
        print(f'cols: {cols}, rows: {rows}')
        
        #reset axis: 
        if self.x[0] != 0:
            x_ = self.x - 2457000
        else:
            x_ = self.x
        
        fig, ax = plt.subplots(rows, cols, figsize=(4*cols, 1.5*rows))
        for j in range(self.total_lightcurves):
            axy = ax[int(j/cols),j%cols]
            
            axy.scatter(x_, self.generated_flux[j], s=2, color='black')
    
            axy.plot(x_, np.zeros_like(x_)+self.flat_params[j], lw=2, color='red')
            
                
        plt.suptitle(f"Output flat models: {self.targetlabel}")
        plt.tight_layout()
        plt.savefig(f"{self.subfolder}/injection-flat-models-plot.png")
        plt.show()
        plt.close()
        return 
    
    def calc_BIC_ratio(self):
        """ 
        calc BIC values + ratios
        -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2))
        """
        self.loglike_flat = np.zeros(self.total_lightcurves)
        self.loglike_powerlaw = np.zeros(self.total_lightcurves)  
        
        for i in range(self.total_lightcurves):
            yerr2 = self.generated_error[i]**2
            y = self.generated_flux[i]
            # FLAT: 
            model = np.zeros_like(self.x) + self.flat_params[i]
            s2 = yerr2 + model**2
            self.loglike_flat[i] = -0.5 * np.nansum((y - model) ** 2 / s2 + np.log(2* np.pi * s2))
            # POWER LAW: 
            model = self.gen_singlerise(self.output_params[i])
            s2 = yerr2 + model**2
            self.loglike_powerlaw[i] = -0.5 * np.nansum((y - model) ** 2 / s2 + np.log(2* np.pi *s2))
        
        self.BIC_flat = 1 * np.log(self.l) - 2 * self.loglike_flat
        self.BIC_powerlaw = 4 * np.log(self.l) - 2 * self.loglike_powerlaw
        self.BIC_diff = self.BIC_powerlaw - self.BIC_flat
        
        self.BIC_ratios = self.BIC_powerlaw / self.BIC_flat
        self.smaller_BIC = np.argmin(np.vstack((self.BIC_flat, self.BIC_powerlaw)), axis=0)
        #this always predicts the power law
        
        self.likelihood_ratio = self.loglike_powerlaw - self.loglike_flat
        # this would only be valid for nested models (single vs double, ie)
        
        self.AIC_flat = 2 - 2*self.loglike_flat
        self.AIC_powerlaw = 2*4 - 2*self.loglike_powerlaw
        self.AIC_min = np.argmin(np.vstack((self.AIC_flat, self.AIC_powerlaw)), axis=0)
        self.relative_likelihood = np.exp( (self.AIC_powerlaw - self.AIC_flat)/2)
        # again, keep predicting the power law 
        return
    
    def BIC_CM(self):
        """ 
        calc CM based on BIC
        """
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        self.cm = confusion_matrix(self.injection_truths, self.smaller_BIC)
        #print(self.cm)
        tn, fp, fn, tp = confusion_matrix(self.injection_truths, self.smaller_BIC).ravel()
        print(tn, fp, fn, tp)
        self.precision = tp / (tp+fp) # tp/tp+fp
        print(f"precision: {self.precision}")
        self.recall = tp / (tp+fn)#tp / (tp + fn)
        print(f"recall: {self.recall}")
        
        fig, ax = plt.subplots(1, figsize=(5,5))
        
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm,
                                      display_labels=['Null','Inj.'])

        disp.plot(ax=ax)
        plt.title(f"{self.targetlabel} CM")
        plt.tight_layout()
        plt.savefig(f"{self.subfolder}{self.targetlabel}-confusionmatrix.png")
        plt.show()
        plt.close()
        
        self.wrongs = np.where(self.injection_truths != self.smaller_BIC)[0]
        print(f"false negatives {self.targetlabel}: {self.wrongs}")
        
        return
    
    def fract_acc_injected(self):
        
        self.s_2 = np.abs(np.abs(self.output_params[self.diff:].round(3) - 
                          self.params_true[self.diff:].round(3))/self.params_true[self.diff:].round(3)) 
        
        print("Average fractional acc per param: ", self.s_2.mean(axis=0).round(3))
        return
    
    def run_all(self, background_type, background, plot=False):
        """ 
        After setting up, run all to produce BIC CM
        """
        self.load_background(background_type, background)
        self.gen_injections(plot=plot)
        self.run_PL_fitting()
        self.run_flat_fitting()
        self.retrieve_singlepower_params()
        self.retrieve_flat_params()
        if plot:
            self.plot_powerlaw_predictions()
            self.plot_flat_predictions()
        self.calc_BIC_ratio()
        self.BIC_CM()
        return
    
    def re_run_wrongs(self, n1=1_000, n2=15_000):
        """
        re-fit to fn/fp 
        """
        
        newfolder = f"{self.save_dir}/rerun_falsenegs/"
        if not os.path.exists(newfolder):
            os.mkdir(newfolder)
        
        for i in self.wrongs: 
            print(f"index: {i}")
            #print(self.x[0])
            dt = self.disctimes[i]
        
            self.trlc = etsMAIN(newfolder, 'nofile', plot=True)
        
            self.trlc.load_single_lc(self.x, self.generated_flux[i], 
                                     self.generated_error[i], 
                                     dt, f"index{i}", "", "", "")

            self.trlc.pre_run_clean(fitType=1)
            self.trlc.run_MCMC(n1, n2, quiet=True)  
            
            print(self.trlc.best_mcmc)
            
            yerr2 = self.generated_error[i]**2
            y = self.generated_flux[i]
            model = self.gen_singlerise(self.trlc.best_mcmc[0])
            s2 = yerr2 + model**2
            loglikepl = -0.5 * np.nansum((y - model) ** 2 / s2 + np.log(2* np.pi *s2))
            
            bic_pl = 4 * np.log(self.l) - 2 * loglikepl
            
            
            if (self.x[0] ==0 ):
                self.x += 2457000
                       
            print(f"index: {i}")
        
            self.trlc = etsMAIN(newfolder, 'nofile', plot=True)
        self.trlc.load_single_lc(self.x, self.generated_flux[i], 
                                 self.generated_error[i], 
                                 self.disctimes[i], f"index{i}", "", "", "")

        self.trlc.pre_run_clean(fitType=20)
        self.trlc.run_MCMC(n1, n2, quiet=True)  
        
        model = np.zeros_like(self.x) + self.trlc.best_mcmc[0][0]
        s2 = yerr2 + model**2
        loglikeflat = -0.5 * np.nansum((y - model) ** 2 / s2 + np.log(2* np.pi * s2))
        
        bic_flat = 1 * np.log(self.l) - 2 * loglikeflat
        
        if (self.x[0] ==0 ):
            self.x += 2457000
          
        print(f"index {i} smaller BIC: {np.argmin((bic_pl, bic_flat))}")
            
            
        return
        
# def get_flux_range(which):
#     all_ = ["2018fwi", "2019sqj", "2020azn"]
#     sec_ = [2, 17, 21]
    
#     name = all_[which]
#     sec = sec_[which]
#     import tessreduce as tr
#     obs = tr.sn_lookup(name)
#     lookup = obs[np.where(np.asarray(obs)[:,2] == sec)[0][0]]
#     tess = tr.tessreduce(obs_list=lookup,plot=False,reduce=True)
#     import etsfit.utils.utilities as ut
#     t, f, e, bg = ut.sigmaclip(tess.lc[0], tess.lc[1], tess.lc[2], None)
#     if which ==1: 
#         mask = np.ones(len(t)).astype(bool)
#         mask[400:530] = False
#         mask[1020:] = False
#         t = t[mask]
#         f = f[mask]
#     print(f"{name}: flux range: {np.max(f) - np.min(f)}")
    
#     lm = tess.to_mag()
#     print(np.ma.masked_invalid(lm[1]).mean())
#     return 

        