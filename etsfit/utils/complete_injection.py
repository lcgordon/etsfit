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
import os
import pandas as pd
rcParams['figure.figsize'] = 8,3
rcParams['font.family'] = 'serif'



class artificial_lc(object):
    
    
    def __init__(self, save_dir, n, rise_=1):
        """ 
        Init all 
        save_dir (str) path
        n (int) # LC to make
        rise_ (int) 1 or 2 
        """
        self.save_dir = save_dir
        self.noise_dir = save_dir
        self.bg_ = ["NoTESS", "2020tld2921", "2018fhw0141", "2020vem3012"]
    
        
        self.n = n
        self.rise_ = rise_
        
        
        if self.rise_ not in (1,2):
            print('NOT A VALID RISE_')
            return 
        
        if self.rise_ == 1:
            self.dim = 4
            self.labels = [r'$t_0$', 'A', r'$\beta$', 'B']
        elif self.rise_ == 2:
            self.dim = 7
            self.labels = [r'$t_0$', r'$t_1$', r'$A_1$', r'$A_2$', 
                           r'$\beta_1$', r'$\beta_2$', 'B']
        
        #GEN BACKGROUNDS HERE IF NOT IN EXISTENCE
        self.generate_all_backgrounds()
        
        dir_ = "{s}{n}-rise-{r}/".format(s=self.save_dir,
                                         n=self.n, r=self.rise_)
        
        if not os.path.exists(dir_):
            print("Making new save folder")
            os.mkdir(dir_)
            
        self.save_dir = dir_
        self.l = 1500
        self.__gen_x()
        
        return
    
    def generate_all_backgrounds(self): 
        """ 
        Make the global files containing the TESS noise to draw from
        Should go in the top directory for all fake data
        this is hardcoded because I am lazy
        
        """
        self.noise_f = "{s}/tess_noise_models.csv".format(s=self.noise_dir)
        
        if os.path.exists(self.noise_f):
            print("Background models to draw from already exist!")
            return
        
        bg1 = "/Users/lindseygordon/research/urop/tessreduce_lc/2020tld2921/2020tld2921-tessreduce" 
        bg2 = "/Users/lindseygordon/research/urop/tessreduce_lc/2018fhw0141/2018fhw0141-tessreduce" 
        bg3 = "/Users/lindseygordon/research/urop/tessreduce_lc/2020vem3012/2020vem3012-tessreduce" 
        bgs = [bg1, bg2, bg3]
        
        di = {}
        t_cuts = [2100.507, 1337.969, 2128.120] #tld, fhw, vem
        #generate models
        from astropy.time import Time
        from astropy.stats import SigmaClip
        for i in range(3): 
            tessreducefile = bgs[i]
            targetlabel = tessreducefile.split("/")[-1].split("-")[0]
            print(targetlabel)
            self.targetlabel = targetlabel
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
            
            di[self.targetlabel] = h
            #put models into dict
        #put dict into csv
        df = pd.DataFrame(di)
        df.to_csv(self.noise_f)
        
        return
      
    def __gen_x(self):
        """ 
        Make x-axis with orbit gap
        """
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
        f = "{s}true-params.csv".format(s=self.save_dir) 
        if os.path.exists(f):
            print('params already exist, loading them in')
            h = pd.read_csv(f)
            self.params_true = h.to_numpy()[:,1:-1]
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
                self.params_true = np.zeros((self.n, self.dim)) ##t0 A beta B
                self.params_true[:,0] = np.random.uniform(5, 20, self.n) #t0
                self.params_true[:,1] = np.random.uniform(0.001, 1.5, self.n) #A1
                # beta is pulled from a unif distro on the arctans
                self.params_true[:,2] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n)
                self.params_true[:,3] = np.random.uniform(1, 20, self.n) *  np.random.choice((-1,1), self.n)
                #B can't get to close to 0 or summary stats look like trash
                
                self.disctimes = {}
                labels = list(range(self.n))
                self.dtimes = self.params_true[:,0] + np.random.uniform(0.5, 6, self.n) + 2457000
                for i in range(self.n):
                    self.disctimes[labels[i]] = self.dtimes[i]
                
            elif self.rise_== 2:
                self.params_true = np.zeros((self.n, self.dim)) ##t0 A beta B
                self.params_true[:,0] = np.random.uniform(1, 20, self.n) #t0
                self.params_true[:,1] = np.random.uniform(self.params_true[:,0], 25, self.n) #t1 defined by t0
                self.params_true[:,2] = np.random.uniform(0.001, 1.5, self.n) #A1
                self.params_true[:,3] = np.random.uniform(0.001, 1.5, self.n) #A2
                self.params_true[:,6] = np.random.uniform(1, 20, self.n) *  np.random.choice((-1,1), self.n) #B
                # beta is pulled from a unif distro on the arctans
                self.params_true[:,4] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n)
                self.params_true[:,5] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n)
                
                self.disctimes = {}
                labels = list(range(self.n))
                self.dtimes = self.params_true[:,0] + np.random.uniform(0.5, 6, self.n) + 2457000
                for i in range(self.n):
                    self.disctimes[labels[i]] = self.dtimes[i]
    
                
            else: 
                print("not a valid rise number (1 or 2)")
            self.__save_true_params(f)
            
        return
    
    def __save_true_params(self, f):
        """ 
        Put the real values into a file for access later
        """
        if self.rise_ == 1:
            di = {'t0':self.params_true[:,0], 
                  'A':self.params_true[:,1],
                  'beta':self.params_true[:,2],
                  'B':self.params_true[:,3], 
                  'disc':self.dtimes}
        elif self.rise_ == 2:
            di = {'t0':self.params_true[:,0], 
                  't1':self.params_true[:,1],
                  'A1':self.params_true[:,2],
                  'A2':self.params_true[:,3],
                  'beta1':self.params_true[:,4],
                  'beta2':self.params_true[:,5],
                  'B':self.params_true[:,6], 
                  'disc':self.dtimes}
        
        df = pd.DataFrame(di)
        df.to_csv(f)
        return
    

    def gen_lc(self, bg=0):
        """ 
        Make LC using given bg + subfolder + save LC into file
        bg = 0-4 
        """
        self.bg = bg
        self.targetlabel = self.bg_[bg]
        # make new subfolder for this run
        self.subfolder = "{s}{tl}/".format(s=self.save_dir, tl = self.targetlabel)
        
        if not os.path.exists(self.subfolder):
            os.mkdir(self.subfolder)
        else:
            print("SUBFOLDER ALREADY EXISTS - YOU MAY BE OVERWRITING RESULTS")
            
        if bg == 0: # no tess noise    
            self.noise_model = np.zeros(self.l)
        else:
            # file to load if it exists: 
            f_noise_run = "{s}{tl}-background.csv".format(s=self.subfolder, tl = self.targetlabel)
            if os.path.exists(f_noise_run):
                #load it
                print("loading existing noise model")
                self.noise_model = pd.read_csv(f_noise_run)['noise_model']
                print("noise shape", self.noise_model.shape)
            else: 
                #generate it
                print("generating noise model for this run")
                # load in the TESS bg file: 
                noise_all = pd.read_csv(self.noise_f)
                noise_use = noise_all[self.bg_[self.bg]]
                #mask nans: 
                noise_use = noise_use[~np.ma.masked_invalid(noise_use).mask]
                self.noise_model = np.random.choice(noise_use, self.l)
                print("noise shape", self.noise_model.shape)
                #save
                di = {'noise_model':self.noise_model}
                df = pd.DataFrame(di)
                df.to_csv(f_noise_run)

        self.flux_fake = np.zeros((self.n, (self.l))) # n l-length arrays
        self.error_fake = np.zeros((self.n, (self.l)))
        
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2

        for i in range(self.n):
            if self.rise_ == 1:
                t0, A, beta, B = self.params_true[i]
                t_ = self.x - t0
                self.flux_fake[i] = (np.heaviside((t_), 1) * A *np.nan_to_num((t_**beta))) + 1 + B
            
            elif self.rise_ == 2:
                t0, t1, A1, A2, beta1, beta2, B = self.params_true[i]
                self.flux_fake[i] = np.piecewise(self.x, [(t0 <= self.x)*(self.x < t1), t1 <= self.x], 
                                                 [func1, func2],
                                                 t0, t1, A1, A2, beta1, beta2) + 1 + B
            
            
            #attach noise: 
            self.flux_fake[i] = self.flux_fake[i] + self.noise_model
            # set error = expectation value of noise model
            self.error_fake[i] += np.mean(self.noise_model)
            
            if bg==0: #if no tess, add 1% mean flux error to all
                self.error_fake[i] += 0.01 * np.mean(self.flux_fake[i])
                    
        self.x += 2457000 #reset time axis because it will get subtracted
        #no need to save lc - always the same when generated
        return

    
    def plot_fake(self, index):
        
        #plt.scatter(self.x, self.flux_fake[index], color='k', s=0.5)
        plt.errorbar(self.x, self.flux_fake[index], 
                     self.error_fake[index], fmt='.k', markersize=0.5, label='data')
        plt.axvline(self.params_true[index][0]+2457000, color='r', label='t0')
        plt.axvline(self.disctimes[index], color='g', label='disc time')
        plt.xlabel("time")
        plt.ylabel('fake flux')
        plt.legend(loc='upper left', fontsize=16)
        plt.title("index:{}".format(index))
        plt.show()
        return
    
    def fit_fakes(self, start=0, n1=500, n2=5000):
        """ 
        Fit models to fake data starting at start index
        """
        
        #check to see if this has already been run: 
        final_file = f"{self.subfolder}/index{self.n-1}/"
        print(final_file)
        if os.path.exists(final_file):
            print("Fitting has already been run")
            return
        
        if self.rise_ == 1: 
            self.__fit_fakes_singlerise(start, n1, n2)
        elif self.rise_ == 2:
            self.__fit_fakes_doublerise(start, n1, n2)
        return 
        
        
    def __fit_fakes_singlerise(self, start=0, n1=500, n2=5000):  
        
        self.s_3 = np.zeros((self.n, self.dim))
        for i in range(start, self.n):
            dt = self.disctimes[i]
            self.trlc = etsMAIN(self.subfolder, 'nofile', plot=False)
        
            self.trlc.load_single_lc(self.x, self.flux_fake[i], self.error_fake[i], 
                                dt, "index{}".format(i), "", "", "")

            self.trlc.pre_run_clean(fitType=1)
            self.trlc.run_MCMC(n1, n2, quiet=True)
            self.s_3[i] = self.trlc.sampler.get_autocorr_time(tol=0)
            
        #save file:
        di = {'act_t':self.s_3[:,0],
              'act_a':self.s_3[:,1],
              'act_beta':self.s_3[:,2],
              'act_B':self.s_3[:,3]}
        df = pd.DataFrame(di)
        filename = "{s}{n}-allfluxes-{tl}-{r}-autocorr.csv".format(s=self.subfolder, 
                                              n=self.n,
                                              tl=self.targetlabel,
                                              r=self.rise_)
        df.to_csv(filename)
            
        return
    
    def __fit_fakes_doublerise(self, start=0, n1=500, n2=5000):  
        
        self.s_3 = np.zeros((self.n, self.dim))
        for i in range(start, self.n):
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
        filename = "{s}{n}-allfluxes-{tl}-{r}-autocorr.csv".format(s=self.subfolder, 
                                              n=self.n,
                                              tl=self.targetlabel,
                                              r=self.rise_)
        df.to_csv(filename)
        return
        

    def s_stats(self):
        """ 
        Calculating s-stats as given in the paper
        """
        max_err = np.maximum(self.upper_error.round(3), self.lower_error.round(3))
        #self.s_1 = np.abs(self.output_params.round(3) / max_err)
        self.s_1 = np.abs(max_err / self.output_params.round(3) )
        self.s_2 = np.abs(np.abs(self.output_params.round(3) - 
                          self.params_true.round(3))/self.params_true.round(3))
        
        # calc average converged/unconverged values
        good = np.nonzero(self.converged_retrieved)
        bad = np.nonzero((self.converged_retrieved - 1))
        
        converged_s1 = self.s_1[good]
        unconverged_s1 = self.s_1[bad]
        converged_s2 = self.s_2[good]
        unconverged_s2 = self.s_2[bad]
        
        converged_s3 = self.s_3.to_numpy()[good]
        unconverged_s3 = self.s_3.to_numpy()[bad]
        
        print(f"converged: s1: {converged_s1.mean(axis=0).round(4)} std {converged_s1.std(axis=0).round(4)}")
        print(f"converged: s2: {converged_s2.mean(axis=0).round(4)} std: {converged_s2.std(axis=0).round(4)}")
        print(f"converged: s3: {converged_s3.mean(axis=0).round(2)[1:]} std {converged_s3.std(axis=0).round(2)[1:]}")
        
        print(f"unconverged: s1: {unconverged_s1.mean(axis=0).round(4)} std {unconverged_s1.std(axis=0).round(4)}")
        print(f"unconverged: s2: {unconverged_s2.mean(axis=0).round(4)} std: {unconverged_s2.std(axis=0).round(4)}")
        print(f"unconverged: s3: {unconverged_s3.mean(axis=0).round(2)[1:]} std: {unconverged_s3.std(axis=0).round(2)[1:]}")
        
        
        
        return
    
    def chi_sq(self):
        """ 
        calc chisquared values
        """
        self.chi_squared = np.zeros(self.n) #n values
        x_ = self.x
        if self.x[0] != 0:
            x_ -= 2457000
        if self.rise_ == 1:
            for i in range(self.n): 
                t0, A, beta, B = self.output_params[i]
                t_ = x_ - t0
                mod = (np.heaviside((t_), 1) * A *np.nan_to_num((t_**beta))) + 1 + B
                self.chi_squared[i] = np.sum( ( mod - self.flux_fake[i] )**2 / self.flux_fake[i])
                
        elif self.rise_ == 2: 
            def func1(x, t0, t1, A1, A2, beta1, beta2):
                return A1 *(x-t0)**beta1
            def func2(x, t0, t1, A1, A2, beta1, beta2):
                return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
            for i in range(self.n): 
                t0, t1, A1, A2, beta1, beta2, B = self.output_params[i]
                mod = np.piecewise(x_, [(t0 <= x_)*(x_ < t1), t1 <= x_], 
                                                 [func1, func2],
                                                 t0, t1, A1, A2, beta1, beta2) + 1 + B
                self.chi_squared[i] = np.sum( ( mod - self.flux_fake[i] )**2 / self.flux_fake[i])
        
        return
    
    def plot_s_separate(self):
        """ 
        plot s1 s2 s3 separately
        """
        h = int(np.rint(self.dim/2))
        size = (10*(h/2),10)
        
        
        if not hasattr(self, 'noise_model'):
            self.__retrieve_noisemodel()

        c = ['black', 'red']
        
        mask_c = np.nonzero(self.converged_retrieved)
        mask_n = np.nonzero((self.converged_retrieved - 1))
        
        print(mask_c, mask_n)
        
        for j in range(3):
            fig, ax = plt.subplots(2, h, figsize=size)
            plotty = [self.s_1, self.s_2, self.s_3.to_numpy()]
            plab = [r'$S_1$',r'$S_2$', r'$S_3$']
            flab = ['S1', 'S2', 'S3']
            for i in range(self.dim):
                ax1 = ax[i%2,int(i/2)]
                ax1.scatter(self.params_true[mask_c][:,i], plotty[j][mask_c][:,i], 
                             color=c[0], s=12,
                            marker="<", label="Converged")
                ax1.scatter(self.params_true[mask_n][:,i], plotty[j][mask_n][:,i], 
                             color=c[1], s=12,
                            marker="<", label="Unconverged")
                ax1.set_xlabel(self.labels[i])
        
            #plot orbit gap line: 
            axy = ax[0,0]
            axy.axvline(self.orbit_gap[0], color='grey', linestyle='dashed', label="Orbit/Noise")
            axy.axvline(self.orbit_gap[1], color='grey', linestyle='dashed')
            axy.legend(fontsize=12)
            
            if self.dim == 7: #on doubles 
                axy = ax[1,0]
                axy.axvline(self.orbit_gap[0], color='grey', linestyle='dashed', label="Orbit/Noise")
                axy.axvline(self.orbit_gap[1], color='grey', linestyle='dashed')
                
                axy = ax[0,3] #plot B limits
                axy.axvline(self.noise_model.min(), color='grey', linestyle='dashed')
                axy.axvline(self.noise_model.max(), color='grey', linestyle='dashed')
                
            else: #single
                #plot noise limits on B:
                axy = ax[1,1]
                axy.axvline(self.noise_model.min(), color='grey', linestyle='dashed')
                axy.axvline(self.noise_model.max(), color='grey', linestyle='dashed')
             
            

            plt.suptitle(f"{plab[j]}; Background Model: {self.bg_[self.bg]}")
            plt.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig("{s}/{l}.png".format(s=self.subfolder, l=flab[j]))
            plt.show()
        
        
        return

    
    def plot_true_param_distros(self, bins=10): 
        
        if self.rise_ == 2:
        
            fig, ax = plt.subplots(1,3, figsize=(15, 5))
                
            ax[0].scatter(self.params_true[:,0], self.params_true[:,1], s=2, color='k')
            ax[0].set_xlabel(r'$t_0$')
            ax[0].set_ylabel(r'$t_1$')
                
            ax[1].scatter(self.params_true[:,2], self.params_true[:,3], s=2, color='k')
            ax[1].set_xlabel(r'$A_1$')
            ax[1].set_ylabel(r'$A_2$')
            
            ax[2].scatter(self.params_true[:,4], self.params_true[:,5], s=2, color='k')
            ax[2].set_xlabel(r'$\beta_1$')
            ax[2].set_ylabel(r'$\beta_2$')
            
            plt.suptitle("Double Power True Param Visualization")
            plt.tight_layout()
            plt.savefig("{s}/doublepower_paramviz.png".format(s=self.save_dir))
            return
        
        elif self.rise_ == 1:
            fig, ax = plt.subplots(2,2, figsize=(10, 10))
            for i in range(self.dim):
                axy = ax[i%2, int(i/2)]
                axy.hist(self.params_true[:,i], bins=bins,color='black', density=True)
                axy.set_xlabel(self.labels[i])
                
            plt.suptitle("Single Power True Param Visualization")
            plt.tight_layout()
            plt.savefig("{s}/singlepower_paramviz.png".format(s=self.save_dir))
        

    def retrieve_params(self, bg=0):
        """ 
        Load in the saved true params and the output params for comparison purposes
        
        """
        self.bg = bg
        #load in true: 
        f = "{s}true-params.csv".format(s=self.save_dir) 
        true_p = pd.read_csv(f)
        self.subfolder = "{s}{tl}/".format(s=self.save_dir, tl = self.bg_[bg])
        print(self.subfolder)
        
        f2 = "{s}{n}-allfluxes-{tl}-{r}-autocorr.csv".format(s=self.subfolder, 
                                              n=self.n,
                                              tl=self.bg_[bg],
                                              r=self.rise_)
        print(f2)
        self.s_3 = pd.read_csv(f2)
        
        # load in calculated params
        import etsfit.utils.batch_analyze as ba
        params_all = {}
        converged_all = {}
        upper_all = {}
        lower_all = {}

        
        for root, dirs, files in os.walk(self.subfolder):
            for name in files:
                if (name.endswith("-output-params.txt")):
                    targ = name.split("-")[0]
                    if targ[0] != "i":
                        continue #oops hit a noise model
                    
                    filepath = root + "/" + name
                    if self.rise_ == 1: 
                        (params,  upper_e, 
                         lower_e,  converg) = ba.extract_singlepower_all(filepath)
                    elif self.rise_ == 2:
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
        
        print("Convergence Rate:", np.sum(self.converged_retrieved)/len(self.converged_retrieved))

        return 
    
   
    
    def __retrieve_noisemodel(self):
        f_noise_run = "{s}{tl}-background.csv".format(s=self.subfolder, tl = self.targetlabel)
        if os.path.exists(f_noise_run):
            #load it
            print("loading existing noise model")
            self.noise_model = pd.read_csv(f_noise_run)['noise_model']
            print("noise shape", self.noise_model.shape)
        else:
            print("cannot load drawn noise model, file does not exist!")
    
    def print_bg_range(self):
        
        print(f"bg:{self.bg_[self.bg]}")
        print(f"min:{self.noise_model.to_numpy().min()}")
        print(f"max:{self.noise_model.to_numpy().max()}")
        print(f"range:{self.noise_model.to_numpy().max() - self.noise_model.to_numpy().min()}")
        
        plt.scatter(self.x, self.noise_model, color='black')
        plt.show()
        return
    
    def print_unconverged_params(self, array_indexes):
        """ 
        print true parameters of unconverged stuff: 
        """
        for i in range(len(array_indexes)):
            ind = array_indexes[i]
            print(f"{ind}: true: {self.params_true[ind].round(2)} estimated: {self.output_params[ind].round(2)}")
        return

    def rerun_unconverged_single(self, n1=5000, n2 = 150_000):
        """ 
        Run a long chain on unconverged items
        """
        if self.rise_ != 1:
            print("wrong function")
            return
        for i in range(self.n):
            if self.converged_retrieved == 0:
                dt = self.disctimes[i]
                self.trlc = etsMAIN(self.subfolder, 'nofile', plot=False)
                self.trlc.load_single_lc(self.x, self.flux_fake[i], self.error_fake[i], 
                                         dt, "index{}".format(i), "", "", "")
                
                self.trlc.pre_run_clean(fitType=1)
                self.trlc.run_MCMC(n1, n2, quiet=True)
                self.s_3[i] = self.trlc.sampler.get_autocorr_time(tol=0)
                
            
        #save file:
        di = {'act_t':self.s_3[:,0],
              'act_a':self.s_3[:,1],
              'act_beta':self.s_3[:,2],
              'act_B':self.s_3[:,3]}
        df = pd.DataFrame(di)
        filename = "{s}{n}-allfluxes-{tl}-{r}-autocorr.csv".format(s=self.subfolder, 
                                              n=self.n,
                                              tl=self.targetlabel,
                                              r=self.rise_)
        df.to_csv(filename)
        return 
    
    def rerun_unconverged_double(self, n1=5000, n2 = 150_000):
        """ 
        Run a long chain on unconverged items
        """
        
        if self.rise_ != 2:
            print("wrong function")
            return
        
        for i in range(self.n):
            if self.converged_retrieved == 0:
                dt = self.disctimes[i]
                self.trlc = etsMAIN(self.subfolder, 'nofile', plot=False)
                self.trlc.load_single_lc(self.x, self.flux_fake[i], self.error_fake[i], 
                                         dt, "index{}".format(i), "", "", "")
                
                self.trlc.pre_run_clean(fitType=3)
                self.trlc.run_MCMC(n1, n2, quiet=True)
                self.s_3[i] = self.trlc.sampler.get_autocorr_time(tol=0)
                
            
        #save file:
        di = {'act_t0':self.s_3[:,0],
              'act_t1':self.s_3[:,1],
              'act_A1':self.s_3[:,2],
              'act_A2':self.s_3[:,3],
              'act_beta1':self.s_3[:,4],
              'act_beta2':self.s_3[:,5],
              'act_B':self.s_3[:,6]}
        df = pd.DataFrame(di)
        filename = "{s}{n}-allfluxes-{tl}-{r}-autocorr.csv".format(s=self.subfolder, 
                                              n=self.n,
                                              tl=self.targetlabel,
                                              r=self.rise_)
        df.to_csv(filename)
        return 
    
    def megaplot(self):
        """ 
        plot the fake data  + retrieved models
        """
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        #10 plot figs
        n_plot = int(np.ceil(self.n/10))
        if self.x[0] != 0:
            x_ = self.x - 2457000
        else:
            x_ = self.x
        
        k = 0
        
        for i in range(n_plot): # i plots
            fig, ax = plt.subplots(5, 2, figsize=(10, 20))
            
            for j in range(10): #j subplots
                axy = ax[int(j/2),j%2]
                axy.axvline(self.dtimes[k]-2457000, color='blue', label='disc. time')
                
                
                axy.scatter(x_, self.flux_fake[k], s=2, color='black')
                #truth
                if self.rise_ == 1:
                    t0, A, beta, B = self.params_true[k]
                    t_ = x_ - t0
                    truemodel = (np.heaviside((t_), 1) * A *np.nan_to_num((t_**beta))) + 1 + B
                
                elif self.rise_ == 2:
                    t0, t1, A1, A2, beta1, beta2, B = self.params_true[k]
                    truemodel = np.piecewise(x_, [(t0 <= x_)*(x_ < t1), t1 <= x_], 
                                                     [func1, func2],
                                                     t0, t1, A1, A2, beta1, beta2) + 1 + B
                axy.plot(x_, truemodel, lw=2, color='blue')
                axy.axvline(t0, color='blue', label='t0')
                # retrieved
                if self.rise_ == 1:
                    t0, A, beta, B = self.output_params[k]
                    t_ = x_ - t0
                    model = (np.heaviside((t_), 1) * A *np.nan_to_num((t_**beta))) + 1 + B
                
                elif self.rise_ == 2:
                    t0, t1, A1, A2, beta1, beta2, B = self.output_params[k]
                    model = np.piecewise(x_, [(t0 <= x_)*(x_ < t1), t1 <= x_], 
                                                     [func1, func2],
                                                     t0, t1, A1, A2, beta1, beta2) + 1 + B
                axy.plot(x_, model, lw=2, color='red')
                k+=1 #update lc being plotted
                
            #save it
            plt.suptitle(f"plot {i}")
            plt.tight_layout()
            plt.savefig(f"{self.subfolder}/{i}-megaplot.png")
        
        

#%% running! 
save_dir = "./research/urop/fake_data/"

bg = 2
lc = artificial_lc(save_dir, 50, rise_=1)
lc.gen_params()
lc.gen_lc(bg=bg)
lc.plot_true_param_distros(bins=20)
lc.fit_fakes(start=0, n1=5000, n2=50_000)

lc.retrieve_params(bg=bg)

lc.megaplot() 
lc.s_stats()
lc.plot_s_separate()

print(f"unconverged indexes: {np.nonzero(lc.converged_retrieved-1)}")
lc.print_unconverged_params(np.nonzero(lc.converged_retrieved-1)[0])

lc.chi_sq()

#lc.rerun_unconverged_double()

