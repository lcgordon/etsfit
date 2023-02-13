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



class artificial_lc(object):
    
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        
        
    def generate_doublerise_fakes(self, n, tessreducefile, TNSFile, tess_background=True):
        """ 
        double power law artifiical light curves (multiple competing processes)
        n = # to make
        tessreducefile = Ia whose noise model ur gonna generate
        TNSFile = u know
        tess_backgrounds = true/false using them
        
        """
        
        print("Generating {} Double Power Artificial Light Curves".format(n))
        
        # make the noise model - even if you won't use it
        self.__tess_noise(tessreducefile, TNSFile)
        
        self.tag = "doublepower"

        self.save_dir = self.save_dir + "{n}-artificial-{tag}-tessbg-{tb}/".format(tag=self.tag,
                                                                              n = n, 
                                                                              tb = tess_background)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.n = n
        self.params_true = np.zeros((self.n, 7)) ##t0 t1 A1 A2 beta1 beta2 B
        
        # x axis with a fake orbit gap of size 1/10th array
        tenth = int(self.l / 10)
        start_ = int(self.l / 2) - int(tenth/2)
        end_ = start_ + tenth
        self.x = np.linspace(0, 28, (self.l+tenth))
        
        mask = np.ones(self.l+tenth)
        mask[start_:end_] = 0
        mask = np.nonzero(mask) # which ones you are keeping
        self.x = self.x[mask]
        
        self.flux_fake = np.zeros((self.n, (self.l))) # n l-length arrays
        self.error_fake = np.zeros((self.n, (self.l)))
        
        self.params_true[:,0] = np.random.uniform(1, 20, self.n) #t0
        self.params_true[:,1] = np.random.uniform(self.params_true[:,0], 25, self.n) #t1 defined by t0
        self.params_true[:,2] = np.random.uniform(0.001, 3, self.n) #A1
        self.params_true[:,3] = np.random.uniform(0.001, 3, self.n) #A2
        self.params_true[:,6] = np.random.uniform(-20, 20, self.n) #B
        # beta is pulled from a unif distro on the arctans
        self.params_true[:,4] = np.tan(np.random.uniform(np.arctan(0.5), np.arctan(4), self.n))
        self.params_true[:,5] = np.tan(np.random.uniform(np.arctan(0.5), np.arctan(4), self.n))
        
        
        def func1(x, t0, t1, A1, A2, beta1, beta2):
            return A1 *(x-t0)**beta1
        def func2(x, t0, t1, A1, A2, beta1, beta2):
            return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
        
        
        
        for i in range(self.n):
            t0, t1, A1, A2, beta1, beta2, B = self.params_true[i]
            self.flux_fake[i] = np.piecewise(self.x, [(t0 <= self.x)*(self.x < t1), t1 <= self.x], 
                                             [func1, func2],
                                             t0, t1, A1, A2, beta1, beta2) + 1 + B
            
            #attach noise: 
            if tess_background:
                self.flux_fake[i] = self.flux_fake[i] + self.noise_model
                # set error = expectation value of noise model
                self.error_fake[i] += np.mean(self.noise_model)
            else:
                self.error_fake[i] += 0.01 #uniform error

        
        self.x += 2457000 #reset time axis because it will get subtracted
        #we also assign an arbitrary discovery time as being 0.5-6 days post-t0. 
        #dict version required to run unfortch
        self.disctimes = {}
        #labels
        labels = list(range(self.n))
        self.dtimes = self.params_true[:,0] + np.random.uniform(0.5, 6, self.n) + 2457000
        for i in range(self.n):
            self.disctimes[labels[i]] = self.dtimes[i]
        
        self.__save_true_params(tess_background)
        self.__save_all_lc(tess_background)
        
        return
    
    def generate_singlerise_fakes(self, n, tessreducefile, TNSFile, tess_background=True):
        """ 
        
        single power laws 
        n = # to make
        tessreducefile = Ia whose noise model ur gonna generate
        TNSFile = u know
        tess_backgrounds = true/false using them
        
        """
        
        print("Generating {} two-Power Artificial Light Curves".format(n))
        
        # make the noise model - even if you won't use it
        self.__tess_noise(tessreducefile, TNSFile)
        self.tag = 'singlepower'
        
        self.save_dir = self.save_dir + "{n}-artificial-{tag}-tessbg-{tb}/".format(tag=self.tag,
                                                                              n = n, 
                                                                              tb = tess_background)
        
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.n = n
        self.params_true = np.zeros((self.n, 4)) ##t0 A beta B
        
        # x axis with a fake orbit gap of size 1/10th array
        tenth = int(self.l / 10)
        start_ = int(self.l / 2) - int(tenth/2)
        end_ = start_ + tenth
        self.x = np.linspace(0, 28, (self.l+tenth))
        
        mask = np.ones(self.l+tenth)
        mask[start_:end_] = 0
        mask = np.nonzero(mask) # which ones you are keeping
        self.x = self.x[mask]
        
        self.flux_fake = np.zeros((self.n, (self.l))) # n l-length arrays
        self.error_fake = np.zeros((self.n, (self.l)))
        
        self.params_true[:,0] = np.random.uniform(1, 20, self.n) #t0
        self.params_true[:,1] = np.random.uniform(0.001, 3, self.n) #A1
        self.params_true[:,3] = np.random.uniform(-20, 20, self.n) #B
        # beta is pulled from a unif distro on the arctans
        self.params_true[:,2] = np.tan(np.random.uniform(np.arctan(0.5), np.arctan(4), self.n))
        
        
        for i in range(self.n):
            t0, A, beta, B = self.params_true[i]
            t_ = self.x - t0
            self.flux_fake[i] = (np.heaviside((t_), 1) * A *np.nan_to_num((t_**beta))) + 1 + B
            
            #attach noise: 
            if tess_background:
                self.flux_fake[i] = self.flux_fake[i] + self.noise_model
                # set error = expectation value of noise model
                self.error_fake[i] += np.mean(self.noise_model)
            else:
                self.error_fake[i] += 0.01 #uniform error

        
        self.x += 2457000 #reset time axis because it will get subtracted
        #we also assign an arbitrary discovery time as being 0.5-6 days post-t0. 
        #dict version required to run unfortch
        self.disctimes = {}
        #labels
        labels = list(range(self.n))
        self.dtimes = self.params_true[:,0] + np.random.uniform(0.5, 6, self.n) + 2457000
        for i in range(self.n):
            self.disctimes[labels[i]] = self.dtimes[i]
        
        self.__save_true_params(tess_background)
        self.__save_all_lc(tess_background)
        
        return
    
    
    def __tess_noise(self, tessreducefile, TNSFile):
        """ 
        Generate a noise model to use for each target that you have
        """
        
        targetlabel = tessreducefile.split("/")[-1].split("-")[0]
        print(targetlabel)
        filename = self.save_dir + targetlabel + "-tessnoise.csv"
        
        if os.path.exists(filename):
            print("noise model exists, loading: ")
            read = pd.read_csv(filename)
            self.cut_flux = read['flux']
            self.l = int(read['orig length'][0])
        else: 
            print('no saved file, generating ')
            print("Making noise model from data (STOP FREAKING OUT ITS SUPPOSED TO RUN MCMC HERE)")
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
            #relocate to mean=0
            self.cut_flux -= np.mean(self.cut_flux)
            
            #trim to 3sigma
            from astropy.stats import SigmaClip
            sigclip = SigmaClip(sigma=3, maxiters=None, cenfunc='median')
            clipped_inds = np.nonzero(np.ma.getmask(sigclip(self.cut_flux)))
            self.cut_time = np.delete(self.cut_time, clipped_inds)
            self.cut_flux = np.delete(self.cut_flux, clipped_inds)
            
            #draw samples from that to fill in a full fake light curve
            self.l = len(time)
            
            #save file:
            di = {'flux':self.cut_flux, 
                  'orig length': (self.l +np.ones(len(self.cut_flux)))}
            df = pd.DataFrame(di)
            df.to_csv(filename)
            
         #make noise model:  
        self.noise_model = np.random.choice(self.cut_flux, self.l)
        return
    
    def __save_true_params(self, tess_background):
        """ 
        Put the real values into a file for access later
        """
        di = {'t0':self.params_true[:,0], 
              'A':self.params_true[:,1],
              'beta':self.params_true[:,2],
              'B':self.params_true[:,3], 
              'disc':self.dtimes}
        
        df = pd.DataFrame(di)
        df.to_csv("{s}{n}-{tag}-true-params-tessbg-{t}.csv".format(s=self.save_dir, n=self.n,
                                                            t=tess_background, tag=self.tag))
        return
    
    def __save_all_lc(self, tess_background):
        """ 
        Put fake LC intoa  file for later
        """
        di = {'t':self.x}
            
        for i in range(self.n):
            s_ = "{}".format(i)
            di[s_] = self.flux_fake[i].T
            
            
        df = pd.DataFrame(di)
        df.to_csv("{s}{n}-{tag}-artificial-flux-tessbg-{t}.csv".format(s=self.save_dir, n=self.n,
                                                            t=tess_background, tag=self.tag))
        
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

    def fit_fakes_1(self, start=0, n1=500, n2=5000):  
        
        self.output_params_1 = np.zeros((self.n, 4)) #always going to be t0, a , beta, b
        self.upper_error_1 = np.zeros((self.n, 4)) #upper error
        self.lower_error_1 = np.zeros((self.n, 4)) #lower error
        self.bic_1 = np.zeros((self.n, 1))
        #self.isright_1 = np.zeros((self.n, 4))
        
        for i in range(start, self.n):
            dt = self.disctimes[i]
            trlc = etsMAIN(self.save_dir, 'nofile')
        
            trlc.load_single_lc(self.x, self.flux_fake[i], self.error_fake[i], 
                                dt, 
                                "index{}".format(i), "00", "0", "0")
        
        
        
            trlc.pre_run_clean(fitType=1)
            #trlc.test_plot()
            trlc.run_MCMC(n1, n2, quiet=True)
            
            self.output_params_1[i] = trlc.best_mcmc[0]
            self.upper_error_1[i] = trlc.upper_error[0]
            self.lower_error_1[i] = trlc.lower_error[0]
            print(trlc.BIC)
            self.bic_1[i] = trlc.BIC
            
            #self.precision_accuracy_1()
        return
    
    def fit_fakes_celerite(self, start=0, n1=500, n2=5000):
        """ 
        fit to fake single powers using celerite model
        """
        self.output_params_cel = np.zeros((self.n, 4)) #always going to be t0, a , beta, b
        self.upper_error_cel = np.zeros((self.n, 4)) #upper error
        self.lower_error_cel = np.zeros((self.n, 4)) #lower error
        self.bic_cel = np.zeros((self.n, 1))
        
        for i in range(start, self.n):
            dt = self.disctimes[i]
        
            trlc = etsMAIN(self.save_dir, 'nofile')
            trlc.load_single_lc(self.x, self.flux_fake[i], self.error_fake[i], 
                                dt, "index{}".format(i), "00", "0", "0")
            
                
            trlc.pre_run_clean(11)
            
            trlc.run_GP_fit(n1=n1, n2=n2, gpUSE='celerite_residual', usebounds=True, 
                           custom_bounds=None, quiet=True)
        
            self.output_params_cel[i] = trlc.best_mcmc[0]
            self.upper_error_cel[i] = trlc.upper_error[0]
            self.lower_error_cel[i] = trlc.lower_error[0]
            #print(trlc.BIC)
            self.bic_cel[i] = trlc.BIC
        
        return

    def precision_accuracy(self, sigma=5):
        """ 
        Calculating precision + accuracy of output params
        
        Precision: error normalized by the true value
        
        Accuracy: error / true
        """
        
        self.accuracy = np.abs(self.output_params_1 - self.params_true)/self.params_true
        #self.precision = np.abs(self.output_params_1 - self.params_true)/self.params_true
        #self.precision = 0.5 * (np.abs(self.lower_error_1) + np.abs(self.upper_error_1))/self.params_true
        self.precision = np.abs(self.output_params_1 - self.params_true)/self.lower_error_1
        return
        

    def plot_accuracy_1(self):
        """ 
        plotting accuracies
        
        """
        fig, ax = plt.subplots(2, 2, figsize=(10,10))
        #x axis are the true param vals
        # y axis are the sigmas for each
        #color coded if in 3 sig
        labels = ["t0", "A", 'beta', 'B']
        for i in range(4):
            cm = np.where(lc.accuracy_sigma[:,i] <= 3, 1, 0)

            ax[int(i/2), i%2].scatter(lc.params_true[:,i], lc.accuracy_sigma[:,i], s=10, 
                                      cmap='RdYlBu', c=cm)
            ax[int(i/2), i%2].set_title(labels[i])
            ax[int(i/2), i%2].set_xlabel(labels[i])
            ax[int(i/2), i%2].set_ylabel('n sigma')
            ax[int(i/2), i%2].axhline(3, linestyle='dashed', color='black', lw=1)
            
        plt.tight_layout()
        plt.savefig("{d}sigma-accuracy1.png".format(d=self.save_dir))
        plt.close()
        
        fig, ax = plt.subplots(2, 2, figsize=(10,10))
        labels = ["t0", "A", 'beta', 'B']
        for i in range(4):
            cm = np.where(self.accuracy_sigma[:,i] <= 5, 1, 0)
            if sum(cm) ==10:
                cm = 'navy'

            ax[int(i/2), i%2].scatter(self.accuracy_sigma[:,i], 
                                      self.upper_error_1[:,i], s=10, 
                                      cmap='RdYlBu', c=cm)
            ax[int(i/2), i%2].set_title(labels[i])
            ax[int(i/2), i%2].set_ylabel(r"1-$\sigma$ Value")
            ax[int(i/2), i%2].set_xlabel('n-$\sigma$ accuracy')
            ax[int(i/2), i%2].axvline(5, linestyle='dashed', color='black', lw=1)
            
        plt.tight_layout()
        plt.savefig("{d}sigma-accuracy2.png".format(d=self.save_dir))
        plt.close()
        return
        

    def fit_fakes_3(self, start=0, n1=500, n2=5000):  
        
        self.output_params_3 = np.zeros((self.n, 7)) #always going to be 7 params
        self.upper_error_3 = np.zeros((self.n, 7)) #upper error
        self.lower_error_3 = np.zeros((self.n, 7)) #lower error
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
            
            self.output_params_3[i] = trlc.best_mcmc[0]
            self.upper_error_3[i] = trlc.upper_error[0]
            self.lower_error_3[i] = trlc.lower_error[0]
            self.bic_3[i] = trlc.BIC

        return
    
    def retrieve_calculated_params(self):
        import etsfit.utils.batch_analyze as ba
        params_all = {}
        converged_all = {}
        upper_all = {}
        lower_all = {}
        
        for root, dirs, files in os.walk(self.save_dir):
            for name in files:
                if (name.endswith("singlepower-output-params.txt")):
                    targ = name.split("-")[0][:-4]
                    
                    if targ[0] != "i":
                        continue #oops hit a noise model
                    
                    
                    filepath = root + "/" + name
                    (params,  upper_e, 
                     lower_e,  converg) = ba.extract_singlepower_all(filepath)
                    
                    params_all[targ] = params
                    upper_all[targ] = upper_e
                    lower_all[targ] = lower_e
                    converged_all[targ] = converg
             
        self.params_retrieved = np.zeros((len(params_all), 4))
        self.upper_error_retrieved = np.zeros((len(params_all), 4))
        self.lower_error_retrieved = np.zeros((len(params_all), 4))
        self.converged_retrieved = np.zeros(len(params_all))
        for i in range(len(params_all)):
            st_ = 'index{}'.format(i)
            self.params_retrieved[i] = params_all[st_]
            self.upper_error_retrieved[i] = upper_all[st_]
            self.lower_error_retrieved[i] = lower_all[st_]
            self.converged_retrieved[i] = converged_all[st_]
        return 
    
        
   
 #%%       
TNSFile = "/Users/lindseygordon/research/urop/august2022crossmatch/tesscut-Ia18th.csv"           
tessreducefile = "/Users/lindseygordon/research/urop/tessreduce_lc/2020tld2921/2020tld2921-tessreduce"        
 
lc = artificial_lc("./research/urop/fake_data/")
lc.generate_singlerise_fakes(1_000, tessreducefile, TNSFile, False)
#lc.plot_fake(1)

lc.fit_fakes_1(n1=5_000, n2=70_000)
#lc.precision_accuracy()


#%%
acc = np.abs(lc.output_params_1 - lc.params_true)/lc.params_true
prec = np.abs(lc.output_params_1 - lc.params_true)/lc.lower_error_1


fig, ax = plt.subplots(1,1, figsize=(5,5))
labels = ['t0', 'A', 'beta', 'B']
for i in range(4):
    ax.scatter(lc.params_true[:,i], acc[:,i], label=labels[i])

ax.set_xlabel('true param')
ax.set_ylabel('accuracy')
ax.set_title("closer to 0 = closer to true")
    
ax.legend()
plt.show()
plt.close()

fig, ax = plt.subplots(1,1, figsize=(5,5))
labels = ['t0', 'A', 'beta', 'B']
for i in range(4):
    ax.scatter(lc.params_true[:,i], prec[:,i], label=labels[i])

ax.set_xlabel('true param')
ax.set_ylabel('precision')
    
ax.legend()
plt.show()
plt.close()

#%% can reload calculated params using batch_analyze: 
    
#saving into csv:

df
