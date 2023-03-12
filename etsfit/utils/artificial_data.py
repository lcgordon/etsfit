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
        Init and produce the subfolder for this n and rise
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
        
            if self.rise_== 1:
                self.params_true = np.zeros((self.n, 4)) ##t0 A beta B
                self.params_true[:,0] = np.random.uniform(5, 20, self.n) #t0
                self.params_true[:,1] = np.random.uniform(0.001, 3, self.n) #A1
                # beta is pulled from a unif distro on the arctans
                self.params_true[:,2] = np.tan(np.random.uniform(np.arctan(0.5), np.arctan(4), self.n))
                self.params_true[:,3] = np.random.uniform(1, 20, self.n) *  np.random.choice((-1,1), self.n)
                #B can't get to close to 0 or summary stats look like trash
                
                self.disctimes = {}
                labels = list(range(self.n))
                self.dtimes = self.params_true[:,0] + np.random.uniform(0.5, 6, self.n) + 2457000
                for i in range(self.n):
                    self.disctimes[labels[i]] = self.dtimes[i]
                
            elif self.rise_== 2:
                self.params_true = np.zeros((self.n, 7)) ##t0 A beta B
                self.params_true[:,0] = np.random.uniform(1, 20, self.n) #t0
                self.params_true[:,1] = np.random.uniform(self.params_true[:,0], 25, self.n) #t1 defined by t0
                self.params_true[:,2] = np.random.uniform(0.001, 3, self.n) #A1
                self.params_true[:,3] = np.random.uniform(0.001, 3, self.n) #A2
                self.params_true[:,6] = np.random.uniform(1, 20, self.n) *  np.random.choice((-1,1), self.n) #B
                # beta is pulled from a unif distro on the arctans
                self.params_true[:,4] = np.tan(np.random.uniform(np.arctan(0.5), np.arctan(4), self.n))
                self.params_true[:,5] = np.tan(np.random.uniform(np.arctan(0.5), np.arctan(4), self.n))
                
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
        di = {'t0':self.params_true[:,0], 
              'A':self.params_true[:,1],
              'beta':self.params_true[:,2],
              'B':self.params_true[:,3], 
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

    def fit_fakes_type1(self, start=0, n1=500, n2=5000):  
        
        self.s_3 = np.zeros((self.n, 4))
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
        
            trlc = etsMAIN(self.subfolder, 'nofile')
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

    def s_stats(self):
        """ 
        Calculating s-stats as given in the paper
        """
        max_err = np.maximum(self.upper_error.round(3), self.lower_error.round(3))
        #self.s_1 = np.abs(self.output_params.round(3) / max_err)
        self.s_1 = np.abs(max_err / self.output_params.round(3) )
        self.s_2 = np.abs(np.abs(self.output_params.round(3) - 
                          self.params_true.round(3))/self.params_true.round(3))
        
        return
        

    def plot_s12(self):
        """ 
        plot the precision and accuracy for the loaded values
        """
        
        
        fig, ax = plt.subplots(2,2, figsize=(10,10))
        
        if not hasattr(self, 'noise_model'):
            self.__retrieve_noisemodel()


        labels = [r'$t_0$', 'A', r'$\beta$', 'B']
        c = ['black', 'red']
        for i in range(4):
            ax1 = ax[int(i/2), i%2]
            
            ax1.scatter(self.params_true[:,i], (self.s_1*100)[:,i], 
                         color=c[0], s=12,
                        marker="<", label=r'$S_1$')
            #ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax1.set_xlabel(labels[i])
            
            ax2 = ax1.twinx()
            #ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax2.scatter(self.params_true[:,i], (self.s_2*100)[:,i], 
                        label = r'$S_2$', color=c[1], s=12, marker="x")
            ax2.tick_params(axis='y', labelcolor=c[1])
            
            lines, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels1 + labels2, loc=0, fontsize=16)

        #plot orbit gap linse: 
        axy = ax[0,0]
        axy.axvline(self.orbit_gap[0], color='grey', linestyle='dashed')
        axy.axvline(self.orbit_gap[1], color='grey', linestyle='dashed')
        
        #plot noise limits on B:
        axy = ax[1,1]
        axy.axvline(self.noise_model.min(), color='grey', linestyle='dashed')
        axy.axvline(self.noise_model.max(), color='grey', linestyle='dashed')
            

        plt.suptitle(r"$S_1$ & $S_2$; Background Model: "+ self.bg_[self.bg])
        plt.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig("{s}/s1s2.png".format(s=self.subfolder))
        plt.show()
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
    
    def retrieve_params(self, bg=0):
        """ 
        Load in the saved true params and the output params for comparison purposes
        
        """
        self.bg = bg
        #load in true: 
        f = "{s}true-params.csv".format(s=self.save_dir) 
        true_p = pd.read_csv(f)
        self.subfolder = "{s}{tl}/".format(s=self.save_dir, tl = self.bg_[bg])
        #print(subfolder)
        
        f2 = "{s}{n}-allfluxes-{tl}-{r}-autocorr.csv".format(s=self.subfolder, 
                                              n=self.n,
                                              tl=self.bg_[bg],
                                              r=self.rise_)
        self.s_3_ = pd.read_csv(f2)
        
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
                    
                    params_all[targ] = params
                    upper_all[targ] = upper_e
                    lower_all[targ] = lower_e
                    converged_all[targ] = converg
        
        #print(params_all)
        # dicts into arrays: 
        p_ =  len(params_all)
        if self.rise_ == 1:
            self.output_params = np.zeros((p_, 4))
            self.upper_error = np.zeros((p_, 4))
            self.lower_error = np.zeros((p_, 4))
            self.converged_retrieved = np.zeros(p_)
            for i in range(p_):
                st_ = 'index{}'.format(i)
                self.output_params[i] = params_all[st_]
                self.upper_error[i] = upper_all[st_]
                self.lower_error[i] = lower_all[st_]
                self.converged_retrieved[i] = converged_all[st_]
             
            self.params_true = true_p.to_numpy()[:,1:5]

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
    


#%%
save_dir = "./research/urop/fake_data/"


lc = artificial_lc(save_dir, 5, rise_=1)
lc.gen_params()
lc.gen_lc(bg=0)
lc.fit_fakes_type1(n1=1000, n2=50_000)
lc.retrieve_params(bg=0)

lc.s_stats()
lc.plot_s12()

#%%
import tessreduce as tr
def mag_plot_download(i, df):
    sec = int(df["Sector"].iloc[i])
    #print(i)
    #time.sleep(40)
    print(df['Name'].iloc[i][3:])
    targ = df['Name'].iloc[i][3:]
    
    try:
        obs = tr.sn_lookup(targ)
        lookup = obs[np.where(np.asarray(obs)[:,2] == sec)[0][0]]
        tess = tr.tessreduce(obs_list=lookup,plot=False,reduce=True)
        
    except ValueError:   
        print("value error - something is wrong with vizier or no target in pixels")
     
    except IndexError:
        print("index error - tesscut thinks it wasn't observed")
    
    except ConnectionResetError:
        print("vizier problems again")
    except TimeoutError:
        print("vizier problems")
    
    except ConnectionError:
        print("more! vizier! problems!")
    
    cdir = "/Users/lindseygordon/.lightkurve-cache/tesscut/"
    holder = ""
    for root, dirs, files in os.walk(cdir):
        for name in files:
            holder = root + "/" + name
            print(holder)
            try:
                filenamepieces = name.split("-")
                sector = str( filenamepieces[1][3:])
                camera = str( filenamepieces[2])
                ccd = str(filenamepieces[3][0])
                os.remove(holder)
                break
            except IndexError:
                print("eek")
                os.remove(holder)
                continue
    #print(sector, camera, ccd)
    
    data_dir = "/Users/lindseygordon/research/urop/tessreduce_lc/"
    
    targlabel = targ + sector + camera + ccd 
    newfolder = data_dir + targlabel + "/"
    if not os.path.exists(newfolder):
        os.mkdir(newfolder)
        filesave = newfolder + targlabel + "-tessreduce.csv"
        tess.save_lc(filesave)
        tess.to_flux()
        filesave = newfolder + targlabel + "-tessreduce-fluxconverted.csv"
        tess.save_lc(filesave)

    #make subfolder to save into 
    targlabel = targ + sector + camera + ccd 
    l_mag = tess.to_mag()
    time = l_mag[0]
    mag = l_mag[1]

    # from astropy.stats import SigmaClip
    # sigclip = SigmaClip(sigma=3, maxiters=None, cenfunc='median')
    
    
    m = int(len(time)/2 ) - 40 
    
    # region = mag[0:m ]
    # time = time[0:m]
    # mask = region != np.inf
    
    # time = time[mask]
    # mag = region[mask]
    
    
    # clipped_inds = np.nonzero(np.ma.getmask(sigclip(mag)))
    # time = np.delete(time, clipped_inds)
    # mag = np.delete(mag, clipped_inds)
    
    
    fig, ax = plt.subplots(1,1, figsize=(8,3))
    from astropy.time import Time
    time = Time(time, format='mjd').jd
    ddate = Time(df["Discovery Date (UT)"].iloc[i]).jd
    #print(ddate)
    
    ax.scatter(time, mag)
    ax.invert_yaxis()
    ax.axvline(ddate, color='green')
    ax.axvline(time[m], color='red')
    ax.set_title(targlabel)
    
    
    print(targlabel, "Mean Mag:", np.ma.masked_invalid(mag[0:m]).mean())
    return tess, ddate

# df = pd.read_csv("/Users/lindseygordon/research/urop/august2022crossmatch/all-tesscut-matches.csv")
data_dir = "/Users/lindseygordon/research/urop/tessreduce_lc/"
file_TNS = "/Users/lindseygordon/research/urop/august2022crossmatch/all-tesscut-matches.csv"

df = pd.read_csv(file_TNS)

candidate_backgrounds = ["2018fhw", "2020vem", #"2019esa", "2020aakp", 
                         "2020tld"]

i = 2
ind = 0
df2 = df[df["Name"].str.contains(candidate_backgrounds[i])]
#print(df2)
tess_, dd_ = mag_plot_download(ind, df2)

#%%
l_mag = tess_.to_mag()
time = l_mag[0]
mag = l_mag[1]


fig, ax = plt.subplots(1,1, figsize=(8,3))
from astropy.time import Time
time = Time(time, format='mjd').jd
ddate = Time(df2["Discovery Date (UT)"].iloc[ind]).jd
#print(ddate)
m = int(len(time)/2 ) - 30
ax.scatter(time, mag, color='black')
ax.invert_yaxis()
ax.axvline(ddate, color='green')
ax.axvline(time[m], color='red')
#ax.set_title(targlabel)
print("Mean Mag:", np.ma.masked_invalid(mag[0:m]).mean())
     
dt = Time(time[1], format='jd') - Time(time[0], format='jd')
print(dt.sec)

if dt.sec > 1700:
    print("in the okay cadence! ")
else: 
    print("need to bin this ")
    n_pts = int(np.rint(1800/dt.sec)) #3 ten minutes = 30 minutes
    t_ = []
    f_ = []
    
    n = int(0)
    m = int(n_pts+1)
    
    while m<len(time):
        t_.append(time[n])
        rang = mag[n:m][~np.ma.masked_invalid(mag[n:m]).mask]
        if len(rang) == 0:
            f_.append(np.nan)
        else: 
            f_.append(np.nanmean(rang))
    
        n+= n_pts
        m+= n_pts  
 
    time = np.asarray(t_)
    mag = np.asarray(f_)

    m = int(len(time)/2 ) - 10

ax.scatter(time, mag, color='blue')

mag2 = mag[0:m]
mag2 = mag2[~np.ma.masked_invalid(mag[0:m]).mask]
t2 = time[0:m]
t2 = t2[~np.ma.masked_invalid(mag[0:m]).mask]
ax.scatter(t2, mag2, color='blue')





from astropy.stats import SigmaClip
sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')

mask = np.ma.getmask(sigclip(mag2))
#print("masking: ", len(mask))
mag3 = mag2[~mask]
print("masking", len(mag3) - len(mag2))
t3 = t2[~mask]
ax.scatter(t3, mag3, color='green')

print("mean mag post cleaning: ", mag3.mean())
print("m:", m, "time", t3[-1]-2457000)
print("range:", mag3.max() - mag3.min())