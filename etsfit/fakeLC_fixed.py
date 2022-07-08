# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:40:19 2022

@author: conta
"""
#%%
from etsfit import etsMAIN
import numpy as np
#creating fake light curves for all of the different kinds of fits

def fake_type1(n):
    """Makes n 1000 point fake light curves with the simplest form of the model """

    import matplotlib.pyplot as plt
    timeaxis = np.arange(0,25,0.025) #25 days of data, 1000 data points (p typical of TESS)
    
    paramsall = np.ones((n,4)) #n light curves, 4 parameters
    intall = np.ones((n,1000)) #n light curves, 1000 data points
    noiseall = np.ones((n,1000)) #n light curves, 1000 data points
    
    for n in range(n):
        t0 = np.random.uniform(2,15)
        A = np.random.uniform(0.0001,0.002)
        beta = np.random.uniform(0.5,4)
        B = np.random.uniform(-0.2,0.2)
        t1 = timeaxis - t0
        model = (np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta))) + 1 + B
        #noise = np.random.uniform(-0.02,0.02, 1000)
        noise = np.random.normal(0,0.5*A, len(model)) #noise cannot be more than 50% of A
        paramsall[n] = [t0,A,beta,B]
        noiseall[n] = noise
        intall[n] = model+noise   
        plt.scatter(timeaxis, intall[n])
        plt.axvline(t0)
        
        #print(beta)
    plt.show()
    plt.close()
    return timeaxis, paramsall, intall, noiseall

t,p, i,n = fake_type1(100)

#%%
#and then feed these in as fake light curves into etsfit
folderSAVE = "/data/uchu/lcgordon/20220707/FakeLCOutput-10000/"
CBV_folder = "/data/uchu/lcgordon/FakeLCOutput/"
quaternion_folder_raw = "string"
quaternion_folder_txt = "string"

bigInfoFile = "/data/uchu/lcgordon/etsfit/tutorials/data/18thmag_Ia.csv"

#random discovery times to use
fakedisctimes = np.random.uniform(1,5,5)

def fit_all_fakes_1(t,i,n,p, fakedisctimes, folderSAVE):
    
    output_onesigma = np.zeros(len(i)) #outputs true/false
    output_twosigma = np.zeros(len(i)) #outputs true/false
    
    for k in range(len(i)): #for each input intensity array
        print("running...", str(k)) #run on it
        etsfakes = etsMAIN(folderSAVE, bigInfoFile, CBV_folder,
                         quaternion_folder_raw, quaternion_folder_txt)
        
        fakelygosbg = np.zeros(len(t)) #this is just to trick the custom input fxn
        disctime = p[k][0] + fakedisctimes[k] #set up fake discovery time
        #(t0 never more than 15) so this never puts it beyond the x axis values
        tag = "Fake-LC-Type1-" + str(k) + "-" #tag for the thingy
        #make the etsfake object
        etsfakes.load_custom_lc(t, i[k], n[k], fakelygosbg, disctime, tag,
                                0, 0, 0)
        #run the etsfake object
        best, upperE, lowerE, bic = etsfakes.run_MCMC(1, None, None, fraction=None, 
                                                      n1=5000, n2=10000)

        #check if within bounds of truth
        onesig = 0
        twosig = 0
        for h in range(4):
            print(h)
            print(best[0])
            print(p[k])
            bestie = best[0][h]
            errorU = upperE[0][h]
            errorL = lowerE[0][h]
            
            #print(p[0][i], bestie, errorU, errorL)
            if (bestie-errorL < p[k][h] < bestie+errorU):
                print("parameter within onesigma")
                onesig = 1 #falls in limits
            else:
                onesig = 0
                
            if (bestie-(2*errorL) < p[k][h] < bestie+(2*errorU)):
                print("parameter within twosigma")
                twosig = 1 #set output to having been within the limits
            else:
                twosig = 0 #output is not within limits of what it should be
        
        output_onesigma[k] = onesig
        output_twosigma[k] = twosig
        trueparamsfile = folderSAVE+tag[:-1] + "-000/" + etsfakes.filesavetag[1:]+"/" +"trueparams.txt"
        with open(trueparamsfile, 'w') as file:
            file.write(str(p[k]))
        
    return output_onesigma,output_twosigma
            
output_onesigma, output_twosigma = fit_all_fakes_1(t,i,n,p, fakedisctimes, folderSAVE)

saveresults = "/data/uchu/lcgordon/20220707/FakeLCOutput-10000/all-results-10000.txt"
with open(saveresults, 'w') as file:
    file.write(str(output_onesigma))
    file.write("\n")
    file.write(str(output_twosigma))
    
#%%
#re-run only the ones that didn't two sigma converge at 40,000 step chain
import gc
folderSAVE = "/data/uchu/lcgordon/20220707/FakeLCOutput-20000/"
CBV_folder = "/data/uchu/lcgordon/FakeLCOutput-20000/"
quaternion_folder_raw = "string"
quaternion_folder_txt = "string"

bigInfoFile = "/data/uchu/lcgordon/etsfit/tutorials/data/18thmag_Ia.csv"

def fit_some_fakes_1(t,i,n,p, disctimes, output_inputtwosigma, folderSAVE):
    
    output_onesigma = np.zeros(len(i))
    output_twosigma = np.zeros(len(i))
    errorbar_size = np.zeros(len(i))
    
    for k in range(len(i)):
        if (output_inputtwosigma[k] == 1):
            output_twosigma[k] = 1
            output_onesigma[k] = 1
            print("skipping...", str(k))
            continue #skip the good ones
        print("running...", str(k))
        etsfakes = etsMAIN(folderSAVE, bigInfoFile, CBV_folder,
                         quaternion_folder_raw, quaternion_folder_txt)
        
        fakelygosbg = np.zeros(len(t))
        disctime = p[k][0] + disctimes[k]
        #fake disc time is t0 plus some random offset up to 5 days 
        #(t0 never more than 15) so this never puts it beyond the x axis values
        tag = "Fake-LC-Type1-" + str(k) + "-"
        etsfakes.load_custom_lc(t, i[k], n[k], fakelygosbg, disctime, tag,
                                0, 0, 0)
        
        
        best, upperE, lowerE, bic = etsfakes.run_MCMC(1, None, None, fraction=None, 
                                                      n1=5000, n2=20000)

        onesig = 0
        twosig = 0
        for h in range(4):
            print(h)
            print(best[0][h])
            print(upperE[0][h])
            print(lowerE[0][h])
            bestie = best[0][h]
            errorU = upperE[0][h]
            errorL = lowerE[0][h]
            
            #print(p[0][i], bestie, errorU, errorL)
            if (bestie-errorL < p[k][h] < bestie+errorU):
                print("parameter within onesigma")
                onesig = 1 #falls in limits
            else:
                onesig = 0
                
            if (bestie-(2*errorL) < p[k][h] < bestie+(2*errorU)):
                print("parameter within twosigma")
                twosig = 1 #set output to having been within the limits
            else:
                twosig = 0 #output is not within limits of what it should be
            

        output_onesigma[k] = onesig
        output_twosigma[k] = twosig
            
        trueparamsfile = folderSAVE+tag[:-1] + "-000/" + etsfakes.filesavetag[1:]+"/" +"trueparams.txt"
        with open(trueparamsfile, 'w') as file:
            file.write(str(p[k]))
        del(etsfakes)
        gc.collect()
        
    return output_onesigma,output_twosigma, errorbar_size
            
output_onesigma, output_twosigma, errorbar_size = fit_some_fakes_1(t,i,n,p,fakedisctimes,
                                                                   output_twosigma, 
                                                                   folderSAVE)

saveresults = "/data/uchu/lcgordon/20220707/FakeLCOutput-20000/all-results-20000.txt"
with open(saveresults, 'w') as file:
    file.write(str(output_onesigma))
    file.write("\n")
    file.write(str(output_twosigma))
    file.write("\n")
    file.write(str(errorbar_size))