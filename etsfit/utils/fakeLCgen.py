# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 12:39:41 2022

@author: conta

Generating artificial light curves for testing

"""
import numpy as np
import matplotlib.pyplot as plt
from etsfit import etsMAIN

def fake_type1(n):
    """Makes n 1000 point fake light curves with the first model type"""

    import matplotlib.pyplot as plt
    timeaxis = np.arange(0,25,0.025) #25 days of data, 1000 data points (p typical of TESS)
    
    paramsall = np.ones((n,4)) #n light curves, 4 parameters
    intall = np.ones((n,1000)) #n light curves, 1000 data points
    noiseall = np.ones((n,1000)) #n light curves, 1000 data points
    t0 = np.random.uniform(2,15,n)
    A = np.random.uniform(0.0001,0.002,n)
    beta = np.random.uniform(0.5,4, n)
    B = np.random.uniform(-0.2,0.2, n)
    #noise = np.random.normal(0,0.5*A, (n, 1000)) #noise cannot be more than 50% of A
    
    for i in range(n):
        noiseall[i] = np.random.normal(0,0.5*A[i], 1000)
        t1 = timeaxis-t0[i]
        intall[i] = (np.heaviside((t1), 1) * A[i] *np.nan_to_num((t1**beta[i]))) + 1 + B[i]
    
    paramsall[:,0] = t0
    paramsall[:,1] = A
    paramsall[:,2] = beta
    paramsall[:,3] = B
    
    return timeaxis, paramsall, intall+noiseall, noiseall

def fit_all_fakes_1(time,intensity,error,trueparams, fakedisctimes, 
                    folderSAVE, customtag=None,saveresults=None):
    """Fits all fake type 1 light curves """
    
    output_onesigma = np.zeros(len(intensity)) #outputs true/false
    output_twosigma = np.zeros(len(intensity)) #outputs true/false
    
    for k in range(len(intensity)): #for each input intensity array
        print("running...", str(k)) #run on it
        etsfakes = etsMAIN(folderSAVE, None, folderSAVE,folderSAVE, folderSAVE)
        
        fakelygosbg = np.zeros(len(time)) #this is just to trick the custom input fxn
        disctime = trueparams[k][0] + fakedisctimes[k] #set up fake discovery time
        #(t0 never more than 15) so this never puts it beyond the x axis values
        if customtag is None:
            tag = "Fake-LC-Type1-" + str(k) + "-" #tag for the thingy
        else:
            tag = customtag
        #make the etsfake object
        etsfakes.load_custom_lc(time, intensity[k], error[k], fakelygosbg, disctime, tag,
                                0, 0, 0)
        #run the etsfake object
        best, upperE, lowerE, bic = etsfakes.run_MCMC(1, None, None, fraction=None, 
                                                      n1=5000, n2=10000)

        #check if within bounds of truth
        onesig = 0
        twosig = 0
        for h in range(4):
            if (best[0][h]-lowerE[0][h] < trueparams[k][h] < best[0][h]+upperE[0][h]):
                print("parameter within onesigma")
                onesig = 1 #falls in limits
            else:
                onesig = 0
                
            if (best[0][h]-(2*lowerE[0][h]) < trueparams[k][h] < best[0][h]+(2*upperE[0][h])):
                print("parameter within twosigma")
                twosig = 1 #set output to having been within the limits
            else:
                twosig = 0 #output is not within limits of what it should be
        
        output_onesigma[k] = onesig
        output_twosigma[k] = twosig
        trueparamsfile = folderSAVE+tag[:-1] + "-000/" + etsfakes.filesavetag[1:]+"/" +"trueparams.txt"
        with open(trueparamsfile, 'w') as file:
            file.write(str(trueparams[k]))
    if saveresults is not None:
        with open(saveresults, 'w') as file:
            file.write(str(output_onesigma))
            file.write("\n")
            file.write(str(output_twosigma))
        
    return output_onesigma,output_twosigma


# =============================================================================
# def fake_type2(n, CBVALL):
#     """Makes n 1000 point fake light curves with the second model type """
# 
#     import matplotlib.pyplot as plt
#     timeaxis = np.arange(0,25,0.025) #25 days of data, 1000 data points (p typical of TESS)
#     
#     paramsall = np.ones((n,4)) #n light curves, 4 parameters
#     intall = np.ones((n,1000)) #n light curves, 1000 data points
#     noiseall = np.ones((n,1000)) #n light curves, 1000 data points
#     t0 = np.random.uniform(2,15,n)
#     A = np.random.uniform(0.0001,0.002,n)
#     beta = np.random.uniform(0.5,4, n)
#     B = np.random.uniform(-0.2,0.2, n)
#     cQ = np.random.uniform(-2,2,n)
#     c1 = np.random.uniform(-2,2,n)
#     c2 = np.random.uniform(-2,2,n)
#     c3 = np.random.uniform(-2,2,n)
#     Qall, CBV1, CBV2, CBV3 = CBVALL
#     
#     for i in range(n):
#         noiseall[i] = np.random.normal(0,0.5*A[i], 1000)
#         t1 = timeaxis-t0[i]
#         intall[i] = (np.heaviside((t1), 1) * A[i] *np.nan_to_num((t1**beta[i]))) + 1 + B[i]
#         intall[i] += cQ * Qall + c1 * CBV1 + c2 * CBV2 + c3 * CBV3
#     
#     paramsall[:,0] = t0
#     paramsall[:,1] = A
#     paramsall[:,2] = beta
#     paramsall[:,3] = B
#     paramsall[:,4] = cQ
#     paramsall[:,5] = c1
#     paramsall[:,6] = c2
#     paramsall[:,7] = c3
#     
#     return timeaxis, paramsall, intall+noiseall, noiseall
# 
# =============================================================================
def fake_type3(n):
    """Makes n 1000 point fake light curves with the first model type"""

    import matplotlib.pyplot as plt
    timeaxis = np.arange(0,25,0.025) #25 days of data, 1000 data points (p typical of TESS)
    
    paramsall = np.ones((n,7)) #n light curves, 7 parameters
    intall = np.ones((n,1000)) #n light curves, 1000 data points
    noiseall = np.ones((n,1000)) #n light curves, 1000 data points
    t0 = np.random.uniform(2,15,n)
    t1 = np.random.uniform(t0,17,n)
    A1 = np.random.uniform(0.0001,0.002,n)
    A2 = np.random.uniform(0.0001,0.002,n)
    beta1 = np.random.uniform(0.5,4, n)
    beta2 = np.random.uniform(0.5,4, n)
    B = np.random.uniform(-0.2,0.2, n)
    #noise = np.random.normal(0,0.5*A, (n, 1000)) #noise cannot be more than 50% of A
    
    paramsall[:,0] = t0
    paramsall[:,1] = t1
    paramsall[:,2] = A1
    paramsall[:,3] = A2
    paramsall[:,4] = beta1
    paramsall[:,5] = beta2
    paramsall[:,6] = B
    
    def func1(x, t0, t1, A1, A2, beta1, beta2):
        return A1 *(x-t0)**beta1
    def func2(x, t0, t1, A1, A2, beta1, beta2):
        return A1 * (x-t0)**beta1 + A2 * (x-t1)**beta2
    
    for i in range(n):
        noiseall[i] = np.random.normal(0,0.5*A1[i], 1000)

        model = np.piecewise(timeaxis, [(t0[i] <= timeaxis)*(timeaxis < t1[i]), 
                            t1[i] <= timeaxis], [func1, func2],
                             t0[i], t1[i], A1[i], A2[i], beta1[i], beta2[i]) + 1 + B[i]
        
        intall[i] = model + noiseall[i]
        
    
    return timeaxis, paramsall, intall, noiseall

def fit_all_fakes_3(time,intensity,error,trueparams, fakedisctimes, 
                    folderSAVE, customtag=None,saveresults=None):
    """Fits all fake type 1 light curves """
    
    output_onesigma = np.zeros(len(intensity)) #outputs true/false
    output_twosigma = np.zeros(len(intensity)) #outputs true/false
    
    for k in range(len(intensity)): #for each input intensity array
        print("running...", str(k)) #run on it
        etsfakes = etsMAIN(folderSAVE, None, folderSAVE,folderSAVE, folderSAVE)
        
        fakelygosbg = np.zeros(len(time)) #this is just to trick the custom input fxn
        disctime = trueparams[k][0] + fakedisctimes[k] #set up fake discovery time
        #(t0 never more than 15) so this never puts it beyond the x axis values
        if customtag is None:
            tag = "Fake-LC-Type1-" + str(k) + "-" #tag for the thingy
        else:
            tag = customtag
        #make the etsfake object
        etsfakes.load_custom_lc(time, intensity[k], error[k], fakelygosbg, disctime, tag,
                                0, 0, 0)
        #run the etsfake object
        best, upperE, lowerE, bic = etsfakes.run_MCMC(3, None, None, fraction=None, 
                                                      n1=5000, n2=10000)

        #check if within bounds of truth
        onesig = 0
        twosig = 0
        for h in range(len(trueparams[0])):
            if (best[0][h]-lowerE[0][h] < trueparams[k][h] < best[0][h]+upperE[0][h]):
                print("parameter within onesigma")
                onesig = 1 #falls in limits
            else:
                onesig = 0
                
            if (best[0][h]-(2*lowerE[0][h]) < trueparams[k][h] < best[0][h]+(2*upperE[0][h])):
                print("parameter within twosigma")
                twosig = 1 #set output to having been within the limits
            else:
                twosig = 0 #output is not within limits of what it should be
        
        output_onesigma[k] = onesig
        output_twosigma[k] = twosig
        trueparamsfile = folderSAVE+tag[:-1] + "-000/" + etsfakes.filesavetag[1:]+"/" +"trueparams.txt"
        with open(trueparamsfile, 'w') as file:
            file.write(str(trueparams[k]))
    if saveresults is not None:
        with open(saveresults, 'w') as file:
            file.write(str(output_onesigma))
            file.write("\n")
            file.write(str(output_twosigma))
        
    return output_onesigma,output_twosigma

timeaxis, paramsall, intall, noiseall= fake_type3(4)
fakedisctimes = np.random.uniform(1,5,4)
folderSAVE = ""
saveresults = ""

output_onesigma,output_twosigma = fit_all_fakes_3(timeaxis,intall,noiseall,
                                                  paramsall, fakedisctimes, 
                                                  folderSAVE, customtag=None,
                                                  saveresults=saveresults)