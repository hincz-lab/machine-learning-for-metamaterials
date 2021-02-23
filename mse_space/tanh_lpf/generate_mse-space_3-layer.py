#script to generate the MSE space for each system
#generates MSE at every point in 0.5nm increments to span the space
# 3 layer systems

import numpy as np
import datetime
import TMM_numba as tmm
import BB_metals as bb
import dielectric_materials as di
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm_notebook as tqdm
from numba import jit
from scipy.optimize import least_squares

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())



#label the main directory and an additional folder where you would like to save the output
#all files should be in this directory
dr = '/home/arl92/Documents/newdata/mse_space/tanh/'
dr_save = '/home/arl92/Documents/newdata/mse_space/tanh/3-layer/results-even/'

#import materials and define wavelength
#these parameters are the same found in the CNN data generation program
wave = np.linspace(450,950,200)*1E-9

#materials
ag = bb.nk_material('Ag',wave)
au = bb.nk_material('Au',wave)
al2o3 = np.loadtxt(dr+'alumina.txt').view(complex) #file containing RI constants taken from the dielectric_materials module
#identical to calling al2o3 = di.nk_material('al2o3',wave) for these wavelengths
d_tio2 = np.loadtxt(dr+'tio2.txt') #file containing RI constants taken from the dielectric_materials module
tio2 = d_tio2[0,:]+1j*d_tio2[1,:]
ito = np.loadtxt(dr+'ito.txt').view(complex) #file containing RI constants taken from the dielectric_materials module

gl = di.nk_Cauchy_Urbach(wave,1.55,0.005) #decent theoretical glass model based on cauchy disperision and previous experimental fits

#define the materials array
materials = np.array([ag,al2o3,ito,au,tio2])

#substrate and superstrate for the materials stack (1. = Void)
n_subst = gl
n_super = 1.

#data is saved in individual text files
#these are the 'ground truth' (drawn) spectra to which the CNN is attempting to optimize
rpt = np.loadtxt(dr+'rp.txt')
rst = np.loadtxt(dr+'rs.txt')
tpt = np.loadtxt(dr+'tp.txt')
tst = np.loadtxt(dr+'ts.txt')


num_mat = 5 #number of materials probed by the system
num_lay = 3 #total layer number in the probed systems

#I have found that the only reasonable way to generate this data is to run multiple iterations of the same
#script on different nodes at the same time. This can be accomplished by uncommenting one of the 'thicks'
#lines at a time for each run. Also the thicknesses in each array can be altered to look at the sets of
#evens, evens+0.5nm, odds, odds+0.5nm, for a total of 20 parallel iterations of the same code. I suggest to
#rename the output folder for each thickness type (o,e,o+0.5,e+0.5) to give a unique identifier when used
#in combination with the s variable in the output file name.I have only provided this single code beacuse 
#the alterations are minor between runs.

#evens
thicks = np.array([10,20,30,40,50,60]);s=1
#thicks = np.array([2,12,22,32,42,52]);s=2
#thicks = np.array([4,14,24,34,44,54]);s=3
#thicks = np.array([6,16,26,36,46,56]);s=4
#thicks = np.array([8,18,28,38,48,58]);s=5

#odds
#thicks = np.array([1,11,21,31,41,51]);s=1
#thicks = np.array([3,13,23,33,43,53]);s=2
#thicks = np.array([5,15,25,35,45,55]);s=3
#thicks = np.array([7,17,27,37,47,57]);s=4
#thicks = np.array([9,19,29,39,49,59]);s=5

#evens+0.5nm
#thicks = np.array([0.5,20.5,30.5,40.5,50.5,60.5]);s=1
#thicks = np.array([2.5,12.5,22.5,32.5,42.5,52.5]);s=2
#thicks = np.array([4.5,14.5,24.5,34.5,44.5,54.5]);s=3
#thicks = np.array([6.5,16.5,26.5,36.5,46.5,56.5]);s=4
#thicks = np.array([8.5,18.5,28.5,38.5,48.5,58.5]);s=5

#odds+0.5nm
#thicks = np.array([1.5,11.5,21.5,31.5,41.5,51.5]);s=1
#thicks = np.array([3.5,13.5,23.5,33.5,43.5,53.5]);s=2
#thicks = np.array([5.5,15.5,25.5,35.5,45.5,55.5]);s=3
#thicks = np.array([7.5,17.5,27.5,37.5,47.5,57.5]);s=4
#thicks = np.array([9.5,19.5,29.5,39.5,49.5,59.5]);s=5

#this is the number on thicknesses probed in this script
numthick = 6

#function called by the parallelization module
#takes the target spectra and 2 looping variables. saves a file for systems with these looping variables
def call_fcn(a,b,rpt,rst,tpt,tst):
    its=0
    #loop over the rest of the materials
    for c in np.delete(np.arange(num_mat),b):
        it=0
        output = np.zeros((int(((num_mat-1))*(numthick**num_lay)),6))
        #define the materials array
        n = np.zeros((num_lay,wave.size),dtype=complex)
        n = np.array([materials[a],materials[b],materials[c]])
        #loop over the thicknesses
        for f in range(numthick):
            for g in range(numthick):
                for h in range(numthick):
                    #define the thicknesses array
                    th = np.array([thicks[f],thicks[g],thicks[h]])*1E-9
                    rp = np.zeros((wave.size,ang.size))
                    rs = np.zeros((wave.size,ang.size))
                    tp = np.zeros((wave.size,ang.size))
                    ts = np.zeros((wave.size,ang.size))
                    #generate the predicted spectral response for the specific choice of materials and thicknesses
                    #spectra are generated from the same TMM code as used in data generation
                    for j in range(0,ang.size):     
                        for i in range(0,wave.size):
                            rp[i,j] = tmm.reflect_amp(1,ang[j], wave[i], n[:,i], th, 1., n_subst[i])
                            rs[i,j] = tmm.reflect_amp(0,ang[j], wave[i], n[:,i], th, 1., n_subst[i])
                            tp[i,j] = tmm.trans_amp(1,ang[j], wave[i], n[:,i], th, 1., n_subst[i])
                            ts[i,j] = tmm.trans_amp(0,ang[j], wave[i], n[:,i], th, 1., n_subst[i])
                    #RMSE is calculated as the root(mean(square(residuals))) between the target spectra and the spectra generated from the materials and thickness choice
                    rrp = np.sqrt(np.mean(np.mean(np.square(rpt-rp))))
                    rrs = np.sqrt(np.mean(np.mean(np.square(rst-rs))))
                    rtp = np.sqrt(np.mean(np.mean(np.square(tpt-tp))))
                    rts = np.sqrt(np.mean(np.mean(np.square(tst-ts))))
                    #this is the mean RMSE for each spectral type
                    #used as a singular metric for the response
                    rmser = np.mean(np.array([rrp,rrs,rtp,rts]))
                    output[it,:] = np.array([it,rrp,rrs,rtp,rts,rmser])
                    it+=1
        #save the RMSE results to file. Produces a single file for each full system of materials choices. 
        #make sure you have a unique file identifier! Currently set up for unique directory (e,o,ect..) - s (1-5) - a, b, and its (unique for c, thickness choices). This can in principle be whatever you want
        np.savetxt(dr_save+str(s)+'_'+str(a)+str(b)+'_'+str(its)+'.txt',output)
        its += 1
    #returns its, can be used to make sure all files have ran in the output log
    return its
            
#a and b are the materials codes for the first two layers of the system
#this should contain all allowed combinations of materials in the first two layers (num_lay*(num_lay-1) elements in each array). Do not include the systems with repeating layers since they are not represented in the CNN training dataset.
a = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])
b = np.array([1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3])

#calculate the RMSE for the theoretical spectra with given parameters
print('Generating Parallel Pool...\n')
cores = multiprocessing.cpu_count()
print('Parallel: %d Found Cores.\n'%(cores))
print('Generating Solutions...\n')
#loop through the a and b arrays
results = Parallel(n_jobs=cores)(delayed(call_fcn)(a[g],b[g],rpt,rst,tpt,tst) for g in range(a.size))
#print the results to file to check that everything ran the correct number of iterations
print(results)
