#genetic algorithm to perform the inverse design problem in thin film metamaterials
#5 layer systems
 
from numba import jit
import numpy as np
import TMM_numba as tmm
from scipy.optimize import least_squares
import h5py
import BB_metals as bb
import dielectric_materials as di
from joblib import Parallel, delayed
import multiprocessing
cores = multiprocessing.cpu_count()
import datetime
from numba.extending import overload

#label the main directory and an additional folder where you would like to save the output
#all files should be in this directory
dr = '/home/arl92/Documents/newdata/'
dr_save = '/home/arl92/Documents/newdata/comparisons/genetic/'

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

#model info taken from the data generation script
#this is to ensure the optimization is constrained to the same domain as probed by the CNN model
#since this data is just the ranges for the CNN dataset, inclusion of this data is necessary for a fair comparison between methods
trange = np.array([1,60])*1E-9 #allowed thicknesses
ang = np.array([25.,45.,65.])  #incident angles
num_ang = ang.size
num_lay = 5
num_mat = 5
num_wave = wave.size
np.random.seed(38947) #seed the RNG in numpy for reproducibility
#####


#data rescaling fcns
def resc_mat(in_mat):
    resc = in_mat / 4
    return resc
def resc_th(in_th):
    resc = in_th* 1E7
    return resc
def resc_ang(in_theta):
    resc = in_theta / 45
    return resc
def resc_psi(in_psi):
    resc = in_psi/90
    return resc
def resc_delt(in_delt):
    resc = in_delt/90
    return resc

#funciton to read in data produced by the generation script 
#optimizaion acts on spectra in the CNN test dataset
def readin_data(filename,nmat,nang,nlay,nwave):
    f = h5py.File(filename,'r')
    arrd = np.array(f.get('data'))
    itl=0
    ith=nang
    theta = (arrd[:,itl:ith])
    itl=ith
    ith +=nmat
    ml1 = (arrd[:,itl:ith])
    itl=ith
    ith +=nmat
    ml2 = (arrd[:,itl:ith])
    itl=ith
    ith +=nmat
    ml3 = (arrd[:,itl:ith])
    itl=ith
    ith +=nmat
    ml4 = (arrd[:,itl:ith])
    itl=ith
    ith +=nmat
    ml5 = (arrd[:,itl:ith])
    itl=ith
    ith +=nlay
    th = (arrd[:,itl:ith])
    itl=ith
    ith +=nwave*nang
    rp = (arrd[:,itl:ith])
    itl=ith
    ith +=nwave*nang
    rs = (arrd[:,itl:ith])
    itl=ith
    ith +=nwave*nang
    tp = (arrd[:,itl:ith])
    itl=ith
    ith +=nwave*nang
    ts = (arrd[:,itl:ith])
    itl=ith
    ith +=nwave*nang
    psi = (arrd[:,itl:ith])
    itl=ith
    delta = (arrd[:,itl:])
    f.close()
    return (ml1,ml2,ml3,ml4,ml5,th,theta,rp,rs,tp,ts,psi,delta)

#simple well in 2d, this can be used to test the ability of the script to find a global minimum
#attempt to optimize this function to check the vailidity of the code
def test_well(location):
    minwell =  np.abs(trange[0]-trange[1])/2+np.min((trange[0],trange[1]))
    x1 = location[0]
    x2 = location[1]
    height = np.array([(x1-minwell) ,(x2-minwell)])
    return height

#tanh transformation constrains the domain to conincide with the thickness ranges seen in the generation script file
#this keeps the otpimization script from optimizing to thicknesses which are outside the total range, and does not allow unphysical thicknesses which give technically correct spectra when transformed by the TMM code
#tanh(thickness_unscaled)  = [min_thickness, max_thickness] 
@jit(nopython=True)
def transform(x):
    xt = (np.tanh(x)*(0.5*(trange[1]-trange[0]))) + (0.5*(trange[1]+trange[0]))
    return xt

#residuals function, returns residuals between the optimum spectra and a TMM generated spectra based on the optimized thicknesses 
#different spectra are concatenated together in a linear array
@jit(nopython=True)
def residuals_fcn_pd(n,x,p,d):
    xt = transform(x)
    ang = np.array([25.,45.,65.])
    psi = np.zeros(wave.size*ang.size,dtype=np.float64)
    delta = np.zeros(wave.size*ang.size,dtype=np.float64)
    for j in range(0,ang.size):
        for i in range(0,wave.size):
            (psi[i+wave.size*j],delta[i+wave.size*j]) = tmm.ellips(ang[j], wave[i], n[:,i], xt, 1., n_subst[i])

    return np.concatenate(((psi-p),(delta-d)))

#calulates the fitness for the current population
#fitness is calculated as the -MSE between the spectra produced by the optimized materials and the target spectra
@jit(nopython=True)
def calcFit(newPop,tp,td):
    fit = np.zeros(newPop.shape[0])
    n = np.zeros((num_lay,wave.size),dtype = np.complex128)
    #loop over population
    for i in range(newPop.shape[0]):
        l = newPop[i,:num_lay]
        mat = newPop[i,num_lay:]
        for j in range(num_lay):
            n[j,:] = materials[np.int(mat[j])]
        #calc output for all angles
        mse = -1*np.mean(np.square(residuals_fcn_pd(n,l,tp,td)))
        fit[i] = mse
        #this is where the test well is inserted into the code for testing the optimzation
        #change the dimensionality of the well as needed
        #fit[i] = -1*np.sum(np.square(test_well(l)))/3
    return fit

#generate a population with random mutations in thickness and materials
#the frequency of these mutations can be tuned as needed.
@jit(nopython=True)
def mutation(offCross):
    #loop over population
    for i in range(offCross.shape[0]):
        #sometimes randomly change the thicknesses with frequency specified by the inequality
        #the normal is weighted so after the tanh rescaling the probability for any thickness is roughly uniform within the allowed domain
        if np.random.rand() < 0.5:
            for k in range(num_lay):
                offCross[i,k] += np.random.normal(0,(1-0.99/np.cosh(offCross[i,k])**2)*2)
        #sometimes randomly change the material in a random layer
        #this is a more abrupt change and happens much less frequently
        #the new material is uniformly chosen
        if np.random.rand() < 0.05:
            offCross[i,np.random.randint(num_lay,2*num_lay)] = np.random.randint(0,num_mat)
    return offCross


#choose the most fit individuals to be parents
#the number of mating parents is given by numMating
@jit(nopython=True)
def matingPool(newPop,fit,numMating):
    parents = np.empty((numMating,newPop.shape[1]))
    for parent in range(numMating):
        max_fit = np.argmax(fit)
        parents[parent,:] = newPop[max_fit,:]
        #reasign the chosen individual to a terrible fitness
        fit[max_fit] = -99999999999999
    return parents
        
#crossover genes between 2 parents
#the layer thickness and materials are co-dependent, since a change in materials moves to a new subspace, thus the crossover is my layer and thickness / material pairs are maintained
@jit(nopython=True)
def crossover(parents,offSize):
    off = np.empty(offSize)
    crossPt = np.ceil(num_lay/2)
    #the crossover is roughly half from each parent, where the first parent contribues more if there is an odd number of layers.
    for i in range(offSize[0]):
        p1 = (2*i-1)%parents.shape[0]
        p2 = (2*i)%parents.shape[0]
        off[i,:crossPt] = parents[p1,:crossPt]
        off[i,num_lay:num_lay+crossPt] = parents[p1,num_lay:num_lay+crossPt]
        off[i,crossPt:num_lay] = parents[p2,crossPt:num_lay]
        off[i,num_lay+crossPt:] = parents[p2,num_lay+crossPt:]
    return off

    
#function to perform the genetic evolution of the population
#you need to input the number of individuals in the population and the target spectra
#returns the optimized population and the history of the most fit individual in each generation
@jit(nopython=True)
def history(solPerPop,tp,td):
    #create an initial population with solPerPop individuals
    #the normal distribution with s=0.85 is 'roughly' a uniform distribution after the tanh transformation is performed
    thicks = np.random.normal(0,0.85,size = (solPerPop,num_lay))
    mats = np.random.randint(0,num_mat,size = (solPerPop,num_lay))
    iniPop = np.concatenate((thicks,mats),axis=1)
    newPop = iniPop
    hist = np.zeros(numGen)
    #evolve the population for numGen generations
    for gen in range(numGen):
        #calculate initial fitness
        fit = calcFit(newPop,tp,td)
        #select parents from the population
        parents = matingPool(newPop,fit,numMating)
        parShape = parents.shape
        #offspring make up the remainder of the population.
        #perform the crossover and mutation for the offspring
        offSize = (popSize[0] - parShape[0],popSize[1])
        offCross = crossover(parents,offSize)
        offMutat = mutation(offCross)
        #new population contains both parents and offspring
        newPop[:parShape[0],:] = parents
        newPop[parShape[0]:,:] = offMutat
        #history of the population fitness
        fit = calcFit(newPop,tp,td)
        #this is nice to show in real time how the population is evolving
        #print('Gen',gen,' max: ',np.max(fit), newPop[np.argmax(fit),:])
        hist[gen] = np.max(fit)
    return (hist,newPop)
        

#generation function to be called by the parallelization module
#opimizes the population and calculates optimum fitness
#returns the most fit individual and the max and std from the population
#the best best individual is the optimization target
@jit(nopython=True)
def gen_fcn(tp,td,solPerPop):
    (hist,optPop) = history(solPerPop,tp,td)
    optFit = calcFit(optPop,tp,td)
    best = optPop[np.argmax(optFit),:]
    best[:num_lay] = transform(best[:num_lay])
    return np.concatenate((best,np.array([np.max(optFit)]),np.array([np.std(optFit)])))

#creates an ensemble of several initial populations and returns optimized populations for each
#the ensemble. can be useful for generating statistics on the obtained optima and a range of optimization times
#returns the best individual overall, the std in fitness for the best from all popuations and the program runtime
def ensemble_fcn(solPerPop,tp,td,ensemble,g):
    results = np.zeros((ensemble,numParams+2))
    #put the actual evolution in a try loop in case the program encounters an exception
    try:
        start = datetime.datetime.now()
        for k in range(ensemble):
            results[k,:] = gen_fcn(tp,td,solPerPop)
        end = datetime.datetime.now()
        best = results[np.argmax(results[:,-2]),:]
        #sys_runtime is used for comparison, is the time to calculate the results for n number of individual populations and save the results
        #need to make sure the optimization works for all popualtions to trust the sys_runtime statistic
        sys_runtime = (end-start)
    except:
    	#return zeros for everything
        best = np.zeros(numParams+2)
        sys_runtime = (datetime.datetime.now()-datetime.datetime.now())
        #check logfile for this error message...
        print('Error in Sys #',g)
    return np.concatenate((best,np.array([np.std(results[:,-2])]),np.array([sys_runtime.seconds+sys_runtime.microseconds*1E-6]).astype('float')))

#Accuracy for materials and thickness MSE metrics
#compared to the data generation materials and thicknesses
def accutest(tm,tt,pm,pt):
    (num_el,num_lay) = tm.shape
    acc = np.zeros(num_lay)
    thicks = np.zeros(num_lay)
    for i in range(num_el):
        for j in range(num_lay):
            if np.array(tm[i,j]).astype('int') == np.array(pm[i,j]).astype('int'):
                acc[j] += 1
            thicks[j] += (tt[i,j]-pt[i,j])**2
    #reurns accuracy % and thickness MSE for each layer averaged over the input data sets
    #this assumes the data rescaling found in the 
    metrics = np.array([acc*100,thicks*1E14])/num_el
    return metrics

#read in data from file
filename = dr+'data_rte_gen5lay5mat_0ge_240000n_v-tma_20201112.h5'
(tm1,tm2,tm3,tm4,tm5,tth,tang,trp,trs,ttp,tts,psi,delta) = readin_data(filename,num_mat,num_ang,num_lay,num_wave)


######################################## Parameters

#general parameters
numParams = int(2*num_lay) #thickness and materials for each layer
numGen = 30 # number of generations for the population (~30 reaches a local maximum in tests)
sample = int(cores) #different systems to probe (n*cores is a good idea for the parallelization)
offset = 220000 #makes sure you are looking at the CNN test portion of the dataset
ensemble = 3 #distinct populations per system (min 3 for the std statistic)

#initialization of Numba functions, just let this part run and do not save output
#helps make runtume faster for actual optimization
popSize = (10,numParams)
numMating = np.int(5)
ensemble_fcn(10,psi[2,:],delta[2,:],ensemble,2)

#GENETIC PARAMETERS
solPerPop = 100
popSize = (solPerPop,numParams)
numMating = np.int(solPerPop/2)

########################################


#parallelized over samples (each sample runs independently on a core)
#output looks nice in the log file...
print('------Individuals:%d------'%(solPerPop))
print('Start: ',datetime.datetime.now())
start = datetime.datetime.now()
results = Parallel(n_jobs=cores,verbose=10)(delayed(ensemble_fcn)(solPerPop,psi[g,:],delta[g,:],ensemble,g) for g in range(offset,sample+offset))
end = datetime.datetime.now()
datasave = np.array(results)

#this is the total program runtime
print('End    : ',end)
print('Runtime: ',(end-start))
#filname contains relevant genetic hyperparameters
filename = dr_save+'genetic_fitresults_5l5m_'+str(offset)+'-'+str(offset+sample)+'_'+'pop-'+str(solPerPop)+'gen-'+str(numGen)+'ens-'+str(ensemble)+'_'+start.strftime('%y')+start.strftime('%m')+start.strftime('%d')+'_'+start.strftime('%H')+start.strftime('%M')+start.strftime('%S')
print(filename)
#save the genetic results to a file
#contians the [optimized material parameters, std over the ensemble, and system runtime] for each system considered
np.savetxt(filename+'.txt', datasave,delimiter=',')

#calculate the statistics for genetic output vs ground truth (spectral RMSE, materials accuracy, layer thickness RMSE)
pt = datasave[:,:num_lay]
pm = datasave[:,num_lay:2*num_lay]
tt = tth[offset:sample+offset,:]
tm = np.transpose(np.array([np.argmax(tm1[offset:sample+offset,:],axis=-1),np.argmax(tm2[offset:sample+offset,:],axis=-1),np.argmax(tm3[offset:sample+offset,:],axis=-1),np.argmax(tm4[offset:sample+offset,:],axis=-1),,np.argmax(tm5[offset:sample+offset,:],axis=-1)]))
met = accutest(tm,tt,pm,pt)
#calculate the mean and std for the fitness (spectral RMSE) and runtime over mutliple samples
#this runtime is an average over the individual runtimes for independent systems and does not include unnecssary overheads or is impacted by outliers like the total program runtime. This is the runtime metric which should be referenced in comparison to other methods.
mfit = np.mean(np.sqrt(-datasave[:,2*num_lay]))
sfit = np.std(np.sqrt(-datasave[:,2*num_lay]))
mtime = np.mean(datasave[:,-1])
stime = np.std(datasave[:,-1])
#print the results to the log file
print('Metrics: ',met)
print('RMSE   : ',mfit,sfit)
print('Ens STD: ',np.mean(datasave[:,-2]))
print('Time   : ',mtime,stime)
#save the results in individual txt files
np.savetxt(filename+'_metrics.txt',met)
np.savetxt(filename+'_statistics.txt',np.concatenate((met[0,:],met[1,:],np.array([mfit,sfit,mtime,stime]))),delimiter=',')


