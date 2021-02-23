#reflect/trans --> structure CNN
#1 layer structures


import warnings
warnings.filterwarnings('ignore')
# -*- coding: utf-8 -*-


import numpy as np
from numpy import round
import tensorflow as tf
from keras import callbacks
from keras.callbacks import LearningRateScheduler as LRS
from keras.models import Model
from keras.layers import *
from keras import optimizers as opt
import keras.backend as K
import matplotlib.pyplot as plt
import datetime
global date
date = datetime.datetime.now()
import h5py
import scipy.io as sio
from contextlib import redirect_stdout
import csv

#make sure you can see the GPU or Keras will attempt to run on CPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#data rescaling functions
#puts the data in roughly (0-1) range
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

#load in the data from the generator file
#this follows the generator format :
#[angle,materials,thickness,rp,rs,tp,ts,psi,delta]
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

    #rescale the input data
    #usually just rescaling angles, thicknesses, psi and delta is OK
    psi = resc_psi(psi)
    delta = resc_delt(delta)
    th = resc_th(th)
    return (ml1,th,theta,rp,rs,tp,ts,psi,delta)

#split the input arrays for multiple independent training / validation sets
def split_data(m1,th,ang,rp,rs,tp,ts,psi,delta,mn,mx):
    tr_m1=m1[mn:mx,:]
    tr_th=th[mn:mx,:]
    tr_ang=ang[mn:mx,:]
    tr_p=psi[mn:mx,:]
    tr_d=delta[mn:mx,:]
    tr_rp=rp[mn:mx,:]
    tr_rs=rs[mn:mx,:]
    tr_tp=tp[mn:mx,:]
    tr_ts=ts[mn:mx,:]
    return (tr_m1,tr_th,tr_ang,tr_rp,tr_rs,tr_tp,tr_ts,tr_p,tr_d)


#read in data from file with the readin_data fcn
#you need to update the num_mat, ect.. variables with your choices from the generator program
num_mat = 5
num_ang = 3
num_lay = 1
num_wave = 200
(ml1,th,ang,rp,rs,tp,ts,psi,delta) = readin_data('/home/arl92/Documents/newdata/data_rte_gen1lay5mat_0ge_240000n_v-tma_20201112.h5',num_mat,num_ang,num_lay,num_wave)

#directort to save data
dr = '/home/arl92/Documents/newdata/rt2mt/'

#ranges for the independent datasets, recommended most data in the training set
#then split the data
tr = 200000   #training
va = tr+10000 #validation
te = va+10000 #test

(m1r,thr,angr,rpr,rsr,tpr,tsr,psir,deltar) = split_data(ml1,th,ang,rp,rs,tp,ts,psi,delta,0,tr)
(m1v,thv,angv,rpv,rsv,tpv,tsv,psiv,deltav) = split_data(ml1,th,ang,rp,rs,tp,ts,psi,delta,tr,va)
(m1e,the,ange,rpe,rse,tpee,tse,psie,deltae) = split_data(ml1,th,ang,rp,rs,tp,ts,psi,delta,va,te)

#the spectra need to be reshaped from [spec(ang1),spec(ang2),...] to [spec;angle] 2-d matrix
#this does not seem particularly efficent but it works
 tr_p = np.zeros((tr,num_wave,num_ang))
 tr_d = np.zeros((tr,num_wave,num_ang))
 p = np.transpose(psir)
 d = np.transpose(deltar)
 for i in range(tr):
     tr_p[i,:,:] = np.transpose(np.reshape(p[:,i],(num_ang,num_wave)))
     tr_d[i,:,:] = np.transpose(np.reshape(d[:,i],(num_ang,num_wave)))

 va_p = np.zeros((va-tr,num_wave,num_ang))
 va_d = np.zeros((va-tr,num_wave,num_ang))
 p = np.transpose(psiv)
 d = np.transpose(deltav)
 for i in range(va-tr):
     va_p[i,:,:] = np.transpose(np.reshape(p[:,i],(num_ang,num_wave)))
     va_d[i,:,:] = np.transpose(np.reshape(d[:,i],(num_ang,num_wave)))

 te_p = np.zeros((te-va,num_wave,num_ang))
 te_d = np.zeros((te-va,num_wave,num_ang))
 p = np.transpose(psie)
 d = np.transpose(deltae)
 for i in range(te-va):
     te_p[i,:,:] = np.transpose(np.reshape(p[:,i],(num_ang,num_wave)))
     te_d[i,:,:] = np.transpose(np.reshape(d[:,i],(num_ang,num_wave)))
   
tr_rp = np.zeros((tr,num_wave,num_ang))
tr_rs = np.zeros((tr,num_wave,num_ang))
p = np.transpose(rpr)
d = np.transpose(rsr)
for i in range(tr):
    tr_rp[i,:,:] = np.transpose(np.reshape(p[:,i],(num_ang,num_wave)))
    tr_rs[i,:,:] = np.transpose(np.reshape(d[:,i],(num_ang,num_wave)))

va_rp = np.zeros((va-tr,num_wave,num_ang))
va_rs = np.zeros((va-tr,num_wave,num_ang))
p = np.transpose(rpv)
d = np.transpose(rsv)
for i in range(va-tr):
    va_rp[i,:,:] = np.transpose(np.reshape(p[:,i],(num_ang,num_wave)))
    va_rs[i,:,:] = np.transpose(np.reshape(d[:,i],(num_ang,num_wave)))

te_rp = np.zeros((te-va,num_wave,num_ang))
te_rs = np.zeros((te-va,num_wave,num_ang))
p = np.transpose(rpe)
d = np.transpose(rse)
for i in range(te-va):
    te_rp[i,:,:] = np.transpose(np.reshape(p[:,i],(num_ang,num_wave)))
    te_rs[i,:,:] = np.transpose(np.reshape(d[:,i],(num_ang,num_wave)))
    
tr_tp = np.zeros((tr,num_wave,num_ang))
tr_ts = np.zeros((tr,num_wave,num_ang))
p = np.transpose(tpr)
d = np.transpose(tsr)
for i in range(tr):
    tr_tp[i,:,:] = np.transpose(np.reshape(p[:,i],(num_ang,num_wave)))
    tr_ts[i,:,:] = np.transpose(np.reshape(d[:,i],(num_ang,num_wave)))

va_tp = np.zeros((va-tr,num_wave,num_ang))
va_ts = np.zeros((va-tr,num_wave,num_ang))
p = np.transpose(tpv)
d = np.transpose(tsv)
for i in range(va-tr):
    va_tp[i,:,:] = np.transpose(np.reshape(p[:,i],(num_ang,num_wave)))
    va_ts[i,:,:] = np.transpose(np.reshape(d[:,i],(num_ang,num_wave)))

te_tp = np.zeros((te-va,num_wave,num_ang))
te_ts = np.zeros((te-va,num_wave,num_ang))
p = np.transpose(tpee)
d = np.transpose(tse)
for i in range(te-va):
    te_tp[i,:,:] = np.transpose(np.reshape(p[:,i],(num_ang,num_wave)))
    te_ts[i,:,:] = np.transpose(np.reshape(d[:,i],(num_ang,num_wave)))


#fuction to build the CNN model 
#takes several hyperparameters as input:
# n_conv = number of convolutoinal layers-1
# nla    = number of dense layers
# np     = number of dense nodes #1
# np1    = number of dense nodes #2
# np2    = number of dense nodes #3
# drop   = dropout rate
# a      = learning rate scaling factor
def define_model(x):
    global st
    global modname
    global model
    global a
    st += 1
    
    n_conv = int(x[0])
    nla = int(x[1])
    np =  int(x[2])
    np1 = int(x[3])
    np2 = int(x[4])
    drop = x[5]
    a = x[6]
    
    def decay(ep):
        global a
        lr = a/((ep)+1)#simple 1 param 1/(t) decay
        return lr
    lr = LRS(decay)
    
    #input layers
    rpin = Input((200,3))
    rsin = Input((200,3))
    tpin = Input((200,3))
    tsin = Input((200,3))
    
    #variable number of convolutional layers
    #independent layers for psi and delta
    convrp = Conv1D(64,4,activation='relu',padding='same')(rpin)
    convrp = MaxPooling1D(pool_size = 2)(convrp)
    for i in range(n_conv):
        convrp = Conv1D(64,4,activation='relu')(convrp)
        convrp = MaxPooling1D(pool_size = 2)(convrp)
    convrp = MaxPooling1D(pool_size = 2)(convrp)
    frp = Flatten()(convrp)

    convrs = Conv1D(64,4,activation='relu',padding='same')(rsin)
    convrs = MaxPooling1D(pool_size = 2)(convrs)
    for i in range(n_conv):
        convrs = Conv1D(64,4,activation='relu')(convrs)
        convrs = MaxPooling1D(pool_size = 2)(convrs)
    convrs = MaxPooling1D(pool_size = 2)(convrs)
    frs = Flatten()(convrs)

    convtp = Conv1D(64,4,activation='relu',padding='same')(tpin)
    convtp = MaxPooling1D(pool_size = 2)(convtp)
    for i in range(n_conv):
        convtp = Conv1D(64,4,activation='relu')(convtp)
        convtp = MaxPooling1D(pool_size = 2)(convtp)
    convtp = MaxPooling1D(pool_size = 2)(convtp)
    ftp = Flatten()(convtp)

    convts = Conv1D(64,4,activation='relu',padding='same')(tsin)
    convts = MaxPooling1D(pool_size = 2)(convts)
    for i in range(n_conv):
        convts = Conv1D(64,4,activation='relu')(convts)
        convts = MaxPooling1D(pool_size = 2)(convts)
    convts = MaxPooling1D(pool_size = 2)(convts)
    fts = Flatten()(convts)

    #add the two polarization states together in some dense layers
    #pass the convolutoinal layers through dense layers, still indepent by spectra type
    r = Add()([frp,frs])
    t = Add()([ftp,fts])
    r = Dense(np,activation='relu')(r)
    r = Dropout(drop)(r)
    t = Dense(np,activation='relu')(t)
    t = Dropout(drop)(t)

    #combine the two spectral types
    a = Add()([r,t])
    #pass the data through more dense layers
    a = Dense(np1,activation='relu')(a)
    a = Dropout(drop)(a)
    for i in range(nla):
        a = Dense(np1,activation='relu')(a)
        a = Dropout(drop)(a)
    
    #indepent layers for the materials and thickness funneling the data to small number of output nodes
    m = Dense(np2,activation='relu')(a)
    t = Dense(np2,activation='relu')(a)
    m = Dense(200,activation='relu')(m)
    t = Dense(200,activation='relu')(t)
    m = Dense(100,activation='relu')(m)
    t = Dense(50,activation='relu')(t)


    #output nodes
    #thickness is a number in so there is no activation
    thout = Dense(1,activation=None)(t)
    #materials guesses a probability array for each material
    #real material is always a normalized array with 1 at the material index i.e. [0 1 0 0 0]
    mout1 = Dense(5,activation='softmax')(m)

    #define the model
    model = Model([rpin,rsin,tpin,tsin],[thout,mout1])
    # compile the model using the adam optimizer and appropriate loss functions
    #the loss weights can be used to tune the relative importance of output data . 
    #I find that increasing the loss weight for the thickness layer tends to improve fitting,
    #since there loss fuctions are different for materials and thicknesses
    model.compile(optimizer='adam',loss=['mse','categorical_crossentropy'],metrics = ['mse','accuracy'],loss_weights=[2,1])
    model.summary()

    #model name can be anything you want
    modname = dr+'general_1lay4matinverse_model_v-tma_convolution_rprstpts_'+str(st)+'step_'+date.strftime("%Y")+date.strftime("%m")+date.strftime("%d")
    #print the model summary to file for records, the modelname in log also helps with identificaiton of the log files
    print('-'*57)
    print(modname)
    print('-'*57)
    with open(modname+'_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
        
#function called to fit the model and test the result
#need to input the hyperparameters into this function
# x[0] = number of convolutoinal layers-1
# x[1] = number of dense layers
# x[2] = number of dense nodes #1
# x[3] = number of dense nodes #2
# x[4] = number of dense nodes #3
# x[5] = dropout rate
# x[6] = learning rate scaling factor
def fit_fcn(x):
    global modname
    global st
    global model
    
    #build the CNN model
    #model is saved as a global variable 
    define_model(x)
    
    #number of epochs to fit
    leng = 300
    #fit the model and return the loss and metric history 
    #fitting based on independent training and validation sets
    #the lr callback is necessary to decay the learning rate as a function of epoch
    #batch size can be tuned as necessary
    history = model.fit([tr_rp,tr_rs,tr_tp,tr_ts],[thr,m1r],epochs=leng,batch_size=128,verbose=2,validation_data = ([va_rp,va_rs,va_tp,va_ts],[thv,m1v]),callbacks=[lr])

    #save model for later uses
    model.save(modname)

    #evaluate and print stats on the model fit to the log file
    (te_loss) = model.evaluate([te_rp,te_rs,te_tp,te_ts],[the,m1e])
    print(te_loss)
    
    #this can be uncommented to save a plot of the learning rate decay over training epoch
    #not necessary, since the graph can be reconstructed from knowning a
    #also doesn't work unless there is a graphical ouput
    
    #try:
    #    plt.plot(range(len(history.history['lr'])),history.history['lr'])
    #    plt.legend(['LearningRate'])
    #    plt.savefig(modname+'loss_fcn.png', dpi=300)
    #except:
    #    print('PlotLoss figure not saved.\n')

    #save the model statistics from the fitting and test evaluation
    try:
        np.savetxt(modname+'_saveresults.txt',te_loss)
        #saves the model training progress history
        sio.savemat(modname+'_history.mat',history.history)
    except:
        print('Model epoch training data not saved.\n')
  

#globals available for the fitting function
global modname
global st
global model

#use as a unique identifier for the output file (or step if training multiple instances)
st=0

#hyperparameters for building the model
hparams = np.array([2,2,379,315,601,0.038915798772616,0.005493252183191])

#fit the function using the described hyperparameters
fit_fcn(hparams)

