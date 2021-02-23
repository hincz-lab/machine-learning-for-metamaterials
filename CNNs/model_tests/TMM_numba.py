# -*- coding: utf-8 -*-
#Implementation of Chillwell(1984) thin film TMM
#arl92@case.edu   09/08/2019

from numba import jit
import numpy as np

#Sample parameters list
##################################################
## 0 = TE, 1 = TM polarization
#rho = 0
##user defined index of refraction for layers, cover(above), and substrate(below)
#n = np.array([1,1,1,13]) 
#n_cover = 1
#n_subst = 1
##user defined thickness of each layer, m
#l = np.array([20,20,20,20])*1E-9
##angle of incidence, degrees
#ang_of_inc = 0
## wavelength of light incident in m
#wavelength = 500E-9;
##################################################

#calculate and return the reflection amplitude for system in given POL state
@jit(nopython=True)
def reflect_amp(rho, ang_of_inc, wavelength, n, l, n_cover, n_subst):
    #calculate parameters from the user input
    #wavenumber
    k = 2*np.pi/wavelength;
    #impedance of FS, ohms
    z0 = 376.730313667
    #direction cosines
    beta = np.multiply(n_cover,np.sin(ang_of_inc*np.pi/180))
    ang = np.arcsin(np.divide(beta,n))
    alpha = np.multiply(n,np.cos(ang))
    ang_s = np.arcsin(beta/n_subst)
    #phase thickness
    phi = k*np.multiply(alpha,l)
    #gamma parameter different for TE and TM
    if rho == 0:
        gamma = alpha/z0
        #gamma in cover(above) and substrate(below)
        gammac = n_cover*np.cos(ang_of_inc*np.pi/180)/z0
        gammas = n_subst*np.cos(ang_s)/z0
    elif rho == 1:
        gamma = z0*np.divide(np.cos(ang),n)
        gammac = z0*np.cos(ang_of_inc*np.pi/180)/n_cover
        gammas = z0*np.cos(ang_s)/n_subst
    else:
        #exception handling
        print("Exception in Pol State. Ending...")
        return
        
    #calculate transfer matrix for entire stack as product
    m = np.eye(2,dtype=np.complex128)
    for i in np.arange(n.size):
        m = np.dot(m,np.array([[np.cos(phi[i]),-1j*np.divide(np.sin(phi[i]),gamma[i])],[-1j*np.multiply(np.sin(phi[i]),gamma[i]),np.cos(phi[i])]]))
    
    #calculate R & T
    #reflection coefficient
    r = (gammac*m[0,0]+gammac*gammas*m[0,1]-m[1,0]-gammas*m[1,1])/(gammac*m[0,0]+gammac*gammas*m[0,1]+m[1,0]+gammas*m[1,1])
    r_amp = np.square(np.abs(r))
    return r_amp


#calculate and return the transmission amplitude for the system in given POL state
@jit(nopython=True)
def trans_amp(rho, ang_of_inc, wavelength, n, l, n_cover, n_subst):
    #calculate parameters from the user input
    #wavenumber
    k = 2*np.pi/wavelength;
    #impedance of FS, ohms
    z0 = 376.730313667
    #direction cosines
    beta = np.multiply(n_cover,np.sin(ang_of_inc*np.pi/180))
    ang = np.arcsin(np.divide(beta,n))
    alpha = np.multiply(n,np.cos(ang))
    ang_s = np.arcsin(beta/n_subst)
    #phase thickness
    phi = k*np.multiply(alpha,l)
    #gamma parameter different for TE and TM
    if rho == 0:
        gamma = alpha/z0
        #gamma in cover(above) and substrate(below)
        gammac = n_cover*np.cos(ang_of_inc*np.pi/180)/z0
        gammas = n_subst*np.cos(ang_s)/z0
    elif rho == 1:
        gamma = z0*np.divide(np.cos(ang),n)
        gammac = z0*np.cos(ang_of_inc*np.pi/180)/n_cover
        gammas = z0*np.cos(ang_s)/n_subst
    else:
        #exception handling
        print("Exception in Pol State. Ending...")
        return
        
    #calculate transfer matrix for entire stack as product
    m = np.eye(2,dtype=np.complex128)
    for i in np.arange(n.size):
        m = np.dot(m,np.array([[np.cos(phi[i]),-1j*np.divide(np.sin(phi[i]),gamma[i])],[-1j*np.multiply(np.sin(phi[i]),gamma[i]),np.cos(phi[i])]]))
    
    #calculate R & T
    #transmission coefficient
    t = (2*gammac)/(gammac*m[0,0]+gammac*gammas*m[0,1]+m[1,0]+gammas*m[1,1])
    t_amp = (np.real(gammas)/np.real(gammac))*np.square(np.abs(t))
    return t_amp


#calculate and return ellipsometric parameters for the system
@jit(nopython=True)
def ellips(ang_of_inc, wavelength, n, l, n_cover, n_subst):
#     if np.any(l <= 1E-12):
#         print('OOB thickness!')
#         return(90.,90.)
    #calculate parameters from the user input
    #wavenumber
    k = 2*np.pi/wavelength;
    #impedance of FS, ohms
    z0 = 376.730313667
     #direction cosines
    beta = np.multiply(n_cover,np.sin(ang_of_inc*np.pi/180))
    ang = np.arcsin(np.divide(beta,n))
    alpha = np.multiply(n,np.cos(ang))
    ang_s = np.arcsin(beta/n_subst)
    #phase thickness
    phi = k*np.multiply(alpha,l)
    #gamma parameter different for TE and TM
    gamma0 = alpha/z0
    gamma1 = z0*np.divide(np.cos(ang),n)
    #gamma in cover(above) and substrate(below)
    gammac0 = n_cover*np.cos(ang_of_inc*np.pi/180)/z0
    gammas0 = n_subst*np.cos(ang_s)/z0   
    gammac1 = z0*np.cos(ang_of_inc*np.pi/180)/n_cover
    gammas1 = z0*np.cos(ang_s)/n_subst

    #calculate transfer matrix for TE polarization (rho = 0)
    m0 = np.eye(2,dtype=np.complex128)
    for i in np.arange(n.size):
        m0 = np.dot(m0,np.array([[np.cos(phi[i]),-1j*np.divide(np.sin(phi[i]),gamma0[i])],[-1j*np.multiply(np.sin(phi[i]),gamma0[i]),np.cos(phi[i])]]))
    #reflection coefficient
    r0 = (gammac0*m0[0,0]+gammac0*gammas0*m0[0,1]-m0[1,0]-gammas0*m0[1,1])/(gammac0*m0[0,0]+gammac0*gammas0*m0[0,1]+m0[1,0]+gammas0*m0[1,1])

    #calculate transfer matrix for TM polarization (rho = 1)
    m1 = np.eye(2,dtype=np.complex128)
    for i in np.arange(n.size):
        m1 = np.dot(m1,np.array([[np.cos(phi[i]),-1j*np.divide(np.sin(phi[i]),gamma1[i])],[-1j*np.multiply(np.sin(phi[i]),gamma1[i]),np.cos(phi[i])]]))
    #reflection coefficient
    r1 = (gammac1*m1[0,0]+gammac1*gammas1*m1[0,1]-m1[1,0]-gammas1*m1[1,1])/(gammac1*m1[0,0]+gammac1*gammas1*m1[0,1]+m1[1,0]+gammas1*m1[1,1])
 
    #calculate Psi and Delta
    psi = np.arctan(np.abs(r1/r0))*(180/np.pi)
    delta = (2*np.pi-np.imag(np.log(r0/r1))-(np.imag(r1/r0)))  #From Giuseppe TMM code
    return (psi,delta)
    
    
#convert psi and delta into n and k values
@jit(nopython=True)
def ellip2nk(psi,delta,th,n_cover):
    #expecting angles in degrees
    th *= np.pi/180
    psi *= np.pi/180
    delta *= np.pi/180
    ncomp = (np.sqrt(1-(4*np.sin(th)*np.sin(th)*np.tan(psi)*np.exp(1j*delta))+(2*np.tan(psi)*np.exp(1j*delta))+(np.tan(psi)*np.tan(psi)*np.exp(1j*delta)))*n_cover*np.sin(th))/(np.cos(th)*(1+(np.tan(psi)*np.exp(1j*delta))))
    n = np.real(ncomp)
    k = np.imag(ncomp)
    return (n,k)


#convert n&k to epsilon
@jit(nopython=True)
def nk2eps(n,k):
    e1 = np.square(n)-np.square(k)
    e2 = 2*np.multiply(n,k)
    return (e1,e2)
