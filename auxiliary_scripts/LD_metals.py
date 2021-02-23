import numpy as np
#Implementation of Lorenz-Drude model for several metals
#Rakic 1998

#Define metal parameters as given in Rakic paper
# -->Is there a more efficient way then dictionaries??
Ag = {
    'f':np.array([0.845,0.065,0.124,0.011,0.840,5.646]), 
    'g':np.array([0.048,3.886,0.452,0.065,0.916,2.419]), 
    'w':np.array([0,0.816,4.481,8.185,9.083,20.29]),
    'wp':9.01
}

Au = {
    'f':np.array([0.760,0.024,0.010,0.071,0.601,4.384]), 
    'g':np.array([0.053,0.241,0.345,0.870,2.494,2.214]), 
    'w':np.array([0,0.415,0.830,2.969,4.304,13.32]),
    'wp':9.03
}

Cu = {
    'f':np.array([0.575,0.061,0.104,0.723,0.638]), 
    'g':np.array([0.030,0.378,1.056,3.213,4.305]), 
    'w':np.array([0,0.291,2.597,5.300,11.18]),
    'wp':10.83
}

Al = {
    'f':np.array([0.532,0.227,0.050,0.166,0.030]), 
    'g':np.array([0.047,0.333,0.312,1.351,3.382]), 
    'w':np.array([0,0.162,1.544,1.808,3.473]),
    'wp':14.98
}

Be = {
    'f':np.array([0.084,0.031,0.140,0.530,0.130]), 
    'g':np.array([0.035,1.664,3.395,4.454,1.802]), 
    'w':np.array([0,0.100,1.032,3.183,4.604]),
    'wp':18.51
}

Cr = {
    'f':np.array([0.168,0.151,0.150,1.149,0.825]), 
    'g':np.array([0.047,3.175,1.305,2.676,1.335]), 
    'w':np.array([0,0.121,0.543,1.970,8.775]),
    'wp':10.75
}

Ni = {
    'f':np.array([0.096,0.100,0.135,0.106,0.729]), 
    'g':np.array([0.048,4.511,1.334,2.178,6.292]), 
    'w':np.array([0,0.174,0.582,1.597,6.089]),
    'wp':15.92
}

Pd = {
    'f':np.array([0.330,0.649,0.121,0.638,0.453]), 
    'g':np.array([0.008,2.950,0.555,4.621,3.236]), 
    'w':np.array([0,0.336,0.501,1.659,1.715]),
    'wp':9.72
}

Pt = {
    'f':np.array([0.333,0.191,0.659,0.547,3.576]), 
    'g':np.array([0.080,0.517,1.838,3.668,8.517]), 
    'w':np.array([0,0.780,1.314,3.141,9.249]),
    'wp':9.59
}

Ti = {
    'f':np.array([0.148,0.899,0.393,0.187,0.001]), 
    'g':np.array([0.082,2.276,2.518,1.663,1.762]), 
    'w':np.array([0,0.777,1.545,2.509,19.43]),
    'wp':7.29
}

W = {
    'f':np.array([0.206,0.054,0.166,0.706,2.590]), 
    'g':np.array([0.064,0.530,1.281,3.332,5.836]), 
    'w':np.array([0,1.004,1.917,3.580,7.498]),
    'wp':13.22
}


mats = {
    'Ag':Ag,
    'Au':Au,
    'Cu':Cu,
    'Al':Al,
    'Be':Be,
    'Cr':Cr,
    'Ni':Ni,
    'Pd':Pd,
    'Pt':Pt,
    'Ti':Ti,
    'W':W
}
#List the available materials
def materials():
    for key in mats:
        print(key)

#Print the current version data to the terminal
def version():
    #UPDATE THIS WHEN CHANGING CODE
    print('20191018 10:30')
        
#convert complex eps to n and k values    
def eps2nk(eps):
    n = np.real(np.sqrt(eps))
    k = np.imag(np.sqrt(eps))
    #complex refractive index
    return (n-1j*k) 
        
#get the complex dielectric function in wavelength range
def eps_material(mat,waverange):
    pen = np.divide(1239.84193E-9,waverange)
    m = mats.get(mat)
    #free electron eps
    epsf = 1-np.divide(np.power(np.multiply(np.sqrt(m.get('f')[0]),m.get('wp')),2),np.multiply(pen,np.subtract(pen,1j*m.get('g')[0])))
    #bound electron eps
    epsb = 0 + 1j*0
    for i in range((m.get('f').size)-1):
        epsb += np.divide(np.multiply(m.get('f')[i+1],np.power(m.get('wp'),2)),(np.subtract(np.power(m.get('w')[i+1],2),np.power(pen,2))+np.multiply(pen,1j*m.get('g')[i+1]))) 
    #complex dielectric function
    return (epsf+epsb)

def nk_material(mat,waverange):
    e = eps_material(mat,waverange)
    n = eps2nk(e)
    return n

        
#calculate reflectivity from semiconductor-metal interface
def refl(ri,n_c):
    ref = np.divide(np.power(n_c-np.real(ri),2)+np.power(-np.imag(ri),2),np.power(n_c+np.real(ri),2)+np.power(-np.imag(ri),2))
    return ref