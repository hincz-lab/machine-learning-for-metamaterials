import numpy as np
from scipy.special import wofz
#Implementation of BB model for several metals
#Rakic et at 1998

#Define metal parameters as given in Rakic paper
# -->Is there a more efficient way then dictionaries??
Ag = {
    'f':np.array([0.821,0.050,0.133,0.051,0.467,4.000]), 
    'g':np.array([0.049,0.189,0.067,0.019,0.117,0.052]), 
    'w':np.array([0,2.025,5.185,4.343,9.809,18.56]),
    's':np.array([0,1.894,0.665,0.189,1.170,0.516]),
    'wp':9.01
}

Au = {
    'f':np.array([0.770,0.054,0.050,0.312,0.719,1.648]), 
    'g':np.array([0.050,0.074,0.035,0.083,0.125,0.179]), 
    'w':np.array([0,0.218,2.885,4.069,6.137,27.97]),
    's':np.array([0,0.742,0.349,0.830,1.246,1.795]),
    'wp':9.03
}

Cu = {
    'f':np.array([0.562,0.076,0.081,0.324,0.726]), 
    'g':np.array([0.030,0.056,0.047,0.113,0.172]), 
    'w':np.array([0,0.416,2.849,4.819,8.136]),
    's':np.array([0,0.562,0.469,1.131,1.719]),
    'wp':10.83
}

Al = {
    'f':np.array([0.526,0.213,0.060,0.182,0.014]), 
    'g':np.array([0.047,0.312,0.315,1.587,2.145]), 
    'w':np.array([0,0.163,1.561,1.827,4.495]),
    's':np.array([0,0.013,0.042,0.256,1.735]),
    'wp':14.98
}

Be = {
    'f':np.array([0.081,0.066,0.067,0.346,0.311]), 
    'g':np.array([0.035,2.956,3.962,2.398,3.904]), 
    'w':np.array([0,0.131,0.469,2.827,4.318]),
    's':np.array([0,0.277,3.167,1.446,0.893]),
    'wp':18.51
}

Cr = {
    'f':np.array([0.154,0.338,0.261,0.817,0.105]), 
    'g':np.array([0.048,4.256,3.957,2.218,6.983]), 
    'w':np.array([0,0.281,0.584,1.919,6.997]),
    's':np.array([0,0.115,0.252,0.225,4.903]),
    'wp':10.75
}

Ni = {
    'f':np.array([0.083,0.357,0.039,0.127,0.654]), 
    'g':np.array([0.022,2.820,0.120,1.822,6.637]), 
    'w':np.array([0,0.317,1.059,4.583,8.825]),
    's':np.array([0,0.606,1.454,0.379,0.510]),
    'wp':15.92
}

Pd = {
    'f':np.array([0.330,0.769,0.093,0.309,0.409]), 
    'g':np.array([0.009,2.343,0.497,2.022,0.119]), 
    'w':np.array([0,0.066,0.502,2.432,5.987]),
    's':np.array([0,0.694,0.027,1.167,1.331]),
    'wp':9.72
}

Pt = {
    'f':np.array([0.333,0.186,0.665,0.551,2.214]), 
    'g':np.array([0.080,0.498,1.851,2.604,2.891]), 
    'w':np.array([0,0.782,1.317,3.189,8.236]),
    's':np.array([0,0.031,0.096,0.766,1.146]),
    'wp':9.59
}

Ti = {
    'f':np.array([0.126,0.427,0.218,0.513,0.0002]), 
    'g':np.array([0.067,1.877,0.100,0.615,4.109]), 
    'w':np.array([0,1.459,2.661,0.805,19.86]),
    's':np.array([0,0.463,0.506,0.799,2.854]),
    'wp':7.29
}

W = {
    'f':np.array([0.197,0.006,0.022,0.136,2.648]), 
    'g':np.array([0.057,3.689,0.227,1.433,4.555]), 
    'w':np.array([0,0.481,0.985,1.962,5.442]),
    's':np.array([0,3.754,0.059,0.273,1.912]),
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
    #UPDATE THIS WHEN CHANGING CODE!!!
    print('20191115 13:00:00')
        

#get the complex dielectric function in wavelength range
def eps_material(mat,waverange,numosc=5):
    pen = np.divide(1239.84193E-9,waverange)
    if numosc <= 0:
        numosc = 1
    m = mats.get(mat)
    #free electron eps
    epsf = 1-np.divide(np.power(np.sqrt(m.get('f')[0])*m.get('wp'),2),np.multiply(pen,pen+1j*m.get('g')[0]))
    #bound electron eps
    epsb = 0 + 1j*0
    for i in range(np.min([(m.get('f').size)-1,numosc])):
        a = np.sqrt(np.power(pen,2)+np.multiply(1j*m.get('g')[i+1],pen))
        epsb += np.multiply(np.divide(1j*np.sqrt(np.pi)*m.get('f')[i+1]*m.get('wp')*m.get('wp'),np.multiply(2*np.sqrt(2)*m.get('s')[i+1],a)),(wofz(np.divide(np.subtract(a,m.get('w')[i+1]),np.sqrt(2)*m.get('s')[i+1]))+wofz(np.divide(a+m.get('w')[i+1],np.sqrt(2)*m.get('s')[i+1]))))
        #complex dielectric function
    return (epsf+epsb)

    
#convert complex eps to n and k values    
def eps2nk(eps):
    n = np.real(np.sqrt(eps))
    k = np.imag(np.sqrt(eps))
    #complex refractive index
    return (n+1j*k) 

def nk_material(mat,waverange):
    e = eps_material(mat,waverange)
    n = eps2nk(e)
    return n

#end