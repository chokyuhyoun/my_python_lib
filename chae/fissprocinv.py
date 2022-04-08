from __future__ import absolute_import, division

# -*- coding: utf-8 -*-
"""
This package contains routines and classes useful for the spectral Analysis of FISS Data

 - Mathematical/satatistical functions 
 
    forpos(): penalaty function for positivity
    GetNoise(): standard noise from an 1-D array data
    Voigt() : Voigt function
        
 - FISS processing  functions
 
    Trad(): Radiation temperature from intensity
    Linecenter():  central wavelegnth of a line
    Recalibrate():  recalibrate wavelengths
    ftandxi():   Temperarure and nonthermal speed from Doppler widths
    fwidth():  Dopplwer widths from temeprature and nontehrmal speed
    fpure():  specifying wavelengths free from blending by weak lines
    fwvline() : central wavegnth of eithr H alpha ot Ca II 8542 line
    fsel() :  specifying wavelengths used for the model fitting
    fsigma() : speficifying noise in intensity
    Photolinewv() : specifying the wavelengths of the line for the photospheric 
            velocity    
    FissBadsteps():  number of bad steps in the raster scan
    FissStrayAsym(): correction for stray light and far wing blue-red asymmetry 
    fTauH2O(): optical thickness model of the terrestrial lines of water vapor 
    FissGetTlines():   determine the model parameters of the terrestrial lines
    FissCorTlines():   correct the intensity profile for the terrestrail lines
    fTauS(): optical thickness model of the Co I line in sunspots.
    FissGetSline(): determien the model parameters of the Co Line
    FissCorSline() : correction for the Co I line
    
 - Three Layer Spectral Inversion functions
    
    ThreeLayerPar(): description of each parameter
    ThreeLayerParControl(): indexes of the free parameters and the constrained parameters
    ThreeLayerDefault() :  default values of parameters and their prior deviations 
    fPhi():  absorption coefficient with damping 
    fPhie(): absorption coefficient without damping
    fThreeLayer():  intensity profiles of a line at three levels 
    ThreeLayerRes(): residual array to be used  for fitting based on least_squares
    ThreeLayerInitial(): initial guess of model parameters  and  their prior deviations
    ThreeLayerModel(): three-layer spectrla inversion of a line profile
    ThreeLayerRadLoss(): radiative losses from the three-layer model parameters

 - Hybrid ( Three Layer + cloud) Spectrla Inversion

    fCloud(): intensity profile of a cloud mode 
    ThreeLayerCloudModel():  hybrid inversion (three layer model + cloud model)

 - GuiThreeLayerModel:  Graphical user interface class for Implementing the Three-Layer Inversion of a profile

    selwv(): re-select the wavelengths to be used for the fitting
    selFont(): set the font class for widgets
    DockWidget():  widget configuration
    Readpar(): read parameters from the widgets
    Writepar() : write paramters into the widgets
    Redraw((): redraw plots reflecting the change in the data and parameters
    Initialize(): initialize parameters
    Renew(): update data
    Fit():  do model fitting
    Export() : save the data and parameters into a pickle file.
    Import(): read data and parameters from the file
     
   
Release Version
--------------------------- 
1.0.0: 2021 September 11 by Jongchul Chae
       
@author: Jongchul Chae
"""
package='fissprocinv v1.0.0'

import numpy as np
import matplotlib.pyplot as plt
from fisspy.image.base import alignoffset
from scipy.optimize import least_squares
from fisspy.read import FISS
import fisspy
from astropy.io import fits
from astropy.time import Time 
from time import time,localtime,strftime, sleep
from scipy import interpolate
import matplotlib.animation as animation
from scipy.special import wofz, expn
from numpy import exp,log, log10, sqrt, array,sin, cos
from scipy.signal import savgol_filter
import fisspy.image.base as fissimage 
from PyQt5 import QtGui, QtCore, QtWidgets
import pickle
from sys import stdout
import glob,os
import multiprocessing as mp
#%%     Mathematical/satatistical functions
def forpos(x):
    """
    To deterine the penalaty function for positivity. 
    The penalty is zero for a positive input, and is equal to the negative input.  

    Parameters
    ----------
    x : array_like
        input(s)

    Returns
    -------
    penlaty:  array_like
        output penalty(s) 

    """
    penalty = x*(x<0)
    return penalty
def GetNoise(x):
    """
    To determine standard noise from an 1-D array data, 
    based on the assumption that the 2nd order differences are much smaller 
    than the noisy fluctuation 

    Parameters
    ----------
    x : array_like
        values

    Returns
    -------
    noise
        standard noise

    """
    kernel=array([1.,-2.,1.])
    a=np.convolve(x,kernel)
    return a[2:-2].std()/sqrt(6.)
def Voigt(u, a):
    """
    To determine the Voigt function

    Parameters
    ----------
    u : array_like
        normalized and centered wavelengths
    a : array_like
        dimensionless damaping parameters

    Returns
    -------
    value
        function value

    """
    z = u + 1j*a
    return  np.real(wofz(z) )         

#%%  FISS processing functions

def Trad(intensity, line, intensity0=1.):
    """
    Radiation temperature corresponding to intensity

    Parameters
    ----------
    intensity : array_like
        intensity(s).
    line : str
        line designation.
    intensity0 : float, optional
        the disk center intensity in normalized unit. The default is 1..

    Returns
    -------
    Trad : array_like
        radiation temperature.

    """
    wv=fwvline(line)
    h = 6.63e-27
    kb = 1.38e-16
    c= 3.e10
    I00 = 2*h*c**2/(wv*1.e-8)**4/wv
    if line == 'Ha':
        Icont = 2.84e6
    if line == 'Ca':
        Icont = 1.76e6          
    Ilambda = (intensity/intensity0)*Icont 
    hnuoverk = h*(c/(wv*1.e-8))/kb # 1.43*10000./wv
    Trad =hnuoverk/(log(1+I00/Ilambda))
    return Trad


def Linecenter(wv0, prof0, dn=2):
    """ 

    To determine the central wavelength of an absorption line using the 2-nd polynomial
        fitting of the line core

    Parameters
    ------------
    wv0 : array_like
        wavelengths

    prof0 : array_like
        intensities

    dn : integer, optional
        half number of data points, default=2
        
    Returns
    ------------
    value:  central wavelength
        
    """    
    s=prof0[dn:-dn].argmin()+dn
    prof1=prof0[s-dn:s+dn+1]
    wv1=wv0[s-dn:s+dn+1]            
    coeff=np.polyfit(wv1, prof1, 2)
    return -coeff[1]/(2*coeff[0])

def Recalibrate(wv, prof, line='Ha'):       
    """
    To recalibrate wavelengths of a spectrum 
    
    Parameters
    ------------
    wv:  array_lke
        wavelengths
    prof: array_like
        intensities
        
    line: str, optional (default='Ha')
        designation of the spectral band
        
    Returns    
    ------------
    wvnew:  array_like
        new wavelengths
    """
 
    if line == 'Ha':
        wvline1 = 6559.580 #  Ti II
        dn1=2
        wvline2 = 6562.817   # H alpha
        dn2=3
        dispersion = 0.01905 # A/pix
        method = 1
    elif line =='Ca':    
        wvline2 =  8536.165 # Si I 
        dn1=2
        wvline1 =  8542.091   # Ca II
        dn2=2
        dispersion = -0.02575 # A/pix
        method = 2
        
    if method == 1:  
#        
#  Use two lines to re-determine both dispersion and wavelength reference        
        pline = abs(wv-wvline1) <= 0.3  
        wpix = np.arange(0., len(wv))
        wpix1 =Linecenter(wv[pline], prof[pline], dn=dn1)        
        pline = abs(wv-wvline2) <= 0.3   
        wpix2 =Linecenter(wv[pline], prof[pline], dn=dn2)       
        a=(wvline1-wvline2)/(wpix1-wpix2)  # dispersion
        b=(wpix1*wvline2-wpix2*wvline1)/(wpix1-wpix2)
        wvnew = a*wv+ b  
        
    if method == 2:
#   
#  Use one line to re-determine the wavelength reference  and 
#    and use the given dispersion     
        pline = abs(wv-wvline1) <= 0.3                     
        wpix = np.arange(0., len(wv)) 
        wpix1 =  Linecenter(wpix[pline], prof[pline], dn=dn1)
        wvnew = dispersion*(wpix-wpix1)+wvline1
        
    return wvnew

def ftandxi(DwHa, DwCa):
    """
    To determine hydrogen temperature and nonthermal speed from the Doppler widths
    of the H alpha line and the Ca II 8542 line

    Parameters
    ----------
    DwHa : array_like
        Doppler width(s) of the H alphalline in unit of Angstrom
    DwCa : array_like
        Doppler width(s) of the Ca II 8542 line in unit of Angstrom

    Returns
    -------
    Temp : array_like
        hydgregen temperature(s) in unit of K
    xi : array_like
        nonthermal speed(s) in unit of km/s

    """

    yHa = (DwHa/6562.8*3.e10)**2
    yCa = (DwCa/8542.1*3.e10)**2    
    delt =array(yHa-yCa)
    delt[delt<0]=1.
    Temp = (1.-1./40.)**(-1.)*(1.67e-24/1.38e-16)/2.*delt
    
    delt = array(yCa - yHa/40.)
    delt[delt<0]=1.
    xi = sqrt((1.-1./40.)**(-1.)*delt)/1.e5   # to km/s
    return Temp, xi

def fwidth(Temp, xi):
    """
    To determine the Dopplwer widths of the H alpha line and Ca II 8542 line

    Parameters
    ----------
    Temp : array_like
        hydrogen temperature(s)
    xi : array_like
        nonthermal speed(s)

    Returns
    -------
    DwHa : array_like
        Doppler width(s) of the H alpha line
    DwCa : array_like
        Doppler width(s) of the Ca II 8542 line

    """
    DwHa = 6562.8*sqrt(xi**2+(2*1.38e-16*Temp/1.67e-24)/1.e10)/3.e5
    DwCa = 8542.1*sqrt(xi**2+(2*1.38e-16*Temp/(40*1.67e-24))/1.e10)/3.e5
    return DwHa, DwCa

def fpure(wv, line='Ha'):
    """
    To determine whether blending by weak lines is absent or not at the
    specified wavelength(s)

    Parameters
    ----------
    wv : array_like
        absolute wavelength(s) in unit of Angstrom
    line : str, optional
        spectral line designation. The default is 'Ha'.

    Returns
    -------
    pure : boolean array_like
        True if blending is not serious.

    """
    if line == 'Ha':
        hw = 0.15
        pure = (abs(wv-(6562.82-3.20)) > hw) * (abs(wv-(6562.82-3.2)) > hw)  \
             * (abs(wv-(6562.82-2.55)) > hw) * (abs(wv-(6562.82-2.23)) > hw)  \
             * (abs(wv-(6562.82+2.65) ) > hw) *  (abs(wv-(6562.82+2.69) ) > hw) \
             * (abs(wv-(6562.82-4.65) ) > hw) *    (abs(wv-(6562.82+3.80) ) > hw) 
               
    if line == 'Ca':
        hw = 0.15
        pure =  (abs(wv-(8542.09-5.95)) > hw) * (abs(wv-8536.45 ) > hw) \
            * (abs(wv-8536.68 ) > hw) * (abs(wv-8537.95) > hw) \
            * (abs(wv-8538.25 ) > hw) * (abs(wv- (8542.09+2.2)) > hw) \
            *(abs(wv-8542.09+2.77) > hw) \
            * (abs(wv-wv.max()) > hw)* (abs(wv-wv.min()) > hw)
                
    return pure  

def fwvline(line):
    """
    To yield the central wavelength of a line

    Parameters
    ----------
    line : str
        line designation.

    Returns
    -------
    wvline : float
        laboratory wavelength of the line.

    """
    if line == 'Ha': wvline = 6562.817
    if line == 'Ca': wvline = 8542.091
    return wvline 
def fsel(wv1, line):
    """
    To determine whether the data are to be selected or not for fitting

    Parameters
    ----------
    wv1 : array_like
        wavelengths.
    line : str
        line designation.

    Returns
    -------
    sel : booleab array_like
        True if selected for fitting.

    """
    sel=fpure(wv1+fwvline(line), line=line)
    if line == 'Ha':
        sel=sel*(abs(wv1)<3.)
    else:    
        sel=sel*(abs(wv1)<3.)
    return sel
def fsigma(wv1, intensity, line='Ha'):
    """
    To calculate the data noise of intensity 

    Parameters
    ----------
    wv1 : array_like
        wavelengths measured from line center.
    intensity : TYPE
        intensities normalized by continuum.
    line : str, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    sigma : array_like
        standard noises.

    """
    
    if line == 'Ha':
        sigma0 = 0.01 
    elif line == 'Ca':
        sigma0 = 0.01
    sigma = sigma0*sqrt(intensity)    
    return sigma
def Photolinewv(line, wvmin, wvmax):
    """
    To specicy the spectral line used to determine photospheric velocity 

    Parameters
    ----------
    line : str
        spectral band designation.
    wvmin : float
        minimum wavelength of the spectral band.
    wvmax : TYPE
        maximum wavelength of the spectral band.

    Returns
    -------
    wvp : float
        laboratory wavelength of the photosperic line.
    dwv : TYPE
        half of the wavelength range to be used 

    """
    if line == 'Ha':
        wvp, dwv = 6559.580, 0.25
    if line == 'Ca':
        wvp,dwv = 8536.165, 0.25 
        if (wvp > (wvmin+2*dwv))*(wvp < (wvmax-2*dwv)): return(wvp, dwv)
        wvp,dwv = 8548.079*(1+(-0.62)/3.e5), 0.25 
        
    return (wvp, dwv)        

#%%
def FissBadsteps(Fissfile):
    """
    To determine the number of bad steps in the  FISS raster scan

    Parameters
    ----------
    Fissfile : str
        FISS file name.

    Returns
    -------
    nbad : integer
        number of bad steps.
    icont : 2d array
        continuum-like raster image.

    """
    fiss = FISS(Fissfile)
    fiss.data = np.flip(fiss.data, 1)    
    icont=(fiss.data[:,:, 50:60]).mean(2)
    s=icont.shape
    N =s[1]
    nbad=0       
    b, bmag=0.,0.
    for i in range(1, N):       
         if i == 1:
             a = icont[:,i-1]
             amag = sqrt((a**2).mean())
         else:
             a = b
             amag = bmag
         b = icont[:,i] 
         bmag = sqrt((b**2).mean())
         if (a*b).mean() <= 0.7*(amag*bmag):  
             nbad= nbad+1
    return  nbad, icont

def FissStrayAsym(wv1, prof0, avprof, line='Ha', pure=None, epsilon=0.027, zeta=0.055):
    """
    To correct a spectral lie profile for stray light and far wing red-blue asymmetry

    Parameters
    ----------
    wv1 : array-like
        wavelengths measured from the line center 
    prof0 : array_lie
        intensities to be corrected
    avprof : array_like
        spatially averaged line profile to be used as a reference
    line : str, optional
        line designation. The default is 'Ha'.
    pure : boolean array_like, optional
        True if not blended. The default is None.
    epsilon: float
        fraction of spatial stray light. The defualt is 0.027
    zeta : float
        fraction of spectral stary light. The default is 0.055

    Returns
    -------
    prof : array_like
        line profile corrected for stray light and far wing red-bue asymmetry

    """
    
    if pure is None: pure=fpure(wv1+fwvline(line), line=line)
    fw=abs(wv1-4.5) < 0.1
    prof0fw = prof0[fw].mean()
    avproffw = avprof[fw].mean()
    
    # correcting for stray light
    zeta= 0.055 #0.055
    epsilon= 0.027
    prof = (prof0/prof0fw -zeta)/(1.-zeta)*(prof0fw/avproffw-epsilon)/(1-epsilon)*avproffw          
    conti = pure * (abs(wv1) > 3.)*(abs(wv1)< 4.)

    # correcting for far blue-red wings asymmetry
    p1model=np.poly1d(np.polyfit(wv1[conti], prof[conti], 1))
    p=p1model(wv1)
    p=p/np.maximum(p.mean(), 0.03)
    prof=prof/p
    return prof

#%%   To remove the terrestrial H2O lines from the spectra
def fTauH2O(wv, rtau, dwv, line='Ha'):
    """
    To determine the profile of water vapor optical thickness of the Earth's atmosphere
    

    Parameters
    ----------
    wv : array_like
        wavelengths
    rtau :float
        relative optical thickness
    dwv : float
        wavelength offset of the optical thickness profile
    line : str, optional
        spectral line designation. The default is 'Ha'.

    Returns
    -------
    tau
        optical thickness profile

    """

    if line == 'Ha':
        wvHaline = 6562.817
        wvlines =array([6568.149, 6558.64, 6560.50, 6561.11, 6562.45, 
                 6563.521, 6564.055, 6564.206, 6565.53])-wvHaline+ 0.020-dwv           
        tau0 = array([7., 1.5, 7.5, 4.5, 3.,  5., 4.2, 14.7, 1.5])/100.*2            
        a = 1.4
        w = 0.022
    if line == 'Ca':
        wvCaline = 8542.091
        wvlines =array([8539.895, 8540.817, 8546.22])+ 0.032-wvCaline-dwv     
        tau0 = array([3.3, 8., 3.4])/100.*2.0       
        a = 1.9
        w = 0.034   
    
    taunormal = 0.
    nlines = len(wvlines)
    V0 = Voigt(0.,a) 
    for pline in range(nlines):        
        taunormal = taunormal + tau0[pline]*Voigt((wv-wvlines[pline])/w, a)/V0
    tau = rtau*taunormal        
    return tau

def FissGetTlines(wave, profile, line='Ha'):
    """
    To determine the parameter relevant to the content of water vapor in the Earth's atmosphere

    Parameters
    ----------
    wave : array_like
        wavelengths
    profile : array_like
        intensities
    line : str, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    array_like
        a two-element array to be used for the correction for terrestrial H2O lines

    """
    

    
    if line == 'Ha': 
            s= ((wave-2.)*(wave-1.) < 0 ) 
    if line == 'Ca':
            s= ((wave+1.7)*(wave+1.) < 0 )  
     
    wave1=wave[s]
    profile1=profile[s]/profile.max()        
    sigma=fsigma(wave1, profile1, line=line)
    
    def resTauH2O(par):       
        tau = fTauH2O(wave1, par[0]**2, par[1], line=line) 
        res= np.convolve(profile1*exp(tau), array([-1, 1.]), mode='same')/sqrt(2.)   
        resD = res[1:-2]/sigma[1:-2] 
        resP = array([(par[0]-0.)/1., par[1]/0.01])
        res  = np.append(resD/sqrt(len(resD)), 0.01*resP/sqrt(len(resP)))                
        return res
           
    par=array([0.5, 0.])
    res_lsq= least_squares(resTauH2O, par, max_nfev=50,jac='2-point' )
    par = np.copy(res_lsq.x)
    return par
def FissCorTlines(wave, profile, par, line='Ha'):
    tau = fTauH2O(wave, par[0]**2, par[1], line=line)
    return profile*exp(tau)
        
#%% To remove the Co I line from the  H alpha sunspot spectra 
def fTauS(wv, rtau, dwv):
    """
    To determine the optical thickness profile of the Co I line in the H alpha band

    Parameters
    ----------
    wv : array_like
        wavelengths measured from the center of the H alpha line.
    rtau : array_like
        relative otpical thickness 
    dwv : array_like
        wavelength offset(s)

    Returns
    -------
    tau : array_like
        optical thickness(es).

    """
        
    wvline =  0.593 + dwv        
    tau0 = 0.15   
    a = 0.00
    w = 0.12
    u = (wv-wvline)/w
    taunormal = tau0*Voigt(u,a)/Voigt(0., a) #exp(-u**2)
    tau = rtau*taunormal
    return tau
def FissGetSline(wv, profile):
    """
    To determine the optical thickness parameter of Co I line from the fitting 

    Parameters
    ----------
    wv : array_like
        wavelengths measured from the H alphe line center
    profile : array_like
        intensities

    Returns
    -------
    par: a two-element array  
        optical thicnkess parameter.

    """      
    if profile.max() > 0.95:
        par = array([0., 0.])
    else:    
        par=array([0.5, 0.])
        s = ((wv-0.25)*(wv-1.) < 0 )  
        wv1 = wv[s]
        profile1=profile[s]/profile.max() 
        sigma=fsigma(wv1, profile1, line='Ha')
        
   
        def resTauS(par):
            tau1 = fTauS(wv1, par[0]**2, par[1])
            res = np.convolve(profile1*exp(tau1), array([-1.,2.,-1.]), mode='same')/sqrt(6.)
            resD = res[2:-3]/sigma[2:-3]  
            resP = array([(par[0]-0.)/1., (par[1]-0.)/0.01])
            res  = np.append(resD/sqrt(len(resD)), 0.01*resP/sqrt(len(resP)))     
            return res
        
        res_lsq= least_squares(resTauS, par, max_nfev=50,jac='2-point' )
        par = np.copy(res_lsq.x)
        par[0] = abs(par[0])
    return par
def FissCorSline(wv, profile, par):
    """
    To correct the H alpha spectral profile for the Co I line blending 

    Parameters
    ----------
    wv : array_like
        wavelengths measured from the H alpha line center.
    profile : array_like
        intensity profile to be corrected.
    par : array_like
        Co I line blending parameters.

    Returns
    -------
    profilenew : array_like
        corrrected line profile.

    """
    tau = fTauS(wv, par[0]**2, par[1])
    profilenew = profile*exp(tau)
    return profilenew

#%%
def ThreeLayerPar(index):
    """
    To yield the description of the specified parameter in the three layer model. 
    
    The first 15 parameters (with indice 0 to 14) are the primary parameters 
    neccessary and sufficient to specify the model. The remaining parameters are 
    the secondary parameters that can be caluclated from the other parameters.

    Parameters
    ----------
    index : int
        index of the parameter.

    Returns
    -------
    descript : str
        decription of the parameter.

    """
    version = 1
    
    if version == 1: 
        description = array([ 'vp', 'log eta', 'log wp', 'log ap', 'log Sp', 'log S2',
       'log tau2', 'log tau1', 'v1', 'v0', 'log w1', 'log w0', 'log S1', 'log S0', 'wg', 'epsD',
       'epsP', 'Radloss2', 'Radloss1'  ])     
    descript =   description[index]      
    return descript
def ThreeLayerParControl(line='Ha'):
    """
    To provide the indexes of the free model parameters and 
    the priorly constrained model parameters

    Parameters
    ----------
    line : str, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    free : integer array
        indexes of free parameters to be determined from the fitting.  
    constr : integer array
        indexes of priorly constrained parameters. 

    """
    if line == 'Ha':
        free   = [1,3,4,5,8,9,10,11,12,13]
        constr = [1,3,8,9,10,11,12,13]
       
    else:    
        free   = [1,3,4,5,8,9,10,11,12,13]
        constr = [1,3, 8,9,10,11,12,13]
    return free, constr

def ThreeLayerDefault(line='Ha'):  
    """
    provide the default model parameters and their prior deviations  

    Parameters
    ----------
    line : str, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    par0 : array_like
        default values of the 15-elements parameters.
    psig : array_like
        prior deviations of the parameters 

    """
    
    if line == 'Ha':
       par0qr =   array([0.2,  0.5,  -0.65, 0.5, 0.0, -0.26, log10(5.), log10(5.),   
                          -0.3,  0,   -0.42,  -0.42,  -0.42, -0.87, 0.05]) 
       psigqr =    array([0.5,  0.05, 0.1, 0.05, 0.2,  0.2, 0.1, 0.1,  
                           1.4, 2.2,  0.03,  0.05,   0.02,  0.08, 1.e-4])   
       par0ar =   array([0.2,  0.50,  -0.67, 0.50,  0.00, -0.21, log10(5.), log10(5.),   
                         -0.5,  0.1,   -0.39,  -0.39,  -0.41, -0.74, 0.05]) 
       psigar =    array([0.4,  0.05,  0.1,  0.05,   0.2, 0.2, 1.e-4,  1.e-4,  
                          1.5,  1.8,   0.04,  0.04,  0.03,  0.10, 1.e-4])         
    elif line == 'Ca':
       par0qr =   array([0,  0.4,  -1.3,   1.40,   0.00,  -0.55,  log10(5.), log10(5.),  
                          0, 0.9,   -0.57, -0.57,   -0.34,  -1.16, 0.05]) 
       psigqr =    array([0.5, 0.03,  0.1,  0.05, 0.1,  0.1,   0.1, 0.1, 
                           2.3, 1.4, 0.04, 0.06,    0.04,  0.24, 1.e-4])*1.
       par0ar =   array([0,  0.4,  -1.3,    1.40,   0.02,   -0.44,  log10(5.), log10(5.),  
                          -0.5, 0.5,   -0.57, -0.57,   -0.30,  -0.71, 0.05]) 
       psigar =    array([0.5, 0.03, 0.1,  0.05,  0.1,  0.1,   0.01, 0.2, 
                           2.2, 1.2, 0.04, 0.06,     0.04,  0.21, 0.001])
    
    par0 = (par0qr+par0ar)/2.
    psig = sqrt(0.5*(psigqr**2+(par0-par0qr)**2)+0.5*(psigar**2+(par0ar-par0)**2))*1.5  

    if line == 'Ha':
          par0[10:12]= -0.50
          psig[10:12] = 0.07
          psig[12:14] = [0.03, 0.3]
          psig[8:10] = 3.0        
    elif line == 'Ca':    
          par0[10:12]= -0.70
          psig[10:12] = 0.07
          psig[12:14] = [0.03, 0.3]
          psig[8:10] = 3.0
               
    return par0, psig

def fPhi(wv1, wvc, w, a, line='Ha'):
    """
     To determine  absorption profile in the presence of damping. The profile 
     is normalzied for the peak value to be equal to 1.

    Parameters
    ----------
    wv1 : array-like
        wavelengths from the line center in A.
    wvc : float
        central wavelength from the line center in A.
    w : float
        Doppler width in A.
    a : float
        dimensioness damping parameter.
    line : str, optional
        line designatio, either 'Ha' or 'Ca'. The default is 'Ha'.

    Returns
    -------
    Phi : array-like
        values of absorpsion coefficient.

    """
    if line == 'Ha':
         u=(wv1-wvc)/w
         Phi = Voigt(u, a)/Voigt(0., a)
    if line == 'Ca':
         u=(wv1-wvc)/w   
         dwv = array([.0857, .1426, .1696, .1952, .2433,.2871])-0.0857
         dA = 10**(array([6.33, 4.15, 3.47, 4.66, 1.94, 3.61])-6.33)
         Phi = 0.  
         for i in [0]:   
             Phi=Phi+Voigt(u-dwv[i]/w, a)*dA[i]
         Phi = Phi/Voigt(0.,a)
    return Phi    

def fPhie(wv1, wvc, w, line='Ha'):
    """
    To determine Gaussian absorption profile normalized to have peak value of 1 

    Parameters
    ----------
    wv1 : array-like
        wavelengths from line center in A.
    wvc : float
        central wavelength from line center in A.
    w : float
        Doppler width in A.
    line :str, optional
        line designation, either 'Ha' or 'Ca'. The default is 'Ha'.

    Returns
    -------
    Phi : array-like
        normalized absorption coefficent.

    """
    if line == 'Ha':
         u=(wv1-wvc)/w
         Phi = exp(-u**2)
    if line == 'Ca':
         u=(wv1-wvc)/w   
         dwv = array([.0857, .1426, .1696, .1952, .2433,.2871])-0.0857
         dA = 10**(array([6.33, 4.15, 3.47, 4.66, 1.94, 3.61])-6.33)
         Phi = 0.  
         for i in [0]:
             Phi=Phi+exp(-(u-dwv[i]/w)**2)*dA[i]
    return Phi    

def fThreeLayer(wv1, p, line='Ha', phonly=False):
    
    """
    To calculate the intensity profiles of a line at three levels  
    

    Parameters
    ----------
    wv1 : array_like
        wavelengths measured from the line center.
    p : array_like
        array of 15 parameters.
    line : str, optional
        line designation. The default is 'Ha'.
    phonly : boolean, optional
        True if the intensity profile at the top of
        the photospehre only I2 is required. The default is False.

    Returns
    -------
    I0 : array_like
        intensity profile at the top of the chromosphere    
    I1 : array_like
        intensity profile in the middle of the chromosphere
    I2 : array_like
        intensity profile at the top of the photosphere

    """
    
#  Change of Variables    
    wvline = fwvline(line)
    wvp = p[0]/3.e5*wvline
    eta, wp, ap, Sp, S2 =10.**p[1:6]    
    tau2, tau1 = 10**p[6:8]    
    wv01, wv00 =p[8:10]/3.e5*wvline
    w1, w0 = 10**p[10:12] 
    S1, S0 = 10**p[12:14]
    wg = p[14]
    wv02 = wvp
    w2 = wp
    
# Photosphereic Contribution
#    V0 = Voigt(0., ap)
#    u=(wv1-wvp)/wp
    rlamb = eta*fPhi(wv1, wvp, wp, ap, line=line) + 1  #Voigt(u, ap)/V0 + 1    
    I2=S2 + (Sp-S2)/rlamb 
    
    if phonly:  return I2, I2, I2
       
#    npoint = 5
    xvalues5p = array([-0.9062, -0.5385,   0.,     0.5385, 0.9062])
    weights5p = array([ 0.2369,  0.4786,   0.5689 ,0.4786, 0.2369])    
    xvalues3p =  array([-0.774597,0, 0.774597])
    weights3p =   array([0.55556, 0.888889, 0.555556])
 
# Lower Chromosphere
    x=1.
    dummy =0.
    for j in range(3):
        xx = xvalues3p[j]*(x-0.)/2+(x+0.)/2
        wvcenter =(wv01+(wv02-wv01)*xx)
        width=(w1 + (w2-w1)*xx)
        a =  (wg*xx) /(w1 + (w2-w1)*xx)      
        dummy += weights3p[j]*fPhi(wv1, wvcenter, width, a, line=line) #Voigt(u,a)/Voigt(0,a) 
    taulamb = tau2*x/2.*dummy    
    a0 = S0
    a1 = (-1.5*S0+2*S1-0.5*S2)
    a2 = (0.5*S0-S1+0.5*S2)

    def Sfromx(x, x0):
        return a0+a1*(x+x0)+ a2*(x+x0)**2
    Integral = 0.
    for i in range(3):        
        x = xvalues3p[i]*(1-0.)/2+(1+0.)/2        
        dummy = 0.
        for j in range(3):
            xx = xvalues3p[j]*(x-0.)/2+(x+0.)/2
#            u = (wv1-(wv01+(wv02-wv01)*xx))/(w1 + (w2-w1)*xx)
            wvcenter =(wv01+(wv02-wv01)*xx)
            width=(w1 + (w2-w1)*xx)
            a =   (wg*xx)/width     
            dummy += weights3p[j]* fPhi(wv1, wvcenter, width, a, line=line) #Voigt(u,a)/Voigt(0,a) 
        tlamb = tau2*x/2.*dummy
           
        #ulamb = (wv1-(wv01+(wv02-wv01)*x))/(w1 + (w2-w1)*x)
        wvcenter = (wv01+(wv02-wv01)*x)
        width = (w1 + (w2-w1)*x)
        a = (wg*x) /width
        S = Sfromx(x, 1.)
        Integral += weights3p[i]*S*exp(-tlamb)* fPhi(wv1, wvcenter, width, a, line=line) # Voigt(ulamb,a)/Voigt(0,a)        
    I1 = I2*exp(-taulamb) + tau2*1./2.*Integral

# Upper Chromosphere         
    x=1.
    dummy =0.
    for j in range(3):
        xx = xvalues3p[j]*(x-0.)/2+(x+0.)/2
       # u = (wv1-(wv00+(wv01-wv00)*xx))/(w0 + (w1-w0)*xx)
        wvcenter =(wv00+(wv01-wv00)*xx)
        width=(w0 + (w1-w0)*xx) #(w1 + (w2-w1)*xx)
        dummy += weights3p[j]*fPhie(wv1, wvcenter, width, line=line) #exp(-u**2) # Voigt(u,0) 
    taulamb = tau1*x/2.*dummy
    Integral = 0.
    for i in range(3):      
        x = xvalues3p[i]*(1-0.)/2+(1+0.)/2   
        dummy = 0.        
        for j in range(3):
            xx = xvalues3p[j]*(x-0.)/2+(x+0.)/2
           # u = (wv1-(wv00+(wv01-wv00)*xx))/(w0 + (w1-w0)*xx)
            wvcenter =wv00+(wv01-wv00)*xx
            width = w0 + (w1-w0)*xx
            dummy += weights3p[j]*fPhie(wv1, wvcenter, width, line=line) #exp(-u**2) #Voigt(u,0) 
        tlamb = tau1*x/2.*dummy           
        #ulamb = (wv1-(wv00+(wv01-wv00)*x))/(w0 + (w1-w0)*x) 
        wvcenter = (wv00+(wv01-wv00)*x)
        width = (w0 + (w1-w0)*x)
        S = Sfromx(x, 0.)
        Integral +=   weights3p[i]*S*exp(-tlamb)*fPhie(wv1, wvcenter, width,  line=line) # exp(-ulamb**2) #Voigt(ulamb,0) 
        
    I0 = I1*exp(-taulamb) + tau1*1./2.*Integral
      
    return I0, I1, I2

def ThreeLayerRes(pf, wv1, intensity,  p0, psig, p, line, free, constr):
    """
     To calculate the residual array to be used  for fitting based on least_squares

    Parameters
    ----------
    pf : array_like
        free parameters to be determined.
    wv1 : array_like
        wavelengths measured from line center .
    intensity : array_Lke TYPE
        intensities DESCRIPTION.
    p0 : array_lke TYPE
        default model parameters .
    psig : array_lke TYPE
        prior deviation of model parameters .
    p : array_lke TYPE
        model parameters .
    line : str
        line designation.
    free  : integer, array_like
         indexes of free parameter
    constr : integer,  array_like
         indexes of constrained parameter      
    Returns
    -------
    Res : array_like TYPE
        residuals DESCRIPTION.

    """
   # free, constr = ThreeLayerParControl(line=line)
    p[free]=pf
    model, modelph, modelch =fThreeLayer(wv1,p, line=line)

    sig = fsigma(wv1, intensity, line=line)    
    resD = (intensity-model)/sig 
    
    resC = (p[constr]-p0[constr])/psig[constr]
    resC = np.append(resC, (p[10]-p[11])/sqrt(psig[10]**2+psig[11]**2))
    resC = np.append(resC, (p[8]-p[9])/sqrt(psig[8]**2+psig[9]**2))
    
    logS1, logS0 = p[12:14]
    logSp, logS2 = p[4:6]
    logw1, logw0 = p[10:12]
    
    if line == 'Ha': 
        resC = np.append(resC, forpos((logw1-logw0)/0.03) )
        resC = np.append(resC, forpos((logS2-logS1)/0.03) )
        resC = np.append(resC, forpos((logS1-logS0)/0.03) )
        resC = np.append(resC, forpos((logS0+1.5)/0.03) )

    if line == 'Ca':
        resC = np.append(resC, forpos((logw1-logw0)/0.03) )
        resC = np.append(resC, forpos((logS1-logS2)/0.03) )
        resC = np.append(resC, forpos((logS1-logS0)/0.03) )
        resC = np.append(resC, forpos((logS0+1.5)/0.03) )
      
    Res = np.append(resD/sqrt(len(resD)), resC/sqrt(len(resC)))
    
    return Res 

#   
def ThreeLayerInitial(wv1, prof,  line='Ha'):
    """
    Determine the initial guess of model parameters  and  their prior deviations

    Parameters
    ----------
    wv1 : array_like TYPE
        wavelengths measured from line center DESCRIPTION.
    prof : array_like TYPE
        intesity profile  DESCRIPTION.
    line : str, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    par : array_like
         model parameters
        
    psig : array_like   
        prior deviations of model parameters
    

    """
# Preparation
    free, constr = ThreeLayerParControl(line=line)
    par0, psig0 = ThreeLayerDefault(line=line) 
    par = np.copy(par0)
    psig = np.copy(psig0)
    profav = savgol_filter(prof, window_length=7, polyorder=2)   
    wvline = fwvline(line)
    pure = fpure(wv1+wvline, line=line)
    sel = fsel(wv1, line)  

    fprofav = interpolate.interp1d(wv1[pure], profav[pure])  
    
    wvp, dwv= Photolinewv (line, wv1.min()+wvline, wv1.max()+wvline)
    pline = abs(wv1+wvline-wvp) <= 2*dwv    
    
    dwc=Linecenter(wv1[pline]+wvline-wvp, profav[pline], dn=2)
    vp = dwc/wvp*3.e5
    par[0] = vp  
 
# Photospheric Layer
        
    eta = 10**par[1]
    ap = 10**par[3]
    wv1a, wv1b = 1.5, 4.
    wv1a = wv1a+vp/3.e5*wvline
    inta = fprofav(wv1a)
    wv1an = -wv1a+vp/3.e5*wvline    
    intan = fprofav(wv1an)
    inta = 0.5*(inta+intan)
    intb = (fprofav(wv1b)+fprofav(-wv1b))/2.    
   
    Tav = Trad(intb,line)
    wha, wca = fwidth(Tav, 1.)
   
    if line == 'Ha':
        wp = wha       
    else:
        wp = wca
    par[2] = log10(wp)

    eta  = 10**par[1]    
    qa = 1./(eta*Voigt(wv1a/wp, ap)/Voigt(0.,ap)+1 )
    qb = 1./(eta*Voigt(wv1b/wp, ap)/Voigt(0.,ap)+1 )
    d1 = (inta-intb)/(qa-qb)
    d0 = (inta*qb-intb*qa)/(qb-qa)
    S2 = d0
    Sp = d1 + S2
    par[4] = log10(np.maximum(Sp, intb*0.5))
    par[5] = log10(np.maximum(S2, intb*0.02))
    if line == 'Ha': 
        par[14]   = wp*ap*0.20
        par[12] =  log10(profav[sel].max())*0.4 + log10(profav[sel].min())*0.6  
    else: 
        par[14]   = wp*ap*0.20
        par[12] =  log10(profav[sel].max())*0.5 + log10(profav[sel].min())*0.5 
    
    int0 = profav[sel].min()
    S0 =  np.maximum(0.7*int0, 10.**-1.5)    
    par[13]=log10(S0)  
    model, modelch, modelph = fThreeLayer(wv1[sel],par, line=line, phonly=True)       
    weight =np.maximum(modelph*1.-profav[sel], 0.)*(abs(wv1[sel])< 0.5)          
    wv0 =( weight*wv1[sel]).sum()/np.maximum(weight.sum(), 1.)
    vc = wv0/wvline*3.e5
    par[9] = vc
    par[8] = vc
            
    return par, psig
def ThreeLayerModel(wv1, prof,  sel=None, par=None, par0=None, psig=None,  free=None, 
                    constr=None, line='Ha'):
    """
    To do the three-layer spectrla inversion of a line profile

    Parameters
    ----------
    wv1 : array_like
        wavelengths (in Angstrom) measured from line center.
    prof : array_like
        profile of intensities (normalized by the disk center continuum intensity).
    sel : boolean, array_like, optional
        True for the indexes of data to be used for fitting. The default is None.
    par : array_like, optional
         initial estimates of the model parameters. The default is None.
    par0 : array_like, optional
        default model parameters. The default is None.
    psig : array_like, optional
        prior deviations of model parameters. The default is None.
    free : integer, array_like, optional
        indexes of free parameters to be determined. The default is None. 
        If None, the indexes are determined by the function ThreeLayerParControl().
    constr : integer, array_like, optional
        indexes of constrained free parameters to be determined. The default is None. 
        If None, the indexes are determined by the function ThreeLayerParControl().
       
    line : str, optional
        line designation The default is 'Ha'.
    Fitting : boolean, optional
        True if the fitting is required. The default is True.

    Returns
    -------
    par : array_like
        model parameters.
    I0 : array_like
        intensity profile at the top of the chromosphere    
    I1 : array_like
        intensity profile in the middle of the chromosphere
    I2 : array_like
        intensity profile at the top of the photosphere
        
    epsD : float
        error of data fitting.
    epsP : float
        deviation of parameters from prior conditions.

    """

    if prof.min() >= 0.9*prof.max() or prof.max() < 0.05 :
       par0, psig = ThreeLayerDefault( line=line)
       par = par0
       I2 = prof
       I0 = prof
       I1 = prof
       epsD=-1.
       epsP =-1.
       return   (par,  I0, I2, I1, epsD, epsP) 
    if par0 is None:
        par0, psig = ThreeLayerInitial(wv1, prof, line=line) 
    
    
    if free is None: 
        free, constr0 = ThreeLayerParControl(line=line)
    if constr is None:
        free0, constr = ThreeLayerParControl(line=line)
      
    if sel is None: sel = fsel(wv1, line)
  
    if par is None:   par = np.copy(par0)
    parf=par[free]
    
    res_lsq= least_squares(ThreeLayerRes, parf,  
            args=(wv1[sel], prof[sel],    par0, psig,  par, line, free, constr), 
              jac='2-point',  max_nfev=100, ftol=1e-3) 
    
    
    parf = np.copy(res_lsq.x)
    par[free] = parf
    residual= res_lsq.fun

    Ndata = len(wv1[sel]) 
    epsD = sqrt((residual[0:Ndata]**2).sum())
    epsP = sqrt((residual[Ndata:]**2).sum())
         
    I0, I1, I2 = fThreeLayer(wv1,par, line=line)
    

    return  (par,  I0, I1, I2,  epsD, epsP)  

def ThreeLayerRadLoss(p, line='Ha'):
    """
    To calculate the radiative losses of the upper chromosphere and
    the lower chromosphere, respectively, from the three-layer model parameters

    Parameters
    ----------
    p : array_like
        parameters of three layer models.
    line : str, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    Radloss01 : float
        radiative loss from the upper chromosphere in unit of kW/m^2
        
    Radloss12 : float
        rdaitive loss from the lower chromosphere in unit of kW/m^2   

    """

    
    xvalues5p = array([-0.9062, -0.5385,   0.,     0.5385, 0.9062])
    weights5p = array([ 0.2369,  0.4786,   0.5689 ,0.4786, 0.2369])    
    xvalues3p =  array([-0.774597,0, 0.774597])
    weights3p =   array([0.55556, 0.888889, 0.555556])
    xvalues2p =  array([-0.57735,0.57735])
    weights2p =   array([1., 1.])

    
#    if line =='Ha':
    lambdaMax = 6.
    tauM = 100.

    wvline = fwvline(line)
    wvp = p[0]/3.e5*wvline
    eta, wp, ap, Sp, S2 =10.**p[1:6]
    
    tau2, tau1 = 10**p[6:8]    
    wv01, wv00 =p[8:10]/3.e5*wvline
    w1, w0 = 10**p[10:12] 
    S1, S0 = 10**p[12:14]
    wg = p[14]
    wv02 = wvp
    w2 = wp
    def gf(wv1, x, level):
        if level == 1:
            #ulamb = (wv1-(wv00+(wv01-wv00)*x))/(w0 + (w1-w0)*x)
            value = fPhie(wv1, wv00+(wv01-wv00)*x, w0 + (w1-w0)*x, line=line)# exp(-ulamb**2)
        elif level == 2:
            
           # ulamb = (wv1-(wv01+(wv02-wv01)*x))/(w1 + (w2-w1)*x)
            wvcenter= (wv01+(wv02-wv01)*x)
            width=(w1 + (w2-w1)*x)
            a = (wg*x) /width
            value = fPhi(wv1, wvcenter, width, a, line=line) #Voigt(ulamb,a)/Voigt(0,a)
        elif level == 3:
            #V0 = Voigt(0., ap)
           # ulamb=(wv1-wvp)/wp
            
            value = (eta*fPhi(wv1, wvp, wp, ap, line=line)+1.) #Voigt(ulamb, ap)/V0 + 1)
        return value

    def dff(wv1, x1, x2, level):
         if level == 3:
             value = (x2-x1)*gf(wv1, x1, level)
         else:    
             xx = xvalues3p*(x2-x1)/2.+(x2+x1)/2.
             wdx = weights3p*(x2-x1)/2.
             value = 0.
             for i in range(len(xx)):
                 value = value +gf(wv1, xx[i], level)*wdx[i]
         return value
    a0 = S0
    a1 = (-1.5*S0+2*S1-0.5*S2)
    a2 = (0.5*S0-S1+0.5*S2)

    def Sf(x, level):
       if level == 1:
           x0 = 0.
           value = a0+a1*(x+x0)+ a2*(x+x0)**2
       elif level == 2:
           x0 = 1.
           value = a0+a1*(x+x0)+ a2*(x+x0)**2
       elif level == 3:
           value = S2 + (Sp-S2)*(tauM*x)
       return value
#        
            
    
    xs = xvalues5p*(1-0)/2.+(1+0.)/2.
    weightdx = weights5p*(1-0)/2. 
    Nx = len(xs)
 
#   Applyng Simpson's 3/8 rule making use of Cubic spline
    
    Nlambda = int(2*lambdaMax/0.2)
    Nlambda = Nlambda+ (3-((Nlambda-1)%3))
    index = np.arange(Nlambda)
    
    w=np.ones(Nlambda, float)*2
    s= index%3 == 0
    w[s]=3.
    w[0]=1.
    w[-1]=1.
    lambdas = (index/(Nlambda-1)-0.5)*lambdaMax
    weightdlambda = (3/8.)*(lambdas[-1]-lambdas[0])/(Nlambda-1)*w
        
    Nlambda = len(lambdas)
    
    
    Nl = 2
    levels = array([1,2,3])
    
    gs = np.zeros((Nlambda, Nx, Nl))
    fs = np.zeros((Nlambda, Nx, Nl))
    
    
    for l in range(Nl):
        for j in range(Nx):
            gs[:,j,l] = gf(lambdas, xs[j], levels[l])
            if j == 0:
                fs[:,j,l] = dff(lambdas, 0., xs[j], levels[l])
            else:
                fs[:,j,l] = fs[:, j-1,l]+dff(lambdas, xs[j-1], xs[j], levels[l])
    taulambda1 =   tau1*( fs[:, -1, 0]+dff(lambdas, xs[-1], 1., 1))
    taulambda2 =   tau2*( fs[:, -1, 1]+dff(lambdas, xs[-1], 1., 2))

    K12 = np.zeros((Nx, Nl))
    K01 = np.zeros((Nx, Nl))
    S = np.zeros((Nx, Nl))
 
    
    for j in range(Nx):            
            tmp = -expn(2, taulambda1-tau1*fs[:,j,0]) + expn(2, taulambda1+taulambda2-tau1*fs[:,j,0])
            tmp = tau1*tmp*gs[:,j,0]
            K12[j,0] = (weightdlambda*tmp).sum()
            tmp = expn(2,tau1*fs[:,j,1])+ expn(2, taulambda2-tau2*fs[:,j,1])
            tmp = tau2*tmp*gs[:,j,1]
            K12[j,1] = (weightdlambda*tmp).sum()
            tmp = expn(2, tau1*fs[:,j,0])+ expn(2, taulambda1-tau1*fs[:,j,0])
            tmp = tau1*tmp*gs[:,j,0]
            K01[j,0] = (weightdlambda*tmp).sum()
            tmp = expn(2,taulambda1+ tau2*fs[:,j,1])- expn(2, tau2*fs[:,j,1])
            tmp = tau2*tmp*gs[:,j,1]
            K01[j,1] = (weightdlambda*tmp).sum()
            if Nl > 2:
                tmp = -expn(2,tauM*fs[:,j,2])+expn(2,  taulambda2+tauM*fs[:,j,2])
                tmp = tauM*tmp*gs[:,j,2]
                K12[j,2] =(weightdlambda*tmp).sum()        
                tmp = -expn(2,taulambda2+tauM*fs[:,j,2])+expn(2, taulambda1+ taulambda2+tauM*fs[:,j,2])
                tmp =tauM* tmp*gs[:,j,2]
                K01[j,2] = (weightdlambda*tmp).sum()
        
            for l in range(Nl):
                S[j,l] = Sf(xs[j], levels[l])       
    
    Radloss01, Radloss12 = 0., 0
    for l in range(Nl):
        Radloss01 =  Radloss01+(2*np.pi)* (weightdx*S[:,l]*K01[:,l]).sum()
        Radloss12 =  Radloss12+(2*np.pi)* (weightdx*S[:,l]*K12[:,l]).sum()
    if line == 'Ha':
       Radloss01 = Radloss01*(2.84e6/1.0e6)  # in kW/m^2
       Radloss12 = Radloss12*(2.84e6/1.0e6)
    elif line == 'Ca':
       Radloss01 = Radloss01*(1.76e6/1.0e6)  
       Radloss12 = Radloss12*(1.76e6/1.0e6)  # in kW/m^2
      
    return Radloss01, Radloss12  
#%%
def fCloud(wv1, p, inprofile, line='Ha'):
    """
    To calculate the intensity profile of a cloud model 

    Parameters
    ----------
    wv1 : array_like
        wavelengths from line center (in A).
    p : array_like
        model parameters. [log S0, log tau0, log w, v]
    inprofile : array_like
        profile of incident light intensity.
    line : str, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    model : array_like
        model profile.

    """
    
    S0 = 10**p[0]
    tau0 = 10**p[1]
    w = 10**p[2]
    wv0 = p[3]/3.e5*fwvline(line)    
    tau = tau0*fPhie(wv1, wv0, w, line =line) #exp(-((wv1-wv0)/w)**2)    
    model = inprofile*exp(-tau)+S0*(1-exp(-tau))
    return model
def CloudRes(p,wv1, inprofile, profile, sigma, p0, psig, line):
    model = fCloud(wv1,p, inprofile, line=line)
    resD = (log10(profile)-log10(model))/(sigma*0.43)
    resC = (p-p0)/psig
    Res = np.append(resD/sqrt(len(resD)), resC/sqrt(len(resC)))
    return Res
def CloudModel(wv1, inprofile, profile, line='Ha'):
    """
    To do the cloud model inversion of a profile

    Parameters
    ----------
    wv1 : array_like
        wavelengths from line center (in A).
    inprofile : array_like
        profile of incident light profile.
    profile : array_like
        observed profile of light intensity.
    line : str, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    par : array_like
        model parameters.
    model : array_like
        model profile of intensity.

    """
    weight = abs(1.-inprofile/profile)
    weight =weight/weight.sum()
    wv0 = (wv1*weight).sum()
    v = wv0/fwvline(line)*3.e5
    sigma = fsigma(wv1,profile, line=line)
    if line == 'Ha':
         p = array([-0.5, -0.3,-0.45, v])
    elif line == 'Ca':
         p = array([-0.5, -0.3,-0.70, v])
    p0 =  np.copy(p)     
    psig = array([0.5, 0.1,  0.05, 2.])
    
    res_lsq= least_squares(CloudRes, p,     args=(wv1, inprofile,  profile, sigma, p0, psig,   line), 
             jac='2-point',  max_nfev=50, ftol=1.e-3) #, bounds=(low[free],up[free]))
   
   
    par = np.copy(res_lsq.x)
    residual= res_lsq.fun
    model = fCloud(wv1, par, inprofile, line=line)
    return par,  model

def ThreeLayerCloudModel(wv1, prof,  wvb=None,  par=None, line='Ha'):
    """
    To do the hybrid inversion (three layer model + cloud model)

    Parameters
    ----------
    wv1 : array_like
        wavelengths from line center in A.
    prof : array_like
        intensity profile.
    wvb : array_like, optional
        range of wavelengths from line center (in A) 
        to be used for cloud model. The default is None.
    par : array_like, optional
        initail guess of three layer model parameters. The default is None.
    line : str, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    parout : array_like
        parameters (three layer parameters and cloud model parameters).
    model : array_like
        three layer model.
    modelc : array_like
        cloud model.
    res : array_like
        residuals.
    sel : boolean, array_like
        True if used for three layer model fitting.
    selc : boolean, array_like
        True if used for cloudl model fitting.
    epsD : float
        data fitting error of three layer model.
    epsP : float
        deviation of parameters from prior values in three layer model.

    """

    pure=fpure(wv1+fwvline(line), line=line)
    sigma = fsigma(wv1,prof, line=line)
    if wvb is None: wvb = array([0., 0.])
    blocked =  (wv1-wvb[0])*(wv1-wvb[1]) < 0
    sel  = pure*(~blocked)*(abs(wv1)<4.)
    selc = pure* blocked*(abs(wv1)<4.)
    
    nselc = len(wv1[selc])
    parfit,  model1, modelch1,  modelph1, epsD, epsP = \
           ThreeLayerModel(wv1, prof,  sel=sel, par=par, line=line) 
    
    model, modelch, modelph = fThreeLayer(wv1,parfit, line=line)       

    if nselc > 1:
        pc, modelc1 = CloudModel(wv1[selc], model[selc], prof[selc], line=line)
    else:
        pc = array([-10., -10., 10., 0.])
       
    modelc = fCloud(wv1, pc, model, line=line)
    parout = np.append(parfit, pc)       
    res = (prof-modelc)/sigma 
    return parout, model, modelc, res, sel, selc, epsD, epsP   
    
#%%

class GuiThreerLayerModel: 
    """
      Graphical user interface class for Implementing the Three-Layer Inversion of a profile  
    """
    def __init__(self, wv1, prof, profref, par=None, eps=None,  radloss = None,
                 line='Ha', hybrid= True, title=None):
        """
        To create a class of IntThreeLayerModel type with initialization

        Parameters
        ----------
        wv1 : array_like
            wavelengths (in A) from line center.
        prof : array_like
            profile of intensity.
        profref : array_like
            refernce profile of intensity.
        par : array_like, optional
            model parameters. The default is None.
        radloss : array_lke
            radiative losses. The default is None.
        eps : array_like
            epsD and epsP. The default is None.
        line : str, optional
            line designation. The default is 'Ha'.
        hybrid : : str, optional
             True if the  hybrid model fitting is allowed. The default is True.
        title : str, optional
            title of the plot. The default is None.
            

        Returns
        -------
        None.

        """
        # Input data 
        self.wv1 = wv1
        self.prof = prof
        self.profref = profref
        self.line = line
        self.hybrid = hybrid
        # Auxialliary data and initialization
        self.par0, self.psig =ThreeLayerInitial(wv1, prof, line=self.line)
        self.free, self.constr=ThreeLayerParControl(line=self.line)
        self.pure = fpure(self.wv1+fwvline(self.line), line=self.line)
        self.pc = array([0,0,0,0.])
        self.wvb = array([0., 0.])
        self.sel = self.selwv()
        self.selc = (self.wv1-self.wvb[0])*(self.wv1-self.wvb[1]) <= 0.
        self.sigma = fsigma(self.wv1, prof, line=self.line)
        
        if par is None:  par = self.par0

        self.par = par
        if radloss is None:  radloss = np.zeros(2)
        self.radloss = radloss
        if eps is None: eps=np.zeros(2)
        self.eps = eps
        self.modelc = prof
      
        # Plot configuration  
        self.fig =  plt.figure(line+ ' profile fitting', figsize=[11, 8])
        self.axes = self.fig.subplots(2, 1, sharex='col')
        ax0=self.axes[0]
        ax1=self.axes[1]
        
      #  Int0, Int1, Int2 = fThreeLayer(self.wv1, self.par, line=self.line)  
        self.iplot0=ax0.plot(self.wv1, self.prof, 'k', linewidth=3, label=r'$I_{obs}$')[0]
        self.iplot02=ax0.plot(self.wv1, prof, 'g', linewidth=1, label=r'$I_2$')[0]
        self.iplot01=ax0.plot(self.wv1, prof, 'b', linewidth=1, label=r'$I_{1}$')[0]
        self.iplot00=ax0.plot(self.wv1, prof, 'r',  linewidth=1, label=r'$I_{0}$')[0]
        if ~(self.hybrid):
            self.iplotref=ax0.plot(self.wv1, self.profref, 'c--', linewidth=1, label=r'$I_{ref}$')[0]  
        if self.hybrid:    
            self.iplot0c=ax0.plot(self.wv1, self.modelc, 'm', linewidth=1, label=r'$I_m$')[0]

        ax0.set_ylabel('Intensity')
        ax0.set_yscale('log')
        ax0.set_ylim([0.05, 2.])
        ax0.set_xlim([-4, 4])
        self.title=title
        ax0.set_title(self.title, fontsize=11)
        
        
        self.txtsrc = self.axes[0].text(0.05, 0.15,     
                 rf'$\log\, S_p$={par[4]:.2f}, $\log\, S_2$={par[5]:.2f}, ' + 
                 rf'$\log\, S_1$={par[12]:.2f}, $\log\, S_0$={par[13]:.2f}',
                     transform=self.axes[0].transAxes, size=11  )
        self.txtwv = self.axes[0].text(0.05, 0.05,     
                 rf'$v_1$={par[8]:.2f}, $v_0$={par[9]:.2f}, ' + 
                 rf'$\log\, w_1$={par[10]:.2f}, $\log\, w_0$={par[11]:.2f}',
                     transform=self.axes[0].transAxes, size=11    )
                
        self.txteps = self.axes[1].text(0.05, 0.05,
                  r'$\epsilon_D$ ='+f'{eps[0]:.2f}, '+r'$\epsilon_P$'+f'={eps[1]:.2f}',
                     transform=self.axes[1].transAxes, size=11        ) 
        if self.hybrid: 
            self.txtcl = self.axes[1].text(0.05, 0.9, '',
                         transform=self.axes[1].transAxes, size=11        ) 


#        Ndata = len(self.wv1[self.sel])
        ax1.plot([-700,700], [0,0], linewidth=1)
        self.iplot1=ax1.plot(self.wv1[self.sel], self.prof[self.sel]*0,'r.', ms=3, )[0]
        if self.hybrid:
            self.iplot1c=ax1.plot(self.wv1[self.selc], self.prof[self.selc]*0, 'm.', ms=3)[0]
        wvline = fwvline(line)
        ax1.set_xlabel(r'$\lambda$ - '+f'{wvline:.2f}'+r' [$\AA$]')
        ax1.set_ylabel(r'$(I_{obs}-I_0)/\sigma$')
        ax1.set_yscale('linear')
        ax1.set_ylim([-10,10])
        ax1.set_xlim([-4, 4])
        ax0.legend(loc='lower right')
        
        
        self.DockWidget()
        self.fig.tight_layout(pad=3)  
        self.Writepar()
        self.Redraw()
    def selwv(self):
        """
        To re-select the wavelengths to be used for the fitting 
        by excluding the wavelegnths bounded by two boundary wavelengths   

        Returns
        -------
        sel : boolean array
             array of selected or not.

        """
        sel=fsel(self.wv1, self.line)*((self.wv1-self.wvb[0])*(self.wv1-self.wvb[1]) >= 0 )
                    
        return sel            
    def selFont(self):
        """
        To set the font (class) used in the widgets

        Returns
        -------
        font : class
            font class.

        """
        # 
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(50)
        return font    
    def DockWidget(self): 
        """
        To configure the widgets

        Returns
        -------
        None.

        """
        
        font = self.selFont()        
        # figure window to variable
        self.root = self.fig.canvas.manager.window

        # create pannel and layout       
        dock = QtWidgets.QDockWidget("Parameters", self.root)
        self.root.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock) 
        panel = QtWidgets.QWidget()
        dock.setWidget(panel)
        vbox = QtWidgets.QVBoxLayout(panel) # 
        
        pVal={}
        pVal0={}
        pLabel={}
        pSig={}
        parnames = ThreeLayerPar(np.arange(15))
        
        self.npar= len(self.par)
        free, constr=ThreeLayerParControl(line=self.line)

        parnames1=np.append(parnames, ['epsD', 'epsP', 'wvb1', 'wvb2'])
        par1=np.append(self.par, [0., 0., 0., 0.])
        labelcolor= ['lightgray','lightgray','lightgray','lightgray','lightgray',
                    'lightgray','lightgray', 'lightgray', 'lightgray', 'lightgray', 
                    'lightgray', 'lightgray','lightgray', 'lightgray', 'lightgray',
                    'lightgray', 'lightgray',     'cyan',   'cyan',  'lightgray']
        
        for i in range(len(free)):  labelcolor[free[i]] = 'lightblue'
        for i in range(len(constr)): labelcolor[constr[i]] = 'lightgreen'
        npar1 = len(par1)
        for i in range(npar1):
            pLabel[i] = QtWidgets.QLabel(parnames1[i])
            pVal[i] = QtWidgets.QLineEdit()
            pVal0[i] = QtWidgets.QLineEdit()
            pSig[i] = QtWidgets.QLineEdit()
            pLabel[i].setFixedWidth(70)
            pVal[i].setFixedWidth(70)
            pVal0[i].setFixedWidth(70)
            pSig[i].setFixedWidth(70)
            if i < self.npar:
                pVal[i].setText(f"{self.par[i]:.3f}")
                pVal0[i].setText(f"{self.par0[i]:.3f}")
                pSig[i].setText(f"{self.psig[i]:.3f}")
            else:
                pVal[i].setText(f"{0.:.3f}")
                pVal0[i].setText(f"{0.:.3f}")
                pSig[i].setText(f"{0.:.3f}")
            pLabel[i].setFont(font)
            pLabel[i].setStyleSheet("background-color:"+ labelcolor[i])           
            pVal[i].setFont(font)
            pVal[i].setStyleSheet("background-color:"+ labelcolor[i])
            pVal0[i].setFont(font)
            pSig[i].setFont(font)
            pVal0[i].setStyleSheet("background-color:"+ labelcolor[i])
            pSig[i].setStyleSheet("background-color:"+ labelcolor[i])
        for i in range(npar1):
            
            if i == 0:
                hbox = QtWidgets.QHBoxLayout(panel)
                tmp=QtWidgets.QLabel('Name')
                tmp.setFixedWidth(70)
                tmp.setFixedHeight(15)
                tmp.setFont(font)
                hbox.addWidget(tmp)
                tmp=QtWidgets.QLabel('Par')
                tmp.setFixedWidth(70)
                tmp.setFixedHeight(15)
                tmp.setFont(font)
                hbox.addWidget(tmp)
                tmp=QtWidgets.QLabel('Par0')
                tmp.setFixedWidth(70)
                tmp.setFixedHeight(15)
                tmp.setFont(font)
                hbox.addWidget(tmp)
                tmp=QtWidgets.QLabel('Psig')
                tmp.setFixedWidth(70)
                tmp.setFixedHeight(15)
                hbox.addWidget(tmp)
                tmp.setFont(font)
                vbox.addLayout(hbox)
            hbox = QtWidgets.QHBoxLayout(panel)
            hbox.addWidget(pLabel[i])
            hbox.addWidget(pVal[i])
            hbox.addWidget(pVal0[i])
            hbox.addWidget(pSig[i])
            vbox.addLayout(hbox)

        self.pVal = pVal 
        self.pVal0 = pVal0
        self.pSig  = pSig
        # create Button
        Btn = [QtWidgets.QPushButton(text='Redraw'), 
               QtWidgets.QPushButton(text='Initialize'),
               QtWidgets.QPushButton(text='Fit'),
 #              QtWidgets.QPushButton(text='Hybrid'),
               QtWidgets.QPushButton(text='Export'),
                QtWidgets.QPushButton(text='Import'),]
        Btn[0].clicked.connect(self.Redraw)
        Btn[1].clicked.connect(self.Initialize)
        Btn[2].clicked.connect(self.Fit)
        Btn[3].clicked.connect(self.Export)
        Btn[4].clicked.connect(self.Import)
        self.Btn = Btn      
        for i in range(len(Btn)):
            if (i == 0) or (i==3) :
                hbox = QtWidgets.QHBoxLayout(panel)
                vbox.addLayout(hbox)
            Btn[i].setFont(font)
            hbox.addWidget(Btn[i])
    
    def Readpar(self):
        """
        To read parameters from the widgets

        Returns
        -------
        None.

        """
        for i in range(self.npar):  
            self.par[i]  = float(self.pVal[i].text()) 
            self.par0[i] = float(self.pVal0[i].text())
            self.psig[i] = float(self.pSig[i].text())
        self.wvb[0] = float(self.pVal[self.npar+2].text())
        self.wvb[1] = float(self.pVal[self.npar+3].text())
        self.sel=self.selwv() 
        self.selc = (self.wv1-self.wvb[0])*(self.wv1-self.wvb[1]) <= 0.
        free, constr=ThreeLayerParControl(line=self.line)
        if self.hybrid and (self.wvb[0] != self.wvb[1])  :
            self.constr = np.append(constr, 13)
        else:
            self.constr = constr 
    
     
    def Writepar(self): 
        """
        To write paramters into the widgets

        Returns
        -------
        None.

        """
        for i in range(self.npar):   
            self.pVal[i].setText(f"{self.par[i]:.3f}")
            self.pVal0[i].setText(f"{self.par0[i]:.3f}")  
            self.pSig[i].setText(f"{self.psig[i]:.3f}") 
        self.pVal[self.npar+2].setText(f"{self.wvb[0]:.2f}")
        self.pVal[self.npar+3].setText(f"{self.wvb[1]:.2f}")
    def Redraw(self):
        """
        To redraw plots reflecting the change in the data and parameters

        Returns
        -------
        None.

        """
        self.Readpar()       
        Int0, Int1, Int2 = fThreeLayer(self.wv1, self.par, line=self.line)  
        self.model = Int0
        self.iplot0.set_ydata(self.prof)                 
        self.iplot00.set_ydata(Int0)
        self.iplot00.set_xdata(self.wv1)
        self.iplot02.set_ydata(Int2)
        self.iplot01.set_ydata(Int1)
        if  (self.wvb[0] != self.wvb[1]) and self.hybrid:                
            resDs = (self.prof[self.selc]-self.modelc[self.selc])/self.sigma[self.selc]
            Ndatas = len(self.wv1[self.selc]) 
            self.iplot0c.set_ydata(self.modelc[self.selc])
            self.iplot0c.set_xdata(self.wv1[self.selc])
            self.iplot1c.set_ydata(resDs)
            self.iplot1c.set_xdata(self.wv1[self.selc])
            epsDs = sqrt((resDs**2).mean()) 
        
        Res = ThreeLayerRes(self.par[self.free], self.wv1[self.sel], 
                          self.prof[self.sel],  self.par0, self.psig,  self.par, 
                          self.line, self.free, self.constr)
        Ndata = len(self.wv1[self.sel])
                    
        if self.wvb[0] == self.wvb[1]:  
            if (self.hybrid):
                self.iplot0c.set_ydata(self.wv1[self.sel]*0)
                self.iplot0c.set_xdata(self.wv1[self.sel])
                self.iplot1c.set_ydata(self.wv1[self.sel]*0-100. )
                self.iplot1c.set_xdata(self.wv1[self.sel])

        self.iplot1.set_xdata(self.wv1[self.sel])
        self.iplot1.set_ydata(Res[0:Ndata]*sqrt(Ndata))
 
        self.pVal[self.npar].setText(f'{self.eps[0]:.2f}')
        self.pVal[self.npar+1].setText(f'{self.eps[1]:.2f}')
        
        self.txtsrc.set_text(     
                 rf'$\log\, S_p$={self.par[4]:.2f}, $\log\, S_2$={self.par[5]:.2f}, ' + 
                 rf'$\log\, S_1$={self.par[12]:.2f}, $\log\, S_0$={self.par[13]:.2f}')
        self.txtwv.set_text(     
                 rf'$v_1$={self.par[8]:.2f}, $v_0$={self.par[9]:.2f}, ' + 
                 rf'$\log\, w_1$={self.par[10]:.3f}, $\log\, w_0$={self.par[11]:.3f} ' )
        
        self.txteps.set_text(
                  r'$\epsilon_D$='+f'{self.eps[0]:.2f}, '
                  +r'$\epsilon_P$'+f'={self.eps[1]:.2f}, Radloss2=' + f'{self.radloss[0]:.1f},' \
                      +' Radloss1='+f'{self.radloss[1]:.1f}') 
        if self.hybrid:
            epsD = self.eps[0]
            if self.wvb[0] != self.wvb[1]:      
                 epsDt = sqrt((epsD**2*Ndata + epsDs**2*Ndatas)/(Ndata+Ndatas))
                 self.txtcl.set_text(     
                      rf'$\log\, S$={self.pc[0]:.2f}, $\log\, \tau$={self.pc[1]:.2f}, ' + 
                     rf'$\log\, w$={self.pc[2]:.3f}, $v$={self.pc[3]:.1f}, $\epsilon_D$='+
                     f'{epsDs:.2f}'+', $\epsilon_t$='+f'{epsDt:.2f}')
            else:       self.txtcl.set_text('')
        self.fig.canvas.draw_idle()
 
    def Initialize(self):
        """
        To initialize par, par0, psig, write parameters into the widgets, 
        draw the plots

        Returns
        -------
        None.

        """
        parint, psig = ThreeLayerInitial(self.wv1, self.prof, line=self.line)
        self.par = np.copy(parint)
        self.par0 = np.copy(parint)
        self.psig = np.copy(psig)
        self.Writepar()
        self.Redraw()    
    def Renew(self, wv1, prof, par=None, eps=None, radloss=None):
        """
        To renew the data

        Parameters
        ----------
        wv1 : array
            wavelengths from line center in A.
        prof : array
            intensities.
        par : array, optional
            model parameters. If None, the parameters are set to the initial
            guess. The default is None.
        eps : 2-element array, optional
            array of epsD and epsP. If None, they are initially set to zero.
            The default is None.
        radloss : 2-element array, optional
            array of rdaloss1 and radloss2. If None, they are set to zero.
            The default is None.

        Returns
        -------
        None.

        """
        self.prof = prof
        self.wv1 = wv1
        self.par0, self.psig =ThreeLayerInitial(wv1, prof, line=self.line)
        if (par is None): par = self.par0
        self.par = par
        if (eps is None): eps = np.zeros(2)
        self.eps = eps
        if (radloss is None): radloss = np.zeros(2) 
        self.radloss = radloss
        self.Writepar()
        self.Redraw()    
    def Fit(self):
        '''
        To do the MLSI fitting to the profile and show the result. 
        Three-Layer inversion is done to the selected wavelengths, and 
        the cloud model inversion, to the blocked wavelengths. 

        Returns
        -------
        None.

        '''
        self.Readpar()
        
        parfit,  model, modelch,  modelph, epsD, epsP = \
            ThreeLayerModel(self.wv1, self.prof,  sel = self.sel,  \
                            par=self.par, par0=self.par0, psig=self.psig, 
                            free=self.free, constr=self.constr, line=self.line)   
        Radloss=ThreeLayerRadLoss(parfit, line=self.line)
        self.radloss = np.flip(Radloss)
        self.eps = array([epsD, epsP])
 
        self.par=np.copy(parfit)
        self.Writepar()
#        print("Fitting of "+self.line+" line took ", tt2-tt1, ' s')
        if self.hybrid:
            self.selc = (self.wv1-self.wvb[0])*(self.wv1-self.wvb[1]) <= 0.
            pc, modelc = CloudModel(self.wv1[self.selc], model[self.selc], self.prof[self.selc], line=self.line)
            modelc = fCloud(self.wv1, pc, model, line=self.line)
            self.pc = pc
            self.modelc = modelc
        self.Redraw()
    def Export(self):  
        """
        To save the data of wavelengths, intensity profile, parameters, 
        and model profile into a pickle file. 

        Returns
        -------
        None.

        """
        data = (self.wv1, self.prof, self.par, self.model)
        file =self.line+"ITLM.pickle"
        with open(file, "wb") as f:
            pickle.dump(data, f)
    def Import(self):
        """
        To read the data of wavelengths, intensity profile, parameters, 
        and model profile from a pickle file, and to present the plots 

        Returns
        -------
        None.

        """
        file =self.line+"ITLM.pickle"
        with open(file, 'rb') as f:
            data = pickle.load(f)
        self.wv1 = data[0]
        self.prof0 = data[1]
        self.par = data[2]
        for i in range(self.npar):   
            self.pVal[i].setText(f"{self.par[i]:.3f}")
            self.pVal0[i].setText(f"{(self.par[i]-self.par0[i])/self.psig[i]:.3f}")

        self.Redraw()
 
#%%
class GuiFissRaster:
    """
    Graphical user interface for FISS raster data file(s) and 
    their assicated inversion parameter files.
    """
    def __init__(self, fHa=None, fCa=None, x=0., y=0., showpar = False, Haparfile=None, Caparfile=None, 
                 HaWv = array([-1,   -0.5,  0,  -0.5, 1., 4.]), 
                 CaWv = array([-0.5, -0.25, 0., 0.25, 1., 4.]) ):
        """
        Intialize FissRasterGui 

        Parameters
        ----------
        fHa : str, optional
            file name of Ha raster data. If it is None, ha data are not used.
            The default is None.
        fCa : str, optional
            file name of Ca II 8542 raster data.  If it is None, Ca data are not used.
            The default is None.
        x : float, optional
            x position in the field of view in Mm. It is 0 at the center. 
            The default is 0.
        y : float, optional
            y position in the field of view in Mm. It is 0 at the center. 
            The default is 0.
        showfit : str, optional
             'dofit'  if the realtime fitting is done to show fit.
             'parfile' if the parameter in file is to be used to show fit. 
             The default is 'dofit'.
        Haparfile : str, optional
            file of  Ha parameter data. If None, the file is assumed to be in the subdirectory 'pars'.
            The default is None.
        Caparfile : str, optional
            file of  Ca parameter data. If None, the file is assumed to be in the subdirectory 'pars'.
            The default is None.
        HaWv : array_like, optional
            wavelegnths for Ha raster image construction. 
            The default is array([-1,   -0.5,  0,  -0.5, 1., 4.]).
        CaWv : array_like, optional
            wavelegnths for Ca II raster image construction. 
            The default is array([-0.5, -0.25, 0., 0.25, 1., 4.]).

        Returns
        -------
        None.

        """
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass
        
        self.showHadata = False
        self.showHapar = False
        self.showCadata = False
        self.showCapar = False
        
        if fHa != None:
            self.fHa = fHa
            if os.path.isfile(self.fHa):
                self.showHadata = True
            if Haparfile == None:
                self.fHapar = ThOutfile(self.fHa)
            else :
                self.fHapar = Haparfile
            if showpar and os.path.isfile(self.fHapar):
                self.showHapar = True  
        if fCa != None:
            self.fCa = fCa
            if os.path.isfile(self.fCa):
                self.showCadata = True
            if Caparfile == None:
                self.fCapar = ThOutfile(self.fCa)
            else :
                self.fCapar = Caparfile
            if showpar and os.path.isfile(self.fCapar):
                self.showCapar = True  
     
                   
        self.SpectralData()
        self.xpix = self.Xpix2Mm(x, invert=True) 
        self.ypix = self.Ypix2Mm(y, invert=True) 
        self.x = x 
        self.y = y 
        
        if self.showHapar:
            subdir='pars'
            if Haparfile == None: 
                t=fHa.find('_c.fts')
                u=fHa.find('FISS_')
                Haparfile = fHa[0:u]+subdir+'\\'+fHa[u:t]+'_par.fts'  
            self.Aparfile = Haparfile    
        if self.showCapar:
            subdir='pars'
            if Caparfile == None: 
                t=fCa.find('_c.fts')
                u=fCa.find('FISS_')
                Caparfile = fCa[0:u]+subdir+'\\'+fCa[u:t]+'_par.fts'   
            self.Bparfile = Caparfile
        self.ParData()
        self.HaWv = HaWv
        self.CaWv = CaWv
        self.imInterp = 'nearest'
        self.scale = 'log'
        self.Raster()
        
        self.Spectrogram()
        if self.showHapar or self.showCapar:
            self.Parmap()
        self.Profile()       

    def SpectralData(self):
        """
        To load the FISS spectral data from the FISS files. In addtion,
            - to re-calibrate  wavelengths 
            - to re-normalize data
            - to construct the reference profile 
            - to spatially align between the Ha data and Ca II data
             

        Returns
        -------
        None.

        """
        if self.showHadata:
            fissHa = FISS(self.fHa)   # 6562.82
            self.wvHaline = fissHa.centralWavelength #6562.817
            fissHa.data = np.flip(fissHa.data,1) 
            icont=fissHa.data[:,:, 50]
            qs=icont > (icont.max()*0.7)
            fissHa.avProfile = fissHa.data[qs,:].mean(0)
            fissHa.wave = Recalibrate(fissHa.wave, fissHa.avProfile, line='Ha')                                
            fissHa.wave = fissHa.wave - fissHa.centralWavelength
            self.harefc = fissHa.avProfile[abs(abs(fissHa.wave)-4.5)<0.05].mean()*1.01
            fissHa.avProfile = fissHa.avProfile/self.harefc
            fissHa.data = fissHa.data/self.harefc
            self.fissHa = fissHa
            self.xDelt = self.fissHa.xDelt
            self.yDelt = self.fissHa.yDelt

        if self.showCadata:
            fissCa = FISS(self.fCa)   # 8542.09  
            self.wvCaline = fissCa.centralWavelength #8542.091                     
            fissCa.data = np.flip(fissCa.data,1)
            icont=fissCa.data[:,:, 50]       
            qs=icont > (icont.max()*0.7)
            fissCa.avProfile = fissCa.data[qs,:].mean(0)
            fissCa.wave = Recalibrate(fissCa.wave, fissCa.avProfile, line='Ca') 
            fissCa.wave = fissCa.wave - fissCa.centralWavelength                    
            self.carefc = fissCa.avProfile[abs(abs(fissCa.wave)-4.5)<0.05].mean()*1.09
            fissCa.avProfile = fissCa.avProfile/self.carefc
            fissCa.data = fissCa.data/self.carefc
            self.fissCa = fissCa       
            self.xDelt = self.fissCa.xDelt
            self.yDelt = self.fissCa.yDelt
        
        if self.showHadata and  self.showCadata:
            sh = alignoffset(fissCa.data[0:250,2:-2,50], fissHa.data[0:250,2:-2,-50])
            dy, dx = sh.round().astype('int')
            
            dy=dy[0]
            dx=dx[0]
            if dy > 0:
                yoffha, yoffca = dy, 0
            else:
                yoffha, yoffca = 0, -dy
            self.yoffha, self.yoffca = yoffha, yoffca   
            if dx > 0:
                xoffha, xoffca = dx, 0    
            else:
                xoffha, xoffca = 0, -dx
            self.xoffha, self.xoffca = xoffha, xoffca
            self.ny = np.maximum(fissHa.ny+self.yoffha, fissCa.ny+self.yoffca) 
            self.nx = np.maximum(fissHa.nx+self.xoffha, fissCa.nx+self.xoffca) 
            
        if self.showHadata and not(self.showCadata): 
            self.ny = fissHa.ny
            self.nx = fissHa.nx 
            self.yoffha=0
            self.xoffha=0
        
        if  not(self.showHadata) and  self.showCadata: 
            self.ny = fissCa.ny
            self.nx = fissCa.nx 
            self.yoffca=0
            
            self.xoffca=0
            
    def ParData(self):
        """
        To load the FISS parameter data from the parameter files.

        Returns
        -------
        None.

        """
        if self.showHapar:
            hA =fits.getheader(self.Aparfile) 
            nparA=hA['NAXIS3']
            parA=fits.getdata(self.Aparfile)
            self.parHa = np.empty((nparA,self.fissHa.ny,self.fissHa.nx), float)
    
            x1, x2, y1, y2 = hA['XSTART'], hA['NAXIS1']+hA['XSTART'],  hA['YSTART'], hA['NAXIS2']+hA['YSTART']       
            for p in range(nparA):
                self.parHa[p,y1:y2, x1:x2]=  parA[p,:, :]*hA[rf'SCALE{p:02d}']  
   
       
        if self.showCapar:
            hB =fits.getheader(self.Bparfile) 
            parB = fits.getdata(self.Bparfile)
            nparB=hB['NAXIS3']
            self.parCa = np.empty((nparB,self.fissCa.ny,self.fissCa.nx), float)
            x1, x2, y1, y2 = hB['XSTART'], hB['NAXIS1']+hB['XSTART'], \
                             hB['YSTART'], hB['NAXIS2']+hB['YSTART']
            for p in range(nparB):
                self.parCa[p,y1:y2, x1:x2]=parB[p,:, :]*hB[rf'SCALE{p:02d}']
        
        
    def Xpix2Mm(self, x, invert = False):
        """
        Convert x position in pixels into position in Mm

        Parameters
        ----------
        x : integer (or float) 
            x position in pixels (or in Mm)
        invert : boolean, optional
            True if x position in Mm is to be converted to x position in pixels. 
            The default is False.

        Returns
        -------
        xnew : float (or integer)
            x position in Mm (or in pixels.)

        """
        if not invert:
            xnew = (x+0.5-self.nx/2.)*self.xDelt*0.725
        else:
            xnew = int(round(x/(self.xDelt*0.725)+self.nx/2-0.5))
        return xnew
    
    def Ypix2Mm(self, y, invert = False):
        """
        Convert y position in pixels into position in Mm

        Parameters
        ----------
        y : integer (or float) 
            y position in pixels (or in Mm)
        invert : boolean, optional
            True if y position in Mm is to be converted to y position in pixels. 
            The default is False.

        Returns
        -------
        ynew : float (or integer)
            y position in Mm (or in pixels.)

        """

        if not invert:
            ynew = (y+0.5-self.ny/2.)*self.yDelt*0.725
        else:
            ynew = int(round(y/(self.yDelt*0.725)+self.ny/2-0.5))
        return ynew

    def Raster(self):
        """
        To first show the raster images from the FISS spectral data

        Returns
        -------
        None.

        """
        
        HaWv = self.HaWv
        CaWv = self.CaWv  # 0.2, 0.7, 4])
        nWv = HaWv.size

        self.rasterextent =np.append(self.Xpix2Mm(array([-0.5, self.nx-0.5]) ), 
                          self.Ypix2Mm(array([-0.5, self.ny-0.5 ])) )
       # Raster
              
        l= -1+ self.showHadata + self.showCadata
        
        if l >= 0: 
            self.figRaster, self.axRaster = plt.subplots(1+l, nWv, num='Raster',
                                figsize=[self.nx/50.*nWv*0.6,3*(1+l)], sharex=True,
                                sharey=True,  clear=True)
            self.figRaster.canvas.mpl_connect('key_press_event', self._onKey)
            
            self.axRaster = self.axRaster.flatten()
            if self.showHadata: 
                for n, hwv in enumerate(HaWv):
                    raster = self.fissHa.getRaster(hwv, hw=0.05)
                    rastermean = np.median(raster)
                    wh = raster >  0.5*rastermean
                    if self.scale == 'log':
                        raster = log10(raster)
                        rastermean = log10(rastermean)
                    rasterbig = np.ones((self.ny,self.nx), float)*rastermean
                    rasterbig[self.yoffha:self.yoffha+self.fissHa.ny, 
                              self.xoffha:self.xoffha+self.fissHa.nx] = raster
                    im = self.axRaster[n].imshow(rasterbig, self.fissHa.cmap,origin='lower',
                                                 extent=self.rasterextent, interpolation=self.imInterp)
                    m=raster[wh].mean()
                    st =raster[wh].std()
                    im.set_clim(m-3*st, m+3.*st)
                    if n == 5: im.set_clim(-0.2, 0.05)
                    if n == 4 or n == 0: im.set_clim(m-4*st, m+2*st)
        
                    sg='+'
                    if hwv < 0.:  sg=r'$-$'
                    self.axRaster[n].set_title(r' H$\alpha$'+sg+f'{abs(hwv):.2f} '+r'$\AA$',fontsize=12)         
            if self.showCadata: 
                for n, cwv in enumerate(CaWv):
                    raster = self.fissCa.getRaster(cwv, hw=0.03)
                    rastermean = np.median(raster)
                    wh = raster >  0.5*rastermean
                    if self.scale == 'log':
                        raster = log10(raster)
                        rastermean = log10(rastermean)
                    rasterbig = np.ones((self.ny,self.nx), float)*rastermean
                    rasterbig[self.yoffca:self.yoffca+self.fissCa.ny, 
                              self.xoffca:self.xoffca+self.fissCa.nx] = raster
                    im = self.axRaster[n+nWv*l].imshow(rasterbig, self.fissCa.cmap,
                                     origin='lower', extent=self.rasterextent,  interpolation=self.imInterp)
                    
                    m=raster[wh].mean()
                    st =raster[wh].std()
                    im.set_clim(m-3*st, m+3*st)
                    if n == 5: im.set_clim(-0.2, 0.0)
                    if n == 4 or n== 0 : im.set_clim(m-4*st, m+2*st)
        
                    sg='+'
                    if cwv < 0.:  sg=r'$-$'
                    self.axRaster[n+nWv*l].set_title('Ca II'+sg+f'{abs(cwv):.2f} '+r'$\AA$',fontsize=12)
                    if n == 0:
                        self.axRaster[n+nWv*l].set_xlabel('x (Mm)')
                        self.axRaster[n+nWv*l].set_ylabel('y (Mm)')
                         
            self.position = []
            for ax in self.axRaster:
                self.position += [ax.scatter(self.x, self.y, 50, marker='+', color='c', linewidth=1)]
            self.figRaster.tight_layout(pad=1)
        
    def Spectrogram(self):
        """
        To first show the spectrograms at the selected slit positions 

        Returns
        -------
        None.

        """
        #spectrogram
        if self.showHadata:
            extentSpectroHa = np.append(array([self.fissHa.wave[0], self.fissHa.wave[-1]]) , 
                                        self.Ypix2Mm(array([-0.5, self.ny-0.5 ])) )
        if self.showCadata:    
            extentSpectroCa = np.append(array([self.fissCa.wave[-1], self.fissCa.wave[0]]) , 
                                        self.Ypix2Mm(array([-0.5, self.ny-0.5 ])) ) 
        if self.showHadata:
            lha  = 1
        else: 
            lha = 0
        if self.showCadata: 
            lca = 1
        else:
            lca =0
        if (lha+lca) >0:    
            self.figSpec = plt.figure('Spectrogram', figsize=[5, 3*(lha+lca)], clear=True)
            self.figSpec.canvas.mpl_connect('key_press_event', self._onKey)
            
        if self.showHadata:
            self.axSpecHa = self.figSpec.add_subplot(lha+lca, 1, 1)
            self.axSpecHa.set_xlabel(r'$\lambda$ - '+f'{self.wvHaline:.2f}'+r' [$\AA$]')
            self.axSpecHa.set_ylabel('y (Mm)')
            self.axSpecHa.set_title(r"H$\alpha$") 
            xpixha = np.minimum(np.maximum(self.xpix - self.xoffha,0), self.fissHa.nx-1)
            specHa = self.fissHa.data[:, xpixha]
            whHa = specHa > 5./self.harefc
            if self.scale == 'log':   
                specHa = log10(specHa)
            specHa1 = np.zeros((self.ny,specHa.shape[1]))+np.median(specHa) 
            specHa1[self.yoffha:self.yoffha+self.fissHa.ny,:]=specHa
            self.imSpecHa = self.axSpecHa.imshow(specHa1, self.fissHa.cmap,origin='lower',
                                                 extent= extentSpectroHa,
                                                 interpolation=self.imInterp)
            self.imSpecHa.set_clim(specHa[whHa].min(), specHa.max())
            self.hlineSpecHa = self.axSpecHa.axhline(self.y,linestyle='dashed',
                                                     color='r', linewidth=0.5)
            self.axSpecHa.set_aspect(adjustable='box', aspect='auto')
        
        if self.showCadata:
            self.axSpecCa = self.figSpec.add_subplot(lha+lca, 1, lha+1)         
            self.axSpecCa.set_xlabel(r'$\lambda$ - '+f'{self.wvCaline:.2f}'+r' [$\AA$]')
            self.axSpecCa.set_ylabel('y (Mm)')
            self.axSpecCa.set_title("Ca II 854.2 nm")
            xpixca = np.minimum(np.maximum(self.xpix - self.xoffca,0), self.fissCa.nx-1)
            specCa = self.fissCa.data[:, xpixca, ::-1]
            whCa = specCa > 5./self.carefc
            if self.scale == 'log':   
                specCa = log10(specCa)
                        
            specCa1 = np.zeros((self.ny,specCa.shape[1]))+np.median(specCa) 
            specCa1[self.yoffca:self.yoffca+self.fissCa.ny,:]=specCa
       
            self.imSpecCa = self.axSpecCa.imshow(specCa1, self.fissCa.cmap, origin='lower', 
                                                 extent= extentSpectroCa,
                                                 interpolation=self.imInterp)
            
            self.imSpecCa.set_clim(specCa[whCa].min(), specCa.max())        
            self.axSpecCa.set_aspect(adjustable='box', aspect='auto')
            self.hlineSpecCa = self.axSpecCa.axhline(self.y,linestyle='dashed',
                                                     color='r', linewidth=0.5)

        self.figSpec.tight_layout()
        
    def Profile(self):
        """
        To first show the spectral profiles of the two lines at the selected position

        Returns
        -------
        None.

        """
#  Profiles
        if (self.xpix < 0) or (self.ypix<0):
            if self.showHadata:  profA0=self.fissHa.avProfile 
            if self.showCadata:  profB0=self.fissCa.avProfile  
        else:      
            if self.showHadata: profA0=self.fissHa.data[self.ypix, self.xpix]
            if self.showCadata: profB0=self.fissCa.data[self.ypix, self.xpix]
        
        if self.showHadata:
            wvA1=self.fissHa.wave
            wvA=wvA1+self.wvHaline
            pureA=fpure(wvA, line='Ha')  
            self.TparHa = FissGetTlines(wvA1, self.fissHa.avProfile, line='Ha')
            self.fissHa.refProfile = FissCorTlines(wvA1, self.fissHa.avProfile, self.TparHa,line='Ha') 
     
            self.fissHa.refProfile = FissStrayAsym(wvA1, self.fissHa.refProfile, 
                                                   self.fissHa.avProfile, pure=pureA)    
        if self.showCadata:
            wvB1=self.fissCa.wave
            wvB=wvB1+self.wvCaline
    
            pureB=fpure(wvB, line='Ca')
     
                         
            self.TparCa = FissGetTlines(wvB1, self.fissCa.avProfile, line='Ca') 
            self.fissCa.refProfile =FissCorTlines(wvB1, self.fissCa.avProfile, self.TparCa,line='Ca') 
                  
            self.fissCa.refProfile = FissStrayAsym(wvB1,self.fissCa.refProfile, 
                                                   self.fissCa.avProfile, pure=pureB)    
        self.Reaction()

    def Parmap(self):        
        """
        To display the maps of the H alpha parameters in a figure and
        those of the Ca II line parameters in anonther figure

        Returns
        -------
        None.

        """
#      Interactive Raster
       
        nd = 12            
        ysize= 8.
        xsize = (self.nx+30.)/self.ny*ysize/3.*4.
        
        if self.showHapar:
            pars = self.parHa
            cmap=self.fissHa.cmap
            self.figIRaster, self.axIRaster = plt.subplots(3, nd//3, num='HaParMaps', 
                                        sharex=True, sharey=True, 
                                        figsize=[xsize,ysize],   clear=True)
            self.axIRaster = self.axIRaster.flatten()
            self.figIRaster.canvas.mpl_connect('key_press_event', self._onKey)
            good = (pars[15]< 2.)*(pars[16]<2.)*(pars[15]>0)     

            for i in range(nd):               
                j = ([4, 5, 12, 13, 16, 0,  8,  9,  15, 17,  10, 11, ])[i]
                if j >=0: 
                    name = ThreeLayerPar(j)
                    data = pars[j]
                elif j == -1:
                    name = 'log SE'
                    data = log10(0.2* 10**pars[12]  + 0.8*10**pars[13])
                elif j == -8:
                    name = '(v0+v1)/2'
                    data = (pars[9]+pars[8])/2.
                elif j == -9:
                    name = '(v0-v1)/2'
                    data = (pars[9]-pars[8])/2
                elif j == -10:
                    name = '(w0+v1)/2'
                    data = (10**pars[11]+10**pars[10])/2.
                elif j == -11:
                    name = '(w0-w1)/2'
                    data = (10**pars[11]-10**pars[10])/2    

                if j in (0,8, 9, -8, -9) : 
                    cmap='seismic'
                if j in (10,11, -10, -11, 17, 18): cmap='afmhot'
                if j in (7,7): cmap='afmhot_r'
                if j in (15, 16):
                    cmap='jet'
                rasterbig = np.ones((self.ny,self.nx), float)*data.mean()
                rasterbig[self.yoffha:self.yoffha+self.fissHa.ny, 
                          self.xoffha:self.xoffha+self.fissHa.nx] = data
 
                tmp= self.axIRaster[i].imshow(rasterbig, cmap=cmap,
                                 origin='lower',  extent=self.rasterextent,  
                                 interpolation=self.imInterp)
                self.axIRaster[i].set_title(name, size=10)
                cbar= self.figIRaster.colorbar(tmp, ax=self.axIRaster[i], shrink=0.6)
                cbar.ax.tick_params(labelsize=10) 
                if (j < 15) or (j == 17) or (j==18):
                    m=data[good].mean()
                    sg=data[good].std()
                    tmp.set_clim(m-3*sg, m+3*sg) 
            
                else:                      
                    tmp.set_clim( 0., 1.5)
                if  j == 4:  tmp.set_clim(-0.2, 0.05) # m-3*sg, m+3*sg)
#                if j == 17: tmp.set_clim(8., 13.)
#                if j == 18: tmp.set_clim(0, 2.5)
                    
                if j in (): tmp.set_clim(-5, 5)
            self.Iposition = []
            for ax in self.axIRaster:
                self.Iposition += [ax.scatter(self.x, self.y, 50, 
                                              marker= '+', color='c', linewidth=1)]
            self.figIRaster.tight_layout(pad=0.5)


        if self.showCapar:
            pars = self.parCa
            cmap=self.fissCa.cmap
            self.figIRasterCa, self.axIRasterCa = plt.subplots(3, nd//3, num='CaParMaps', 
                                        sharex=True, sharey=True,    
                                        figsize=[xsize,ysize],   clear=True)
            self.axIRasterCa = self.axIRasterCa.flatten()            
            self.figIRasterCa.canvas.mpl_connect('key_press_event', self._onKey)
                          
            good = (pars[15]< 2.)*(pars[16]<2.)*(pars[15]>0)     

            for i in range(nd):               
                j = ([4, 5, 12, 13, 16, 0,  8,  9,  15, 17,  10, 11, ])[i]
                if j >=0: 
                    name = ThreeLayerPar(j)
                    data = pars[j]
                elif j == -1:
                    name = 'log SE'
                    data = log10(0.2* 10**pars[12]  + 0.8*10**pars[13])
                elif j == -8:
                    name = '(v0+v1)/2'
                    data = (pars[9]+pars[8])/2.
                elif j == -9:
                    name = '(v0-v1)/2'
                    data = (pars[9]-pars[8])/2
                elif j == -10:
                    name = '(w0+v1)/2'
                    data = (10**pars[11]+10**pars[10])/2.
                elif j == -11:
                    name = '(w0-w1)/2'
                    data = (10**pars[11]-10**pars[10])/2    

                if j in (0,8, 9, -8, -9) : 
                    cmap='seismic'
                if j in (10,11, -10, -11, 17, 18): cmap='afmhot'
                if j in (7,7): cmap='afmhot_r'
                if j in (15, 16):
                    cmap='jet'
                rasterbig = np.ones((self.ny,self.nx), float)*data.mean()
                rasterbig[self.yoffca:self.yoffca+self.fissCa.ny, 
                          self.xoffca:self.xoffca+self.fissCa.nx] = data
               
                tmp= self.axIRasterCa[i].imshow(rasterbig, cmap=cmap,
                                 origin='lower',  extent=self.rasterextent,  
                                 interpolation=self.imInterp)
                self.axIRasterCa[i].set_title(name, size=10)
               
                cbar=self.figIRasterCa.colorbar(tmp, ax=self.axIRasterCa[i], shrink=0.6)
                cbar.ax.tick_params(labelsize=10) 

                if (j < 15) or (j == 17) or (j==18):
                    m=data[good].mean()
                    sg=data[good].std()
                    tmp.set_clim(m-3*sg, m+3*sg) 
            
                else:                      
                    tmp.set_clim( 0., 1.5)
                if  j == 4:  tmp.set_clim(-0.2, 0.05) # m-3*sg, m+3*sg)
                    
                if j in (): tmp.set_clim(-5, 5)
            self.IpositionCa = []
            for ax in self.axIRasterCa:
                self.IpositionCa += [ax.scatter(self.x, self.y, 50,  
                                                marker='+', color='c', linewidth=1)]
            self.figIRasterCa.tight_layout(pad=0.5)    
        
    def Reaction(self, xchange=False, ychange= False):
        """
        To do actions when the mouse position change is detected
            - to redraw the spectrograms
            - to refresh the mouse positions on the spectrograms, rasters, 
                and parmeter maps
            - to redraw the spectral profiles after processing
            - to fit the profiles when  the parameter file data are not used.     

        Parameters
        ----------
        xchange : boolean, optional
            True if x position changes. The default is False.
        ychange : booleand, optional
            True if y poisition changes. The default is False.

        Returns
        -------
        None.

        """
        # Spectrogram

        if self.showHadata:
            xpixha = np.minimum(np.maximum(self.xpix - self.xoffha,0), self.fissHa.nx-1)
            ypixha = np.minimum(np.maximum(self.ypix - self.yoffha,0), self.fissHa.ny-1)
        if self.showCadata:
            xpixca = np.minimum(np.maximum(self.xpix - self.xoffca,0), self.fissCa.nx-1)
            ypixca = np.minimum(np.maximum(self.ypix - self.yoffca,0), self.fissCa.ny-1)
              
        if xchange:
            if self.showHadata:
                specHa = self.fissHa.data[:, xpixha]
                if self.scale == 'log':   specHa = log10(specHa)
                specHa1 = np.zeros((self.ny,specHa.shape[1]))+np.median(specHa)
                specHa1[self.yoffha:self.yoffha+self.fissHa.ny,:]=specHa
                self.imSpecHa.set_data(specHa1)
            if self.showCadata:
                specCa = self.fissCa.data[:, xpixca, ::-1]
                if self.scale == 'log':    specCa = log10(specCa)
                specCa1 = np.zeros((self.ny,specCa.shape[1]))+np.median(specCa) 
                specCa1[self.yoffca:self.yoffca+self.fissCa.ny,:]=specCa                 
                self.imSpecCa.set_data(specCa1)
            
        if ychange:
  
            if self.showHadata: self.hlineSpecHa.set_ydata(self.y)
            if self.showCadata: self.hlineSpecCa.set_ydata(self.y)
                           
        # Raster
        for i in range(len(self.position)):
            self.position[i].set_offsets([self.x, self.y])
            
        # IRaster
        if self.showHapar: 
          for i in range(len(self.Iposition)):
            self.Iposition[i].set_offsets([self.x, self.y])
        if self.showCapar: 
          for i in range(len(self.IpositionCa)):
            self.IpositionCa[i].set_offsets([self.x, self.y])
            
        # Profiles
        title= rf'$(x,y)$=({self.x:.2f},{self.y:.2f}) Mm = ({self.xpix:3},{self.ypix:3}) px'

        if self.showHadata:
            wvA1=self.fissHa.wave
            wvA = wvA1 + self.wvHaline
            profA0=self.fissHa.data[ypixha, xpixha]
            pureA=fpure(wvA, line='Ha')
            TparHa=FissGetTlines(wvA1, profA0, line='Ha')
            profA0 = FissCorTlines(wvA1, profA0, TparHa, line='Ha')
            spar= FissGetSline(wvA1, profA0)
    
            profA0 = FissCorSline(wvA1, profA0, spar)
            profA = FissStrayAsym(wvA1, profA0, self.fissHa.avProfile, pure=pureA)    
            if hasattr(self, 'haint'):            
                if self.showHapar:
                    self.haint.Renew(wvA1, profA,par= self.parHa[0:15, ypixha, xpixha],
                                          eps=self.parHa[15:17, ypixha, xpixha], 
                                          radloss=self.parHa[17:19,ypixha,xpixha])
                else:
                    self.haint.Renew(wvA1, profA)
                    t1=time()
                    self.haint.Initialize()               
                    self.haint.Fit()
                    t2=time()
                    print('Ha fit took ', t2-t1, ' s')
                    
                
            else:
                if self.showHapar:
                    self.haint = GuiThreerLayerModel(wvA1, profA, self.fissHa.refProfile, \
                                    par=self.parHa[0:15, ypixha, xpixha], \
                                    eps=self.parHa[15:17, ypixha,xpixha],
                                    radloss = self.parHa[17:19, ypixha, xpixha], line='Ha')
                else:
                    self.haint = GuiThreerLayerModel(wvA1, profA, self.fissHa.refProfile, line='Ha')                   
                    self.haint.Initialize()  
                    self.haint.Fit()
                    
            self.haint.axes[0].set_title(title, fontsize=11)
        
        if self.showCadata:
            wvB1=self.fissCa.wave
            wvB= wvB1+ self.wvCaline
                    
            profB0=self.fissCa.data[ypixca, xpixca]
            # if ((profA0.std() - 0.01*profA0.mean()) <= 0.) or ((profB0.std() - 0.01*profB0.mean()) <= 0.):
            #     return -1
            
            pureB=fpure(wvB, line='Ca')
            TparCa=FissGetTlines(wvB1, profB0, line='Ca')
            profB0 =  FissCorTlines(wvB1, profB0,  TparCa, line='Ca')
            profB = FissStrayAsym(wvB1, profB0, self.fissCa.avProfile, pure=pureB)
            if hasattr(self, 'caint'):
                if self.showCapar:
                    self.caint.Renew(wvB1,profB,self.parCa[0:15, ypixca,xpixca],
                                self.parCa[15:17, ypixca, xpixca],
                                self.parCa[17:19, ypixca, xpixca])
                else: 
                    
                    self.caint.Renew(wvB1,profB)
                    t1=time()
                    self.caint.Initialize()  
                    self.caint.Fit()
                    t2=time()
                    print('Ca fit took ', t2-t1, ' s')
            else:
                if self.showCapar:
                    self.caint = GuiThreerLayerModel(wvB1, profB, self.fissCa.refProfile, 
                                par=self.parCa[0:15, ypixca, xpixca], 
                                radloss = self.parCa[17:19, ypixca, xpixca], 
                                eps=self.parCa[15:17, ypixca, xpixca], line='Ca')
                else:
                    self.caint = GuiThreerLayerModel(wvB1, profB, self.fissCa.refProfile, line='Ca')
                    self.caint.Initialize()
                    self.caint.Fit()    
                
            self.caint.axes[0].set_title(title, fontsize=11)
        
    def _onKey(self, event):
        """
        To respond to each motion event. There are five keys for the designation
            space bar: read (x, y) from the current mouse position
            right: increase x position by one pixel
            left: decrease y poisition by one pixel
            up: increase y position by one pixel
            down: decrease y position by one pixel

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        eventinmap = (event.inaxes in self.axRaster)
        eventinsp = False
        if self.showHadata: eventinsp = eventinsp or (event.inaxes == self.axSpecHa)
        if self.showCadata: eventinsp = eventinsp or (event.inaxes == self.axSpecCa)
        if self.showHapar: 
            eventinmap = eventinmap or  (event.inaxes in self.axIRaster)  
        if self.showCapar:    
            eventinmap = eventinmap    or (event.inaxes in self.axIRasterCa)
        
        if (event.key == 'right') and eventinmap:
            if self.xpix < self.nx-1:
                self.xpix += 1
            self.xpix = np.minimum(np.maximum(self.xpix, 0), self.nx-1)     
            self.x = self.Xpix2Mm(self.xpix) 
            self.Reaction(xchange=True)
        elif (event.key == 'left') and eventinmap:
            if self.xpix > 0:
                self.xpix -= 1
            self.xpix = np.minimum(np.maximum(self.xpix, 0), self.nx-1)     
            self.x = self.Xpix2Mm(self.xpix) 
            self.Reaction(xchange=True)
        elif event.key == 'up':
            if self.ypix < self.ny-1:
                self.ypix += 1
            self.ypix=  np.minimum(np.maximum(self.ypix,0), self.ny-1) 
            self.y = self.Ypix2Mm(self.ypix) 
            self.Reaction(ychange=True)
        elif event.key == 'down':
            if self.ypix > 0:
                self.ypix -= 1
            self.ypix=  np.minimum(np.maximum(self.ypix,0), self.ny-1) 
            self.y = self.Ypix2Mm(self.ypix) 
            self.Reaction(ychange=True)
        elif event.key == ' ' and eventinmap:
            x = event.xdata
            y = event.ydata
            self.xpix = self.Xpix2Mm(x, invert=True)
            self.xpix = np.minimum(np.maximum(self.xpix, 0), self.nx-1) 
            self.x = self.Xpix2Mm(self.xpix) 
            self.ypix = self.Ypix2Mm(y, invert=True)
            self.ypix=  np.minimum(np.maximum(self.ypix,0), self.ny-1) 
            self.y = self.Ypix2Mm(self.ypix) 
            self.Reaction(xchange=True, ychange=True)
        elif event.key == ' ' and eventinsp:
            y = event.ydata
            self.ypix=  self.Ypix2Mm(y, invert=True) 
            self.ypix=  np.minimum(np.maximum(self.ypix,0), self.ny-1) 
            self.y = self.Ypix2Mm(self.ypix) 
            self.Reaction(ychange=True)
        
        
        if self.showHapar: 
            self.figIRaster.canvas.draw_idle()
        if self.showCapar:    
            self.figIRasterCa.canvas.draw_idle()
        self.figRaster.canvas.draw_idle()
        self.figSpec.canvas.draw_idle()
        
    def _Help(self):
        helpBox = plt.figure('Interactive Key', figsize=[5, 2.])
        ax = helpBox.add_subplot(111)
        ax.set_position([0,0,1,1])
        ax.text(0.05,0.9,'right: Move to right')
        ax.text(0.05,0.7,'left: Move to left')
        ax.text(0.05,0.5,'up: Move to up')
        ax.text(0.05,0.3,'down: Move to down')
        ax.text(0.05,0.1,'spacebar: Move to mouse point')
        
#%%
class FissAlign:
    
    def __init__(self, fHas, fCas,  refframe = None, align=False ):
        """
        To generate an instance of the class FissAlign

        Parameters
        ----------
        fHas : array of strings
            Names of FISS H alpha data files.
        fCas : array of strings
            Names of FISS Ca II data files.
        refframe : integer, optional
            refrerence frame number. If None, the middle frame is considered
            as the refeence. The default is None.
        align : boolean, optional
            If True, the align parameters are calculated. If False,  
            the align parameters are read from a pickle file. The default is False.

        Returns
        -------
        Name
            the name of the class instance.

        """
    
        nf= len(fHas)
        if len(fCas) != nf:
            print('Inputs fHas and FCas in FissAlign should have the same number of elements!')
            return -1
        
        self.nf = nf
        self.fHa = fHas
        self.fCa = fCas
        fHapar=[]
        fCapar=[]
        for f in range(nf):
            fHapar= np.append(fHapar,ThOutfile(self.fHa[f]))
            fCapar = np.append(fCapar, ThOutfile(self.fCa[f]))
        self.fHapar = fHapar
        self.fCapar = fCapar
          
        self.refline = 'Ha'
        self.parfile = False
        if not self.parfile:
            self.nx = (fits.getheader(fHas[0]))['NAXIS3']
        else:
            self.nx = (fits.getheader(fHas[0]))['NAXIS1']

        if refframe == None: 
            refframe = self.nf//2
        self.refframe = refframe   
        if align:
            self.AlignAll()
            self.DefBoard()
            with open('fissalign.pkl', 'wb') as file:
                pickle.dump((self.alignparHa, self.alignparCa, 
                             self.xrefrange, self.yrefrange, self.refstime, self.stimes, self.dtdays), file)
        elif os.path.isfile('fissalign.pkl'):
            with open('fissalign.pkl', 'rb') as file:
                data = pickle.load(file)
            self.alignparHa = data[0]
            self.alignparCa = data[1]
            self.xrefrange = data[2]
            self.yrefrange = data[3]
            self.refstime = data[4]
            self.stimes = data[5]
            self.dtdays = data[6]
#        self.DefBoard()
    
    def CoTrans(self, xi, yi, par,  direction = 'rtoi'): 
        """
        To do coordinate transform from reference frame to individual frame
        or from individual frame to reference frame
        
        All the coordinates are measured in unit of pixels. 
        The coordinate x  in each frame starts with 0 at the Solar easternmost pixel 
        and increases westward.   
        
        The coordinate y in each frame starts with 0 at the Solar southernmost pixel
        and increases northward.
        
        Parameters
        ----------
        xi : array_like
            input x-coordinate value(s)  in the reference frame if direction is 
            'rtoi', or in the rindividual frame if direction is 'itor'.
            
        yi : array_like
            input y-coordinate value(s) in the reference frame if direction is 
            'rtoi', or in the individual frame if direction is 'itor'.
                      
        par : array_like
            coordinate transform parameters.
              xcr, ycr, xc, yc, theta, dx, dy = par   
        direction : str, optional
            direction of coordinate transform. The default is 'rtoi'.
        
        Returns
        -------
        xo : array_like
            ouput x-coordinate value(s)  in the  individual frame if direction is 
            'rtoi', or in the reference frame if direction is 'itor'.
        yo : array_like
            ouput y-coordinate value(s)  in the  individual frame if direction is 
            'rtoi', or in the reference frame if direction is 'itor'.
        
        """
        
        yxratio, xcr, ycr, xc, yc, theta, dx, dy = par   
        
        if direction == 'rtoi':
            xref, yref = xi, yi
            xframe =  (xref - xcr)*cos(theta) + (yref-ycr)*sin(theta)/yxratio + xc + dx
            yframe = -(xref - xcr)*sin(theta)*yxratio + (yref-ycr)*cos(theta) + yc + dy
            xo, yo = xframe, yframe
        elif direction == 'itor' :             
            xframe, yframe = xi, yi
            xref =  (xframe -xc -dx)*cos(-theta)+ (yframe-yc-dy)*sin(-theta)/yxratio + xcr 
            yref = -(xframe -xc -dx)*sin(-theta)*yxratio + (yframe-yc-dy)*cos(-theta) + ycr 
            xo, yo = xref, yref
        
        return xo, yo
    def DefBoard(self):
        """
        To determine the x range  (self.xrefrange)  and y range (self.yrefrange) 
        of the reference board where all the aligned maps are to be put.         
        
        Returns
        -------
        None.

        """
        if self.refline == 'Ha':
            alignparref = self.alignparHa[:,self.refframe]
        else:
            alignparref = self.alignparCa[:,self.refframe]
        
        xrefmin, xrefmax, yrefmin, yrefmax = 0., alignparref[3]*2-1, 0., alignparref[4]*2-1
        
        for f in range(self.nf):
             for line in ('Ha', 'Ca'):
                if line == 'Ha':  
                    alignpar = self.alignparHa[:,f]
                else: 
                    alignpar = self.alignparCa[:, f]
                xfmin, xfmax, yfmin, yfmax = 0., alignpar[3]*2-1, 0., alignpar[4]*2-1
                xf = array([xfmin, xfmax, xfmax, xfmin])
                yf = array([yfmin, yfmin, yfmax, yfmax])
                xref, yref = self.CoTrans(xf, yf, alignpar, direction='itor')
                xrefmin = np.minimum(xrefmin, xref.min())
                yrefmin = np.minimum(yrefmin, yref.min())
                xrefmax = np.maximum(xrefmax, xref.max())
                yrefmax = np.maximum(yrefmax, yref.max())
#        print(xrefmin, xrefmax, yrefmin, yrefmax)
        xrefmin, xrefmax = int(xrefmin-1), int(xrefmax+1)
        yrefmin, yrefmax = int(yrefmin-1), int(yrefmax+1)
#        print(xrefmin, xrefmax, yrefmin, yrefmax)
        
        self.xrefrange = [xrefmin, xrefmax]
        self.yrefrange = [yrefmin, yrefmax]       
        
    def AlignAll(self, quiet=False):                   
        self.AlignLine(quiet=quiet)
        self.AlignHaCa(quiet=True)

        
    def AlignLine(self,  quiet=False):
        """
        To determine the series of align parameters from the series of 
        FISS data (either H alpha or Ca II) files. The line specified by 
        self.refline is used.

        Parameters
        ----------
        quiet : boolean, optional
            If True, no print message, no plot. The default is False.

        Returns
        -------
        None.

        """
 #       self.cors = np.zeros(self.nf)   
        nf = self.nf
        fref = self.refframe         
        alignpars = np.zeros((8, nf), float)
        yxratio = 1.
        alignpars[0] = yxratio
       
        if self.refline == 'Ha':
            filef = self.fHa[fref]  
        else:
            filef = self.fCa[fref]
        
#        if not self.parfile:
        fissr = FISS(filef)                
        imr=np.flip(fissr.data[:,:, 50:55].mean(2), 1)
        
        xcr = fissr.nx/2
        ycr = fissr.ny/2        
        refstime = fissr.date
            
        # else:
        #     p=4
        #     h = fits.getheader(filef)
        #     data = fits.getdata(filef)
        #     imr = data[p]*h[rf'SCALE{p:02d}']   
        #     xcr = h['NAXIS1']/2
        #     ycr = h['NAXIS2']/2
        #     tstr = h['FILEORIG']
        #     refstime = tstr[5:9]+'-'+tstr[9:11]+'-'+tstr[11:13]+'T' \
        #         + tstr[14:16]+':'+tstr[16:18]+':'+tstr[18:20]
        self.refstime = refstime       

        alignpars[1] = xcr
        alignpars[2] = ycr
        xc, yc =xcr, ycr
        alignpars[3,fref] = xc
        alignpars[4,fref] = yc
         
         # in day       
        nx1=np.minimum(self.nx, 100)
        ny1= 200
        xref = (np.arange(nx1)-nx1/2+xcr)+np.zeros(ny1)[:, None]
        yref = np.zeros(nx1)+np.arange(ny1)[:, None]-ny1/2+ycr
        im0 = imr
        im0sm = self.GetInt(im0, xref, yref)
        ref0sm = im0sm
        
        imgs = np.zeros((nf, ny1, nx1), float)
        imgsal = np.zeros((nf, ny1, nx1), float)
        stimes=[]
        dtdays=[]
        index=[]
        for k in range(0, nf):
            if k < fref:                
                f0 = fref - k 
                f = fref - k-1
            elif k == fref:
                 f= fref
                 imgs[f,:, :]=ref0sm
                 imgsal[f, :,:]=ref0sm
                 stime= refstime
                 dtday=0.
            elif k > fref:
                f0 = k-1
                f = k 
            if k != fref:    
                if self.refline == 'Ha':
                    file = self.fHa[f]  
                else:
                    file = self.fCa[f]
#                if not self.parfile:
                fiss = FISS(file)
                im=np.flip(fiss.data[:,:, 50:55].mean(2),1)
                stime = fiss.date
                dtday = Time(stime).jd - Time(refstime).jd
                
                # else:              
                #     h=fits.getheader(file)
                #     data = fits.getdata(file)
                #     im = data[p]*h[rf'SCALE{p:02d}']            
                #     tstr = h['FILEORIG']
                #     stime = tstr[5:9]+'-'+tstr[9:11]+'-'+tstr[11:13]+'T' \
                #         + tstr[14:16]+':'+tstr[16:18]+':'+tstr[18:20]
                #     dtday = Time(stime).jd - Time(reftime).jd   
                                    
                theta = -dtday*(2*np.pi)
                alignpar = alignpars[:,f]
                s=im.shape
                alignpar[3:6] = s[-1]/2, s[-2]/2, theta
                alignpar[6:8] = alignpars[6:8, f0]
                xf, yf = self.CoTrans(xref, yref, alignpar, direction='rtoi')
                imsm = self.GetInt(im, xf, yf)
                for rep in range(2):                 
                    sh, cor=alignoffset(imsm, im0sm,cor=True)
                    sh = sh.flatten()
                    alignpar[6:8] =  alignpar[6:8]+ [sh[1], sh[0]]
                    xf, yf = self.CoTrans(xref, yref, alignpar, direction='rtoi')
                    imsm = self.GetInt(im, xf, yf)
                im0sm = imsm
                imgsal[f,:,:]= self.GetInt(im, xf, yf)
                imgs[f,:,:]= self.GetInt(im, xref, yref)
                if not quiet: print('f=', f, ', dx=', alignpar[6], ', cor=', cor)
                alignpars[:, f] = alignpar
 #               self.cors[f] = cor
            stimes = np.append(stimes, stime)
            dtdays = np.append(dtdays, dtday)
            index =  np.append(index, f)
        self.stimes =np.array(stimes)[np.argsort(index)]
        self.dtdays =np.array(dtdays)[np.argsort(index)]
        if self.refline == 'Ha':
            self.alignparHa = alignpars 
            cmap=fisspy.cm.ha
        else:
            self.alignparCa = alignpars
            cmap=fisspy.cm.ca

        if not quiet:  
            fig, ax = plt.subplots(1,2, sharex=True, sharey = True, clear=True)
            im0=ax[0].imshow(imgs[0],origin='lower', cmap=cmap)
            
            im1=ax[1].imshow(imgsal[0], origin="lower", cmap=cmap)
            txt = ax[0].text(10,10, 'Frame = '+rf"{0-self.refframe:04.0f}" )
            m=imgs[self.refframe].mean()
            sigma=imgs[self.refframe].std()
            #print('m1, m2=', m1, m2)
            im0.set_clim(m-2.5*sigma, m+2*sigma)
            im1.set_clim(m-2.5*sigma, m+2*sigma)
            fig.tight_layout(pad=1)        
            def update(frame): 
                  im0.set_data(imgs[frame])
                  im1.set_data(imgsal[frame]) 
                  txt.set_text('Frame = '+rf"{frame-self.refframe:04.0f}")
                  
            ani = animation.FuncAnimation(fig, update, frames=imgs.shape[0], interval=50)
            ani.save('alignall.mp4',fps=5)
            plt.pause(1.) 
            
                   
            
    def AlignHaCa(self, quiet=False):
        """
        To determine the series of align parameters in a line based on the
        series of align parameters in the reference line, by using
        the relative shift between two kinds of data.

        Parameters
        ----------
        quiet : boolean, optional
            If Ture, no message print, not plot. The default is False.

        Returns
        -------
        None.

        """        
        nf = self.nf
        fref = self.refframe
        if self.refline == 'Ha':
            self.alignparCa = np.copy(self.alignparHa)
        else:
            self.alignparHa = np.copy(self.alignparCa)


        if not self.parfile:
            fissHa = FISS( self.fHa[fref])
            #fissHa.data = np.flip(fissHa.data,1)
            imHa=np.flip(fissHa.data[:,:, 50:55].mean(2),1)
                    
            fissCa = FISS(self.fCa[fref])
            #fissCa.data = np.flip(fissCa.data,1) 
            imCa=np.flip(fissCa.data[:,:, 50:55].mean(2),1)                
        else:
            p=4
            h=fits.getheader(self.fHa[fref])
            data = fits.getdata(self.fHa[fref])
            imHa = data[p]*h[rf'SCALE{p:02d}']
            h=fits.getheader(self.fCa[fref])
            data = fits.getdata(self.fCa[fref])
            imCa = data[p]*h[rf'SCALE{p:02d}']
                
        nx1=np.minimum(self.nx, 100)
        ny1= 200
        
        if self.refline == 'Ha':
            s=imCa.shape
            self.alignparCa[3,:]=s[-1]/2
            self.alignparCa[4,:]=s[-2]/2 
            xcr = self.alignparHa[1,fref]
            ycr = self.alignparHa[2,fref]
                    
            xref = (np.arange(nx1)-nx1/2+xcr)+np.zeros(ny1)[:, None]
            yref = np.zeros(nx1)+np.arange(ny1)[:, None]-ny1/2+ycr
            imref = self.GetInt(imHa, xref, yref)
            alignpar = self.alignparCa[:, fref]
            xf, yf = self.CoTrans(xref, yref, alignpar, direction='rtoi')
            imf = self.GetInt(imCa, xf, yf)
            cmref = fisspy.cm.ha
            cmf = fisspy.cm.ca

        if self.refline == 'Ca':
            s=imHa.shape
            self.alignparHa[3,:]=s[-1]/2
            self.alignparHa[4,:]=s[-2]/2 
            xcr = self.alignparCa[1,fref]
            ycr = self.alignparCa[2,fref]
                    
            xref = (np.arange(nx1)-nx1/2+xcr)+np.zeros(ny1)[:, None]
            yref = np.zeros(nx1)+np.arange(ny1)[:, None]-ny1/2+ycr
            imref = self.GetInt(imCa, xref, yref)
            
            alignpar = self.alignparHa[:, fref]
            xf, yf = self.CoTrans(xref, yref, alignpar, direction='rtoi')
            imf = self.GetInt(imHa, xf, yf)
            
            cmref = fisspy.cm.ca
            cmf = fisspy.cm.ha
        
            
        for rep in range(3):                 
            sh, cor=alignoffset(imf, imref,cor=True)
            sh = sh.flatten()
            alignpar[6:8] =  alignpar[6:8]+ [sh[1], sh[0]]
            xf, yf = self.CoTrans(xref, yref, alignpar, direction='rtoi')
            if self.refline == 'Ha':
                imf = self.GetInt(imCa, xf, yf)
            elif self.refline == 'Ca':
                imf = self.GetInt(imHa, xf, yf)
                
 #           if not quiet: im1=ax[1].imshow(imf, origin="lower", cmap=cmf)
            dx=alignpar[6]
            dy=alignpar[7]
            if not quiet: print('rep=', rep,', sh=', sh,', cor=', cor,', dx=',dx, ', dy=',dy)
        if not quiet:

            fig, ax = plt.subplots(1,2, sharex=True, sharey = True, clear=True)
            im0=ax[0].imshow(imref, origin='lower', cmap=cmref)
            im1=ax[1].imshow(imf, origin="lower", cmap=cmf)
            fig.tight_layout(pad=1)        

        if self.refline == 'Ha':    
            self.alignparCa[6,:]=self.alignparHa[6,:] + dx
            self.alignparCa[7,:]=self.alignparHa[7,:] + dy
        elif self.refline == 'H=Ca':    
            self.alignparHa[6,:]=self.alignparCa[6,:] + dx
            self.alignparHa[7,:]=self.alignparCa[7,:] + dy

            
    def GetInt(self, image, x, y, x0=0.,  y0=0., grid=False): 
        """
        To determine the image values at the specified position(s) using interpolation

        Parameters
        ----------
        image : 2-D array
            input image adta.
        x : array-like
            x positions.
        y : array-like
            y positions.
        x0 : float, optional
            the x-value of the origin. The default is 0.
        y0 : float, optional
            the y-value of the origin. The default is 0.
        grid : boolean, optional
            If set, the 1-D x values and the 1-D y values are used to
            construct the 2-D grids. The default is False.

        Returns
        -------
        values : array-like
            interpolated image values.

        """
        s=image.shape
        xx = x0+np.arange(s[-1])
        yy = y0+np.arange(s[-2])
        fimage = interpolate.RectBivariateSpline(yy,xx,image,kx=3,ky=3) #,fill_value=image.mean())
        values = fimage(y, x, grid=grid)
        return values
    def GetSpecSeries(self, xframe, yframe, line='Ha', frame=None, frange=None, quiet=False):
        """
        To obtain a time seris of spectra 

        Parameters
        ----------
        xframe : integer or float
            x position in a specified frame.
        yframe : integer or float
            y poisyion in a specified frame.
        line : str, optional
            spectral line designation. The default is 'Ha'.
        frame : integer, optional
            frame number where xframe and yframe are defined. If None, it is set
            to the reference frame (self.refframe). The default is None.
        frange : 2-element array, optional
            Starting and ending frame numbers. If None, it is set to [0, self.nf].
            The default is None.
        quiet : str, optional
            If True, no messages, no plots. The default is False.

        Returns
        -------
        spectra : 2-D array
            Spectral-temporal map of intensity.

        """
        if frame is None: frame = self.refframe
        if frange is None: frange = [0, self.nf]
        if line == 'Ha':
            alignpars = self.alignparHa
            files = self.fHa
            cmap = fisspy.cm.ha
        elif line == 'Ca':
            alignpars = self.alignparCa
            files = self.fCa
            cmap = fisspy.cm.ca
        xref, yref = self.CoTrans(xframe, yframe, alignpars[:,frame], direction ='itor')
        frange[1] = np.maximum(frange[1], frange[0]+1) 
        for f in range(frange[0], frange[1]):
            xf, yf = self.CoTrans(xref, yref, alignpars[:, f], direction='rtoi') 
            header = fits.getheader(files[f])
            
            nx = header['NAXIS3']
            ny = header['NAXIS2']
            x = nx - 1 - xf  # to determine from flipped data
            y = yf
            i, j = int(x), int(y)
            fiss = FISS(files[f], x1=i, x2=i+2)
            data0 = fiss.data[:,0,:]
            data1 = fiss.data[:,1,:]
            i = np.minimum(nx-2, np.maximum(0,i))
            j = np.minimum(ny-2, np.maximum(0,j))
            wx, wy = x-i, y-j              
            spectrum = (1-wy)*(1-wx)*data0[j,:]  +  wy*(1-wx)*data0[j+1,:]  \
                + (1-wy)*wx *data1[j,:]  + wy*wx*data1[j+1,:]
            if f == frange[0]:
                spectra = np.zeros((frange[1]-frange[0], fiss.data.shape[-1]))    
            if line == 'Ca': spectrum = np.flip(spectrum)
            spectra[f-frange[0]] = spectrum
        if not quiet:
            fig, ax = plt.subplots(1,1, sharex=True, sharey = True, clear=True)
            im0=ax.imshow(spectra,origin='lower', cmap=cmap)   
        return spectra    
    def GetFitParSeries(self, xframe, yframe, line='Ha', frange=None, 
                        quiet=True, frame=None):
        """
        To determine time series of MLSI parameters at a specified point

        Parameters
        ----------
        xframe : float
            x position  at a specified frame in pixels.
        yframe : float
            y position at a specified frame in pixels.
        line : str, optional
            line designation. The default is 'Ha'.
        frange : a 2-eleement array, optional
            the range of frame numbers. If None, the whole frames are used.
            The default is None.
        quiet : boolean, optional
            If False, no message, no plot. The default is True.
        frame : integer, optional
            number of where the position is spefied. If none, frame number is 
            set to self.refframe. The default is None.

        Returns
        -------
        pars : array
            time series of parameters. The time series of the p-th parameter
            is specified by pars[p,:] or simply pars[p].  
        """
        
        if frange is None: frange = [0, self.nf]
        if frame == None: frame = self.refframe
        if line == 'Ha':
            alignpars = self.alignparHa
            files = self.fHapar
            cmap = fisspy.cm.ha
        elif line == 'Ca':
            alignpars = self.alignparCa
            files = self.fCapar
            cmap = fisspy.cm.ca
        xref, yref = self.CoTrans(xframe, yframe, alignpars[:,frame], direction ='itor')
        frange[1] = np.maximum(frange[1], frange[0]+1) 
        for f in range(frange[0], frange[1]):
            xf, yf = self.CoTrans(xref, yref, alignpars[:, f], direction='rtoi') 
            header = fits.getheader(files[f])
            
            nx = header['NAXIS1']
            ny = header['NAXIS2']
            npar= header['NAXIS3']
            x = xf  # to determine from flipped data
            y = yf
            i, j = int(x), int(y)
            
            data = fits.getdata(files[f])
            
            
            
            data0 = np.empty((npar, ny), float)
            data1 = np.empty((npar, ny), float)
    
            for p in range(npar):
                data0[p,:] = data[p,:, i]*header[f'SCALE{p:02d}']
                data1[p,:] = data[p,:, i+1]*header[f'SCALE{p:02d}']            
            
            i = np.minimum(nx-2, np.maximum(0,i))
            j = np.minimum(ny-2, np.maximum(0,j))
            wx, wy = x-i, y-j              
            par = (1-wy)*(1-wx)*data0[:,j]  +  wy*(1-wx)*data0[:,j+1]  \
                + (1-wy)*wx *data1[:,j]  + wy*wx*data1[:, j+1]
            if f == frange[0]:
                pars = np.zeros((frange[1]-frange[0], len(par)))    
            pars[f-frange[0],:] = par
        if not quiet:
            #fig, ax = plt.subplots(1,1, sharex=True, sharey = True, clear=True)
            plt.figure()
#            print(pars.shape, pars[:,9].shape, self.dtdays.shape)
            plt.plot(self.dtdays*(24*60.), pars[:,9])            
            
        return pars  
 
    def GetRasterWave(self, wvs=None, line='Ha',  quiet=False, moviefile=None):
        """
        To obtain the spectral series of raster images

        Parameters
        ----------
        wvs: array, optional
            an array of wavelengths from line center. If None, 201 wavelengths are 
            selected from -1.5 A (H alpha line) or -1.0 (Ca II 8542) to
            1.5 A or 1.0 A with equal intevals.  The default is None.
            
        line : str, optional
            line designation. 'Ha' for the H alpha line, 'Ca' for the Ca II 8542 line.
            The default is 'Ha'.    
        quiet : boolean, optional
            If True, no messages, no plots.  The default is False.
        moviefile : str, optional
            name of movie file (without extension). If specified, the mp4 movie file is created. 
            The default is None.

        Returns
        -------
        imge :  3-d array
            an array of images.

        """        
        f=self.refframe
        if line == 'Ha':
            file = self.fHa[f]
            alignpar = self.alignparHa[:,f]
            cmap = fisspy.cm.ha
        else:
            file = self.fCa[f]
            alignpar = self.alignparCa[:,f]
            cmap = fisspy.cm.ca
        fiss = FISS(file)


        nwv = (fiss.data.shape)[-1]
        if (wvs is None):
            if line == 'Ha': 
                wvmax=1.5
            else:
                wvmax = 1.0
            nwvsel = 201
            wvs = (np.arange(nwvsel)-100.)/100.*wvmax
        else:
            nwsel = len(wvs)
            
        for k in range(nwvsel):
 
            wv1 =wvs[k]
 
            sel=abs(fiss.wave-fiss.centralWavelength-wv1)<0.025
            raster = log10(np.flip(fiss.data[:,:, sel].mean(2), 1))    
            
                        
            if k == 0:
                nyraster, nxraster = raster.shape          
                imgs = np.zeros((nwvsel, nyraster, nxraster))
    
            imgs[k, :, :] = raster- np.median(raster)

        if not quiet:  
            fig, ax = plt.subplots(1,1, sharex=True, sharey = True, clear=True)
            im0=ax.imshow(imgs[0],origin='lower', cmap=cmap, 
                          extent=(-0.5, nxraster+0.5, -0.5, nyraster+0.5))
#            m=imgs[:,nyraster//4:nyraster//4*3, nxraster//4:nxraster//4*3].mean()
#            sigma=imgs[:, nyraster//4:nyraster//4*3, nxraster//4:nxraster//4*3].std()
      
            m=0.
            sigma = 0.1
            im0.set_clim(m-2.5*sigma, m+1.5*sigma)

            txt=ax.text(10,10, r'$\lambda$ ='+f'{wvs[0]:0.2f}' + r'$\AA$')
            fig.tight_layout(pad=1)        
            def update(frame): 
                  im0.set_data(imgs[frame])
 
 #                 im1.set_data(imgsal[frame]) 
                  txt.set_text(r'$\lambda$ ='+f'{wvs[frame]:0.2f}' +r'$\AA$')
                  
            ani = animation.FuncAnimation(fig, update, frames=imgs.shape[0], interval=50)
            
            if not (moviefile == None): 
 #               FFwriter=animation.FFMpegWriter()
                ani.save(moviefile, fps=5 )#  writer=FFwriter)
            plt.pause(60.) 
            
        return imgs
    def GetRasterSeries(self, wv1, line='Ha', quiet=False, moviefile=None):
        """
        To determine a time series of monochromatic raster images

        Parameters
        ----------
        wv1 : float
            wavelength from line center in A.
        line : str, optional
            line designation, either 'Ha' or 'Ca'. The default is 'Ha'.
        quiet : boolean, optional
            If True, no messages, no plots. The default is False.
        moviefile : str, optional
            name of movie file (without extension). If specified, the mp4 file
            is created.The default is None.

        Returns
        -------
        imgs :  3-d array
            an array of monochromatic images
        """
        xrefmin, xrefmax = self.xrefrange
        yrefmin, yrefmax = self.yrefrange
        nxbig = xrefmax - xrefmin+1
        nybig = yrefmax - yrefmin +1
        
        xrefbig = (np.arange(nxbig)+xrefmin)+np.zeros(nybig)[:, None]
        yrefbig = np.zeros(nxbig)+(np.arange(nybig)+yrefmin)[:, None]
        
        nframe= self.nf
        
        for f in range(nframe):
            if line == 'Ha':
                file = self.fHa[f]
                alignpar = self.alignparHa[:,f]
                cmap = fisspy.cm.ha
            else:
                file = self.fCa[f]
                alignpar = self.parCa[:,f]
                cmap = fisspy.cm.ca
            fiss = FISS(file)
            sel=abs(fiss.wave-fiss.centralWavelength-wv1)<0.05
            raster = log10(np.flip(fiss.data[:,:, sel].mean(2), 1))
            xf, yf = self.CoTrans(xrefbig, yrefbig, alignpar, direction='rtoi') 
            
            nyraster, nxraster = raster.shape
            raster1 = np.zeros((nyraster+2, raster.shape[-1]+2))+raster.mean()
            raster1[1:-1, 1:-1] = raster
            if f == 0:
                imgs = np.zeros((nframe, nybig, nxbig))+raster.mean()
    
            imgs[f, :, :] =self.GetInt(raster1, xf+1, yf+1)
            if not quiet: 
                print('f=', f)
#                plt.imshow(imgs[f], origin='lower')

        if not quiet:  
            fig, ax = plt.subplots(1,1, sharex=True, sharey = True, clear=True)
            im0=ax.imshow(imgs[0],origin='lower', cmap=cmap, 
                          extent=(xrefmin-0.5, xrefmax+0.5, yrefmin-0.5, yrefmax+0.5))
            m=imgs[:,nybig//4:nybig//4*3, nxbig//4:nxbig//4*3].mean()
            sigma=imgs[:, nybig//4:nybig//4*3, nxbig//4:nxbig//4*3].std()
            
            im0.set_clim(m-3*sigma, m+3*sigma)
            fig.tight_layout(pad=1)        
            def update(frame): 
                  im0.set_data(imgs[frame])
 #                 im1.set_data(imgsal[frame]) 
 #                 txt.set_text('Frame = '+rf"{frame-self.refframe:04.0f}")
                  
            ani = animation.FuncAnimation(fig, update, frames=imgs.shape[0], interval=100)
            if not (moviefile == None): ani.save(moviefile+'.mp4',fps=5)
            plt.pause(10.) 
            
            
        return imgs
    def GetParmapSeries(self, line='Ha', quiet=False, moviefile=None, savefile=None):
        """
        To determine time series of maps of parameters.

        Parameters
        ----------
        line : str, optional
            line designation, either 'Ha' or 'Ca'. The default is 'Ha'.
        quiet : boolean, optional
            If True, no messages, no plots. The default is False.
        moviefile : str, optional
            name of movie file (w/o extension). If specified, the movie file is
            generated. The default is None.
        savefile : str, optional
            name of fits file (w/o extension). If specifed, the 4-d array is 
            saved in the fits file. The default is None.

        Returns
        -------
        pardata : 4-d array
            arrays of maps of parameters. pardata[p,:,:,:] or simply paradta[p] 
            refers to the time series array of maps of p-th parameter. 
        """
        xrefmin, xrefmax = self.xrefrange
        yrefmin, yrefmax = self.yrefrange
        nxbig = xrefmax - xrefmin+1
        nybig = yrefmax - yrefmin +1
        
        xrefbig = (np.arange(nxbig)+xrefmin)+np.zeros(nybig)[:, None]
        yrefbig = np.zeros(nxbig)+(np.arange(nybig)+yrefmin)[:, None]
        
        nframe= self.nf
        
        for f in range(nframe):
            if line == 'Ha':
                file = self.fHapar[f]
                alignpar = self.alignparHa[:,f]
                cmap = fisspy.cm.ha
            else:
                file = self.fCapar[f]
                alignpar = self.alignparCa[:,f]
                cmap = fisspy.cm.ca
            xf, yf = self.CoTrans(xrefbig, yrefbig, alignpar, direction='rtoi') 
            
            h = fits.getheader(file)
            data = fits.getdata(file)
        
            if f == 0:
                nx = h['NAXIS1']
                ny = h['NAXIS2']
                npar= h['NAXIS3']            
                pardata = np.zeros((npar, nframe, nybig, nxbig), float)
            if f == self.refframe:
                href = h
            for p in range(npar):  
                data1= data[p]*h[f'SCALE{p:02d}']
                buff = np.zeros((ny+2, nx+2))+ data1.mean()
                buff[1:-1, 1:-1] =data1                
                pardata[p,f, :, :] =self.GetInt(buff, xf+1, yf+1)
            if not quiet: 
                print('f=', f)
#                plt.imshow(imgs[f], origin='lower')

        if not quiet:  
            fig, ax = plt.subplots(1,1, sharex=True, sharey = True, clear=True)
            im0=ax.imshow(pardata[9,0],origin='lower', cmap='seismic', 
                          extent=(xrefmin-0.5, xrefmax+0.5, yrefmin-0.5, yrefmax+0.5))
            m=pardata[9,:,nybig//4:nybig//4*3, nxbig//4:nxbig//4*3].mean()
            sigma=pardata[9,:, nybig//4:nybig//4*3, nxbig//4:nxbig//4*3].std()
            
            im0.set_clim(m-3*sigma, m+3*sigma)
            fig.tight_layout(pad=1)        
            def update(frame): 
                  im0.set_data(pardata[9,frame])
 #                 im1.set_data(imgsal[frame]) 
 #                 txt.set_text('Frame = '+rf"{frame-self.refframe:04.0f}")
                  
            ani = animation.FuncAnimation(fig, update, frames=nframe, interval=100)
            if not (moviefile == None): ani.save(moviefile+'.mp4',fps=5)
            plt.pause(1.)
            
        if not (savefile == None):
            
            pari=np.empty((npar,nframe, nybig, nxbig), np.int16)            
            par_scale=np.empty(npar, float)
            for p in range(npar): 
                par_scale[p] = href[f'SCALE{p:02d}'] #abs(pardata[p]).max()/32000.
        #        print(par_scale[p]) 
            for p in range(npar): 
                pari[p] = np.int16(np.round(pardata[p]/par_scale[p]))
            HDUout=fits.PrimaryHDU(pari)
            HDUout.header.set('REFFRAME', self.refframe)
            if self.refline == 'Ha': 
                reffile = self.fHa[self.refframe]
            elif self.refline == 'Ca':
                reffile = self.fCa[self.refframe]
            s=reffile.find('FISS_')  
            HDUout.header.set('REFFILE',reffile[s:] )
            HDUout.header.set('XREF0', self.xrefrange[0])
            HDUout.header.set('YREF0', self.yrefrange[0])
            for p in range(npar):
                HDUout.header.set(f'SCALE{p:02d}', par_scale[p], 'for '+line+' '+ ThreeLayerPar(p)) 
            for f in range(nframe):
                HDUout.header.set(f'TIME{f:03d}', self.stimes[f]) 
        
            HDUout.writeto(savefile+'.fts', overwrite=True, checksum=True)
    
        return pardata
    
#%%
def ThInvertFile(Infile, Outfile=None, x1=None, x2=None, y1=None, y2=None, 
                logfile = None, quiet = True):
    """
    To do three-layer spectral inversion of a FISS file. If the input FISS file
    has too poor quality, the inversion is not done. 

    Parameters
    ----------
    Infile : str
        name of input FISS data file.
    Outfile : str
        name of output model parameters file.
    x1 : integer, optional
        starting number of step. The default is None.
    x2 : integer, optional
        ending number of step. The default is None.
    y1 : integer, optional
        starting number of position along the slit. The default is None.
    y2 : integer, optional
        starting number of position along the slit. The default is None.
    logfile : str, optional
        File name of process log file. If not set, log information is written
        into 'TheInvertFile_log.txt'. The default is none.
    quiet : boolean, optional
        If True, the processing information is printed in the log file. 
        The default is True.

    Returns
    -------
    None.

    """
    if not Outfile: Outfile = ThOutfile(Infile)
    t=Infile.find('_c.fts')
    u=Infile.find('FISS_')    
    message=str('Inversion of  %s  starting at %s \n' % (Infile[u:t],  
                        strftime("%Y-%m-%d %H:%M:%S KST", localtime() ))     )
    if not logfile:
        logfile = 'TheInvertFile_log.txt'
    if not quiet:     
        stdout.write(message)   
    else: 
        with  open(logfile,'a') as outfile:  outfile.write(message)  
    nbad, ic = FissBadsteps(Infile)
    s =ic.shape
    npoint=s[0]*s[1]
    if nbad  > 5:
        message=str('Inversion of  %s was not done because of poor quality.\n' 
                              % (Infile[u:t]))
        if not quiet:
            stdout.write(message)   
        else: 
            with  open(logfile,'a') as outfile:  outfile.write(message)      
        success = False    
        return success
    
    t1=time()
    
    fiss = FISS(Infile) 
    wvline = fiss.centralWavelength
        
    fiss.data = np.flip(fiss.data, 1)
    icont=fiss.data[:,:, 50]
    qs=icont > (icont.max()*0.7)
    fiss.avProfile = fiss.data[qs,:].mean(0)
    
    
    if abs(wvline-6562.817)  < 10.: line='Ha'
    if abs(wvline-8542.091)  < 10.: line='Ca'
    
    fiss.wave=Recalibrate(fiss.wave, fiss.avProfile, line=line)
          
    wv1=fiss.wave-wvline 
    if line == 'Ha':
        refc = fiss.avProfile[abs(abs(wv1)-4.5)<0.05].mean()*1.01
    elif line =='Ca':
        refc = fiss.avProfile[abs(abs(wv1)-4.5)<0.05].mean()*1.09
    fiss.avProfile = fiss.avProfile/refc
    fiss.data = fiss.data/refc
    
    pure=fpure(fiss.wave, line=line)
    if  (x1 is None): x1=0
    if  (x2 is None): x2=fiss.nx
    if  (y1 is None): y1=0
    if  (y2 is None): y2=fiss.ny
    
    npar=19
    par = np.zeros((npar, y2-y1, x2-x1), float)
    
    Tpar = FissGetTlines(wv1, fiss.avProfile, line=line)
    fiss.refProfile = FissCorTlines(wv1, fiss.avProfile, Tpar, line=line) 
    fiss.refProfile = FissStrayAsym(wv1, fiss.refProfile, fiss.avProfile, pure=pure)    
   
   
    for xpix in range(x1,x2):
      if not quiet: print('xpix=', xpix)  
      for ypix in range(y1,y2):   
#        if not quiet: print('ypix=', ypix)
        prof0 = fiss.data[ypix,xpix]
        Tpar = FissGetTlines(wv1, prof0, line=line)
        prof0 = FissCorTlines(wv1,prof0, Tpar, line=line) 
        if line == 'Ha':
            spar= FissGetSline(wv1, prof0)
            prof0 = FissCorSline(wv1, prof0, spar)  
        prof = FissStrayAsym(wv1, prof0, fiss.avProfile, pure=pure)
              
        pp, I0, I1, I2, epsD, epsP = ThreeLayerModel(wv1, prof,line=line)          
        Radloss=ThreeLayerRadLoss(pp, line=line)
        par[:,ypix-y1, xpix-x1] =np.append(pp, 
                   (epsD, epsP, Radloss[1], Radloss[0] ) )
                
    par_scale=np.zeros(npar, float)
    for p in range(npar): 
        par_scale[p] = abs(par[p,:,:]).max()/32000.
        
    pari= np.zeros((npar,y2-y1,x2-x1), np.int16)
    for p in range(npar): 
        pari[p,:,:] = np.int16(np.round(par[p,:,:]/par_scale[p]))
    hdu=fits.PrimaryHDU(pari)
    header=hdu.header
    header.set('xstart', x1)
    header.set('ystart', y1)
    for p in range(npar): 
        header.set(rf'SCALE{p:02d}', par_scale[p], 'for '+ ThreeLayerPar(p) ) 
    s=Infile.find('FISS_')
    header.set('fileorig', Infile[s:])
    header.set('package', package, 'used for processing')
    hdu.writeto(Outfile,overwrite=True)    

    t2=time()

    message= str('Mean computing time for 1 point = %.0f [ms] \n' % ( (t2-t1)/npoint*1000))
    clock_time= strftime("%Y-%m-%d %H:%M:%S KST", localtime())
    message2= str('Inversion of %s finished at %s and took %.2f min  \n' 
                  % (Infile[u:t],clock_time,(t2-t1)/60. ) )
    if not quiet:
         stdout.write(message)   
         stdout.write(message2) 
    else: 
        with  open(logfile,'a') as outfile:   
            outfile.write(message)
            outfile.write(message2) 
   
    success = True      
    return success           

    
def ThOutfile(Infile):
    """
    To construct the name of output parameter file for a FISS spectral file.
    The parameter data file is saved in to subdirectory 'pars'. 

    Parameters
    ----------
    Infile : str
        Name of FISS data file.

    Returns
    -------
    Outfile : str
        Name of output parameter file.

    """
    t=Infile.find('_c.fts') 
    s=Infile.find('FISS_')
    outdir = Infile[0:s]+'pars' 
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    Outfile = outdir+'\\'+Infile[s:t]+'_par.fts' 
    return Outfile
# def ThInvertDirMp(Indir, Ainvert = True, Binvert = True, overwrite=False):
#     """
#     To apply ThreeLayer Inversion to the files in a directory

#     Parameters
#     ----------
#     Indir : str
#         name directory containing  the input FISS files.
#     Ainvert : boolean, optional
#         True if all the Ha files are to be inverted. The default is True.
#     Binvert : boolean, optional
#         True if all the Ca II files are to be inverted. The default is True.
#     overwrite : boolean, optional
#         True if the parameter files are obtained and overwritten even if 
#         parameter files already exist.  The default is False.

#     Returns
#     -------
#     None.

#     """
     
#     Fiterable =[""]
#     if Ainvert:
#         filesA=glob.glob(Indir+'FISS*A1_c.fts')  
#         for InfileA in filesA:
#             OutfileA = ThOutfile(InfileA) 
#             if not os.path.isfile(OutfileA) or overwrite:
#                 Fiterable.append((InfileA, OutfileA))
#     if Binvert:    
#         filesB=glob.glob(Indir+'FISS*B1_c.fts')       
#         for InfileB in filesB:
#             OutfileB = ThOutfile(InfileB)             
#             if not os.path.isfile(OutfileB) or overwrite:
#                 Fiterable.append((InfileB, OutfileB))
#     Fiterable=Fiterable[1:]
#     pool = mp.Pool(2)
#     pool.map(ThInvertFile, Fiterable)
#     pool.close()
#     pool.join()
    



# parHa=array([-0.395 ,  0.4923, -0.644 ,  0.4858, -0.004 , -0.1789,  0.699 ,
#          0.699 , -0.5748, -0.404 , -0.3948, -0.3964, -0.3125, -0.5347,
#         0.144 ])
# # # array([-0.793 ,  0.5071, -0.64  ,  0.5277,  0.0373, -0.2528,  0.699 ,
# # #         0.699 ,  1.7834,  2.7822, -0.405 , -0.4254, -0.3472, -0.692 ,
# # #         0.145 ])
# # # array([ 0.227 ,  0.4912, -0.647 ,  0.4944, -0.0152, -0.2489,  0.699 ,
# # #         0.699 ,  0.3699,  0.0563, -0.4802, -0.5794, -0.4581, -0.9239,
# # #         0.143 ])

# parCa=array([-0.75  ,  0.3741, -1.268 ,  1.295 , -0.0092, -0.3656,  0.699 ,
#         0.699 ,  0.8927, -3.5456, -0.6919, -0.6911, -0.1665, -0.3544,
#         0.271 ])
# # array([-1.46  ,  0.3884, -1.265 ,  1.3843,  0.0478, -0.5274,  0.699 ,
# #         0.699 ,  2.0839,  2.133 , -0.6556, -0.6486, -0.3028, -0.5778,
# #         0.273 ])
# # array([-0.239 ,  0.3773, -1.27  ,  1.321 , -0.0087, -0.5586,  0.699 ,
# #         0.699 ,  0.579 ,  1.6504, -0.8214, -0.8899, -0.422 , -1.1711,
# #         0.27  ])

# RlHa=ThreeLayerRadLoss(parHa, line='Ha')


# RlCa=ThreeLayerRadLoss(parCa, line='Ca')

# print("Radiative loss in Ha=", RlHa, ", in Ca=", RlCa )

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# fig, ax = plt.subplots()
# im=ax.imshow(imgs[0,:,:], origin='lower')
# def update(frame):
#     im.set_data(imgs[frame,:,:])
#     return im,
# ani = FuncAnimation(fig, update, frames=np.arange(0, 200))
# plt.show()


# fig, ax = plt.subplots(1,2, sharex=True, sharey = True, clear=True)
  
# im0=ax[0].imshow(imgs[0],origin='lower')
# im1=ax[1].imshow(imgsal[0], origin="lower")
# fig.tight_layout(pad=1)        
# def update(frame): 
#       im0.set_data(imgs[frame])
#       im1.set_data(imgsal[frame])
    
# ani = animation.FuncAnimation(fig, update, frames=imgs.shape[0])
# plt.show()  
#        ani.save('test.mp4',fps=5)
 