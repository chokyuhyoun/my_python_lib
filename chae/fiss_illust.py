 # -*- coding: utf-8 -*-
"""
This contains a set of example programs illustrating the functions and classes 
in the package fissprocinv

@author: user
"""

#--------------------------------------------------------------------------------           
#                  Main Part
#
import numpy as np
import os, glob
from os.path import dirname
from os import chdir
#from fissprocinv import ThreeLayerPar, AniParFile, GuiFissRaster, FissAlign
#from fissprocinv import ThInvertFile, ThOutfile, ThInvertDirMp, AlignParFiles, AniParFile
import fissprocinv as fpi
import matplotlib.pyplot as plt
from time import time 
from fisspy.read import FISS
import fisspy
from scipy import interpolate
from scipy.ndimage import convolve
import multiprocessing as mp
from astropy.io import fits
from astropy.time import Time
#import matplotlib as mpl
#mpl.rcParams['lines.linewidth'] = 2
#mpl.rcParams['backend']='Qt5Agg'
db=None
plt.close('all')
scriptDir = dirname(__file__)
dataDir = scriptDir.replace('python', 'data')
chdir(scriptDir)
 
if 1:
    datadir='data\\'
    filesA=(glob.glob(datadir+'FISS_*A1_c.fts')) 
    filesB=(glob.glob(datadir+'FISS_*B1_c.fts'))  
    fHa = filesA[0]
    fCa = filesB[0] 
    print('fHa=', fHa, ', fCa=', fCa)           

# GUI fiss raster witht par data         

if 0:    
    db = fpi.GuiFissRaster(fHa=fHa) 
if 0:    
    db = fpi.GuiFissRaster(fHa=fHa, showpar=True)         
if 0:    
    db = fpi.GuiFissRaster(fHa=fHa, fCa=fCa)          
if 0:    
    db = fpi.GuiFissRaster(fHa=fHa, fCa=fCa, showpar=True)  

# Taking a look at individual profiles

if 0: # Processing for the reference profile 

    fissHa = FISS(fHa)   # 6562.82
    wvHaline = fissHa.centralWavelength #6562.817
    fissHa.data = np.flip(fissHa.data,1) 
          # constructing the average profile
    icont=fissHa.data[:,:, 50]
    qs=icont > (icont.max()*0.7)
    fissHa.avProfile = fissHa.data[qs,:].mean(0)
          # Recalibrating wavelengths 
    fissHa.wave = fpi.Recalibrate(fissHa.wave, fissHa.avProfile, line='Ha')                                
    fissHa.wave = fissHa.wave - fissHa.centralWavelength
            # Re-normalizing the data
    harefc = fissHa.avProfile[abs(abs(fissHa.wave)-4.5)<0.05].mean()*1.01
    fissHa.avProfile = fissHa.avProfile/harefc
    fissHa.data = fissHa.data/harefc
            # Processings for the reference profile    
    wvA1=fissHa.wave
    wvA = wvA1 + wvHaline
            # Specifying the unblended wavelengths
    pureA=fpi.fpure(wvA, line='Ha')    
            # Correction for Tereestrial Lines
    TparHa = fpi.FissGetTlines(wvA1, fissHa.avProfile, line='Ha')
    fissHa.refProfile = fpi.FissCorTlines(wvA1, fissHa.avProfile, TparHa,line='Ha')
            # Correction for stray light and far wing blue-red asymmetry
    fissHa.refProfile = fpi.FissStrayAsym(wvA1, fissHa.refProfile, 
                                                   fissHa.avProfile, pure=pureA)    
    plt.figure()
    plt.plot(wvA1, fissHa.avProfile)
    plt.plot(wvA1, fissHa.refProfile) 
    plt.yscale('log')
    plt.ylim(0.05, 2)

    if 1:  # Processing the profile at a specific point
        ypixha, xpixha = 180, 50   

        profA0=fissHa.data[ypixha, xpixha]
        TparHa=fpi.FissGetTlines(wvA1, profA0, line='Ha')
        profA = fpi.FissCorTlines(wvA1, profA0, TparHa, line='Ha')
        spar= fpi.FissGetSline(wvA1, profA)
        profA = fpi.FissCorSline(wvA1, profA, spar)
        profA = fpi.FissStrayAsym(wvA1, profA, fissHa.avProfile, pure=pureA) 
        fprof = interpolate.interp1d(wvA1[pureA], profA[pureA])         
        wvnew = wvA1[abs(wvA1) <3.]    
        profnew = fprof(wvnew)
        plt.figure()
        plt.plot(wvA1, profA0)
        plt.plot(wvA1, profA) 
        plt.plot(wvnew, profnew, '.g', ms=1)
        plt.yscale('log')
        plt.ylim(0.05, 2)

 
    if 1:   # Fitting the profile using the interactive tool
        
        haint = fpi.GuiThreerLayerModel(wvA1, profA, fissHa.refProfile, line='Ha')  
        haint.Fit()
        pars=haint.par
        print('Fit parameters=', pars)
        
# Three Layer Inversion of H alpha data file    
if 0:   
    fHa='C:\\work\\FISS\\data\\20140603\\FISS_20140603_165321_A1_c.fts'
    fpi.ThInvertFile(fHa,  quiet=False)
    
# Three Layer Inversion of Ca II data file        
if 0:   
    fCa='C:\\work\\FISS\\data\\20140603\\FISS_20140603_165321_B1_c.fts'
    fpi.ThInvertFile(fCa,  quiet=False)  

# Apply Three-Layer Inversion to all the files in a directory    
if 0:  
        #  - Multiprocessing should be done in the main level
        #  - Warning:  this processes will take long, even up to a day or longer
        #        depending the speed of the computer
        
    Indir = datadir
    Ainvert = True
    Binvert = True
    overwrite = True
    Flist =[""]
    if Ainvert:
        filesA=glob.glob(Indir+'FISS*A1_c.fts')  
        for InfileA in filesA:
            OutfileA = fpi.ThOutfile(InfileA) 
            if not os.path.isfile(OutfileA) or overwrite:
                Flist.append(InfileA)
    if Binvert:    
        filesB=glob.glob(Indir+'FISS*B1_c.fts')       
        for InfileB in filesB:
            OutfileB =fpi.ThOutfile(InfileB)             
            if not os.path.isfile(OutfileB) or overwrite:
                Flist.append(InfileB)
    Flist=Flist[1:]
    if __name__ == "__main__":   
        pool = mp.Pool(np.maximum(1,os.cpu_count()//4))
        pool.map(fpi.ThInvertFile, Flist)
        pool.close()
        pool.join()
                                                     
# Prepare a class for aligning a time series of FISS files        
if 1:
 
    fa = fpi.FissAlign(filesA, filesB, align=True)

    xx, yy = 60, 120
#   Wavelegnth-lapse Movie of  the reference raster      
    if 0:      # H alpha
        imgshawv =fa.GetRasterWave( line='Ha', moviefile='hawv')
    if 0:      # Ca II
        imgscawv = fa.GetRasterWave( line='Ca', moviefile='cawv')
# Time series of spectra at a given position        
    if 1:
        spectraha = fa.GetSpecSeries(xx, yy, line='Ha')
    if 0:
        spectraca = fa.GetSpecSeries(xx, yy, line='Ca') 
        
#   Time-lapse Movie of monochromatic images with a fiexed wavelength      
    if 0:
        imgshacen = fa.GetRasterSeries(0., line='Ha', moviefile='hacen')
    if 0:    
        imgshab05 = fa.GetRasterSeries(-1., line='Ha', moviefile='hab10')
#  Time series of model parameters at a given position
        
    if 1:
        pars = fa.GetFitParSeries(xx, yy, line='Ha')
        parsca = fa.GetFitParSeries(xx, yy, line='Ca')
        fig, ax = plt.subplots(1,1, figsize=[12, 6])
        #ax=ax.flatten()
        ax.plot(fa.dtdays*(24*60), pars[:, 9]-np.median(pars[:,9]) )
        ax.plot(fa.dtdays*(24*60),(parsca[:,9]-np.median(parsca[:,9])))
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('v0 (km/s')
#        ax.set_xlim(0, 30.)
#        ax.plot([-100,100], [0,0], linewidth=0.5)
# Time series of aligned maps of model parameters        
    if 0:
        pardata = fa.GetParmapSeries(line='Ha', quiet=False, 
                                     savefile='par20140603A') 
# Reading from a file of aligned maps of model parameters        
    if 0:
        parsfile='par20140603A.fts'
        h=fits.getheader(parsfile)
        xref0 = h['XREF0']
        yref0 = h['YREF0']
        pars = fits.getdata(parsfile)
        nframe  = h['NAXIS3']
        refframe = h['REFFRAME']
        dtdays = np.empty(nframe, float)
        for f in range(nframe):
           dtdays[f]=Time(h[f'TIME{f:03d}']).jd 
        dtdays=dtdays-dtdays[refframe]   
        p = 9
        v = pars[9,:,yy-yref0, xx-xref0]*h[f'SCALE{p:02d}']    
        fig, ax = plt.subplots(1,1, figsize=[12, 6])
        ax.plot(fa.dtdays*(24*60), v-np.median(v) )
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('v0 (km/s')
#        ax.set_xlim(0, 30.)
#        ax.plot([-100,100], [0,0], linewidth=0.5)        
        
        
    # Outfile=datadir+'Data2014060319'    
    # AlignParFiles(filesA, filesB, Outfile) 
    # AniParFile(Outfile)
            