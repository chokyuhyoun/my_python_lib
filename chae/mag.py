# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:47:32 2020

@author: user
;
; NAME :
;          B_LFF
; PURPOSE :
;          Linear Force-Free Extrapolation of Magnetic Field
; CALLING SEQUENCE:
;          B_LFF, bz0, z, bx, by, bz, alpha=alpha, seehafer=seehafer,  green=green
;
; INPUT :
;           bz0 : 2-d array of vertical field at z=0 plane
;           z   : 1-d array of heights (in unit of pixels)
; OUTPUT :
;           bx, by, bz : 3-d arrays of field components
; MODIFICATION HISTORY:
;        October 2002  Jongchul Chae, generalized from a_pot.pro
;                    Reference: Nakagawa and Raadu 1972, Solar Physics, 25, 127
;                                       Seehafer 1978, Solar Physics, 58, 215 (keyword SEEHAFER)
;        2009 October, J. Chae:
;                      Implemented the Green fucntion method (keywod GREEN)
;                             Chiu and Hilton 1977
;        2021 March, J. Chae:  coded in Python
;


"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt,  cos, sin, fft
from astropy.io import fits
from scipy.ndimage import rotate
from scipy.interpolate import interpn
import glob,pickle
from matplotlib.collections import LineCollection
def b_lff( bz0, z,  alpha=0. ):        
    nx1=(bz0.shape)[-1]
    ny1=(bz0.shape)[-2]
    nz = len(z)
    bx = np.empty((nz,ny1,nx1), float)
    by = np.empty((nz,ny1,nx1), float)
    bz = np.empty((nz,ny1,nx1), float)
    
    nx2=nx1*2
    ny2=ny1*2
    bz0e = np.zeros((ny2,nx2), float)
    bz0e[0:ny1,0:nx1]=bz0-bz0.mean()
    
    fbz0 = fft.fft2(bz0e)
    
    x = np.roll(np.arange(-nx2//2, nx2//2,dtype=float), -nx2//2) * (np.ones(ny2, float))[:, None]
    y = np.ones(nx2, float) * (np.roll(np.arange(-ny2//2, ny2//2, dtype=float), -ny2//2))[:,None]
    
    small=0.01
    for j in range(nz):
        zz = np.maximum(z[j], small)
        xt = x/zz
        yt = y/zz
        zt = zz*alpha
        rt1 = np.maximum(sqrt(xt**2+yt**2), small)
        rt  = sqrt(xt**2+yt**2+1.)
            
        Gamma =(1./rt * cos (zt*rt)-cos(zt))/rt1/zz
        Gamma_z = (1./rt*cos(zt*rt)*(1-1./rt**2) - zt/rt**2*sin(zt*rt) + zt*sin(zt))/rt1/zz**2
        Gx= (xt/rt1)*Gamma_z + zt*Gamma*(yt/rt1)/zz
        Gy= (yt/rt1)*Gamma_z - zt*Gamma*(xt/rt1)/zz
        if zz < 1:  den = (1./rt**3/zz**2).sum()/(2*np.pi)  
        else:  den = 1.
        Gz=(1/rt**3/zz**2)*(cos(zt*rt)+zt*rt*sin(zt*rt))/den
    
        tmp=fft.ifft2( fbz0*fft.fft2(Gx/(2*np.pi))) 
                      
        bx[j,:,:]=(tmp.real)[0:ny1, 0:nx1]
        tmp=fft.ifft2(fbz0*fft.fft2(Gy/(2*np.pi))  )
        by[j,:,:]= (tmp.real )[0:ny1, 0:nx1]
        tmp=fft.ifft2(fbz0*fft.fft2(Gz/(2*np.pi))  )
        bz[j,:,:]= (tmp.real )[0:ny1, 0:nx1] + bz0.mean()
    return bx, by, bz
def b_line(bx, by, bz, z,  r0, dx=1., dy=1., ds=1.):
    
    nx = bx.shape[-1]
    ny = bx.shape[-2]
    nz = bx.shape[-3]
    x=np.arange(nx)*dx
    y=np.arange(ny)*dy
    
    grids = (z,y,x)
    if nz != len(z):
        print("The number of z values should be equal to the z-elements of bx")
        return
    
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    zmin = z.min()
    zmax = z.max()
    
    r1 = np.copy(r0)
   
     
    b1 = np.array([interpn(grids, bz, r1, bounds_error=False)[0], 
                   interpn(grids, by, r1, bounds_error=False)[0], 
                   interpn(grids, bx, r1, bounds_error=False)[0]])
    if b1[0] < 0: ds=-1.
    c1 = b1/ np.linalg.norm(b1)   
    
    c2guess = np.copy(c1)
    r=r1.reshape(1,3)
    b=b1.reshape(1,3)
    inside = True 
    npoint=0
    while inside:
        converge = False
        itn = 0
        while not converge and itn < 20:  
            c2= c2guess    
            c= (c1+c2)/2.
            r2 = r1 + c*ds    
            b2 = np.array([interpn(grids, bz, r2, bounds_error=False)[0],
                           interpn(grids, by, r2, bounds_error=False)[0], 
                           interpn(grids, bx, r2, bounds_error=False)[0]])
            c2guess = b2/np.linalg.norm(b2)
            converge = np.linalg.norm(c2-c2guess) <= 0.001
            itn = itn +1
        npoint=npoint+1    
 #       print('npoint=', npoint, ', iternumber=', itn)
        inside = (r2[2]-xmin)*(r2[2]-xmax) < 0. \
            and  (r2[1]-ymin)*(r2[1]-ymax) < 0.  \
            and  (r2[0]-zmin)*(r2[0]-zmax) < 0. 
        if inside:
            r1=np.copy(r2)
            c1=np.copy(c2)
            b1=np.copy(b2)
            r=np.append(r, [r1], axis=0)
            b=np.append(b, [b1], axis=0)
    return r, b
def bext(angle=20., x1=220, y1=200, nz=2, draw=True, alpha=0.):
 #   filename="C:\\work\\Data\\hmi.m_45s.2013.07.17_19_*_TAI.magnetogram.fits"
    files=glob.glob("C:\\work\\Data\\hmi.m_45s.2013.07.17_19_*_TAI.magnetogram.fits")  
 #   filev="C:\\work\Data\\hmi.B_720s.20130717_193600_TAI.normal_field.fits"
    if 0:
        bz1=0.
        for filename in files:
            data=np.nan_to_num(np.copy(fits.getdata(filename)), -1.e5)
            data = rotate(data, 180.)
            bz1=bz1+data[1050:1050+800,2280:2280+800]
        bz1 = bz1/len(files)
        with open('data.pickle', 'wb') as f:
            pickle.dump(bz1, f)
    else: 
        with open('data.pickle', 'rb') as f:
            bz1=pickle.load(f)
    bz0 = (rotate(bz1, angle, reshape=False))[400-256:400+256, 400-256:400+256]
    

    z= np.arange(nz, dtype=float) #np.arange(10, dtype=float)
    colors = plt.cm.viridis(np.linspace(0,1,nz))
    #z=np.array([0.,  0.5, 1.0, 5., 20])
    bx,by,bz = b_lff(bz0-bz0.mean(),z, alpha=alpha)
   
    
 
    nx,ny=104,95
    x2 = x1+nx
    y2 = y1+ny
    extent = np.array([x1-0.5, x2-0.5, y1-0.5, y2-0.5])
    if draw:  
        plt.figure(figsize=[8,8])
        plt.imshow(bz0[y1:y2,x1:x2], extent=extent, origin='lower')
        plt.clim(-200,200)
       
        plt.contour(bz0[y1:y2,x1:x2], [-600,-200,-20, 20, 200,600,1100], \
                    colors=['b','b','b', 'r', 'r', 'r', 'r'],extent=extent, \
                        origin='lower', linewidths=1.5)
    if 0:    
        x = np.arange(-28, 32, 4)+358-x1
        y = np.arange(-28, 32, 4)+270-y1
        xs, ys = np.meshgrid(x, y)  
        xs = np.ravel(xs)
        ys = np.ravel(ys)
        bzp = bz[0,ys,xs]
        ss=np.argsort(abs(bzp))
        xs=xs[ss]
        ys=ys[ss]
        for i in range(len(xs)):
                r1, b1 = b_line(bx[:,y1:y2,x1:x2],by[:,y1:y2,x1:x2],
                                bz[:,y1:y2,x1:x2],z, np.array([0.,ys[i], xs[i]]) )
                ls = 5
                r1[:,-1]=r1[:,-1]+x1
                r1[:,-2]=r1[:,-2]+y1
                for s in range(r1.shape[-2]//ls):   
                    zz=round(r1[ls*s+ls//2,-3])
                    plt.plot(r1[ls*s:ls*(s+1)+1,-1], r1[ls*s:ls*(s+1)+1, -2], \
                             color=colors[zz],linewidth=2)
    
    
   
    return bx[:,y1:y2,x1:x2],by[:,y1:y2,x1:x2], bz[:,y1:y2,x1:x2],z
##
