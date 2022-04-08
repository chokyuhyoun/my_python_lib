# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:11:21 2020

@author: chokh
"""

import matplotlib.pyplot as plt
import numpy as np
# from astropy.convolution import convolve
from scipy.signal import convolve2d, convolve
from scipy.stats.stats import pearsonr

# ;   NAME: NAVE_POINT  =  Non-linear Affine Velocity Estimator
# ;                       at the  selected spatial points
# ;
# ;   PURPOSE:  Determine 6 parameters defining the affine velocity field
# ;             at the selected points
# ;
# ;   CALLING SEQUENCE:
# ;
# ;            Result = Nave_Point (fim, sim, x, y, fwhm)
# ;   INPUTS:
# ;            fim        Image at t1
# ;            sim        Image at t2;
# ;            x, y       arrays of x and y values of positions
# ;            fwhm       fwhms of the window function (can be arbitrary positive number )
# ;                          fwhmx, fwhmy
# ;
# ;
# ;
# ;   KEYWORD INPUTS:
# ;            adv        if set,  the advection equation is solved.
# ;            conti      if set, the continuity equation is solved.
# ;            noise      standard deviation of the noise in the difference between two
# ;                       images
# ;
# ;
# ;   OUTPUTS:
# ;            Result           parameters at  all the pixels of the images
# ;              Result(0,*) : x-component of velocity (U_0)
# ;              Result(1,*) : y-component of velocity (V_0)
# ;              Result(2,*) : x-derivative of x-component (U_x)
# ;              Result(3,*) : y-derivative of y-component (V_y)
# ;              Result(4,*) : y-derivative of x-component (U_y)
# ;              Result(5,*) : x-derivative of y-component (V_x)
# ;              Result(6,*) :  mu  (optional)
# ;              Result(7,*) :  cnst (optional)
# ;
# ;              It is assumed that the velocity field around (x_0, y_0) is of the form
# ;
# ;                              vx = U_0 + U_x * (x-x_0) + U_y * (y-y_0)
# ;                              vy = V_0 + V_x * (x-x_0) + V_y * (y-y_0)
# ;  KEYWORD OUTPUTS
# ;           Sigma         uncertainties in the parameters
# ;
# ;   REMARKS:
# ;
# ;        -.  The time unit is the time interval between two successive images,
# ;            and the length unit is the pixel size.
# ;
# ;
# ;   HISTORY
# ;
# ;              2007 Jana: J. Chae,  first coded.
# ;              2007 July:
# ;              2008 October:    Refined the definition of window and FWHM
# ;              2008 Decemeber:  positions are allowed to be off the grids
# ;                               (x- and y-values may be non-integers).
# ;              2009 February,  made more efficient
# ;              2020 June:       Written in Python by K. Cho

def nave_point(fim0, sim0, x, y, fwhm, itermax=5, adv=False, 
               source=False, noise=1, np_deriv=3, background=False,
               corr=False):
    #x, y --> image coordinate    
    fim = np.asfarray(fim0)
    sim = np.asfarray(sim0)
    
    qsw = 1 if (source == True) else 0
    if adv == True:
        psw, qsw = 0, 0
    else:
        psw = 1
    s = fim.shape
    ny, nx = s
            
    # Constructing derivatives
    if np_deriv == 3:
        kernel = np.array([-1., 0., 1.])*0.5  # 3 points diff.
    if np_deriv == 5:
        kernel = np.array([0.12019, -0.74038, 0., 0.74038, -0.12019]) # 5 points diff.
    
    fim_x = -convolve(fim, kernel[None, :])
    fim_y = -convolve(fim, kernel[:, None])
    sim_x = -convolve(sim, kernel[None, :])
    sim_y = -convolve(sim, kernel[:, None])
    npoint = np.size(x)
    
    npar = 8 if (background == True) else (6+qsw)
    par = np.zeros([npoint, npar])  # U_0, V_0, U_x, V_y, U_y, V_x and optionally mu
    
    deta = np.zeros(npoint)
    gres = np.zeros(npoint)
    chisq = np.zeros(npoint)
    cor = np.zeros(npoint)
    sigma = np.zeros([npoint, npar])
    
    w = loc_win(fwhm)
    w1 = w/noise**2
    
    nys, nxs = w.shape
    ys, xs = np.mgrid[0:nys, 0:nxs]
    ys = ys-nys//2
    xs = xs-nxs//2
    
    A = np.zeros([npar, npar])
    B = np.zeros(npar)
    d = np.zeros([npar, nys, nxs])
    
    for ip in range(npoint):
        x0 = x[ip] if np.size(x) > 1 else x
        y0 = y[ip] if np.size(y) > 1 else y
        for iter in range(itermax):
            delxh = 0.5*(par[ip, 0]+par[ip, 2]*xs+par[ip, 4]*ys)
            delyh = 0.5*(par[ip, 1]+par[ip, 5]*xs+par[ip, 3]*ys)
            for sgn in [-1, 1]:
                xx = x0 + xs + sgn*delxh
                yy = y0 + ys + sgn*delyh
                sx = xx.astype(int)
                ex = xx - sx
                sy = yy.astype(int)
                ey = yy - sy
                w00 = (1.-ex)*(1.-ey)
                w10 = ex*(1.-ey)
                w01 = (1.-ex)*ey
                w11 = ex*ey
                if sgn == -1:
                    Fv = fim[sy,   sx]*w00 + fim[sy,   sx+1]*w10 \
                        +fim[sy+1, sx]*w01 + fim[sy+1, sx+1]*w11
                    Fxv = fim_x[sy,   sx]*w00 + fim_x[sy,   sx+1]*w10 \
                         +fim_x[sy+1, sx]*w01 + fim_x[sy+1, sx+1]*w11
                    Fyv = fim_y[sy,   sx]*w00 + fim_y[sy,   sx+1]*w10 \
                         +fim_y[sy+1, sx]*w01 + fim_y[sy+1, sx+1]*w11
                elif sgn == 1:
                    Sv = sim[sy,   sx]*w00 + sim[sy,  sx+1]*w10 \
                        +sim[sy+1, sx]*w01 + sim[sy+1,sx+1]*w11
                    Sxv = sim_x[sy,   sx]*w00 + sim_x[sy,   sx+1]*w10 \
                         +sim_x[sy+1, sx]*w01 + sim_x[sy+1, sx+1]*w11
                    Syv = sim_y[sy,   sx]*w00 + sim_y[sy,   sx+1]*w10 \
                         +sim_y[sy+1, sx]*w01 + sim_y[sy+1, sx+1]*w11
            nu = -(par[ip, 2]+par[ip, 3])*psw
            if qsw:
                nu = nu-qsw*par[ip, 6]
            sdiv = np.exp(-nu*0.5)
            fdiv = 1./sdiv
            gv = Sv*sdiv-Fv*fdiv
            if npar == 8:
                gv = gv+par[ip, 7]
            if iter != itermax:
                tx =  (Sxv*sdiv + Fxv*fdiv)*0.5
                t0 = -(Sv *sdiv + Fv *fdiv)*0.5
                ty =  (Syv*sdiv + Fyv*fdiv)*0.5
                d[0, :, :] = tx                 # U0
                d[1, :, :] = ty                 # V0
                d[2, :, :] = tx*xs - psw*t0     # Ux
                d[3, :, :] = ty*ys - psw*t0     # Vy
                d[4, :, :] = tx*ys              # Uy
                d[5, :, :] = ty*xs              # Vx
                if qsw:
                    d[6, :, :] = -qsw*t0
                if npar == 8:
                    d[7, :, :] = 1.
                for i in range(npar):
                    for j in range(i+1):
                        A[i, j] = sum(sum(d[j, :, :]*d[i, :, :]*w1))
                        A[j, i] = A[i, j]
                for i in range(npar):
                    B[i] = -sum(sum(gv*d[i, :, :]*w1))
                # ww, *_ = np.linalg.svd(A)
                pp, *_ = np.linalg.lstsq(A, B, rcond=None)
                par[ip, :] = pp + par[ip, :]
            else:
                # deta[ip] = min(np.log10(ww))
                gres[ip] = gv[nys*0.5, nxs*0.5]
                chisq[ip] = sum(sum(gv**2*w1))
                sigma[ip, :] = np.sqrt(np.diag(np.linalg.inv(A)))
                tmp = Sv*sdiv
                cor[ip] = pearsonr(tmp, Fv*fdiv)
    if corr == True:
        return par.astype(float), cor
    else:
        return par.astype(float)
            
# ; NAME: NAVE_TRACK
# ;
# ; PURPOSE:
# ;         Determine the position of a Lagrangian trajectory at t2
# ;         that was at a specified position at t1
# ;
# ; CALLING SEQUENCE:
# ;          pos2=Nave_track(im1, im2, fwhm, pos1, sigma=sigma)
# ; INPUTS:
# ;         im1     Image at t1
# ;         im2     Image at t2
# ;         fwhm    fwhm of the window to be used in NAVE
# ;         pos1    Position(s) at t1
# ;                      pos1[0,*] =x
# ;                      pos1[1,*] =y
# ; OUTPUTS:
# ;         pos2    Positon(s) at t2
# ; Keyword Output
# ;         cor     correlation
# ; REQUIRED ROUTINES:
# ;         NAVE_POINT
# ; HISTORY:
# ;          2008 April: J. Chae first coded.
# ;          2008 November
# ;          2008 December
# ;          2020 June:     Written in Python by K. Cho

def nave_track(im1, im2, fwhm, xp0, yp0, noise=1., itmax=2):
    #xp0, yp0 --> image coordinate
    shape = xp0.shape
    xp1 = np.asfarray(xp0.flatten())
    yp1 = np.asfarray(yp0.flatten())
    dx, dy = 0., 0.
    for k in range(1, itmax):
        par = nave_point(im1, im2, xp1+0.5*dx, yp1+0.5*dy, fwhm, 
                         noise=noise)
        dx = par[:, 0]
        dy = par[:, 1]
    xp2 = xp1*1.
    yp2 = yp1*1.
    xp2 = xp1 + dx
    yp2 = yp1 + dy
    return np.reshape(xp2, shape), np.reshape(yp2, shape)
            
def loc_win(win_par, profile='Gaussian'):
    if np.size(win_par) >= 2:
        fwhmx, fwhmy = win_par[0:2]
        theta_deg = win_par[2] if (np.size(win_par) >= 3) else 0.
    else:
        fwhmx, fwhmy = win_par*np.array([1, 1])
        theta_deg = 0.
    
    theta = theta_deg*np.pi/180.
    hwhm = np.array([fwhmx, fwhmy])*0.5
    
    hh = round(max(hwhm)+1e-5)
    mf = 1 if (profile == 'top-hat') else 2
    nxs = 2.*hh*mf+1.
    nys = 2.*hh*mf+1.
    ys, xs = np.mgrid[0:nys, 0:nxs] - nxs//2
    
    r2 = ((xs*np.cos(theta) + ys*np.sin(theta))/hwhm[0])**2 + \
        ((-xs*np.sin(theta) + ys*np.cos(theta))/hwhm[1])**2
    
    if profile == 'top-hat':
        w = r2*(r2 <= 1.)
    elif profile == 'Gaussian':
        w = np.exp(-np.log(2.)*r2)
    elif profile == 'hanning':
        w = (1.+np.cos(np.pi*0.5*(np.sqrt(r2).clip(min=2.))))*0.5
    return w
    
               
    