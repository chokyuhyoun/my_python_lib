# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:12:13 2020

@author: chokh
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.fftpack import fftn, fftfreq, fftshift, ifftn, ifftshift, fft, ifft
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Ring2DKernel
import sys, time#, keyboard
import fisspy.image.base as fp
from interpolation.splines import LinearSpline

def alignoffset(image0, template0, cor= None):
    
    st=template0.shape
    si=image0.shape
    ndim=image0.ndim

    if ndim==1:
        if not st[-1]==si[-1]:
            raise ValueError('Image and template are incompatible')
        if not ('float' in str(image0.dtype) and 'float' in str(template0.dtype)):
            image0=image0.astype(float)
            template0=template0.astype(float)
    
        nx=st[-1]
        template=template0.copy()
        image=image0.copy()
    
        image-=image.mean()
        template-=template.mean()
    
        sigx=nx/6.
        gx=np.arange(-nx/2,nx/2,1)
        gauss=np.exp(-0.5*((gx/sigx)**2))**0.5

        corr=ifft(ifft(template*gauss)*fft(image*gauss)).real
   
        s=np.argmax(corr)
        x0=s-nx*(s>nx/2)
        x1=0.5*(corr[s-1]-corr[s+1-nx])/(corr[s+1-nx]+corr[s-1]-2.*corr[s])
        x=x0+x1

        if cor:
            img = shift(image, -x)
            xx = np.arange(nx) + x
            kx = np.logical_and(xx >= 0, xx <= nx - 1)
            cor = (img*template)[kx].sum()/np.sqrt((img[kx]**2).sum() *
                          (template[kx]**2).sum())
            return x, cor
        else:
            return x
                             
    else:
        return fp.alignoffset(image0, template0, cor)

def shift(image, sh):
    if image.ndim == 1:
        x = np.arange(image)
        xp = x - sh
        int1d = interp1d(x, image, kind='cubic')
        return int1d(xp)
    else:
        return fp.shift(image, sh)


def interpol2d(image0, xxp0, yyp0, *args, missing=0):
    shape = np.array(image0.shape)
    if len(args) == 0:
        xxp = xxp0
        yyp = yyp0
        smin = [0, 0]
        smax = shape - [1, 1]
    else: 
        smin = np.array([yyp0[0, 0], xxp0[0, 0]])
        smax = np.array([yyp0[-1, -1], xxp0[-1, -1]])
        xxp = args[0]
        yyp = args[1]
        
    order = shape
    interp = LinearSpline(smin, smax, order, image0)
    a = np.array((yyp.flatten(), xxp.flatten()))
    b = interp(a.T)
    res = b.reshape(xxp.shape)
    if missing is not False:
        mask = ((xxp > smax[1]) | (yyp > smax[0]) | \
                (xxp < smin[1]) | (yyp < smin[0]))
        res[mask]=missing
    # import pdb; pdb.set_trace()
    return res

def img_rot_align(data, ref, data_angle=0., ref_angle=0., 
                  shift=False, cor=False, missing=0, 
                  xr=False, yr=False): # angle in degree

    if ref_angle != 0:
        n_ref = fp.rot(ref, ref_angle)
    else:
        n_ref = ref        
    shape = np.array(data.shape)
    if xr == False:
        xr = [0, shape[1]]
    if yr == False:
        yr = [0, shape[0]]
    n_ref1 = n_ref[yr[0]:yr[1], xr[0]:xr[1]]
    smin = [0, 0]
    smax = shape - [1, 1]
    order = shape
    interp = LinearSpline(smin, smax, order, data)
    yc, xc = (shape-1)*0.5
    y, x = np.mgrid[yr[0]:yr[1], xr[0]:xr[1]]

    delyx = np.array([0., 0.])
    for i in range(2):
        xt, yt = fp.rot_trans(x+delyx[1], y+delyx[0] ,xc, yc, 
                              data_angle)
        pt = np.array((yt.reshape(yt.size), xt.reshape(xt.size)))
        n_data = interp(pt.T)
        n_data1 = n_data.reshape(x.shape)
        delyx2, corr = fp.alignoffset(n_data1, n_ref1, cor=True)
        delyx += delyx2.flatten()
        
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    xt, yt = fp.rot_trans(x+delyx[1], y+delyx[0], xc, yc, data_angle)
    pt = np.array((yt.reshape(yt.size), xt.reshape(xt.size)))
    aligned_data1 = interp(pt.T)
    aligned_data = aligned_data1.reshape(x.shape)
#    import pdb; pdb.set_trace()
    if missing != -1:
        inside = (xt <= np.max(x)) & (xt >= np.min(x)) & \
                 (yt <= np.max(y)) & (yt >= np.min(y))
        aligned_data[~inside]=missing
    if shift:
        if corr:
            return aligned_data, delyx , corr 
        else:
            return aligned_data, delyx 
    else:        
        return aligned_data
    
def cr_mem(image, size=2.0, itmax=30, display=False, reference=None):
    size = max(size, 2.)
    size2 = size/2
    sz = image.shape
    
    y = np.clip(image, 0.01, None)
    mm = np.median(image)
    
    if reference:
        m = reference
    else:
        m = np.exp(ringfilter(np.log(y), size, size+1))
    y1 = np.log(np.clip(abs(y-m), m*0.1, None))
    xd = derivatives(y1, second=True)
    yd = np.transpose(derivatives(np.transpose(y1), second=True))
    
    fwhm = size
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    tmp = (np.sqrt(xd**2+yd**2) > (3/sigma**2))*1.
    # xd, yd, y1 = 0, 0, 0
    ss = ringfilter(tmp.astype('float'), 0, size2)
    ss[:, [0, sz[1]-1]] = 1.
    ss[[0, sz[0]-1], :] = 1.
    # import pdb; pdb.set_trace()
    x = image.astype('float')
    l1 = ss > 0.
    l2 = ss <= 0.
    if np.sum(l1) == 0:
        return x
    if np.sum(l1) >= 0:
        x[l1] = m[l1]
    goodness = np.sum(l1)/(np.sum(l1)+np.sum(l2))*100
    
    if display:
        fig, ax = plt.subplots()
        im01 = ax.imshow(np.log(y/mm), vmin=0.25, vmax=4., origin='lower')
        if np.sum(l1) >= 1:
            # yp, xp = np.mgrid[0:sz[0], 0:sz[1]]
            # ax.plot(xp[l1], yp[l1], '+', color='gray')
            pos = np.argwhere(l1)
            ax.plot(pos[:, 1], pos[:, 0], '+', color='gray')
    if itmax == 0:
        return x
    
    k = 0
    m1, m2 = 0, 1
    if (itmax > 0) & (np.sum(l1) >= 1):
        while ((max(abs(m2), abs(m1)) <= 0.001) | (k == itmax)):
            x0 = x*1.
            k += 1
            m = ringfilter(x, size2, size2+1)
            x[l1] = m[l1]
            m1 = np.nanmin((x-x0)/x0)
            m2 = np.nanmax((x-x0)/x0)
            if display:
                print('k=', k)
                print(m2, m1)
                ax.imshow(np.log(x/mm), vmin=0.25, vmax=4., origin='lower')
    return x
            
   
def ringfilter(image, r1, r2):
    # s = np.round(2*r2)
    # if s < (2*r2):
    #     s += 1
    # if (s % 2) == 0:
    #     s += 1
    # xx, yy = np.mgrid[0:s, 0:s]-s//2
    # r = np.sqrt(xx**2+yy**2)
    # ker = ((r >= r1) & (r <= r2))*1.
    # ker = ker/np.sum(ker)
    # return scipy.signal.convolve2d(image.astype('float'), ker, mode='same', 
                                   # boundary='fill')
    ker = Ring2DKernel(r1, r2-r1)
    return convolve(image.astype('float'), ker, boundary='extend') 
    
def derivatives(image, second=False):
    s = image.shape
    u = np.zeros(image.shape)
    second_der = np.zeros(image.shape)
    for i in range(1, s[1]-2):
        sigma = 0.5*second_der[:, i-1]+2
        second_der[:, i] = -0.5/sigma
        u[:, i] = image[:, i+1]-2.*image[:, i]+image[:, i-1]
        u[:, i] = (3*u[:, i]-0.5*u[:, i-1])/sigma
    
    for k in range(s[1]-2, 0, -1):
        second_der[:, k] = second_der[:, k]*second_der[:, k+1]+u[:, k]
        
    if second:
        return second_der
    
    first_der = (np.roll(image, -1, axis=1)-image.astype(float))  \
                -second_der/3-np.roll(second_der, -1, axis=1)/6
    first_der[:, s[1]-1] = image[:, s[1]-1].astype(float)-image[:, s[1]-2]  \
                          -second_der[:, s[1]-1]/6
    return first_der                               

def filtering(data, dt=1, i_min=2., f_min=8., power=False):
    data[np.isnan(data)] = 0
    freq_high = 1./60./i_min
    freq_low = 1./60./f_min
    freq_t0 = fftshift(fftfreq(data.shape[0], dt))
    freq_t = freq_t0[:, None, None]
    fil_band = (abs(freq_t) > freq_low) & (abs(freq_t) < freq_high)
    fft_res0 = fftshift(fftn(data, axes=0), axes=0)
    fil_fft_res = fft_res0*fil_band
    if not power :
        return ifftn(ifftshift(fil_fft_res, axes=0), axes=0).real
    else :
        return abs(fil_fft_res)**2, freq_t0
    
    
