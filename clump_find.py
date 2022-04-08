# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:47:38 2020

@author: chokh
"""

import numpy as np

def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):

        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian
#%%
def clump_find(data, level=0, min_area=4, dist_sq_crit=2):
    sz = data.shape
    n_map = np.zeros(sz, dtype=int)
    # select = data >= level
    
    # Second derivative along gradient & perpendicular direction -> find convex
    grady = np.gradient(data, axis=0)
    gradx = np.gradient(data, axis=1)
    hess = hessian(data)
    grad_orthox = -grady
    grad_orthoy = gradx
    
    convex_grad = (    hess[1, 1]*gradx**2 \
                   + 2*hess[0, 1]*gradx*grady \
                   +   hess[0, 0]*grady**2)*0.5
    convex_ortho = (   hess[1, 1]*grad_orthox**2 \
                   + 2*hess[0, 1]*grad_orthox*grad_orthoy \
                   +   hess[0, 0]*grad_orthoy**2)*0.5                  
    select = (convex_grad <= 0) & (convex_ortho <= 0) & (data > level)

    xpos, ypos = np.where(select)
    no = np.zeros(np.sum(select), dtype=int)
    no[0] = 1
    count = 2
    dist_sq_arr =    (xpos[:, None]-xpos[None, :])**2  \
                   + (ypos[:, None]-ypos[None, :])**2

    for i in range(1, np.sum(select)):
       # dist_sq = (xpos[0:i]-xpos[i])**2 + (ypos[0:i]-ypos[i])**2
       dist_sq = dist_sq_arr[i, 0:i]
       if np.any(dist_sq <= dist_sq_crit):
           group = dist_sq <= dist_sq_crit
           assign = min(no[0:i][group])
           for j in np.unique(no[0:i][group]):
               no[no == j] = assign
           no[i] = assign
       else:
           no[i] = count
           count += 1
    value, counts = np.unique(no, return_counts=True)
    real_value = value[counts >= min_area]
    # hist, bins = np.histogram(no, bins=range(min(no), max(no)+2, 1))
    # real_value = bins[0:-1][hist >= min_area]
    for i in range(1, len(real_value)+1):
        n_map[xpos[no == real_value[i-1]], ypos[no == real_value[i-1]]] = i
    return n_map    
#%%        
def clump_find3d(data, level=0, min_area=4, dist_crit=np.sqrt(2)):
    sz = data.shape               # 3d array
    n_data = np.zeros(sz, dtype=int)
    binary_data = np.zeros(sz, dtype=int)
    # Second derivative along gradient & perpendicular direction -> find convex
    count = 2
    for k in range(sz[1]):
        img = data[k]
        grady = np.gradient(img, axis=0)
        gradx = np.gradient(img, axis=1)
        hess = hessian(img)
        grad_orthox = -grady
        grad_orthoy = gradx
        
        convex_grad = (    hess[1, 1]*gradx**2 \
                       + 2*hess[0, 1]*gradx*grady \
                       +   hess[0, 0]*grady**2)*0.5
        convex_ortho = (   hess[1, 1]*grad_orthox**2 \
                       + 2*hess[0, 1]*grad_orthox*grad_orthoy \
                       +   hess[0, 0]*grad_orthoy**2)*0.5                  
        select = (convex_grad <= 0) & (convex_ortho <= 0) & (img > level)
        xpos, ypos = np.where(select)
        dist_sq_arr =    (xpos[:, None]-xpos[None, :])**2  \
                       + (ypos[:, None]-ypos[None, :])**2
                       
        temp_map = np.zeros(img.shape)
        no = np.arange(len(xpos)) + 1
        for ii in range(0, len(xpos)-2):   # find clumpy structure in 2D map
            same_clump = (dist_sq_arr[ii] <= dist_crit**2)
            same_clump[ii:] = False
            if np.sum(same_clump) > 0:
                assign = min(no[same_clump])
                for j in np.unique(no[same_clump]):
                    no[no == j] = assign
                no[ii] = assign
        value, counts = np.unique(no, return_counts=True)
        real_values = value[counts >= min_area]
        for ii, real_value in enumerate(real_values):
            temp_map[xpos[no == real_value], ypos[no == real_value]] = ii+1
        
            
            
        for i in range(1, np.sum(select)):
           dist_sq = (xpos[0:i]-xpos[i])**2 + (ypos[0:i]-ypos[i])**2
           if np.any(dist_sq <= dist_crit**2):
               group = dist_sq <= dist_sq_crit
               assign = min(no[0:i][group])
               for j in np.unique(no[0:i][group]):
                   no[no == j] = assign
               no[i] = assign
           else:
               no[i] = count
               count += 1
    value, counts = np.unique(no, return_counts=True)
    real_value = value[counts >= min_area]
    # hist, bins = np.histogram(no, bins=range(min(no), max(no)+2, 1))
    # real_value = bins[0:-1][hist >= min_area]
    for i in range(1, len(real_value)+1):
        n_map[xpos[no == real_value[i-1]], ypos[no == real_value[i-1]]] = i
    return n_map    