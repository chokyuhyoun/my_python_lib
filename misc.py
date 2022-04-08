# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:27:14 2020

@author: chokh
"""
from glob import glob
from mpl_toolkits.mplot3d import Axes3D, proj3d
from scipy import interpolate
from interpolation.splines import LinearSpline
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import subprocess
from mylib import misc

#%%
class show3d(object):
        
    def __init__(self, vol, cont=None, lims=None, display=True, 
                 vmin=None, vmax=None, levels=None, cont_kwargs=None,
                 xlabel=None, ylabel=None, zlabel=None, 
                 xarray=None, yarray=None, zarray=None):
        self.vol = vol
        self.shape = np.array(self.vol.shape)
        self.cont = cont
        self.vmin = max(np.nanmean(self.vol)-3*np.nanstd(self.vol), 
                        np.nanmin(self.vol)) \
            if vmin == None else vmin
        self.vmax = max(np.nanmean(self.vol)+3*np.nanstd(self.vol), 
                        np.nanmax(self.vol)) \
            if vmax == None else vmax
        self.levels = () if levels == None else (levels, )
        self.xarray = np.arange(self.shape[2]) if np.all(xarray) == None else xarray
        self.yarray = np.arange(self.shape[1]) if np.all(yarray) == None else yarray
        self.zarray = np.arange(self.shape[0]) if np.all(zarray) == None else zarray

        xextent = np.array([-0.5, self.shape[2]-0.5]) if np.all(xarray) == None \
                  else misc.minmax(self.xarray) + np.array([-0.5, 0.5])*(self.xarray[1]-self.xarray[0])
        yextent = np.array([-0.5, self.shape[1]-0.5]) if np.all(yarray) == None \
                  else misc.minmax(self.yarray) + np.array([-0.5, 0.5])*(self.yarray[1]-self.yarray[0])
        zextent = np.array([-0.5, self.shape[0]-0.5]) if np.all(zarray) == None \
                  else misc.minmax(self.zarray) + np.array([-0.5, 0.5])*(self.zarray[1]-self.zarray[0])
        self.extents = [xextent, yextent, zextent]
        self.lims = [xextent, yextent, zextent] if np.all(lims) == None \
                  else lims
        
        self.cont_kwargs = dict(colors='black', linewidths=1, #linestyles='solid', 
                                alpha=0.5) \
            if cont_kwargs == None else cont_kwargs
        self.cont_sz = np.array(self.cont.shape) if np.all(self.cont) != None else 0
        self.cont_exist = False
            
        self.zpos, self.ypos, self.xpos = self.shape//2
        self.zpos = 0   
        self.value = self.vol[self.zpos, self.ypos, self.xpos]         
        self.xvalue = self.xarray[self.xpos]
        self.yvalue = self.yarray[self.ypos]
        self.zvalue = self.zarray[self.zpos]
        
        self.xlabel = 'X (pix)' if xlabel == None else xlabel
        self.ylabel = 'Y (pix)' if ylabel == None else ylabel
        self.zlabel = 'Z (pix)' if zlabel == None else zlabel
        self.volume_explorer()
    
    def volume_explorer(self):
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
            plt.rcParams['keymap.home'].remove('home')
        except:
            pass

        self.fig = plt.figure(figsize=[9.5, 9.5])
        self.fig.canvas.set_window_title(r'show3d')
        self.ax_xy = self.fig.add_axes([0.1, 0.1, 0.35, 0.35], autoscale_on=False, 
                                         xlabel=self.xlabel, ylabel=self.ylabel, 
                                         xlim=self.lims[0], ylim=self.lims[1]) 
        self.ax_xz = self.fig.add_axes([0.1, 0.55, 0.35, 0.35], autoscale_on=False, 
                                         xlabel=self.xlabel, ylabel=self.zlabel, 
                                         xlim=self.lims[0], ylim=self.lims[2])
        self.ax_zy = self.fig.add_axes([0.62, 0.1, 0.35, 0.35], autoscale_on=False, 
                                         xlabel=self.zlabel, ylabel=self.ylabel, 
                                         xlim=self.lims[2], ylim=self.lims[1])
        self.ax_xy.set_title('('+str(self.xpos)+', '+str(self.ypos)+', '
                                      +str(self.zpos)+') = '
                                      +str(format(self.value, "10.2e")))        
        self.ax_xz.set_title('Y = '+str(self.ypos))
        self.ax_zy.set_title('X = '+str(self.xpos))
        
        self.img_xy = self.ax_xy.imshow(self.vol[self.zpos, :, :], 
                                          origin='lower', interpolation='none', 
                                          aspect='auto', vmin=self.vmin, vmax=self.vmax)
        self.img_xz = self.ax_xz.imshow(self.vol[:, self.ypos, :], 
                                          origin='lower', interpolation='none', 
                                          aspect='auto')
        self.img_zy = self.ax_zy.imshow(self.vol[:, :, self.xpos].T, 
                                          origin='lower', interpolation='none', 
                                          aspect='auto')
        self.img_xy.set_extent((self.extents[0][0], self.extents[0][1], \
                                self.extents[1][0], self.extents[1][1]))
        self.img_xz.set_extent((self.extents[0][0], self.extents[0][1], \
                                self.extents[2][0], self.extents[2][1]))
        self.img_zy.set_extent((self.extents[2][0], self.extents[2][1], \
                                self.extents[1][0], self.extents[1][1]))
        if np.array_equal(self.shape, self.cont_sz):
            if self.levels == None:
                self.levels = np.median(self.cont)
            self.draw_cont(*self.levels, **self.cont_kwargs)
            self.cont_exist = True
            
        self.img_xy.hor = self.ax_xy.plot(self.ax_xy.get_xlim(), 
                                            self.ypos*np.array([1, 1]), 
                                            '--', color='gray')
        self.img_xy.vert = self.ax_xy.plot(self.xpos*np.array([1, 1]), 
                                            self.ax_xy.get_ylim(), 
                                            '--', color='gray')
        self.img_xz.hor = self.ax_xz.plot(self.ax_xy.get_xlim(), 
                                            self.zpos*np.array([1, 1]), 
                                            '--', color='gray')
        self.img_xz.vert = self.ax_xz.plot(self.xpos*np.array([1, 1]), 
                                                self.ax_xz.get_ylim(), 
                                                '--', color='gray')
        self.img_zy.hor = self.ax_zy.plot(self.ax_xz.get_ylim(), 
                                            self.ypos*np.array([1, 1]),
                                            '--', color='gray')
        self.img_zy.vert = self.ax_zy.plot(self.zpos*np.array([1, 1]), 
                                             self.ax_xy.get_ylim(), 
                                             '--', color='gray')
        self.cax = self.fig.add_axes([0.45, 0.1, 0.02, 0.35])
        self.colorbar = plt.colorbar(self.img_xy, cax=self.cax)
        self.fig.text(0.55, 0.9, r'$\leftarrow$, $\rightarrow$ : $\pm$ X')
        self.fig.text(0.75, 0.9, r'$\uparrow$, $\downarrow$ : $\pm$ Y')
        self.fig.text(0.55, 0.87, r'PgUp, PgDn : $\pm$ Z')
        self.fig.text(0.75, 0.87, r'Home, End : Last, First Z')
        self.fig.text(0.55, 0.84, r'Ctrl : ${\times}$ 5')
        self.fig.text(0.75, 0.84, r'Shift : $\times$ 10')
        self.fig.text(0.55, 0.81, r'+ : Full extent')
        self.fig.text(0.75, 0.81, r'Q : Quit')
        self.fig.text(0.55, 0.78, r'0, 5, 9 : First corner, Center, Last corner' )
        self.t_xval = self.fig.text(0.55, 0.67, self.xlabel+' = '+str(format(self.xarray[self.xpos], "10.3e")))
        self.t_yval = self.fig.text(0.55, 0.64, self.ylabel+' = '+str(format(self.yarray[self.ypos], "10.3e")))
        self.t_zval = self.fig.text(0.55, 0.61, self.zlabel+' = '+str(format(self.zarray[self.zpos], "10.3e")))
        self.value = self.vol[self.zpos, self.ypos, self.xpos]
        self.t_val = self.fig.text(0.55, 0.58, 
                                      '('+str(self.xpos)+', '+str(self.ypos)+', '+str(self.zpos)
                                      +') = '+str(format(self.value, "10.3e")), 
                                      fontweight='bold')
        self.img_update()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()

    def on_key(self, event):
        co1, co2 = 1, 1
        if 'ctrl' in event.key:
            co1 = 5
        if 'shift' in event.key:
            co2 = 10
        if 'left' in event.key:
            self.xpos = (self.xpos-co1*co2+self.shape[2]) % self.shape[2]
        elif 'right' in event.key:
            self.xpos = (self.xpos+co1*co2) % self.shape[2]
        elif 'pageup' in event.key:
            self.zpos = (self.zpos+co1*co2) % self.shape[0]
        elif 'pagedown' in event.key:
            self.zpos = (self.zpos-co1*co2+self.shape[0]) % self.shape[0]
        elif 'home' in event.key:
            self.zpos = self.shape[0]-1
        elif 'end' in event.key:
            self.zpos = 0
        elif 'up' in event.key:
            self.ypos = (self.ypos+co1*co2) % self.shape[1]
        elif 'down' in event.key:
            self.ypos = (self.ypos-co1*co2+self.shape[1]) % self.shape[1]
        elif event.key == ' ' and event.inaxes == self.ax_xy:
            self.xpos = int(event.xdata)
            self.ypos = int(event.ydata)
        elif event.key == ' ' and event.inaxes == self.ax_xz:
            self.xpos = int(event.xdata)
            self.zpos = int(event.ydata)
        elif event.key == ' ' and event.inaxes == self.ax_zy:
            self.zpos = int(event.xdata)
            self.ypos = int(event.ydata)
        elif '0' in event.key:
            self.xpos, self.ypos, self.zpos = 0, 0, 0
        elif '5' in event.key:
            self.zpos, self.ypos, self.xpos = self.shape//2
        elif '9' in event.key:
            self.zpos, self.ypos, self.xpos = np.array(self.shape)-np.array([1, 1, 1])
        elif '-' in event.key:
            self.lims = self.extents
            self.ax_xy.set_xlim(self.lims[0])
            self.ax_xy.set_ylim(self.lims[1])
            self.ax_xz.set_ylim(self.lims[2])
        elif 'q' in event.key:
            del(self)
            plt.close(fig='Volume Explorer')
            return
        self.img_update()            
        

    def img_update(self):
        self.img_xy.set_data(self.vol[self.zpos, :, :])
        self.img_xz.set_data(self.vol[:, self.ypos, :])
        self.img_zy.set_data(self.vol[:, :, self.xpos].T)

        xextent = misc.minmax(self.xarray) + np.array([-0.5, 0.5])*(self.xarray[1]-self.xarray[0])
        yextent = misc.minmax(self.yarray) + np.array([-0.5, 0.5])*(self.yarray[1]-self.yarray[0])
        zextent = misc.minmax(self.zarray) + np.array([-0.5, 0.5])*(self.zarray[1]-self.zarray[0])
        self.extents = [xextent, yextent, zextent]
        
        if np.all(self.cont) != None:
            self.draw_cont(*self.levels, **self.cont_kwargs)
        self.lims = [self.ax_xy.get_xlim(), self.ax_xy.get_ylim(), self.ax_xz.get_ylim()]             
        self.img_xy.hor[0].set_data(self.lims[0], self.yarray[self.ypos]*np.array([1, 1]))
        self.img_xy.vert[0].set_data(self.xarray[self.xpos]*np.array([1, 1]), self.lims[1])
        self.img_xz.hor[0].set_data(self.lims[0], self.zarray[self.zpos]*np.array([1, 1]))
        self.img_xz.vert[0].set_data(self.xarray[self.xpos]*np.array([1, 1]), self.lims[2])
        self.img_zy.hor[0].set_data(self.lims[2], self.yarray[self.ypos]*np.array([1, 1]))
        self.img_zy.vert[0].set_data(self.zarray[self.zpos]*np.array([1, 1]), self.lims[1])
        self.value = self.vol[self.zpos, self.ypos, self.xpos] 
        xy_title1 = str(self.zpos) if self.zlabel == 'Z (pix)' \
               else str(format(self.zarray[self.zpos], "10.3e"))
        xz_title1 = str(self.ypos) if self.ylabel == 'Y (pix)' \
               else str(format(self.yarray[self.ypos], "10.3e"))               
        zy_title1 = str(self.xpos) if self.xlabel == 'X (pix)' \
               else str(format(self.xarray[self.xpos], "10.3e"))                       
        self.ax_xy.set_title(self.zlabel+' = '+xy_title1)        
        self.ax_xz.set_title(self.ylabel+' = '+xz_title1)        
        self.ax_zy.set_title(self.xlabel+' = '+zy_title1)        
        self.t_xval.set_text(self.xlabel+' = '+str(format(self.xarray[self.xpos], "10.3e")))
        self.t_yval.set_text(self.ylabel+' = '+str(format(self.yarray[self.ypos], "10.3e")))
        self.t_zval.set_text(self.zlabel+' = '+str(format(self.zarray[self.zpos], "10.3e")))
        self.t_val.set_text('('+str(self.xpos)+', '+str(self.ypos)+', '+str(self.zpos)
                            +') = '+str(format(self.value, "10.3e")))
        self.ax_xy.set_xlabel(self.xlabel)
        self.ax_xy.set_ylabel(self.ylabel)
        self.ax_xz.set_xlabel(self.xlabel)
        self.ax_xz.set_ylabel(self.zlabel)
        self.ax_zy.set_xlabel(self.zlabel)
        self.ax_zy.set_ylabel(self.ylabel)
        self.vmin, self.vmax = self.img_xy.get_clim()
        self.img_xz.set_clim(self.vmin, self.vmax)
        self.img_zy.set_clim(self.vmin, self.vmax)
        self.img_xy.set_extent((self.extents[0][0], self.extents[0][1], \
                                self.extents[1][0], self.extents[1][1]))
        self.img_xz.set_extent((self.extents[0][0], self.extents[0][1], \
                                self.extents[2][0], self.extents[2][1]))
        self.img_zy.set_extent((self.extents[2][0], self.extents[2][1], \
                                self.extents[1][0], self.extents[1][1]))
        self.ax_xz.set_xlim(self.lims[0])
        self.ax_zy.set_ylim(self.lims[1])
        self.ax_zy.set_xlim(self.lims[2])
        self.ax_xz.set_position([self.ax_xy.get_position().x0, 
                                 self.ax_xz.get_position().y0, 
                                 self.ax_xy.get_position().width, 
                                 self.ax_xz.get_position().height])
        self.ax_zy.set_position([self.ax_zy.get_position().x0, 
                                 self.ax_xy.get_position().y0, 
                                 self.ax_zy.get_position().width, 
                                 self.ax_xy.get_position().height])
        self.cax.set_position([self.cax.get_position().x0, 
                               self.ax_xy.get_position().y0, 
                               self.cax.get_position().width, 
                               self.ax_xy.get_position().height])
        self.fig.canvas.draw_idle()

    def set_clim(self, *args):
        if len(args) == 1 : clim1, clim2 = args[0][0], args[0][1]
        else : clim1, clim2 = args[0], args[1]
        self.vmin, self.vmax = clim1, clim2
        self.img_xy.set_clim(self.vmin, self.vmax)
        self.img_update()
        
    def set_xlim(self, *args):
        if len(args) == 1 : xlim1, xlim2 = args[0][0], args[0][1]
        else : xlim1, xlim2 = args[0], args[1]
        self.lims[0] = (xlim1, xlim2)
        self.ax_xy.set_xlim(self.lims[0])
        self.img_update()

    def set_ylim(self, *args):
        if len(args) == 1 : ylim1, ylim2 = args[0][0], args[0][1]
        else : ylim1, ylim2 = args[0], args[1]
        self.lims[1] = (ylim1, ylim2)
        self.ax_xy.set_ylim(self.lims[1])
        self.img_update()

    def set_zlim(self, *args):
        if len(args) == 1 : zlim1, zlim2 = args[0][0], args[0][1]
        else : zlim1, zlim2 = args[0], args[1]        
        self.lims[2] = (zlim1, zlim2)
        self.ax_xz.set_ylim(self.lims[2])
        self.img_update()
    
    def set_cmap(self, cmap):
        self.img_xy.set_cmap(cmap)    
        self.img_xz.set_cmap(cmap)    
        self.img_zy.set_cmap(cmap)    

    def draw_cont(self, *args, **kwargs):
        self.cont_off()
        self.cont_xy = self.ax_xy.contour(self.cont[self.zpos, :, :], 
                                            *args, **kwargs, 
                                            extent=(self.extents[0][0], self.extents[0][1], \
                                                    self.extents[1][0], self.extents[1][1]))
        self.cont_xz = self.ax_xz.contour(self.cont[:, self.ypos, :], 
                                            *args, **kwargs, 
                                            extent=(self.extents[0][0], self.extents[0][1], \
                                                    self.extents[2][0], self.extents[2][1]))
        self.cont_zy = self.ax_zy.contour(self.cont[:, :, self.xpos].T,
                                            *args, **kwargs, 
                                            extent=(self.extents[2][0], self.extents[2][1], \
                                                    self.extents[1][0], self.extents[1][1]))
        self.cont_exist = True
        
    def cont_off(self):
        if self.cont_exist == True:
            for dum in self.cont_xy.collections:
                dum.remove()
            for dum in self.cont_xz.collections:
                dum.remove()
            for dum in self.cont_zy.collections:
                dum.remove()
            self.cont_exist = False
    



def adj_3sigma(data, sig=3):
    vmin = np.max([np.mean(data) - sig*np.std(data), np.min(data)])
    vmax = np.min([np.mean(data) + sig*np.std(data), np.max(data)])
    return (vmin, vmax)

def find_procedure(word, directory=None):
    if directory == None:
        directory = os.getcwd()
        print(f'directory = {directory}')
    files = glob(directory+r'\**\*.py', recursive=True)
    found = []
    for file in files:
        f = open(file, 'rt', encoding='UTF8')
        lines = f.readlines()
        for i, line in enumerate(lines):
            if word in line:
                print(file+' : line '+str(i+1))
                found.append(file)
    return     

def minmax(data, delta=False):
    if delta:
        return np.max(data) - np.min(data)
    else:
        return np.array([np.nanmin(data), np.nanmax(data)])

def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad="1%")
    ax_pos= ax.get_position()
    cax = fig.add_axes([ax_pos.x1+ax_pos.width*0.02, ax_pos.y0, 
                        ax_pos.width*0.05, ax_pos.height])
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def imshow(*args, axis=False, origin='lower', **kwargs):
    fig, ax = plt.subplots()
    im = []
    for arg in args:
        im0 = ax.imshow(arg, origin=origin, **kwargs)
        im0.set_clim(adj_3sigma(arg))
        im.append(im0)
    if len(im) == 1:
        im = im0
    cb = colorbar(im0)
    # cax01 = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 
    #                       0.03, ax.get_position().height])
    # cb01 = fig.colorbar(im0, cax = cax01)
    # ax.set_aspect('equal')
    if axis:
        return im, ax
    else:
        return im

def plot(*arg, axis=False):
    fig, ax = plt.subplots()
    p = ax.plot(*arg)
    if axis:
        return p, ax
    else:
        return p
    
def grid(data):  # z, y, x order
    shape = data.shape
    if data.ndim == 2:
        return np.mgrid[0:shape[0], 0:shape[1]]
    if data.ndim == 3:
        return np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    return -1

def arr_eq(arr, val, diff=False):
    if type(val) == np.ndarray:
        match = np.zeros(val.size)
        for i in range(val.size):
            match[i] = abs(arr-val[i]).argmin()
    else:
        match = abs(arr-val).argmin()
    if diff:
        return match, val-arr[match]
    return match

class Arrow3D(mpatches.FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        mpatches.FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer, *args, **kwargs):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        mpatches.FancyArrowPatch.draw(self, renderer, *args, **kwargs)
        
def rebin(ndarray, new_shape, operation='mean'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    Number of output dimensions must match number of input dimensions.
    https://gist.github.com/derricw/95eab740e1b08b78c03f
    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray        



def ffmpeg(imglist, fpsi=20, output='video.mp4', no_buffer=False):
    """
    FFMPEG
    
    Using the ffmpeg make a video file from images.
    The output video is saved at the same location of images.
    
    Parameters
    ----------
    imglist : list
        List of image filename.
    fpsi   : int
        Integer value of Frame Per Second.
    output : str
        Output video name with extension. 
            * Default is video.mp4
            
    Returns
    -------
    video : file
        Video file.
        
    Example
    -------
    >>> import fisspy
    >>> fisspy.ffmpeg(file,10,'fiss.mp4')
    """
    FFMPEG_BIN = os.path.dirname(os.path.abspath(__file__))+r'\ffmpeg '
    
    exten=output.split('.')[-1]
    if exten == 'mp4':
        codec='-vcodec libx264'
    elif exten == 'avi':
        codec='-vcodec libxvid'
    elif exten == 'mov':
        codec='-codec mpeg4'
    else:
        ValueError('The given output extension is not supported !')
    
    n=len(imglist)
    if n == 0:
        raise ValueError('Image list has no element!')
    
    fps=str(fpsi)
    dir=os.path.dirname(imglist[0])
    if bool(dir):
        os.chdir(dir)
    else:
        os.chdir(os.getcwd())

    f=open('img_list.tmp','w')
    for i in imglist:
        f.write("file '"+os.path.basename(i)+"'\n")
    if no_buffer == False:
        for i in range(fpsi):
            f.write("file '"+os.path.basename(imglist[-1])+"'\n")
    f.close()
    
    # cmd=(FFMPEG_BIN+
    #      # ' -r '+fps+' -f concat -i img_list.tmp'+
    #      # ' -c:v '+codec+' -pix_fmt yuv420p -q:v 1 -y '+output)
    #      ' -r '+fps+' -f concat -i img_list.tmp '+codec+' -crf 18'+  
    #      ' -vf "fps='+fps+', format=yuv420p, '+ 
    #      ' scale=trunc(iw/2)*2:trunc(ih/2)*2:out_range=full"' + 
    #      ' -y '+ output + ' -nostats -loglevel 0')
    cmd = ('ffmpeg -r '+fps+' -f concat -i img_list.tmp '+codec+' -crf 18'+  
         ' -vf "fps='+fps+', format=yuv420p,'+ 
         ' scale=trunc(iw/2)*2:trunc(ih/2)*2:out_range=full"' + 
         ' -y '+ output)
    # res = subprocess.check_output(cmd, shell=True)
    res = os.system(cmd)
    os.remove('img_list.tmp')
    return res

def interpol2d(image0, xxp0, yyp0, *args, missing=-1):
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
    if missing != -1:
        mask = (xxp > np.max(xxp0)) | (yyp > np.max(yyp0)) | \
               (xxp < np.min(xxp0)) | (yyp < np.min(yyp0))
        res[mask]=missing
    return res

def SlitPos(Points, ds=1.):
    NP = (Points.shape)[0]
    t = np.arange(NP, dtype='float')
    if NP == 2:
        fx = interpolate.interp1d(t, Points[:,0], fill_value="extrapolate")
        fy = interpolate.interp1d(t, Points[:,1], fill_value="extrapolate")
    if NP == 3:
        fx = interpolate.interp1d(t, Points[:,0], kind='quadratic', fill_value="extrapolate")
        fy = interpolate.interp1d(t, Points[:,1], kind='quadratic', fill_value="extrapolate")
    if NP > 3:
        fx = interpolate.interp1d(t, Points[:,0], kind='quadratic', fill_value="extrapolate")
        fy = interpolate.interp1d(t, Points[:,1], kind='quadratic', fill_value="extrapolate")
    x0 = Points[0,0]
    y0 = Points[0,1]
    s0=0.
    x=x0
    y=y0
    s=s0
    t0 = t[0]
    dt = 0.1
    
    while t0 < t[NP-1]:
        for rep in range(2):
            t1 = t0+dt
            x1 = fx(t1)
            y1 = fy(t1)
            dxdt  =  (x1-x0)/dt
            dydt  =  (y1-y0)/dt
            dsdt = np.sqrt(dxdt**2+dydt**2)    
            dt = ds/dsdt
        x0 = x1
        y0 = y1
        t0 = t1 
        s0=s0+ds
        x = np.append(x, x0)
        y = np.append(y, y0)
        s = np.append(s, s0)
    return x, y, s

class blink(object):
    def __init__(self, *args):
        if len(args) == 1:
            args = args[0]
        self.imgs = args
        self.num = len(args)
        self.fig = args[0].get_figure()
        self.i = 0
        print('press spacebar')
        self.t01 = self.fig.text(0.05, 1-0.05, str(self.i)+'\nExit: w', 
                                 ha='left', va='top', size=15)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()
        
    def on_key(self, event):
        if event.key == ' ':
            for img in self.imgs:
                img.set_visible(False)
            self.imgs[self.i].set_visible(True)
            self.t01.set_text(str(self.i)+'\nExit: w')
            self.i = (self.i+1) % self.num
            plt.show()
        elif event.key == 'w':
            for img in self.imgs:
                img.set_visible(True)
            self.t01.remove()
            print('Finished')
            plt.show()
            
            return
        self.fig.canvas.draw_idle()


#%%
def get_click(num=4):
    fig = plt.gcf()
    global coords
    coords = []
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print(f'x = {ix:.5}, y = {iy:.5}')
    
        # global coords
        coords.append((ix, iy))
    
        if len(coords) >= num:
            fig.canvas.mpl_disconnect(cid)
        return coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    res = np.array(coords)
    return res
