# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:35:10 2020

@author: chokh
"""

import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import rc

class show3d(object):
        
    def __init__(self, vol):
        self.vol=vol
    
    def volume_explorer(self, xlim=None, ylim=None, zlim=None, vmin=None, vmax=None):
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
            # rc('text', usetex=Tru)
        except:
            pass

        self.fig = plt.figure(figsize=[9.5, 9.5])    

        self.sz = np.array(self.vol.shape)
        if not xlim:
            self.xlim = (-0.5, self.sz[2]-0.5)
        else:
            self.xlim = xlim
        if not ylim:
            self.ylim = (-0.5, self.sz[1]-0.5)
        else:
            self.ylim = ylim
        if not zlim:
            self.zlim = (-0.5, self.sz[0]-0.5)
        else:
            self.zlim = zlim            
        
        if not vmin:
            self.vmin = max(self.vol.mean()-3*self.vol.std(), self.vol.min())
        else:
            self.vmin = vmin
        if not vmax:
            self.vmax = min(self.vol.mean()+3*self.vol.std(), self.vol.max())
        else:
            self.vmax = vmax
        self.zpos, self.ypos, self.xpos = self.sz//2
        self.zpos = 0   
        self.value = self.vol[self.zpos, self.ypos, self.xpos]         
        self.fig.add_axes([0.1, 0.1, 0.35, 0.35], autoscale_on=False, 
                          xlabel='X pix (2nd axis)', ylabel='Y pix (1st axis)', 
                          xlim=self.xlim, ylim=self.ylim) 
        self.fig.add_axes([0.1, 0.55, 0.35, 0.35], autoscale_on=False, 
                          xlabel='X pix (2nd axis)', ylabel='Z pix (0th axis)', 
                          xlim=self.xlim, ylim=self.zlim)
        self.fig.add_axes([0.6, 0.1, 0.35, 0.35], autoscale_on=False, 
                          xlabel='Z pix (0th axis)', ylabel='Y pix (1st axis)', 
                          xlim=self.zlim, ylim=self.ylim)

        self.fig.axes[0].set_title('('+str(self.xpos)+', '+str(self.ypos)+', '
                                      +str(self.zpos)+') = '
                                      +str(format(self.value, "10.2e")))        
        self.fig.axes[1].set_title('Y = '+str(self.ypos))
        self.fig.axes[2].set_title('X = '+str(self.xpos))
        
        self.img_xy = self.fig.axes[0].imshow(self.vol[self.zpos, :, :], 
                                              origin='lower', 
                                              aspect='auto', 
                                              vmin=self.vmin, vmax=self.vmax)
        self.img_xz = self.fig.axes[1].imshow(self.vol[:, self.ypos, :], 
                                              origin='lower',
                                              aspect='auto', 
                                              vmin=self.img_xy.get_clim()[0], 
                                              vmax=self.img_xy.get_clim()[1])
        self.img_zy = self.fig.axes[2].imshow(self.vol[:, :, self.xpos].T, 
                                              origin='lower', 
                                              aspect='auto', 
                                              vmin=self.img_xy.get_clim()[0], 
                                              vmax=self.img_xy.get_clim()[1])
        self.img_xy.hor = self.fig.axes[0].plot(self.xlim, 
                                                self.ypos*np.array([1, 1]), 
                                                '--', color='gray')
        self.img_xy.vert = self.fig.axes[0].plot(self.xpos*np.array([1, 1]), 
                                                 self.ylim, 
                                                 '--', color='gray')
        self.img_xz.hor = self.fig.axes[1].plot(self.xlim, 
                                                self.zpos*np.array([1, 1]), 
                                                '--', color='gray')
        self.img_xz.vert = self.fig.axes[1].plot(self.xpos*np.array([1, 1]), 
                                                 self.zlim, 
                                                 '--', color='gray')
        self.img_zy.hor = self.fig.axes[2].plot(self.zlim, 
                                                self.ypos*np.array([1, 1]),
                                                '--', color='gray')
        self.img_zy.vert = self.fig.axes[2].plot(self.zpos*np.array([1, 1]), 
                                                 self.ylim, 
                                                 '--', color='gray')
        cax = plt.axes([0.45, 0.1, 0.02, 0.35])
        self.colorbar = plt.colorbar(self.img_xy, cax=cax)
        ax = self.fig.add_axes([0.6, 0.55, 0.35, 0.35])
        # ax.set_position([0,0,1,1])
        ax.set_axis_off()
        ax.text(0.05, 0.9, r'$\leftarrow$, $\rightarrow$ : $\pm$ X')
        ax.text(0.05, 0.8, r'$\uparrow$, $\downarrow$: $\pm$ Y')
        ax.text(0.05, 0.7, r'PgUp, PgDn: $\pm$ Z')
        ax.text(0.05, 0.6, r'Home, End: Last, First Z')
        ax.text(0.05, 0.5, r'Ctrl: ${\times}$ 5')
        ax.text(0.05, 0.4, r'Ctrl+Shift: $\times$ 50')
        ax.text(0.05, 0.3, r'Q: Quit')

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()

    def on_key(self, event):
        if 'ctrl' in event.key:
            co1 = 5
        else:
            co1 = 1
        if 'shift' in event.key:
            co2 = 10
        else:
            co2 = 1
        if 'left' in event.key:
            self.xpos = (self.xpos-co1*co2+self.sz[2]) % self.sz[2]
        elif 'right' in event.key:
            self.xpos = (self.xpos+co1*co2) % self.sz[2]
        elif 'pageup' in event.key:
            self.zpos = (self.zpos+co1*co2) % self.sz[0]
        elif 'pagedown' in event.key:
            self.zpos = (self.zpos-co1*co2+self.sz[0]) % self.sz[0]
        elif 'home' in event.key:
            self.zpos = self.sz[0]-1
        elif 'end' in event.key:
            self.zpos = 0
        elif 'up' in event.key:
            self.ypos = (self.ypos+co1*co2) % self.sz[1]
        elif 'down' in event.key:
            self.ypos = (self.ypos-co1*co2+self.sz[1]) % self.sz[1]
        elif 'q' in event.key:
            del(self)
            plt.close(fig='Volume Explorer')
            return
        self.img_update()            

    def img_update(self):
        self.img_xy.set_data(self.vol[self.zpos, :, :])
        self.img_xz.set_data(self.vol[:, self.ypos, :])
        self.img_zy.set_data(self.vol[:, :, self.xpos].T)
        self.img_xy.hor[0].set_data(self.xlim, self.ypos*np.array([1, 1]))
        self.img_xy.vert[0].set_data(self.xpos*np.array([1, 1]), self.ylim)
        self.img_xz.hor[0].set_data(self.xlim, self.zpos*np.array([1, 1]))
        self.img_xz.vert[0].set_data(self.xpos*np.array([1, 1]), self.zlim)
        self.img_zy.hor[0].set_data(self.zlim, self.ypos*np.array([1, 1]))
        self.img_zy.vert[0].set_data(self.zpos*np.array([1, 1]), self.ylim)
        self.value = self.vol[self.zpos, self.ypos, self.xpos] 
        self.fig.axes[0].set_title('('+str(self.xpos)+', '+str(self.ypos)+', '
                                      +str(self.zpos)+') = '
                                      +str(format(self.value, "10.2e")))        
        self.fig.axes[1].set_title('Y = '+str(self.ypos))
        self.fig.axes[2].set_title('X = '+str(self.xpos))
        self.img_xz.set_clim(self.img_xy.get_clim())
        self.img_zy.set_clim(self.img_xy.get_clim())
        self.fig.axes[1].set_xlim(self.fig.axes[0].get_xlim())
        self.fig.axes[2].set_ylim(self.fig.axes[0].get_ylim())
        self.fig.canvas.draw_idle()

    def set_clim(self, clim1, clim2):
        self.img_xy.set_clim(clim1, clim2)
        self.img_update()