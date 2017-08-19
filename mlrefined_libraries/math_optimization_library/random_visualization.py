import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from IPython.display import display, HTML
import copy
        
# custom plot for spiffing up plot of a two mathematical functions
def double_plot(func,num_samples,**kwargs):    
    ## arguments user can tweak from control panel ##
    wmax = 1
    if 'wmax' in kwargs:
        wmax = kwargs['wmax']   
    view = [10,50]
    if 'view' in kwargs:
        view = kwargs['view']
    
    #### setup figure ####
    fig = plt.figure(figsize = (8,6)) 
    gs = gridspec.GridSpec(2, 2,wspace=0.3, hspace=0.8) 

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1],projection='3d')
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3],projection='3d')

    ###### create 2d panels ######
    # range over which to plot
    w = np.linspace(-wmax,wmax,1000)
     
    # create even grid of sample points, random sample points
    w_even = np.linspace(-wmax,wmax,num_samples)
    w_rand = 2*wmax*np.random.rand(num_samples) - wmax

    ### plot first 2d function - with even grid of points ###
    f = [func(val) for val in w]
    ax1.plot(w,f,color = 'k',zorder = 2, linewidth = 2)
    ax1.plot(w,[s*0 for s in f],color = 'k',zorder = 1, linewidth = 1)    # horizontal axis
   
    ax3.plot(w,f,color = 'k',zorder = 2, linewidth = 2)
    ax3.plot(w,[s*0 for s in f],color = 'k',zorder = 1, linewidth = 1)    # horizontal axis

    f_even = [func(val) for val in w_even]
    ax1.scatter(w_even,f_even,s = 50,c = 'lime',edgecolor = 'k',linewidth = 0.7,zorder = 3)       
    ax1.scatter(w_even,[s*0 for s in w_even],s = 50,c = 'b',edgecolor = 'k',linewidth = 0.7,zorder = 3)      
    
    ### plot second 2d function - with random grid of points ###
    f_rand = [func(val) for val in w_rand]
    ax3.scatter(w_rand,f_rand,s = 50,c = 'lime',edgecolor = 'k',linewidth = 0.7,zorder = 3)            
    ax3.scatter(w_rand,[s*0 for s in w_rand],s = 50,c = 'b',edgecolor = 'k',linewidth = 0.7,zorder = 3)   
    
    ### cleanup panels ###
    # label 2d panels, put axis lines on, etc.,
    ax1.set_xlabel('$w$',fontsize = 12)
    ax1.set_title('$g(w)$',fontsize = 12)
    ax1.grid(False, which='both')
   
    ax3.set_xlabel('$w$',fontsize = 12)
    ax3.set_title('$g(w)$',fontsize = 12)
    ax3.grid(False, which='both')
    
    
    ###### create 2d panels ######   
    w = np.linspace(-wmax,wmax,200)
    w1_vals, w2_vals = np.meshgrid(w,w)
    w1_vals.shape = (len(w)**2,1)
    w2_vals.shape = (len(w)**2,1)
    h = np.concatenate((w1_vals,w2_vals),axis=1)
    func_vals = np.asarray([func(s) for s in h])

    w1_vals.shape = (len(w),len(w))
    w2_vals.shape = (len(w),len(w))
    func_vals.shape = (len(w),len(w))

    ### plot function and z=0 for visualization ###
    ax2.plot_surface(w1_vals, w2_vals, func_vals, alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=0.7,edgecolor = 'k',zorder = 2)
        
    ax4.plot_surface(w1_vals, w2_vals, func_vals, alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=0.7,edgecolor = 'k',zorder = 2)       
        
    ### plot even vals ###
    w = np.linspace(-wmax,wmax,num_samples)
    w1_vals, w2_vals = np.meshgrid(w,w)
    w1_vals.shape = (len(w)**2,1)
    w2_vals.shape = (len(w)**2,1)
    h = np.concatenate((w1_vals,w2_vals),axis=1)
    f_even = np.asarray([func(s) for s in h])
    ax2.scatter(w1_vals,w2_vals,f_even,s = 50,c = 'lime',edgecolor = 'k',linewidth = 0.7,zorder = 3)       
    f_even = [s*0 for s in f_even]
    ax2.scatter(w1_vals,w2_vals,f_even,s = 50,c = 'b',edgecolor = 'k',linewidth = 0.7,zorder = 3)       
    
    ### plot random samples ###
    w_rand1 = 2*wmax*np.random.rand(num_samples**2) - wmax
    w_rand1.shape = (len(w_rand1),1)
    w_rand2 = 2*wmax*np.random.rand(num_samples**2) - wmax
    w_rand2.shape = (len(w_rand2),1)
    h = np.concatenate((w_rand1,w_rand2),axis=1)
    f_rand = [func(val) for val in h]
    ax4.scatter(w_rand1,w_rand2,f_rand,s = 50,c = 'lime',edgecolor = 'k',linewidth = 0.7,zorder = 3)       
    f_rand = [s*0 for s in f_rand]
    ax4.scatter(w_rand1,w_rand2,f_rand,s = 50,c = 'b',edgecolor = 'k',linewidth = 0.7,zorder = 3)       
    
        
    ### cleanup panels ###
    ax2.set_xlabel('$w_1$',fontsize = 12)
    ax2.set_ylabel('$w_2$',fontsize = 12,rotation = 0)
    ax2.set_title('$g(w_1,w_2)$',fontsize = 12)
    ax2.view_init(view[0],view[1])
    
    # clean up axis
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    ax2.xaxis.pane.set_edgecolor('white')
    ax2.yaxis.pane.set_edgecolor('white')
    ax2.zaxis.pane.set_edgecolor('white')
    
    ax2.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax2.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax2.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    ax2.set_xticks([-1,0,1])
    ax2.set_xticklabels([-1,0,1])
    
    ax2.set_yticks([-1,0,1])
    ax2.set_yticklabels([-1,0,1])
    
    ax2.set_zticks([0,1,2])
    ax2.set_zticklabels([0,1,2])
    
    # label axis
    ax4.set_xlabel('$w_1$',fontsize = 12)
    ax4.set_ylabel('$w_2$',fontsize = 12,rotation = 0)
    ax4.set_title('$g(w_1,w_2)$',fontsize = 12)
    ax4.view_init(view[0],view[1])
    
    # clean up axis
    ax4.xaxis.pane.fill = False
    ax4.yaxis.pane.fill = False
    ax4.zaxis.pane.fill = False

    ax4.xaxis.pane.set_edgecolor('white')
    ax4.yaxis.pane.set_edgecolor('white')
    ax4.zaxis.pane.set_edgecolor('white')
    
    ax4.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax4.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax4.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    ax4.set_xticks([-1,0,1])
    ax4.set_xticklabels([-1,0,1])
    
    ax4.set_yticks([-1,0,1])
    ax4.set_yticklabels([-1,0,1])
  
    ax4.set_zticks([0,1,2])
    ax4.set_zticklabels([0,1,2])


