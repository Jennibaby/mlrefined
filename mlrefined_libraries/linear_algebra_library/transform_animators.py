# import custom JS animator
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only

# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
import time
from matplotlib import gridspec
import copy

# import autograd functionality
import numpy as np
import math

# make the adjustable grid
def make_warpable_grid(horz_min,horz_max,vert_min,vert_max):
    s = np.linspace(-10,10,40)
    s.shape = (len(s),1)
    g = np.array([-10,-10])
    g.shape = (1,len(g))
    e = np.linspace(-10,10,200)
    e.shape = (len(e),1)
    f = np.ones((200,1))
    f.shape = (len(f),1)
    for a in s:
        t = a*f
        h = np.concatenate((e,t),axis = 1)
        i = np.concatenate((t,e),axis = 1)
        j = np.concatenate((h,i),axis = 0)
        g = np.concatenate((g,j),axis = 0)

    grid = g[1:,:]
    return grid

# animator for showing grid of points transformed by linear transform
def transform2d_animator(mat1,**kwargs):  
    if len(mat1.shape) > 2 or len(np.argwhere(np.asarray(mat1.shape) > 2)) > 0:
        print ('input matrix must be 2x2')
        return 
    orig_mat1 = copy.deepcopy(mat1)
                                    
    # define number of frames
    num_frames = 100
    if 'num_frames' in kwargs:
        num_frames = kwargs['num_frames']
     
    # define convex-combo parameter - via num_frames
    alphas = np.linspace(0,1,num_frames)

    # define grid of points via meshgrid
    viewx = 5
    viewgap = 0.1*viewx
    grid = make_warpable_grid(horz_min=-viewx,horz_max=viewx,vert_min=-viewx,vert_max=viewx)
    orig_grid = copy.deepcopy(grid)
    
    # initialize figure
    fig = plt.figure(figsize = (16,8))
    artist = fig
    
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,3, 1]) 
    ax1 = plt.subplot(gs[0]); ax1.axis('off');
    ax3 = plt.subplot(gs[2]); ax3.axis('off');
    
    # plot input function
    ax = plt.subplot(gs[1])

    # animate
    def animate(k):
        # clear the panel
        ax.cla()
        
        # print rednering update
        if np.mod(k+1,25) == 0:
            print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
        if k == num_frames - 1:
            print ('animation rendering complete!')
            time.sleep(1.5)
            clear_output()  
        
        # get current lambda, define current matrix
        alpha = alphas[k]
        mat1 = alpha*orig_mat1 + (1 - alpha)*np.eye(2)

        # compute current transformation of points and plot
        grid = np.dot(mat1,orig_grid.T).T
            
        # plot points
        for i in range(80):
            ax.plot(grid[200*i:(i+1)*200,0],grid[200*i:(i+1)*200,1],color = [0.75,0.75,0.75],linewidth = 1,zorder = 0)   
                          
        # plot x and y axes, and clean up
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k', linewidth=1)
        plt.axvline(x=0, color='k', linewidth=1)
   
        # return artist to render
        ax.set_xlim([-viewx - viewgap,viewx + viewgap])
        ax.set_ylim([-viewx - viewgap,viewx + viewgap])
        
        return artist,
        
    anim = animation.FuncAnimation(fig, animate,frames=num_frames, interval=num_frames, blit=True)
        
    return(anim)     
        