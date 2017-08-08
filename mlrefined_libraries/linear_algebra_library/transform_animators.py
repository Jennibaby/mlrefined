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
        
    # grab points if input
    pts = []
    orig_pts = []
    if 'pts' in kwargs:
        pts = kwargs['pts']
        orig_pts = copy.deepcopy(pts)
     
    # define convex-combo parameter - via num_frames
    alphas = np.linspace(0,1,num_frames)

    # define grid of points via meshgrid
    viewx = 5
    viewgap = 0.1*viewx
    viewx2 = 10
    grid = make_warpable_grid(horz_min=-viewx2,horz_max=viewx2,vert_min=-viewx2,vert_max=viewx2)
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
            
        # plot warped grid
        for i in range(80):
            ax.plot(grid[200*i:(i+1)*200,0],grid[200*i:(i+1)*200,1],color = [0.75,0.75,0.75],linewidth = 1,zorder = 0)  
            
        # plot input points
        if len(orig_pts) > 0:
            pts = np.dot(mat1,orig_pts.T).T
            ax.plot(pts[:,0],pts[:,1],c = 'k',linewidth = 3)
                          
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
        
    # animate the method
def inner_product_visualizer(**kwargs):
    # set number of frames for animation
    num_frames = 300                          # number of slides to create - the input range [-3,3] is divided evenly by this number
    if 'num_frames' in kwargs:
        num_frames = kwargs['num_frames']
    
    # initialize figure
    fig = plt.figure(figsize = (16,8))
    artist = fig

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1],wspace=0.3, hspace=0.05) 
    ax1 = plt.subplot(gs[0]);
    ax2 = plt.subplot(gs[1]); 
    
    # create dataset for unit circle
    v = np.linspace(0,2*np.pi,100)
    s = np.sin(v)
    s.shape = (len(s),1)
    t = np.cos(v)
    t.shape = (len(t),1)
    
    # user defined starting point on the circle
    start = 0
    if 'start' in kwargs:
        start = kwargs['start']
        
    # create span of angles to plot over
    v = np.linspace(start,2*np.pi + start,num_frames)
    y = 0.87*np.sin(v)
    x = 0.87*np.cos(v)
    
    # create linspace for sine/cosine plots
    w = np.linspace(start,2*np.pi + start,300)
    
    # define colors for sine / cosine
    colors = ['salmon','cornflowerblue']
    
    # print update
    print ('starting animation rendering...')
    
    # animation sub-function
    def animate(k):
        # clear panels for next slide
        ax1.cla()
        ax2.cla()
        
        # print rendering update
        if np.mod(k+1,25) == 0:
            print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
        if k == num_frames - 1:
            print ('animation rendering complete!')
            time.sleep(1.5)
            clear_output()
        
        ### setup left panel ###
        # plot circle with lines in left panel
        ax1.plot(s,t,color = 'k',linewidth = 3)
        
        # plot moving arrow
        ax1.arrow(0, 0, x[k], y[k], head_width=0.1, head_length=0.1, fc='k', color = colors[1],linewidth=3,zorder = 3)
        
        # plot fixed arrow
        ax1.arrow(0, 0, 0.87, 0, head_width=0.1, head_length=0.1, fc='k', color = 'k',linewidth=3,zorder = 3)

        # clean up panel
        ax1.grid(True, which='both')
        ax1.axhline(y=0, color='k')
        ax1.axvline(x=0, color='k')
        
        ### setup right panel ###
        # determine closest value in space of sine/cosine input
        current_angle = v[k]
        ind = np.argmin(np.abs(w - current_angle))
        p = w[:ind+1]
        
        # plot sine wave so far
        ax2.plot(p,np.cos(p),color = colors[1],linewidth=4,zorder = 3)
        
        # cleanup plot
        ax2.grid(True, which='both')
        ax2.axhline(y=0, color='k')
        ax2.axvline(x=0, color='k')   
        ax2.set_xlim([-0.3 + start,2*np.pi + 0.3 + start])
        ax2.set_ylim([-1.2,1.2])
        
        # add legend
        ax2.legend([r'cos$(\theta)$'],loc='center left', bbox_to_anchor=(0.33, 1.05),fontsize=18,ncol=2)

        return artist,

    anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

    return(anim)