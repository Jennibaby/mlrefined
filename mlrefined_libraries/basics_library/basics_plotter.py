import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from IPython.display import display, HTML

# custom plot for spiffing up plot of a single mathematical function
def single_plot(table,**kwargs):
    xlabel = '$w_1$'
    ylabel = '$w_2$'
    zlabel = '$g(w_1,w_2)$'
    plot_type = 'continuous'
    fontsize = 15
    guides = 'off'
    rotate_ylabel = 90
    label_fontsize = 15
    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    if 'ylabel' in kwargs:
        ylabel = kwargs['ylabel']
    if 'zlabel' in kwargs:
        zlabel = kwargs['zlabel']
    if 'fontsize' in kwargs:
        fontsize = kwargs['fontsize']
        
    if 'plot_type' in kwargs:
        plot_type = kwargs['plot_type']
    if 'rotate_ylabel' in kwargs:
        rotate_ylabel = kwargs['rotate_ylabel']   
    if 'label_fontsize' in kwargs:
        label_fontsize = kwargs['label_fontsize']
    if 'guides' in kwargs:
        guides = kwargs['guides']       
       
    # is the function 2-d or 3-d?
    dim = np.shape(table)[1]
    
    # single two dimensonal plot
    plt.style.use('ggplot')
    if dim == 2:
        fig = plt.figure(figsize = (12,4))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,3, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # plot input function
        ax2 = plt.subplot(gs[1])
        
        # choose plot type
        if plot_type == 'continuous':
            ax2.plot(table[:,0], table[:,1], c='r', linewidth=2,zorder = 3)
        if plot_type == 'scatter':
            ax2.scatter(table[:,0], table[:,1], c='r', s=50,edgecolor='k',linewidth=1)
            
            # if the guides are turned on, plot them
            if guides == 'on':
                ax2.plot(table[:,0], table[:,1], c='r', linewidth=2,zorder = 2,alpha = 0.25)

        # plot x and y axes, and clean up
        ax2.grid(True, which='both')
        ax2.axhline(y=0, color='k', linewidth=1)
        ax2.axvline(x=0, color='k', linewidth=1)
        
        # set viewing limits
        w = table[:,0]
        wrange = max(w) - min(w)
        wgap = wrange*0.15
        wmax = max(w) + wgap
        wmin = min(w) - wgap
        ax2.set_xlim([wmin,wmax])
        
        g = table[:,1]
        grange = max(g) - min(g)
        ggap = grange*0.25
        gmax = max(g) + ggap
        gmin = min(g) - ggap
        ax2.set_ylim([gmin,gmax])

        ax2.set_xlabel(xlabel,fontsize = label_fontsize)
        ax2.set_ylabel(ylabel,fontsize = label_fontsize,rotation = rotate_ylabel,labelpad = 20)
        plt.show()
    
    # single 3-d function plot
    if dim == 3:    
        # plot the line
        fig = plt.figure(figsize = (15,6))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,2, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # plot input function
        ax2 = plt.subplot(gs[1],projection='3d')
        ax2.plot_surface(table[:,0], table[:,1], table[:,2], alpha = 0.3,color = 'r',rstride=10, cstride=10,linewidth=2,edgecolor = 'k')

        # plot x and y axes, and clean up
        ax2.set_xlabel(xlabel,fontsize = fontsize)
        ax2.set_ylabel(ylabel,fontsize = fontsize,rotation = 0)
        ax2.set_zlabel(zlabel,fontsize = fontsize)

        # clean up plot and set viewing angle
        ax2.view_init(10,30)
        plt.show()
        
# custom plot for spiffing up plot of a two mathematical functions
def double_plot(table1,table2,**kwargs): 
    # get labeling arguments
    xlabel = '$w_1$'
    ylabel_1 = '$g$'
    ylabel_2 = '$g$'
    fontsize = 15
    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    if 'ylabel_1' in kwargs:
        ylabel_1 = kwargs['ylabel_1']
    if 'ylabel_2' in kwargs:
        ylabel_2 = kwargs['ylabel_2']
    if 'fontsize' in kwargs:
        fontsize = kwargs['fontsize']
        
    # plot the functions 
    fig = plt.figure(figsize = (15,4))
    plt.style.use('ggplot')
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122); 
    plot_type = 'continuous'
    if 'plot_type' in kwargs:
        plot_type = kwargs['plot_type']
    if plot_type == 'scatter':
        ax1.scatter(table1[:,0], table1[:,1], c='r', s=20,zorder = 3)
        ax2.scatter(table2[:,0], table2[:,1], c='r', s=20,zorder = 3)
    if plot_type == 'continuous':
        ax1.plot(table1[:,0], table1[:,1], c='r', linewidth=2,zorder = 3)
        ax2.plot(table2[:,0], table2[:,1], c='r', linewidth=2,zorder = 3)

    # plot x and y axes, and clean up
    ax1.set_xlabel(xlabel,fontsize = fontsize)
    ax1.set_ylabel(ylabel_1,fontsize = fontsize,rotation = 0,labelpad = 20)
    ax2.set_xlabel(xlabel,fontsize = fontsize)
    ax2.set_ylabel(ylabel_2,fontsize = fontsize,rotation = 0,labelpad = 20)
    
    ax1.grid(True, which='both'), ax2.grid(True, which='both')
    ax1.axhline(y=0, color='k', linewidth=1), ax2.axhline(y=0, color='k', linewidth=1)
    ax1.axvline(x=0, color='k', linewidth=1), ax2.axvline(x=0, color='k', linewidth=1)
    plt.show()
    
# plot pandas to html table centered in notebook
def table_plot(table):
    # display table mcdonalds revenue values
    display(HTML('<center>' + table.to_html(index=False) + '</center>'))