# import standard plotting 
import matplotlib.pyplot as plt
from matplotlib import gridspec
from IPython.display import clear_output

# other basic libraries
import math
import time
import copy
import autograd.numpy as np

# patch / convex hull libraries
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull

# import optimizer class from same library
from . import optimimzers

class Visualizer:
    '''
    Demonstrate multiclass logistic regression classification
    
    '''
    
    #### initialize ####
    def __init__(self,data):                
        # define the input and output of our dataset
        self.x = np.asarray(data[:,:-1])
        self.x.shape = (len(self.x),np.shape(data)[1]-1); self.x = self.x.T;
        self.y = data[:,-1]
        self.y.shape = (len(self.y),1)
    
        # colors for viewing classification data 'from above'
        self.colors = [[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5],'mediumaquamarine']
        
    ###### cost functions ######
    # counting cost for multiclass classification - used to determine best weights
    def counting_cost(self,W):        
        # pre-compute predictions on all points
        y_hats = W[0,:] + np.dot(self.x.T,W[1:,:])

        # compute counting cost
        cost = 0
        for p in range(len(self.y)):
            # pluck out current true label, predicted label
            y_p = int(self.y[p][0]) - 1        # subtract off one due to python indexing
            y_hat_p = int(np.argmax(y_hats[p])) 

            # update cost
            cost += np.abs(np.sign(y_hat_p - y_p))
        return cost
    
    # multiclass logistic - base function
    def standard(self,W):        
        # pre-compute predictions on all points
        y_hats = W[0,:] + np.dot(self.x.T,W[1:,:])

        # compute counting cost
        cost = 0
        for p in range(len(self.y)):
            # pluck out current true label, predicted label
            y_p = int(self.y[p][0]) - 1    # subtract off one due to python indexing
            max_val = max(y_hats[p,:])

            # update cost
            cost += max_val - y_hats[p,y_p]
        return cost
    
    ###### plotting functions ######  
    # show data
    def show_dataset(self):
        # initialize figure
        fig = plt.figure(figsize = (8,4))
        artist = fig
        gs = gridspec.GridSpec(1, 3,width_ratios = [1,3,1]) 

        # setup current axis
        ax = plt.subplot(gs[1],aspect = 'equal'); 
        
        # run axis through data plotter
        self.plot_data(ax)
        
        # determine plotting ranges
        minx = min(min(self.x[0,:]),min(self.x[1,:]))
        maxx = max(max(self.x[0,:]),max(self.x[1,:]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # dress panel
        ax.set_xlim(minx,maxx)
        ax.set_ylim(minx,maxx)
        
        plt.show()
        
    # show coloring of entire space
    def show_complete_coloring(self,w_hist,**kwargs):
        # determine best set of weights from history
        cost_evals = []
        for i in range(len(w_hist)):
            W = w_hist[i]
            cost = self.counting_cost(W)
            cost_evals.append(cost)
        ind = np.argmin(cost_evals)
        self.W = w_hist[ind]
        
        # generate input range for viewing range
        minx = min(min(self.x[0,:]),min(self.x[1,:]))
        maxx = max(max(self.x[0,:]),max(self.x[1,:]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # initialize figure
        fig = plt.figure(figsize = (8,4))
        
        show_cost = False
        if 'show_cost' in kwargs:
            show_cost = kwargs['show_cost']
        if show_cost == True:   
            gs = gridspec.GridSpec(1, 3,width_ratios = [1,1,1]) 
            ax3 = plt.subplot(gs[2])
            ax3.plot(cost_evals)
        else:
            gs = gridspec.GridSpec(1, 2,width_ratios = [1,1]) 

        # setup current axis
        ax = plt.subplot(gs[0],aspect = 'equal');
        ax2 = plt.subplot(gs[1],aspect = 'equal');
        
        # plot panel with all data and separators
        self.plot_data(ax)
        self.plot_data(ax2)
        self.plot_all_separators(ax)
                
        ### draw multiclass boundary on right panel
        r = np.linspace(minx,maxx,2000)
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        o = np.ones((len(r)**2,1))
        h = np.concatenate([o,w1_vals,w2_vals],axis = 1)
        pts = np.dot(h,self.W)
        g_vals = np.argmax(pts,axis = 1)

        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))
        
        # plot contour
        C = len(np.unique(self.y))
        ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = range(0,C+1),linewidths = 2.75,zorder = 4)
        ax2.contourf(w1_vals,w2_vals,g_vals+1,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))
        ax.contourf(w1_vals,w2_vals,g_vals+1,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))

        # dress panel
        ax.set_xlim(minx,maxx)
        ax.set_ylim(minx,maxx)
        ax.axis('off')
        
        ax2.set_xlim(minx,maxx)
        ax2.set_ylim(minx,maxx)
        ax2.axis('off')     
        
    ### compare grad descent runs - given cost to counting cost ###
    def compare_to_counting(self,cost,**kwargs):
        # parse args
        num_runs = 1
        if 'num_runs' in kwargs:
            num_runs = kwargs['num_runs']
        max_its = 200
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        alpha = 10**-3
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']  
        steplength_rule = 'none'
        if 'steplength_rule' in kwargs:
            steplength_rule = kwargs['steplength_rule']
        version = 'unnormalized'
        if 'version' in kwargs:
            version = kwargs['version'] 
        algo = 'gradient_descent'
        if 'algo' in kwargs:
            algo = kwargs['algo']
         
        #### perform all optimizations ###
        g = self.standard
        if cost == 'standard':
            g = self.standard
        if cost == 'softmax':
            g = self.softmax
        g_count = self.counting_cost
        
        # create instance of optimizers
        self.opt = optimimzers.MyOptimizers()
        
        # run optimizer
        big_w_hist = []
        C = len(np.unique(self.y))
        for j in range(num_runs):
            # create random initialization
            w_init =  np.random.randn(C,np.shape(self.x)[0]+1)

            # run algo
            if algo == 'gradient_descent':# run gradient descent
                w_hist = self.opt.gradient_descent(g = g,w = w_init,version = version,max_its = max_its, alpha = alpha,steplength_rule = steplength_rule)
            elif algo == 'newtons_method':
                w_hist = self.opt.newtons_method(g = g,w = w_init,max_its = max_its)
            big_w_hist.append(w_hist)
            
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (8,4))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);
        
        #### start runs and plotting ####
        for j in range(num_runs):
            w_hist = big_w_hist[j]
            
            # evaluate counting cost / other cost for each weight in history, then plot
            count_evals = []
            cost_evals = []
            for k in range(len(w_hist)):
                w = w_hist[k]
                g_eval = g(w)
                cost_evals.append(g_eval)
                
                count_eval = g_count(w)
                count_evals.append(count_eval)
                
            # plot each 
            ax1.plot(np.arange(0,len(w_hist)),count_evals[:len(w_hist)],linewidth = 2)
            ax2.plot(np.arange(0,len(w_hist)),cost_evals[:len(w_hist)],linewidth = 2)
                
        #### cleanup plots ####
        # label axes
        ax1.set_xlabel('iteration',fontsize = 13)
        ax1.set_ylabel('num misclassifications',rotation = 90,fontsize = 13)
        ax1.set_title('number of misclassifications',fontsize = 14)
        ax1.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        
        ax2.set_xlabel('iteration',fontsize = 13)
        ax2.set_ylabel('cost value',rotation = 90,fontsize = 13)
        title = cost + ' cost'
        ax2.set_title(title,fontsize = 14)
        ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        
        plt.show()
        
    
    
    #### utility functions ####           
    #plot data
    def plot_data(self,ax):
        # initialize figure, plot data, and dress up panels with axes labels etc.
        num_classes = np.size(np.unique(self.y))
                
        # color current class
        for a in range(0,num_classes):
            t = np.argwhere(self.y == a+1)
            t = t[:,0]
            ax.scatter(self.x[0,t],self.x[1,t], s = 50,color = self.colors[a],edgecolor = 'k',linewidth = 1.5)
        
    # plot separators
    def plot_all_separators(self,ax):
        # determine plotting ranges
        minx = min(min(self.x[0,:]),min(self.x[1,:]))
        maxx = max(max(self.x[0,:]),max(self.x[1,:]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # initialize figure, plot data, and dress up panels with axes labels etc.
        num_classes = np.size(np.unique(self.y))
                
        # color current class
        r = np.linspace(minx,maxx,400)
        for a in range(0,num_classes):
            # get current weights
            w = self.W[:,a]
            
            # draw subproblem separator
            z = - w[0]/w[2] - w[1]/w[2]*r
            r = np.linspace(minx,maxx,400)
            ax.plot(r,z,linewidth = 2,color = self.colors[a],zorder = 3)
            ax.plot(r,z,linewidth = 2.75,color = 'k',zorder = 2)