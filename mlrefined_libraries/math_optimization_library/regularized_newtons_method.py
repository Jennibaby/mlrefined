# import custom JS animator
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only

# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math

# simple first order taylor series visualizer
class visualizer:
    '''
    Illustrating how to regularize Newton's method to deal with nonconvexity.  Using a custom slider
    widget we can visualize the result of adding a pure weighted quadratic to the second derivative
    at each step of Newton's method.  Each time the slider is moved a new complete run of regularized
    Newton's method is illustrated, where at each step going from left to right the weight on the 
    pure quadratic is increased.
    
    For a non-convex function we can see how that - without reglarizing - we will climb to a local maximum,
    since at each step the quadratic approximation is concave.  However if the regularization parameter is set
    large enough the sum quadratic is made convex, and we can descend.  If the weight is made too high we 
    completely drown out second derivative and have gradient descent.
    ''' 
    
    def __init__(self,**args):
        self.g = args['g']                          # input function
        self.grad = compute_grad(self.g)            # first derivative of input function
        self.hess = compute_grad(self.grad)         # second derivative of input function
        self.w_init = float( -2.3)                  # initial point
        self.w_hist = []
        self.beta_range = np.linspace(0,2,20)       # range of regularization parameter to try
        self.max_its = 10
        
    ######## newton's method ########
    # run newton's method
    def run_newtons_method(self,beta):
        w_val = self.w_init
        self.w_hist = []
        self.w_hist.append(w_val)
        w_old = np.inf
        j = 0
        while (w_old - w_val)**2 > 10**-5 and j < self.max_its:
            # update old w and index
            w_old = w_val
            j+=1
            
            # plug in value into func and derivative
            grad_val = float(self.grad(w_val))
            hess_val = float(self.hess(w_val))

            # take newtons step
            curvature = hess_val + beta
            if abs(curvature) > 10**-2:
                w_val = w_val - grad_val/curvature
            
            # record
            self.w_hist.append(w_val)

    # animate the method
    def draw_it_newtons(self,**args):
        # let the user define the range of regularization parameters to try as well as the initial point of all runs
        if 'beta_range' in args:
            self.beta_range = args['beta_range']
        if 'w_init' in args:
            self.w_init = float(args['w_init'])
        if 'max_its' in args:
            self.max_its = float(args['max_its'])
    
        # initialize figure
        fig = plt.figure(figsize = (15,7))
        artist = fig
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # generate function for plotting on each slide
        w_plot = np.linspace(-3,3,200)
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.5
        w_vals = np.linspace(-2.5,2.5,50)
 
        # animation sub-function
        def animate(k):
            # clear the previous panel for next slide
            ax1.cla()
            ax2.cla()
            
            # plot function 
            ax1.plot(w_plot,g_plot,color = 'k',zorder = 0)               # plot function
            
            # plot initial point and evaluation
            if k == 0:
                w_val = self.w_init
                g_val = self.g(w_val)
                ax1.scatter(w_val,g_val,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 2)            # plot point of tangency
                ax1.scatter(w_val,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 2, marker = 'X')

            # plot function alone first along with initial point
            if k > 0:
                beta = self.beta_range[k-1]
                
                # run gradient descent method
                self.w_hist = []
                self.run_newtons_method(beta = beta)
        
                # colors for points
                s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
                s.shape = (len(s),1)
                t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
                t.shape = (len(t),1)
                s = np.vstack((s,t))
                self.colorspec = []
                self.colorspec = np.concatenate((s,np.flipud(s)),1)
                self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
                # plot everything for each iteration 
                for j in range(len(self.w_hist)):  
                    w_val = self.w_hist[j]
                    g_val = self.g(w_val)
                    ax1.scatter(w_val,g_val,s = 90,c = self.colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    ax1.scatter(w_val,0,s = 90,facecolor = self.colorspec[j],marker = 'X',edgecolor = 'k',linewidth = 0.7, zorder = 2)
                    
                    # plug in value into func and derivative
                    g_val = self.g(w_val)
                    g_grad_val = self.grad(w_val)
                    g_hess_val = self.hess(w_val)

                    # determine width of plotting area for second order approximator
                    width = 0.5
                    if g_hess_val < 0:
                        width = - width

                    # compute second order approximation
                    wrange = np.linspace(w_val - 3,w_val + 3, 100)
                    h = g_val + g_grad_val*(wrange - w_val) + 0.5*(g_hess_val + beta)*(wrange - w_val)**2 

                    # plot all
                    ax1.plot(wrange,h,color = self.colorspec[j],linewidth = 2,alpha = 0.4,zorder = 1)      # plot approx
            
                    ### plot all on cost function decrease plot
                    ax2.scatter(j,g_val,s = 90,c = self.colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    
                    # clean up second axis
                    ax2.set_xlabel('iteration',fontsize = 13)
                    ax2.set_ylabel('cost function value',fontsize = 13)
                    ax2.set_xticks(np.arange(len(self.w_hist)))
                    
                    # plot connector between points for visualization purposes
                    if j > 0:
                        w_old = self.w_hist[j-1]
                        w_new = self.w_hist[j]
                        g_old = self.g(w_old)
                        g_new = self.g(w_new)
                        ax2.plot([j-1,j],[g_old,g_new],color = self.colorspec[j],linewidth = 2,alpha = 0.4,zorder = 1)      # plot approx

            # fix viewing limits
            ax1.set_xlim([-3,3])
            ax1.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            
            # draw axes
            ax1.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)

            return artist,

        anim = animation.FuncAnimation(fig, animate,frames=len(self.beta_range)+1, interval=len(self.beta_range)+1, blit=True)

        return(anim)