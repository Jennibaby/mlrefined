# import custom JS animator
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only

# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
import time

# simple first order taylor series visualizer
class visualizer:
    '''
    Illustrate gradient descent, Newton method, and Secant method for minimizing an input function, illustrating
    surrogate functions at each step.  A custom slider mechanism is used to progress each algorithm, and points are
    colored from green at the start of an algorithm, to yellow as it converges, and red as the final point.
    ''' 
     
    ######## gradient descent ########
    # run gradient descent 
    def run_gradient_descent(self):
        w = self.w_init
        self.w_hist = []
        self.w_hist.append(w)
        w_old = np.inf
        j = 0
        for j in range(int(self.max_its)):
            # update old w and index
            w_old = w
            
            # plug in value into func and derivative
            grad_eval = float(self.grad(w))
            
            # take gradient descent step
            w = w - self.alpha*grad_eval
            
            # record
            self.w_hist.append(w)

    # animate the gradient descent method
    def draw_it_gradient_descent(self,**kwargs):
        self.g = kwargs['g']                            # input function
        self.grad = compute_grad(self.g)              # gradient of input function
        self.hess = compute_grad(self.grad)           # hessian of input function
        self.w_init =float( -2)                       # user-defined initial point (adjustable when calling each algorithm)
        self.alpha = 10**-4                           # user-defined step length for gradient descent (adjustable when calling gradient descent)
        self.max_its = 20                             # max iterations to run for each algorithm
        self.w_hist = []                              # container for algorithm path
        
        # get new initial point if desired
        if 'w_init' in kwargs:
            self.w_init = float(kwargs['w_init'])
            
        # take in user defined step length
        if 'alpha' in kwargs:
            self.alpha = float(kwargs['alpha'])
            
        # take in user defined maximum number of iterations
        if 'max_its' in kwargs:
            self.max_its = float(kwargs['max_its'])
            
        # initialize figure
        fig = plt.figure(figsize = (9,4))
        artist = fig
        
        # remove whitespace from figure
        #fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        #fig.subplots_adjust(wspace=0.01,hspace=0.01)

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,4,1]) 

        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        ax = plt.subplot(gs[1]); 

        # generate function for plotting on each slide
        w_plot = np.linspace(-3.1,3.1,200)
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.1
        width = 30
        
        # run gradient descent method
        self.w_hist = []
        self.run_gradient_descent()
        
        # colors for points --> green as the algorithm begins, yellow as it converges, red at final point
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        self.colorspec = []
        self.colorspec = np.concatenate((s,np.flipud(s)),1)
        self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
        # animation sub-function
        num_frames = 2*len(self.w_hist)+2
        print ('starting animation rendering...')
        def animate(t):
            ax.cla()
            k = math.floor((t+1)/float(2))
            
            # print rendering update            
            if np.mod(t+1,25) == 0:
                print ('rendering animation frame ' + str(t+1) + ' of ' + str(num_frames))
            if t == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # plot function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 2)                           # plot function
            
            # plot initial point and evaluation
            if k == 0:
                w_val = self.w_init
                g_val = self.g(w_val)
                ax.scatter(w_val,g_val,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 3, marker = 'X')            # plot point of tangency
                ax.scatter(w_val,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 3)
                
                # draw dashed line connecting w axis to point on cost function
                s = np.linspace(0,g_val)
                o = np.ones((len(s)))
                ax.plot(o*w_val,s,'k--',linewidth=1)

            # plot all input/output pairs generated by algorithm thus far
            if k > 0:
                # plot all points up to this point
                for j in range(min(k-1,len(self.w_hist))):  
                    w_val = self.w_hist[j]
                    g_val = self.g(w_val)
                    ax.scatter(w_val,g_val,s = 90,c = self.colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3,marker = 'X')            # plot point of tangency
                    ax.scatter(w_val,0,s = 90,facecolor = self.colorspec[j],edgecolor = 'k',linewidth = 0.7, zorder = 2)
                    
            # plot surrogate function and travel-to point
            if k > 0 and k < len(self.w_hist) + 1:          
                # grab historical weight, compute function and derivative evaluations
                w = self.w_hist[k-1]
                g_eval = self.g(w)
                grad_eval = float(self.grad(w))
            
                # determine width to plot the approximation -- so its length == width defined above
                div = float(1 + grad_eval**2)
                w1 = w - math.sqrt(width/div)
                w2 = w + math.sqrt(width/div)

                # use point-slope form of line to plot
                wrange = np.linspace(w1,w2, 100)
                h = g_eval + grad_eval*(wrange - w)

                # plot tangent line
                ax.plot(wrange,h,color = 'lime',linewidth = 2,zorder = 1)      # plot approx

                # plot tangent point
                ax.scatter(w,g_eval,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 3,marker = 'X')            # plot point of tangency
            
                # plot next point learned from surrogate
                if np.mod(t,2) == 0:
                    # create next point information
                    w_zero = w - self.alpha*grad_eval
                    g_zero = self.g(w_zero)
                    h_zero = g_eval + grad_eval*(w_zero - w)

                    # draw dashed line connecting the three
                    vals = [0,h_zero,g_zero]
                    vals = np.sort(vals)

                    s = np.linspace(vals[0],vals[2])
                    o = np.ones((len(s)))
                    ax.plot(o*w_zero,s,'k--',linewidth=1)

                    # draw intersection at zero and associated point on cost function you hop back too
                    ax.scatter(w_zero,h_zero,s = 100,c = 'k', zorder = 3,marker = 'X')
                    ax.scatter(w_zero,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 3)
                    ax.scatter(w_zero,g_zero,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 3, marker = 'X')            # plot point of tangency
                 
            # fix viewing limits
            ax.set_xlim([-3,3])
            ax.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            
            # place title
            ax.set_xlabel(r'$w$',fontsize = 14)
            ax.set_ylabel(r'$g(w)$',fontsize = 14,rotation = 0,labelpad = 25)

            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)

