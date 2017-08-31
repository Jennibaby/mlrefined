# import custom JS animator
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only

# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
from IPython.display import clear_output
import time
from matplotlib import gridspec

# simple first order taylor series visualizer
class visualizer:
    '''
    Illustrate gradient descent, Newton method, and Secant method for minimizing an input function, illustrating
    surrogate functions at each step.  A custom slider mechanism is used to progress each algorithm, and points are
    colored from green at the start of an algorithm, to yellow as it converges, and red as the final point.
    ''' 
    def __init__(self,**args):
        self.g = args['g']                            # input function
        self.grad = compute_grad(self.g)              # gradient of input function
        self.hess = compute_grad(self.grad)           # hessian of input function
        self.w_init =float( -2)                       # user-defined initial point (adjustable when calling each algorithm)
        self.alpha = 10**-4                           # user-defined step length for gradient descent (adjustable when calling gradient descent)
        self.max_its = 20                             # max iterations to run for each algorithm
        self.w_hist = []                              # container for algorithm path
        self.colors = [[0,1,0.25],[0,0.75,1]]    # set of custom colors used for plotting

    ######## newton's method ########
    # run newton's method
    def run_newtons_method(self):
        w = self.w_init
        self.w_hist = []
        self.w_hist.append(w)
        w_old = np.inf
        j = 0
        while (w_old - w)**2 > 10**-5 and j < self.max_its:
            # update old w and index
            w_old = w
            j+=1
            
            # plug in value into func and derivative
            grad_eval = float(self.grad(w))
            hess_eval = float(self.hess(w))

            # take newtons step
            w = w - grad_eval/(hess_eval + 10**-5)
            
            # record
            self.w_hist.append(w)

    # animate the method
    def draw_it_newtons(self,**kwargs):
        # get new initial point if desired
        if 'w_init' in kwargs:
            self.w_init = float(kwargs['w_init'])
            
        # take in user defined maximum number of iterations
        if 'max_its' in kwargs:
            self.max_its = float(kwargs['max_its'])
            
        wmax = 3
        if 'wmax' in kwargs:
            wmax = kwargs['wmax']
            
        # initialize figure
        fig = plt.figure(figsize = (10,5))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # plot input function
        ax = plt.subplot(gs[1],aspect = 'equal')
        
        # generate function for plotting on each slide
        w_plot = np.linspace(-wmax,wmax,1000)
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.1
        w_vals = np.linspace(-2.5,2.5,50)
        width = 1
        
        # run newtons method
        self.w_hist = []
        self.run_newtons_method()
        
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
        print ('starting animation rendering...')
        num_frames = 2*len(self.w_hist)+2
        def animate(t):
            ax.cla()
            k = math.floor((t+1)/float(2))
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if t == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()

            # plot function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 1)                           # plot function
            
            # plot initial point and evaluation
            if k == 0:
                w_val = self.w_init
                g_val = self.g(w_val)
                ax.scatter(w_val,g_val,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, marker = 'X',zorder = 2)            # plot point of tangency
                ax.scatter(w_val,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 2)
                # draw dashed line connecting w axis to point on cost function
                s = np.linspace(0,g_val)
                o = np.ones((len(s)))
                ax.plot(o*w_val,s,'k--',linewidth=1,zorder = 0)
                
            # plot all input/output pairs generated by algorithm thus far
            if k > 0:
                # plot all points up to this point
                for j in range(min(k-1,len(self.w_hist))):  
                    w_val = self.w_hist[j]
                    g_val = self.g(w_val)
                    ax.scatter(w_val,g_val,s = 90,c = self.colorspec[j],edgecolor = 'k',marker = 'X',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    ax.scatter(w_val,0,s = 90,facecolor = self.colorspec[j],edgecolor = 'k',linewidth = 0.7, zorder = 2)
                          
            # plot surrogate function and travel-to point
            if k > 0 and k < len(self.w_hist) + 1:          
                # grab historical weight, compute function and derivative evaluations    
                w_eval = self.w_hist[k-1]

                # plug in value into func and derivative
                g_eval = self.g(w_eval)
                g_grad_eval = self.grad(w_eval)
                g_hess_eval = self.hess(w_eval)

                # determine width of plotting area for second order approximator
                width = 0.5
                if g_hess_eval < 0:
                    width = - width

                # setup quadratic formula params
                a = 0.5*g_hess_eval
                b = g_grad_eval - 2*0.5*g_hess_eval*w_eval
                c = 0.5*g_hess_eval*w_eval**2 - g_grad_eval*w_eval - width

                # solve for zero points
                w1 = (-b + math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)
                w2 = (-b - math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)

                # compute second order approximation
                wrange = np.linspace(w1,w2, 100)
                h = g_eval + g_grad_eval*(wrange - w_eval) + 0.5*g_hess_eval*(wrange - w_eval)**2 

                # plot tangent line
                ax.plot(wrange,h,color = self.colors[1],linewidth = 2,zorder = 2)      # plot approx

                # plot tangent point
                ax.scatter(w_eval,g_eval,s = 100,c = 'm',edgecolor = 'k', marker = 'X',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
                # plot next point learned from surrogate
                if np.mod(t,2) == 0:
                    # create next point information
                    w_zero = w_eval - g_grad_eval/(g_hess_eval + 10**-5)
                    g_zero = self.g(w_zero)
                    h_zero = g_eval + g_grad_eval*(w_zero - w_eval) + 0.5*g_hess_eval*(w_zero - w_eval)**2

                    # draw dashed line connecting the three
                    vals = [0,h_zero,g_zero]
                    vals = np.sort(vals)

                    s = np.linspace(vals[0],vals[2])
                    o = np.ones((len(s)))
                    ax.plot(o*w_zero,s,'k--',linewidth=1)

                    # draw intersection at zero and associated point on cost function you hop back too
                    ax.scatter(w_zero,h_zero,s = 100,c = 'b',linewidth=0.7, marker = 'X',edgecolor = 'k',zorder = 3)
                    ax.scatter(w_zero,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 3)
                    ax.scatter(w_zero,g_zero,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, marker = 'X',zorder = 3)            # plot point of tangency
            
            # fix viewing limits on panel
            ax.set_xlim([-wmax,wmax])
            ax.set_ylim([min(-0.3,min(g_plot) - ggap),max(max(g_plot) + ggap,0.3)])
            
            # add horizontal axis
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            
            # label axes
            ax.set_xlabel('$w$',fontsize = 12)
            ax.set_ylabel('$g(w)$',fontsize = 12,rotation = 0,labelpad = 12)
            
            # set tickmarks
            ax.set_xticks(-np.arange(-round(wmax), round(wmax) + 1, 1.0))
            ax.set_yticks(np.arange(round(min(g_plot) - ggap), round(max(g_plot) + ggap) + 1, 1.0))

            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

        return(anim)
    
    
    ######## secant method #########
    # run secant method
    def run_secant_method(self):
        # get initial point
        w2 = self.w_init
        
        # create second point nearby w_old
        w1 = w2 - 0.5
        g2 = self.g(w2)
        g1 = self.g(w1)
        if g1 > g2:
            w1 = w2 + 0.5
        
        # setup container for history
        self.w_hist = []
        self.w_hist.append(w2)
        self.w_hist.append(w1)
        
        # start loop
        w_old = np.inf
        j = 0
        while abs(w1 - w2) > 10**-5 and j < self.max_its:  
            # plug in value into func and derivative
            g1 = float(self.grad(w1))
            g2 = float(self.grad(w2))
                        
            # take newtons step
            w = w1 - g1*(w1 - w2)/(g1 - g2 + 10**-6)
            
            # record
            self.w_hist.append(w)
            
            # update old w and index
            j+=1
            w2 = w1
            w1 = w
    
    # animate the method
    def draw_it_secant(self,**args):
        # get new initial point if desired
        if 'w_init' in args:
            self.w_init = float(args['w_init'])
            
        # take in user defined maximum number of iterations
        if 'max_its' in args:
            self.max_its = float(args['max_its'])
            
        # initialize figure
        fig = plt.figure(figsize = (6,6))
        artist = fig
        ax = fig.add_subplot(111)

        # generate function for plotting on each slide
        w_plot = np.linspace(-3.1,3.1,200)
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.1
        w_vals = np.linspace(-2.5,2.5,50)
        width = 1
        
        # run secant method
        self.w_hist = []
        self.run_secant_method()
        
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
        print ('starting animation rendering...')
        def animate(t):
            ax.cla()
            k = math.floor((t+1)/float(2))
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # plot function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 2)                           # plot function
            
            # plot initial point and evaluation
            if k == 0:
                w_val = self.w_init
                g_val = self.g(w_val)
                ax.scatter(w_val,g_val,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 2)            # plot point of tangency
                ax.scatter(w_val,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 2, marker = 'X')
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
                    
                    ax.scatter(w_val,g_val,s = 90,c = self.colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    ax.scatter(w_val,0,s = 90,facecolor = self.colorspec[j],marker = 'X',edgecolor = 'k',linewidth = 0.7, zorder = 2)
            
            # plot surrogate function and travel-to point
            if k > 0 and k < len(self.w_hist):    
                # grab historical weights, form associated secant line
                w2 = self.w_hist[k-1]
                w1 = self.w_hist[k]
                g2 = self.g(w2)
                g1 = self.g(w1)
                grad2 = self.grad(w2)
                grad1 = self.grad(w1)

                # determine width of plotting area for second order approximator
                width = 0.5
                g_hess_val = (grad1 - grad2)/(w1 - w2)
                if g_hess_val < 0:
                    width = - width
            
                # setup quadratic formula params
                a = 0.5*g_hess_val
                b = grad1 - 2*0.5*g_hess_val*w1
                c = 0.5*g_hess_val*w1**2 - grad1*w1 - width
            
                # solve for zero points
                wa = (-b + math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)
                wb = (-b - math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)

                # compute second order approximation
                wrange = np.linspace(wa,wb, 100)
                h = g1 + grad1*(wrange - w1) + 0.5*g_hess_val*(wrange - w1)**2 
                ax.plot(wrange,h,color = 'b',linewidth = 2,zorder = 1)      # plot approx

                # plot intersection points
                ax.scatter(w2, g2, s = 100, c='m',edgecolor = 'k',linewidth = 0.7,zorder = 3)
                ax.scatter(w1, g1, s = 100, c='m',edgecolor = 'k',linewidth = 0.7,zorder = 3)

                # plot next point learned from surrogate
                if np.mod(t,2) == 0:
                    # create next point information
                    w_zero = w1 - grad1/(g_hess_val + 10**-5)
                    g_zero = self.g(w_zero)
                    h_zero = g1 + grad1*(w_zero - w1) + 0.5*g_hess_val*(w_zero - w1)**2

                    # draw dashed linen connecting the three
                    vals = [0,h_zero,g_zero]
                    vals = np.sort(vals)

                    s = np.linspace(vals[0],vals[2])
                    o = np.ones((len(s)))
                    ax.plot(o*w_zero,s,'k--',linewidth=1)

                    # draw intersection at zero and associated point on cost function you hop back too
                    ax.scatter(w_zero,h_zero,s = 100,c = 'k', zorder = 3)
                    ax.scatter(w_zero,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 3, marker = 'X')
                    ax.scatter(w_zero,g_zero,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
            # fix viewing limits
            ax.set_xlim([-3,3])
            ax.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)

            # place title
            ax.set_title('Secant method',fontsize = 15)

            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=2*len(self.w_hist), interval=2*len(self.w_hist), blit=True)

        return(anim)  