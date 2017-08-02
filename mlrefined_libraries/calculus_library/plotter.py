import sys
sys.path.append('../')
from mlrefined_libraries import basics_library as baslib
import numpy as np

# a short function for plotting function and derivative values over a large range for input function g
def ad_derval_plot(MyTuple,g,**kwargs):
    # specify range of input for our function and its derivative
    w = np.linspace(-10,10,1000) 
    if 'w' in kwargs:
        w = kwargs['w']
  
    # recurse to create higher order derivative object
    order = 1
    if 'order' in kwargs:
        order = kwargs['order']
    
    # initialize objects
    valder_objs = []
    for u in w:
        # loop over and make deep object for higher order derivatives
        s = MyTuple(val = u)
        for i in range(order-1):
            s = MyTuple(val = s)
        valder_objs.append(s)

    # collect function and derivative values to plot
    results = [g(w) for w in valder_objs]
    
    # loop over and collect final derivative value
    g = []
    dgdw = []
    for r in results:
        val = r.val
        der = r.der
        for i in range(order-1):
            val = val.val
            der = der.der
        g.append(val)
        dgdw.append(der)

    # generate original function
    function_table = np.stack((w,g), axis=1) 

    # generate derivative function
    derivative_table = np.stack((w,dgdw), axis=1) 

    # use custom plotter to show both functions
    baslib.basics_plotter.double_plot(table1 = function_table, table2 = derivative_table,plot_type = 'continuous',xlabel = '$w$',ylabel_1 = '$g(w)$',ylabel_2 = r'$\frac{\mathrm{d}^' + str(order) +  '}{\mathrm{d}w^' + str(order) +  '}g(w)$',fontsize = 14)

# plotter for function and derivative equations
def derval_eq_plot(g,dgdw,**kwargs):
    # specify range of input for our function and its derivative
    w = np.linspace(-10,10,1000) 
    if 'w' in kwargs:
        w = kwargs['w']
        
    # make real function / derivative values
    g_vals = g(w)
    dgdw_vals = dgdw(w)

    # generate original function
    function_table = np.stack((w,g_vals), axis=1) 

    # generate derivative function
    derivative_table = np.stack((w,dgdw_vals), axis=1) 

    # use custom plotter to show both functions
    baslib.basics_plotter.double_plot(table1 = function_table, table2 = derivative_table,plot_type = 'continuous',xlabel = '$w$',ylabel_1 = '$g(w)$',ylabel_2 = r'$\frac{\mathrm{d}}{\mathrm{d}w}g(w)$',fontsize = 14)