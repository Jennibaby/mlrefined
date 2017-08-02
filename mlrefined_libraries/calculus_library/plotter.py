import sys
sys.path.append('../')
from mlrefined_libraries import basics_library as baslib
import numpy as np

# a short function for plotting function and derivative values over a large range for input function g
def ad_derval_plot(MyTuple,g):
    # specify range of input for our function and its derivative
    w = np.linspace(-10,10,1000)    

    # define function/derivative Evals objects for each initial value in w_range
    valder_objs = [MyTuple(val = u) for u in w]

    # collect function and derivative values to plot
    results = [g(w) for w in valder_objs]
    g = [r.val for r in results]
    dgdw = [r.der for r in results]

    # generate original function
    function_table = np.stack((w,g), axis=1) 

    # generate derivative function
    derivative_table = np.stack((w,dgdw), axis=1) 

    # use custom plotter to show both functions
    baslib.basics_plotter.double_plot(table1 = function_table, table2 = derivative_table,plot_type = 'continuous',xlabel = '$w$',ylabel_1 = '$g(w)$',ylabel_2 = r'$\frac{\mathrm{d}}{\mathrm{d}w}g(w)$',fontsize = 14)

# plotter for function and derivative equations
def derval_eq_plot(g,dgdw):
    # specify range of input for our function and its derivative
    w = np.linspace(-10,10,1000)   

    # make real function / derivative values
    g_vals = g(w)
    dgdw_vals = dgdw(w)

    # generate original function
    function_table = np.stack((w,g_vals), axis=1) 

    # generate derivative function
    derivative_table = np.stack((w,dgdw_vals), axis=1) 

    # use custom plotter to show both functions
    baslib.basics_plotter.double_plot(table1 = function_table, table2 = derivative_table,plot_type = 'continuous',xlabel = '$w$',ylabel_1 = '$g(w)$',ylabel_2 = r'$\frac{\mathrm{d}}{\mathrm{d}w}g(w)$',fontsize = 14)