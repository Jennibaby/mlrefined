# Machine Learning Refined: Python Jupyter notebook collection

See our [blog here](https://jermwatt.github.io/mlrefined/index.html) for interactive versions of the notebooks in this repo.  These posts describe a range of topics in machine learning / deep learning including a wide variety of topics in supervised learning, mathematical optimization and automatic differentiation / the back propagation algorithm, and reinforcement learning.

In order to effectively run the Jupyter notebooks contained in this repo on your own machine we strongly recommend using the Anaconda Python 3 distribution [which can be downloaded here](https://www.anaconda.com/download/) since the default install contains most of the library dependencies used here as well as as Jupyter notebook, with the exception of autograd [which can be cloned here](https://github.com/HIPS/autograd).

To re-run the animations contained withiin these jupyter notebooks you can initialize your jupyter session with the following adjusted command in place of the standard 'jupyter notebook' initialization command - which increases the rate you can plot images to a jupyter notebook cell

jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000 
        
- - -
This repository contains various supplementary Jupyter notebooks, Python and MATLAB files, presentations associated with the textbook Machine Learning Refined (Cambridge University Press). Visit [http://www.mlrefined.com](http://www.mlrefined.com) for free chapter downloads and tutorials, and [our Amazon site here for details regarding a hard copy of the text](https://www.amazon.com/Machine-Learning-Refined-Foundations-Applications/dp/1107123526/ref=sr_1_1?ie=UTF8&qid=1471025359&sr=8-1&keywords=machine+learning+refined).
