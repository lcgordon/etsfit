.. etsfit documentation master file, created by
   sphinx-quickstart on Fri Oct  6 08:45:52 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to etsfit's documentation!
==================================


This is a project to quickly run MCMC on early time supernova light curves. It was originally designed for TESS data on Type Ia's, but should 
function as a plug-and-play way to drop in a new log likelihood function for any model and any data and get a uniform set of plots out. 

.. toctree::
   :maxdepth: 1
   :caption: etsfit main: 

   source/etsfit.etsfit
   source/modules

.. toctree::
   :maxdepth: 1
   :caption: Main Submodules:

   source/etsfit.utils.default_plots
   source/etsfit.utils.gp_plots
   source/etsfit.utils.MCMC
   source/etsfit.utils.parameter_retrieval
   source/etsfit.utils.utilities


.. toctree::
   :maxdepth: 1
   :caption: other links
   
   source/page

   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
