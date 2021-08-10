# Forecasting constraints for neutrino mass with peculiar velocity surveys

## About
This repository contains code to produce Fisher information forecasts for galaxy surveys to constrain the sum of neutrino masses. The code allows you to produce forecasts for surveys with redshift information, peculiar velocity information, or a survey that has both.

This code is heavily inspired by/based off the code here: https://github.com/CullanHowlett/PV_fisher

## Technologies required

This code was created with Python 3 (3.7.6). You will need to also install the CLASS package, see instructions at https://lesgourg.github.io/class_public/class.html.
Other python packages required used include: scipy, numpy, pandas, os, time, pickle, matplotlib.

## How to use 

The main code to produce forecasts ```main_PV_forecasts.py``` can be run without modifying this file, by simplying running
``` python run_main_forecasting_script.py
```
from your terminal. This produces forecasts in as many redshift bins as desired and writes the fisher matrices and covariance matrices for each bin to a file, if desired. It also 
adds the fisher matrices together and produces a forecast for information across all bins. If you want to add an estimate of the fisher information from the Planck 2018 results, this can also be included in the Fisher forecast (the fisher information estimate for Planck will be added to the total fisher information across all redshift bins).

Edit the settings in ```run_main_forecasting_script.py``` to change the forecasting settings.

If you want a forecast for cosmological parameters where nuisance parameters are treated as separate free parameters in different redshift bins, you can use the script
```read_in_matrices_get_forecasts.py```.

You need to specify some information correctly in the script, and you need to already have fisher matrices for the different redshift bins written to a file (which can be achieved with ```run_main_forecasting_script.py``` for a survey). Then when you run ```read_in_matrices_get_forecasts.py```, it will treat parameters like the galaxy bias, as a different parameter in separate redshift bins for your forecasts.



