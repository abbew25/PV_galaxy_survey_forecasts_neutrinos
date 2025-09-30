# Forecasting constraints for neutrino mass with peculiar velocity surveys

## About
This repository contains code to produce Fisher information forecasts for galaxy surveys to constrain the sum of neutrino masses. The code allows you to produce forecasts for surveys with redshift information, peculiar velocity information, or a survey that has both.

This code is heavily inspired by/based off the code here: https://github.com/CullanHowlett/PV_fisher

## Technologies required

This code has been updated to run on python 3.11. You will need to also install the CLASS package, see instructions at https://lesgourg.github.io/class_public/class.html.
Other python packages required used include: scipy, cython, numpy, pandas, os, time, pickle, matplotlib, loguru, rich, pydantic. 

## How to use 

The main code to produce forecasts ```main_PV_forecasts.py``` can be run without modifying this file, by simplying running
``` 
python run_main_forecasting_script.py
```
from your terminal. This produces forecasts in as many redshift bins as desired and writes the fisher matrices and covariance matrices for each bin to a file, if desired. It also 
adds the fisher matrices together and produces a forecast for information across all bins. If you want to add an estimate of the fisher information from the Planck 2018 results, this can also be included in the Fisher forecast (the fisher information estimate for Planck will be added to the total fisher information across all redshift bins).

Edit the settings in ```run_main_forecasting_script.py``` to change the forecasting settings.

If you want a forecast for cosmological parameters where nuisance parameters are treated as separate free parameters in different redshift bins, you can use the script
```read_in_matrices_get_forecasts.py```.

You need to specify some information correctly in the script, and you need to already have fisher matrices for the different redshift bins written to a file (which can be achieved with ```run_main_forecasting_script.py``` for a survey). Then when you run ```read_in_matrices_get_forecasts.py```, it will treat parameters like the galaxy bias, as a different parameter in separate redshift bins for your forecasts (rather than as the same parameter across all bins, by adding the information on the bias in each bin together).

There are example files above for the required number density of objects in the redshift survey and PV survey, and the example_results folder has an example of the results from running my codes with the current settings, with the example number density files.

To run forecasts, all you need are files with the number density of redshifts and peculiar velocities for the surveys you are forecasting for as a function of redshift, and you need to know the sky area for your surveys and the overlap between your redshift and peculiar velocity survey. Then just choose the rest of the settings you want in ```run_main_forecasting_script.py```.

## More information 

A blog post about the project I am working related to this code can be found here: https://astrolaureate.github.io/projects/neutrinospeculiarvelocities 

I also used this code to produce the results in this work https://arxiv.org/abs/2112.10302 

