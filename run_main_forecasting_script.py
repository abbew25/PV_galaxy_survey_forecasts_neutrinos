# Author: Abbe Whitford 


# Script to run main_PV_forecasts.py without altering that code.


import os 
import numpy as np
import pickle
from rich.console import Console
from enum import StrEnum
cosmo_variable = StrEnum("variable", "H0 As Och2 Obh2 mnu Neff n_s b_g r_g sigma_u sigma_g")

#--------------------------------------------------------------------------------------------------------------------------------------


include_Planck_base_LCDM_2018 = True            # set to True or False to include estimate of the Fisher information from Planck in forecast  
plot_ellipses = True                            # plot confidence ellipses of sum of neutrino masses with each other parameter in Fisher matrix
write_data_to_files = True                      # writes the fisher matrix in each redshift bin and final covariance matrix to a file + other information relating to forecast.
MCMC_chains_option = 2                          # flag for what information to include from Planck MCMC chains 
                                                # 1 = Planck PlikHM 2018 TTTEEE + lowE + lowl, 
                                                # 2 = Planck PlikHM 2018 TTTEEE + lowE + lowl + lensing (no lensing if N_eff is free parameter though)
write_power_spectra_to_files = True             # write the derivatives of the power spectra to files w.r.t. cosmo params to save recalculating them later.
                                                # Note: these files can be large, and need to be recomputed if you change the cosmology (code does this automatically).                                                                                   

data_from_MCMC_chains = 'No planck data included'
if MCMC_chains_option == 1:
    data_from_MCMC_chains = 'Planck 2018 PlikHM TT + TE + EE + lowE + lowl'
elif MCMC_chains_option == 2:
    data_from_MCMC_chains = 'Planck 2018 PlikHM TT + TE + EE + lowE + lowl + lensing /Planck 2018 PlikHM TT + TE + EE + lowE + lowl (no lensing if N_eff is free)'
else:
    raise Exception('Flags for including information from Planck may only be: 1, 2')


save_file_folder = str(os.getcwd()) + '/example_results/' # specify path for saving information if write_data_to_files = True 

divide_by_h3_on_read_in = False                # don't modify this, should probably never be set to TRUE
multiply_by_h3_on_read_in = True               # for number density files - set to True if the number density of objects in h^3/Mpc^3
multiply_by1eminus6_on_read_in = True          # for number density files - number densities in example files is multiplied by 1e6 to save precision
                                               # set to True to mutiply by 1e-6 on read in of files 

#-------------------------------------------------------------------------------------------------------------------------------------------------
#set up fiducial cosmological parameters and some other stuff: need to define H0, h, As, Obh, M_nu, Och, Om, tau and ns and the neutrino hierarchy 

d_a = 0.0001                                    # size of step in scalefactor to get growth rate using 0.5 * P/a * dP/da (best not to change this)
H0 = 67.32                                      # Hubble constant
h = H0/100.0                                    # little h
As = np.exp(3.0448)*1e-10                       # normalization of the linear matter power spectrum
Obh = 0.022383                                  # normalized density of baryonic matter * h^2 (H0/100.0 km/s/MPC)
m_nu = 0.058                                    # sum of masses of neutrinos (eV)
Och = 0.12011                                   # normalized density of cold dark matter * h^2 (H0/100.0 km/s/MPC)
Om = (Obh + Och + m_nu/93.14)/(h**2)                  # density of matter (baryonics + neutrinos + cold dark matter)
tau = 0.0543                                    # optical depth to reionization
ns = 0.96605                                    # the spectral index of the matter power spectrum


q_perp = 1.0                                    # AP effect perpendicular distortion parameter (don't change this!) 
q_parallel = 1.0                                # AP effect parallel distortion parameter (don't change this!)


#n_h = 'degenerate'
n_h = 'normal'                                  # neutrino hierarchy - options are 'normal' or 'inverted' or 'degenerate'
#n_h = 'inverted'

# nuisance parameters: setting the values  -------------------------------------- 

r_g = 1.0                                       # correlation between the matter and velocity power spectra
b_g = 1.34                                      # galaxy bias (scale-independent, but redshift dependent)
sigma_uh = 13.0                                 # parameter related to the nonlinear motion of galaxies, related to FoG Effect
sigma_gh = 4.24                                 # parameter related to the velocity dispersion between pairs of galaxies, related to FoG Effect 
linear = True                                   # use linear CLASS power spectra or use non-linear CLASS (halofit) power spectra
                                            
lin_or_nonlin = 'linear'
if not linear:
    lin_or_nonlin = 'nonlinear'

# step sizes for differentiation w.r.t. H0, As, mnu, Obh2, Och2 (best values for these **may** depend on the values of cosmological parameters but
# these have been found to work well for cosmology defined above)
del_H0 = 0.54*5
del_Obh2 = 0.00006
del_As = 0.029*1e-10
del_Och2 = 0.00015
del_Mnu = 0.001
del_Neff = 0.025
del_ns = 0.01

# other settings: 

c = 299792.458                                  # The speed of light in km/s ( probably best not to alter this!!!! )
num_redshift_bins = 5                           # number of redshift bins - however many separate bins you want                                             
zmax = 0.50                                     # maximum redshift to include information from in forecasting analysis
zmin = 0.00                                     # minimum redshift to include information from in forecasting analysis
kmax = 0.2*h                                    # The maximum k mode to include information from in forecasting analysis
kmin_overall = 0                                # don't change this - value of kmin_overall is determined later by forecasting script.
kmin_overall_index = 0                          # don't alter this.
numk_vals = 1000                                # number of k points to consider (optional)

# We need to know the independent survey area for our redshift and peculiar velocity surveys and then the area of overlap between the surveys 
# (redshift survey area only first, then PV survey area only second, then overlap, in units of pi*steradians)
survey_area = [0.0, 0.0, 1.3186] 
error_rand = 300.0                              # The observational error due to random non-linear velocities 
error_dist = 0.20                               # The percentage error on the distance indicator (Typically 0.05 - 0.10 for SNe IA, 
                                                # 0.2 or more for Tully-Fisher or Fundamental Plane) 

#  A list of flags for the parameters we are interested in varying in the analysis/free parameters - comment out parameters you don't want
# Ordering of parameters here is best and shouldn't be altered, just comment out lines for parameters you do not want to include in forecasting analysis.
Data = [                                        
    0, #cosmo_variable.H0,                  # H0
    1, #cosmo_variable.As,                  # As
    2, #cosmo_variable.Obh2,                # Obh
    3, #cosmo_variable.Och2,                # Och
    4, #cosmo_variable.mnu,                 # mnu
    13, #cosmo_variable.Neff,                # N_eff (effective number of neutrino species)  
    14, #cosmo_variable.n_s,                 # ns
    7, #cosmo_variable.b_g,                 # galaxy bias b_g
    8, #cosmo_variable.r_g,                 # r_g
    9, #cosmo_variable.sigma_u,             # sigma_u
    10 #cosmo_variable.sigma_g,             # sigma_g
        ]  


nparams = len(Data)                             # The number of free parameters (don't alter this)
verbosity = 1                                   # How much output to give: = 0 only gives less information, =1 gives more information
dm2_atm = 2.5e-3                                # setting the atmospheric neutrino mass splitting (best not to alter this too much)
dm2_sol = 7.45e-5                               # setting the solar neutrino mass splitting (best not to alter this too much)  
num_mus = 100                                   # number of mus to use when integrating over mu in z_eff_integrand and mu_integrand with trapezoid rule 
                                                # (100 is good enough, a large number of mus may significantly slows down the code)   



nbar_file = [

str(os.getcwd()) + r'/example_redshift_number_densities.csv', # number density of redshifts first in list
str(os.getcwd()) + r'/example_PV_number_densities.csv'        # number density of PVs second 

]

# DESI BGS with PV - short names for density files being read in, 
# to use when writing power spectra data to files (if write_power_spectra_to_files = True) so that one has reference to which
# redshifts the power spectra were calculated for 
nbar_file_brev = ['exampleredshifts', 'examplePVs'] 



forecasting_params = {

    'include_Planck_base_LCDM_2018': include_Planck_base_LCDM_2018,
    'plot_ellipses': plot_ellipses,
    'write_data_to_files': write_data_to_files,
    'write_power_spectra_to_files': write_power_spectra_to_files,
    'MCMC_chains_option': MCMC_chains_option,
    'data_from_MCMC_chains': data_from_MCMC_chains,
    'save_file_folder': save_file_folder,
    'divide_by_h3_on_read_in': divide_by_h3_on_read_in,
    'multiply_by_h3_on_read_in': multiply_by_h3_on_read_in,
    'multiply_by1eminus6_on_read_in': multiply_by1eminus6_on_read_in,
    'd_a': d_a,      
    'H0': H0,
    'h': h,
    'As': As,
    'Obh': Obh,
    'm_nu': m_nu,
    'Och': Och,
    'Om': Om,
    'tau': tau,
    'ns': ns,
    'q_perp': q_perp,
    'q_parallel': q_parallel,
    'n_h': n_h,
    'r_g': r_g,
    'b_g': b_g,
    'sigma_uh': sigma_uh,
    'sigma_gh': sigma_gh,
    'linear': linear,
    'del_H0': del_H0,
    'del_Obh2': del_Obh2,
    'del_As': del_As,
    'del_Och2': del_Och2,
    'del_Mnu': del_Mnu,
    'del_Neff': del_Neff,
    'del_ns': del_ns,
    'lin_or_nonlin': lin_or_nonlin,
    'c': c,
    'num_redshift_bins': num_redshift_bins,                                         
    'zmax': zmax, 
    'zmin': zmin, 
    'kmax': kmax, 
    'kmin_overall': kmin_overall,
    'kmin_overall_index': kmin_overall_index, 
    'numk_vals': numk_vals,
    'survey_area':survey_area,                                  
    'error_rand': error_rand, 
    'error_dist': error_dist,
    'Data': Data,
    'nparams': nparams,
    'verbosity': verbosity,
    'dm2_atm': dm2_atm,
    'dm2_sol': dm2_sol,
    'num_mus': num_mus,
    'nbar_file': nbar_file,
    'nbar_file_brev': nbar_file_brev

}


pickle.dump(forecasting_params, open('forecasting_params.p', 'wb'))

from subprocess import call
call(["python", "main_PV_forecasts.py"])

console = Console(style='bold green')

console.log('Finished running forecasts ')
