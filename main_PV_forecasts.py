# import libraries
import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp 
from scipy.integrate import simps, quad
from scipy.interpolate import CubicSpline
import pandas as pd
from classy import Class
import time 
import pickle 
import os 
from matplotlib.patches import Ellipse
import functions_PV_forecasts as cosmo # functions for derivatives 
import numpy.typing as npt 
from loguru import logger
from rich.console import Console
from enum import StrEnum
skiprows = 1  # number of rows skipped when reading in number density files in read_nz()
cosmo_variable = StrEnum("variable", "H0 As Och2 Obh2 mnu Neff n_s b_g r_g sigma_u sigma_g")

mapping_index_dict = { 
    cosmo_variable.H0: 0,
    cosmo_variable.As: 1,
    cosmo_variable.Obh2: 2,
    cosmo_variable.Och2: 3,
    cosmo_variable.mnu: 4,
    cosmo_variable.Neff: 13,
    cosmo_variable.n_s: 14,
    cosmo_variable.b_g: 7,
    cosmo_variable.r_g: 8,
    cosmo_variable.sigma_u: 9,
    cosmo_variable.sigma_g: 10
} 
 
 
#--------------------------------------------------------------------------------------------------------------------------------------
# importing settings 
console = Console(style="blue")

include_Planck_base_LCDM_2018 = pickle.load( open('forecasting_params.p', 'rb') )['include_Planck_base_LCDM_2018']
plot_ellipses = pickle.load( open('forecasting_params.p', 'rb') )['plot_ellipses']
write_data_to_files = pickle.load( open('forecasting_params.p', 'rb') )['write_data_to_files']
write_power_spectra_to_files = pickle.load( open('forecasting_params.p', 'rb') )['write_power_spectra_to_files']
MCMC_chains_option = pickle.load( open('forecasting_params.p', 'rb') )['MCMC_chains_option']                    
data_from_MCMC_chains = pickle.load( open('forecasting_params.p', 'rb') )['data_from_MCMC_chains']         
save_file_folder = pickle.load( open('forecasting_params.p', 'rb') )['save_file_folder']                                         
divide_by_h3_on_read_in = pickle.load( open('forecasting_params.p', 'rb') )['divide_by_h3_on_read_in']
multiply_by_h3_on_read_in = pickle.load( open('forecasting_params.p', 'rb') )['multiply_by_h3_on_read_in']
multiply_by1eminus6_on_read_in = pickle.load( open('forecasting_params.p', 'rb') )['multiply_by1eminus6_on_read_in']
d_a = pickle.load( open('forecasting_params.p', 'rb') )['d_a']
H0 = pickle.load( open('forecasting_params.p', 'rb') )['H0']
h = pickle.load( open('forecasting_params.p', 'rb') )['h']
As = pickle.load( open('forecasting_params.p', 'rb') )['As']
Obh = pickle.load( open('forecasting_params.p', 'rb') )['Obh']
m_nu = pickle.load( open('forecasting_params.p', 'rb') )['m_nu']
Och = pickle.load( open('forecasting_params.p', 'rb') )['Och']
Om = pickle.load( open('forecasting_params.p', 'rb') )['Om']
tau = pickle.load( open('forecasting_params.p', 'rb') )['tau']
ns = pickle.load( open('forecasting_params.p', 'rb') )['ns']
q_perp = pickle.load( open('forecasting_params.p', 'rb') )['q_perp']
q_parallel = pickle.load( open('forecasting_params.p', 'rb') )['q_parallel']
n_h = pickle.load( open('forecasting_params.p', 'rb') )['n_h']
r_g = pickle.load( open('forecasting_params.p', 'rb') )['r_g']
b_g = pickle.load( open('forecasting_params.p', 'rb') )['b_g']
sigma_uh = pickle.load( open('forecasting_params.p', 'rb') )['sigma_uh']
sigma_gh = pickle.load( open('forecasting_params.p', 'rb') )['sigma_gh']
linear = pickle.load( open('forecasting_params.p', 'rb') )['linear']
del_H0 = pickle.load( open('forecasting_params.p', 'rb') )['del_H0']
del_Obh2 = pickle.load( open('forecasting_params.p', 'rb') )['del_Obh2']
del_As = pickle.load( open('forecasting_params.p', 'rb') )['del_As']
del_Och2 = pickle.load( open('forecasting_params.p', 'rb') )['del_Och2']
del_Mnu = pickle.load( open('forecasting_params.p', 'rb') )['del_Mnu']
del_Neff = pickle.load( open('forecasting_params.p', 'rb') )['del_Neff']
del_ns = pickle.load( open('forecasting_params.p', 'rb') )['del_ns']
lin_or_nonlin = pickle.load( open('forecasting_params.p', 'rb') )['lin_or_nonlin']
c = pickle.load( open('forecasting_params.p', 'rb') )['c']
num_redshift_bins = pickle.load( open('forecasting_params.p', 'rb') )['num_redshift_bins']                                             
zmax = pickle.load( open('forecasting_params.p', 'rb') )['zmax']
zmin = pickle.load( open('forecasting_params.p', 'rb') )['zmin']
kmax = pickle.load( open('forecasting_params.p', 'rb') )['kmax']
kmin_overall = pickle.load( open('forecasting_params.p', 'rb') )['kmin_overall']
kmin_overall_index = pickle.load( open('forecasting_params.p', 'rb') )['kmin_overall_index']
numk_vals = pickle.load( open('forecasting_params.p', 'rb') )['numk_vals']
survey_area = pickle.load( open('forecasting_params.p', 'rb') )['survey_area']                          
error_rand = pickle.load( open('forecasting_params.p', 'rb') )['error_rand']
error_dist = pickle.load( open('forecasting_params.p', 'rb') )['error_dist'] 
Data = pickle.load( open('forecasting_params.p', 'rb') )['Data']
nparams = len(Data)
verbosity = pickle.load( open('forecasting_params.p', 'rb') )['verbosity']
dm2_atm = pickle.load( open('forecasting_params.p', 'rb') )['dm2_atm']
dm2_sol = pickle.load( open('forecasting_params.p', 'rb') )['dm2_sol']
num_mus = pickle.load( open('forecasting_params.p', 'rb') )['num_mus']
nbar_file = pickle.load( open('forecasting_params.p', 'rb') )['nbar_file']
nbar_file_brev = pickle.load( open('forecasting_params.p', 'rb') )['nbar_file_brev']


#--------------------------------------------------------------------------------------------------------------------------    


# initializing some variables for reading in the power spectra - initializing so the variables are global
pmm_array, pmt_array, ptt_array, delta_ks, kvals, growth_rate_array = [], [], [], [], [], []

pmm_0_kmin = 0.0

N_redshifts_arr, N_bar_arr, r_array, delta_r_array, redshift_dist_spline = [], [], [], [], []

dPdH_arr_gg, dPdObh2_arr_gg, dPdOch2_arr_gg, dPdMnu_arr_gg, dPdAs_arr_gg, dPdNeff_arr_gg, dPdns_arr_gg  = [], [], [], [], [], [], []
dPdH_arr_gu, dPdObh2_arr_gu, dPdOch2_arr_gu, dPdMnu_arr_gu, dPdAs_arr_gu, dPdNeff_arr_gu, dPdns_arr_gu  = [], [], [], [], [], [], []
dPdH_arr_uu, dPdObh2_arr_uu, dPdOch2_arr_uu, dPdMnu_arr_uu, dPdAs_arr_uu, dPdNeff_arr_uu, dPdns_arr_uu  = [], [], [], [], [], [], []


galaxy_bias_array = []

planck_18_information = []


data_analysis_dictionary = {
        'kmax/h:': kmax/h,
        'central param vals:': [H0, As*1e9, Obh, Och, m_nu, r_g, b_g, sigma_gh, sigma_uh, ns, tau],
        'step_sizes_cosmo:': [del_H0, del_As*1e9, del_Obh2, del_Och2, del_Mnu, del_Neff, del_ns],
        'linear or nonlinear:': lin_or_nonlin,
        '+ Planck data?:': data_from_MCMC_chains,
        'zmin, zmax:': [zmin, zmax],
        'num redshift bins:': num_redshift_bins,
        'number density files:': nbar_file,
        'neutrino hierarchy:': n_h,
        'survey area:': survey_area,
        'error distance indicator:': error_dist,
        'random error due to nonlinear motion:': error_rand,
        'free param flags:': Data
                            }

if write_data_to_files:

        with open((save_file_folder + r'forecast_details.txt'), 'w') as f:
            for k in data_analysis_dictionary:
                print(k, data_analysis_dictionary[k], file=f)
       

fisher_matrices = {}

covariance_matrices = {}

uncertainties_dictionary = {}

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

def shrink_sqr_matrix(sqr_matrix_obj: npt.NDArray, flags: list):
    ''' 
    Function that removed the rows and columns of a square matrix (numpy matrix) if the rows 
    and columns that a diagonal element of the matrix coincides with is zero.
    e.g. 1 2 3 4
         2 1 9 0   ----- >     1 2 4
         4 5 0 9               2 1 0
         4 3 2 1               4 3 1
    
    The third row and column has been removed since M_(2, 2) <= 1e-7
    '''
    a = 0
    b = False 
    for i in np.arange(sqr_matrix_obj.shape[0]):
        if sqr_matrix_obj[i,i] == 0.0:
            a = i
            b = True 
    if b:
        sqr_matrix_obj = np.delete(sqr_matrix_obj, a, 0)
        sqr_matrix_obj = np.delete(sqr_matrix_obj, a, 1)
        flags = np.delete(flags, a, 0)
        sqr_matrix_obj, flags = shrink_sqr_matrix(sqr_matrix_obj, flags)
        return sqr_matrix_obj, flags
    else:
        return sqr_matrix_obj, flags 


# function to get the Planck 2018 MCMC chains for H0, As, Obh2, Och2
def get_Planck18_MCMC(Data_list: list):

    global planck_18_information

    Obh2_index = 0+2
    Och2_index = 1+2
    Mnu__index = 4+2
    H0___index = 28+2
    As___index = 42+2
    ws___index = 0
    Neff_index = 5+2
    ns___index = 7+2

    chain1 = ''
    chain2 = ''
    chain3 = ''
    chain4 = ''

    cwd = os.getcwd()

    if MCMC_chains_option == 1: # TT + TE + EE + lowE + lowl 

        Obh2_index = 0+2
        Och2_index = 1+2
        Mnu__index = 4+2
        H0___index = 28+2
        As___index = 42+2
        ws___index = 0
        ns___index = 6+2

        chain1 = pd.read_csv((cwd + r'/Planck_files/base_mnu/plikHM_TTTEEE_lowl_lowE/base_mnu_plikHM_TTTEEE_lowl_lowE_1.txt'), 
        sep="    ", header=None, engine = 'python')
        chain2 = pd.read_csv((cwd + r'/Planck_files/base_mnu/plikHM_TTTEEE_lowl_lowE/base_mnu_plikHM_TTTEEE_lowl_lowE_2.txt'), 
        sep="    ", header=None, engine = 'python')
        chain3 = pd.read_csv((cwd + r'/Planck_files/base_mnu/plikHM_TTTEEE_lowl_lowE/base_mnu_plikHM_TTTEEE_lowl_lowE_3.txt'), 
        sep="    ", header=None, engine = 'python')
        chain4 = pd.read_csv((cwd + r'/Planck_files/base_mnu/plikHM_TTTEEE_lowl_lowE/base_mnu_plikHM_TTTEEE_lowl_lowE_4.txt'), 
        sep="    ", header=None, engine = 'python')

        if 13 in Data_list:

            ws___index = 0
            Obh2_index = 0+2
            Och2_index = 1+2
            Mnu__index = 4+2
            H0___index = 29+2
            As___index = 43+2
            Neff_index = 5+2
            ns___index = 7+2

            chain1 = pd.read_csv((cwd + r'/Planck_files/base_mnu_nnu/plikHM_TTTEEE_lowl_lowE/base_nnu_mnu_plikHM_TTTEEE_lowl_lowE_1.txt'), 
            sep="    ", header=None, engine = 'python')
            chain2 = pd.read_csv((cwd + r'/Planck_files/base_mnu_nnu/plikHM_TTTEEE_lowl_lowE/base_nnu_mnu_plikHM_TTTEEE_lowl_lowE_2.txt'), 
            sep="    ", header=None, engine = 'python')
            chain3 = pd.read_csv((cwd + r'/Planck_files/base_mnu_nnu/plikHM_TTTEEE_lowl_lowE/base_nnu_mnu_plikHM_TTTEEE_lowl_lowE_3.txt'), 
            sep="    ", header=None, engine = 'python')
            chain4 = pd.read_csv((cwd + r'/Planck_files/base_mnu_nnu/plikHM_TTTEEE_lowl_lowE/base_nnu_mnu_plikHM_TTTEEE_lowl_lowE_4.txt'), 
            sep="    ", header=None, engine = 'python')


    elif MCMC_chains_option == 2: # TT + TE + EE + lowE + lowl + lensing (no lensing if N_eff is free though)

        Obh2_index = 0+2
        Och2_index = 1+2
        Mnu__index = 4+2
        H0___index = 28+2
        As___index = 42+2
        ws___index = 0
        ns___index = 6+2

        chain1 = pd.read_csv((cwd + r'/Planck_files/base_mnu/plikHM_TTTEEE_lowl_lowE_lensing/base_mnu_plikHM_TTTEEE_lowl_lowE_lensing_1.txt'), 
        sep="    ", header=None, engine = 'python')
        chain2 = pd.read_csv((cwd + r'/Planck_files/base_mnu/plikHM_TTTEEE_lowl_lowE_lensing/base_mnu_plikHM_TTTEEE_lowl_lowE_lensing_2.txt'), 
        sep="    ", header=None, engine = 'python')
        chain3 = pd.read_csv((cwd + r'/Planck_files/base_mnu/plikHM_TTTEEE_lowl_lowE_lensing/base_mnu_plikHM_TTTEEE_lowl_lowE_lensing_3.txt'),
        sep="    ", header=None, engine = 'python')
        chain4 = pd.read_csv((cwd + r'/Planck_files/base_mnu/plikHM_TTTEEE_lowl_lowE_lensing/base_mnu_plikHM_TTTEEE_lowl_lowE_lensing_4.txt'), 
        sep="    ", header=None, engine = 'python')

        if 13 in Data_list:

            ws___index = 0
            Obh2_index = 0+2
            Och2_index = 1+2
            Mnu__index = 4+2
            H0___index = 29+2
            As___index = 43+2
            Neff_index = 5+2
            ns___index = 7+2

            chain1 = pd.read_csv((cwd + r'/Planck_files/base_mnu_nnu/plikHM_TTTEEE_lowl_lowE/base_nnu_mnu_plikHM_TTTEEE_lowl_lowE_1.txt'), 
            sep="    ", header=None, engine = 'python')
            chain2 = pd.read_csv((cwd + r'/Planck_files/base_mnu_nnu/plikHM_TTTEEE_lowl_lowE/base_nnu_mnu_plikHM_TTTEEE_lowl_lowE_2.txt'), 
            sep="    ", header=None, engine = 'python')
            chain3 = pd.read_csv((cwd + r'/Planck_files/base_mnu_nnu/plikHM_TTTEEE_lowl_lowE/base_nnu_mnu_plikHM_TTTEEE_lowl_lowE_3.txt'), 
            sep="    ", header=None, engine = 'python')
            chain4 = pd.read_csv((cwd + r'/Planck_files/base_mnu_nnu/plikHM_TTTEEE_lowl_lowE/base_nnu_mnu_plikHM_TTTEEE_lowl_lowE_4.txt'), 
            sep="    ", header=None, engine = 'python')


    else:
        message = f'MCMC chains option is not valid: {MCMC_chains_option}. (in get_Planck18_MCMC()).' 
        logger.error(message)
        # raise Exception('MCMC chains option is not valid. (in get_Planck18_MCMC()).')
    
    chain1 = pd.DataFrame(chain1)
    chain2 = pd.DataFrame(chain2)
    chain3 = pd.DataFrame(chain3)
    chain4 = pd.DataFrame(chain4)


    chain_Obh2 = np.concatenate((chain1[Obh2_index], chain2[Obh2_index], chain3[Obh2_index], chain4[Obh2_index]))
    chain_Och2 = np.concatenate((chain1[Och2_index], chain2[Och2_index], chain3[Och2_index], chain4[Och2_index]))
    chain_H_0_ = np.concatenate((chain1[H0___index], chain2[H0___index], chain3[H0___index], chain4[H0___index]))
    chain_Mnu_ = np.concatenate((chain1[Mnu__index], chain2[Mnu__index], chain3[Mnu__index], chain4[Mnu__index]))
    chain_A_s_ = np.concatenate((chain1[As___index], chain2[As___index], chain3[As___index], chain4[As___index]))*1e-9
    chain_Neff = np.concatenate((chain1[Neff_index], chain2[Neff_index], chain3[Neff_index], chain4[Neff_index]))
    chain_ns__ = np.concatenate((chain1[ns___index], chain2[ns___index], chain3[ns___index], chain4[ns___index]))

    weights1 = chain1[ws___index]
    weights2 = chain2[ws___index]
    weights3 = chain3[ws___index]
    weights4 = chain4[ws___index]

    weights = np.concatenate((weights1, weights2, weights3, weights4))


    chains_all = 0 
    chains_param_indices = []

    for count, item in enumerate(Data_list):

        if item == 0 or item == 1 or item == 2 or item == 3 or item == 4 or item == 13 or item == 14:
            chains_param_indices.append(item)

        if count == 0:
            if item == 0:
                chains_all = chain_H_0_
            elif item == 1:
                chains_all = chain_A_s_
            elif item == 2:
                chains_all = chain_Obh2
            elif item == 3:
                chains_all = chain_Och2
            elif item == 4:
                chains_all = chain_Mnu_
            elif item == 13:
                chains_all = chain_Neff
            elif item == 14:
                chains_all = chain_ns__
            else:
                continue
        else:
            if item == 0:
                chains_all = np.vstack((chains_all, chain_H_0_))
            elif item == 1:
                chains_all = np.vstack((chains_all, chain_A_s_))
            elif item == 2:
                chains_all = np.vstack((chains_all, chain_Obh2))
            elif item == 3:
                chains_all = np.vstack((chains_all, chain_Och2))
            elif item == 4:
                chains_all = np.vstack((chains_all, chain_Mnu_))
            elif item == 13:
                chains_all = np.vstack((chains_all, chain_Neff))
            elif item == 14:
                chains_all = np.vstack((chains_all, chain_ns__))
            else:
                continue

    # get the covariance matrix for these chains
    planck_18_cov_matrix = np.matrix(np.cov(chains_all, aweights = weights))
    # get the Fisher information for these chains from the covariance matrix 
    planck_18_info_matrix = np.linalg.inv(planck_18_cov_matrix)

    # now need to add it to the full fisher information matrix for all parameters
    # that vary in our analysis
    planck_18_information = np.zeros((len(Data_list), len(Data_list)))
    planck_18_information = np.matrix(planck_18_information)

    for i in Data_list:
        for j in Data_list:

            # get the index of i and j in planck_18_information
            indexii = Data_list.index(i)
            indexjj = Data_list.index(j)

            if i in chains_param_indices and j in chains_param_indices:
                indexi = chains_param_indices.index(i)
                indexj = chains_param_indices.index(j)
                planck_18_information[indexii,indexjj] = planck_18_info_matrix[indexi, indexj]
                
            else:
                planck_18_information[indexii,indexjj] = 0
    
    if write_data_to_files:

        with open((save_file_folder + r'planck_information_matrix.txt'), 'w') as f:
            print(data_from_MCMC_chains, file=f)
            print('# Planck Information:', file=f)
            for line in planck_18_information:
                np.savetxt(f, line)
            

        with open((save_file_folder + r'planck_cov_matrix.txt'), 'w') as f:
            print('# Planck covariance matrix:', file=f)
            for line in planck_18_cov_matrix:
                np.savetxt(f, line)
           






# function used to integrate to compute proper distance from Friedmann Equation
def E_z_inverse(z: float):
    '''
    Compute the inverse of the E(z) function (from the first Friedmann Equation).
    '''
    return 1.0/(np.sqrt((Om*(1.0+z)**3)+(1.0-Om)))



def E_z(z: float):
    '''
    Compute E(z) (from the first Friedmann Equation).
    '''
    return (np.sqrt((Om*(1.0+z)**3)+(1.0-Om)))





# function that computes the proper distance as a function of redshift (dimensionless)
def rz(red: npt.Union[float, npt.NDArray]):
    '''
    Calculates the proper radial distance to an object at redshift z for the given cosmological model.
    '''
    try:
        d_com = (c/H0)*quad(E_z_inverse, 0.0, red, epsabs = 5e-5)[0]
        return d_com
    except:
        distances = np.zeros(len(red))  
        for i, z in enumerate(red):
            distances[i] = (c/H0)*quad(E_z_inverse, 0.0, z, epsabs = 5e-5)[0]
        return distances 





# read in the files that get the galaxy number density as a function of redshift 
def read_nz():
    '''
    Read in the files to get the number density of galaxies for the density field
    and the velocity field. 
    This function also computes the distance to each redshift bin we consider, the width of the 
    distance bins and creates a spline for redshift with distance that can be used globally. 
    '''

    start_time = time.time()

    global N_bar_arr
    Nredshiftsarror = 0
    for i in np.arange(2): # redshift file then velocity file

        try: # try to read in the file
            number_density_data = pd.read_csv(r'%s' % (nbar_file[i]), header=None, engine='python', 
            delim_whitespace=True, names = ["n_red", "n_bar"], skiprows = skiprows)
        except: # raise an exception if it cannot be read in
            message = f'File could not be read in: error in (read_nz()). File name: {nbar_file[i]}'
            logger.error(message)
            # raise Exception("File could not be read in: error in (read_nz()).")

        if i == 0: # save the redshifts array 
            Nredshiftsarror = np.array(number_density_data["n_red"])

         # save the number density for vel. and pos. in a list
        n_in_bar_arr = np.array(number_density_data["n_bar"])

        if multiply_by1eminus6_on_read_in:
            n_in_bar_arr = n_in_bar_arr*1e-6

        if divide_by_h3_on_read_in:
            n_in_bar_arr = n_in_bar_arr*(1/(h**3))

        if multiply_by_h3_on_read_in:
            n_in_bar_arr = n_in_bar_arr*(h**3)

        N_bar_arr.append(n_in_bar_arr)
            
    N_bar_arr = np.array(N_bar_arr)
    
    # need to check the number of redshift bins match for velocity and position data 
    if (len(N_bar_arr[0]) != len(N_bar_arr[1])):
        message = f'The length of the redshift bins for the velocity and position density files are not the same: error in (read_nz()).'
        logger.error(message)
        # raise Exception("The length of the redshift bins for the velocity and position density files are not the same: error in (read_nz()).")


    # create a redshift-distance spline to use globally 
    global redshift_dist_spline
    nbins = 500                                 
    redshifts = np.linspace(0, 2.0, nbins)
    distances = rz(redshifts)
    redshift_dist_spline = CubicSpline(redshifts, distances)

    # calculate k min OVERALL - this will be used to define the value of f0 and define fbar(k)
    # as well as b0 value and bbar(k)
    global kmin_overall
    global zmax
    kmin_overall = np.pi/rz(zmax)

    # make the redshift array from the redshift array that was read in, array of 
    # distances, array of delta redshifts
    Nredshiftsarror2 = []
    rarray = []
    deltararray = []

    for i in range(len(Nredshiftsarror)-1):
        Nredshiftsarror2.append((Nredshiftsarror[i+1] + Nredshiftsarror[i])/2.0)
        rarray.append(rz(Nredshiftsarror2[i]))
        deltararray.append((rz(Nredshiftsarror[i+1]) - rz(Nredshiftsarror[i])))

        
    # making our arrays numpy arrays 
    # also saving them to global variables
    global N_redshifts_arr, r_array, delta_r_array
    N_redshifts_arr = np.array(Nredshiftsarror2) 
    r_array = np.array(rarray) 
    delta_r_array = np.array(deltararray)

    # set the last values for each array 
    N_redshifts_arr = np.append(N_redshifts_arr, (zmax + Nredshiftsarror[len(Nredshiftsarror)-1])/2.0 )
    r_array = np.append(r_array, rz(N_redshifts_arr[len(N_redshifts_arr)-1]) )
    delta_r_array = np.append( delta_r_array, rz(zmax) - rz(Nredshiftsarror[len(Nredshiftsarror)-1]) ) 
    
    end_time = time.time()

    console.log('Run time for read_nz(): ', round(end_time - start_time,3), ' seconds.')
    ##################################################################################################################3






# function to get in the power spectra you want + the growth rate at each k value and redshift ,
# and derivative of power spectra w.r.t. all parameters of interest at each k and zval (for H0, obh2, och2, As, mnu)
def read_power():
    '''
    Function to set up the power spectra that are needed,
    compute numerical derivatives that are needed for the Fisher matrix,
    compute the growth rate and galaxy bias as a function of redshift. 
    '''
    start_time = time.time()

    k1 = 1e-4
    k2 = 3.0

    global kvals, pmm_0_kmin
    kvals = np.linspace(k1, k2, numk_vals)
    global kmin_overall_index, kmin_overall
    kmin_overall_index = np.argmin(abs(kvals - kmin_overall))
    original_kmin_overall = kmin_overall
    kmin_overall = kvals[kmin_overall_index]
    while kmin_overall < original_kmin_overall:
        kmin_overall = kvals[kmin_overall_index+1]
        kmin_overall_index = kmin_overall_index+1
    global delta_ks
    delta_ks = kvals[1:len(kvals)] - kvals[0:len(kvals)-1]


    global pmm_array, pmt_array, ptt_array, growth_rate_array, galaxy_bias_array

                                                                                     
    global dPdH_arr_gg, dPdObh2_arr_gg, dPdOch2_arr_gg, dPdMnu_arr_gg, dPdAs_arr_gg, dPdNeff_arr_gg, dPdns_arr_gg
    global dPdH_arr_gu, dPdObh2_arr_gu, dPdOch2_arr_gu, dPdMnu_arr_gu, dPdAs_arr_gu, dPdNeff_arr_gu, dPdns_arr_gu
    global dPdH_arr_uu, dPdObh2_arr_uu, dPdOch2_arr_uu, dPdMnu_arr_uu, dPdAs_arr_uu, dPdNeff_arr_uu, dPdns_arr_uu

    # get the matter /velocity power spectra and growth rate etc. for all the redshifts we need

    # make a string that has the values of parameters 
    CCStr = str(Obh) + '_' + str(m_nu) + '_' + str(H0) + '_' + str(Och) + '_' + str(As*1e9) 
    CCStr = CCStr + '_' + str(sigma_gh) + '_' + str(sigma_uh) + '_' + str(r_g) + '_' + str(b_g) 
    CCStr = CCStr + '_' + str(ns) + '_' + str(tau) + '_' + str(lin_or_nonlin) + '_' + n_h + '_' + nbar_file_brev[0] + '_' + nbar_file_brev[1]
    CCStr = 'powerspectradata/' + CCStr

    cwd = os.getcwd()
    if not os.path.exists((cwd + '/powerspectradata/')): # checking folder path to write files to later exists 
        os.makedirs((cwd + '/powerspectradata/'))


    pmm_0_kmin = []
    pmm_kmin_array = []
    galaxy_bias_array = []

    try: 

        pmm_array = pickle.load(open(CCStr + '_powerspectra.p', 'rb'))['pmm_array']
        pmt_array = pmm_array
        ptt_array = pmm_array

        growth_rate_array = pickle.load(open(CCStr + '_powerspectra.p', 'rb'))['growth_rate_array']

        pmm_kmin_array = pickle.load(open(CCStr + '_powerspectra.p', 'rb'))['pmm_kmin_array']

        pmm_0_kmin = pickle.load(open(CCStr + '_powerspectra.p', 'rb'))['pmm_0_kmin']

    except:

        pmm_array = cosmo.run_class(Obh, Och, H0, As, m_nu, n_h, N_redshifts_arr, k1, k2, numk_vals, dm2_atm, dm2_sol, del_Mnu, m_nu,
        'lin_ks', tau, ns, Obh, Och, H0, As, m_nu, np.linspace(0.0, 1.0, num_mus), lin_or_nonlin, 0.0)[0]
        pmt_array = pmm_array
        ptt_array = pmm_array


        growth_rate_array = cosmo.compute_f_at_many_redshifts(Obh, Och, H0, As, m_nu, n_h, N_redshifts_arr, k1, k2,
        np.linspace(0.0, 1.0, num_mus), numk_vals, d_a, linear, m_nu, Obh, Och, H0, As, m_nu, 'lin_ks', tau, ns, 0.0, delta_mnu_max=del_Mnu)[0]
                
                
        # get the value of the power spectrum at z = 0, kmin (in linear theory specifically)
        pmm_kmin = cosmo.run_class(Obh, Och, H0, As, m_nu, n_h, N_redshifts_arr, k1, k2, numk_vals, dm2_atm, dm2_sol, del_Mnu, m_nu,
        'lin_ks', tau, ns, Obh, Och, H0, As, m_nu, np.linspace(0.0, 1.0, num_mus), 'linear', 0.0)[0]
        pmm_kmin_array = pmm_kmin[kmin_overall_index, :]

        pmm_0_kmin = cosmo.run_class(Obh, Och, H0, As, m_nu, n_h, [0.0], k1, k2, numk_vals, dm2_atm, dm2_sol, del_Mnu, m_nu,
        'lin_ks', tau, ns, Obh, Och, H0, As, m_nu, np.linspace(0.0, 1.0, num_mus), 'linear', 0.0)[0][kmin_overall_index, 0]


        string = CCStr + '_powerspectra.p'

        power_spectra_data_dictionary = {'pmm_array': pmm_array,
                                        'growth_rate_array': growth_rate_array,
                                        'pmm_kmin_array': pmm_kmin_array,
                                        'pmm_0_kmin': pmm_0_kmin, }

        if write_power_spectra_to_files:

            pickle.dump(power_spectra_data_dictionary, open(string, 'wb'))



    for i in np.arange(len(N_redshifts_arr)):

        galaxy_bias = (b_g*(np.sqrt(pmm_0_kmin)/np.sqrt(pmm_kmin_array[i])))
        galaxy_bias_array.append(galaxy_bias)

    galaxy_bias_array = np.array(galaxy_bias_array)

    if growth_rate_array.shape == (num_mus, numk_vals, len(N_redshifts_arr)):
        growth_rate_array = growth_rate_array[0,:,:]

    
    # compute the derivatives of RSPS w.r.t. cosmological parameters 
    central_params = [Obh, m_nu, H0, Och, As, sigma_gh, galaxy_bias_array, r_g, sigma_uh]
    mus = np.linspace(0, 1.0, num_mus)



    if cosmo_variable.H0 in Data:

        
        # make a string with values of parameters + step size of cosmological variable in consideration 

        check_str = CCStr + '_' + 'deltaH0' + '_' + str(del_H0) + '_.p'

        try:

            dPdH_arr_gg = pickle.load( open(check_str, 'rb') )['dPdH_arr_gg']
            dPdH_arr_gu = pickle.load( open(check_str, 'rb') )['dPdH_arr_gu']
            dPdH_arr_uu = pickle.load( open(check_str, 'rb') )['dPdH_arr_uu']

        except:

            dPdH_arr_gg, dPdH_arr_gu, dPdH_arr_uu = cosmo.get_rsp_dP_dx_cosmo_many_redshifts(cosmo_variable.H0, N_redshifts_arr, central_params, del_H0, k1, k2, 
            numk_vals, mus, d_a, n_h, lin_or_nonlin, 'lin_ks', tau, ns, 0.0, dm2_atm, dm2_sol)[0:3]

            dictionary_2_store_derivatives = { 'dPdH_arr_gg': dPdH_arr_gg, 
                                               'dPdH_arr_gu': dPdH_arr_gu,
                                               'dPdH_arr_uu': dPdH_arr_uu  }

            if write_power_spectra_to_files:                                   

                pickle.dump( dictionary_2_store_derivatives, open(check_str, 'wb') )



    if cosmo_variable.As in Data:

        # make a string with values of parameters + step size of cosmological variable in consideration 

        check_str = CCStr + '_' + 'deltaAs' + '_' + str(del_As*1e9) + '_.p'

        try:

            dPdAs_arr_gg = pickle.load( open(check_str, 'rb') )['dPdAs_arr_gg']
            dPdAs_arr_gu = pickle.load( open(check_str, 'rb') )['dPdAs_arr_gu']
            dPdAs_arr_uu = pickle.load( open(check_str, 'rb') )['dPdAs_arr_uu']

        except: 

            
            dPdAs_arr_gg, dPdAs_arr_gu, dPdAs_arr_uu = cosmo.get_rsp_dP_dx_cosmo_many_redshifts(cosmo_variable.As, N_redshifts_arr, central_params, del_As, k1, k2, 
            numk_vals, mus, d_a, n_h, lin_or_nonlin, 'lin_ks', tau, ns, 0.0, dm2_atm, dm2_sol)[0:3]

            dictionary_2_store_derivatives = { 'dPdAs_arr_gg': dPdAs_arr_gg, 
                                               'dPdAs_arr_gu': dPdAs_arr_gu,
                                               'dPdAs_arr_uu': dPdAs_arr_uu  }

            if write_power_spectra_to_files:

                pickle.dump( dictionary_2_store_derivatives, open(check_str, 'wb') )


    if cosmo_variable.Obh2 in Data:

        # make a string with values of parameters + step size of cosmological variable in consideration

        check_str = CCStr + '_' + 'deltaObh2' + '_' + str(del_Obh2) + '_.p'

        try:

            dPdObh2_arr_gg = pickle.load( open(check_str, 'rb') )['dPdObh2_arr_gg']
            dPdObh2_arr_gu = pickle.load( open(check_str, 'rb') )['dPdObh2_arr_gu']
            dPdObh2_arr_uu = pickle.load( open(check_str, 'rb') )['dPdObh2_arr_uu']

        except:


            dPdObh2_arr_gg, dPdObh2_arr_gu, dPdObh2_arr_uu = cosmo.get_rsp_dP_dx_cosmo_many_redshifts(cosmo_variable.Obh2, N_redshifts_arr, central_params, del_Obh2, k1, k2, 
            numk_vals, mus, d_a, n_h, lin_or_nonlin, 'lin_ks', tau, ns, 0.0, dm2_atm, dm2_sol)[0:3]


            dictionary_2_store_derivatives = { 'dPdObh2_arr_gg': dPdObh2_arr_gg, 
                                               'dPdObh2_arr_gu': dPdObh2_arr_gu,
                                               'dPdObh2_arr_uu': dPdObh2_arr_uu  }

            if write_power_spectra_to_files:
                
                pickle.dump( dictionary_2_store_derivatives, open(check_str, 'wb') )


    if cosmo_variable.Och2 in Data:
        
        # make a string with values of parameters + step size of cosmological variable in consideration 

        check_str = CCStr + '_' + 'deltaOch2' + '_' + str(del_Och2) + '_.p'

        try:

            dPdOch2_arr_gg = pickle.load( open(check_str, 'rb') )['dPdOch2_arr_gg']
            dPdOch2_arr_gu = pickle.load( open(check_str, 'rb') )['dPdOch2_arr_gu']
            dPdOch2_arr_uu = pickle.load( open(check_str, 'rb') )['dPdOch2_arr_uu']

        except:

            
            dPdOch2_arr_gg, dPdOch2_arr_gu, dPdOch2_arr_uu = cosmo.get_rsp_dP_dx_cosmo_many_redshifts(cosmo_variable.Och2, N_redshifts_arr, central_params, del_Och2, k1, k2, 
            numk_vals, mus, d_a, n_h, lin_or_nonlin, 'lin_ks', tau, ns, 0.0, dm2_atm, dm2_sol)[0:3]


            
            dictionary_2_store_derivatives = { 'dPdOch2_arr_gg': dPdOch2_arr_gg, 
                                               'dPdOch2_arr_gu': dPdOch2_arr_gu,
                                               'dPdOch2_arr_uu': dPdOch2_arr_uu  }

            if write_power_spectra_to_files:
                
                pickle.dump( dictionary_2_store_derivatives, open(check_str, 'wb') )


    if cosmo_variable.mnu in Data:

        # make a string with values of parameters + step size of cosmological variable in consideration 

        check_str = CCStr + '_' + 'deltaMnu' + '_' + str(del_Mnu) + '_.p'

        try:

            dPdMnu_arr_gg = pickle.load( open(check_str, 'rb') )['dPdMnu_arr_gg']
            dPdMnu_arr_gu = pickle.load( open(check_str, 'rb') )['dPdMnu_arr_gu']
            dPdMnu_arr_uu = pickle.load( open(check_str, 'rb') )['dPdMnu_arr_uu']

        except:


            dPdMnu_arr_gg, dPdMnu_arr_gu, dPdMnu_arr_uu = cosmo.get_rsp_dP_dx_cosmo_many_redshifts(cosmo_variable.mnu, N_redshifts_arr, central_params, del_Mnu, k1, k2, 
            numk_vals, mus, d_a, n_h, lin_or_nonlin, 'lin_ks', tau, ns, 0.0, dm2_atm, dm2_sol)[0:3]

            
            dictionary_2_store_derivatives = { 'dPdMnu_arr_gg': dPdMnu_arr_gg, 
                                               'dPdMnu_arr_gu': dPdMnu_arr_gu,
                                               'dPdMnu_arr_uu': dPdMnu_arr_uu  }

            if write_power_spectra_to_files:
                
                pickle.dump( dictionary_2_store_derivatives, open(check_str, 'wb') )


    if cosmo_variable.Neff in Data:

        # make a string with values of parameters + step size of cosmological variable in consideration 

        check_str = CCStr + '_' + 'deltaNeff' + '_' + str(del_Neff) + '_.p'

        try:

            dPdNeff_arr_gg = pickle.load( open(check_str, 'rb') )['dPdNeff_arr_gg']
            dPdNeff_arr_gu = pickle.load( open(check_str, 'rb') )['dPdNeff_arr_gu']
            dPdNeff_arr_uu = pickle.load( open(check_str, 'rb') )['dPdNeff_arr_uu']

        except:


            dPdNeff_arr_gg, dPdNeff_arr_gu, dPdNeff_arr_uu = cosmo.get_rsp_dP_dx_cosmo_many_redshifts(cosmo_variable.Neff, N_redshifts_arr, central_params, del_Neff, k1, k2, 
            numk_vals, mus, d_a, n_h, lin_or_nonlin, 'lin_ks', tau, ns, 0.0, dm2_atm, dm2_sol)[0:3]

            
            dictionary_2_store_derivatives = { 'dPdNeff_arr_gg': dPdNeff_arr_gg, 
                                               'dPdNeff_arr_gu': dPdNeff_arr_gu,
                                               'dPdNeff_arr_uu': dPdNeff_arr_uu  }

            if write_power_spectra_to_files:
                
                pickle.dump( dictionary_2_store_derivatives, open(check_str, 'wb') )

    if cosmo_variable.n_s in Data:

        check_str = CCStr + '_' + 'delta_ns' + '_' + str(del_ns) + '_.p'

        try:

            dPdns_arr_gg = pickle.load( open(check_str, 'rb') )['dPdns_arr_gg']
            dPdns_arr_gu = pickle.load( open(check_str, 'rb') )['dPdns_arr_gu']
            dPdns_arr_uu = pickle.load( open(check_str, 'rb') )['dPdns_arr_uu']

        except:

            dPdns_arr_gg, dPdns_arr_gu, dPdns_arr_uu = cosmo.get_rsp_dP_dx_cosmo_many_redshifts(cosmo_variable.n_s, N_redshifts_arr, central_params, del_ns, k1, k2, 
            numk_vals, mus, d_a, n_h, lin_or_nonlin, 'lin_ks', tau, ns, 0.0, dm2_atm, dm2_sol)[0:3]

            
            dictionary_2_store_derivatives = { 'dPdns_arr_gg': dPdns_arr_gg, 
                                               'dPdns_arr_gu': dPdns_arr_gu,
                                               'dPdns_arr_uu': dPdns_arr_uu  }

            if write_power_spectra_to_files:
                
                pickle.dump( dictionary_2_store_derivatives, open(check_str, 'wb') )


    end_time = time.time()

    console.log('Run time for read_power(): ', round(end_time-start_time, 3), 
    ' seconds = ',round((end_time-start_time)/60.0, 3) , ' minutes.') 

    ####################################################################################################################



# function to calculate the effective redshift


def z_eff_integrand(mu: float, datalist1: list):
    '''
    Function to compute the effective redshift of a redshift bin. 
    '''
    k_index, k, zminv, zmaxv = datalist1  # [numk, k, zmin_iter, zmax_iter]
    P_gg, P_uu = 0.0, 0.0

    # set up arrays to hold power spectrum information as a function of z for a single k mode 
    pmm_k = pmm_array[k_index, :]
    ptt_k = ptt_array[k_index, :]
    f___k = growth_rate_array[k_index, :]
    
    
    # initialize variables for integrals 
    dVeff = 0.0         # effective volume element
    zdVeff = 0.0        # z x effective volume element 

    for zz in range(len(N_redshifts_arr)): # loop over all redshifts in redshift bin 

        zval = N_redshifts_arr[zz]

        if (zval < zminv): 
            continue
        elif (zval > zmaxv):
            break

        r_sum = 0.0
        rval = r_array[zz]
        deltarval = delta_r_array[zz]

        # get the power spectra and growth rate at z
        pmm = pmm_k[zz]
        ptt = ptt_k[zz]
        f = f___k[zz]
        
        # get the value of the galaxy bias at z
        b_g_z = galaxy_bias_array[zz] # get the galaxy bias at z 

        # set up redshift space spectra 

       
        P_gg = cosmo.gg_redshift_s_at_k_real(b_g_z, r_g, f, sigma_gh/h, mu, k, zval, H0, 1.0, 1.0)*pmm
        P_uu = cosmo.uu_redshift_s_at_k_real(sigma_uh/h, mu, f, k, zval, H0, Om, 1.0-Om, 1.0, 1.0)*ptt

    
        # We need to do the overlapping and non-overlapping parts of the redshifts and PV surveys separately
        for s in np.arange(3):
            surv_sum = 0
            error_obs = 0
            error_noise = 0
            n_g = 0
            n_u = 0

            if survey_area[s] > 0:

                if s == 0: # redshift 
                    n_g = N_bar_arr[0][zz]

                elif s == 1: # velocity 
                    error_obs = H0*error_dist*rval              # Percentage error * distance * H0 in km/s (factor of 100.0 comes from hubble parameter)
                    error_noise = error_rand**2 + error_obs**2     # Error_noise is in km^{2}s^{-2}
                    n_u = N_bar_arr[1, zz]/error_noise 

                else: # i == 2, overlap              
                    error_obs = H0*error_dist*rval              # Percentage error * distance * H0 in km/s (factor of 100.0 comes from hubble parameter)
                    error_noise = error_rand**2 + error_obs**2     # Error_noise is in km^{2}s^{-2}
                    n_u = N_bar_arr[1, zz]/error_noise                  
                    n_g = N_bar_arr[0, zz]
                
                
                value1 = n_g/(1.0 + n_g*P_gg)
                value2 = n_u/(1.0 + n_u*P_uu)

                surv_sum += value1**2 + value2**2

                surv_sum *= survey_area[s]
                r_sum += surv_sum

            
        dVeff += rval*rval*deltarval*r_sum
        zdVeff += zval*rval*rval*deltarval*r_sum
        

    return zdVeff/dVeff


# get covariance matrix derivatives 
def get_dCdx_matrix_elements(val_o: cosmo_variable, list_vals: list):
    '''
    Function to get the values for the derivatives of the covariance matrices.
    '''

    kv, zvalv, pmmv, pmtv, pttv, muv, f, D_g, D_u, b_g_z, zindex, kindex = list_vals 

    df_dx, dp_mm_dx = 0, 0

    m1, m2, m3, m4 = 0, 0, 0, 0

    # get the derivatives for all values of mu at fixed z and fixed k
    if val_o == cosmo_variable.H0:  # H0
        
        m1 = dPdH_arr_gg[:,kindex, zindex]
        m2 = dPdH_arr_gu[:,kindex, zindex]
        m3 = m2
        m4 = dPdH_arr_uu[:,kindex, zindex]
    
    elif val_o == cosmo_variable.As: # As
        
        m1 = dPdAs_arr_gg[:,kindex, zindex]
        m2 = dPdAs_arr_gu[:,kindex, zindex]
        m3 = m2
        m4 = dPdAs_arr_uu[:,kindex, zindex]

    elif val_o == cosmo_variable.Obh2: # Obh2
        
        m1 = dPdObh2_arr_gg[:,kindex, zindex]
        m2 = dPdObh2_arr_gu[:,kindex, zindex]
        m3 = m2
        m4 = dPdObh2_arr_uu[:,kindex, zindex]

    elif val_o == cosmo_variable.Och2: # Och2

        m1 = dPdOch2_arr_gg[:,kindex, zindex]
        m2 = dPdOch2_arr_gu[:,kindex, zindex]
        m3 = m2
        m4 = dPdOch2_arr_uu[:,kindex, zindex]
        
    elif val_o == cosmo_variable.mnu: # mnu
        
        m1 = dPdMnu_arr_gg[:,kindex, zindex]
        m2 = dPdMnu_arr_gu[:,kindex, zindex]
        m3 = m2
        m4 = dPdMnu_arr_uu[:,kindex, zindex]



    elif val_o == cosmo_variable.b_g: # b_g (calculate analytically)

        a = 1.0/(1.0 + zvalv)
        H_z = H0*E_z(zvalv)

        m1 = (2.0*b_g_z + 2.0*r_g*f*(muv**2) )*pmmv*(D_g**2)
        m2 = (a*H_z*muv/kv)*( f*r_g )*pmtv*D_g*D_u
        m3 = m2
        m4 = 0
        

    elif val_o == cosmo_variable.r_g: # rg (calculate analytically)

        df_dx = 0
        dp_mm_dx = 0
        dmu_real_dx = 0
        dk_real_dx = 0

        m1 = cosmo.dP_gg_dx(val_o, b_g_z, r_g, f, df_dx, muv, kv, sigma_gh/h, pmmv, dp_mm_dx, zvalv, H0, dmu_real_dx, 
        dk_real_dx, 1.0, 1.0, 0.0, 0.0)
        m2 = cosmo.dP_gu_dx(val_o, b_g_z, r_g, f, df_dx, muv, kv, sigma_gh/h, sigma_uh/h, pmmv, dp_mm_dx, H0, zvalv, Om, 
        1.0-Om, dk_real_dx, dmu_real_dx, 1.0, 1.0, 0.0, 0.0)
        m3 = m2
        m4 = 0


    elif val_o == cosmo_variable.sigma_u: # sigmau (calculate analytically)

        df_dx = 0
        dp_mm_dx = 0
        dmu_real_dx = 0
        dk_real_dx = 0

        m1 = 0
        m2 = cosmo.dP_gu_dx(val_o, b_g_z, r_g, f, df_dx, muv, kv, sigma_gh/h, sigma_uh/h, pmmv, 
        dp_mm_dx, H0, zvalv, Om, 1.0-Om, dk_real_dx, dmu_real_dx, 1.0, 1.0, 0.0, 0.0)
        m3 = m2
        m4 = cosmo.dP_uu_dx(val_o, muv, kv, f, df_dx, sigma_uh/h, pmmv, dp_mm_dx, H0, 
        zvalv, Om, 1.0-Om, dk_real_dx, dmu_real_dx, 1.0, 1.0, 0.0, 0.0)


    elif val_o == cosmo_variable.sigma_g: # sigmag (calculate analytically)

        df_dx = 0
        dp_mm_dx = 0
        dmu_real_dx = 0
        dk_real_dx = 0

        m1 = cosmo.dP_gg_dx(val_o, b_g_z, r_g, f, df_dx, muv, kv, sigma_gh/h, pmmv, dp_mm_dx, zvalv, 
        H0, dmu_real_dx, dk_real_dx, 1.0, 1.0, 0.0, 0.0)
        m2 = cosmo.dP_gu_dx(val_o, b_g_z, r_g, f, df_dx, muv, kv, sigma_gh/h, sigma_uh/h, pmmv, 
        dp_mm_dx, H0, zvalv, Om, 1.0-Om, dk_real_dx, dmu_real_dx, 1.0, 1.0, 0.0, 0.0)
        m3 = m2
        m4 = 0


    elif val_o == cosmo_variable.N_eff: # varying N_eff

        m1 = dPdNeff_arr_gg[:, kindex, zindex]
        m2 = dPdNeff_arr_gu[:, kindex, zindex]
        m3 = m2        
        m4 = dPdNeff_arr_uu[:, kindex, zindex]
        

    elif val_o == cosmo_variable.n_s: # varying n_s

        m1 = dPdns_arr_gg[:, kindex, zindex]
        m2 = dPdns_arr_gu[:, kindex, zindex]
        m3 = m2        
        m4 = dPdns_arr_uu[:, kindex, zindex]
        

    else:
        message = 'get dCdx_matrix_elements(): val_o (input param for derivatives) = ' + str(val_o) 
        logger.error(message)
        # raise Exception('get dCdx_matrix_elements(): o (input param for derivatives) can only be 0, 1, 2 ... 10, 11 or 12')


    return m1, m2, m3, m4



def mu_integrand(mu: float, datalist1: list): # function to be integrated over mu for Fisher matrix elements, 
    # at some k and mu values 
    '''
    Function to compute the integral of the Fisher matrix elements over mu for a single value of k and z.
    '''

    k_index, k, zminv, p1, p2, zmaxv = datalist1  # [numk, k, zmin_iter, Data[i], Data[j], zmax_iter]
    P_gg, P_uu, P_ug = 0, 0, 0

    # getting P_xx(k, z) for full range of redshifts
    pmm_k = pmm_array[k_index, :]
    pmt_k = pmt_array[k_index, :]
    ptt_k = ptt_array[k_index, :]
    f___k = growth_rate_array[k_index, :]

    result_sum = 0

    D_g = np.sqrt(1.0/(1.0+0.5*(k**2*mu**2*(sigma_gh/h)**2)))          # This is unitless
    D_u = np.sin(k*sigma_uh/h)/(k*sigma_uh/h)                           # This is unitless

    for zz in range(len(N_redshifts_arr)): # looping through redshift  
        r_sum = 0
        zval = N_redshifts_arr[zz] 

        if (zval < zminv): 
            continue
        elif (zval > zmaxv):
            break

        rval = r_array[zz]
        deltarval = delta_r_array[zz]
        pmm = pmm_k[zz]
        pmt = pmt_k[zz]
        ptt = ptt_k[zz]
        f = f___k[zz]


        # get the galaxy bias at this redshift 
        b_g_z = galaxy_bias_array[zz]


        datalist_for_derivs = [k, zval, pmm, pmt, ptt, mu, f, D_g, D_u, b_g_z, zz, k_index]
        
        # initialize covariance matrix elements 
        ci1, ci2, ci3, ci4 = 0, 0, 0, 0

        #  derivatives w.r.t. to p1 and p2 here 
        dcdx1, dcdx2, dcdx3, dcdx4 = get_dCdx_matrix_elements(p1, datalist_for_derivs)
        dcdy1, dcdy2, dcdy3, dcdy4 = get_dCdx_matrix_elements(p2, datalist_for_derivs)

        # need to calculate P_uu, P_ug, P_gg etc. 

        
        P_gg = cosmo.gg_redshift_s_at_k_real(b_g_z, r_g, f, sigma_gh/h, mu, k, zval, H0, 1.0, 1.0)*pmm
        P_uu = cosmo.uu_redshift_s_at_k_real(sigma_uh/h, mu, f, k, zval, H0, Om, 1.0-Om, 1.0, 1.0)*ptt
        P_ug = cosmo.gu_redshift_s_at_k_real(b_g_z, r_g, f, sigma_gh/h, sigma_uh/h, mu, k, zval, H0, 
        Om, 1.0-Om, 1.0, 1.0)*pmt
        

        # ----------------------------------------------------------------------------------------------------
        # need to do overlapping and nonoverlapping surveys seperately 
        for s in np.arange(3):
            surv_sum = 0
            error_obs = 0
            error_noise = 0
            n_g = 0
            n_u = 0

            if survey_area[s] > 0:

                if s == 0: # redshift 
                    n_g = N_bar_arr[0, zz]

                elif s == 1: # velocity 
                    error_obs = H0*error_dist*rval                 # Percentage error * distance * H0 - units end up in km/s 
                    error_noise = error_rand**2 + error_obs**2     # Error_noise is in km^{2}s^{-2}
                    n_u = N_bar_arr[1, zz]/error_noise 

                else: # s == 2, overlap              
                    error_obs = H0*error_dist*rval                 # Percentage error * distance * H0 - units end up in km/s 
                    error_noise = error_rand**2 + error_obs**2     # Error_noise is in km^{2}s^{-2}
                    n_u = N_bar_arr[1, zz]/error_noise                  
                    n_g = N_bar_arr[0, zz]
                

                if (n_g == 0 and n_u == 0):
                    continue

                # get the determinant of the covariance matrix C^-1
                det = 1.0 + n_u*n_g*(P_gg*P_uu - P_ug*P_ug) + n_u*P_uu + n_g*P_gg

                # get the inverse covariance matrix: C^-1

                ci1 = n_u*n_g*P_uu + n_g
                ci2 = - n_g*n_u*P_ug
                ci3 = ci2 
                ci4 = n_g*n_u*P_gg + n_u
                

                # multiply the matrices ( dC_dx1 * C^-1 * dC_dx2 * C^-1) and take the trace of the result
                surv_sum += ci1*dcdx1*(ci1*dcdy1 + ci2*dcdy3) + ci1*dcdx2*(ci3*dcdy1 + ci4*dcdy3)
                surv_sum += ci2*dcdx3*(ci1*dcdy1 + ci2*dcdy3) + ci2*dcdx4*(ci3*dcdy1 + ci4*dcdy3)
                surv_sum += ci3*dcdx1*(ci1*dcdy2 + ci2*dcdy4) + ci3*dcdx2*(ci3*dcdy2 + ci4*dcdy4)
                surv_sum += ci4*dcdx3*(ci1*dcdy2 + ci2*dcdy4) + ci4*dcdx4*(ci3*dcdy2 + ci4*dcdy4)

                surv_sum /= det**2
                surv_sum *= survey_area[s]
                r_sum += surv_sum

        result_sum += (rval**2)*deltarval*r_sum

    return result_sum



# ---------------------------------------------------------------------------------------------------------------




# run the main code to compute the Fisher matrix 
if __name__ == "__main__": 

    # setting up ---------------------------------------------------------------------------------------------
    # read in the survey number densities + set up arrays for integration
    read_nz() 
    
    # read in the power spectra, growth rate array, derivatives of power spectra w.r.t. different parameters of interest 
    read_power()
    
    Dataln = [mapping_index_dict[Data[i]] for i in range(nparams)] # list of parameters to be varied numerically
    # get the Planck 2018 results covariance matrix for the base LCDM parameters 
    if include_Planck_base_LCDM_2018 and (0 in Dataln or 1 in Dataln or 2 in Dataln or 3 in Dataln or 4 in Dataln or 13 in Dataln or 14 in Dataln):
        get_Planck18_MCMC(Dataln)

    ks = kvals
    mus = np.linspace(0, 1.0, num_mus)
    

    # --------------------------------------------------------------------------------------------------------


    # do some checks to prevent the Fisher matrix being singular ---------------------------------------------
    if not ((survey_area[0] > 0.0) or (survey_area[2] > 0.0)): # if position and velocity surveys
        # are NOT overlapping AND their is NO density field information,
        # then there is no density field information at all


        for i in np.arange(nparams): 

            if Data[i] == cosmo_variable.b_g: # bg

                message = "ERROR: b_g is a free parameter, but there is no information in the density field (Fisher matrix will be singular)."
                logger.error(message)

            elif Data[i] == cosmo_variable.r_g: # rg

                message = "ERROR: r_g is a free parameter, but there is no information in the density field (Fisher matrix will be singular)."
                logger.error(message)

            elif Data[i] == cosmo_variable.sigma_g: # sigmag

                message = "ERROR: sigma_g is a free parameter, but there is no information in the density field (Fisher matrix will be singular)."
                logger.error(message)



    if not (((survey_area[1] > 0.0) or (survey_area[2] > 0.0))): # if there is NO overlap between surveys
        # AND there is NO information in the velocity field,
        # then there is no velocity field information at all

        for i in np.arange(nparams): 

            if Data[i] == cosmo_variable.sigma_u: # sigma_u
                message = "ERROR: sigma_u is a free parameter, but there is no information in the velocity field (Fisher matrix will be singular)."
                logger.error(message)
     # --------------------------------------------------------------------------------------------------------
            
    # now start calculating the Fisher matrix
    
    # we calculate the Fisher matrix in the redshift bins we want to split up into then sum 
    # up the matrices to get Fisher_matrix_total (the total Fisher information)

    Fisher_matrix_total = np.zeros((nparams, nparams))
    Fisher_matrix_total = np.matrix(Fisher_matrix_total)
    inverted_Fisher_matrix_total = np.zeros((nparams, nparams))
    inverted_Fisher_matrix_total = np.matrix(inverted_Fisher_matrix_total) 

    console.log("Evaluating the Fisher Matrix for %d redshift bins between [z_min = %.3f, z_max = %.3f]" % (num_redshift_bins, zmin, zmax))
    
    # iterating through redshift bins 
    for ziter in range(num_redshift_bins): 
        flags = Dataln 
        # for each redshift bin we will calculate: 
        # 1) the effective redshift
        # 2) the Fisher matrix at this effective redshift 
        # 3) print out error bars for each parameter of interest (taken from the inverted
        # Fisher matrix) to the terminal and effective redshift for this matrix

        zbinwidth = (zmax-zmin)/(num_redshift_bins)
        zmin_iter = ziter*zbinwidth + zmin
        zmax_iter = (ziter+1.0)*zbinwidth + zmin

        rzmax = redshift_dist_spline(zmax_iter)  # max distance to a galaxy in redshift bin
        kmin = np.pi/rzmax                       # k mode value for max distance 

        # give information of min and max k modes
        if (verbosity > 0):
            console.log("Evaluating the Fisher matrix with [k_min = %.6f, k_max = %.6f] and [z_min = %.3f, z_max = %.3f]" % (kmin, kmax, zmin_iter, zmax_iter))


        # Calculate the effective redshift (which has been based on the sum of the S/N for the density and velocity fields)

        # trapezoidal rule over k
        k_sum1, k_sum2 = 0.0, 0.0
        count = 0
        for numk in range(len(delta_ks)):

            k = kvals[numk]+0.5*delta_ks[numk]
            deltak = delta_ks[numk]

            if k < kmin: 
                continue
            elif k > kmax:
                continue 

            datalist1 = [numk, k, zmin_iter, zmax_iter]

            # integration method 2 using simpson rule (faster)
            mus = np.linspace(0.0, 1.0, num_mus)
            zeffs = z_eff_integrand(mus, datalist1)
            result = simps(zeffs, mus)

            k_sum1 += k*k*deltak*result
            k_sum2 += k*k*deltak

        z_eff = k_sum1/k_sum2
        if (verbosity > 0): 
            console.log("Effective redshift for this redshift bin, z_eff = %.6f" % z_eff, style="bold yellow")

        
        
        # Calculate the fisher matrix, integrating over k, mu, and r (r is effectively integrating over the survey volume).
        # As the input spectra are tabulated we'll just use the trapezoid rule to integrate over k
        Fisher_matrix = np.zeros((nparams, nparams))

        # here we will just do the integral for mu and k first (integral over redshift within mu_integrand)
        for i in range(0, nparams):
            for j in range(i, nparams): # getting F_ij where i and j are some parameter we want to allow to vary, in a loop and saving 
                # in Fisher matrix 

           
                k_sum = 0.0

                for numk in range(len(delta_ks)):

                    k = kvals[numk]+0.5*delta_ks[numk]

                    if k < kmin: 
                        continue
                    elif k > kmax:
                        continue 


                    deltak = delta_ks[numk]

                    datalist1 = [numk, k, zmin_iter, Data[i], Data[j], zmax_iter]

                    # integration method 2 using simpson rule
                    mus = np.linspace(0.0, 1.0, num_mus)
                    fmus = mu_integrand(mus, datalist1)
                    result = simps(fmus, mus)

                    k_sum += k*k*deltak*result                      # adding up contribution from all ks (with trapezoidal rule)

                Fisher_matrix[i, j] = k_sum/(np.pi*4)
                if i != j:
                    Fisher_matrix[j, i] = k_sum/(np.pi*4)

        
        # save array as a matrix object 
        Fisher_matrix = np.matrix(Fisher_matrix)

        if write_data_to_files:

            fisher_matrices[('zbin=%s-%s' % (zmin_iter, zmax_iter))] = Fisher_matrix

        # add Fisher matrix in this redshift bin to the complete Fisher matrix 
        Fisher_matrix_total = Fisher_matrix_total + Fisher_matrix

        # print the Fisher matrix to the terminal for this redshift bin 
        if (verbosity > 0):
            console.log("Fisher Matrix for this redshift bin:", style='yellow')
            console.log("==================")
            if include_Planck_base_LCDM_2018 and (0 in Dataln or 1 in Dataln or 2 in Dataln or 3 in Dataln or 4 in Dataln or 13 in Dataln or 14 in Dataln):
                console.log('(Including Planck 2018 information: ' + data_from_MCMC_chains + ')', style='yellow')
                console.log(np.matrix(Fisher_matrix) + planck_18_information, style='yellow') 
            else:
                print(Fisher_matrix, style='yellow')
        
        # now invert the Fisher matrix
        if include_Planck_base_LCDM_2018 and (0 in Dataln or 1 in Dataln or 2 in Dataln or 3 in Dataln or 4 in Dataln or 13 in Dataln or 14 in Dataln):
            Fisher_matrix = np.matrix(Fisher_matrix) + planck_18_information
        else:
            Fisher_matrix = np.matrix(Fisher_matrix)
        inverted_Fisher_matrix = 0
        try: 
            inverted_Fisher_matrix = np.linalg.inv(Fisher_matrix)
        except:
            console.log('Fisher matrix is singular, cannot be inverted in this redshift bin. ', style='bold red')
            console.log('This may be because sigma_u is a free parameter but you have no velocity field information in this bin.', style='bold red')
            console.log('Removing rows and columns corresponding to parameter we have no information for and reinverting matrix:', style='bold red')

            Fisher_matrix, flags = shrink_sqr_matrix(Fisher_matrix, Dataln)
            
            inverted_Fisher_matrix = np.linalg.inv(Fisher_matrix)

        if write_data_to_files:

            covariance_matrices[('zbin=%s-%s' % (zmin_iter, zmax_iter))] = inverted_Fisher_matrix

     
        # calculate b_g_z at z_eff and b_g_z*sigma8 at z_eff 
        pmm_kmin_to_interp = pmm_array[kmin_overall_index, :]
        if N_redshifts_arr[len(N_redshifts_arr)-1] < N_redshifts_arr[len(N_redshifts_arr)-2]:
            pmm_kmin_interp = CubicSpline(N_redshifts_arr[0:len(N_redshifts_arr)-1], pmm_kmin_to_interp[0:len(N_redshifts_arr)-1])
        else:
            pmm_kmin_interp  = CubicSpline(N_redshifts_arr, pmm_kmin_to_interp)
        b_g_zeff = b_g*np.sqrt(pmm_0_kmin)/np.sqrt(pmm_kmin_interp(z_eff))

        
        if verbosity > 0 and not (isinstance(inverted_Fisher_matrix, int)):

            console.log("============================================", style="bold yellow")
            for i in range(len(flags)): # print the error bars for each parameter, as determined from the inverted Fisher matrix inverse for this redshift bin

                # H0, As, Obh, Och, mnu, bg, rg, sigmau, sigmag, N_eff, ns
                # 0 , 1,  2,   3,    4,  7,  8,    9,      10,     13,  14

                if (flags[i] == 0):
                    console.log("H0 = %.6f pm %.6f" % (H0, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    console.log(" %.4f percent error on H0" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/H0)  )


                elif (flags[i] == 1):
                    console.log("As = %.6f * 1e-9 pm %.6f * 1e-9" % (As*1e9, np.sqrt(inverted_Fisher_matrix[i,i])*1e9) )
                    console.log(" %.4f percent error on As" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/As)  )


                elif (flags[i] == 2):
                    console.log("Obh = %.6f pm %.6f" % (Obh, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    console.log(" %.4f percent error on Obh" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/Obh)  )

                    
                elif (flags[i] == 3):
                    console.log("Och = %.6f pm %.6f" % (Och, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    console.log(" %.4f percent error on Och" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/Och)  )

                    
                elif (flags[i] == 4):
                    console.log("m_nu = %.6f pm %.6f" % (m_nu, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    console.log(" %.4f percent error on m_nu" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/m_nu)  )


                elif (flags[i] == 7):
                    console.log("b_g = %.6f pm %.6f" % (b_g_zeff, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    console.log(" %.4f percent error on b_g" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/b_g_zeff)  )

                
                elif (flags[i] == 8):
                    console.log("r_g = %.6f pm %.6f" % (r_g, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    console.log(" %.4f percent error on r_g" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/r_g)  )


                elif (flags[i] == 9):
                    console.log("sigma_u = %.6f pm %.6f" % (sigma_uh/h, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    console.log(" %.4f percent error on sigma_u" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/(sigma_uh/h))    ) 

                
                elif (flags[i] == 10):
                    console.log("sigma_g = %.6f pm %.6f" % (sigma_gh/h, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    console.log(" %.4f percent error on sigma_g" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/(sigma_gh/h))  )


                elif (flags[i] == 13):
                    console.log("Neff = %.6f pm %.6f" % (3.046, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    console.log(" %.4f percent error on Neff" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/(3.046))  )

                elif (flags[i] == 14):
                    console.log("ns = %.6f pm %.6f" % (ns, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    console.log(" %.4f percent error on ns" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/(ns))  )

            console.log("============================================", style="bold yellow")

        # print the inverted Fisher matrix to the terminal for this redshift bin 
        if verbosity > 0 and not (isinstance(inverted_Fisher_matrix, int)):
            console.log("Covariance Matrix for this redshift bin:", style='yellow')
            console.log("==================", style='yellow')
            console.log(inverted_Fisher_matrix, style='yellow')

        string_data = ''

        if write_data_to_files and not (isinstance(inverted_Fisher_matrix, int)):


            for i in range(len(flags)):

                if (flags[i] == 0):
                    string_data = string_data + "H0 = %.6f pm %.6f" % (H0, np.sqrt(inverted_Fisher_matrix[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on H0" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/H0) 
                    string_data = string_data + '\n'

                elif (flags[i] == 1):
                    string_data = string_data + "As = %.6f * 1e-9 pm %.6f * 1e-9" % (As*1e9, np.sqrt(inverted_Fisher_matrix[i,i])*1e9) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on As" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/As)  
                    string_data = string_data + '\n'


                elif (flags[i] == 2):
                    string_data = string_data + "Obh = %.6f pm %.6f" % (Obh, np.sqrt(inverted_Fisher_matrix[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on Obh" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/Obh)  
                    string_data = string_data + '\n'

                    
                elif (flags[i] == 3):
                    string_data = string_data + "Och = %.6f pm %.6f" % (Och, np.sqrt(inverted_Fisher_matrix[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on Och" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/Och)  
                    string_data = string_data + '\n'

                    
                elif (flags[i] == 4):
                    string_data = string_data + "m_nu = %.6f pm %.6f" % (m_nu, np.sqrt(inverted_Fisher_matrix[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on m_nu" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/m_nu)  
                    string_data = string_data + '\n'


                elif (flags[i] == 7):
                    string_data = string_data + "b_g = %.6f pm %.6f" % (b_g_zeff, np.sqrt(inverted_Fisher_matrix[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on b_g" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/b_g_zeff)  
                    string_data = string_data + '\n'

                
                elif (flags[i] == 8):
                    string_data = string_data + "r_g = %.6f pm %.6f" % (r_g, np.sqrt(inverted_Fisher_matrix[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on r_g" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/r_g)  
                    string_data = string_data + '\n'


                elif (flags[i] == 9):
                    string_data = string_data + "sigma_u = %.6f pm %.6f" % (sigma_uh/h, np.sqrt(inverted_Fisher_matrix[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on sigma_u" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/(sigma_uh/h))    
                    string_data = string_data + '\n'

                
                elif (flags[i] == 10):
                    string_data = string_data + "sigma_g = %.6f pm %.6f" % (sigma_gh/h, np.sqrt(inverted_Fisher_matrix[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data +" %.4f percent error on sigma_g" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/(sigma_gh/h))  
                    string_data = string_data + '\n'


                elif (flags[i] == 13):
                    string_data = string_data + "Neff = %.6f pm %.6f" % (3.046, np.sqrt(inverted_Fisher_matrix[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data +" %.4f percent error on Neff" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/(3.046))  
                    string_data = string_data + '\n'


                elif (flags[i] == 14):
                    string_data = string_data + "ns = %.6f pm %.6f" % (ns, np.sqrt(inverted_Fisher_matrix[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data +" %.4f percent error on ns" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/(ns))  
                    string_data = string_data + '\n'


            uncertainties_dictionary[('zbin=%s-%s' % (zmin_iter, zmax_iter))] = string_data
            
        # -------------------------------------------------------------------------------------------------------------------------------

    # calculate the full Fisher matrix constraints, sum of multiple redshift bins 
    if (num_redshift_bins > 1):

        rzmax = rz(zmax)
        kmin = np.pi/rzmax

        if (verbosity > 0):
            console.log("Finally, evaluating the Fisher Matrix for all redshift bins: ", \
            " [k_min = %.5f, k_max = %.5f] and [z_min = %.3f, z_max = %.3f]" % (kmin, kmax, zmin, zmax))

         # Calculate the effective redshift (which I base on the sum of the S/N for the density and velocity fields)
        k_sum1, k_sum2 = 0.0, 0.0
        for numk in range(len(delta_ks)):

            k = kvals[numk]+0.5*delta_ks[numk]
            deltak = delta_ks[numk]

            if k < kmin: 
                continue
            elif k > kmax:
                continue 

            datalist1 = [numk, k, zmin, zmax]

            # integration method 2 using simpson rule (faster)
            mus = np.linspace(0.0, 1.0, num_mus)
            zeffs = z_eff_integrand(mus, datalist1)
            result = simps(zeffs, mus)

            k_sum1 += k*k*deltak*result
            k_sum2 += k*k*deltak

        
        z_eff = k_sum1/k_sum2
        if (verbosity > 0): 
            console.log("Effective redshift z_eff = %.6f" % z_eff, style="bold yellow")


        if (verbosity > 0):
            console.log("Fisher Matrix for all redshift bins:", style='yellow')
            console.log("======================", style='yellow')
            if include_Planck_base_LCDM_2018 and (0 in Dataln or 1 in Dataln or 2 in Dataln or 3 in Dataln or 4 in Dataln or 13 in Dataln or 14 in Dataln):
                console.log('Including Planck 2018 results:', style='yellow')
                console.log(Fisher_matrix_total + planck_18_information, style='yellow')
            else:
                console.log(Fisher_matrix_total, style='yellow')

        if write_data_to_files and include_Planck_base_LCDM_2018:

            fisher_matrices['sum of fisher information + any planck/extra info'] = Fisher_matrix_total + planck_18_information

        # Now invert the Fisher matrix
        if include_Planck_base_LCDM_2018 and (0 in Dataln or 1 in Dataln or 2 in Dataln or 3 in Dataln or 4 in Dataln or 13 in Dataln or 14 in Dataln):
            inverted_Fisher_matrix_total = np.linalg.inv(Fisher_matrix_total + planck_18_information)
        else:
            inverted_Fisher_matrix_total = np.linalg.inv(Fisher_matrix_total)

        if write_data_to_files:

            covariance_matrices['cov matrix for all fisher information + any planck/extra info'] = inverted_Fisher_matrix_total

        # calculate b0 at z_eff and b0*sigma8 at z_eff 
        pmm_kmin_to_interp = pmm_array[kmin_overall_index, :]
        if N_redshifts_arr[len(N_redshifts_arr)-1] < N_redshifts_arr[len(N_redshifts_arr)-2]:
            pmm_kmin_interp = CubicSpline(N_redshifts_arr[0:len(N_redshifts_arr)-1], pmm_kmin_to_interp[0:len(N_redshifts_arr)-1])
        else:
            pmm_kmin_interp = CubicSpline(N_redshifts_arr, pmm_kmin_to_interp)
        b_g_zeff = b_g*np.sqrt(pmm_0_kmin)/np.sqrt(pmm_kmin_interp(z_eff))


        if verbosity > 0:

            for i in range(nparams):

                # H0, As, Obh, Och, mnu, bg, rg, sigmau, sigmag, N_eff
                # 0 , 1,  2,   3,    4,  7,  8,    9,      10,     13

                if (Dataln[i] == 0):
                    console.log("H0 = %.6f pm %.6f" % (H0, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    console.log(" %.4f percent error on H0" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/H0)  )


                elif (Dataln[i] == 1):
                    console.log("As = %.6f * 1e-9 pm %.6f *1e-9" % (As*1e9, np.sqrt(inverted_Fisher_matrix_total[i,i])*1e9) )
                    console.log(" %.4f percent error on As" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/As)  )


                elif (Dataln[i] == 2):
                    console.log("Obh = %.6f pm %.6f" % (Obh, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    console.log(" %.4f percent error on Obh" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/Obh)  )


                elif (Dataln[i] == 3):
                    console.log("Och = %.6f pm %.6f" % (Och, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    console.log(" %.4f percent error on Och" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/Och)  )


                elif (Dataln[i] == 4):
                    console.log("m_nu = %.6f pm %.6f" % (m_nu, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    console.log(" %.4f percent error on m_nu" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/m_nu)  )


                elif (Dataln[i] == 7):
                    console.log("b_g = %.6f pm %.6f" % (b_g_zeff, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    console.log(" %.4f percent error on b_g" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/b_g_zeff)  )

                
                elif (Dataln[i] == 8):
                    console.log("r_g = %.6f pm %.6f" % (r_g, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    console.log(" %.4f percent error on r_g" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/r_g)  )


                elif (Dataln[i] == 9):
                    console.log("sigma_u = %.6f pm %.6f" % (sigma_uh/h, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    console.log(" %.4f percent error on sigma_u" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/(sigma_uh/h))    )

                elif (Dataln[i] == 10):
                    console.log("sigma_g = %.6f pm %.6f" % (sigma_gh/h, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    console.log(" %.4f percent error on sigma_g" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/(sigma_gh/h))  )

                elif (Dataln[i] == 13):
                    console.log("Neff = %.6f pm %.6f" % (3.046, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    console.log(" %.4f percent error on Neff" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/(3.046))  )


                elif (Dataln[i] == 14):
                    console.log("n_s = %.6f pm %.6f" % (ns, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    console.log(" %.4f percent error on n_s" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/(ns))  )

        if (verbosity > 0):
            console.log("Covariance Matrix:", style='yellow')
            console.log("======================", style='yellow')
            console.log(inverted_Fisher_matrix_total, style='yellow')


        if write_data_to_files:

            string_data = ''

            for i in range(nparams):

                if (Dataln[i] == 0):
                    string_data = string_data + "H0 = %.6f pm %.6f" % (H0, np.sqrt(inverted_Fisher_matrix_total[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on H0" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/H0) 
                    string_data = string_data + '\n'

                elif (Dataln[i] == 1):
                    string_data = string_data + "As = %.6f * 1e-9 pm %.6f * 1e-9" % (As*1e9, np.sqrt(inverted_Fisher_matrix_total[i,i])*1e9) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on As" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/As)  
                    string_data = string_data + '\n'


                elif (Dataln[i] == 2):
                    string_data = string_data + "Obh = %.6f pm %.6f" % (Obh, np.sqrt(inverted_Fisher_matrix_total[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on Obh" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/Obh)  
                    string_data = string_data + '\n'

                    
                elif (Dataln[i] == 3):
                    string_data = string_data + "Och = %.6f pm %.6f" % (Och, np.sqrt(inverted_Fisher_matrix_total[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on Och" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/Och)  
                    string_data = string_data + '\n'

                    
                elif (Dataln[i] == 4):
                    string_data = string_data + "m_nu = %.6f pm %.6f" % (m_nu, np.sqrt(inverted_Fisher_matrix_total[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on m_nu" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/m_nu)  
                    string_data = string_data + '\n'


                elif (Dataln[i] == 7): 
                    string_data = string_data + "b_g = %.6f pm %.6f" % (b_g_zeff, np.sqrt(inverted_Fisher_matrix_total[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on b_g" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/b_g_zeff)  
                    string_data = string_data + '\n'

                
                elif (Dataln[i] == 8):
                    string_data = string_data + "r_g = %.6f pm %.6f" % (r_g, np.sqrt(inverted_Fisher_matrix_total[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on r_g" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/r_g)  
                    string_data = string_data + '\n'


                elif (Dataln[i] == 9):
                    string_data = string_data + "sigma_u = %.6f pm %.6f" % (sigma_uh/h, np.sqrt(inverted_Fisher_matrix_total[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on sigma_u" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/(sigma_uh/h))    
                    string_data = string_data + '\n'

                
                elif (Dataln[i] == 10):
                    string_data = string_data + "sigma_g = %.6f pm %.6f" % (sigma_gh/h, np.sqrt(inverted_Fisher_matrix_total[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data + " %.4f percent error on sigma_g" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/(sigma_gh/h))  
                    string_data = string_data + '\n'


                elif (Dataln[i] == 13):
                    string_data = string_data + "Neff = %.6f pm %.6f" % (3.046, np.sqrt(inverted_Fisher_matrix_total[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data +" %.4f percent error on Neff" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/(3.046))  
                    string_data = string_data + '\n'


                elif (Dataln[i] == 14):
                    string_data = string_data + "ns = %.6f pm %.6f" % (ns, np.sqrt(inverted_Fisher_matrix_total[i,i])) 
                    string_data = string_data + '\n'
                    string_data = string_data +" %.4f percent error on ns" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/(ns))  
                    string_data = string_data + '\n'


            uncertainties_dictionary[('total fisher information covariance matrix')] = string_data

     
    # now plot some 2D confidence ellipses with sum of neutrino masses

    # firstly saving a few things
    if num_redshift_bins == 1:
        inverted_Fisher_matrix_total = inverted_Fisher_matrix


    if plot_ellipses and 4 in Dataln:
    
        
        vals = {'0': [H0, 'H_0', r'$H_0$'],                   '7': [b_g_zeff, 'bg', r'$b_g$'],
                '1': [As*1e9, 'A_sx1e9', r'$A_s \times 10^{9}$'], '8': [r_g, 'rg', r'$r_g$'],
                '2': [Obh, 'Obh2', r'$\Omega_bh^2$'],          '9': [sigma_uh/h, 'sigmau', r'$\sigma_u$'],
                '3': [Och, 'Och2', r'$\Omega_ch^2$'],          '10': [sigma_gh/h, 'sigmag', r'$\sigma_g$'],
                '4': [m_nu, 'Mnu', r'$M_{\nu}$'],             '13': [3.046, 'Neff', r'$N_{eff}$'],
                '14': [ns, 'ns', r'$n_s$'],
                     }


        for i in np.arange(len(Data)):

            if Dataln[i] != 4:

                mnu_index = Dataln.index(4)
                data_flag = Dataln[i]
                

                sigma_otherval_sqrd = 0
                cross_correlation_sqrd = 0 
                cross_correlation = 0
    
                if Dataln[i] == 1:
                    sigma_otherval_sqrd = inverted_Fisher_matrix_total[i,i]*1e18
                    cross_correlation_sqrd = (inverted_Fisher_matrix_total[mnu_index, i]*1e9)**2
                    cross_correlation = inverted_Fisher_matrix_total[mnu_index, i]*1e9
                else:
                    sigma_otherval_sqrd = inverted_Fisher_matrix_total[i, i]
                    cross_correlation_sqrd = (inverted_Fisher_matrix_total[mnu_index, i])**2
                    cross_correlation = inverted_Fisher_matrix_total[mnu_index, i]

                sigma_mnu_sqrd = inverted_Fisher_matrix_total[mnu_index, mnu_index]
        
                a = 0.5*(sigma_otherval_sqrd + sigma_mnu_sqrd) + np.sqrt( 0.25*((sigma_mnu_sqrd - sigma_otherval_sqrd)**2) + cross_correlation_sqrd )
                b = 0.5*(sigma_otherval_sqrd + sigma_mnu_sqrd) - np.sqrt( 0.25*((sigma_mnu_sqrd - sigma_otherval_sqrd)**2) + cross_correlation_sqrd )

                width1 = 2.0*np.sqrt(a)*1.52
                height1 = 2.0*np.sqrt(b)*1.52

                width2 = 2.0*np.sqrt(a)*2.48
                height2 = 2.0*np.sqrt(b)*2.48

                width3 = 2.0*np.sqrt(a)*3.44
                height3 = 2.0*np.sqrt(b)*3.44

                theta = np.arctan2(2.0*(cross_correlation), (sigma_mnu_sqrd - sigma_otherval_sqrd) )/2.0


                ell1 = Ellipse(xy=(m_nu, vals[str(data_flag)][0]), width=width1, height=height1, angle=theta*180.0/np.pi)
                ell2 = Ellipse(xy=(m_nu, vals[str(data_flag)][0]), width=width2, height=height2, angle=theta*180.0/np.pi)
                ell3 = Ellipse(xy=(m_nu, vals[str(data_flag)][0]), width=width3, height=height3, angle=theta*180.0/np.pi)

                ells = [ell1, ell2, ell3]
                labels = [r'68.3%', r'95.4%', r'99.7%']
                    
                plt.figure()
                ax = plt.gca()
                ax.add_patch(ell1)
                ax.add_patch(ell2)
                ax.add_patch(ell3)
                ell1.set(alpha=(0.2), facecolor = 'r')
                ell2.set(alpha=(0.2), facecolor = 'r')
                ell3.set(alpha=(0.2), facecolor = 'r')
                ax.scatter(m_nu, vals[str(data_flag)][0], color='white')
                plt.xlabel(r'$M_{\nu}$')
                plt.ylabel( vals[str(data_flag)][2] )
                plt.xlim([0, m_nu+width3/2.0 + 0.05])
                plotname = save_file_folder+'mnu_%s_contour.png' % (vals[str(data_flag)][1])
                plt.savefig(plotname)
                

        
    # write fisher matrices, covariance matrices, uncertainties on parameters, to a file
    if write_data_to_files:

        with open((save_file_folder + r'fisher_matrices_only_survey_info.txt'), 'w') as f:
            for k in fisher_matrices:
                print(k, file=f)  
               
                for line in fisher_matrices[k]:
                    np.savetxt(f, line)
                    
        with open((save_file_folder + r'covariance_matrices_plus_any_planck_or_therprobes_info.txt'), 'w') as f:
            for k in covariance_matrices:
                print(k, file=f)  
                
                if not isinstance(covariance_matrices[k], int):
                    for line in covariance_matrices[k]:
                        np.savetxt(f, line)
                else:
                    print('covariance matrix is singular in this redshift bin')

        with open((save_file_folder + r'parameter_1sigma_uncertainties_each_bin.txt'), 'w') as f:
            for k in uncertainties_dictionary:
                print(k, file=f)  
                print(uncertainties_dictionary[k], file=f)


        with open((save_file_folder + r'extra_details.txt'), 'w') as f:
            print('Value of effective redshift for all fisher information (matrices summed) in this forecasting run: ', z_eff, file=f)  
            



     # ----------------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------------------------





