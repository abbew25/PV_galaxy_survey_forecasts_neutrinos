# Author: Abbe Whitford 

# script to read in fisher matrices and get correct forecast
# by treating b_g, sigma_g, sigma_u, rg etc (any nuisance parameters) as separate free parameters in different redshift bins 


# imports --------------------------------------------------------------------------------
import numpy as np
import os 
import pandas as pd 

# functions ------------------------------------------------------------------------------

def shrink_sqr_matrix(sqr_matrix_obj):
    ''' 
    Function that removed the rows and columns of a square matrix (numpy matrix) if the rows 
    and columns that a diagonal element of the matrix coincides with is zero.
    e.g. 1 2 3 4
         2 1 9 0   ----- >     1 2 4
         4 5 0 9               2 1 0
         4 3 2 1               4 3 1
    
    The third row and column has been removed since M_(3, 3) <= 1e-7
    '''
    a = 0
    b = False 
    for i in np.arange(sqr_matrix_obj.shape[0]):
        if sqr_matrix_obj[i,i] <= 1e-7:
            a = i
            b = True 
            
    if b:
        sqr_matrix_obj = np.delete(sqr_matrix_obj, a, 0)
        sqr_matrix_obj = np.delete(sqr_matrix_obj, a, 1)
        sqr_matrix_obj = shrink_sqr_matrix(sqr_matrix_obj)
        return sqr_matrix_obj
    else:
        return sqr_matrix_obj


# function to get the Planck 2018 MCMC chains for H0, As, Obh2, Och2
def get_Planck18_MCMC(Data_list, MCMC_chains_option):


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

        if 13 in Data:

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
        raise Exception('MCMC chains option is not valid. (in get_Planck18_MCMC()).')
    
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

    return planck_18_information
    
  

# ---------------------------------------------------------------------------------------------------------------------------

# setting up information to read in  
# A list of flags for the parameters we are interested in varying in the analysis/free parameters - comment out parameters you don't want
# Ordering of parameters here is best and shouldn't be altered (cosmo parameters before nuisance parameters), 
# just comment out lines for parameters that were not included in the forecasting analysis.

Data = [                                        
0,                # H0
1,                # As
2,                # Obh
3,                # Och
4,                # mnu
13,               # N_eff (effective number of neutrino species)  
14,               # n_s (the spectral index)
7,                # galaxy bias b_g
#8,               # r_g 
9,                # sigmau 
10,               # sigmag                            
        ]

Data_list_cosmo_params = Data 

file_path_main = str(os.getcwd()) + '/example_results/'
file_path = file_path_main + 'fisher_matrices_only_survey_info.txt' 
number_free_params = 10 # total number of free parameters in each redshift bin (len(Data) in forecasting analysis) - user needs to set this 
number_params_nueb = 7 # number of parameters that will be treated as the same parameter in all redshift bins (any cosmological parameters, and not nuisance parameters)
num_unique_params = number_free_params - number_params_nueb
num_redshift_bins = 5 # total number of matrices we will read in 
include_Planck = True # do we want to get an estimate of the fisher information from Planck on cosmo parameters and add it to our fisher information?
check_ill_conditioned = False  # do we want to check if the matrix is not symmetric or is ill-conditioned?
print_fisher_matrix = False    # do we want to check the Fisher matrix?
Planck_info_flag = 1           # flag for what information to include from Planck MCMC chains 
                               # 1 = Planck PlikHM 2018 TTTEEE + lowE + lowl, 
                               # 2 = Planck PlikHM 2018 TTTEEE + lowE + lowl + lensing (no lensing if N_eff is free parameter though)     
Neff_free = False 
if 13 in Data:
    Neff_free = True
include_lensing = False 
if Planck_info_flag == 2:
    include_lensing = True 

# setting up the fisher matrix to invert  
total_free_params_all_bins = number_params_nueb + num_redshift_bins*num_unique_params
total_fisher = np.zeros((total_free_params_all_bins, total_free_params_all_bins))




# reading in the matrices 
for zbin in np.arange(num_redshift_bins):
    # loading the fisher info in the redshift bin
    fishermatrix = np.array(np.loadtxt(file_path, delimiter=' ', skiprows=(1+zbin*(1 + number_free_params)), max_rows=number_free_params))
    # adding information on cosmological parameters
    total_fisher[0:number_params_nueb, 0:number_params_nueb] += fishermatrix[0:number_params_nueb, 0:number_params_nueb]
    # adding information on nuisance parameters that are treated as separate in each redshift bin
    k = zbin*num_unique_params
    total_fisher[0:number_params_nueb, (number_params_nueb+k):(number_free_params+k)] += fishermatrix[0:number_params_nueb, number_params_nueb:number_free_params]
    total_fisher[(number_params_nueb+k):(number_free_params+k), 0:number_params_nueb] += fishermatrix[number_params_nueb:number_free_params, 0:number_params_nueb]
    total_fisher[(number_params_nueb+k):(number_free_params+k),(number_params_nueb+k):(number_free_params+k)] += fishermatrix[number_params_nueb:number_free_params,number_params_nueb:number_free_params]



total_fisher = shrink_sqr_matrix(total_fisher) # getting rid of free parameters in redshift bins where there is no information 
# e.g. for a PV + redshift forecast, there is no information on sigma_u in high redshift bins so the rows and columns corresponding to sigma_u in these 
# higher bins needs to be removed to avoid having a singular fisher matrix 



if print_fisher_matrix:
    for i in np.arange(total_fisher.shape[0]):
        print(str(np.array_str(total_fisher[i,:],  precision = 2)).replace('\n', ' '))


# now adding the Planck information 

Planck_fisher_estimate = get_Planck18_MCMC(Data, Planck_info_flag)
Planck_fisher_estimate = shrink_sqr_matrix(Planck_fisher_estimate)
if include_Planck:
    total_fisher[0:number_params_nueb, 0:number_params_nueb] += Planck_fisher_estimate
    

# get covariance matrix 
cov = np.linalg.inv(total_fisher)


# check if ill conditioned 
if check_ill_conditioned:
    print('If matrix is not ill-conditioned and is symmetric, the following number will be close to zero: -----')
    print(sum(sum(total_fisher*cov))-sum(sum(total_fisher.T*cov))) # checking the matrix is symmetric and is not ill-conditioned  
    

# print error on mnu to terminal
mnu_error = np.sqrt(cov[4,4])
mnu_per_error = 100.0*mnu_error/0.058
results_mnu = 'mnu error: ' + str(mnu_error) + ' eV, % error: ' + str(mnu_per_error)
print(results_mnu)


# now write the results to an appropriate file location with a descriptive name
# write the results to a file 
if include_Planck:
    if include_lensing or Neff_free:
        with open((file_path_main + 'final_results_mnu.txt'), 'w') as f:
            print(results_mnu, file=f)  

        with open((file_path_main + 'cov_final.txt'), 'w') as f:
            print(cov, file=f)  

        with open((file_path_main + 'fisher_final.txt'), 'w') as f:
            print(total_fisher, file=f)  
    else:
        with open((file_path_main + 'final_results_mnu_nolensing.txt'), 'w') as f:
            print(results_mnu, file=f)  

        with open((file_path_main + 'cov_final_nolensing.txt'), 'w') as f:
            print(cov, file=f)  

        with open((file_path_main + 'fisher_final_nolensing.txt'), 'w') as f:
            print(total_fisher, file=f)  
else: # No planck info included 
    with open((file_path_main + 'final_results_mnu_no_planck.txt'), 'w') as f:
        print(results_mnu, file=f)  

    with open((file_path_main + 'cov_final_no_planck.txt'), 'w') as f:
        print(cov, file=f)  

    with open((file_path_main + 'fisher_final_no_planck.txt'), 'w') as f:
        print(total_fisher, file=f)  
