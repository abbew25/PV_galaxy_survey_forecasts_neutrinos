# Author: Abbe Whitford

# functions for main script, main_PV_forecasts.py

# Import modules/libraries --------------------------------------
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd                          
from classy import Class
from scipy.optimize import fsolve
from scipy.integrate import quad
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import numpy.typing as npt 
from loguru import logger
from rich.console import Console
from enum import StrEnum

# setting up some new object things 
cosmo_variable = StrEnum("variable", "H0 As Och2 Obh2 mnu Neff n_s b_g r_g sigma_u sigma_g")

RSPS_variable = StrEnum("P_variable", "P_gg P_uu P_ug")

# function definitions ----------------------------------------------------------------------------



# get the mass of each neutrino eigenstate, given the hierarchy (excluding degenerate hierarchy as an option),
# sum of masses and neutrino oscillation mass differences
# this code is inspired by some code I found here: https://github.com/lesgourg/class_public/blob/master/notebooks/neutrinohierarchy.ipynb
def get_masses(sum_masses_true: float, hierarchy: str, 
    del_mnu_max: float, sum_masses_central_true: float, 
    delta_m_squared_atm: float = 2.5e-3, delta_m_squared_sol: float = 7.5e-5):
    ''' 
    This function is used to get the mass of each neutrino given the sum of neutrino masses, the mass splittings 
    and the neutrino hierarchy, appropriately for the forecasting analysis. 
    '''
    if (sum_masses_central_true-del_mnu_max <= (np.sqrt(delta_m_squared_sol) + np.sqrt(delta_m_squared_atm))):
        
        return 0, 0, sum_masses_true # just let 2 of the neutrinos have no mass (similar to normal hierarchy)
        
    elif (sum_masses_true <= (np.sqrt(delta_m_squared_sol) + np.sqrt(delta_m_squared_atm))):

        return 0, 0, sum_masses_true # just let 2 of the neutrinos have no mass (similar to normal hierarchy)

    else:
        if hierarchy == 'normal':
            m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
            m1 = fsolve(m1_func,sum_masses_true/3.,(sum_masses_true,delta_m_squared_atm,delta_m_squared_sol),full_output=True)[0]
            m1 = m1[0]
            m2 = (delta_m_squared_sol + m1**2.)**0.5
            m3 = (delta_m_squared_atm + m1**2.)**0.5
            
            return m1,m2,m3
        elif hierarchy == 'inverted': # assume inverted hierarchy
            if sum_masses_central_true-del_mnu_max <= (np.sqrt(delta_m_squared_atm - delta_m_squared_sol) + np.sqrt(delta_m_squared_atm)):
                # 2 equally massive neutrinos, 1 with no mass
                m1 = sum_masses_true/2.0
                m2 = sum_masses_true/2.0
                m3 = 0
               
                return m1,m2,m3
            elif sum_masses_true <= (np.sqrt(delta_m_squared_atm - delta_m_squared_sol) + np.sqrt(delta_m_squared_atm)):
                # 2 equally massive neutrinos, 1 with no mass
                m1 = sum_masses_true/2.0
                m2 = sum_masses_true/2.0
                m3 = 0
            
                return m1,m2,m3
            else: # use an inverted hierarchy
                m3_func = lambda m3, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + d_m_sq_sol + m3**2. - 2.*M_tot*m3 - 2.*M_tot*(d_m_sq_atm+m3**2.)**0.5 + 2.*m3*(d_m_sq_atm+m3**2.)**0.5
                m3 = fsolve(m3_func,sum_masses_true/3.,(sum_masses_true,delta_m_squared_atm,delta_m_squared_sol),full_output=True)[0]
                m3 = m3[0]
                m1 = (m3**2. + delta_m_squared_atm -delta_m_squared_sol)**0.5
                m2 = (delta_m_squared_atm + m3**2.)**0.5
                #print('4')
                return m3, m1, m2
        else: # assume degenerate hierarchy
            m1 = sum_masses_true/3.0
            m2 = m1
            m3 = m1
            return m1, m2, m3


# get angular diameter distance integral
def angular_diameter_distance(Om: float, H0: float, z: float, c: float = 299792.458):
    '''
    Returns the angular diameter distance to redshift z assuming flat cosmology with H0, Om as given 
    by first two arguments of this function.
    Inputs:
        z = redshift
        Om = normalized matter density
        H0 = Hubble constant 
    '''
    integral = quad(get_Hubble_z_inv, 0.0, z, args = (Om, H0))[0]
    Ang_diam_dist = c*integral/(1.0+z)
    return Ang_diam_dist


# get Hubble function (H(z))
def get_Hubble_z(z: float, Om: float, H0: float):
    '''
    Returns H(z).
    Inputs:
        z = redshift
        Om = normalized matter density
        H0 = Hubble constant 
    '''
    E_z = np.sqrt( Om*((1+z)**3) + (1-Om))
    H_z = H0*E_z
    return H_z


# get inverse Hubble function (for integration)
def get_Hubble_z_inv(z: float, Om: float, H0: float):
    '''
    Returns 1/H(z).
    Inputs:
        z = redshift
        Om = normalized matter density
        H0 = Hubble constant 
    '''
    return 1.0/get_Hubble_z(z, Om, H0)


# get normalized matter density from sum of neutrino mass, baryon and CDM density
def get_Om_0(Obh2: float, Och2: float, Mnu: float, H0: float):
    '''
    Returns Om at the present day given H0, mnu, Obh2, Och2 etc.
    Using the relation Omega_nu h^2 = sum(m_nu)/(93.14 eV)
    '''
    h = H0/100.0
    Om = (Obh2 + Och2 + (Mnu)/(93.14))/(h**2)
    return Om


# function to compute q_perpendicular (parameter related to AP Effect, appropriate for a cosmological fit to data)
def distortion_parallel(As_t: float, Obh2_t: float, Och2_t: float,
    H0_t: float, Mnu_t: float, As_fid: float,                     
    Obh2_fid: float, Och2_fid: float, 
    H0_fid: float, Mnu_fid: float, z: float):
    '''
    Returns q_parallel, given values of cosmological parameters for a REAL cosmology (ending with _t)
    and cosmological parameters for an assumed fiducial cosmology, and the redshift z.
    '''
    Om_t = get_Om_0(Obh2_t, Och2_t, Mnu_t, H0_t)
    Om_fid = get_Om_0(Obh2_fid, Och2_fid, Mnu_fid, H0_fid) 

    H_z_t = get_Hubble_z(z, Om_t, H0_t)
    H_z_fid = get_Hubble_z(z, Om_fid, H0_fid)

    q_para = (H_z_fid)/(H_z_t)
   
    return q_para


# function to compute q_parallel (parameter related to AP Effect, appropriate for a cosmological fit to data)
def distortion_perpendicular(As_t: float, Obh2_t: float, Och2_t: float, 
    H0_t: float, Mnu_t: float, As_fid: float, 
    Obh2_fid: float, Och2_fid: float, 
    H0_fid: float, Mnu_fid: float, z: float):
    '''
    Returns q_perpendicular, given values of cosmological parameters for a REAL cosmology (ending with _t)
    and cosmological parameters for an assumed fiducial cosmology, and the redshift z.
    '''
    Om_t = get_Om_0(Obh2_t, Och2_t, Mnu_t, H0_t)
    D_A_z_t = angular_diameter_distance(Om_t, H0_t, z)

    Om_fid = get_Om_0(Obh2_fid, Och2_fid, Mnu_fid, H0_fid)
    D_A_z_fid = angular_diameter_distance(Om_fid, H0_fid, z)
    
    q_perp = 0
    if z == 0: 
        q_perp = H0_fid / H0_t
    else:
        q_perp = (D_A_z_t)/(D_A_z_fid)
        

    return q_perp


def distortion_ratio_F(As_t: float, Obh2_t: float, Och2_t: float, 
    H0_t: float, Mnu_t: float, As_fid: float, 
    Obh2_fid: float, Och2_fid: float, 
    H0_fid: float, Mnu_fid: float, z: float):
    '''
    Returns F = q_parallel/q_perpendicular.
    '''
    q_parallel = distortion_parallel(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
    q_perpend = distortion_perpendicular(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
    F = q_parallel/q_perpend
    return F


# function to get mu_real (real values of mu) from mu_fiducial (values of mu observer believes they are measuring based on fiducial cosmology)
def get_mus_realobs(mus: float, As_t: float,  # get the actual observations of mu given some some set of mus to be the 'real' mu
    Obh2_t: float, Och2_t: float, H0_t: float, 
    Mnu_t: float, As_fid: float, Obh2_fid: float, 
    Och2_fid: float, H0_fid: float, Mnu_fid: float, z: float):
    '''
    Returns mu (observed) computed from parameters for real and fidicual cosmologies and mu (actual). 
    '''
    F = distortion_ratio_F(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
    mus_obs = ((mus)/(F))*(1.0/(np.sqrt(1.0 + (mus**2)*( 1.0/(F**2) - 1.0 ))))
    
    return mus_obs


# # function to get mu_observed (observed values of mu) from mu_actual 
# def get_mus_obsreal(mus_obs: float, As_t: float,  # get the actual mu given some some set of mus to be the observations
#     Obh2_t: float, Och2_t: float, H0_t: float, 
#     Mnu_t: float, As_fid: float, Obh2_fid: float, 
#     Och2_fid: float, H0_fid: float, Mnu_fid: float, z: float):
#     '''
#     Returns mu (real) computed from parameters for real and fidicual cosmologies and mu (observed). 
#     '''
#     F = distortion_ratio_F(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
#     mu_real = mus_obs * F * (np.sqrt(1.0 + (mus_obs**2)*(F**2 - 1.0))) 
#     return mu_real 

# function to get k_real (real k modes) from k_fiducial (values of mk modes observer believes they are measuring based on fiducial cosmology)
def get_ks_realobs(ks: float, As_t: float, Obh2_t: float, 
    Och2_t: float, H0_t: float, Mnu_t: float, 
    As_fid: float, Obh2_fid: float, Och2_fid: float, 
    H0_fid: float, Mnu_fid: float, z: float, mus: float):
    '''
    Returns k (observed) computed from parameters for real and fidicial cosmo and mu and k (actual).
    '''
    F = distortion_ratio_F(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
    q_perp = distortion_perpendicular(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
    
    ks_obs = ((ks)/(q_perp))*np.sqrt(1.0 + (mus**2)*( 1.0/(F**2) - 1.0 ))

    return ks_obs


# # function to get k_observed (observed k modes) from k_actual 
# def get_ks_obsreal(ks_obs: float, As_t: float, Obh2_t: float,
#     Och2_t: float, H0_t: float, Mnu_t: float, 
#     As_fid: float, Obh2_fid: float, Och2_fid: float, 
#     H0_fid: float, Mnu_fid: float, z: float, mus: float):
#     '''
#     Returns k (real) computed from parameters for real and fiducial cosmologies and k (observed).
#     '''
#     F = distortion_ratio_F(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
#     q_perp = distortion_perpendicular(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
#     ks_real = ks_obs * q_perp / (np.sqrt(1.0 + (mus**2)*(F**2 - 1.0)))
#     return ks_real


# function to just get the power spectrum of matter (linear) from CLASS----------------------------
def run_class(omega_b_t: float, omega_cdm_t: float, H0_t: float, As_t: float, 
m_nu_t: float, neutrino_hierarchy: str, zed: list, kmin: float, kmax: float, 
knum: int, del_mnu_max: float, sum_masses_central_t: float, kspaceoption: str, tau_t: float, ns_t: float, 
omega_b_fid: float, omega_cdm_fid: float, H0_fid: float, As_fid: float, m_nu_fid: float, mus_arr: npt.NDArray, 
linornonlin: str, N_eff_deviation: float):
    ''' 
    Function to compute the matter power spectrum from Class() (which is equal to the velocity divergence and cross correlation of the velocity and 
    density power spectrum in linear theory).
    Returns Pk, k_obs and mu_obs (assuming mu is supplied).
    zed always need to be in a vector (even for a single z).
    mu always needs to be in a vector (even for a single mu).
    Shape of Pk returned, k_real returned and mu_real returned will depend on whether AP effect is taken into account or not (whether fiducial
    and real cosmological model differ or not).
    '''
    # Set the CLASS parameters
    M = Class()
    # get the neutrino masses
    m1, m2, m3 = get_masses(m_nu_t, neutrino_hierarchy, del_mnu_max, sum_masses_central_t) # true neutrino masses

    mass_input_string = str(m1)+','+str(m2)+','+str(m3) 
    if (m1 < 0 or m2 < 0 or m3 < 0): # we have a problem
        print(m1 + m2 + m3)
        msg = 'Neutrino masses are set to: ', str(m1) + '+ ' + '(0,'+str(m2-m1)+','+str(m3-m1) + '). ' 
        msg = msg + 'Neutrino masses are unphysical, sum of masses is most likely too small for set mass eigenstate differences from neutrino oscillations.'
        logger.error(msg)
        raise (ValueError)
    
    
    if linornonlin == 'linear':
        # set up TRUE cosmology
        M.set({"omega_b": omega_b_t, "omega_cdm": omega_cdm_t, "H0": H0_t, "A_s": As_t, "N_ur": 0.106+N_eff_deviation, 
        "N_ncdm": 3.0, "T_ncdm": "0.71,0.71,0.71",
        "m_ncdm": mass_input_string, "tau_reio": tau_t, "n_s": ns_t})
        M.set({"output": "mPk", "P_k_max_1/Mpc": 5.0, "z_max_pk": 2.0})
        
    elif linornonlin == 'nonlinear':
        # set up TRUE cosmology
        M.set({"omega_b": omega_b_t, "omega_cdm": omega_cdm_t, "H0": H0_t, "A_s": As_t, "N_ur": 0.106+N_eff_deviation, 
        "N_ncdm": 3.0, "T_ncdm": "0.71,0.71,0.71",
        "m_ncdm": mass_input_string, "tau_reio": tau_t, "n_s": ns_t})
        M.set({"output": "mPk", "P_k_max_1/Mpc": 3.0, "z_max_pk": 2.0, 'non linear': 'Halofit'})
        
    else:
        msg = 'linornonlin not set to appropriate string, options are \'linear\' or \'nonlinear\' (run_class()).'
        logger.error(msg)
        raise (ValueError)

    M.compute() 
    
    
    # now to compute the power spectrum at the observed (k, mu) given (k, mu actual) -------------------------------------------

    # first make an array with all the fiducial values of k
    ks_true = []
    if kspaceoption == 'log':
        ks_true = np.logspace(np.log10(kmin), np.log10(kmax), knum, base=10) # 1/ MPC units
    else:
        ks_true = np.linspace(kmin, kmax, knum) # 1/MPC units

    # lets first make an array of k_fid, mu_fid then k_true, mu_true corresponding to k_fid, mu_fid

    if ((omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid) == (omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t)): # no need to worry about AP effect

        Pk = np.zeros((knum, len(zed)))
        for reds in np.arange(len(zed)):
            Pk[:,reds] = np.array([M.pk_cb_lin(ki, zed[reds]) for ki in ks_true]) # (MPC ^3)
        return Pk, ks_true, mus_arr

    else: # we need to worry about the AP effect 
        
        k_obs_grid = []
        # getting the power spectrum for multiple redshifts

        k_obs_grid = np.zeros((len(mus_arr), len(ks_true), len(zed)))   

        mus_obs = np.zeros((len(mus_arr), len(zed)))

        for zz in np.arange(len(zed)):
            for muu in np.arange(len(mus_arr)):

                k_obs_grid[muu,:, zz] = get_ks_realobs(ks_true, As_t, omega_b_t, omega_cdm_t, H0_t, m_nu_t,
                As_fid, omega_b_fid, omega_cdm_fid, H0_fid, m_nu_fid, zed[zz], mus_arr[muu])

            mus_obs[:, zz] = get_mus_realobs(mus_arr, As_t, omega_b_t, omega_cdm_t, H0_t, m_nu_t, 
            As_fid, omega_b_fid, omega_cdm_fid, H0_fid, m_nu_fid, zed[zz]).flatten()

        # get the matter power spectra at k'
        Pk = np.zeros((len(mus_arr), knum, len(zed)))

        for zz in np.arange(len(zed)):
            for muu in np.arange(len(mus_arr)):
                Pk[muu, :, zz] = np.array([M.pk_cb_lin(ki, zed[zz]) for ki in k_obs_grid[muu, :, zz]])

        return Pk, k_obs_grid, mus_obs

       
# function to get the growth rate at a single value of z (compute numerical approx. to dln(sqrt(P(k))))/dln(a) using central /backwards finite difference)
def growth_rate(omega_b_t: float, omega_cdm_t: float, H0_t: float, As_t: float, 
m_nu_t: float, neutrino_hierarchy: str, zed: float, kmin: float, kmax: float, knum, 
d_a: float, del_mnu_max: float, sum_masses_central_t: float, kspaceoption: str, tau_t: float, 
ns_t: float, omega_b_fid: float, omega_cdm_fid: float, H0_fid: float, As_fid: float, m_nu_fid: float, 
mus_arr: npt.NDArray, N_eff_deviation: float):

    ''' 
    This function allows one to compute f numerically using the finite difference method for a single value of z.
    mus_fid_arr needs to be an array.
    Shape of f(k) returned, k_real returned and mu_real returned will depend on whether AP effect is taken into account or not (whether fiducial
    and real cosmological model differ or not).
    '''
    scalefactors = np.array([1.0, 1.0-d_a]) # set up scalefactors to get power spectrum at 
    if zed != 0:
        scalefactors = np.linspace(1.0/(1.0+zed) + d_a, 1.0/(1.0+zed) - d_a, 3)
    zs = 1.0/scalefactors - 1.0

    pk, ks, mus = run_class(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, zs, kmin, kmax, 
    knum, del_mnu_max, sum_masses_central_t, kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid,
    H0_fid, As_fid, m_nu_fid, mus_arr, 'linear', N_eff_deviation)

    growthrate = 0
    da = abs(scalefactors[1] - scalefactors[0])

    # now finally getting the growth rate 
    if zed == 0: # use backwards difference (finite diff.) since we can't do a central finite difference in z at z = 0 

        if pk.shape == (knum, len(zs)):
            growthrate = 0.5*(scalefactors[0]/(pk[:,0]))*(pk[:,0] - pk[:,1])/(da)
        elif pk.shape == (len(mus_arr), knum, len(zs)):
            growthrate = 0.5*(scalefactors[0]/(pk[:,:,0]))*(pk[:,:,0] - pk[:,:,1])/(da)
        else:
            msg = 'Shape of pk is unexpected. (growth_rate()), shape of pk is: ' + str(pk.shape)
            logger.error(msg)
            raise (ValueError)
    else: # use central difference (finite diff.)

        if pk.shape == (knum, len(zs)):
            growthrate = 0.5*(scalefactors[1]/(pk[:,1]))*(pk[:,0] - pk[:,2])/(2*da)
        elif pk.shape == (len(mus_arr), knum, len(zs)):
            growthrate = 0.5*(scalefactors[1]/(pk[:,:,1]))*(pk[:,:,0] - pk[:,:,2])/(2*da)
        else:
            msg = 'Shape of pk is unexpected. (growth_rate()), shape of pk is: ' + str(pk.shape)
            logger.error(msg)
            raise (ValueError)

    return growthrate, ks, mus


# get the derivative of the growth rate w.r.t. some parameter of choice using just a central finite difference
def derivative_growth_rate(param: cosmo_variable, delta: float, omega_b_t: float, omega_cdm_t: float, 
H0_t: float, As_t: float, m_nu_t: float, neutrino_hierarchy: str, 
zed: float, kmin: float, kmax: float, knum: int, d_a: float, linear: str, 
del_mnu_max: float, sum_masses_central_t: float, kspaceoption: str, tau_t: float, 
ns_t: float, omega_b_fid: float, omega_cdm_fid: float, H0_fid: float, As_fid: float, 
m_nu_fid: float, mus_arr: npt.NDArray, N_eff_deviation: float):
    
    ''' 
    This function allows one to compute df/dx where x is either Omega_bh2, Omega_ch2, mnu, As or H0. 
    This function only works for a single value of z. 
    df/dx is returned. 
    param = 1 gets df/dH0
    param = 2 gets df/dMnu 
    param = 3 gets df/dObh
    param = 4 gets df/dOch 
    param = 9 gets df/dAs 
    param = 10 gets df/dNeff
    param = 11 gets df_dns
    '''

    if param == cosmo_variable.H0: 
        lowf = growth_rate(omega_b_t, omega_cdm_t, H0_t-delta, As_t, m_nu_t, neutrino_hierarchy, zed, 
        kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        highf = growth_rate(omega_b_t, omega_cdm_t, H0_t+delta, As_t, m_nu_t, neutrino_hierarchy, 
        zed, kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        df_dH = (highf - lowf)/(2*delta)
        return df_dH

    elif param == cosmo_variable.mnu:
        lowf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t-delta, neutrino_hierarchy, zed, 
        kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        highf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t+delta, neutrino_hierarchy, 
        zed, kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        df_dmnu = (highf - lowf)/(2*delta)
        return df_dmnu

    elif param == cosmo_variable.Obh2:
        lowf = growth_rate(omega_b_t-delta, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, zed, 
        kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        highf = growth_rate(omega_b_t+delta, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, 
        zed, kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        df_dob = (highf - lowf)/(2*delta)
        return df_dob

    elif param == cosmo_variable.Och2:
        lowf = growth_rate(omega_b_t, omega_cdm_t-delta, H0_t, As_t, m_nu_t, neutrino_hierarchy, zed, 
        kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        highf = growth_rate(omega_b_t, omega_cdm_t+delta, H0_t, As_t, m_nu_t, neutrino_hierarchy, 
        zed, kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        df_doc = (highf - lowf)/(2*delta)
        return df_doc

    elif param == cosmo_variable.As:
        lowf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t-delta, m_nu_t, neutrino_hierarchy, zed, 
        kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        highf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t+delta, m_nu_t, neutrino_hierarchy, 
        zed, kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        df_da = (highf - lowf)/(2*delta)
        return df_da

    elif param == cosmo_variable.Neff: # Neff
        lowf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, zed, 
        kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, -N_eff_deviation)[0]

        highf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, 
        zed, kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        df_da = (highf - lowf)/(2*delta)
        return df_da

    elif param == cosmo_variable.n_s: # n_s
        
        lowf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, zed, 
        kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t-delta, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        highf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, 
        zed, kmin, kmax, knum, d_a, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t+delta, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_arr, N_eff_deviation)[0]

        df_da = (highf - lowf)/(2*delta)
        return df_da

    else:
        raise Exception('param value (o) is probably not correct (derivative_growth_rate())')

# ---------------------------------------------------------------------------------------------------------


# get galaxy galaxy power spectrum / P_mm  (prefactor)
def gg_redshift_s(bg: float, rg: float, f_t: npt.NDArray, sigmag: float, 
mu: float, ks: npt.NDArray, q_para: float, q_perp: float): 
    'Function to get the prefactor for P_gg(k_fid).'  
    Dg2 = 1/(1 + ((ks*sigmag*mu)**2)/2)
    res = (bg**2)*Dg2/(q_para*(q_perp**2))
    res += (2*rg*(mu**2)*bg*f_t)*Dg2/(q_para*(q_perp**2))
    res += ((mu**4)*(f_t**2))*Dg2/(q_para*(q_perp**2))
    return res


# get galaxy div velocity power spectrum / P_mm  (prefactor)
def gu_redshift_s(bg: float, rg: float, f_t: npt.NDArray, sigmag: float, sigmau: float,
mu: float, ks: npt.NDArray, z: float, H0_t: float, omegam_t: float, omegalambda_t: float, 
q_para: float, q_perp: float): 
    'Function to get the prefactor for P_gu(k).'  
    Dg = 1/(np.sqrt(1 + ((ks*sigmag*mu)**2)/2))
    a = 1.0/(1+z)
    H_z_t = H0_t*np.sqrt( omegam_t*((1.0+z)**3) + omegalambda_t )
    Dmu = np.sinc(ks*sigmau/np.pi)   
    res = a*H_z_t*mu*(rg*bg*f_t)*Dmu*Dg/(ks*q_para*(q_perp**2))
    res += a*H_z_t*mu*((mu*f_t)**2)*Dmu*Dg/(ks*q_para*(q_perp**2))            
    return res


# div v. div v. power spectrum / P_mm (true for linear case)  - (need to multiply by P_mm)
def uu_redshift_s(sigmau: float, mu: float, f_t: npt.NDArray, ks: npt.NDArray, z: float, 
H0_t: float, omegam_t: float, omegalambda_t: float, q_para: float, q_perp: float):  
    'Function to get the prefactor for P_uu(k).'  
    a = 1.0/(1+z)
    H_z_t = H0_t*np.sqrt( omegam_t*((1.0+z)**3) + omegalambda_t )
    Dmu = np.sinc(ks*sigmau/np.pi)
    res = ((a*H_z_t*mu*f_t/ks)**2)*(Dmu**2)/(q_para*(q_perp**2))
    return res

# ---------------------------------------------------------------------------------------------------------


# function to compute the derivative of real k (k = k(k_fid, z, cosmological parameters...)) with respect to cosmological parameters 
def dk_obs_dx(param: cosmo_variable, dF_dx: float, dq_perp_dx: float, mu: float, k: npt.NDArray, F: float, q_perp: float):
    '''
    Function to compute the derivative of k(k_fid, mu_fid, F_distortion_ratio) with respect
    to H0, Obh2, Och2 or Mnu.
    '''
    if param == cosmo_variable.H0 or param == cosmo_variable.mnu or param == cosmo_variable.Obh2 or param == cosmo_variable.Och2: #H0, Mnu, Obh, Och respectively

        p1 = np.sqrt( 1.0 + (mu**2)*( 1.0/(F**2)  - 1.0 ) )
        p2 = 1.0/p1
        res = -1.0*((k)/(q_perp**2))*p1*dq_perp_dx - ((k*(mu**2))/(q_perp*(F**3)))*p2*dF_dx
        return res
    

    else: # Any other derivatives are always zero 
        msg = 'param value (o) is probably not correct / derivative is zero. (dk_obs_dx())'
        logger.error(msg)   
        raise (ValueError)



# function to compute the derivative of the distortion ratio F with respect to cosmological parameters
def dF_distortion_dx(param: cosmo_variable, F: float, H0_t: float, z: float, Om_t: float, H0_fid: float, Om_fid: float, q_perp: float, q_para: float):
    '''
    Function to compute the derivative of F = q_parallel/q_perp with respect
    to H0, Obh2, Och2 or Mnu (all other derivatives = 0).
    '''
    
    dq_para_dx = dq_para_distortion_dx(param, H0_t, H0_fid, Om_t, Om_fid, z) 
    dq_perp_dx = dq_perp_distortion_dx(param, H0_t, H0_fid, Om_t, Om_fid, z) 
    
    dF_dx = 1.0/q_perp * dq_para_dx - (q_para/(q_perp**2))*dq_perp_dx
    
    return dF_dx
        

# function to compute the derivative of real mu with respect to cosmological parameters 
def dmu_obs_dx(dF_dx: float, mu: float, F: float, param: cosmo_variable):
    '''
    Function to compute the derivative of mu(mu_fid, F_distortion_ratio) with respect
    to H0, Obh2, Och2 or Mnu (all other derivatives = 0).
    '''
    if param == cosmo_variable.H0 or param == cosmo_variable.Obh2 or param == cosmo_variable.Och2 or param == cosmo_variable.mnu:
        p1 = 1.0/np.sqrt( 1.0 + (mu**2)*( 1.0/(F**2) - 1.0 ))
        p2 = p1**3
        res = (- (mu/(F**2))*p1 + (((mu**3)/(F**4)))*p2 )*dF_dx 
        return res

    else:
        msg = 'param value (o) is probably not correct / derivative is zero. (dmu_obs_dx()).'
        logger.error(msg)
        raise (ValueError)
        
        
        

# function to compute the derivative of the perpendicular distortion parameter with respect to cosmological parameters 
def dq_perp_distortion_dx(param: cosmo_variable, H0_t: float, H0_fid: float, Om_t: float, 
Om_fid: float, z: float, c: float = 299792.458):
    '''
    Function to compute the derivative of q_perpendicular with respect
    to H0, Obh2, Och2 or Mnu (all other derivatives = 0).
    '''
    D_a_real = angular_diameter_distance(Om_t, H0_t, z)
    D_a_fid = angular_diameter_distance(Om_fid, H0_fid, z)
    h_t = H0_t/100.0
    q_perp = D_a_real/D_a_fid if z != 0 else H0_fid/H0_t 
    E_z = get_Hubble_z(z, Om_t, H0_t)/H0_t 
    
    if param == cosmo_variable.H0: # H0
        
        if z == 0:
            return -1.0 * q_perp / H0_t 
        
        else: 

            funcEz = lambda zi: np.sqrt(Om_t * (1.0 + zi)**3 + (1.0 - Om_t)) 
            functoint = lambda zi: (Om_t * (1.0 + zi)**3 - Om_t) * (1.0 / funcEz(zi)**3)
            integral = quad(functoint, 0.0, z)[0]
            res = -1.0 * q_perp / H0_t + (1.0/(D_a_fid)) * (c / (H0_t**2)) * (1.0/(1.0 + z)) * integral
            return res 
        
    elif param == cosmo_variable.Obh2 or param == cosmo_variable.Och2 or param == cosmo_variable.mnu: # Mnu, Obh, Och
            
        if z == 0:
            return 0.0
        
        else: 
            funcEz = lambda zi: np.sqrt(Om_t * (1.0 + zi)**3 + (1.0 - Om_t)) 
            functoint = lambda zi: ((1.0 + zi)**3 - 1.0) * (1.0 / h_t**2) * -0.5 / funcEz(zi)**3 
            if param == cosmo_variable.mnu:
                functoint = lambda zi: ((1.0 + zi)**3 - 1.0) * (1.0 / (93.14 * h_t**2)) * -0.5 / funcEz(zi)**3 

            integral = quad(functoint, 0.0, z)[0] 
            integral = integral * (c / H0_t) * 1.0/(1.0 + z) 
            
            res = 1.0/(D_a_fid) * integral
        
            return res 
            
    else:
        msg = 'param value (o) is probably not correct / derivative is zero. (dq_perp_distortion_dx()).'
        logger.error(msg)
        raise (ValueError)
    
        
        
# function to compute the derivative of the parallel distortion parameter with respect to cosmological parameters
def dq_para_distortion_dx(param: cosmo_variable, H0_t: float, H0_fid: float, Om_t: float, Om_fid: float, z):
    '''
    Function to compute the derivative of q_parallel with respect
    to H0, Obh2, Och2 or Mnu (all other cosmo derivatives = 0).
    '''

    H_z_t = get_Hubble_z(z, Om_t, H0_t)
    H_z_f = get_Hubble_z(z, Om_fid, H0_fid)
    h = H0_t/100.0
    q_para = H_z_f/H_z_t
    E_z = H_z_t/H0_t
    
    if param == cosmo_variable.H0: #H0
        
        if z == 0:
            return -1.0 * q_para / H0_t
        
        else: 
            
            omegamfunc = Om_t * (1.0 + z)**3 - Om_t 
            res = -1.0 * q_para / H0_t * (1.0 - 1.0/E_z**2 * omegamfunc) 
            return res

    elif param == cosmo_variable.Obh2 or param == cosmo_variable.Och2 or param == cosmo_variable.mnu: # Mnu, Obh or Och

        if z == 0:
            return 0.0
        
        else: 
            dE_dx = (1.0*(1.0 + z)**3 - 1.0) * (1.0/(h**2)) * 1.0/E_z  * 0.5 
            if param == cosmo_variable.mnu:
                dE_dx = (1.0*(1.0 + z)**3 - 1.0) * (1.0/(93.14*h**2)) * 1.0/E_z * 0.5 
                
            res = - q_para / H_z_t * dE_dx * H0_t
            return res

    else:
        msg = 'param value (o) is probably not correct (dq_para_distortion_dx()).'
        logger.error(msg)
        raise (ValueError)
    
   
# get derivatives of galaxy galaxy power spectrum w.r.t. relevant parameter (need to pass in P_mm and dP_dx)
def dP_gg_dx(param: cosmo_variable, bg: float, rg: float, f_t: npt.NDArray, df_dx_t: npt.NDArray, mu_obs: float, k_obs: npt.NDArray, 
sigmag: float, P_mm_t: npt.NDArray, dP_mm_dx_t: npt.NDArray, z: float, H0_t: float, dmu_obs_dx: float, dk_obs_dx: npt.NDArray, 
q_para: float, q_perp: float, dq_para_dx: float, dq_perp_dx: float, includeAP: bool = True): 
    ''' 
    Function to semi-analytically calculate dP_gg_dx where x is some parameter.
    Only dP/dx is returned.
    The user needs to pass in P_mm(k), dP_mm_dx(k), f(k), df_dx, 
    where x can be:
        1 = H0
        2 = Mnu
        3 = Obh
        4 = Och
        9 = As
        5 = sigma_g
        6 = galaxy bias 
        7 = r_g
        10 = Neff
        11 = ns
    THe user also needs to pass in dk_real_dx and dmu_real_dx. 
    '''
    Dg2 = 1/(1 + ( (mu_obs*k_obs*sigmag)**2 )/2)

    if param == cosmo_variable.H0: # varying H0
        res = (bg**2)*dP_mm_dx_t*Dg2/(q_para*(q_perp**2))
        res += (bg**2)*P_mm_t*(((k_obs*mu_obs*sigmag)**2)/H0_t)*(Dg2**2)/(q_para*(q_perp**2))

        res += 2*rg*bg*f_t*(mu_obs**2)*dP_mm_dx_t*Dg2/(q_para*(q_perp**2))
        res += (2*rg*bg*f_t*(mu_obs**2))*P_mm_t*(((k_obs*mu_obs*sigmag)**2)/H0_t)*(Dg2**2)/(q_para*(q_perp**2)) 
        res += (2*rg*bg*df_dx_t*(mu_obs**2))*P_mm_t*Dg2/(q_para*(q_perp**2))

        res += (f_t**2)*(mu_obs**4)*Dg2*dP_mm_dx_t/(q_para*(q_perp**2))
        res += (f_t**2)*(mu_obs**4)*P_mm_t*(((k_obs*mu_obs*sigmag)**2)/H0_t)*(Dg2**2)/(q_para*(q_perp**2)) 
        res += 2*f_t*(mu_obs**4)*(df_dx_t)*P_mm_t*Dg2/(q_para*(q_perp**2))

        if includeAP: 
        #res += (all extra terms due to AP Effect need to be added here)
            res += 4.0*dmu_obs_dx*(bg*rg*f_t*mu_obs)*Dg2*P_mm_t/(q_para*(q_perp**2))
            res += 4.0*dmu_obs_dx*((mu_obs**3)*(f_t**2))*Dg2*P_mm_t/(q_para*(q_perp**2))

            res += -1.0*(bg**2)*(sigmag**2)*(Dg2**2)*k_obs*(mu_obs**2)*dk_obs_dx*P_mm_t/(q_para*(q_perp**2))
            res += -1.0*(bg**2)*(sigmag**2)*(Dg2**2)*mu_obs*(k_obs**2)*dmu_obs_dx*P_mm_t/(q_para*(q_perp**2))

            res += -1.0*( 2.0*rg*bg*f_t*(mu_obs**2) )*(sigmag**2)*(Dg2**2)*k_obs*(mu_obs**2)*dk_obs_dx*P_mm_t/(q_para*(q_perp**2))
            res += -1.0*( 2.0*rg*bg*f_t*(mu_obs**2) )*(sigmag**2)*(Dg2**2)*mu_obs*(k_obs**2)*dmu_obs_dx*P_mm_t/(q_para*(q_perp**2))

            res += -1.0*( (f_t**2)*(mu_obs**4) )*(sigmag**2)*(Dg2**2)*k_obs*(mu_obs**2)*dk_obs_dx*P_mm_t/(q_para*(q_perp**2))
            res += -1.0*( (f_t**2)*(mu_obs**4) )*(sigmag**2)*(Dg2**2)*mu_obs*(k_obs**2)*dmu_obs_dx*P_mm_t/(q_para*(q_perp**2))

            res += -1.0*(1.0/((q_perp*q_para)**2))*(dq_para_dx)*( bg**2 + 2.0*rg*bg*f_t*(mu_obs**2) + (f_t**2)*(mu_obs**4))*Dg2*P_mm_t
            res += -1.0*(2.0/((q_para)*(q_perp**3)))*(dq_perp_dx)*( bg**2 + 2.0*rg*bg*f_t*(mu_obs**2) + (f_t**2)*(mu_obs**4))*Dg2*P_mm_t

        return res
    
    elif param == cosmo_variable.Obh2 or param == cosmo_variable.Och2 or param == cosmo_variable.mnu or param == cosmo_variable.As: # varying obh, och, mnu, As
        res = ( bg**2 )*dP_mm_dx_t*Dg2/(q_para*(q_perp**2))
        #res = 0
        res += (2*rg*bg*f_t*(mu_obs**2))*dP_mm_dx_t*Dg2/(q_para*(q_perp**2))
        res += ( 2*rg*bg*df_dx_t*(mu_obs**2) )*P_mm_t*Dg2/(q_para*(q_perp**2))

        res += ((mu_obs**4)*(f_t**2))*dP_mm_dx_t*Dg2/(q_para*(q_perp**2))
        res += ( 2.0*f_t*(mu_obs**4)*(df_dx_t) )*P_mm_t*Dg2/(q_para*(q_perp**2))

        #res += (all extra terms due to AP Effect need to be added here)
        if includeAP:
            res += 4.0*dmu_obs_dx*(bg*rg*f_t*mu_obs)*Dg2*P_mm_t/(q_para*(q_perp**2))
            res += 4.0*dmu_obs_dx*((mu_obs**3)*(f_t**2))*Dg2*P_mm_t/(q_para*(q_perp**2))

            res += -1.0*(bg**2)*(sigmag**2)*(Dg2**2)*k_obs*(mu_obs**2)*dk_obs_dx*P_mm_t/(q_para*(q_perp**2))
            res += -1.0*(bg**2)*(sigmag**2)*(Dg2**2)*mu_obs*(k_obs**2)*dmu_obs_dx*P_mm_t/(q_para*(q_perp**2))

            res += -1.0*( 2.0*rg*bg*f_t*(mu_obs**2) )*(sigmag**2)*(Dg2**2)*k_obs*(mu_obs**2)*dk_obs_dx*P_mm_t/(q_para*(q_perp**2))
            res += -1.0*( 2.0*rg*bg*f_t*(mu_obs**2) )*(sigmag**2)*(Dg2**2)*mu_obs*(k_obs**2)*dmu_obs_dx*P_mm_t/(q_para*(q_perp**2))

            res += -1.0*( (f_t**2)*(mu_obs**4) )*(sigmag**2)*(Dg2**2)*k_obs*(mu_obs**2)*dk_obs_dx*P_mm_t/(q_para*(q_perp**2))
            res += -1.0*( (f_t**2)*(mu_obs**4) )*(sigmag**2)*(Dg2**2)*mu_obs*(k_obs**2)*dmu_obs_dx*P_mm_t/(q_para*(q_perp**2))

            res += -1.0*(1.0/((q_perp*q_para)**2))*(dq_para_dx)*( bg**2 + 2.0*rg*bg*f_t*(mu_obs**2) + (f_t**2)*(mu_obs**4))*Dg2*P_mm_t
            res += -1.0*(2.0/((q_para)*(q_perp**3)))*(dq_perp_dx)*( bg**2 + 2.0*rg*bg*f_t*(mu_obs**2) + (f_t**2)*(mu_obs**4))*Dg2*P_mm_t

        return res

    elif param == cosmo_variable.sigma_g: # varying sigma g
        Dg4 = Dg2**2
        res = -P_mm_t*( bg**2 + 2*rg*bg*f_t*(mu_obs**2) + (mu_obs**4)*(f_t**2))*( (sigmag*((mu_obs*k_obs)**2)) )*Dg4/(q_para*(q_perp**2))
        return res 

    elif param == cosmo_variable.b_g: # varying galaxy bias
        res = (2*bg + 2*rg*f_t*(mu_obs**2))*Dg2*P_mm_t/(q_para*(q_perp**2))
        return res

    elif param == cosmo_variable.r_g: # varying rg
        res = (2*bg*f_t*(mu_obs**2))*Dg2*P_mm_t/(q_para*(q_perp**2))
        return res

    elif param == cosmo_variable.n_s or param == cosmo_variable.Neff: # n_s (11) or Neff (10)

        res = ( bg**2 + 2*rg*bg*f_t*(mu_obs**2) + (mu_obs**4)*(f_t**2) )*dP_mm_dx_t
        res += ( 2.0*rg*bg*(mu_obs**2) + 2.0*f_t*(mu_obs**4)  )*df_dx_t*P_mm_t
        res *= Dg2/(q_para*(q_perp**2))
        return res 

    else:
        msg = 'param value (o) is probably not correct / derivative is zero. (dP_gg_dx())'
        logger.error(msg)
        raise (ValueError)

# get derivatives of galaxy velocity div. power spectrum w.r.t. relevant parameters (need to pass in P_theta-delta and dP_dx)
def dP_gu_dx(param: cosmo_variable, bg: float, rg: float, f_t: npt.NDArray, df_dx_t: npt.NDArray, mu_obs: float, k_obs: npt.NDArray, 
sigmag: float, sigmau: float, P_mm_t: npt.NDArray, dP_mm_dx_t: npt.NDArray, H0_t: float, z: float, omegam_t: float, omegalambda_t: float, 
dk_obs_dx: npt.NDArray, dmu_obs_dx: float, q_para: float, q_perp: float, dq_para_dx: float, dq_perp_dx: float, includeAP: bool = True):
    ''' 
    Function to semi-analytically calculate dP_gu_dx where x is some parameter.
    The user needs to pass in P_mm(k) (P_mtheta), dP_mm_dx(k) (dP_mtheta_dx), f(k), 
    df_dx where x can be:
        1 = H0
        2 = Mnu
        3 = Obh
        4 = Och
        9 = As
        5 = sigma_g
        6 = galaxy bias 
        7 = r_g
        8 = sigma_u
        10 = Neff
        11 = ns 
    The user also needs to pass in dk_real_dx and dmu_real_dx. 
    '''
    
    Dg = (1/(np.sqrt(1 + ((mu_obs*sigmag*k_obs)**2)/2)))
    Du = np.sinc(k_obs*sigmau/np.pi)
    a = 1.0/(1.0+z)
    h_t = H0_t/100.0
    u = np.sqrt( omegam_t*((1+z)**3) + omegalambda_t )
    H_z_t = H0_t*np.sqrt( omegam_t*((1+z)**3) + omegalambda_t )
    if param == cosmo_variable.H0: # varying H0

        res = (a*mu_obs/k_obs)*( rg*bg*f_t + ((mu_obs*f_t)**2))*(P_mm_t)*Dg*Du*(1/u)*(1.0/(q_para*(q_perp**2)))
        res += (a*H_z_t*mu_obs/k_obs)*( rg*bg*f_t + ((mu_obs*f_t)**2))*Dg*Du*dP_mm_dx_t*(1.0/(q_para*(q_perp**2)))
        res += (a*H_z_t*mu_obs/k_obs)*(  rg*bg*df_dx_t + 2*(mu_obs**2)*f_t*df_dx_t  )*Du*Dg*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += (a*H_z_t*mu_obs/k_obs)*( rg*bg*f_t + ((mu_obs*f_t)**2) )*(P_mm_t)*Dg*( np.sinc(sigmau*k_obs/np.pi) - np.cos(sigmau*k_obs) )*(1.0/(q_para*(q_perp**2)))/(H0_t)
        res += (a*H_z_t*mu_obs/k_obs)*( rg*bg*f_t + ((mu_obs*f_t)**2) )*(P_mm_t)*Du*((mu_obs*k_obs*sigmag)**2)*(Dg**3)*(1.0/(q_para*(q_perp**2)))/(2*H0_t)
        # res += extra terms for AP effect
        
        if includeAP: 
            res += -1.0*((a*H_z_t*mu_obs)/(k_obs**2))*dk_obs_dx*(f_t*rg*bg)*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += -1.0*((a*H_z_t*mu_obs)/(k_obs**2))*dk_obs_dx*((f_t**2)*(mu_obs**2))*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += ((a*H_z_t)/(k_obs))*dmu_obs_dx*(f_t*rg*bg)*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += ((a*H_z_t)/(k_obs))*dmu_obs_dx*(3.0*(f_t**2)*(mu_obs**2))*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += -1.0*((a*H_z_t*mu_obs)/(2*k_obs))*(f_t*rg*bg + (f_t**2)*(mu_obs**2))*Du*(Dg**3)*(sigmag**2)*((k_obs**2)*mu_obs*dmu_obs_dx)*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += -1.0*((a*H_z_t*mu_obs)/(2*k_obs))*(f_t*rg*bg + (f_t**2)*(mu_obs**2))*Du*(Dg**3)*(sigmag**2)*((mu_obs**2)*k_obs*dk_obs_dx)*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += ((a*H_z_t*mu_obs)/(k_obs))*(f_t*bg*rg)*Dg*dk_obs_dx*( np.cos(k_obs*sigmau)/k_obs - np.sin(k_obs*sigmau)/((k_obs**2)*sigmau) )*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += ((a*H_z_t*mu_obs)/(k_obs))*((f_t**2)*(mu_obs**2))*Dg*dk_obs_dx*( np.cos(k_obs*sigmau)/(k_obs) - np.sin(k_obs*sigmau)/((k_obs**2)*sigmau) )*P_mm_t*(1.0/(q_para*(q_perp**2)))

            res += -1.0*(1.0/((q_para*q_perp)**2))*dq_para_dx*((a*H_z_t*mu_obs)/(k_obs))*( f_t*bg*rg + (f_t**2)*(mu_obs**2) )*Dg*Du*P_mm_t
            res += -1.0*(2.0/((q_para)*(q_perp**3)))*dq_perp_dx*((a*H_z_t*mu_obs)/(k_obs))*( f_t*bg*rg + (f_t**2)*(mu_obs**2) )*Dg*Du*P_mm_t
        return res  

    elif param == cosmo_variable.Obh2 or param == cosmo_variable.Och2 or param == cosmo_variable.mnu or param == cosmo_variable.As: # varying obh, och, mnu, As
        res = (a*H_z_t*mu_obs/k_obs)*( rg*bg*f_t + ((mu_obs*f_t)**2))*Dg*Du*dP_mm_dx_t*(1.0/(q_para*(q_perp**2)))
        res += (a*H_z_t*mu_obs/k_obs)*(rg*bg*df_dx_t)*Du*P_mm_t*Dg*(1.0/(q_para*(q_perp**2)))
        res += (a*H_z_t*mu_obs/k_obs)*(2*(mu_obs**2))*f_t*df_dx_t*Du*P_mm_t*Dg*(1.0/(q_para*(q_perp**2)))
        
        if param == cosmo_variable.mnu:
            u = np.sqrt( omegam_t*((1+z)**3) + omegalambda_t )
            dH_dx_t = (H0_t/(2*93.14*u*(h_t**2)))*( (1+z)**3 - 1 )
            res += (a*mu_obs/k_obs)*(rg*bg*f_t + ((mu_obs*f_t)**2))*Du*Dg*P_mm_t*(dH_dx_t)*(1.0/(q_para*(q_perp**2)))
            
            if includeAP:
            # + extra terms for AP EFFECT!!!
                res += -1.0*((a*H_z_t*mu_obs)/(k_obs**2))*dk_obs_dx*(f_t*rg*bg)*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += -1.0*((a*H_z_t*mu_obs)/(k_obs**2))*dk_obs_dx*((f_t**2)*(mu_obs**2))*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += ((a*H_z_t)/(k_obs))*dmu_obs_dx*(f_t*rg*bg)*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += ((a*H_z_t)/(k_obs))*dmu_obs_dx*(3.0*(f_t**2)*(mu_obs**2))*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += -1.0*((a*H_z_t*mu_obs)/(2*k_obs))*(f_t*rg*bg + (f_t**2)*(mu_obs**2))*Du*(Dg**3)*(sigmag**2)*((k_obs**2)*mu_obs*dmu_obs_dx)*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += -1.0*((a*H_z_t*mu_obs)/(2*k_obs))*(f_t*rg*bg + (f_t**2)*(mu_obs**2))*Du*(Dg**3)*(sigmag**2)*((mu_obs**2)*k_obs*dk_obs_dx)*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += ((a*H_z_t*mu_obs)/(k_obs))*(f_t*bg*rg)*Dg*dk_obs_dx*( np.cos(k_obs*sigmau)/k_obs - np.sin(k_obs*sigmau)/((k_obs**2)*sigmau) )*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += ((a*H_z_t*mu_obs)/(k_obs))*((f_t**2)*(mu_obs**2))*Dg*dk_obs_dx*( np.cos(k_obs*sigmau)/k_obs - np.sin(k_obs*sigmau)/((k_obs**2)*sigmau) )*P_mm_t*(1.0/(q_para*(q_perp**2)))

                res += -1.0*(1.0/((q_para*q_perp)**2))*dq_para_dx*((a*H_z_t*mu_obs)/(k_obs))*( f_t*bg*rg + (f_t**2)*(mu_obs**2) )*Dg*Du*P_mm_t
                res += -1.0*(2.0/((q_para)*(q_perp**3)))*dq_perp_dx*((a*H_z_t*mu_obs)/(k_obs))*( f_t*bg*rg + (f_t**2)*(mu_obs**2) )*Dg*Du*P_mm_t
                
        elif param == cosmo_variable.Obh2 or param == cosmo_variable.Och2: 
            u = np.sqrt( omegam_t*((1+z)**3) + omegalambda_t )
            dH_dx_t = (H0_t/(2*u*(h_t**2)))*( (1+z)**3 - 1 )
            res += (a*mu_obs/k_obs)*(rg*bg*f_t + ((mu_obs*f_t)**2))*Du*Dg*P_mm_t*(dH_dx_t)*(1.0/(q_para*(q_perp**2)))
            # + extra terms for AP EFFECT!!!
            if includeAP:
                res += -1.0*((a*H_z_t*mu_obs)/(k_obs**2))*dk_obs_dx*(f_t*rg*bg)*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += -1.0*((a*H_z_t*mu_obs)/(k_obs**2))*dk_obs_dx*((f_t**2)*(mu_obs**2))*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += ((a*H_z_t)/(k_obs))*dmu_obs_dx*(f_t*rg*bg)*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += ((a*H_z_t)/(k_obs))*dmu_obs_dx*(3.0*(f_t**2)*(mu_obs**2))*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += -1.0*((a*H_z_t*mu_obs)/(2*k_obs))*(f_t*rg*bg + (f_t**2)*(mu_obs**2))*Du*(Dg**3)*(sigmag**2)*((k_obs**2)*mu_obs*dmu_obs_dx)*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += -1.0*((a*H_z_t*mu_obs)/(2*k_obs))*(f_t*rg*bg + (f_t**2)*(mu_obs**2))*Du*(Dg**3)*(sigmag**2)*((mu_obs**2)*k_obs*dk_obs_dx)*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += ((a*H_z_t*mu_obs)/(k_obs))*(f_t*bg*rg)*Dg*dk_obs_dx*( np.cos(k_obs*sigmau)/k_obs - np.sin(k_obs*sigmau)/((k_obs**2)*sigmau) )*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += ((a*H_z_t*mu_obs)/(k_obs))*((f_t**2)*(mu_obs**2))*Dg*dk_obs_dx*( np.cos(k_obs*sigmau)/k_obs - np.sin(k_obs*sigmau)/((k_obs**2)*sigmau) )*P_mm_t*(1.0/(q_para*(q_perp**2)))

                res += -1.0*(1.0/((q_para*q_perp)**2))*dq_para_dx*((a*H_z_t*mu_obs)/(k_obs))*( f_t*bg*rg + (f_t**2)*(mu_obs**2) )*Dg*Du*P_mm_t
                res += -1.0*(2.0/((q_para)*(q_perp**3)))*dq_perp_dx*((a*H_z_t*mu_obs)/(k_obs))*( f_t*bg*rg + (f_t**2)*(mu_obs**2) )*Dg*Du*P_mm_t

        return res

    elif param == cosmo_variable.sigma_g: # varying sigma g
        res = -1.0*(1.0/(q_para*(q_perp**2)))*(a*H_z_t*mu_obs/(2*k_obs))*(rg*bg*f_t + (mu_obs*f_t)**2)*Du*P_mm_t*(sigmag*(mu_obs*k_obs)**2)/((np.sqrt(1 + ((mu_obs*sigmag*k_obs)**2)/2))**3)
        return res

    elif param == cosmo_variable.b_g: # varying galaxy bias
        res = (1.0/(q_para*(q_perp**2)))*a*H_z_t*mu_obs*rg*f_t*Du*P_mm_t*Dg/k_obs
        return res

    elif param == cosmo_variable.r_g: # varying rg
        res = (1.0/(q_para*(q_perp**2)))*(a*H_z_t*bg*f_t*Du*P_mm_t*mu_obs)*Dg/k_obs
        return res

    elif param == cosmo_variable.sigma_u: # varying sigmau
        res = (1.0/(q_para*(q_perp**2)))*(a*H_z_t*mu_obs/k_obs)*( rg*bg*f_t + (mu_obs*f_t)**2 )*P_mm_t*((k_obs*sigmau*np.cos(k_obs*sigmau)) - (np.sin(k_obs*sigmau)))/(k_obs*(sigmau**2))
        res *= Dg
        return res

    elif param == cosmo_variable.n_s or param == cosmo_variable.Neff: # Neff or ns
        res = ( rg*bg*f_t + (mu_obs*f_t)**2 )*dP_mm_dx_t
        res += ( rg*bg + 2.0*(mu_obs**2)*f_t )*df_dx_t*P_mm_t
        res *= (a*H_z_t*mu_obs/k_obs)*(1.0/(q_para*(q_perp**2)))*Dg
        return res

    else:
        msg = 'param value (o) is probably not correct / derivative is zero. (dP_gu_dx())'
        logger.error(msg)
        raise (ValueError)

# get derivatives of vel div. vel div. power spectrum w.r.t. relevant parameters (need to pass in P_thetatheta and dP_dx)
def dP_uu_dx(param: cosmo_variable, mu_obs: float, k_obs: npt.NDArray, f_t: npt.NDArray, df_dx_t: npt.NDArray, sigmau: float, 
P_mm_t: npt.NDArray, dP_mm_dx_t: npt.NDArray, H0_t: float, z: float, omegam_t: float, omegalambda_t: float, dk_obs_dx: npt.NDArray, dmu_obs_dx: float,
q_para: float, q_perp: float, dq_para_dx: float, dq_perp_dx: float, includeAP: bool = True):
    ''' 
    Function to semi-analytically calculate dP_uu_dx where x is some parameter.
    The user needs to pass in P_mm(k) (P_theta theta), dP_mm_dx(k) (dP_theta theta_dx), 
    f(k), df_dx where x can be:
        1 = H0
        2 = Mnu
        3 = Obh
        4 = Och
        9 = As
        8 = sigma_u
        10 = Neff
        11 = ns 
    THe user also needs to pass in dk_real_dx and dmu_real_dx. 
    '''
    h_t = H0_t/100
    Du2 = (np.sinc(k_obs*sigmau/np.pi))**2
    H_z_t = H0_t*np.sqrt( omegam_t*((1+z)**3) + omegalambda_t )
    a = 1.0/(1.0 + z)
    u = np.sqrt( omegam_t*((1+z)**3) + omegalambda_t )

    if param == cosmo_variable.H0: # varying H0
        res = ((a*mu_obs/k_obs)**2)*((f_t**2))*(Du2)*P_mm_t*( 2*H_z_t*(1/u) )*(1.0/(q_para*(q_perp**2)))
        res += ((a*mu_obs*H_z_t/k_obs)**2)*(2*f_t)*df_dx_t*(Du2)*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += ((a*mu_obs*H_z_t/k_obs)**2)*(f_t**2)*Du2*dP_mm_dx_t*(1.0/(q_para*(q_perp**2)))
        res += ((a*H_z_t*mu_obs/k_obs)**2)*P_mm_t*(f_t**2)*(2*np.sinc(k_obs*sigmau/np.pi))*((np.sinc(k_obs*sigmau/np.pi) - np.cos(sigmau*k_obs))/H0_t)*(1.0/(q_para*(q_perp**2)))
        # res += Extra terms for AP Effect
        if includeAP: 
            res += 2.0*mu_obs*dmu_obs_dx*(((a*H_z_t*f_t)/(k_obs))**2)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += -2.0*(((a*H_z_t*f_t*mu_obs)/(k_obs))**2)*(dk_obs_dx/k_obs)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += 2.0*(((a*H_z_t*f_t*mu_obs)/(k_obs))**2)*np.sqrt(Du2)*dk_obs_dx*P_mm_t*( np.cos(k_obs*sigmau)/(k_obs) - np.sin(k_obs*sigmau)/((k_obs**2)*sigmau) )*(1.0/(q_para*(q_perp**2)))

            res += (-1.0/((q_perp*q_para)**2))*(((a*H_z_t*f_t*mu_obs)/(k_obs))**2)*Du2*P_mm_t*dq_para_dx
            res += (-2.0/(q_para*(q_perp**3)))*(((a*H_z_t*f_t*mu_obs)/(k_obs))**2)*Du2*P_mm_t*dq_perp_dx
        return res

    elif param == cosmo_variable.Obh2 or param == cosmo_variable.Och2 or param == cosmo_variable.mnu or param == cosmo_variable.As: # varying obh, och, mnu, As
        res = ((a*mu_obs*H_z_t/k_obs)**2)*(2*f_t*df_dx_t)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += ((a*mu_obs*H_z_t/k_obs)**2)*((f_t)**2)*Du2*dP_mm_dx_t*(1.0/(q_para*(q_perp**2)))
        
        if param == cosmo_variable.mnu:
            dH_dxsqrd = (H0_t/(2*93.14*u*(h_t**2)))*( (1+z)**3 - 1 )*(2*H_z_t)
            res += ((a*mu_obs/k_obs)**2)*((f_t)**2)*Du2*P_mm_t*(dH_dxsqrd)*(1.0/(q_para*(q_perp**2)))
            # res += EXTRA TERMS AP EFFECT
            if includeAP:
                res += 2.0*mu_obs*dmu_obs_dx*(((a*H_z_t*f_t)/(k_obs))**2)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += -2.0*(((a*H_z_t*f_t*mu_obs)/(k_obs))**2)*(dk_obs_dx/k_obs)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += 2.0*(((a*H_z_t*f_t*mu_obs)/(k_obs))**2)*np.sqrt(Du2)*dk_obs_dx*P_mm_t*( np.cos(k_obs*sigmau)/(k_obs) - np.sin(k_obs*sigmau)/((k_obs**2)*sigmau) )*(1.0/(q_para*(q_perp**2)))

                res += (-1.0/((q_perp*q_para)**2))*(((a*H_z_t*f_t*mu_obs)/(k_obs))**2)*Du2*P_mm_t*dq_para_dx
                res += (-2.0/(q_para*(q_perp**3)))*(((a*H_z_t*f_t*mu_obs)/(k_obs))**2)*Du2*P_mm_t*dq_perp_dx

        elif param == cosmo_variable.Obh2 or param == cosmo_variable.Och2:
            dH_dxsqrd =  (H0_t/(2*u*(h_t**2)))*( (1+z)**3 - 1 )*(2*H_z_t)
            res += ((a*mu_obs/k_obs)**2)*((f_t)**2)*Du2*P_mm_t*(dH_dxsqrd)*(1.0/(q_para*(q_perp**2)))
            # res += extra terms AP EFFECT
            if includeAP:
                res += 2.0*mu_obs*dmu_obs_dx*(((a*H_z_t*f_t)/(k_obs))**2)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += -2.0*(((a*H_z_t*f_t*mu_obs)/(k_obs))**2)*(dk_obs_dx/k_obs)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
                res += 2.0*(((a*H_z_t*f_t*mu_obs)/(k_obs))**2)*np.sqrt(Du2)*dk_obs_dx*P_mm_t*( np.cos(k_obs*sigmau)/(k_obs) - np.sin(k_obs*sigmau)/((k_obs**2)*sigmau) )*(1.0/(q_para*(q_perp**2)))

                res += (-1.0/((q_perp*q_para)**2))*(((a*H_z_t*f_t*mu_obs)/(k_obs))**2)*Du2*P_mm_t*dq_para_dx
                res += (-2.0/(q_para*(q_perp**3)))*(((a*H_z_t*f_t*mu_obs)/(k_obs))**2)*Du2*P_mm_t*dq_perp_dx

        return res

    elif param == cosmo_variable.sigma_u: # varying sigma_u
        res = (1.0/(q_para*(q_perp**2)))*((a*H_z_t*mu_obs*f_t/k_obs)**2)*(2*np.sinc(k_obs*sigmau/np.pi))*(k_obs*sigmau*np.cos(k_obs*sigmau) - np.sin(k_obs*sigmau))*P_mm_t/(k_obs*(sigmau**2))
        return res

    elif param == cosmo_variable.Neff or param == cosmo_variable.n_s: # varying Neff or varying ns
        res = ((a*H_z_t*mu_obs/k_obs)**2)*(f_t**2)*dP_mm_dx_t
        res += 2.0*((a*H_z_t*mu_obs/k_obs)**2)*df_dx_t*f_t*P_mm_t
        res *= (1.0/(q_para*(q_perp**2)))*Du2
        return res   

    else:
        msg = 'param value (o) is probably not correct / derivative is zero. (dP_uu_dx())'
        logger.error(msg)
        raise (ValueError)


# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------


# function to compute derivatives or make a plot of some derivatives for some parameter (given by o), 
# some power spectrum (given by P) and some redshift z.

# this function computes the derivatives of the chosen power spectrum w.r.t. some parameter for all 
# values of mu and 3 different stepsizes. The derivatives here are computed both completely numerically 
# (finite difference method) and in a semi-analytic way (analytic derivatives where possible, 
# otherwise the finite difference method is used) so the user may ensure the they are robust to 
# different methods of computing them and that the choice of stepsize also has minimal 
# effect on the results.

def derivative_power_spectra(o: cosmo_variable, P: RSPS_variable, z: float, plot_or_write2file: str, central_params: list[float],
delta_max_param: float, kmin: float, kmax: float, num_k_points: int, num_mus: int, da: float, neutrino_hier: str, 
lin_or_nonlin: str, kspaceoption: str, tau: float, ns: float):
    """ 
    Function to compute derivatives or make a plot of some derivatives for some parameter (given by o), 
    some power spectrum (given by P) and some redshift z.

    To compute the derivative of chosen power spectrum P w.r.t. some parameter x, set o to the corresponding value of the cosmo_variable enum,
    and P to the corresponding value of the RSPS_variable enum.  
    
    To just make a plot of some of the derivatives, set parameter plot_or_write2file to string 'plot1', 
    to make a plot of the derivatives and their relative difference from each other, set plot_or_write2file to 'plot2',
    else it will write the derivatives to a file.
    central_params should be a list of the central values used for each parameter specified above, in the order:
    omega_b, sum neutrino masses, H0, omega_CDM, As, sigmag*h, bg, rg, sigmau*h.
    Assume N_ur central value is always 0.00641. (such that N_eff = 3.046.)
    kmin and kmax are the next parameters - min. and max. values for power spectrum
    num_k_points - number of k values used
    num_mus - number of points mu (where mu is the cosine of the angle between the observers line of site and the 
    k space vector) between 0 and 1 to compute the derivatives at 
    da - size of step in scalefactor when computing the growth rate (involves a derivative of the 
    power spectrum w.r.t. to scalefactor)
    neutrino_hier - options are only set it to 'inverted' or 'normal' hierarchy for neutrinos - any other string gives a degenerate hierarchy
    lin_or_nonlin - setting this to 'linear' will get the linear matter power spectrum from CLASS (in which case it is 
    exactly equal to the velocity-velocity and galaxy-velocity spectra) where as setting it to 'nonlinear' 
    will retrieve the HALOFIT power spectrum from CLASS
    The derivatives of the power spectra also take into account the Alcock-Paczynski Effect where relevant.  
     
    The central values of all central parameters are taken to be the users 'fiducial' cosmological parameters
    which are equivalent to the true cosmology (thus the distortion parameters/ratios are all equal to one). If the 
    true and fiducial cosmological parameters are not the same then the distortion parameters differ from one, and the 
    fiducial values of k and mu differ to the real values of k and mu.
    The derivatives take into account how mu and k change with respect to changing the real cosmology compared to the 
    fiducial cosmological parameters (which stay constant).
    """


    linear = True # use linear power spectrum, rather than the nuCopter nonlinear power spectra 
    if lin_or_nonlin != 'linear':  
        linear = False
    plot1 = True 
    plot2 = True
    if plot_or_write2file != 'plot1': 
        plot1 = False
    if plot_or_write2file != 'plot2':
        plot2 = False
    num_steps = 7  # number of intervals slices we divide our parameter that we vary into (allows for 
    # 3 step sizes about the central value)
    ks_fiducial = []


    # set up either logarithmic or linear spacing for the ks (depending on what the user prefers, logarithmic
    # is better for plotting a smooth power spectrum on a log scale)
    if kspaceoption == 'log':
        ks_fiducial = np.logspace(np.log10(kmin), np.log10(kmax), num_k_points, base=10) # k values (k in 1/MPC)
    else:
        ks_fiducial = np.linspace(kmin, kmax, num_k_points)

    mus = np.linspace(0, 1, num_mus) # set the mu values 

    # unpack the list of central values for each parameter
    Obh_central, mnu_central, H0_central, Och_central, As_central, sigmag_central, bg_central, rg_central, sigmau_central = central_params
    
    param_variation = [] # initialising a list that will store the values of our parameter we vary
    deltas = np.zeros(num_steps)   # initializing a list that tells us the step sizes for our parameters from
    # the central values of cosmological parameters  

    if o == cosmo_variable.H0: # varying H0
        param_variation = np.linspace(H0_central-delta_max_param, H0_central+delta_max_param, num_steps)
    elif o == cosmo_variable.mnu: # varying sum of neutrino masses
        param_variation = np.linspace(mnu_central-delta_max_param, mnu_central+delta_max_param, num_steps)
    elif o == cosmo_variable.Obh2: # varying omega_b
        param_variation = np.linspace(Obh_central-delta_max_param, Obh_central+delta_max_param, num_steps)
    elif o == cosmo_variable.Och2: # varying omega_cdm
        param_variation = np.linspace(Och_central-delta_max_param, Och_central+delta_max_param, num_steps)
    elif o == cosmo_variable.sigma_g: # varying sigma_g
        param_variation = np.linspace(sigmag_central-delta_max_param, sigmag_central+delta_max_param, num_steps)
    elif o == cosmo_variable.b_g: # galaxy bias
        param_variation = np.linspace(bg_central-delta_max_param, bg_central+delta_max_param, num_steps)
    elif o == cosmo_variable.r_g: # correlation between the velocity and density fields
        param_variation = np.linspace(rg_central-delta_max_param, rg_central+delta_max_param, num_steps)
    elif o == cosmo_variable.sigma_u: # sigma_u
        param_variation = np.linspace(sigmau_central-delta_max_param, sigmau_central+delta_max_param, num_steps)
    elif o == cosmo_variable.As: # As
        param_variation = np.linspace(As_central-delta_max_param, As_central+delta_max_param, num_steps)
    elif o == cosmo_variable.Neff: #N_eff
        param_variation = np.linspace(-delta_max_param, delta_max_param, num_steps)
    elif o == cosmo_variable.n_s: #n_s
        param_variation = np.linspace(ns-delta_max_param, ns+delta_max_param, num_steps)
    
    else:
        msg = 'Parameter chosen is not an option (in derivative_power_spectra()).'
        logger.error(msg)
        raise (ValueError)

    for i in range(num_steps): # get differences between central value of parameter we are varying and shifted value
            deltas[i] = abs(param_variation[int((num_steps-1)/2.0)] - param_variation[i])
    deltas = np.delete(deltas, int((num_steps-1)/2.0)) 


    # initialise arrays to store results to later -----------------------------------------------

    # store matter power spectrum for all k_obs, mu_obs, and values of varying parameter
    matter_power_spectrum_arr = np.zeros((num_steps, num_mus, num_k_points))

    # store observed ks
    ks_obs_arr = np.zeros((num_steps, num_mus, num_k_points))

    # store observed mus
    mus_obs_arr = np.zeros((num_steps, num_mus))

    # store growth rates
    growth_rates_arr = np.zeros((num_steps, num_mus, num_k_points))

    # store redshift space power spectrum for all mu, k, and values of varying parameter
    results_redshiftspacepowerspectrum_arr = np.zeros((num_steps, num_mus, num_k_points))  

    # store matter power spectrum derivatives for all k, and values of varying parameter
    matter_central_diffs = np.zeros((int((num_steps-1)/2), num_mus, num_k_points))

    # store redshift space power spectrum derivatives for all mu and k, and stepsizes 
    # (only simple finite diff method results)
    results_findiff_derivatives = np.zeros((int((num_steps-1)/2), num_mus, num_k_points))  

    # store redshift space power spectrum derivatives for all mu and k and stepsizes ( semi analytic method results)
    results_findiff_semi_analytic_derivatives = np.zeros((int((num_steps-1)/2), num_mus, num_k_points))

    # ------------------------------------------------------------------------------------------


    # initializing values for variables

    h = H0_central/100.0 # need to initialise little h
    H0 = H0_central # initialising all other cosmo parameters 
    Obh = Obh_central
    Och = Och_central
    As = As_central
    mnu = mnu_central
    bg = bg_central
    rg = rg_central
    sigmag = (sigmag_central)/h # MPC
    sigmau = sigmau_central/h # MPC     
    N_eff_deviation = 0.0 
    delta_mnu_max = 0.0
    ns_val = ns 

    # calculate power spectrum while varying parameter of choice
    for i in np.arange(num_steps): 
        # first just get delta-delta/ delta-theta /theta-theta power spectrum

        # changing parameter that is varied for the loop 
        if o == cosmo_variable.H0:
            H0 = param_variation[i]
            h = param_variation[i]/100.0
            sigmag = sigmag_central/h
            sigmau = sigmau_central/h
        elif o == cosmo_variable.mnu:
            mnu = param_variation[i]
            delta_mnu_max = delta_max_param
        elif o == cosmo_variable.Obh2:
            Obh = param_variation[i]
        elif o == cosmo_variable.Och2:
            Och = param_variation[i]
        elif o == cosmo_variable.sigma_g:
            sigmag = param_variation[i]/h # MPC
        elif o == cosmo_variable.b_g:
            bg = param_variation[i]
        elif o == cosmo_variable.r_g: 
            rg = param_variation[i]
        elif o == cosmo_variable.sigma_u: 
            sigmau = param_variation[i]/h # MPC
        elif o == cosmo_variable.As:
            As = param_variation[i]
        elif o == cosmo_variable.Neff:
            N_eff_deviation = param_variation[i]
        elif o == cosmo_variable.n_s:
            ns_val = param_variation[i]

        # get omega_m
        omegam = get_Om_0(Obh, Och, mnu, H0)
        
        
        Pk_cb, ks, mus = run_class(Obh, Och, H0, As, mnu, neutrino_hier, [z], kmin, kmax, num_k_points,
        delta_mnu_max, mnu_central, kspaceoption, tau, ns_val, Obh_central, Och_central, H0_central, As_central, mnu_central, mus, lin_or_nonlin,
        N_eff_deviation)
        #print(mus.shape)

        f = growth_rate(Obh, Och, H0, As, mnu, neutrino_hier, z, kmin, kmax, num_k_points, da,
        delta_mnu_max, mnu_central, kspaceoption, tau, ns_val, Obh_central, Och_central, H0_central, As_central, mnu_central, mus,
        N_eff_deviation)[0]


        if Pk_cb.shape == (num_mus, num_k_points, 1) and ks.shape == (num_mus, num_k_points, 1) and f.shape == (num_mus, num_k_points):

            matter_power_spectrum_arr[i,:,:] = Pk_cb[:,:,0] # storing the matter power spectrum for all ks and mus (real)
            ks_obs_arr[i,:,:] = ks[:,:,0]
            mus_obs_arr[i,:] = mus[:,0]
            growth_rates_arr[i,:,:] = f


        elif Pk_cb.shape == (num_k_points, 1) and ks.shape == (num_k_points,) and f.shape == (num_k_points,):

            mus = mus.flatten()
            for muu in np.arange(num_mus):
                matter_power_spectrum_arr[i,muu,:] = Pk_cb[:,0] # just storing the same matter power spectrum since the k modes don't change with mu
                ks_obs_arr[i,muu,:] = ks 
                growth_rates_arr[i,muu,:] = f 
                mus_obs_arr[i,:] = mus


        else:
            print('shape Pk_cb: ', Pk_cb.shape)
            print('shape ks: ', ks.shape)
            print('shape mus:', mus.shape)
            print('shape f: ', f.shape)
            msg = 'The shapes of the Pk_cb, ks, mus, and f arrays are not what we expect (derivative_power_spectra())'
            logger.error(msg)
            raise ValueError()

            

        # now compute the redshift space power spectrum for all the needs ks and mus 
        q_para = distortion_parallel(As, Obh, Och, H0, mnu, As_central, Obh_central, Och_central, H0_central, mnu_central, z)
        q_perp = distortion_perpendicular(As, Obh, Och, H0, mnu, As_central, Obh_central, Och_central, H0_central, mnu_central, z)
        for j in np.arange(len(mus)): # calculate redshift space power spectrum of choice for all mus 

            if P == RSPS_variable.P_gg: # redshift galaxy galaxy
            
                pkrss = matter_power_spectrum_arr[i,j,:]*(gg_redshift_s(bg, rg, growth_rates_arr[i, j, :], 
                sigmag, mus_obs_arr[i,j], ks_obs_arr[i,j,:], q_para, q_perp)) 
                results_redshiftspacepowerspectrum_arr[i,j,:] = pkrss

            elif P == RSPS_variable.P_ug: # redshift galaxy velocity
                
                pkrss = matter_power_spectrum_arr[i,j,:]*(gu_redshift_s(bg, rg, growth_rates_arr[i, j, :], 
                sigmag, sigmau, mus_obs_arr[i,j], ks_obs_arr[i,j,:], z, H0, omegam, 1.0-omegam, q_para, q_perp))
                results_redshiftspacepowerspectrum_arr[i,j,:] = pkrss 

            elif P == RSPS_variable.P_uu: # redshift velocity velocity 
                
                pkrss = matter_power_spectrum_arr[i,j,:]*(uu_redshift_s(sigmau, mus_obs_arr[i, j], 
                growth_rates_arr[i,j,:], ks_obs_arr[i,j,:], z, H0, omegam, 1.0-omegam, q_para, q_perp)) 
                results_redshiftspacepowerspectrum_arr[i,j,:] = pkrss


    # --------------------------------------------------------------------------------------- 

    # get derivatives with just simple central difference method for matter power spectrum 
    if o in [cosmo_variable.H0, cosmo_variable.As, cosmo_variable.Och2, 
             cosmo_variable.Obh2, cosmo_variable.mnu, cosmo_variable.n_s, cosmo_variable.Neff]: # derivatives are zero if the varied paramater is not a cosmological parameter
        for j in np.arange((int((num_steps-1)/2))):
            matter_central_diffs[j,:,:] = (matter_power_spectrum_arr[num_steps-1-j,:] 
            - matter_power_spectrum_arr[j,:,:])/(2.0*deltas[num_steps-2-j])
                   
    # ---------------------------------------------------------------------------------------

    # get derivatives with just simple central difference method for redshift 
    # space power spectrum of choice (no expansion/no analytic derivatives)
    for j in np.arange(int((num_steps-1)/2)):
        if o == cosmo_variable.sigma_g or o == cosmo_variable.sigma_u: # need to include 1/h in delta
            results_findiff_derivatives[j,:,:] = (results_redshiftspacepowerspectrum_arr[num_steps-1-j,:,:] 
            - results_redshiftspacepowerspectrum_arr[j,:,:])/(2.0*deltas[num_steps-2-j]/(H0_central/100.0))
        else:
            results_findiff_derivatives[j,:,:] = (results_redshiftspacepowerspectrum_arr[num_steps-1-j,:,:] 
            - results_redshiftspacepowerspectrum_arr[j,:,:])/(2.0*deltas[num_steps-2-j])

    # ---------------------------------------------------------------------------------------

    # get derivatives with semi analytic / fully analytic method (where possible) for
    # redshift space power spectrum
    # need to take derivatives about the 'central' values for all parameters that can be varied
        
    # set the correct central values for little h, sigmag, sigmau, omegam, etc. 
    h = (H0_central/100)
    H0 = H0_central
    sigmag = sigmag_central/h
    sigmau = sigmau_central/h
    bg = bg_central
    rg = rg_central
    Obh = Obh_central
    Och = Och_central
    mnu = mnu_central
    As = As_central
    omegam = get_Om_0(Obh, Och, mnu, H0)
    N_eff_deviation = 0
    delta_mnu_max = 0.0
    ns_val = ns 
    if o == cosmo_variable.mnu:
        delta_mnu_max = delta_max_param

    # get the central values for the growth rate as a function of (k, u) which should = (k, u) fiducial
    f = growth_rates_arr[3, :, :]
    # we expect that all the ks don't change with mu when the fiducial cosmology equals the true cosmology 

    # get derivatives with the three different step sizes
    for i in np.arange(int((num_steps-1)/2)):

        stepsize = deltas[num_steps-2-i] 
        
        df_dx = np.zeros((num_mus, num_k_points))
        F = 1.0
        dF_dx = 0.0 
        q_perp = 1.0
        q_para = 1.0
        q_parallel = 1.0
        dq_perp_dx = 0.0  
        dq_para_dx = 0.0


        if o in [cosmo_variable.H0, cosmo_variable.As, cosmo_variable.Och2, 
                  cosmo_variable.Obh2, cosmo_variable.mnu, cosmo_variable.n_s, cosmo_variable.Neff]: # varying cosmological parameter x = non-zero df_dx
            # getting derivative of the growth rate w.r.t. parameter being varied
            if o in [cosmo_variable.H0, cosmo_variable.Och2, cosmo_variable.Obh2, cosmo_variable.mnu, cosmo_variable.As]:
                df_dx = derivative_growth_rate(o, stepsize, Obh, Och, H0, As, mnu, neutrino_hier, 
                z, kmin, kmax, num_k_points, da, linear, delta_mnu_max, 
                mnu_central, kspaceoption, tau, ns, Obh_central, Och_central, H0_central, As_central, mnu_central,
                mus, N_eff_deviation)
            elif o == cosmo_variable.Neff or o == cosmo_variable.n_s:
                df_dx[:,:] = derivative_growth_rate(o, stepsize, Obh, Och, H0, As, mnu, neutrino_hier, 
                z, kmin, kmax, num_k_points, da, linear, delta_mnu_max, 
                mnu_central, kspaceoption, tau, ns, Obh_central, Och_central, H0_central, As_central, mnu_central,
                mus, N_eff_deviation)
            if o in [cosmo_variable.H0, cosmo_variable.Och2, cosmo_variable.Obh2, cosmo_variable.mnu]:
                
                Om_central = get_Om_0(Obh_central, Och_central, mnu_central, H0_central) 
                dF_dx = dF_distortion_dx(o, F, H0, z, omegam, H0_central, Om_central, q_perp, q_para)
                dq_perp_dx = dq_perp_distortion_dx(o, H0, H0_central, omegam, Om_central, z)
                dq_para_dx = dq_para_distortion_dx(o, H0, H0_central, omegam, Om_central, z)

        else:
            df_dx = np.zeros((num_mus, num_k_points)) # growth rate only changes w.r.t. cosmological parameters
            dF_dx = 0.0
            dq_perp_dx = 0.0 
            dq_para_dx = 0.0


        for muu in np.arange(len(mus)): 


            dkobs_dx = 0
            dmuobs_dx = 0
                
            if o in [cosmo_variable.H0, cosmo_variable.Och2, cosmo_variable.Obh2, cosmo_variable.mnu]: # k and mu only change w.r.t. these cosmological parameters
                
                dkobs_dx = dk_obs_dx(o, dF_dx, dq_perp_dx, mus[muu], ks_fiducial, F, q_perp)
                dmuobs_dx =dmu_obs_dx(dF_dx, mus[muu], F, o)
                
            else:
                dkobs_dx = 0
                dmuobs_dx = 0
                
            if P == RSPS_variable.P_gg:
    
                results_findiff_semi_analytic_derivatives[i, muu, :] = dP_gg_dx(o, bg, rg, growth_rates_arr[3, muu, :], df_dx[muu,:],
                mus_obs_arr[3,muu], ks_obs_arr[3, muu, :], sigmag, matter_power_spectrum_arr[3,muu,:], matter_central_diffs[i, muu,:],
                z, H0, dmuobs_dx, dkobs_dx, q_parallel, q_perp, dq_para_dx, dq_perp_dx, includeAP=True)
            
            if P == RSPS_variable.P_ug:
                results_findiff_semi_analytic_derivatives[i, muu, :] = dP_gu_dx(o, bg, rg, growth_rates_arr[3,muu,:], df_dx[muu,:], 
                mus_obs_arr[3,muu], ks_obs_arr[3,muu,:], sigmag, sigmau, matter_power_spectrum_arr[3, muu, :], 
                matter_central_diffs[i, muu, :], H0, z, omegam, 1.0-omegam, dkobs_dx, dmuobs_dx, q_parallel, q_perp, dq_para_dx, dq_perp_dx, includeAP=True)
                    
            if P == RSPS_variable.P_uu:
                results_findiff_semi_analytic_derivatives[i, muu, :] = dP_uu_dx(o, mus_obs_arr[3, muu], ks_obs_arr[3, muu, :], 
                growth_rates_arr[3,muu,:], df_dx[muu,:], sigmau, matter_power_spectrum_arr[3, muu, :], matter_central_diffs[i, muu, :],
                H0, z, omegam, 1.0-omegam, dkobs_dx, dmuobs_dx, q_parallel, q_perp, dq_para_dx, dq_perp_dx, includeAP=True)




    # --------------------------finished computing derivatives ------------------------------------------------------------------------------
    
    mapping_P = {RSPS_variable.P_gg: 0, RSPS_variable.P_ug: 1, RSPS_variable.P_uu: 2} 
    mapping_var = {cosmo_variable.H0: 0, cosmo_variable.mnu: 1, cosmo_variable.Obh2: 2, cosmo_variable.Och2: 3, 
    cosmo_variable.sigma_g: 4, cosmo_variable.b_g: 5, cosmo_variable.r_g: 6, cosmo_variable.sigma_u: 7, cosmo_variable.As: 8, 
    cosmo_variable.Neff: 9, cosmo_variable.n_s: 10} 

    if plot1 == True:  # plot some of the derivatives

        ks_obs_arr /= (H0_central/100.0) # put ks into units of k/h MPc
        results_findiff_derivatives *= (H0_central/100.0)**3 # putting into units of (MPC/h)^3/x
        results_findiff_semi_analytic_derivatives *= (H0_central/100.0)**3 # putting into units of (MPC/h)^3/x, x being the units of the parameter
        ks_fiducial /= (H0_central/100.0) # put ks into units of k/h MPc
        # that is being varied 

        # getting some mus we would like to plot 
        mu_indices = np.array(([np.argmin(np.abs(mus - 0.1)), np.argmin(np.abs(mus - 0.5)), np.argmin(np.abs(mus - 0.9)) ]))  
         

        # set up figure for plotting derivatives
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)


        # loop through derivative results using 1st method (just finite difference and plot)
        for j in np.arange(int((num_steps-1)/2)):

            if o == cosmo_variable.H0:
                label = (r"$\Delta H_0 = %.5f $ - M1" %  (deltas[j]) ) 
            if o == cosmo_variable.mnu:
                label = (r"$\Delta \sum m_{\nu} = %.6f $ - M1" %  (deltas[j]) ) 
            if o == cosmo_variable.Obh2:
                label = (r"$\Delta \Omega_bh^2 = %.6f $ - M1" %  (deltas[j]) ) 
            if o == cosmo_variable.Och2:
                label = (r"$\Delta \Omega_{cdm}h^2 = %.3f $ - M1" %  (deltas[j]) ) 
            if o == cosmo_variable.sigma_g:
                label = (r"$\Delta \sigma_g = %.3f $ - M1" %  (deltas[j]) ) 
            if o == cosmo_variable.b_g:
                label = (r"$\Delta b_g=%.3f $ - M1" %  (deltas[j]) ) 
            if o == cosmo_variable.r_g:
                label = (r"$\Delta r_g=%.3f $ - M1" %  (deltas[j]) ) 
            if o == cosmo_variable.sigma_u:
                label = (r"$\Delta \sigma_{u}=%.3f $ - M1" %  (deltas[j]) )
            if o == cosmo_variable.As:
                label = (r"$\Delta A_s=%.4f \times 10^{-9} $ - M1" %  (deltas[j]*1e9) ) 
            if o == cosmo_variable.Neff:
                label = (r"$\Delta N_{eff}=3.046+%.4f $ - M1" %  (deltas[j]) ) 
            if o == cosmo_variable.n_s:
                label = (r'$ \Delta n_s=%.4f $ - M1' % (deltas[j]) )


            dP_dtheta = results_findiff_derivatives[j, mu_indices[0], :]
            ax1.semilogx(ks_obs_arr[3,mu_indices[0],:], dP_dtheta, label = label )
            dP_dtheta = results_findiff_derivatives[j, mu_indices[1], :]
            ax2.semilogx(ks_obs_arr[3,mu_indices[1],:], dP_dtheta, label = label )
            dP_dtheta = results_findiff_derivatives[j, mu_indices[2], :]
            ax3.semilogx(ks_obs_arr[3,mu_indices[2],:], dP_dtheta, label = label )


        # loop through derivative results using 2nd method (finite diff. + analytic derivatives)

        for j in np.arange(int((num_steps-1)/2)):

            if o == cosmo_variable.H0:
                label = (r"$\Delta H_0 = %.5f $ - M2" %  (deltas[j]) ) 
            if o == cosmo_variable.mnu:
                label = (r"$\Delta \sum m_{\nu}  = %.6f $ - M2" %  (deltas[j]) ) 
            if o == cosmo_variable.Obh2:
                label = (r"$\Delta \Omega_bh^2 = %.6f $ - M2" %  ( deltas[j])) 
            if o == cosmo_variable.Och2:
                label = (r"$\Delta \Omega_{cdm}h^2 = %.3f $ - M2" %  (deltas[j]) ) 
            if o == cosmo_variable.sigma_g:
                label = (r"$\Delta \sigma_g = %.3f $ - M2" %  (deltas[j]) ) 
            if o == cosmo_variable.b_g:
                label = (r"$\Delta b_g=%.3f $ - M2" %  (deltas[j]) ) 
            if o == cosmo_variable.r_g:
                label = (r"$\Delta r_g=%.3f $ - M2" %  (deltas[j]) ) 
            if o == cosmo_variable.sigma_u:
                label = (r"$\Delta \sigma_{u}=%.3f $ - M2" %  (deltas[j]) ) 
            if o == cosmo_variable.As:
                label = (r"$\Delta A_s=%.4f \times 10^{-9} $ - M2" %  (deltas[j]*1e9) ) 
            if o == cosmo_variable.Neff:
                label = (r"$\Delta N_{eff}=3.046+%.4f $ - M2" %  (deltas[j]) ) 
            if o == cosmo_variable.n_s:
                label = (r'$ \Delta n_s=%.4f $ - M2' % (deltas[j]) )

            
            dP_dtheta = results_findiff_semi_analytic_derivatives[j, mu_indices[0], :]
            ax1.semilogx(ks_obs_arr[3,mu_indices[0],:], dP_dtheta, linestyle = '--', label = label )
            dP_dtheta = results_findiff_semi_analytic_derivatives[j, mu_indices[1], :]
            ax2.semilogx(ks_obs_arr[3,mu_indices[1],:], dP_dtheta, linestyle = '--', label = label )
            dP_dtheta = results_findiff_semi_analytic_derivatives[j, mu_indices[2], :]
            ax3.semilogx(ks_obs_arr[3,mu_indices[2],:], dP_dtheta, linestyle = '--', label = label )


        ylabels = [
        r"$ \frac{dP(k, \mu)}{dH_0}$ ", r"$ \frac{dP(k, \mu)}{d\sum m_{\nu} }$", 
        r"$ \frac{dP(k, \mu)}{d\Omega_bh^2}$" , r"$ \frac{dP(k, \mu)}{d\Omega_{cdm}h^2}$", 
        r"$ \frac{dP(k, \mu)}{d\sigma_g}$", r"$ \frac{dP(k, \mu)}{db_g}$",
        r"$ \frac{dP(k, \mu)}{dr_g}$",r"$ \frac{dP(k, \mu)}{d\sigma_{u}}$" , r"$ \frac{dP(k, \mu)}{dA_s}$",
        r"$ \frac{dP(k, \mu)}{dN_{eff}}$",'', '', r'$ \frac{dP(k, \mu)}{dn_s} $'
                  ]

        xlabel = r"$k (\mathrm{h Mpc})^{-1}$"
        ylabel = ylabels[mapping_var[o]]

        # set up plot labels 
        ax1.set_xlim([np.min(ks), np.max(ks)])
        ax1.set_xlabel(xlabel, fontsize = 14)
        ax1.set_ylabel(ylabel, fontsize = 14)
        ax1.text(5e-4, 0, r"$ \mu = %.2f $" %  (mus[mu_indices[0]])  )
        ax2.set_xlim([np.min(ks), np.max(ks)])
        ax2.set_xlabel(xlabel, fontsize = 14)
        ax2.set_ylabel(ylabel, fontsize = 14)
        ax2.text(5e-4, 0, r"$ \mu = %.2f $" %  (mus[mu_indices[1]])  )
        ax3.set_xlim([np.min(ks), np.max(ks)])
        ax3.set_xlabel(xlabel, fontsize = 14)
        ax3.set_ylabel(ylabel, fontsize = 14)
        ax3.text(5e-4, 0, r"$ \mu = %.2f $" %  (mus[mu_indices[2]])  )


        ax1.legend(loc = 'upper right', fontsize = 'small')

        parameters2bevaried = [
        r"$H_0$ about %.3f km/s/MPC" % H0_central,
        r"$\sum m_{\nu} $ about $ %.4f $ eV" % mnu_central,
        r"$\Omega_{b}h^2$ about %.5f" % Obh_central,
        r"$\Omega_{cdm}h^2$ about %.5f" % Och_central,
        r"$\sigma_g$ about $\frac{4.24}{h}$ MPC",
        r"$b_g$ about $ 1 $",
        r"$r_g$ about $ 1 $",
        r"$\sigma_{u}$ about $\frac{13.0}{h}$ MPC",
        r"$A_s$ about $%.3f \times 10^{-9}$" % (As_central*1e9),
        r'$N_{eff}$ about 3.046',
         '', '', r'$ n_s $ about %.4f' % (ns)
                              ]

        
        power_spectrum_chosen = [
        "galaxy galaxy",
        "galaxy velocity div.",
        "velocity div velocity div." ]


        plt.suptitle("Derivative of the %s power spectrum (%s) w.r.t. variation of %s at z = %.3f " % (power_spectrum_chosen[mapping_P[P]], neutrino_hier, 
        parameters2bevaried[mapping_var[o]], z) )
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        plt.show()

        # ----------------------------------------------------------------------------------------

    elif plot2 == True:

        h = H0_central/100.0 # need to initialise little h
        ks_obs_arr /= (H0_central/100.0) # put ks into units of k/h MPc
        results_findiff_derivatives *= (H0_central/100.0)**3 # putting into units of (MPC/h)^3/x
        results_findiff_semi_analytic_derivatives *= (H0_central/100.0)**3 # putting into units of (MPC/h)^3/x, x being the units of the parameter
        ks_fiducial /= (H0_central/100.0) # put ks into units of k/h MPc
        # that is being varied 
        # that is being varied 
    
        # getting mus we would like to plot 
        mu_indices = np.array(([np.argmin(np.abs(mus - 0.3)), np.argmin(np.abs(mus - 0.7))]))  

        # set up figure for plotting derivatives
        #fig = plt.Figure()
        
        #ax = plt.subplot(211)
        #ax1 = plt.subplot(212)

        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

        ax = plt.subplot(gs[1])
        ax1 = plt.subplot(gs[0])

        count = 0
        count1 = 0
        #count2 = 0
        #count3 = 0
        colors = cm.rainbow(np.linspace(0, 1, 12))
        colors1 = colors[6:]
        colors = colors[0:6]


        # getting power spectra to find relative difference from 
        dP_dtheta0_mu1 = results_findiff_derivatives[0, mu_indices[0], :]
        dP_dtheta0_mu2 = results_findiff_derivatives[0, mu_indices[1], :]


        # dP_dthetasemimu1 = results_findiff_semi_analytic_derivatives[0, mu_indices[0], :]
        # dP_dthetasemimu2 = results_findiff_semi_analytic_derivatives[0, mu_indices[1], :] 
        
        # print(dP_dtheta0_mu1/dP_dthetasemimu1)
        # print(dP_dtheta0_mu2/dP_dthetasemimu2)


        # loop through derivative results using 1st method (just finite difference and plot)
        for j in np.arange(int((num_steps-1)/2)):

            if o == cosmo_variable.H0:
                label = (r"$\Delta H_0 = %.2f $," %  (deltas[j]) + " M1"  ) 
            if o == cosmo_variable.mnu:
                label = (r"$\Delta \sum m_{\nu} = %.6f $," %  (deltas[j]) + " M1" ) 
            if o == cosmo_variable.Obh2:
                label = (r"$\Delta \Omega_bh^2 = %.5f $," %  (deltas[j]) + " M1" ) 
            if o == cosmo_variable.Och2:
                label = (r"$\Delta \Omega_{cdm}h^2 = %.3f $," %  (deltas[j]) + " M1" ) 
            if o == cosmo_variable.sigma_g:
                label = (r"$\Delta \sigma_g = %.3f $," %  (deltas[j]) + " M1" ) 
            if o == cosmo_variable.b_g:
                label = (r"$\Delta b_g=%.3f $," %  (deltas[j]) + " M1" ) 
            if o == cosmo_variable.r_g:
                label = (r"$\Delta r_g=%.3f $," %  (deltas[j]) + " M1" ) 
            if o == cosmo_variable.sigma_u:
                label = (r"$\Delta \sigma_{u}=%.3f $," %  (deltas[j]) + " M1" )
            if o == cosmo_variable.As:
                label = (r"$\Delta A_s=%.4f \times 10^{-9} $," %  (deltas[j]*1e9) + " M1" ) 
            if o == cosmo_variable.Neff:
                label = (r"$\Delta N_{eff}=3.046+%.4f $" %  (deltas[j]) ) 
            if o == cosmo_variable.n_s:
                label = (r'$ \Delta n_s=%.4f $' % (deltas[j]) )


            dP_dtheta = results_findiff_derivatives[j, mu_indices[0], :]
            
            ax.semilogx(ks_fiducial, abs(dP_dtheta-dP_dtheta0_mu1)/dP_dtheta0_mu1, label = label+r", $ \mu = %.2f $" % mus[mu_indices[0]], color = colors[count])
            #ax.semilogx(ks_obs_arr[3,mu_indices[0],:]/(h), abs(dP_dtheta-dP_dtheta0_mu1)/dP_dtheta0_mu1, label = label, color = colors[count])
            ax1.semilogx(ks_fiducial, dP_dtheta, label = label+r", $ \mu = %.2f $" % mus[mu_indices[0]], color = colors[count] )
            #ax1.semilogx(ks_obs_arr[3,mu_indices[0],:]/(h), dP_dtheta, label = label, color = colors[count] )

            count += 1


            dP_dtheta = results_findiff_derivatives[j, mu_indices[1], :]

            ax.semilogx(ks_fiducial, abs(dP_dtheta-dP_dtheta0_mu2)/dP_dtheta0_mu2, linestyle = '-.', label = label+r", $ \mu = %.2f $" % mus[mu_indices[1]], color = colors[count1] )
            #ax.semilogx(ks_obs_arr[3,mu_indices[1],:]/(h), abs(dP_dtheta-dP_dtheta0_mu2)/dP_dtheta0_mu2, linestyle = '-.', label = label, color = colors1[count1] )
            ax1.semilogx(ks_fiducial, dP_dtheta, linestyle = '-.', label = label+r", $ \mu = %.2f $" % mus[mu_indices[1]], color = colors[count1] )
            #ax1.semilogx(ks_obs_arr[3,mu_indices[1],:]/(h), dP_dtheta, linestyle = '-.', label = label, color = colors1[count1] )
            count1 += 1



        # loop through derivative results using 2nd method (finite diff. + analytic derivatives)
        for j in np.arange(int((num_steps-1)/2)):

            if o == cosmo_variable.H0:
                label = (r"$\Delta H_0 = %.2f $," %  (deltas[j]) + " M2"  ) 
            if o == cosmo_variable.mnu:
                label = (r"$\Delta \sum m_{\nu}  = %.6f $," %  (deltas[j]) + " M2" ) 
            if o == cosmo_variable.Obh2:
                label = (r"$\Delta \Omega_bh^2 = %.5f $," %  ( deltas[j]) + " M2") 
            if o == cosmo_variable.Och2:
                label = (r"$\Delta \Omega_{cdm}h^2 = %.3f $," %  (deltas[j]) + " M2" ) 
            if o == cosmo_variable.sigma_g:
                label = (r"$\Delta \sigma_g = %.3f $," %  (deltas[j]) + " M2" ) 
            if o == cosmo_variable.b_g:
                label = (r"$\Delta b_g=%.3f $," %  (deltas[j]) + " M2"  ) 
            if o == cosmo_variable.r_g:
                label = (r"$\Delta r_g=%.3f $," %  (deltas[j]) + " M2" ) 
            if o == cosmo_variable.sigma_u:
                label = (r"$\Delta \sigma_{u}=%.3f $," %  (deltas[j]) + " M2" ) 
            if o == cosmo_variable.As:
                label = (r"$\Delta A_s=%.4f \times 10^{-9} $," %  (deltas[j]*1e9) + " M2" ) 
            if o == cosmo_variable.Neff:
                label = (r"$\Delta N_{eff}=3.046+%.4f $" %  (deltas[j]) ) 
            if o == cosmo_variable.n_s:
                label = (r'$ \Delta n_s=%.4f $' % (deltas[j]) )

            dP_dtheta = results_findiff_semi_analytic_derivatives[j, mu_indices[0], :]
                
            ax.semilogx(ks_fiducial, abs(dP_dtheta-dP_dtheta0_mu1)/dP_dtheta0_mu1, label = label+r", $ \mu = %.2f $" % mus[mu_indices[0]], color = colors[count])
            #ax.semilogx(ks_obs_arr[3,mu_indices[0],:]/h, abs(dP_dtheta-dP_dtheta0_mu1)/dP_dtheta0_mu1, label = label, color = colors[count])
            ax1.semilogx(ks_fiducial, dP_dtheta, label = label+r", $ \mu = %.2f $" % mus[mu_indices[0]], color = colors[count] )
            #ax1.semilogx(ks_obs_arr[3,mu_indices[0],:]/h, dP_dtheta, label = label, color = colors[count] )
            count += 1


            dP_dtheta = results_findiff_semi_analytic_derivatives[j, mu_indices[1], :]

            ax.semilogx(ks_fiducial, abs(dP_dtheta-dP_dtheta0_mu2)/dP_dtheta0_mu2, linestyle = '-.', label = label+r", $ \mu = %.2f $" % mus[mu_indices[1]], color = colors[count1])
            #ax.semilogx(ks_obs_arr[3,mu_indices[1],:]/h, abs(dP_dtheta-dP_dtheta0_mu2)/dP_dtheta0_mu2, linestyle = '-.', label = label, color = colors1[count1])
            ax1.semilogx(ks_fiducial, dP_dtheta, linestyle = '-.', label = label+r", $ \mu = %.2f $" % mus[mu_indices[1]], color = colors[count1] )
            #ax1.semilogx(ks_obs_arr[3,mu_indices[1],:]/h, dP_dtheta, linestyle = '-.', label = label, color = colors1[count1] )
            count1 += 1



        # ylabels = [
        # r"$ \frac{\Delta \frac{dP(k, \mu)}{dH_0}}{\frac{dP(k, \mu)}{dH_0}}$",
        # r"$ \frac{\Delta \frac{dP(k, \mu)}{d\sum m_{\nu} }}{\frac{dP(k, \mu)}{d\sum m_{\nu} }}$",
        # r"$ \frac{\Delta \frac{dP(k, \mu)}{d\Omega_bh^2}}{\frac{dP(k, \mu)}{d\Omega_bh^2}}$",
        # r"$ \frac{\Delta \frac{dP(k, \mu)}{d\Omega_{cdm}h^2}}{\frac{dP(k, \mu)}{d\Omega_{cdm}h^2}}$",
        # r"$ \frac{\Delta \frac{dP(k, \mu)}{d\sigma_g}}{\frac{dP(k, \mu)}{d\sigma_g}}$",
        # r"$ \frac{\Delta \frac{dP(k, \mu)}{db_g}}{\frac{dP(k, \mu)}{db_g}}$",
        # r"$ \frac{\Delta \frac{dP(k, \mu)}{dr_g}}{\frac{dP(k, \mu)}{dr_g}}$",
        # r"$ \frac{\Delta \frac{dP(k, \mu)}{d\sigma_{u}}}{\frac{dP(k, \mu)}{d\sigma_{u}}}$",
        # r"$ \frac{\Delta \frac{dP(k, \mu)}{dA_s}}{\frac{dP(k, \mu)}{dA_s}}$",
        # r"$ \frac{\Delta \frac{dP(k, \mu)}{dN_{eff}}}{\frac{dP(k, \mu)}{dN_{eff}}} $"
        #           ]


        ylabels = [
        r"% diff. $ \frac{dP(k, \mu)}{dH_0}$ ", 
        r"% diff. $ \frac{dP(k, \mu)}{d\sum m_{\nu} }$", 
        r"% diff. $ \frac{dP(k, \mu)}{d\Omega_bh^2}$" , 
        r"% diff. $ \frac{dP(k, \mu)}{d\Omega_{cdm}h^2}$", 
        r"% diff. $ \frac{dP(k, \mu)}{d\sigma_g}$", 
        r"% diff. $ \frac{dP(k, \mu)}{db_g}$",
        r"% diff. $ \frac{dP(k, \mu)}{dr_g}$",
        r"% diff. $ \frac{dP(k, \mu)}{d\sigma_{u}}$" , 
        r"% diff. $ \frac{dP(k, \mu)}{dA_s}$",
        r"% diff. $ \frac{dP(k, \mu)}{dN_{eff}}$", '', '',
        r"% diff. $ \frac{dP(k, \mu)}{dn_s}$"
        ]


        ylabelsax1 = [
        r"$ \frac{dP(k, \mu)}{dH_0}$ ", r"$ \frac{dP(k, \mu)}{d\sum m_{\nu} }$", 
        r"$ \frac{dP(k, \mu)}{d\Omega_bh^2}$" , r"$ \frac{dP(k, \mu)}{d\Omega_{cdm}h^2}$", 
        r"$ \frac{dP(k, \mu)}{d\sigma_g}$", r"$ \frac{dP(k, \mu)}{db_g}$",
        r"$ \frac{dP(k, \mu)}{dr_g}$",r"$ \frac{dP(k, \mu)}{d\sigma_{u}}$" , r"$ \frac{dP(k, \mu)}{dA_s}$",
        r"$ \frac{dP(k, \mu)}{dN_{eff}}$", '', '',
        r"$ \frac{dP(k, \mu)}{dn_s}$"
                  ]


        xlabel = r"$k (\mathrm{h Mpc})^{-1}$"
        ylabel = ylabels[mapping_var[o]]
        ylabelax1 = ylabelsax1[mapping_var[o]]


        ax.set_xlim([np.min(ks), np.max(ks)])
        ax.set_xlabel(xlabel, fontsize = 17)
        ax.set_ylabel(ylabel, fontsize = 17)
        ax.tick_params(axis='both', which='major', labelsize=17)
        ax.tick_params(axis='both', which='minor', labelsize=17)
        
        ax1.legend(loc = 'upper right', fontsize = 12, ncol = 1)#, bbox_to_anchor = (-1.5, 0.5))
        #ax1.axhline(0.0, linestyle = '--', color = 'k')

        ax1.set_xlim([np.min(ks), np.max(ks)])
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        ax1.set_ylabel(ylabelax1, fontsize = 17)
        ax1.tick_params(axis='both', which='major', labelsize=17)
        ax1.tick_params(axis='both', which='minor', labelsize=17)
        ax.set_ylim([-0.05, 0.05])

        ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

        plt.subplots_adjust(hspace = 0.1, wspace = 0.0)
        

        parameters2bevaried = [
        r"$H_0$ about %.3f km/s/MPC" % H0_central,
        r"$\sum m_{\nu} $ about $ %.4f $ eV" % mnu_central,
        r"$\Omega_{b}h^2$ about %.5f" % Obh_central,
        r"$\Omega_{cdm}h^2$ about %.5f" % Och_central,
        r"$\sigma_g$ about $\frac{4.24}{h}$ MPC",
        r"$b_g$ about $ 1 $",
        r"$r_g$ about $ 1 $",
        r"$\sigma_{u}$ about $\frac{13.0}{h}$ MPC",
        r"$A_s$ about $%.3f \times 10^{-9}$" % (As_central*1e9),
        r"$N_{eff}$ about 3.046", '', '', r'$ n_s $ about %.4f ' % (ns)
                              ]

        
        power_spectrum_chosen = [
        "galaxy galaxy",
        "galaxy velocity div.",
        "velocity div velocity div." ]

        plt.suptitle("Derivative of the %s power spectrum (%s) w.r.t. variation of %s at z = %.3f " % (power_spectrum_chosen[mapping_P[P]], neutrino_hier, 
        parameters2bevaried[mapping_var[o]], z) )
        #plt.tight_layout()
        plt.show()



        # ----------------------------------------------------------------------------------------

    else: # write the derivatives results to a file

        spectra = ['GG', 'GU', 'UU']
        parameters = ['H0', 'mnu', 'ombh2', 'omch2', 'sigg', 'b', 'r', 'sigmau', 'As', 'Neff', 'ns' ]

        if o not in [cosmo_variable.As, cosmo_variable.sigma_g, cosmo_variable.sigma_u]: 

        # numerical derivatives 

            for i in range(int((num_steps-1)/2)):
                data = pd.DataFrame(results_findiff_derivatives[i,:,:])
                data.to_csv(r'Derivatives_RSPSpectra_M1_numerical_%s_%s_%s_Delta=%.6f_z=%.3f.csv' % (spectra[mapping_P[P]], 
                parameters[mapping_var[o]], lin_or_nonlin, deltas[i], z), 
                float_format='%.3f', na_rep="NAN!")

        # analytical / semi-analytical/numerical derivatives

            if o != cosmo_variable.Neff:
                for i in range(int((num_steps-1)/2)):
                    data = pd.DataFrame(results_findiff_semi_analytic_derivatives[i,:,:])
                    data.to_csv(r'Derivatives_RSPSpectra_M2_semianalytical_%s_%s_%s_Delta=%.6f_z=%.3f.csv' % (spectra[mapping_P[P]], 
                    parameters[mapping_var[o]], lin_or_nonlin, deltas[i], z), 
                    float_format='%.3f', na_rep="NAN!")

        # power spectrum with varying parameter

            for i in range(num_steps):
                data = pd.DataFrame(results_redshiftspacepowerspectrum_arr[i,:,:])
                data.to_csv(r'PowerSpectra_%s_%s_%s_value=%.6f_z=%.3f.csv' % (spectra[mapping_P[P]], 
                parameters[mapping_var[o]], lin_or_nonlin, param_variation[i], z), 
                float_format='%.3f', na_rep="NAN!")

        elif o == cosmo_variable.As:

        # numerical derivatives

            for i in range(int((num_steps-1)/2)):
                data = pd.DataFrame(results_findiff_derivatives[i,:,:])
                data.to_csv(r'Derivatives_RSPSpectra_M1_numerical_%s_%s_%s_Delta=%.6f_z=%.3f.csv' % (spectra[mapping_P[P]], 
                parameters[mapping_var[o]], lin_or_nonlin, deltas[i]*1e9, z), 
                float_format='%.3f', na_rep="NAN!")

        # analytical / semi-analytical/numerical derivatives


            for i in range(int((num_steps-1)/2)):
                data = pd.DataFrame(results_findiff_semi_analytic_derivatives[i,:,:])
                data.to_csv(r'Derivatives_RSPSpectra_M2_semianalytical_%s_%s_%s_Delta=%.6f_z=%.3f.csv' % (spectra[mapping_P[P]], 
                parameters[mapping_var[o]], lin_or_nonlin, deltas[i]*1e9, z), 
                float_format='%.3f', na_rep="NAN!")

        # power spectrum with varying parameter

            for i in range(num_steps):
                data = pd.DataFrame(results_redshiftspacepowerspectrum_arr[i,:,:])
                data.to_csv(r'PowerSpectra_%s_%s_%s_value=%.6f_z=%.3f.csv' % (spectra[mapping_P[P]], 
                parameters[mapping_var[o]], lin_or_nonlin, param_variation[i]*1e9, z), 
                float_format='%.3f', na_rep="NAN!")

        else:

        # numerical derivatives 

            for i in range(int((num_steps-1)/2)):
                data = pd.DataFrame(results_findiff_derivatives[i,:,:])
                data.to_csv(r'Derivatives_RSPSpectra_M1_numerical_%s_%s_%s_Delta=%.6f_z=%.3f.csv' % (spectra[mapping_P[P]], 
                parameters[mapping_var[o]], lin_or_nonlin, deltas[i](H0_central/100), z), 
                float_format='%.3f', na_rep="NAN!")

        # analytical / semi-analytical/numerical derivatives


            for i in range(int((num_steps-1)/2)):
                data = pd.DataFrame(results_findiff_semi_analytic_derivatives[i,:,:])
                data.to_csv(r'Derivatives_RSPSpectra_M2_semianalytical_%s_%s_%s_Delta=%.6f_z=%.3f.csv' % (spectra[mapping_P[P]], 
                parameters[mapping_var[o]], lin_or_nonlin, deltas[i](H0_central/100), z), 
                float_format='%.3f', na_rep="NAN!")

        # power spectrum with varying parameter

            for i in range(num_steps):
                data = pd.DataFrame(results_redshiftspacepowerspectrum_arr[i,:,:])
                data.to_csv(r'PowerSpectra_%s_%s_%s_value=%.6f_z=%.3f.csv' % (spectra[mapping_P[P]], 
                parameters[mapping_var[o]], lin_or_nonlin, param_variation[i]/(H0_central/100), z), 
                float_format='%.3f', na_rep="NAN!")

    # -----------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------

    return 0



# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------

# compute the growth rate f = dln(sqrt(P))/dln(a) for an arbitrary number of redshifts, ks and mus 
def compute_f_at_many_redshifts(omega_b_t: float, omega_cdm_t: float, H0_t: float, As_t: float, m_nu_t: float, 
neutrino_hierarchy: str, zed_list: list, kmin: float, kmax: float, mu_s: npt.NDArray, knum: int, d_a: float, 
sum_masses_central_t: float, omega_b_fid: float, omega_cdm_fid: float, H0_fid: float, As_fid: float, m_nu_fid: float,  
kspaceoption: str, tau_t: float, ns_t: float, N_eff_deviation: float, delta_mnu_max: float = 0.0):

    '''
    Quickly computes f(k) at any number of redshifts, ks and mus.
    zed_list must be a list/array.
    mu_s_fid must be a list/array.
    Returns f(k) array (3d), ks real (3D array) mus real (2D array) 
    '''
    
    # ---------------------------------------------------------------------------------------

    # set up scalefactors to get power spectrum at for each value of z we need (for each zed in zed_list!)
    all_redshifts_list = [] # this list will contain arrays of redshifts for each z in zed_list
    for zed in np.arange(len(zed_list)):
        scalefactors = 0 # initializing the scalefactors we need 
        if zed_list[zed] == 0:
            scalefactors = np.array([1.0, 1.0-d_a])
        else:
            scalefactors = np.linspace(1.0/(1+zed_list[zed]) + d_a, 1.0/(1+zed_list[zed]) - d_a, 3)
        all_redshifts_list.append(1.0/scalefactors-1.0)
    
    # ---------------------------------------------------------------------------------------
    
    # Set the CLASS parameters
    M = Class()
    # get the neutrinom masses
    m1, m2, m3 = get_masses(m_nu_t, neutrino_hierarchy, delta_mnu_max, sum_masses_central_t)
    
    mass_input_string = str(m1)+','+str(m2)+','+str(m3) 
    if (m1 < 0 or m2 < 0 or m3 < 0): # we have a problem
        print(m1 + m2 + m3)
        print('Neutrino masses are set to: ', str(m1) + '+ ' + '(0,'+str(m2-m1)+','+str(m3-m1) + ')' )
        msg = 'Neutrino masses are unphysical, sum of masses is most likely too small for set mass eigenstate differences from neutrino oscillations.'
        logger.error(msg)
        raise (ValueError)

    # set up cosmology
    M.set({"omega_b": omega_b_t, "omega_cdm": omega_cdm_t, "H0": H0_t, "A_s": As_t, "N_ur": 0.106+N_eff_deviation, 
    "N_ncdm": 3.0, "T_ncdm": "0.71, 0.71, 0.71",
    "m_ncdm": mass_input_string, "tau_reio": tau_t, "n_s": ns_t})
    M.set({"output": "mPk", "P_k_max_1/Mpc": 3.0, "z_max_pk": 2.0})

    M.compute() 
    
    # ---------------------------------------------------------------------------------------

    # calculate the growth rate numerically for the list of redshifts zed_list we want 
    growth_rates_arr = np.zeros((len(mu_s), knum, len(zed_list)))

    ks_obs_array = np.zeros((len(mu_s), knum, len(zed_list)))

    mu_obs_array = np.zeros((len(mu_s), len(zed_list)))

    # ---------------------------------------------------------------------------------------

    # setting up the values of k that we need 
    ks = [] 
    if kspaceoption == 'log':
        ks = np.logspace(np.log10(kmin), np.log10(kmax), knum, base=10) # 1/ MPC units
    else:
        ks = np.linspace(kmin, kmax, knum) # 1/MPC units 

    # ---------------------------------------------------------------------------------------



    if (omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t) == (omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid):

        # now get the power spectrum at all the redshifts needed to compute the growth rate
        for zed in np.arange(len(zed_list)): # looping through main outer redshift array we want 
            
            Pk_list_at_zed = np.zeros((knum, len(all_redshifts_list[zed])))
            for zz in np.arange(len(all_redshifts_list[zed])): # looping through inner zed list

                Pk_list_at_zed[:, zz] = np.array([M.pk_cb_lin(ki, all_redshifts_list[zed][zz]) for ki in ks]) # (MPC ^3)

            growthrate = 0
            scalefactors = 1.0/(1.0 + all_redshifts_list[zed]) # convert zz redshifts (inner redshifts for this iteration of the loop!)
            # back to scalefactors we want 
            da = abs(scalefactors[1] - scalefactors[0]) # calculating the step size in scalefactor da 
            if zed_list[zed] == 0: # use backwards difference method about z = 0 (finite diff.)
                growthrate = 0.5*(scalefactors[0]/(Pk_list_at_zed[:,0]))*(Pk_list_at_zed[:,0] - Pk_list_at_zed[:,1])/(da)
            else: # use the central difference method (finite diff.)
                growthrate = 0.5*(scalefactors[1]/(Pk_list_at_zed[:,1]))*(Pk_list_at_zed[:,0] - Pk_list_at_zed[:,2])/(2*da)

            growth_rates_arr[:,:,zed] = growthrate

        return growth_rates_arr, ks, mu_s

    else:
        # compute ks real and mus real for each z in zed_list

        for zz in np.arange(len(zed_list)):

            mu_obs_array[:, zz] = get_mus_realobs(mu_s, As_t, omega_b_t, omega_cdm_t, H0_t, m_nu_t,
            As_fid, omega_b_fid, omega_cdm_fid, H0_fid, m_nu_fid, zed_list[zz])

            for muu in np.arange(len(mu_s)):

                ks_obs_array[muu, :, zz] = get_ks_realobs(ks, As_t, omega_b_t, omega_cdm_t, H0_t,
                m_nu_t, As_fid, omega_b_fid, omega_cdm_fid, H0_fid, m_nu_fid, zed_list[zz], mu_s[muu])
    
    
        # now get the power spectrum at all the redshifts needed to compute the growth rate
        for zed in np.arange(len(zed_list)): # looping through main outer redshift array we want 

            Pk_list_at_zed = np.zeros((len(mu_s), knum, len(all_redshifts_list[zed])))
            for zz in np.arange(len(all_redshifts_list[zed])): # looping through inner zed list

                for muu in np.arange(len(mu_s)): # looping through mu

                    # compute ks real!
                    ks_real_at_zz_mu = get_ks_realobs(ks, As_t, omega_b_t, omega_cdm_t, H0_t, m_nu_t, As_fid, omega_b_fid, omega_cdm_fid,
                    H0_fid, m_nu_fid, all_redshifts_list[zed][zz], mu_s[muu])
                    

                    Pk_list_at_zed[muu, :, zz] = np.array([M.pk_cb_lin(ki, all_redshifts_list[zed][zz]) for ki in ks_real_at_zz_mu]) # (MPC ^3)

            growthrate = 0
            scalefactors = 1.0/(1.0 + all_redshifts_list[zed]) # convert zz redshifts (inner redshifts for this iteration of the loop!)
            # back to scalefactors we want 
            da = abs(scalefactors[1] - scalefactors[0]) # calculating the step size in scalefactor da 
            if zed_list[zed] == 0: # use backwards difference method about z = 0 (finite diff.)
                growthrate = 0.5*(scalefactors[0]/(Pk_list_at_zed[:,:,0]))*(Pk_list_at_zed[:,:,0] - Pk_list_at_zed[:,:,1])/(da)
            else: # use the central difference method (finite diff.)
                growthrate = 0.5*(scalefactors[1]/(Pk_list_at_zed[:,:,1]))*(Pk_list_at_zed[:,:,0] - Pk_list_at_zed[:,:,2])/(2*da)

            growth_rates_arr[:,:,zed] = growthrate
        
    
        return growth_rates_arr, ks_obs_array, mu_obs_array




# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------


# function to get the derivative of the RSP power spectrum, 
# at many ks, mus and zs, 
# for a single parameter,
# using a single simple 2nd order finite difference , quickly
def get_rsp_dP_dx_cosmo_many_redshifts(o: cosmo_variable, zed_list: list, central_params: list, delta_param: float, 
kmin: float, kmax: float, num_k_points: int, mu_s: npt.NDArray, d_a: float, neutrino_hier: str, lin_or_nonlin: str, kspaceoption: str, 
tau: float, ns: float, N_eff_deviation: float, dm2_atm: float = 2.44e-3, dm2_sol: float = 7.53e-5):

    '''
    Quickly computes dP_xx_dy for a single parameter y for all redshift space power spectra,
    for any number of z, k and mu and returns a 3D array (for each RSPS)
    with the derivatives. ks and mus are also returned.
    mu_s_fid and zed_list MUST be a list, even for a single z or mu value.
    This function may still take a while to run if the number of redshifts and mus is long.
    '''
    
    linear = True # use linear power spectrum
    if lin_or_nonlin != 'linear':  
        linear = False

    #print(kmax_fid, kmin_fid)
    ks = [] # initializing an object to store fiducial values of k
    if kspaceoption == 'log':
        ks = np.logspace(np.log10(kmin), np.log10(kmax), num_k_points, base=10) # k values (k in 1/MPC)
    else:
        ks = np.linspace(kmin, kmax, num_k_points)


    Obh_central, mnu_central, H0_central, Och_central, As_central, sigmagh, b_g, r_g, sigmauh = central_params

    Obh = Obh_central
    mnu = mnu_central
    H0 = H0_central
    As = As_central
    Och = Och_central
    h = H0/100.0
    sigmag = sigmagh/h
    sigmau = sigmauh/h
    delta_mnu_max = 0.0
    N_eff_deviation = 0.0
    ns_val = ns 


    param_variation = [] # initialising an object that will store the values of our parameter we vary
    
    if o == cosmo_variable.H0: # varying H0
        param_variation = np.linspace(H0_central-delta_param, H0_central+delta_param, 2)
    elif o == cosmo_variable.mnu: # varying sum of neutrino masses
        param_variation = np.linspace(mnu_central-delta_param, mnu_central+delta_param, 2)
    elif o == cosmo_variable.Obh2: # varying omega_b
        param_variation = np.linspace(Obh_central-delta_param, Obh_central+delta_param, 2)
    elif o == cosmo_variable.Och2: # varying omega_cdm
        param_variation = np.linspace(Och_central-delta_param, Och_central+delta_param, 2)
    elif o == cosmo_variable.As: # As
        param_variation = np.linspace(As_central-delta_param, As_central+delta_param, 2)
    elif o == cosmo_variable.Neff: # N_eff
        param_variation = np.linspace(-delta_param, delta_param, 2)
    elif o == cosmo_variable.n_s: # ns
        param_variation = np.linspace(ns-delta_param, ns+delta_param, 2)
    else:
        raise Exception('Parameter chosen is not an option. (get_rsp_dP_dx_cosmo_many_redshifts()) ')

    # set up arrays to store power spectra and derivatives in 

    matter_power_spectra_arr = np.zeros((2, len(mu_s), num_k_points, len(zed_list)))

    growth_rate_arr = np.zeros((2, len(mu_s), num_k_points, len(zed_list)))

    ks_obs_arr =  np.zeros((2, len(mu_s), num_k_points, len(zed_list)))

    mus_obs_arr = np.zeros((2, len(mu_s), len(zed_list)))


    redshift_space_power_spectra_gg =  np.zeros((2, len(mu_s), num_k_points, len(zed_list)))

    redshift_space_power_spectra_gu =  np.zeros((2, len(mu_s), num_k_points, len(zed_list)))

    redshift_space_power_spectra_uu =  np.zeros((2, len(mu_s), num_k_points, len(zed_list)))


    power_spectra_derivatives_gg = np.zeros((len(mu_s), num_k_points, len(zed_list)))

    power_spectra_derivatives_gu = np.zeros((len(mu_s), num_k_points, len(zed_list)))

    power_spectra_derivatives_uu = np.zeros((len(mu_s), num_k_points, len(zed_list)))


    # calculate power spectrum while varying parameter of choice
    for i in np.arange(2): 
        # first just get delta-delta/ delta-theta /theta-theta power spectrum

        # changing parameter that is meant to be varied
        if o == cosmo_variable.H0:
            H0 = param_variation[i]
            h = param_variation[i]/100.0
            sigmag = sigmagh/h
            sigmau = sigmauh/h
        elif o == cosmo_variable.mnu:
            mnu = param_variation[i]
            delta_mnu_max = delta_param
        elif o == cosmo_variable.Obh2:
            Obh = param_variation[i]
        elif o == cosmo_variable.Och2:
            Och = param_variation[i]
        elif o == cosmo_variable.As:
            As = param_variation[i]
        elif o == cosmo_variable.Neff:
            N_eff_deviation = param_variation[i]
        elif o == cosmo_variable.n_s:
            ns_val = param_variation[i]
        # get omega_m
        omegam = get_Om_0(Obh, Och, mnu, H0)


        Pk_cb, ks_obs, mus_obs = run_class(Obh, Och, H0, As, mnu, neutrino_hier, zed_list, kmin, 
        kmax, num_k_points, delta_mnu_max, mnu_central, kspaceoption, tau, ns_val, 
        Obh_central, Och_central, H0_central, As_central, mnu_central, mu_s, lin_or_nonlin, N_eff_deviation)
            

        f = compute_f_at_many_redshifts(Obh, Och, H0, As, mnu, neutrino_hier, zed_list, kmin, kmax, mu_s, 
        num_k_points, d_a, mnu_central, Obh_central, Och_central, H0_central, As_central, mnu_central, 
        kspaceoption, tau, ns_val, N_eff_deviation, delta_mnu_max=delta_mnu_max)[0]
            

        if Pk_cb.shape == (len(mu_s), num_k_points, len(zed_list)) and ks_obs.shape == (len(mu_s), num_k_points, len(zed_list)) and mus_obs.shape == (len(mu_s), len(zed_list)) and f.shape == (len(mu_s), num_k_points, len(zed_list)):

            matter_power_spectra_arr[i,:,:,:] = Pk_cb
            ks_obs_arr[i,:,:,:] = ks_obs
            mus_obs_arr[i,:,:] = mus_obs
            growth_rate_arr[i,:,:,:] = f


        elif Pk_cb.shape == (num_k_points, len(zed_list)) and ks_obs.shape == (num_k_points,) and mus_obs.shape == (len(mu_s),) and f.shape == (len(mu_s), num_k_points, len(zed_list)):

            for muu in np.arange(len(mu_s)):
                matter_power_spectra_arr[i,muu,:,:] = Pk_cb
                for zz in np.arange(len(zed_list)):
                    ks_obs_arr[i,muu,:,zz] = ks
                        
            for zz in np.arange(len(zed_list)):
                mus_obs_arr[i,:,zz] = mu_s

            growth_rate_arr[i,:,:,:] = f


        else:
            print('shape Pk_cb: ', Pk_cb.shape)
            print('shape ks: ', ks_obs.shape)
            print('shape mus:', mus_obs.shape)
            print('shape f: ', f.shape)
            msg = 'Shape of Pk_cb / k / mu /growth rate array returned from run_class() is unexpected and cannot be adapted ' + 'to store in matter_power_spectrum_arr / ks_obs_arr / mus_obs_arr / growth_rates_arr (derivative_power_spectra()).'
            logger.error(msg)
            raise (ValueError)
        

        for zz in np.arange(len(zed_list)): # looping through redshift array

            q_para = distortion_parallel(As, Obh, Och, H0, mnu, As_central, Obh_central, Och_central, H0_central, mnu_central, zed_list[zz])
            q_perp = distortion_perpendicular(As, Obh, Och, H0, mnu, As_central, Obh_central, Och_central, H0_central, mnu_central, zed_list[zz])

            for muu in np.arange(len(mu_s)): # looping through mu
                    
                redshift_space_power_spectra_gg[i, muu, :, zz] = gg_redshift_s(b_g[zz], r_g, growth_rate_arr[i, muu, :, zz],
                sigmag, mus_obs_arr[i, muu, zz], ks_obs_arr[i, muu, :, zz], q_para, 
                q_perp)*matter_power_spectra_arr[i,muu,:,zz]
                
                redshift_space_power_spectra_gu[i, muu, :, zz] = gu_redshift_s(b_g[zz], r_g, growth_rate_arr[i, muu, :, zz],
                sigmag, sigmau, mus_obs_arr[i, muu, zz], ks_obs_arr[i, muu, :, zz], zed_list[zz], H0, omegam, 
                1.0-omegam, q_para, q_perp)*matter_power_spectra_arr[i,muu,:,zz]

                redshift_space_power_spectra_uu[i, muu, :, zz] = uu_redshift_s(sigmau, mus_obs_arr[i, muu,zz], 
                growth_rate_arr[i, muu, :, zz], ks_obs_arr[i, muu, :, zz], zed_list[zz], H0, omegam, 
                1.0-omegam, q_para, q_perp)*matter_power_spectra_arr[i,muu,:,zz]

        
    # --------------------------------------------------------------------------------------- 

    # get derivatives with just simple central difference method for redshift 
    # space power spectrum 

    power_spectra_derivatives_gg = (redshift_space_power_spectra_gg[1, :, :, :] 
    - redshift_space_power_spectra_gg[0, :, :, :])/(2.0*delta_param)

    power_spectra_derivatives_gu = (redshift_space_power_spectra_gu[1, :, :, :] 
    - redshift_space_power_spectra_gu[0, :, :, :])/(2.0*delta_param)

    power_spectra_derivatives_uu = (redshift_space_power_spectra_uu[1, :, :, :] 
    - redshift_space_power_spectra_uu[0, :, :, :])/(2.0*delta_param)


    return power_spectra_derivatives_gg, power_spectra_derivatives_gu, power_spectra_derivatives_uu, ks_obs_arr, mus_obs_arr      
        
    



#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    #baryons, neutrinos, H0, CDM, As, sigmag*h, rg, bg, sigmau*h, f0
    Obh = 0.022383
    Och = 0.12011
    H0 = 67.32
    h = H0/100.0
    As = 2.10058e-9
    sigmagh = 4.24
    sigmauh = 13.0
    mnu = 0.058
    r = 1.0
    b = 1.0
    f0 = 1.0
    params = [Obh, mnu, H0, Och, As, sigmagh, b, r, sigmagh]
    zval = 0.0
    tau = 0.092
    ns = 0.996
    dm2_atm = 2.5e-3
    dm2_sol = 7.6e-5

    # pk_neff1 = run_class(Obh, Och, H0, As, mnu, 'normal', [0.0], 1e-3, 1.0, 
    # 100, dm2_atm, dm2_sol, 0.0, mnu, 'log', tau, ns, Obh, Och,
    # H0, As, mnu, [], 'linear',0.0)[0]

    # pk_neff2 = run_class(Obh, Och, H0, As, mnu, 'normal', [0.0], 1e-3, 1.0, 
    # 100, dm2_atm, dm2_sol, 0.0, mnu, 'log', tau, ns, Obh, Och,
    # H0, As, mnu, [], 'linear',0.0)[0]

    # ks = np.logspace(np.log10(1e-3), np.log10(1), 100, base=10)
    

    # plt.legend()
    # plt.show()
    # good step sizes
    # 0.1 (for H0)
    # 0.00015*5 Obh2 (5 sigma)
    # 0.0012 Och2 (5 sigma)
    # 0.029e-10 As
    # choosing 0.001*5 for mnu (seems to work well enough )
    # choosing 
    # 0.5 for rg, bg, sigmau*h, sigmag*h, f0

    # H0 = 1, neutrinos = 2, baryons = 3, CDM = 4, sigmag = 5, bg = 6, rg = 7, sigmau = 8, As = 9
   
    # H0 
    derivative_power_spectra(cosmo_variable.H0, RSPS_variable.P_gg, 1.0, 'plot2', params, 3.0, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # mnu
    #derivative_power_spectra(cosmo_variable.mnu, RSPS_variable.P_gg, 1.0, 'plot2', params, 0.001, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # # # Obh
    # derivative_power_spectra(cosmo_variable.Obh2, RSPS_variable.P_gg, 1., 'plot2', params, 0.0006, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # # # Och
    #derivative_power_spectra(cosmo_variable.Och2, RSPS_variable.P_gg, 1., 'plot2', params, 0.006, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)

    # for ztest in [0.1, 0.5, 1.0, 1.5, 2.0]:
        
        
    #     mu = 0.5 
    #     oldpara = dq_para_distortion_dx(cosmo_variable.H0, H0_t=H0, H0_fid=H0, Om_t=get_Om_0(Obh, Och, mnu, H0),
    #                             Om_fid=get_Om_0(Obh, Och, mnu, H0), z=ztest)
    #     oldperp = dq_perp_distortion_dx(cosmo_variable.H0, H0_t=H0, H0_fid=H0, Om_t=get_Om_0(Obh, Och, mnu, H0),
    #                             Om_fid=get_Om_0(Obh, Och, mnu, H0), z=ztest)
    #     oldF = dF_distortion_dx(cosmo_variable.H0, 1.0, H0, ztest, get_Om_0(Obh, Och, mnu, H0), H0, get_Om_0(Obh, Och, mnu, H0), 1.0, 1.0)
        
    #     oldk = dk_obs_dx(cosmo_variable.H0, oldF, oldperp, mu, 1.0, 1.0, 1.0)
    #     oldmu = dmu_obs_dx(oldF, mu, 1.0, cosmo_variable.H0)
        
    #     newk = (get_ks_realobs(1.0, As, Obh, Och, H0+0.1, mnu, As, Obh, Och, H0, mnu, ztest, mu) -
    #     get_ks_realobs(1.0, As, Obh, Och, H0-0.1, mnu, As, Obh, Och, H0, mnu, ztest, mu))/(2*0.1)
    #     newmu = (get_mus_realobs(mu, As, Obh, Och, H0+0.1, mnu, As, Obh, Och, H0, mnu, ztest) -
    #     get_mus_realobs(mu, As, Obh, Och, H0-0.1, mnu, As, Obh, Och, H0, mnu, ztest))/(2*0.1)

    #     print(ztest, "dk/dH0 old:", oldk, "new:", newk, "rel diff:", (newk-oldk)/max(abs(oldk),1e-12))

    #     print(ztest, "dmu/dH0 old:", oldmu, "new:", newmu, "rel diff:", (newmu-oldmu)/max(abs(oldmu),1e-12))
        
        # new = (dq_para_distortion_dx(cosmo_variable.H0, H0_t=H0+0.1, H0_fid=H0, Om_t=get_O  
        # new = (distortion_parallel(As, Obh, Och, H0+0.1, mnu, As, Obh, Och, H0, mnu, ztest) -
        # distortion_parallel(As, Obh, Och, H0-0.1, mnu, As, Obh, Och, H0, mnu, ztest))/(2*0.1) 
        # print(ztest, "dq_para/dH0 old:", old, "new:", new, "rel diff:", (new-old)/max(abs(old),1e-12))

    # other parameters
    # sigmag
    #derivative_power_spectra(cosmo_variable.sigma_g, RSPS_variable.P_gg, 0.2, 'plot2', params, 0.1, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # bg
    #derivative_power_spectra(cosmo_variable.b_g, RSPS_variable.P_gg, 0.2, 'plot2', params, 0.1, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # rg
    #derivative_power_spectra(7, 1, 0.0, 'plot1', params, 0.1, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # sigmau
    #derivative_power_spectra(cosmo_variable.sigma_u, RSPS_variable.P_ug, 0.2, 'plot2', params, 0.1, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # As
    #derivative_power_spectra(cosmo_variable.As, RSPS_variable.P_gg, 0.2, 'plot2', params, 0.1*1e-10, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # N_eff
    #derivative_power_spectra(cosmo_variable.Neff, RSPS_variable.P_gg, 0.2, 'plot2', params, 0.001, 1e-4, 0.2*(0.6732), 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)

    # print(pk.shape)
    # print(k.shape)

    # plt.loglog(k, pk[0,0,:])
    # plt.show()

    

    # test newly defined functions: compute_f_at_many_redshifts(), get_rsp_dP_dx_cosmo_many_redshifts()
    
    # n_h = 'normal'
    # # f = compute_f_at_many_redshifts(Obh, Och, H0, As, mnu, n_h, 
    # # [0.0, 0.001, 0.005], 1e-4, 1.0, np.linspace(0.0, 1.0, 3), 1000, 0.0001, 'linear', mnu,
    # # Obh, Och, H0, As, mnu, 'log', tau, ns)[0]

    # ks = np.logspace(np.log10(1e-4), np.log10(1.0), 1000, base=10.0)


    # # plt.semilogx(ks, f[0,:,0], label = 'z = 0')
    # # plt.semilogx(ks, f[0,:,1], label = 'z = 0.001')
    # # plt.semilogx(ks, f[0,:,2], label = 'z = 0.005')
    # # plt.legend()
    # # plt.show()
    # # #get_rsp_dP_dx_cosmo_many_redshifts()

    # dP_dObh2_1 = get_rsp_dP_dx_cosmo_many_redshifts(1, [0.0], params, 0.1, 1e-4, 1.0, 1000, mnu,
    # np.linspace(0.0, 1.0, 1), 0.0001, n_h, 'linear', 'log', tau, ns)[0]

    # # dP_dObh2_2 = get_rsp_dP_dx_cosmo_many_redshifts(1, [0.0], params, 0.5, 1e-4, 1.0, 1000, mnu,
    # # np.linspace(0.0, 1.0, 1), 0.0001, n_h, 'linear', 'log', tau, ns)[0]

    # # dP_dObh2_3 = get_rsp_dP_dx_cosmo_many_redshifts(1, [0.0], params, 1.0, 1e-4, 1.0, 1000, mnu,
    # # np.linspace(0.0, 1.0, 3), 0.0001, n_h, 'linear', 'log', tau, ns)[0]


    # plt.semilogx(ks, dP_dObh2_1[0,:,0], label = 'delta = 0.1')
    # # plt.semilogx(ks, dP_dObh2_2[0,:,0], label = 'delta = 0.5')
    # # plt.semilogx(ks, dP_dObh2_3[0,:,0], label = 'delta = 1.0')
    # # plt.legend()
    # plt.show()

