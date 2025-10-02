# Author: Abbe Whitford

# functions for main script, main_PV_forecasts.py

# Import modules/libraries --------------------------------------
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd                           
from classy import Class
from scipy.optimize import fsolve
from scipy.integrate import quad
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick





# function definitions ----------------------------------------------------------------------------



# get the mass of each neutrino eigenstate, given the hierarchy (excluding degenerate hierarchy as an option),
# sum of masses and neutrino oscillation mass differences
# this code is inspired by some code I found here: https://github.com/lesgourg/class_public/blob/master/notebooks/neutrinohierarchy.ipynb
def get_masses(sum_masses_true, hierarchy, del_mnu_max, sum_masses_central_true, delta_m_squared_atm = 2.5e-3, delta_m_squared_sol = 7.5e-5):
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
def angular_diameter_distance(Om, H0, z, c = 299792.458):
    '''
    Returns the angular diameter distance to redshift z assuming flat cosmology with H0, Om as given 
    by first two arguments of this function.
    Inputs:
        z = redshift
        Om = normalized matter density
        H0 = Hubble constant 
    '''
    integral = quad(get_Hubble_z_inv, 0.0, z, args = (Om, H0))[0]
    Ang_diam_dist = c*integral/(1+z)
    return Ang_diam_dist


# get Hubble function (H(z))
def get_Hubble_z(z, Om, H0):
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
def get_Hubble_z_inv(z, Om, H0):
    '''
    Returns 1/H(z).
    Inputs:
        z = redshift
        Om = normalized matter density
        H0 = Hubble constant 
    '''
    E_z = np.sqrt( Om*((1+z)**3) + (1-Om))
    H_z = H0*E_z
    return 1.0/H_z


# get normalized matter density from sum of neutrino mass, baryon and CDM density
def get_Om_0(Obh2, Och2, Mnu, H0):
    '''
    Returns Om at the present day given H0, mnu, Obh2, Och2 etc.
    '''
    h = H0/100.0
    Om = (Obh2 + Och2 + (Mnu)/(93.14))/(h**2)
    return Om


# function to compute q_perpendicular (parameter related to AP Effect, appropriate for a cosmological fit to data)
def distortion_parallel(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z):
    '''
    Returns q_parallel, given values of cosmological parameters for a REAL cosmology (ending with _t)
    and cosmological parameters for an assumed fiducial cosmology, and the redshift z.
    '''
    Om_t = get_Om_0(Obh2_t, Och2_t, Mnu_t, H0_t)
    Om_fid = get_Om_0(Obh2_fid, Och2_fid, Mnu_fid, H0_fid) 

    H_z_t = get_Hubble_z(z, Om_t, H0_t)
    H_z_fid = get_Hubble_z(z, Om_fid, H0_fid)

    q_para = (H_z_fid*H0_t)/(H0_fid*H_z_t)

    return q_para


# function to compute q_parallel (parameter related to AP Effect, appropriate for a cosmological fit to data)
def distortion_perpendicular(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z):
    '''
    Returns q_perpendicular, given values of cosmological parameters for a REAL cosmology (ending with _t)
    and cosmological parameters for an assumed fiducial cosmology, and the redshift z.
    '''
    Om_t = get_Om_0(Obh2_t, Och2_t, Mnu_t, H0_t)
    D_A_z_t = angular_diameter_distance(Om_t, H0_t, z)

    Om_fid = get_Om_0(Obh2_fid, Och2_fid, Mnu_fid, H0_fid)
    D_A_z_fid = angular_diameter_distance(Om_fid, H0_fid, z)
    
    q_perp = 0
    if z == 0: # this is true, you can check using L'Hopital 's rule in lim z -> zero
        q_perp = 1.0
    else:
        q_perp = (D_A_z_t*H0_t)/(D_A_z_fid*H0_fid)

    return q_perp


def distortion_ratio_F(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z):
    '''
    Returns F = q_parallel/q_perpendicular.
    '''
    q_parallel = distortion_parallel(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
    q_perpend = distortion_perpendicular(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
    F = q_parallel/q_perpend
    return F


# function to get mu_real (real values of mu) from mu_fiducial (values of mu observer believes they are measuring based on fiducial cosmology)
def get_mus_real(mus_fid, As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z):
    '''
    Returns mu (actual/real) computed from parameters for real and fidicual cosmologies and values for mu fiducial.  
    '''
    F = distortion_ratio_F(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
    mus_real = ((mus_fid)/(F))*(1.0/(np.sqrt(1.0 + (mus_fid**2)*( 1.0/(F**2) - 1.0 ))))
    
    return mus_real 


# function to get k_real (real k modes) from k_fiducial (values of mk modes observer believes they are measuring based on fiducial cosmology)
def get_ks_real(ks_fid, As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z, mus_fid):
    '''
    Returns k (actual/real) computed from parameters for real and fidicial cosmo and mu and k fiducial.
    '''
    F = distortion_ratio_F(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
    q_perp = distortion_perpendicular(As_t, Obh2_t, Och2_t, H0_t, Mnu_t, As_fid, Obh2_fid, Och2_fid, H0_fid, Mnu_fid, z)
    
    ks_real = ((ks_fid)/(q_perp))*np.sqrt(1.0 + (mus_fid**2)*( 1.0/(F**2) - 1.0 ))

    return ks_real




# function to just get the power spectrum of matter (linear) from CLASS----------------------------
def run_class(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, zed, kmin_fid, kmax_fid, 
knum, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid,
H0_fid, As_fid, m_nu_fid, mus_fid_arr, linornonlin, N_eff_deviation):
    ''' 
    Function to compute the matter power spectrum from Class() (which is equal to the velocity divergence and cross correlation of the velocity and 
    density power spectrum in linear theory).
    Returns Pk, k_real and mu_real (assuming mu_fiducial is supplied).
    zed always need to be in a vector (even for a single z).
    mu always needs to be in a vector (even for a single mu).
    Shape of Pk returned, k_real returned and mu_real returned will depend on whether AP effect is taken into account or not (whether fiducial
    and real cosmological model differ or not).
    '''
    # Set the CLASS parameters
    M = Class()
    # get the neutrino masses
    m1, m2, m3 = get_masses(m_nu_t, neutrino_hierarchy, del_mnu_max, sum_masses_central_t, dm2_atm, dm2_sol) # true neutrino masses

    mass_input_string = str(m1)+','+str(m2)+','+str(m3) 
    if (m1 < 0 or m2 < 0 or m3 < 0): # we have a problem
        print(m1 + m2 + m3)
        print('Neutrino masses are set to: ', str(m1) + '+ ' + '(0,'+str(m2-m1)+','+str(m3-m1) + ')' )
        raise Exception('Neutrino masses are unphysical, sum of masses is most likely too small for set mass eigenstate differences from neutrino oscillations.')
    
    
    if linornonlin == 'linear':
        # set up TRUE cosmology
        M.set({"omega_b": omega_b_t, "omega_cdm": omega_cdm_t, "H0": H0_t, "A_s": As_t, "N_ur": 0.00641+N_eff_deviation, 
        "N_ncdm": 3.0,  
        "m_ncdm": mass_input_string, "tau_reio": tau_t, "n_s": ns_t})
        M.set({"output": "mPk", "P_k_max_1/Mpc": 5.0, "z_max_pk": 2.0})
        
    elif linornonlin == 'nonlinear':
        # set up TRUE cosmology
        M.set({"omega_b": omega_b_t, "omega_cdm": omega_cdm_t, "H0": H0_t, "A_s": As_t, "N_ur": 0.00641+N_eff_deviation, 
        "N_ncdm": 3.0, 
        "m_ncdm": mass_input_string, "tau_reio": tau_t, "n_s": ns_t})
        M.set({"output": "mPk", "P_k_max_1/Mpc": 3.0, "z_max_pk": 2.0, 'non linear': 'Halofit'})
        
    else:
        raise Exception('linornonlin not set to appropriate string, options are \'linear\' or \'nonlinear\' (run_class()).')


    M.compute() 
    
    # now to compute the power spectrum at the true (k, mu) given (k_fid, mu_fid) -------------------------------------------

    # first make an array with all the fiducial values of k
    ks_fid = []
    if kspaceoption == 'log':
        ks_fid = np.logspace(np.log10(kmin_fid), np.log10(kmax_fid), knum, base=10) # 1/ MPC units
    else:
        ks_fid = np.linspace(kmin_fid, kmax_fid, knum) # 1/MPC units

    # lets first make an array of k_fid, mu_fid then k_true, mu_true corresponding to k_fid, mu_fid

    if ((omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid) == (omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t)): # no need to worry about AP effect

        Pk = np.zeros((knum, len(zed)))
        for reds in np.arange(len(zed)):
            Pk[:,reds] = np.array([M.pk_cb_lin(ki, zed[reds]) for ki in ks_fid]) # (MPC ^3)
        return Pk, ks_fid, mus_fid_arr

    else: # we need to worry about the AP effect 
        
        k_real_grid = []
        # getting the power spectrum for multiple redshifts

        k_real_grid = np.zeros((len(mus_fid_arr), len(ks_fid), len(zed)))   

        mus_real = np.zeros((len(mus_fid_arr), len(zed)))

        for zz in np.arange(len(zed)):
            for muu in np.arange(len(mus_fid_arr)):

                k_real_grid[muu,:, zz] = get_ks_real(ks_fid, As_t, omega_b_t, omega_cdm_t, H0_t, m_nu_t,
                As_fid, omega_b_fid, omega_cdm_fid, H0_fid, m_nu_fid, zed[zz], mus_fid_arr[muu])

            mus_real[:, zz] = get_mus_real(mus_fid_arr, As_t, omega_b_t, omega_cdm_t, H0_t, m_nu_t, 
            As_fid, omega_b_fid, omega_cdm_fid, H0_fid, m_nu_fid, zed[zz])

        # get the power spectra 
        Pk = np.zeros((len(mus_fid_arr), knum, len(zed)))

        for zz in np.arange(len(zed)):
            for muu in np.arange(len(mus_fid_arr)):
                Pk[muu, :, zz] = np.array([M.pk_cb_lin(ki, zed[zz]) for ki in k_real_grid[muu, :, zz]])

        return Pk, k_real_grid, mus_real

       
# function to get the growth rate at a single value of z (compute numerical approx. to dln(sqrt(P(k))))/dln(a) using central /backwards finite difference)
def growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, zed, kmin_fid, kmax_fid, knum, 
d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid,
H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation):

    ''' 
    This function allows one to compute f numerically using the finite difference method for a single value of z.
    mus_fid_arr needs to be an array.
    Shape of f(k) returned, k_real returned and mu_real returned will depend on whether AP effect is taken into account or not (whether fiducial
    and real cosmological model differ or not).
    '''
    scalefactors = np.array([1, 1-d_a]) # set up scalefactors to get power spectrum at 
    if zed == 0:
        scalefactors = np.array([1, 1-d_a])
    else:
        scalefactors = np.linspace(1/(1+zed) + d_a, 1/(1+zed) - d_a, 3)
    zs = 1/scalefactors - 1

    pk = 0 
    k_real = 0
    mu_real = mus_fid_arr

    pk, k_real, mu_real = run_class(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, zs, kmin_fid, kmax_fid, 
    knum, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid,
    H0_fid, As_fid, m_nu_fid, mus_fid_arr, 'linear', N_eff_deviation)

    growthrate = 0
    da = abs(scalefactors[1] - scalefactors[0])

    # now finally getting the growth rate 
    if zed == 0: # use backwards difference (finite diff.)

        if pk.shape == (knum, len(zs)):
            growthrate = 0.5*(scalefactors[0]/(pk[:,0]))*(pk[:,0] - pk[:,1])/(da)
        elif pk.shape == (len(mus_fid_arr), knum, len(zs)):
            growthrate = 0.5*(scalefactors[0]/(pk[:,:,0]))*(pk[:,:,0] - pk[:,:,1])/(da)
        else:
            print('shape pk: ', pk.shape)
            raise Exception('Shape of pk is unexpected. (growth_rate())')
    else: # use forwards difference (finite diff.)

        if pk.shape == (knum, len(zs)):
            growthrate = 0.5*(scalefactors[1]/(pk[:,1]))*(pk[:,0] - pk[:,2])/(2*da)
        elif pk.shape == (len(mus_fid_arr), knum, len(zs)):
            growthrate = 0.5*(scalefactors[1]/(pk[:,:,1]))*(pk[:,:,0] - pk[:,:,2])/(2*da)
        else:
            print('shape pk: ', pk.shape)
            raise Exception('Shape of pk is unexpected. (growth_rate())')

    return growthrate, k_real, mu_real


# get the derivative of the growth rate w.r.t. some parameter of choice using just a central finite difference
def derivative_growth_rate(param, delta, omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, 
    zed, kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
    kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation):
    
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

    if param == 1: 
        lowf = growth_rate(omega_b_t, omega_cdm_t, H0_t-delta, As_t, m_nu_t, neutrino_hierarchy, zed, 
        kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        highf = growth_rate(omega_b_t, omega_cdm_t, H0_t+delta, As_t, m_nu_t, neutrino_hierarchy, 
        zed, kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        df_dH = (highf - lowf)/(2*delta)
        return df_dH

    elif param == 2:
        lowf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t-delta, neutrino_hierarchy, zed, 
        kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        highf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t+delta, neutrino_hierarchy, 
        zed, kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        df_dmnu = (highf - lowf)/(2*delta)
        return df_dmnu

    elif param == 3:
        lowf = growth_rate(omega_b_t-delta, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, zed, 
        kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        highf = growth_rate(omega_b_t+delta, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, 
        zed, kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        df_dob = (highf - lowf)/(2*delta)
        return df_dob

    elif param == 4:
        lowf = growth_rate(omega_b_t, omega_cdm_t-delta, H0_t, As_t, m_nu_t, neutrino_hierarchy, zed, 
        kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        highf = growth_rate(omega_b_t, omega_cdm_t+delta, H0_t, As_t, m_nu_t, neutrino_hierarchy, 
        zed, kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        df_doc = (highf - lowf)/(2*delta)
        return df_doc

    elif param == 9:
        lowf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t+delta, m_nu_t, neutrino_hierarchy, zed, 
        kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        highf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t-delta, m_nu_t, neutrino_hierarchy, 
        zed, kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        df_da = (highf - lowf)/(2*delta)
        return df_da

    elif param == 10: # Neff
        lowf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, zed, 
        kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, -N_eff_deviation)[0]

        highf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, 
        zed, kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        df_da = (highf - lowf)/(2*delta)
        return df_da

    elif param == 11: # ns 
        
        lowf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, zed, 
        kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t-delta, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        highf = growth_rate(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, 
        zed, kmin_fid, kmax_fid, knum, d_a, linear, dm2_atm, dm2_sol, del_mnu_max, sum_masses_central_t, 
        kspaceoption, tau_t, ns_t+delta, omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid, mus_fid_arr, N_eff_deviation)[0]

        df_da = (highf - lowf)/(2*delta)
        return df_da

    else:
        raise Exception('param value (o) is probably not correct (derivative_growth_rate())')

# ---------------------------------------------------------------------------------------------------------


# get galaxy galaxy power spectrum / P_mm  (prefactor)
def gg_redshift_s_at_k_real(bg, rg, f_t, sigmag, mu_real, ks_real, z, H0_t, q_para, q_perp): 
    'Function to get the prefactor for P_gg(k_fid).'  
    Dg2 = 1/(1 + ((ks_real*sigmag*mu_real)**2)/2)
    res = (bg**2)*Dg2/(q_para*(q_perp**2))
    res += (2*rg*(mu_real**2)*bg*f_t)*Dg2/(q_para*(q_perp**2))
    res += ((mu_real**4)*(f_t**2))*Dg2/(q_para*(q_perp**2))
    return res


# get galaxy div velocity power spectrum / P_mm  (prefactor)
def gu_redshift_s_at_k_real(bg, rg, f_t, sigmag, sigmau, mu_real, ks_real, z, H0_t, omegam_t, omegalambda_t, q_para, q_perp): 
    'Function to get the prefactor for P_gu(k).'  
    Dg = 1/(np.sqrt(1 + ((ks_real*sigmag*mu_real)**2)/2))
    a = 1.0/(1+z)
    H_z_t = H0_t*np.sqrt( omegam_t*((1.0+z)**3) + omegalambda_t )
    Dmu = np.sinc(ks_real*sigmau/np.pi)   
    res = a*H_z_t*mu_real*(rg*bg*f_t)*Dmu*Dg/(ks_real*q_para*(q_perp**2))
    res += a*H_z_t*mu_real*((mu_real*f_t)**2)*Dmu*Dg/(ks_real*q_para*(q_perp**2))            
    return res


# div v. div v. power spectrum / P_mm (true for linear case)  - (need to multiply by P_mm)
def uu_redshift_s_at_k_real(sigmau, mu_real, f_t, ks_real, z, H0_t, omegam_t, omegalambda_t, q_para, q_perp):  
    'Function to get the prefactor for P_uu(k).'  
    a = 1.0/(1+z)
    H_z_t = H0_t*np.sqrt( omegam_t*((1.0+z)**3) + omegalambda_t )
    Dmu = np.sinc(ks_real*sigmau/np.pi)
    res = ((a*H_z_t*mu_real*f_t/ks_real)**2)*(Dmu**2)/(q_para*(q_perp**2))
    return res

# ---------------------------------------------------------------------------------------------------------



# function to compute the derivative of real k (k = k(k_fid, z, cosmological parameters...)) with respect to cosmological parameters 
def dk_real_dx(param, dF_dx, dq_perp_dx, mu_fid, k_fid, F, q_perp, q_parallel):
    '''
    Function to compute the derivative of k(k_fid, mu_fid, F_distortion_ratio) with respect
    to H0, Obh2, Och2 or Mnu.
    '''
    if param == 1 or param == 2 or param == 3 or param == 4: #H0, Mnu, Obh, Och respectively

        p1 = np.sqrt( 1.0 + (mu_fid**2)*( 1.0/(F**2)  - 1.0 ) )
        p2 = 1.0/p1
        res = -1.0*((k_fid)/(q_perp**2))*p1*dq_perp_dx - ((k_fid*(mu_fid**2))/(q_perp*(F**3)))*p2*dF_dx
        return res

    else: # Any other derivatives are always zero 

        raise Exception('param value (o) is probably not correct (dk_real_dx())')


# function to integrate numerically (its for another function, to get dmu/dx etc. where x is a cosmological parameter)
def func_2_integrate(z, Om, H0):
    '''
    Function to integrate to compute derivatives of k and mu real w.r.t. cosmo parameters. 
    '''
    return H0/(get_Hubble_z(z, Om, H0)**3) 


# another weird function to integrate numerically 
def func_2_integrate_2(z, Om, H0):
    '''
    Function to integrate to compute derivatives of k and mu real w.r.t. cosmo parameters. 
    '''
    return H0*((1.0+z)**3 - 1.0)/(get_Hubble_z(z, Om, H0)**3) 


# function to compute the derivative of the distortion ratio F with respect to cosmological parameters
def dF_distortion_dx(param, F, H0_t, z, Om_t, c = 299792.458):
    '''
    Function to compute the derivative of F = q_perp/q_parallel with respect
    to H0, Obh2, Och2 or Mnu (all other derivatives = 0).
    '''
    H_z_t = get_Hubble_z(z, Om_t, H0_t)
    E_z = H_z_t/H0_t
    h_t = H0_t/100.0
    D_a_z = angular_diameter_distance(Om_t, H0_t, z)
    if param == 1: # H0
        if z == 0:
            res = 0.0
            return res
        else:
            weird_integral = quad(func_2_integrate, 0.0, z, args = (Om_t, H0_t))[0]
            res = -1.0*F/(H_z_t*E_z)
            res += c*F/(D_a_z*(1.0+z))*weird_integral
            return res

    elif param == 2 or param == 3 or param == 4: # Mnu, Obh, Och
        if z == 0:
            res = 0.0
            return res
        else: 
            weird_integral = quad(func_2_integrate_2, 0.0, z, args = (Om_t, H0_t))[0]
            res = -1.0*((F*H0_t)/(2.0*(h_t**2)*H_z_t*E_z))*((1.0 + z)**3 - 1.0)
            res += 1.0*((c*F)/((1.0+z)*D_a_z))*(H0_t/(2.0*(h_t**2)))*weird_integral
            if param == 2:
                res /= 93.14
            return res

    else:
        raise Exception('param value (o) is probably not correct (dF_distortion_dx()).')


# function to compute the derivative of real mu with respect to cosmological parameters 
def dmu_real_dx(dF_dx, mu_fid, F, param, q_parallel, q_perp, k_fid):
    '''
    Function to compute the derivative of mu(mu_fid, F_distortion_ratio) with respect
    to H0, Obh2, Och2 or Mnu (all other derivatives = 0).
    '''
    if param == 1 or param == 2 or param == 3 or param == 4: 
        p1 = 1.0/np.sqrt( 1.0 + (mu_fid**2)*( 1.0/(F**2) - 1.0 ))
        p2 = p1**3
        res = (- (mu_fid/(F**2))*p1 + (((mu_fid**3)/(F**4)))*p2 )*dF_dx 
        return res

    else:

        raise Exception('param value (o) is probably not correct (dmu_real_dx()).')


# function to compute the derivative of the perpendicular distortion parameter with respect to cosmological parameters 
def dq_perp_distortion_dx(param, H0_t, H0_fid, Om_t, Om_fid, z, c = 299792.458):
    '''
    Function to compute the derivative of q_perpendicular with respect
    to H0, Obh2, Och2 or Mnu (all other derivatives = 0).
    '''
    D_a_real = angular_diameter_distance(Om_t, H0_t, z)
    D_a_fid = angular_diameter_distance(Om_fid, H0_fid, z)
    h_t = H0_t/100.0
    if param == 1: # H0

        if z == 0:
            res = 0.0
            return res
        else:
            integral = quad(func_2_integrate, 0.0, z, args=(Om_fid, H0_fid))[0]
            res = (D_a_real/(D_a_fid*H0_fid)) - ((c*H0_t)/(D_a_fid*H0_fid))*(1.0/(1.0 + z))*integral
            return res

    elif param == 2 or param == 3 or param == 4: # H0, Mnu, Obh, Och

        if z == 0:
            res = 0.0
            return res
        else: 
            integral = quad(func_2_integrate_2, 0.0, z, args = (Om_t, H0_t))[0]
            res = -1.0*((c*(H0_t**2))/(2.0*(h_t**2)*D_a_fid*H0_fid*(1.0+z)))*integral
            if param == 2:
                res /= 93.14
            return res

    else:
        raise Exception('param value (o) is probably not correct (dq_perp_distortion_dx()).')
    

# function to compute the derivative of the parallel distortion parameter with respect to cosmological parameters
def dq_para_distortion_dx(param, H0_t, H0_fid, Om_t, Om_fid, z, c = 299792.458):
    '''
    Function to compute the derivative of q_parallel with respect
    to H0, Obh2, Och2 or Mnu (all other cosmo derivatives = 0).
    '''

    H_z_t = get_Hubble_z(z, Om_t, H0_t)
    H_z_f = get_Hubble_z(z, Om_fid, H0_fid)
    h = H0_t/100.0

    if param == 1: #H0

        if z == 0:
            res = 0
            return res
        else:
            res = (H_z_f/H0_fid)*( 1.0/H_z_t - 1.0*(H0_t**2)/((H_z_t**3)))
            return res

    elif param == 2 or param == 3 or param == 4: # Mnu, Obh or Och

        res = -0.5*((H_z_f*(H0_t**3))/(H0_fid*(H_z_t**3)*(h**2)))*( (1+z)**3 - 1.0 )
        if param == 2:
            res /= 93.14
        return res

    else:

        raise Exception('param value is probably not correct (dQ_para_distortion_dx()).')


# get derivatives of galaxy galaxy power spectrum w.r.t. relevant parameter (need to pass in P_mm and dP_dx)
def dP_gg_dx(param, bg, rg, f_t, df_dx_t, mu_real, k_real, sigmag, P_mm_t, dP_mm_dx_t, z, H0_t, dmu_real_dx, dk_real_dx, q_para, q_perp,
    dq_para_dx, dq_perp_dx): 
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
    Dg2 = 1/(1 + ( (mu_real*k_real*sigmag)**2 )/2)

    if param == 1: # varying H0
        res = (bg**2)*dP_mm_dx_t*Dg2/(q_para*(q_perp**2))
        res += (bg**2)*P_mm_t*(((k_real*mu_real*sigmag)**2)/H0_t)*(Dg2**2)/(q_para*(q_perp**2))

        res += 2*rg*bg*f_t*(mu_real**2)*dP_mm_dx_t*Dg2/(q_para*(q_perp**2))
        res += (2*rg*bg*f_t*(mu_real**2))*P_mm_t*(((k_real*mu_real*sigmag)**2)/H0_t)*(Dg2**2)/(q_para*(q_perp**2)) 
        res += (2*rg*bg*df_dx_t*(mu_real**2))*P_mm_t*Dg2/(q_para*(q_perp**2))

        res += (f_t**2)*(mu_real**4)*Dg2*dP_mm_dx_t/(q_para*(q_perp**2))
        res += (f_t**2)*(mu_real**4)*P_mm_t*(((k_real*mu_real*sigmag)**2)/H0_t)*(Dg2**2)/(q_para*(q_perp**2)) 
        res += 2*f_t*(mu_real**4)*(df_dx_t)*P_mm_t*Dg2/(q_para*(q_perp**2))

        #res += (all extra terms due to AP Effect need to be added here)
        res += 4.0*dmu_real_dx*(bg*rg*f_t*mu_real)*Dg2*P_mm_t/(q_para*(q_perp**2))
        res += 4.0*dmu_real_dx*((mu_real**3)*(f_t**2))*Dg2*P_mm_t/(q_para*(q_perp**2))

        res += -1.0*(bg**2)*(sigmag**2)*(Dg2**2)*k_real*(mu_real**2)*dk_real_dx*P_mm_t/(q_para*(q_perp**2))
        res += -1.0*(bg**2)*(sigmag**2)*(Dg2**2)*mu_real*(k_real**2)*dmu_real_dx*P_mm_t/(q_para*(q_perp**2))

        res += -1.0*( 2.0*rg*bg*f_t*(mu_real**2) )*(sigmag**2)*(Dg2**2)*k_real*(mu_real**2)*dk_real_dx*P_mm_t/(q_para*(q_perp**2))
        res += -1.0*( 2.0*rg*bg*f_t*(mu_real**2) )*(sigmag**2)*(Dg2**2)*mu_real*(k_real**2)*dmu_real_dx*P_mm_t/(q_para*(q_perp**2))

        res += -1.0*( (f_t**2)*(mu_real**4) )*(sigmag**2)*(Dg2**2)*k_real*(mu_real**2)*dk_real_dx*P_mm_t/(q_para*(q_perp**2))
        res += -1.0*( (f_t**2)*(mu_real**4) )*(sigmag**2)*(Dg2**2)*mu_real*(k_real**2)*dmu_real_dx*P_mm_t/(q_para*(q_perp**2))

        res += -1.0*(1.0/((q_perp*q_para)**2))*(dq_para_dx)*( bg**2 + 2.0*rg*bg*f_t*(mu_real**2) + (f_t**2)*(mu_real**4))*Dg2*P_mm_t
        res += -1.0*(2.0/((q_para)*(q_perp**3)))*(dq_perp_dx)*( bg**2 + 2.0*rg*bg*f_t*(mu_real**2) + (f_t**2)*(mu_real**4))*Dg2*P_mm_t

        return res
    
    elif param == 2 or param == 3 or param == 4 or param == 9: # varying obh, och, mnu, As
        res = ( bg**2 )*dP_mm_dx_t*Dg2/(q_para*(q_perp**2))
        #res = 0
        res += (2*rg*bg*f_t*(mu_real**2))*dP_mm_dx_t*Dg2/(q_para*(q_perp**2))
        res += ( 2*rg*bg*df_dx_t*(mu_real**2) )*P_mm_t*Dg2/(q_para*(q_perp**2))

        res += ((mu_real**4)*(f_t**2))*dP_mm_dx_t*Dg2/(q_para*(q_perp**2))
        res += ( 2.0*f_t*(mu_real**4)*(df_dx_t) )*P_mm_t*Dg2/(q_para*(q_perp**2))

        #res += (all extra terms due to AP Effect need to be added here)

        res += 4.0*dmu_real_dx*(bg*rg*f_t*mu_real)*Dg2*P_mm_t/(q_para*(q_perp**2))
        res += 4.0*dmu_real_dx*((mu_real**3)*(f_t**2))*Dg2*P_mm_t/(q_para*(q_perp**2))

        res += -1.0*(bg**2)*(sigmag**2)*(Dg2**2)*k_real*(mu_real**2)*dk_real_dx*P_mm_t/(q_para*(q_perp**2))
        res += -1.0*(bg**2)*(sigmag**2)*(Dg2**2)*mu_real*(k_real**2)*dmu_real_dx*P_mm_t/(q_para*(q_perp**2))

        res += -1.0*( 2.0*rg*bg*f_t*(mu_real**2) )*(sigmag**2)*(Dg2**2)*k_real*(mu_real**2)*dk_real_dx*P_mm_t/(q_para*(q_perp**2))
        res += -1.0*( 2.0*rg*bg*f_t*(mu_real**2) )*(sigmag**2)*(Dg2**2)*mu_real*(k_real**2)*dmu_real_dx*P_mm_t/(q_para*(q_perp**2))

        res += -1.0*( (f_t**2)*(mu_real**4) )*(sigmag**2)*(Dg2**2)*k_real*(mu_real**2)*dk_real_dx*P_mm_t/(q_para*(q_perp**2))
        res += -1.0*( (f_t**2)*(mu_real**4) )*(sigmag**2)*(Dg2**2)*mu_real*(k_real**2)*dmu_real_dx*P_mm_t/(q_para*(q_perp**2))

        res += -1.0*(1.0/((q_perp*q_para)**2))*(dq_para_dx)*( bg**2 + 2.0*rg*bg*f_t*(mu_real**2) + (f_t**2)*(mu_real**4))*Dg2*P_mm_t
        res += -1.0*(2.0/((q_para)*(q_perp**3)))*(dq_perp_dx)*( bg**2 + 2.0*rg*bg*f_t*(mu_real**2) + (f_t**2)*(mu_real**4))*Dg2*P_mm_t

        return res

    elif param == 5: # varying sigma g
        Dg4 = Dg2**2
        res = -P_mm_t*( bg**2 + 2*rg*bg*f_t*(mu_real**2) + (mu_real**4)*(f_t**2))*( (sigmag*((mu_real*k_real)**2)) )*Dg4/(q_para*(q_perp**2))
        return res 

    elif param == 6: # varying galaxy bias
        res = (2*bg + 2*rg*f_t*(mu_real**2))*Dg2*P_mm_t/(q_para*(q_perp**2))
        return res

    elif param == 7: # varying rg
        res = (2*bg*f_t*(mu_real**2))*Dg2*P_mm_t/(q_para*(q_perp**2))
        return res

    elif param == 11 or param == 10: # n_s (11) or Neff (10)

        res = ( bg**2 + 2*rg*bg*f_t*(mu_real**2) + (mu_real**4)*(f_t**2) )*dP_mm_dx_t
        res += ( 2.0*rg*bg*(mu_real**2) + 2.0*f_t*(mu_real**4)  )*df_dx_t*P_mm_t
        res *= Dg2/(q_para*(q_perp**2))
        return res 

    else:
        raise ValueError('param value (o) is probably not correct')


# get derivatives of galaxy velocity div. power spectrum w.r.t. relevant parameters (need to pass in P_theta-delta and dP_dx)
def dP_gu_dx(param, bg, rg, f_t, df_dx_t, mu_real, k_real, sigmag, sigmau, P_mm_t, dP_mm_dx_t, H0_t, z, omegam_t, omegalambda_t, dk_real_dx, dmu_real_dx,
    q_para, q_perp, dq_para_dx, dq_perp_dx):
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
    Dg = (1/(np.sqrt(1 + ((mu_real*sigmag*k_real)**2)/2)))
    Du = np.sinc(k_real*sigmau/np.pi)
    a = 1.0/(1.0+z)
    h_t = H0_t/100.0
    u = np.sqrt( omegam_t*((1+z)**3) + omegalambda_t )
    H_z_t = H0_t*np.sqrt( omegam_t*((1+z)**3) + omegalambda_t )
    if param == 1: # varying H0
        res = (a*mu_real/k_real)*( rg*bg*f_t + ((mu_real*f_t)**2))*(P_mm_t)*Dg*Du*(1/u)*(1.0/(q_para*(q_perp**2)))
        res += (a*H_z_t*mu_real/k_real)*( rg*bg*f_t + ((mu_real*f_t)**2))*Dg*Du*dP_mm_dx_t*(1.0/(q_para*(q_perp**2)))
        res += (a*H_z_t*mu_real/k_real)*(  rg*bg*df_dx_t + 2*(mu_real**2)*f_t*df_dx_t  )*Du*Dg*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += (a*H_z_t*mu_real/k_real)*( rg*bg*f_t + ((mu_real*f_t)**2) )*(P_mm_t)*Dg*( np.sinc(sigmau*k_real/np.pi) - np.cos(sigmau*k_real) )*(1.0/(q_para*(q_perp**2)))/(H0_t)
        res += (a*H_z_t*mu_real/k_real)*( rg*bg*f_t + ((mu_real*f_t)**2) )*(P_mm_t)*Du*((mu_real*k_real*sigmag)**2)*(Dg**3)*(1.0/(q_para*(q_perp**2)))/(2*H0_t)
        # res += extra terms for AP effect
        res += -1.0*((a*H_z_t*mu_real)/(k_real**2))*dk_real_dx*(f_t*rg*bg)*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += -1.0*((a*H_z_t*mu_real)/(k_real**2))*dk_real_dx*((f_t**2)*(mu_real**2))*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += ((a*H_z_t)/(k_real))*dmu_real_dx*(f_t*rg*bg)*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += ((a*H_z_t)/(k_real))*dmu_real_dx*(3.0*(f_t**2)*(mu_real**2))*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += -1.0*((a*H_z_t*mu_real)/(2*k_real))*(f_t*rg*bg + (f_t**2)*(mu_real**2))*Du*(Dg**3)*(sigmag**2)*((k_real**2)*mu_real*dmu_real_dx)*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += -1.0*((a*H_z_t*mu_real)/(2*k_real))*(f_t*rg*bg + (f_t**2)*(mu_real**2))*Du*(Dg**3)*(sigmag**2)*((mu_real**2)*k_real*dk_real_dx)*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += ((a*H_z_t*mu_real)/(k_real))*(f_t*bg*rg)*Dg*dk_real_dx*( np.cos(k_real*sigmau)/k_real - np.sin(k_real*sigmau)/((k_real**2)*sigmau) )*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += ((a*H_z_t*mu_real)/(k_real))*((f_t**2)*(mu_real**2))*Dg*dk_real_dx*( np.cos(k_real*sigmau)/(k_real) - np.sin(k_real*sigmau)/((k_real**2)*sigmau) )*P_mm_t*(1.0/(q_para*(q_perp**2)))

        res += -1.0*(1.0/((q_para*q_perp)**2))*dq_para_dx*((a*H_z_t*mu_real)/(k_real))*( f_t*bg*rg + (f_t**2)*(mu_real**2) )*Dg*Du*P_mm_t
        res += -1.0*(2.0/((q_para)*(q_perp**3)))*dq_perp_dx*((a*H_z_t*mu_real)/(k_real))*( f_t*bg*rg + (f_t**2)*(mu_real**2) )*Dg*Du*P_mm_t
        return res
        
    elif param == 2 or param == 3 or param == 4 or param == 9: # varying obh, och, mnu, As
        res = (a*H_z_t*mu_real/k_real)*( rg*bg*f_t + ((mu_real*f_t)**2))*Dg*Du*dP_mm_dx_t*(1.0/(q_para*(q_perp**2)))
        res += (a*H_z_t*mu_real/k_real)*(rg*bg*df_dx_t)*Du*P_mm_t*Dg*(1.0/(q_para*(q_perp**2)))
        res += (a*H_z_t*mu_real/k_real)*(2*(mu_real**2))*f_t*df_dx_t*Du*P_mm_t*Dg*(1.0/(q_para*(q_perp**2)))
        if param == 2:
            u = np.sqrt( omegam_t*((1+z)**3) + omegalambda_t )
            dH_dx_t = (H0_t/(2*93.14*u*(h_t**2)))*( (1+z)**3 - 1 )
            res += (a*mu_real/k_real)*(rg*bg*f_t + ((mu_real*f_t)**2))*Du*Dg*P_mm_t*(dH_dx_t)*(1.0/(q_para*(q_perp**2)))
            # + extra terms for AP EFFECT!!!
            res += -1.0*((a*H_z_t*mu_real)/(k_real**2))*dk_real_dx*(f_t*rg*bg)*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += -1.0*((a*H_z_t*mu_real)/(k_real**2))*dk_real_dx*((f_t**2)*(mu_real**2))*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += ((a*H_z_t)/(k_real))*dmu_real_dx*(f_t*rg*bg)*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += ((a*H_z_t)/(k_real))*dmu_real_dx*(3.0*(f_t**2)*(mu_real**2))*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += -1.0*((a*H_z_t*mu_real)/(2*k_real))*(f_t*rg*bg + (f_t**2)*(mu_real**2))*Du*(Dg**3)*(sigmag**2)*((k_real**2)*mu_real*dmu_real_dx)*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += -1.0*((a*H_z_t*mu_real)/(2*k_real))*(f_t*rg*bg + (f_t**2)*(mu_real**2))*Du*(Dg**3)*(sigmag**2)*((mu_real**2)*k_real*dk_real_dx)*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += ((a*H_z_t*mu_real)/(k_real))*(f_t*bg*rg)*Dg*dk_real_dx*( np.cos(k_real*sigmau)/k_real - np.sin(k_real*sigmau)/((k_real**2)*sigmau) )*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += ((a*H_z_t*mu_real)/(k_real))*((f_t**2)*(mu_real**2))*Dg*dk_real_dx*( np.cos(k_real*sigmau)/k_real - np.sin(k_real*sigmau)/((k_real**2)*sigmau) )*P_mm_t*(1.0/(q_para*(q_perp**2)))

            res += -1.0*(1.0/((q_para*q_perp)**2))*dq_para_dx*((a*H_z_t*mu_real)/(k_real))*( f_t*bg*rg + (f_t**2)*(mu_real**2) )*Dg*Du*P_mm_t
            res += -1.0*(2.0/((q_para)*(q_perp**3)))*dq_perp_dx*((a*H_z_t*mu_real)/(k_real))*( f_t*bg*rg + (f_t**2)*(mu_real**2) )*Dg*Du*P_mm_t

        elif param == 3 or param == 4: 
            u = np.sqrt( omegam_t*((1+z)**3) + omegalambda_t )
            dH_dx_t = (H0_t/(2*u*(h_t**2)))*( (1+z)**3 - 1 )
            res += (a*mu_real/k_real)*(rg*bg*f_t + ((mu_real*f_t)**2))*Du*Dg*P_mm_t*(dH_dx_t)*(1.0/(q_para*(q_perp**2)))
            # + extra terms for AP EFFECT!!!
            res += -1.0*((a*H_z_t*mu_real)/(k_real**2))*dk_real_dx*(f_t*rg*bg)*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += -1.0*((a*H_z_t*mu_real)/(k_real**2))*dk_real_dx*((f_t**2)*(mu_real**2))*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += ((a*H_z_t)/(k_real))*dmu_real_dx*(f_t*rg*bg)*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += ((a*H_z_t)/(k_real))*dmu_real_dx*(3.0*(f_t**2)*(mu_real**2))*Dg*Du*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += -1.0*((a*H_z_t*mu_real)/(2*k_real))*(f_t*rg*bg + (f_t**2)*(mu_real**2))*Du*(Dg**3)*(sigmag**2)*((k_real**2)*mu_real*dmu_real_dx)*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += -1.0*((a*H_z_t*mu_real)/(2*k_real))*(f_t*rg*bg + (f_t**2)*(mu_real**2))*Du*(Dg**3)*(sigmag**2)*((mu_real**2)*k_real*dk_real_dx)*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += ((a*H_z_t*mu_real)/(k_real))*(f_t*bg*rg)*Dg*dk_real_dx*( np.cos(k_real*sigmau)/k_real - np.sin(k_real*sigmau)/((k_real**2)*sigmau) )*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += ((a*H_z_t*mu_real)/(k_real))*((f_t**2)*(mu_real**2))*Dg*dk_real_dx*( np.cos(k_real*sigmau)/k_real - np.sin(k_real*sigmau)/((k_real**2)*sigmau) )*P_mm_t*(1.0/(q_para*(q_perp**2)))

            res += -1.0*(1.0/((q_para*q_perp)**2))*dq_para_dx*((a*H_z_t*mu_real)/(k_real))*( f_t*bg*rg + (f_t**2)*(mu_real**2) )*Dg*Du*P_mm_t
            res += -1.0*(2.0/((q_para)*(q_perp**3)))*dq_perp_dx*((a*H_z_t*mu_real)/(k_real))*( f_t*bg*rg + (f_t**2)*(mu_real**2) )*Dg*Du*P_mm_t

        return res

    elif param == 5: # varying sigma g
        res = -1.0*(1.0/(q_para*(q_perp**2)))*(a*H_z_t*mu_real/(2*k_real))*(rg*bg*f_t + (mu_real*f_t)**2)*Du*P_mm_t*(sigmag*(mu_real*k_real)**2)/((np.sqrt(1 + ((mu_real*sigmag*k_real)**2)/2))**3)
        return res

    elif param == 6: # varying galaxy bias
        res = (1.0/(q_para*(q_perp**2)))*a*H_z_t*mu_real*rg*f_t*Du*P_mm_t*Dg/k_real
        return res

    elif param == 7: # varying rg
        res = (1.0/(q_para*(q_perp**2)))*(a*H_z_t*bg*f_t*Du*P_mm_t*mu_real)*Dg/k_real
        return res

    elif param == 8: # varying sigmau
        res = (1.0/(q_para*(q_perp**2)))*(a*H_z_t*mu_real/k_real)*( rg*bg*f_t + (mu_real*f_t)**2 )*P_mm_t*((k_real*sigmau*np.cos(k_real*sigmau)) - (np.sin(k_real*sigmau)))/(k_real*(sigmau**2))
        res *= Dg
        return res

    elif param == 10 or param == 11: # Neff or ns 

        res = ( rg*bg*f_t + (mu_real*f_t)**2 )*dP_mm_dx_t
        res += ( rg*bg + 2.0*(mu_real**2)*f_t )*df_dx_t*P_mm_t
        res *= (a*H_z_t*mu_real/k_real)*(1.0/(q_para*(q_perp**2)))*Dg
        return res 

    else:
        raise ValueError('param value (o) is probably not correct')



# get derivatives of vel div. vel div. power spectrum w.r.t. relevant parameters (need to pass in P_thetatheta and dP_dx)
def dP_uu_dx(param, mu_real, k_real, f_t, df_dx_t, sigmau, P_mm_t, dP_mm_dx_t, H0_t, z, omegam_t, omegalambda_t, dk_real_dx, dmu_real_dx,
    q_para, q_perp, dq_para_dx, dq_perp_dx):
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
    Du2 = (np.sinc(k_real*sigmau/np.pi))**2
    H_z_t = H0_t*np.sqrt( omegam_t*((1+z)**3) + omegalambda_t )
    a = 1.0/(1.0 + z)
    u = np.sqrt( omegam_t*((1+z)**3) + omegalambda_t )
    if param == 1: # varying H0
        res = ((a*mu_real/k_real)**2)*((f_t**2))*(Du2)*P_mm_t*( 2*H_z_t*(1/u) )*(1.0/(q_para*(q_perp**2)))
        res += ((a*mu_real*H_z_t/k_real)**2)*(2*f_t)*df_dx_t*(Du2)*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += ((a*mu_real*H_z_t/k_real)**2)*(f_t**2)*Du2*dP_mm_dx_t*(1.0/(q_para*(q_perp**2)))
        res += ((a*H_z_t*mu_real/k_real)**2)*P_mm_t*(f_t**2)*(2*np.sinc(k_real*sigmau/np.pi))*((np.sinc(k_real*sigmau/np.pi) - np.cos(sigmau*k_real))/H0_t)*(1.0/(q_para*(q_perp**2)))
        # res += Extra terms for AP Effect
        res += 2.0*mu_real*dmu_real_dx*(((a*H_z_t*f_t)/(k_real))**2)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += -2.0*(((a*H_z_t*f_t*mu_real)/(k_real))**2)*(dk_real_dx/k_real)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += 2.0*(((a*H_z_t*f_t*mu_real)/(k_real))**2)*np.sqrt(Du2)*dk_real_dx*P_mm_t*( np.cos(k_real*sigmau)/(k_real) - np.sin(k_real*sigmau)/((k_real**2)*sigmau) )*(1.0/(q_para*(q_perp**2)))

        res += (-1.0/((q_perp*q_para)**2))*(((a*H_z_t*f_t*mu_real)/(k_real))**2)*Du2*P_mm_t*dq_para_dx
        res += (-2.0/(q_para*(q_perp**3)))*(((a*H_z_t*f_t*mu_real)/(k_real))**2)*Du2*P_mm_t*dq_perp_dx
        return res
    
    elif param == 2 or param == 3 or param == 4 or param == 9: # varying obh, och, mnu, As
        res = ((a*mu_real*H_z_t/k_real)**2)*(2*f_t*df_dx_t)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
        res += ((a*mu_real*H_z_t/k_real)**2)*((f_t)**2)*Du2*dP_mm_dx_t*(1.0/(q_para*(q_perp**2)))
        if param == 2:
            dH_dxsqrd = (H0_t/(2*93.14*u*(h_t**2)))*( (1+z)**3 - 1 )*(2*H_z_t)
            res += ((a*mu_real/k_real)**2)*((f_t)**2)*Du2*P_mm_t*(dH_dxsqrd)*(1.0/(q_para*(q_perp**2)))
            # res += EXTRA TERMS AP EFFECT
            res += 2.0*mu_real*dmu_real_dx*(((a*H_z_t*f_t)/(k_real))**2)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += -2.0*(((a*H_z_t*f_t*mu_real)/(k_real))**2)*(dk_real_dx/k_real)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += 2.0*(((a*H_z_t*f_t*mu_real)/(k_real))**2)*np.sqrt(Du2)*dk_real_dx*P_mm_t*( np.cos(k_real*sigmau)/(k_real) - np.sin(k_real*sigmau)/((k_real**2)*sigmau) )*(1.0/(q_para*(q_perp**2)))

            res += (-1.0/((q_perp*q_para)**2))*(((a*H_z_t*f_t*mu_real)/(k_real))**2)*Du2*P_mm_t*dq_para_dx
            res += (-2.0/(q_para*(q_perp**3)))*(((a*H_z_t*f_t*mu_real)/(k_real))**2)*Du2*P_mm_t*dq_perp_dx

        elif param == 3 or param == 4:
            dH_dxsqrd =  (H0_t/(2*u*(h_t**2)))*( (1+z)**3 - 1 )*(2*H_z_t)
            res += ((a*mu_real/k_real)**2)*((f_t)**2)*Du2*P_mm_t*(dH_dxsqrd)*(1.0/(q_para*(q_perp**2)))
            # res += extra terms AP EFFECT
            res += 2.0*mu_real*dmu_real_dx*(((a*H_z_t*f_t)/(k_real))**2)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += -2.0*(((a*H_z_t*f_t*mu_real)/(k_real))**2)*(dk_real_dx/k_real)*Du2*P_mm_t*(1.0/(q_para*(q_perp**2)))
            res += 2.0*(((a*H_z_t*f_t*mu_real)/(k_real))**2)*np.sqrt(Du2)*dk_real_dx*P_mm_t*( np.cos(k_real*sigmau)/(k_real) - np.sin(k_real*sigmau)/((k_real**2)*sigmau) )*(1.0/(q_para*(q_perp**2)))

            res += (-1.0/((q_perp*q_para)**2))*(((a*H_z_t*f_t*mu_real)/(k_real))**2)*Du2*P_mm_t*dq_para_dx
            res += (-2.0/(q_para*(q_perp**3)))*(((a*H_z_t*f_t*mu_real)/(k_real))**2)*Du2*P_mm_t*dq_perp_dx

        return res

    elif param == 8: # varying sigmau
        res = (1.0/(q_para*(q_perp**2)))*((a*H_z_t*mu_real*f_t/k_real)**2)*(2*np.sinc(k_real*sigmau/np.pi))*(k_real*sigmau*np.cos(k_real*sigmau) - np.sin(k_real*sigmau))*P_mm_t/(k_real*(sigmau**2))
        return res

    elif param == 10 or param == 11: # varying Neff or varying ns 

        res = ((a*H_z_t*mu_real/k_real)**2)*(f_t**2)*dP_mm_dx_t
        res += 2.0*((a*H_z_t*mu_real/k_real)**2)*df_dx_t*f_t*P_mm_t
        res *= (1.0/(q_para*(q_perp**2)))*Du2
        return res 

    else:
        raise ValueError('param value (o) is probably not correct')



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

def derivative_power_spectra(o, P, z, plot_or_write2file, central_params,
delta_max_param, kmin_fiducial, kmax_fiducial, num_k_points, num_mus, da, neutrino_hier, 
lin_or_nonlin, kspaceoption, tau, ns, dm2_atm = 2.5e-3, dm2_sol = 7.6e-5):
    """ 
    Function to compute derivatives or make a plot of some derivatives for some parameter (given by o), 
    some power spectrum (given by P) and some redshift z.
    To compute derivatives of the galaxy-galaxy redshift space power spectrum, enter P = 1.
    To compute derivatives of the galaxy-velocity redshift space power spectrum, enter P = 2.
    To compute derivatives of the velocity-velocity redshift space power spectrum, enter P = 3.
    To compute the derivative of chosen power spectrum w.r.t. some parameter x, set o to:
    x = H0 ->                                   set o = 1 
    x = sum neutrino masses ->                  set o = 2 
    x = omega_b ->                              set o = 3
    x = omega_cdm ->                            set o = 4 
    x = sigma_g ->                              set o = 5 (parameter related to the pairwise 
                                                            velocity dispersion between galaxies)
    x = b ->                                    set o = 6 (galaxy bias)
    x = r ->                                    set o = 7 (correlation between velocity and density fields)
    x = sigma_u ->                              set o = 8 (parameter related to non-linear motion of galaxies)
    x = As ->                                   set o = 9 
    x = N_eff ->                                set o = 10 (the effective number of neutrino species)
    x = n_s ->                                  set o = 11 (the spectral index)
    
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

    if P > 3 or P <= 0:
        raise Exception('P must be set to either 1, 2, 3 (use help() for more info.). ') 


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
        ks_fiducial = np.logspace(np.log10(kmin_fiducial), np.log10(kmax_fiducial), num_k_points, base=10) # k values (k in 1/MPC)
    else:
        ks_fiducial = np.linspace(kmin_fiducial, kmax_fiducial, num_k_points)

    mus_fiducial = np.linspace(0, 1, num_mus) # set the mu values 


    # unpack the list of central values for each parameter
    Obh_central, mnu_central, H0_central, Och_central, As_central, sigmag_central, bg_central, rg_central, sigmau_central = central_params
    
    param_variation = [] # initialising a list that will store the values of our parameter we vary
    deltas = np.zeros(num_steps)   # initializing a list that tells us the step sizes for our parameters from
    # the central values of cosmological parameters  

    if o == 1: # varying H0
        param_variation = np.linspace(H0_central-delta_max_param, H0_central+delta_max_param, num_steps)
    elif o == 2: # varying sum of neutrino masses
        param_variation = np.linspace(mnu_central-delta_max_param, mnu_central+delta_max_param, num_steps)
    elif o == 3: # varying omega_b
        param_variation = np.linspace(Obh_central-delta_max_param, Obh_central+delta_max_param, num_steps)
    elif o == 4: # varying omega_cdm
        param_variation = np.linspace(Och_central-delta_max_param, Och_central+delta_max_param, num_steps)
    elif o == 5: # varying sigma_g
        param_variation = np.linspace(sigmag_central-delta_max_param, sigmag_central+delta_max_param, num_steps)
    elif o == 6: # galaxy bias
        param_variation = np.linspace(bg_central-delta_max_param, bg_central+delta_max_param, num_steps)
    elif o == 7: # correlation between the velocity and density fields
        param_variation = np.linspace(rg_central-delta_max_param, rg_central+delta_max_param, num_steps)
    elif o == 8: # sigma_u
        param_variation = np.linspace(sigmau_central-delta_max_param, sigmau_central+delta_max_param, num_steps)
    elif o == 9: # As
        param_variation = np.linspace(As_central-delta_max_param, As_central+delta_max_param, num_steps)
    elif o == 10: #N_eff
        param_variation = np.linspace(-delta_max_param, delta_max_param, num_steps)
    elif o == 11: #n_s
        param_variation = np.linspace(ns-delta_max_param, ns+delta_max_param, num_steps)
    
    else:
        print('Parameter chosen is not an option (in derivative_power_spectra()).')
        exit()

    for i in range(num_steps): # get differences between central value of parameter we are varying and shifted value
            deltas[i] = abs(param_variation[int((num_steps-1)/2.0)] - param_variation[i])
    deltas = np.delete(deltas, int((num_steps-1)/2.0)) 


    # initialise arrays to store results to later -----------------------------------------------

    # store matter power spectrum for all k_real, mu_real, and values of varying parameter
    matter_power_spectrum_arr = np.zeros((num_steps, num_mus, num_k_points))

    # store real ks
    ks_real_arr = np.zeros((num_steps, num_mus, num_k_points))

    # store real mus 
    mus_real_arr = np.zeros((num_steps, num_mus))

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
        if o == 1:
            H0 = param_variation[i]
            h = param_variation[i]/100.0
            sigmag = sigmag_central/h
            sigmau = sigmau_central/h
        elif o == 2:
            mnu = param_variation[i]
            delta_mnu_max = delta_max_param
        elif o == 3:
            Obh = param_variation[i]
        elif o == 4:
            Och = param_variation[i]
        elif o == 5: 
            sigmag = param_variation[i]/h # MPC
        elif o == 6: 
            bg = param_variation[i]
        elif o == 7: 
            rg = param_variation[i]
        elif o == 8: 
            sigmau = param_variation[i]/h # MPC
        elif o == 9:
            As = param_variation[i]
        elif o == 10:
            N_eff_deviation = param_variation[i]
        elif o == 11:
            ns_val = param_variation[i]

        # get omega_m
        omegam = (Obh + Och + (mnu/93.14))/(h**2)

        
        
        Pk_cb, ks, mus = run_class(Obh, Och, H0, As, mnu, neutrino_hier, [z], kmin_fiducial, kmax_fiducial, num_k_points, dm2_atm, dm2_sol,
        delta_mnu_max, mnu_central, kspaceoption, tau, ns_val, Obh_central, Och_central, H0_central, As_central, mnu_central, mus_fiducial, lin_or_nonlin,
        N_eff_deviation)


        f = growth_rate(Obh, Och, H0, As, mnu, neutrino_hier, z, kmin_fiducial, kmax_fiducial, num_k_points, da, linear, dm2_atm, dm2_sol,
        delta_mnu_max, mnu_central, kspaceoption, tau, ns_val, Obh_central, Och_central, H0_central, As_central, mnu_central, mus_fiducial,
        N_eff_deviation)[0]


        if Pk_cb.shape == (num_mus, num_k_points, 1) and ks.shape == (num_mus, num_k_points, 1) and mus.shape == (num_mus, 1) and f.shape == (num_mus, num_k_points):

            matter_power_spectrum_arr[i,:,:] = Pk_cb[:,:,0] # storing the matter power spectrum for all ks and mus (real)
            ks_real_arr[i,:,:] = ks[:,:,0]
            mus_real_arr[i,:] = mus[:,0]
            growth_rates_arr[i,:,:] = f


        elif Pk_cb.shape == (num_k_points, 1) and ks.shape == (num_k_points,) and mus.shape == (num_mus,) and f.shape == (num_k_points,):

            for muu in np.arange(num_mus):
                matter_power_spectrum_arr[i,muu,:] = Pk_cb[:,0] # just storing the same matter power spectrum since the k modes don't change with mu
                ks_real_arr[i,muu,:] = ks 
                growth_rates_arr[i,muu,:] = f 
                mus_real_arr[i,:] = mus


        else:
            print('shape Pk_cb: ', Pk_cb.shape)
            print('shape ks: ', ks.shape)
            print('shape mus:', mus.shape)
            print('shape f: ', f.shape)
            raise Exception(('Shape of Pk_cb / k / mu /growth rate array returned from run_class() is unexpected and cannot be adapted ' + 
            'to store in matter_power_spectrum_arr / ks_real_arr / mus_real_arr / growth_rates_arr (derivative_power_spectra()).'))

            

        # now compute the redshift space power spectrum for all the needs ks and mus 
        q_para = distortion_parallel(As, Obh, Och, H0, mnu, As_central, Obh_central, Och_central, H0_central, mnu_central, z)
        q_perp = distortion_perpendicular(As, Obh, Och, H0, mnu, As_central, Obh_central, Och_central, H0_central, mnu_central, z)
        for j in np.arange(len(mus_fiducial)): # calculate redshift space power spectrum of choice for all mus 

            if P == 1: # redshift galaxy galaxy
            
                pkrss = matter_power_spectrum_arr[i,j,:]*(gg_redshift_s_at_k_real(bg, rg, growth_rates_arr[i, j, :], 
                sigmag, mus_real_arr[i,j], ks_real_arr[i,j,:], z, H0, q_para, q_perp)) 
                results_redshiftspacepowerspectrum_arr[i,j,:] = pkrss

            elif P == 2: # redshift galaxy velocity
                
                pkrss = matter_power_spectrum_arr[i,j,:]*(gu_redshift_s_at_k_real(bg, rg, growth_rates_arr[i, j, :], 
                sigmag, sigmau, mus_real_arr[i,j], ks_real_arr[i,j,:], z,H0, omegam, 1-omegam, q_para, q_perp))
                results_redshiftspacepowerspectrum_arr[i,j,:] = pkrss 

            elif P == 3: # redshift velocity velocity 
                
                pkrss = matter_power_spectrum_arr[i,j,:]*(uu_redshift_s_at_k_real(sigmau, mus_real_arr[i, j], 
                growth_rates_arr[i,j,:], ks_real_arr[i,j,:], z, H0, omegam, 1-omegam, q_para, q_perp)) 
                results_redshiftspacepowerspectrum_arr[i,j,:] = pkrss


    # --------------------------------------------------------------------------------------- 

    # get derivatives with just simple central difference method for matter power spectrum 
    if o < 5 or o == 9 or o == 10 or o == 11: # derivatives are zero if the varied paramater is not a cosmological parameter
        for j in np.arange((int((num_steps-1)/2))):
            matter_central_diffs[j,:,:] = (matter_power_spectrum_arr[num_steps-1-j,:] 
            - matter_power_spectrum_arr[j,:,:])/(2.0*deltas[num_steps-2-j])
                   
    # ---------------------------------------------------------------------------------------

    # get derivatives with just simple central difference method for redshift 
    # space power spectrum of choice (no expansion/no analytic derivatives)
    for j in np.arange(int((num_steps-1)/2)):
        if o == 5 or o == 8: # need to include 1/h in delta
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
    hsqrd = h**2
    bg = bg_central
    rg = rg_central
    Obh = Obh_central
    Och = Och_central
    mnu = mnu_central
    As = As_central
    omegam = ( Obh_central + Och_central + (mnu_central/93.14) )/(hsqrd)
    N_eff_deviation = 0
    delta_mnu_max = 0.0
    ns_val = ns 
    if o == 2:
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
        q_parallel = 1.0
        dq_perp_dx = 0.0  
        dq_para_dx = 0.0


        if o < 5 or o == 9 or o == 13 or o == 10: # varying cosmological parameter x = non-zero df_dx 
            # getting derivative of the growth rate w.r.t. parameter being varied
            if o < 5 or o == 9:
                df_dx = derivative_growth_rate(o, stepsize, Obh, Och, H0, As, mnu, neutrino_hier, 
                z, kmin_fiducial, kmax_fiducial, num_k_points, da, linear, dm2_atm, dm2_sol, delta_mnu_max, 
                mnu_central, kspaceoption, tau, ns, Obh_central, Och_central, H0_central, As_central, mnu_central,
                mus_fiducial, N_eff_deviation)
            elif o == 10 or o == 13:
                df_dx[:,:] = derivative_growth_rate(o, stepsize, Obh, Och, H0, As, mnu, neutrino_hier, 
                z, kmin_fiducial, kmax_fiducial, num_k_points, da, linear, dm2_atm, dm2_sol, delta_mnu_max, 
                mnu_central, kspaceoption, tau, ns, Obh_central, Och_central, H0_central, As_central, mnu_central,
                mus_fiducial, N_eff_deviation)
            if o < 5:
                dF_dx = dF_distortion_dx(o, F, H0, z, omegam)
                Om_f = (Obh_central + Och_central + mnu_central/93.14)/(h**2)
                dq_perp_dx = dq_perp_distortion_dx(o, H0, H0_central, omegam, Om_f, z)
                dq_para_dx = dq_para_distortion_dx(o, H0, H0_central, omegam, Om_f, z)
                

        else:
            df_dx = np.zeros((num_mus, num_k_points)) # growth rate only changes w.r.t. cosmological parameters
            dF_dx = 0.0
            dq_perp_dx = 0.0 
            dq_para_dx = 0.0


        for muu in np.arange(len(mus_fiducial)): 


            dk_r_dx = 0
            dmu_r_dx = 0
                
            if o < 5:
                dk_r_dx = dk_real_dx(o, dF_dx, dq_perp_dx, mus_fiducial[muu], ks_fiducial, F, q_perp, q_parallel)
                dmu_r_dx =dmu_real_dx(dF_dx, mus_fiducial[muu], F, o, q_parallel, q_perp, ks_fiducial)
                    
            else:
                dk_r_dx = 0
                dmu_r_dx = 0
                
            if P == 1:
                results_findiff_semi_analytic_derivatives[i, muu, :] = dP_gg_dx(o, bg, rg, growth_rates_arr[3, muu, :], df_dx[muu,:],
                mus_real_arr[3,muu], ks_real_arr[3, muu, :], sigmag, matter_power_spectrum_arr[3,muu,:], matter_central_diffs[i, muu,:],
                z, H0, dmu_r_dx, dk_r_dx, q_parallel, q_perp, dq_para_dx, dq_perp_dx)
                
            if P == 2:
                results_findiff_semi_analytic_derivatives[i, muu, :] = dP_gu_dx(o, bg, rg, growth_rates_arr[3,muu,:], df_dx[muu,:], 
                mus_real_arr[3,muu], ks_real_arr[3,muu,:], sigmag, sigmau, matter_power_spectrum_arr[3, muu, :], 
                matter_central_diffs[i, muu, :], H0, z, omegam, 1.0-omegam, dk_r_dx, dmu_r_dx, q_parallel, q_perp, dq_para_dx, dq_perp_dx)
                    
            if P == 3:
                results_findiff_semi_analytic_derivatives[i, muu, :] = dP_uu_dx(o, mus_real_arr[3, muu], ks_real_arr[3, muu, :], 
                growth_rates_arr[3,muu,:], df_dx[muu,:], sigmau, matter_power_spectrum_arr[3, muu, :], matter_central_diffs[i, muu, :],
                H0, z, omegam, 1.0-omegam, dk_r_dx, dmu_r_dx, q_parallel, q_perp, dq_para_dx, dq_perp_dx)




    # --------------------------finished computing derivatives ------------------------------------------------------------------------------

    if plot1 == True:  # plot some of the derivatives

        ks_real_arr /= (H0_central/100.0) # put ks into units of k/h MPc
        results_findiff_derivatives /= (H0_central/100.0)**3 # putting into units of (MPC/h)^3/x
        results_findiff_semi_analytic_derivatives /= (H0_central/100.0)**3 # putting into units of (MPC/h)^3/x, x being the units of the parameter
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

            if o == 1:
                label = (r"$\Delta H_0 = %.5f $ - M1" %  (deltas[j]) ) 
            if o == 2:
                label = (r"$\Delta \sum m_{\nu} = %.6f $ - M1" %  (deltas[j]) ) 
            if o == 3:
                label = (r"$\Delta \Omega_bh^2 = %.6f $ - M1" %  (deltas[j]) ) 
            if o == 4:
                label = (r"$\Delta \Omega_{cdm}h^2 = %.3f $ - M1" %  (deltas[j]) ) 
            if o == 5:
                label = (r"$\Delta \sigma_g = %.3f $ - M1" %  (deltas[j]) ) 
            if o == 6:
                label = (r"$\Delta b_g=%.3f $ - M1" %  (deltas[j]) ) 
            if o == 7:
                label = (r"$\Delta r_g=%.3f $ - M1" %  (deltas[j]) ) 
            if o == 8:
                label = (r"$\Delta \sigma_{u}=%.3f $ - M1" %  (deltas[j]) )
            if o == 9:
                label = (r"$\Delta A_s=%.4f \times 10^{-9} $ - M1" %  (deltas[j]*1e9) ) 
            if o == 10:
                label = (r"$\Delta N_{eff}=3.046+%.4f $ - M1" %  (deltas[j]) ) 
            if o == 11:
                label = (r'$ \Delta n_s=%.4f $ - M1' % (deltas[j]) )


            dP_dtheta = results_findiff_derivatives[j, mu_indices[0], :]
            ax1.semilogx(ks_fiducial, dP_dtheta, label = label )
            dP_dtheta = results_findiff_derivatives[j, mu_indices[1], :]
            ax2.semilogx(ks_fiducial, dP_dtheta, label = label )
            dP_dtheta = results_findiff_derivatives[j, mu_indices[2], :]
            ax3.semilogx(ks_fiducial, dP_dtheta, label = label )


        # loop through derivative results using 2nd method (finite diff. + analytic derivatives)

        for j in np.arange(int((num_steps-1)/2)):

            if o == 1:
                label = (r"$\Delta H_0 = %.5f $ - M2" %  (deltas[j]) ) 
            if o == 2:
                label = (r"$\Delta \sum m_{\nu}  = %.6f $ - M2" %  (deltas[j]) ) 
            if o == 3:
                label = (r"$\Delta \Omega_bh^2 = %.6f $ - M2" %  ( deltas[j])) 
            if o == 4:
                label = (r"$\Delta \Omega_{cdm}h^2 = %.3f $ - M2" %  (deltas[j]) ) 
            if o == 5:
                label = (r"$\Delta \sigma_g = %.3f $ - M2" %  (deltas[j]) ) 
            if o == 6:
                label = (r"$\Delta b_g=%.3f $ - M2" %  (deltas[j]) ) 
            if o == 7:
                label = (r"$\Delta r_g=%.3f $ - M2" %  (deltas[j]) ) 
            if o == 8:
                label = (r"$\Delta \sigma_{u}=%.3f $ - M2" %  (deltas[j]) ) 
            if o == 9:
                label = (r"$\Delta A_s=%.4f \times 10^{-9} $ - M2" %  (deltas[j]*1e9) ) 
            if o == 10:
                label = (r"$\Delta N_{eff}=3.046+%.4f $ - M2" %  (deltas[j]) ) 
            if o == 11:
                label = (r'$ \Delta n_s=%.4f $ - M2' % (deltas[j]) )

            
            dP_dtheta = results_findiff_semi_analytic_derivatives[j, mu_indices[0], :]
            ax1.semilogx(ks_fiducial, dP_dtheta, linestyle = '--', label = label )
            dP_dtheta = results_findiff_semi_analytic_derivatives[j, mu_indices[1], :]
            ax2.semilogx(ks_fiducial, dP_dtheta, linestyle = '--', label = label )
            dP_dtheta = results_findiff_semi_analytic_derivatives[j, mu_indices[2], :]
            ax3.semilogx(ks_fiducial, dP_dtheta, linestyle = '--', label = label )


        ylabels = [
        r"$ \frac{dP(k, \mu)}{dH_0}$ ", r"$ \frac{dP(k, \mu)}{d\sum m_{\nu} }$", 
        r"$ \frac{dP(k, \mu)}{d\Omega_bh^2}$" , r"$ \frac{dP(k, \mu)}{d\Omega_{cdm}h^2}$", 
        r"$ \frac{dP(k, \mu)}{d\sigma_g}$", r"$ \frac{dP(k, \mu)}{db_g}$",
        r"$ \frac{dP(k, \mu)}{dr_g}$",r"$ \frac{dP(k, \mu)}{d\sigma_{u}}$" , r"$ \frac{dP(k, \mu)}{dA_s}$",
        r"$ \frac{dP(k, \mu)}{dN_{eff}}$",'', '', r'$ \frac{dP(k, \mu)}{dn_s} $'
                  ]

        xlabel = r"$k (\mathrm{h Mpc})^{-1}$"
        ylabel = ylabels[o-1]

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


        plt.suptitle("Derivative of the %s power spectrum (%s) w.r.t. variation of %s at z = %.3f " % (power_spectrum_chosen[P-1], neutrino_hier, 
        parameters2bevaried[o-1], z) )
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        plt.show()

        # ----------------------------------------------------------------------------------------

    elif plot2 == True:

        ks_real_arr /= (H0_central/100.0) # put ks into units of k/h MPc
        results_findiff_derivatives /= (H0_central/100.0)**3 # putting into units of (MPC/h)^3/x
        results_findiff_semi_analytic_derivatives /= (H0_central/100.0)**3 # putting into units of (MPC/h)^3/x, x being the units of the parameter
        # that is being varied 
        ks_fiducial /= (H0_central/100.0) # put ks into units of k/h MPC

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



        # loop through derivative results using 1st method (just finite difference and plot)
        for j in np.arange(int((num_steps-1)/2)):

            if o == 1:
                label = (r"$\Delta H_0 = %.2f $," %  (deltas[j]) + " M1"  ) 
            if o == 2:
                label = (r"$\Delta \sum m_{\nu} = %.6f $," %  (deltas[j]) + " M1" ) 
            if o == 3:
                label = (r"$\Delta \Omega_bh^2 = %.5f $," %  (deltas[j]) + " M1" ) 
            if o == 4:
                label = (r"$\Delta \Omega_{cdm}h^2 = %.3f $," %  (deltas[j]) + " M1" ) 
            if o == 5:
                label = (r"$\Delta \sigma_g = %.3f $," %  (deltas[j]) + " M1" ) 
            if o == 6:
                label = (r"$\Delta b_g=%.3f $," %  (deltas[j]) + " M1" ) 
            if o == 7:
                label = (r"$\Delta r_g=%.3f $," %  (deltas[j]) + " M1" ) 
            if o == 8:
                label = (r"$\Delta \sigma_{u}=%.3f $," %  (deltas[j]) + " M1" )
            if o == 9:
                label = (r"$\Delta A_s=%.4f \times 10^{-9} $," %  (deltas[j]*1e9) + " M1" ) 
            if o == 10:
                label = (r"$\Delta N_{eff}=3.046+%.4f $" %  (deltas[j]) ) 
            if o == 11:
                label = (r'$ \Delta n_s=%.4f $' % (deltas[j]) )


            dP_dtheta = results_findiff_derivatives[j, mu_indices[0], :]
            
            #ax.semilogx(ks_fiducial, abs(dP_dtheta-dP_dtheta0_mu1)/dP_dtheta0_mu1, label = label+r", $ \mu = %.2f $" % mus[mu_indices[0]], color = colors[count])
            ax.semilogx(ks_fiducial, abs(dP_dtheta-dP_dtheta0_mu1)/dP_dtheta0_mu1, label = label, color = colors[count])
            #ax1.semilogx(ks_fiducial, dP_dtheta, label = label+r", $ \mu = %.2f $" % mus[mu_indices[0]], color = colors[count] )
            ax1.semilogx(ks_fiducial, dP_dtheta, label = label, color = colors[count] )

            count += 1


            dP_dtheta = results_findiff_derivatives[j, mu_indices[1], :]

            #ax.semilogx(ks_fiducial, abs(dP_dtheta-dP_dtheta0_mu2)/dP_dtheta0_mu2, linestyle = '-.', label = label+r", $ \mu = %.2f $" % mus[mu_indices[1]], color = colors[count] )
            ax.semilogx(ks_fiducial, abs(dP_dtheta-dP_dtheta0_mu2)/dP_dtheta0_mu2, linestyle = '-.', label = label, color = colors1[count1] )
            #ax1.semilogx(ks_fiducial, dP_dtheta, linestyle = '-.', label = label+r", $ \mu = %.2f $" % mus[mu_indices[1]], color = colors[count] )
            ax1.semilogx(ks_fiducial, dP_dtheta, linestyle = '-.', label = label, color = colors1[count1] )
            count1 += 1



        # loop through derivative results using 2nd method (finite diff. + analytic derivatives)
        for j in np.arange(int((num_steps-1)/2)):

            if o == 1:
                label = (r"$\Delta H_0 = %.2f $," %  (deltas[j]) + " M2"  ) 
            if o == 2:
                label = (r"$\Delta \sum m_{\nu}  = %.6f $," %  (deltas[j]) + " M2" ) 
            if o == 3:
                label = (r"$\Delta \Omega_bh^2 = %.5f $," %  ( deltas[j]) + " M2") 
            if o == 4:
                label = (r"$\Delta \Omega_{cdm}h^2 = %.3f $," %  (deltas[j]) + " M2" ) 
            if o == 5:
                label = (r"$\Delta \sigma_g = %.3f $," %  (deltas[j]) + " M2" ) 
            if o == 6:
                label = (r"$\Delta b_g=%.3f $," %  (deltas[j]) + " M2"  ) 
            if o == 7:
                label = (r"$\Delta r_g=%.3f $," %  (deltas[j]) + " M2" ) 
            if o == 8:
                label = (r"$\Delta \sigma_{u}=%.3f $," %  (deltas[j]) + " M2" ) 
            if o == 9:
                label = (r"$\Delta A_s=%.4f \times 10^{-9} $," %  (deltas[j]*1e9) + " M2" ) 
            if o == 10:
                label = (r"$\Delta N_{eff}=3.046+%.4f $" %  (deltas[j]) ) 
            if o == 11:
                label = (r'$ \Delta n_s=%.4f $' % (deltas[j]) )

            dP_dtheta = results_findiff_semi_analytic_derivatives[j, mu_indices[0], :]
                
            #ax.semilogx(ks_fiducial, abs(dP_dtheta-dP_dtheta0_mu1)/dP_dtheta0_mu1, label = label+r", $ \mu = %.2f $" % mus[mu_indices[0]], color = colors[count])
            ax.semilogx(ks_fiducial, abs(dP_dtheta-dP_dtheta0_mu1)/dP_dtheta0_mu1, label = label, color = colors[count])
            #ax1.semilogx(ks_fiducial, dP_dtheta, label = label+r", $ \mu = %.2f $" % mus[mu_indices[0]], color = colors[count] )
            ax1.semilogx(ks_fiducial, dP_dtheta, label = label, color = colors[count] )
            count += 1


            dP_dtheta = results_findiff_semi_analytic_derivatives[j, mu_indices[1], :]

            #ax.semilogx(ks_fiducial, abs(dP_dtheta-dP_dtheta0_mu2)/dP_dtheta0_mu2, linestyle = '-.', label = label+r", $ \mu = %.2f $" % mus[mu_indices[1]], color = colors[count])
            ax.semilogx(ks_fiducial, abs(dP_dtheta-dP_dtheta0_mu2)/dP_dtheta0_mu2, linestyle = '-.', label = label, color = colors1[count1])
            #ax1.semilogx(ks_fiducial, dP_dtheta, linestyle = '-.', label = label+r", $ \mu = %.2f $" % mus[mu_indices[1]], color = colors[count] )
            ax1.semilogx(ks_fiducial, dP_dtheta, linestyle = '-.', label = label, color = colors1[count1] )
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
        ylabel = ylabels[o-1]
        ylabelax1 = ylabelsax1[o-1]


        ax.set_xlim([np.min(ks_fiducial), np.max(ks_fiducial)])
        ax.set_xlabel(xlabel, fontsize = 17)
        ax.set_ylabel(ylabel, fontsize = 17)
        ax.tick_params(axis='both', which='major', labelsize=17)
        ax.tick_params(axis='both', which='minor', labelsize=17)
        
        ax1.legend(loc = 'upper right', fontsize = 12, ncol = 1)#, bbox_to_anchor = (-1.5, 0.5))
        #ax1.axhline(0.0, linestyle = '--', color = 'k')

        ax1.set_xlim([np.min(ks_fiducial), np.max(ks_fiducial)])
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

        plt.suptitle("Derivative of the %s power spectrum (%s) w.r.t. variation of %s at z = %.3f " % (power_spectrum_chosen[P-1], neutrino_hier, 
        parameters2bevaried[o-1], z) )
        #plt.tight_layout()
        plt.show()



        # ----------------------------------------------------------------------------------------

    else: # write the derivatives results to a file

        spectra = ['GG', 'GU', 'UU']
        parameters = ['H0', 'mnu', 'ombh2', 'omch2', 'sigg', 'b', 'r', 'sigmau', 'As', 'Neff', 'ns' ]

        if o != 5 or o != 8 or o != 9:

        # numerical derivatives 

            for i in range(int((num_steps-1)/2)):
                data = pd.DataFrame(results_findiff_derivatives[i,:,:])
                data.to_csv(r'Derivatives_RSPSpectra_M1_numerical_%s_%s_%s_Delta=%.6f_z=%.3f.csv' % (spectra[P-1], 
                parameters[o-1], lin_or_nonlin, deltas[i], z), 
                float_format='%.3f', na_rep="NAN!")

        # analytical / semi-analytical/numerical derivatives

            if o != 10:
                for i in range(int((num_steps-1)/2)):
                    data = pd.DataFrame(results_findiff_semi_analytic_derivatives[i,:,:])
                    data.to_csv(r'Derivatives_RSPSpectra_M2_semianalytical_%s_%s_%s_Delta=%.6f_z=%.3f.csv' % (spectra[P-1], 
                    parameters[o-1], lin_or_nonlin, deltas[i], z), 
                    float_format='%.3f', na_rep="NAN!")

        # power spectrum with varying parameter

            for i in range(num_steps):
                data = pd.DataFrame(results_redshiftspacepowerspectrum_arr[i,:,:])
                data.to_csv(r'PowerSpectra_%s_%s_%s_value=%.6f_z=%.3f.csv' % (spectra[P-1], 
                parameters[o-1], lin_or_nonlin, param_variation[i], z), 
                float_format='%.3f', na_rep="NAN!")

        elif o == 9:

        # numerical derivatives

            for i in range(int((num_steps-1)/2)):
                data = pd.DataFrame(results_findiff_derivatives[i,:,:])
                data.to_csv(r'Derivatives_RSPSpectra_M1_numerical_%s_%s_%s_Delta=%.6f_z=%.3f.csv' % (spectra[P-1], 
                parameters[o-1], lin_or_nonlin, deltas[i]*1e9, z), 
                float_format='%.3f', na_rep="NAN!")

        # analytical / semi-analytical/numerical derivatives


            for i in range(int((num_steps-1)/2)):
                data = pd.DataFrame(results_findiff_semi_analytic_derivatives[i,:,:])
                data.to_csv(r'Derivatives_RSPSpectra_M2_semianalytical_%s_%s_%s_Delta=%.6f_z=%.3f.csv' % (spectra[P-1], 
                parameters[o-1], lin_or_nonlin, deltas[i]*1e9, z), 
                float_format='%.3f', na_rep="NAN!")

        # power spectrum with varying parameter

            for i in range(num_steps):
                data = pd.DataFrame(results_redshiftspacepowerspectrum_arr[i,:,:])
                data.to_csv(r'PowerSpectra_%s_%s_%s_value=%.6f_z=%.3f.csv' % (spectra[P-1], 
                parameters[o-1], lin_or_nonlin, param_variation[i]*1e9, z), 
                float_format='%.3f', na_rep="NAN!")

        else:

        # numerical derivatives 

            for i in range(int((num_steps-1)/2)):
                data = pd.DataFrame(results_findiff_derivatives[i,:,:])
                data.to_csv(r'Derivatives_RSPSpectra_M1_numerical_%s_%s_%s_Delta=%.6f_z=%.3f.csv' % (spectra[P-1], 
                parameters[o-1], lin_or_nonlin, deltas[i](H0_central/100), z), 
                float_format='%.3f', na_rep="NAN!")

        # analytical / semi-analytical/numerical derivatives


            for i in range(int((num_steps-1)/2)):
                data = pd.DataFrame(results_findiff_semi_analytic_derivatives[i,:,:])
                data.to_csv(r'Derivatives_RSPSpectra_M2_semianalytical_%s_%s_%s_Delta=%.6f_z=%.3f.csv' % (spectra[P-1], 
                parameters[o-1], lin_or_nonlin, deltas[i](H0_central/100), z), 
                float_format='%.3f', na_rep="NAN!")

        # power spectrum with varying parameter

            for i in range(num_steps):
                data = pd.DataFrame(results_redshiftspacepowerspectrum_arr[i,:,:])
                data.to_csv(r'PowerSpectra_%s_%s_%s_value=%.6f_z=%.3f.csv' % (spectra[P-1], 
                parameters[o-1], lin_or_nonlin, param_variation[i]/(H0_central/100), z), 
                float_format='%.3f', na_rep="NAN!")

    # -----------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------

    return 0



# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------

# compute the growth rate f = dln(sqrt(P))/dln(a) for an arbitrary number of redshifts, ks and mus 
def compute_f_at_many_redshifts(omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t, neutrino_hierarchy, 
    zed_list, kmin_fid, kmax_fid, mu_s_fid, knum, d_a, linear, sum_masses_central_t,
    omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid,  
    kspaceoption, tau_t, ns_t, N_eff_deviation, delta_mnu_max = 0.0, dm2_atm = 2.5e-3, dm2_sol = 7.6e-5):

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
    m1, m2, m3 = get_masses(m_nu_t, neutrino_hierarchy, delta_mnu_max, sum_masses_central_t, dm2_atm, dm2_sol)
    
    mass_input_string = str(m1)+','+str(m2)+','+str(m3) 
    if (m1 < 0 or m2 < 0 or m3 < 0): # we have a problem
        print(m1 + m2 + m3)
        print('Neutrino masses are set to: ', str(m1) + '+ ' + '(0,'+str(m2-m1)+','+str(m3-m1) + ')' )
        raise Exception('Neutrino masses are unphysical, sum of masses is most likely too small for set mass eigenstate differences from neutrino oscillations.')

    
    # set up cosmology
    M.set({"omega_b": omega_b_t, "omega_cdm": omega_cdm_t, "H0": H0_t, "A_s": As_t, "N_ur": 0.00641+N_eff_deviation, 
    "N_ncdm": 3.0, 
    "m_ncdm": mass_input_string, "tau_reio": tau_t, "n_s": ns_t})
    M.set({"output": "mPk", "P_k_max_1/Mpc": 3.0, "z_max_pk": 2.0})

    M.compute() 
    
    # ---------------------------------------------------------------------------------------

    # calculate the growth rate numerically for the list of redshifts zed_list we want 
    growth_rates_arr = np.zeros((len(mu_s_fid), knum, len(zed_list)))

    ks_real_array = np.zeros((len(mu_s_fid), knum, len(zed_list)))

    mu_real_array = np.zeros((len(mu_s_fid), len(zed_list)))

    # ---------------------------------------------------------------------------------------

    # setting up the values of k that we need 
    ks_fiducial = [] 
    if kspaceoption == 'log':
        ks_fiducial = np.logspace(np.log10(kmin_fid), np.log10(kmax_fid), knum, base=10) # 1/ MPC units
    else:
        ks_fiducial = np.linspace(kmin_fid, kmax_fid, knum) # 1/MPC units 

    # ---------------------------------------------------------------------------------------



    if (omega_b_t, omega_cdm_t, H0_t, As_t, m_nu_t) == (omega_b_fid, omega_cdm_fid, H0_fid, As_fid, m_nu_fid):

        # now get the power spectrum at all the redshifts needed to compute the growth rate
        for zed in np.arange(len(zed_list)): # looping through main outer redshift array we want 
            
            Pk_list_at_zed = np.zeros((knum, len(all_redshifts_list[zed])))
            for zz in np.arange(len(all_redshifts_list[zed])): # looping through inner zed list

                Pk_list_at_zed[:, zz] = np.array([M.pk_cb_lin(ki, all_redshifts_list[zed][zz]) for ki in ks_fiducial]) # (MPC ^3)

            growthrate = 0
            scalefactors = 1.0/(1.0 + all_redshifts_list[zed]) # convert zz redshifts (inner redshifts for this iteration of the loop!)
            # back to scalefactors we want 
            da = abs(scalefactors[1] - scalefactors[0]) # calculating the step size in scalefactor da 
            if zed_list[zed] == 0: # use backwards difference method about z = 0 (finite diff.)
                growthrate = 0.5*(scalefactors[0]/(Pk_list_at_zed[:,0]))*(Pk_list_at_zed[:,0] - Pk_list_at_zed[:,1])/(da)
            else: # use the central difference method (finite diff.)
                growthrate = 0.5*(scalefactors[1]/(Pk_list_at_zed[:,1]))*(Pk_list_at_zed[:,0] - Pk_list_at_zed[:,2])/(2*da)

            growth_rates_arr[:,:,zed] = growthrate

        return growth_rates_arr, ks_fiducial, mu_s_fid

    else:
        # compute ks real and mus real for each z in zed_list

        for zz in np.arange(len(zed_list)):

            mu_real_array[:, zz] = get_mus_real(mu_s_fid, As_t, omega_b_t, omega_cdm_t, H0_t, m_nu_t,
            As_fid, omega_b_fid, omega_cdm_fid, H0_fid, m_nu_fid, zed_list[zz])

            for muu in np.arange(len(mu_s_fid)):

                ks_real_array[muu, :, zz] = get_ks_real(ks_fiducial, As_t, omega_b_t, omega_cdm_t, H0_t,
                m_nu_t, As_fid, omega_b_fid, omega_cdm_fid, H0_fid, m_nu_fid, zed_list[zz], mu_s_fid[muu])
    
    
        # now get the power spectrum at all the redshifts needed to compute the growth rate
        for zed in np.arange(len(zed_list)): # looping through main outer redshift array we want 

            Pk_list_at_zed = np.zeros((len(mu_s_fid), knum, len(all_redshifts_list[zed])))
            for zz in np.arange(len(all_redshifts_list[zed])): # looping through inner zed list

                for muu in np.arange(len(mu_s_fid)): # looping through mu

                    # compute ks real!
                    ks_real_at_zz_mu = get_ks_real(ks_fiducial, As_t, omega_b_t, omega_cdm_t, H0_t, m_nu_t, As_fid, omega_b_fid, omega_cdm_fid,
                    H0_fid, m_nu_fid, all_redshifts_list[zed][zz], mu_s_fid[muu])
                    

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
        
    
        return growth_rates_arr, ks_real_array, mu_real_array




# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------


# function to get the derivative of the RSP power spectrum, 
# at many ks, mus and zs, 
# for a single parameter,
# using a single simple 2nd order finite difference , quickly
def get_rsp_dP_dx_cosmo_many_redshifts(o, zed_list, central_params, delta_param, kmin_fid, kmax_fid, num_k_points, 
mu_s_fid, d_a, neutrino_hier, lin_or_nonlin, kspaceoption, tau, ns, N_eff_deviation, dm2_atm = 2.5e-3, dm2_sol = 7.6e-5):

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
    ks_fiducial = [] # initializing an object to store fiducial values of k
    if kspaceoption == 'log':
        ks_fiducial = np.logspace(np.log10(kmin_fid), np.log10(kmax_fid), num_k_points, base=10) # k values (k in 1/MPC)
    else:
        ks_fiducial = np.linspace(kmin_fid, kmax_fid, num_k_points)


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
    
    if o == 1: # varying H0
        param_variation = np.linspace(H0_central-delta_param, H0_central+delta_param, 2)
    elif o == 2: # varying sum of neutrino masses
        param_variation = np.linspace(mnu_central-delta_param, mnu_central+delta_param, 2)
    elif o == 3: # varying omega_b
        param_variation = np.linspace(Obh_central-delta_param, Obh_central+delta_param, 2)
    elif o == 4: # varying omega_cdm
        param_variation = np.linspace(Och_central-delta_param, Och_central+delta_param, 2)
    elif o == 9: # As
        param_variation = np.linspace(As_central-delta_param, As_central+delta_param, 2)
    elif o == 10: # N_eff
        param_variation = np.linspace(-delta_param, delta_param, 2)
    elif o == 11: # ns
        param_variation = np.linspace(ns-delta_param, ns+delta_param, 2)
    else:
        raise Exception('Parameter chosen is not an option. (get_rsp_dP_dx_cosmo_many_redshifts()) ')

    # set up arrays to store power spectra and derivatives in 

    matter_power_spectra_arr = np.zeros((2, len(mu_s_fid), num_k_points, len(zed_list)))


    growth_rate_arr = np.zeros((2, len(mu_s_fid), num_k_points, len(zed_list)))


    ks_real_arr =  np.zeros((2, len(mu_s_fid), num_k_points, len(zed_list)))


    mus_real_arr = np.zeros((2, len(mu_s_fid), len(zed_list)))


    redshift_space_power_spectra_gg =  np.zeros((2, len(mu_s_fid), num_k_points, len(zed_list)))

    redshift_space_power_spectra_gu =  np.zeros((2, len(mu_s_fid), num_k_points, len(zed_list)))

    redshift_space_power_spectra_uu =  np.zeros((2, len(mu_s_fid), num_k_points, len(zed_list)))


    power_spectra_derivatives_gg = np.zeros((len(mu_s_fid), num_k_points, len(zed_list)))

    power_spectra_derivatives_gu = np.zeros((len(mu_s_fid), num_k_points, len(zed_list)))

    power_spectra_derivatives_uu = np.zeros((len(mu_s_fid), num_k_points, len(zed_list)))


    # calculate power spectrum while varying parameter of choice
    for i in np.arange(2): 
        # first just get delta-delta/ delta-theta /theta-theta power spectrum

        # changing parameter that is meant to be varied
        if o == 1:
            H0 = param_variation[i]
            h = param_variation[i]/100.0
            sigmag = sigmagh/h
            sigmau = sigmauh/h
        elif o == 2:
            mnu = param_variation[i]
            delta_mnu_max = delta_param
        elif o == 3:
            Obh = param_variation[i]
        elif o == 4:
            Och = param_variation[i]
        elif o == 9:
            As = param_variation[i]
        elif o == 10:
            N_eff_deviation = param_variation[i]
        elif o == 11: # ns 
            ns_val = param_variation[i]
        # get omega_m
        omegam = (Obh + Och + (mnu/93.14))/(h**2)


        Pk_cb, ks_r, mus_r = run_class(Obh, Och, H0, As, mnu, neutrino_hier, zed_list, kmin_fid, 
        kmax_fid, num_k_points, dm2_atm, dm2_sol, delta_mnu_max, mnu_central, kspaceoption, tau, ns_val, 
        Obh_central, Och_central, H0_central, As_central, mnu_central, mu_s_fid, lin_or_nonlin, N_eff_deviation)
            

        f = compute_f_at_many_redshifts(Obh, Och, H0, As, mnu, neutrino_hier, zed_list, kmin_fid, kmax_fid, mu_s_fid, 
        num_k_points, d_a, linear, mnu_central, Obh_central, Och_central, H0_central, As_central, mnu_central, 
        kspaceoption, tau, ns_val, N_eff_deviation, delta_mnu_max=delta_mnu_max)[0]
            

        if Pk_cb.shape == (len(mu_s_fid), num_k_points, len(zed_list)) and ks_r.shape == (len(mu_s_fid), num_k_points, len(zed_list)) and mus_r.shape == (len(mu_s_fid), len(zed_list)) and f.shape == (len(mu_s_fid), num_k_points, len(zed_list)):

            matter_power_spectra_arr[i,:,:,:] = Pk_cb
            ks_real_arr[i,:,:,:] = ks_r
            mus_real_arr[i,:,:] = mus_r
            growth_rate_arr[i,:,:,:] = f


        elif Pk_cb.shape == (num_k_points, len(zed_list)) and ks_r.shape == (num_k_points,) and mus_r.shape == (len(mu_s_fid),) and f.shape == (len(mu_s_fid), num_k_points, len(zed_list)):

            for muu in np.arange(len(mu_s_fid)):
                matter_power_spectra_arr[i,muu,:,:] = Pk_cb
                for zz in np.arange(len(zed_list)):
                    ks_real_arr[i,muu,:,zz] = ks_fiducial
                        
            for zz in np.arange(len(zed_list)):
                mus_real_arr[i,:,zz] = mu_s_fid

            growth_rate_arr[i,:,:,:] = f


        else:
            print('shape Pk_cb: ', Pk_cb.shape)
            print('shape ks: ', ks_r.shape)
            print('shape mus:', mus_r.shape)
            print('shape f: ', f.shape)
            raise Exception(('Shape of Pk_cb / k / mu /growth rate array returned from run_class() is unexpected and cannot be adapted ' + 
            'to store in matter_power_spectrum_arr / ks_real_arr / mus_real_arr / growth_rates_arr (derivative_power_spectra()).'))


        

        

        for zz in np.arange(len(zed_list)): # looping through redshift array

            q_para = distortion_parallel(As, Obh, Och, H0, mnu, As_central, Obh_central, Och_central, H0_central, mnu_central, zed_list[zz])
            q_perp = distortion_perpendicular(As, Obh, Och, H0, mnu, As_central, Obh_central, Och_central, H0_central, mnu_central, zed_list[zz])

            for muu in np.arange(len(mu_s_fid)): # looping through muuuuuu

                
                    
                redshift_space_power_spectra_gg[i, muu, :, zz] = gg_redshift_s_at_k_real(b_g[zz], r_g, growth_rate_arr[i, muu, :, zz],
                sigmag, mus_real_arr[i, muu, zz], ks_real_arr[i, muu, :, zz], zed_list[zz], H0, q_para, 
                q_perp)*matter_power_spectra_arr[i,muu,:,zz]

                    
                
                redshift_space_power_spectra_gu[i, muu, :, zz] = gu_redshift_s_at_k_real(b_g[zz], r_g, growth_rate_arr[i, muu, :, zz],
                sigmag, sigmau, mus_real_arr[i, muu, zz], ks_real_arr[i, muu, :, zz], zed_list[zz], H0, omegam, 
                1.0-omegam, q_para, q_perp)*matter_power_spectra_arr[i,muu,:,zz]

                    

                redshift_space_power_spectra_uu[i, muu, :, zz] = uu_redshift_s_at_k_real(sigmau, mus_real_arr[i, muu,zz], 
                growth_rate_arr[i, muu, :, zz], ks_real_arr[i, muu, :, zz], zed_list[zz], H0, omegam, 
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


    return power_spectra_derivatives_gg, power_spectra_derivatives_gu, power_spectra_derivatives_uu, ks_real_arr, mus_real_arr      
        
    



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
    derivative_power_spectra(11, 1, 1.0, 'plot2', params, 0.01, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # mnu
    #derivative_power_spectra(2, 1, 0.0, 'plot2', params, 0.0025, 1.0e-4, 1.0, 2000, 100, 0.0001, 'inverted', 'linear', 'log', tau, ns)
    # Obh   
    #derivative_power_spectra(3, 1, 0.0, 'plot2', params, 0.00012, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # Och
    #derivative_power_spectra(4, 1, 0.0, 'plot2', params, 0.006, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    

    # other parameters
    # sigmag
    #derivative_power_spectra(5, 1, 0.5, 'plot1', params, 0.1, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # bg
    #derivative_power_spectra(6, 2, 1.0, 'plot1', params, 0.1, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # rg
    #derivative_power_spectra(7, 1, 0.0, 'plot1', params, 0.1, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # sigmau
    #derivative_power_spectra(8, 2, 0.0, 'plot1', params, 0.1, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # As
    #derivative_power_spectra(9, 1, 0.0, 'plot2', params, 0.1*1e-10, 1.0e-4, 1.0, 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)
    # N_eff
    #derivative_power_spectra(10, 3, 0.0, 'plot2', params, 0.001, 1e-4, 0.2*(0.6732), 2000, 100, 0.0001, 'normal', 'linear', 'log', tau, ns)

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

