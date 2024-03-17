import numpy as np
from warnings import warn

def shannon(spec):
    '''
    Shannon entropy of a probability distribution

    Args:
        spec (ndarray): probability distribution
    
    Returns:
        S (float): Shannon entropy of spec
    '''
    spec = np.asarray(spec)
    if np.any(spec < 0):
        if np.any(np.abs(spec[spec < 0]) > 1e-6):
            warn("Warning: spec has negative entries!")
    elif np.any(spec > 1):
        print(spec)
        raise ValueError("spec has entries larger than 1")
    spec = spec[spec > 0]
    return -np.sum(spec * np.log(spec)) 

def get_cost_fqi(gamma, Gamma, inactive_indices):

    '''
    Sum of all inactive orbital entropy

    Args:
        gamma (ndarray): current 1RDM
        Gamma (ndarray): current 2RDM
        inactive_indices (list): indices of inactive orbitals
    
    Returns:
        cost_fun (float): S(rho_i) for all i in inactive_indices

    '''


    inds = np.asarray(inactive_indices) 
    nu = gamma[2*inds, 2*inds]
    nd = gamma[2*inds+1, 2*inds+1]
    nn = Gamma[inds, inds, inds, inds]
    spec = np.array([1-nu-nd+nn, nu-nn, nd-nn, nn])
    cost_fun = shannon(spec)
    
    return np.sum(cost_fun)