import numpy as np

def shannon(spec):
    '''
    Shannon entropy of a probability distribution

    Args:
        spec (ndarray): probability distribution
    
    Returns:
        S (float): Shannon entropy of spec
    '''
    # FIXME: can spec be negative? if yes, is it ok to just discard the negative part?
    spec = np.asarray(spec)
    if np.any(spec < 0):
        if np.any(np.abs(spec[spec < 0]) > 1e-6):
            raise ValueError("spec has negative entries")
    elif np.any(spec > 1):
        print(spec)
        raise ValueError("spec has entries larger than 1")
    spec = spec[spec > 0]
    return -np.sum(spec * np.log(spec)) 

def get_cost_fqi(gamma, Gamma, inactive_indices):

    '''
    Sum of all inactive orbita entropy

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
    #spec = np.clip(spec, a_min=1e-15, a_max=None)
    #cost_fun = -np.sum(spec * np.log(spec), axis=0)
    cost_fun = shannon(spec)
    return np.sum(cost_fun)