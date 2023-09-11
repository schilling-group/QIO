import numpy as np
from qicas import prep_rdm12
from old_code.tools import gamma_Gamma_prep
from orb_rot import jacobi_cost_full

def prep_rand_rdm12(no=5):
    dm1 = np.random.rand(no,no)
    dm1 = (dm1 + dm1.T)/2
    dm2 = np.random.rand(no,no,no,no)
    dm2 = (dm2 + dm2.transpose(0,1,3,2) + dm2.transpose(1,0,2,3) + dm2.transpose(1,0,3,2))/4
    return dm1, dm2

def test_prep_rdm12():
    # prepare random symmetric 1- and 2-RDM
    # prepare 1- and 2-RDM in spin-orbital indices
    dm1, dm2 = prep_rand_rdm12(no=5)
    rdm1, rdm2 = prep_rdm12(dm1, dm2)
    
    # use old function to prepare 1- and 2-RDM in spin-orbital indices
    gamma, Gamma = gamma_Gamma_prep(dm1, dm2)

    # check if the 1-RDM is correct
    assert np.allclose(rdm1, gamma)
    # check if the 2-RDM is correct
    assert np.allclose(rdm2, Gamma)

def test_jacobi_cost_full():
    inactive_indices = [0,1]
    inds = np.array(inactive_indices)
    dm1, dm2 = prep_rand_rdm12(no=5)
    rdm1, rdm2 = prep_rdm12(dm1, dm2)
    cost_old = jacobi_cost_full(rdm1, rdm2, inactive_indices)

    def jacobi_cost_full_new(gamma, Gamma, inds):
        
        nu = gamma[2*inds, 2*inds]
        nd = gamma[2*inds+1, 2*inds+1]
        nn = Gamma[inds, inds, inds, inds]
        spec = np.array([1-nu-nd+nn, nu-nd, nd-nn, nn])
        spec = np.clip(spec, a_min=1e-15, a_max=None)
        cost_fun = -np.sum(spec * np.log(spec), axis=0)
        return np.sum(cost_fun)


    cost_new = jacobi_cost_full_new(rdm1, rdm2, inds)

    assert np.allclose(cost_old, cost_new)

    



