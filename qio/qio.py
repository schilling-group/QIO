from functools import reduce
import logging

import numpy as np
from qio.grad.jacobi import minimize_orb_corr_jacobi, reorder_occ
from qio.grad.gradient import minimize_orb_corr_GD


class QIO:
    def __init__(self, mf, sol, act_space=None, logger=None, log_file='qio.log'):
        """
        Attributes:
            mf (pyscf.scf object): mean-field object
            sol: solver for obtaining 1- and 2-RDM
            act_space (tuple): tuple of (n_cas, n_act_e), where n_cas is the number of active orbitals
                and n_act_e is the number of active electrons
            n_core (int): number of core orbitals
            no (int): number of non-frozen orbitals
            max_cycle (int): max number of cycle of jacobi rotations
            level_shift (float): level shift in the diagonal hessian for orbital optimization
            max_M (int): max bond dimension in DMRG
        
        Saved results:
            mo_coeff (ndarray): optimized orbitals in spatial-orbital indices
            gamma (ndarray): optimized 1-RDM in spatial-orbital indices
            Gamma (ndarray): optimized 2-RDM in spatial-orbital indices
        """
        self.mf = mf
        self.no = len(mf.mo_coeff)
        self.nocc = int(mf.mol.nelectron//2)
        self.n_cas, self.n_act_e = act_space
        self.n_core = self.nocc - int(self.n_act_e//2)
        self.active_indices = np.array(list(range(self.n_core, self.n_core+self.n_cas)))
        self.sol = sol
        self.max_cycle = 100
        self.thresh = 1e-4
        self.level_shift = 1e-3
        self.step_size = 1.
        # deprecated attributes for maintaining active space electron numbers
        self.mu = 0.
        self.mu_rate = 1.0
        # stored results
        self.mo_coeff = None
        self.s_val = None
        self.occ_num = None
        #
        self.verbose = mf.verbose

        # logging setup
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger('qio')
            # Create a file handler
            file_handler_set = any(type(handler) is logging.FileHandler
                                     for handler in self.logger.handlers)
            if not file_handler_set:
                self.logger.setLevel(logging.INFO)
                file_handler = logging.FileHandler(log_file, mode='w')
                formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            # Create a console handler
            stream_handler_set = any(type(handler) is logging.StreamHandler
                                     for handler in self.logger.handlers)

            if not stream_handler_set:
                # Create and add the StreamHandler to the logger
                stream_handler = logging.StreamHandler()
                self.logger.addHandler(stream_handler)

        

        # debug flag
        self.debug = False
    
    def dump_flags(self):
        print("="*35+"QICAS FLAGS"+"="*35)
        self.logger.info("max_cycle = %d", self.max_cycle)
        self.logger.info("n_core = %d", self.n_core)
        self.logger.info("thresh = %e", self.thresh)
        self.logger.info("level_shift = %e", self.level_shift)
        self.logger.info("step_size = %f", self.step_size)
        self.logger.info("solver = %s", self.sol.__class__.__name__)


    def kernel(self, inactive_indices, mo_coeff, method='newton-raphson'):

        '''
        Performs QIO procedure:
            1. run solver, such as DMRG or TCCSD, on all orbitals to be optimized
            2. run orbital optimization

        Args:
            active_indices (list): indices of active orbitals
            inactive_indices (list): indices of inactive orbitals
            method (str): method of orbital optimization, currently only '2d_jacobi' is supported
                in the future we will add 'gradient_descent'

        Returns:
            mo_coeff (ndarray): optimized orbital coefficients

        '''
        if mo_coeff is None:
            mo_coeff = self.mf.mo_coeff

        self.sol.kernel(mo_coeff.copy())
        dm1 = self.sol.make_rdm1()
        dm2 = self.sol.make_rdm2()

        # calculate the number of electrons in the active space
        n_act_e = np.sum(dm1.diagonal()[self.active_indices])
        self.logger.info("Number of active electrons = %.6f", n_act_e)

        if self.debug:
            assert np.isclose(np.sum(dm1.diagonal()), self.mf.mol.nelectron, atol=1e-6)
            assert np.isclose(np.einsum('iijj->', dm2), self.mf.mol.nelectron*(self.mf.mol.nelectron-1), atol=1e-6)
            dm1_T = dm1.copy().T
            dm2_T = dm2.copy().transpose((1,0,3,2))
            np.allclose(dm1_T, dm1)
            np.allclose(dm2_T, dm2)
            # check if dm1 is positive semidefinite
            occ, _ = np.linalg.eigh(dm1)
            if np.any(occ < 0):
                raise ValueError("Negative eigenvalues in dm1")
            if self.debug:
                assert np.isclose(np.sum(dm1.diagonal()), self.mf.mol.nelectron, atol=1e-6)
                assert np.isclose(np.einsum('iijj->', dm2), self.mf.mol.nelectron*(self.mf.mol.nelectron-1), atol=1e-6)

        gamma,Gamma = prep_rdm12(dm1,dm2)

        
        if method.upper() == '2D_JACOBI' or method.upper() == '2DJACOBI' or method.upper() == 'JACOBI':
            # Orbital rotation block
            _,U,self.gamma,self.Gamma = minimize_orb_corr_jacobi(gamma, Gamma, inactive_indices, self.max_cycle)
        elif method.upper() == 'NEWTON-RAPHSON' or method.upper() == 'NEWTON_RAPHSON' or method.upper() == 'NEWTON' or method.upper() == 'NR':
            U, self.gamma, self.Gamma, self.mu = minimize_orb_corr_GD(gamma, Gamma, inactive_indices, self.active_indices, 
                                                           thresh=self.thresh, max_cycle=self.max_cycle, step_size=self.step_size,
                                                           level_shift=self.level_shift)
        else:
            raise NotImplementedError('Only 2d_jacobi and newton-raphson are supported')

        V, self.s_val, self.occ_num = reorder_occ(self.gamma.copy(),self.Gamma.copy())
        U_ = np.matmul(V,U)
        self.mo_coeff = mo_coeff @ U_.T

        return self.mo_coeff



def prep_rdm12(dm1, dm2):
    '''
    Prepare the 1- and 2-RDM (splitting 1-RDM into spin parts and fix prefactor of 2-RDM)
    This only works for singlet states.
    For other spin states, one should run spin unrestricted DMRG and get the 
    spin 1- and 2-RDMs.

    Args:
        dm1 (ndarray): spatial-orbital 1RDM from pyscf
        dm2 (ndarray): spatial-orbital 2RDM from pyscf

    Returns:
        rdm1(ndarray): prepared 1RDM in spin-orbital indices
        rdm2(ndarray): prepared relevant part of the 2RDM in orbital indices and spin (up,down,down,up)
    '''
    no = len(dm1)
    rdm1 = np.zeros((2*no, 2*no))
    rdm1[::2, ::2] = dm1 / 2
    rdm1[1::2, 1::2] = dm1 / 2
    rdm2 = dm2.transpose((0,2,3,1)).copy()
    rdm2 = (2 * rdm2 + rdm2.transpose((0, 1, 3, 2))) / 6.

    return rdm1,rdm2


def get_overlap(fcivector, cisdvec):
    '''
    Compute overlap between two vectors

    Args:
        fcivector (ndarray): FCI vector
        cisdvec (ndarray): CISD vector

    Returns:
        overlap (float): overlap between fcivector and cisdvec
    '''
    overlap = np.dot(fcivector, cisdvec) / (np.linalg.norm(fcivector) * np.linalg.norm(cisdvec))
    return overlap




