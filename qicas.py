from functools import reduce
import warnings
import logging

from pyscf import lib, dmrgscf, mcscf, cc
import os
import numpy as np
from solver.jacobi import minimize_orb_corr_jacobi, reorder, reorder_fast, reorder_occ
from solver.gradient import minimize_orb_corr_GD
from tccsd import make_tailored_ccsd


class QICAS:
    def __init__(self, mf, mc, act_space=None):
        """
        Attributes:
            mf (pyscf.scf object): mean-field object
            mc (pyscf.mcscf object): mcscf object
            act_space (tuple): tuple of (n_cas, n_act_e), where n_cas is the number of active orbitals
                and n_act_e is the number of active electrons
            no (int): number of non-frozen orbitals
            max_cycle (int): max number of cycle of jacobi rotations
            max_M (int): max bond dimension in DMRG
        
        Saved results:
            mo_coeff (ndarray): optimized orbitals in spatial-orbital indices
            gamma (ndarray): optimized 1-RDM in spatial-orbital indices
            Gamma (ndarray): optimized 2-RDM in spatial-orbital indices
        """
        self.mf = mf
        self.no = len(mf.mo_coeff)
        self.n_cas, self.n_act_e = act_space
        self.n_core = self.mf.mol.nelectron // 2 - self.n_cas
        self.mc = mc
        self.max_cycle = 100
        self.thresh = 1e-4
        self.step_size = 1.
        # max bond dimension in DMRG
        self.max_M = 200
        self.tcc_e_tot = None

        # settings for tcc solver
        self.tcc_max_cycle = 100
        self.tcc_level_shift = 0.

        #
        self.verbose = mf.verbose

        # logging setup
        self.logger = logging.getLogger('info')
        self.logger.setLevel(logging.INFO)
        # Create a file handler
        file_handler = logging.FileHandler('qicas.log')
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

        # debug flag
        self.debug = False
    
    def dump_flags(self):
        print("="*35+"QICAS FLAGS"+"="*35)
        self.logger.info("max_cycle = %d", self.max_cycle)
        self.logger.info("thresh = %e", self.thresh)
        self.logger.info("step_size = %f", self.step_size)
        self.logger.info("max_M = %d", self.max_M)
        self.logger.info("tcc_e_tot = %s", str(self.tcc_e_tot))

    def kernel(self, inactive_indices, mo_coeff, is_tcc=False, method='newton-raphson'):

        '''
        Performs QICAS procedure:
            1. run DMRG on all orbitals to be optimized
            2. run orbital optimization
            3. run CASCI on optimized active space

        Args:
            active_indices (list): indices of active orbitals
            inactive_indices (list): indices of inactive orbitals
            no (int): number of non-frozen orbitals
            max_cycle (int): max number of cycle of jacobi rotations
            is_tcc (bool): whether to use tailored CCSD solver to obtain rdm1 and rdm2
            method (str): method of orbital optimization, currently only '2d_jacobi' is supported
                in the future we will add 'gradient_descent'

        Returns:
            etot (float): post-QICAS CASCI energy
            n_closed (int): number of closed orbitals predicted by QICAS

        '''
        if mo_coeff is None:
            mo_coeff = self.mf.mo_coeff

        if is_tcc:
            # using TCCSD to get initial RDM1 and RDM2
            tcc = cc.CCSD(self.mf, mo_coeff=mo_coeff.copy())
            mc = mcscf.CASCI(self.mf, self.n_cas, self.n_act_e)
            #mc = fci_prep(mc=mc, mol=self.mf.mol, maxM=self.max_M, tol=1e-5)
            mc.verbose = self.verbose
            mc.canonicalization = True 
            mc.sorting_mo_energy = True
            mc.fix_spin_(ss=0)
            mc.natorb = True
            mc.kernel(mo_coeff.copy())

            self.logger.info('CASCI energy = %.6f',mc.e_tot)
            tcc, t1, t2 = make_tailored_ccsd(tcc, mc)
            tcc.verbose = self.verbose
            tcc.max_cycle = self.tcc_max_cycle
            tcc.level_shift = self.tcc_level_shift
            self.logger.info('mo energy = %s', str(tcc._scf.mo_energy))
            tcc.kernel(t1=t1, t2=t2)
            self.tcc_e_tot = tcc.e_tot
            self.logger.info('TCCSD energy = %.6f',tcc.e_tot)
            dm1 = tcc.make_rdm1()
            dm2 = tcc.make_rdm2()
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

        else:
            # DMRG and RDMs prep block
            mc = mcscf.CASCI(self.mf, self.no, self.mf.mol.nelectron)
            mc = dmrgci_prep(mc=mc, mol=self.mf.mol, maxM=self.max_M, tol=1e-5)
            edmrg = mc.kernel(mo_coeff)[0]
            self.logger.info('DMRG energy = %6.f',edmrg)
            # the spin argument requires special modification to the local block2main code
            #dm1, dm2 = mc.fcisolver.make_rdm12(0, self.no, self.mf.mol.nelectron, spin=True) 
            dm1, dm2 = mc.fcisolver.make_rdm12(0, self.no, self.mf.mol.nelectron) 
            if self.debug:
                assert np.isclose(np.sum(dm1.diagonal()), self.mf.mol.nelectron, atol=1e-6)
                assert np.isclose(np.einsum('iijj->', dm2), self.mf.mol.nelectron*(self.mf.mol.nelectron-1), atol=1e-6)

        gamma,Gamma = prep_rdm12(dm1,dm2)


        if method.upper() == '2D_JACOBI' or method.upper() == '2DJACOBI' or method.upper() == 'JACOBI':
            # Orbital rotation block
            _,U,self.gamma,self.Gamma = minimize_orb_corr_jacobi(gamma,Gamma,inactive_indices,self.max_cycle)
        elif method.upper() == 'NEWTON-RAPHSON' or method.upper() == 'NEWTON_RAPHSON' or method.upper() == 'NEWTON' or method.upper() == 'NR':
            U,self.gamma,self.Gamma = minimize_orb_corr_GD(gamma,Gamma,inactive_indices, thresh=self.thresh, max_cycle=self.max_cycle, step_size=self.step_size,
                                                            logger=self.logger)
        else:
            raise NotImplementedError('Only 2d_jacobi and newton-raphson are supported')

        V = reorder_occ(self.gamma.copy(),self.Gamma.copy(),self.n_cas)
        U_ = np.matmul(V,U)
        self.mo_coeff = mo_coeff @ U_.T

        # Post-QICAS CASCI block
        mycas = mcscf.CASCI(self.mf,self.n_cas, self.n_act_e)

        mycas.fix_spin_(ss=0)
        mycas.canonicalization = True
        mycas.natorb = True
        mycas.sorting_mo_energy = True
        mycas.verbose = self.verbose
        etot = mycas.kernel(self.mo_coeff)[0]


        return etot


def dmrgci_prep(mc, mol, maxM, tol=1E-12):

    '''
    Prepare a dmrgci object

    Args:
        mc: a pyscf casci object
        mol: a pyscf mol object
        maxM (int): maximal bond dimension
        stages (int or list): stages specifications [bond_dim, n_sweeps, noise]
        tol: schedueled tolerance for termination of DMRG sweeps

    Returns:
        mc: target casci object 

    '''
    print("Using dmrg from ", dmrgscf.__file__)
    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=maxM, tol=tol)
    mc.fcisolver.runtimeDir = lib.param.TMPDIR
    mc.fcisolver.scratchDirectory = lib.param.TMPDIR
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 4))
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB
    mc.wfnsym='A1g'
    mc.canonicalization = False
    mc.natorb = False
    
    mc.fcisolver.restart = False
    mc.fcisolver.scheduleSweeps =[50] 
    mc.fcisolver.scheduleMaxMs = [maxM] 
    mc.fcisolver.scheduleTols = [1e-6] 
    mc.fcisolver.scheduleNoises = [0] 
    return mc



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







