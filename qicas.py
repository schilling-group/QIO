from functools import reduce
import warnings
import logging

from pyscf import lib, dmrgscf, mcscf, cc, ci, symm

import numpy as np
from solver.jacobi import minimize_orb_corr_jacobi, reorder, reorder_fast, reorder_occ
from solver.gradient import minimize_orb_corr_GD
from tccsd import make_tailored_ccsd, make_no
import os

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''



class QICAS:
    def __init__(self, mf, mc, act_space=None, n_core=0, logger=None):
        """
        Attributes:
            mf (pyscf.scf object): mean-field object
            mc (pyscf.mcscf object): mcscf object
            act_space (tuple): tuple of (n_cas, n_act_e), where n_cas is the number of active orbitals
                and n_act_e is the number of active electrons
            n_core (int): number of core orbitals
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
        self.nocc = int(mf.mol.nelectron//2)
        self.n_cas, self.n_act_e = act_space
        self.n_core = self.nocc - int(self.n_act_e//2)
        self.active_indices = np.array(list(range(self.n_core, self.n_core+self.n_cas)))
        self.mu = 0.
        self.mu_rate = 1.0
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
        self.casci_natorb = True
        self.casci_max_cycle = 300
        self.casci_ss_shift = 2.0
        self.tcc = None
        self.mc = None

        # stored results
        self.mo_coeff = None
        self.s_val = None
        self.occ_num = None
        #
        self.verbose = mf.verbose

        # logging setup
        if logger is None:
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
        else:
            self.logger = logger

        # debug flag
        self.debug = False
    
    def dump_flags(self):
        print("="*35+"QICAS FLAGS"+"="*35)
        self.logger.info("max_cycle = %d", self.max_cycle)
        self.logger.info("thresh = %e", self.thresh)
        self.logger.info("step_size = %f", self.step_size)
        self.logger.info("max_M = %d", self.max_M)
        self.logger.info("n_core = %d", self.n_core)
        self.logger.info("tcc_e_tot = %s", str(self.tcc_e_tot))
        self.logger.info("tcc_max_cycle = %d", self.tcc_max_cycle)
        self.logger.info("tcc_level_shift = %f", self.tcc_level_shift)
        self.logger.info("tcc_casci_natorb = %s", str(self.casci_natorb))


    def kernel(self, inactive_indices, mo_coeff, is_ccsd=False,is_tcc=False, method='newton-raphson'):

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
            self.mf.mo_coeff = mo_coeff.copy()
            mc = mcscf.CASCI(self.mf, self.n_cas, self.n_act_e)
            #mo_coeff = mc.sort_mo(list(range(1,11,1)))
            #mc.frozen = self.n_core
            mc.verbose = self.verbose
            mc.canonicalization = True
            mc.sorting_mo_energy = False 
            mc.fcisolver.max_cycle = self.casci_max_cycle
            mc.fix_spin_(ss=0, shift=self.casci_ss_shift)
            #mc.fix_spin_(ss=0)
            mc.tol = 1e-8
            mc.fcisolver.level_shift = 0.1
            mc.fcisolver.threads = 1
            mc.natorb = self.casci_natorb
            mc.kernel(mo_coeff.copy())
            #mc_rdm1 = mc.fcisolver.make_rdm1()
            #mc_rdm1 = np.einsum('ij,pi,qj->pq', mc_rdm1, mo_coeff, mo_coeff)
            #mc_rdm2 = mc.make_rdm2()
            #mc.mo_coeff, mc.ci, occ = mc.cas_natorb(sort=False)


            
            self.tcc = cc.CCSD(self.mf, mo_coeff=mo_coeff.copy())

            self.logger.info('CASCI energy = %.6f',mc.e_tot)
            self.tcc, t1, t2, is_good_ref = make_tailored_ccsd(self.tcc, mc)
            self.tcc.verbose = self.verbose
            #tcc.max_cycle = self.tcc_max_cycle
            max_cycle = self.max_cycle
            mu0 = self.mu
            if is_good_ref:
                self.tcc.max_cycle = self.tcc_max_cycle
                self.max_cycle = max_cycle 
                #self.mu = mu0
                #self.casci_natorb = False
            else:
                self.tcc.max_cycle = 0
                self.max_cycle = 10
                #self.mu = 0.
                #self.casci_natorb = True
            self.tcc.level_shift = self.tcc_level_shift
            #tcc.diis_start_cycle = 10
            #tcc.direct = True
            #tcc.diis_space = 20
            #self.logger.info('mc mo energy = %s', str(mc.mo_energy))
            self.tcc.kernel()

            self.tcc_e_tot = self.tcc.e_tot
            #if not is_good_ref:
            #    n_cas = self.n_cas
            #    cas_ncore = mc.ncore
            #    nelec_cas = sum(mc.nelecas)
            #    cas_nocc = nelec_cas//2
            #    cas_nvir = n_cas - cas_nocc
            #    nocc = mc.ncore + cas_nocc
            #    tcc.t1[cas_ncore:nocc, :cas_nvir] = t1[cas_ncore:nocc, :cas_nvir]
            #    tcc.t2[cas_ncore:nocc, cas_ncore:nocc, :cas_nvir, :cas_nvir] = t2[cas_ncore:nocc, cas_ncore:nocc, :cas_nvir, :cas_nvir]
            self.logger.info('TCCSD energy = %.6f',self.tcc.e_tot)
            dm1 = self.tcc.make_rdm1()
            dm2 = self.tcc.make_rdm2()

            # calculate the number of electrons in the active space
            n_act_e = np.sum(dm1.diagonal()[self.active_indices])
            self.logger.info("Number of active electrons = %.6f", n_act_e)

            # make NO in inactive space 
            #n_core = (self.mf.mol.nelectron - self.n_act_e)//2
            #inactive_subspace = list(range(n_core)) + list(range(self.n_cas+n_core,self.no))
            #mo_coeff, _ = make_no(dm1, mo_coeff, inactive_subspace)

            #tcc = cc.CCSD(self.mf, mo_coeff=mo_coeff.copy())

            #self.logger.info('CASCI energy = %.6f',mc.e_tot)
            #tcc, t1, t2 = make_tailored_ccsd(tcc, mc)
            #tcc.verbose = self.verbose
            #tcc.max_cycle = self.tcc_max_cycle
            #tcc.level_shift = self.tcc_level_shift
            #self.logger.info('mc mo energy = %s', str(mc.mo_energy))
            #tcc.kernel()
            #self.tcc_e_tot = tcc.e_tot
            #self.logger.info('TCCSD energy = %.6f',tcc.e_tot)
            #dm1 = tcc.make_rdm1()
            #dm2 = tcc.make_rdm2()

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
        elif is_ccsd:
            from pyscf.ci import cisd
            ccsd = cc.CCSD(self.mf, mo_coeff=mo_coeff.copy())
            ccsd.verbose = self.verbose
            ccsd.level_shift = self.tcc_level_shift
            ccsd.max_cycle = self.tcc_max_cycle
            ccsd.kernel()

            def make_rdm1():
                """Make 1-RDM using C1 and C2 from CCSD"""
                t1, t2 = ccsd.t1, ccsd.t2
                c0 = 1.
                c1 = t1.copy()
                c2 = t2 + np.einsum('ia,jb->ijab', t1, t1) 
                mycisd = cisd.CISD(ccsd._scf)
                cisdvec = cisd.amplitudes_to_cisdvec(c0, c1, c2)
                cisdvec /= np.linalg.norm(cisdvec)
                return mycisd.make_rdm1(cisdvec)

            def make_rdm2():
                """Make 2-RDM using C1 and C2 from CCSD"""
                t1, t2 = ccsd.t1, ccsd.t2
                c0 = 1.
                c1 = t1
                c2 = t2 + np.einsum('ia,jb->ijab', t1, t1) 
                mycisd = cisd.CISD(ccsd._scf)
                cisdvec = cisd.amplitudes_to_cisdvec(c0, c1, c2)
                cisdvec /= np.linalg.norm(cisdvec)
                return mycisd.make_rdm2(cisdvec)
            ccsd.make_rdm1 = make_rdm1
            ccsd.make_rdm2 = make_rdm2
            dm1 = ccsd.make_rdm1()
            dm2 = ccsd.make_rdm2()
            self.tcc_e_tot = ccsd.e_tot
        else:
            # DMRG and RDMs prep block
            mc = mcscf.CASCI(self.mf, self.no, self.mf.mol.nelectron)
            mc = dmrgci_prep(mc=mc, mol=self.mf.mol, maxM=self.max_M, tol=1e-5)
            edmrg = mc.kernel(mo_coeff)[0]
            self.logger.info('DMRG energy = %16.8f',edmrg)
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
            U,self.gamma,self.Gamma, self.mu = minimize_orb_corr_GD(gamma,Gamma,inactive_indices, self.active_indices, mu=self.mu, mu_rate=self.mu_rate,
                                                           thresh=self.thresh, max_cycle=self.max_cycle, step_size=self.step_size,
                                                           logger=self.logger)
        else:
            raise NotImplementedError('Only 2d_jacobi and newton-raphson are supported')

        V, self.s_val, self.occ_num = reorder_occ(self.gamma.copy(),self.Gamma.copy(),self.n_cas)
        U_ = np.matmul(V,U)
        self.mo_coeff = mo_coeff @ U_.T
        #self.mu = 0.
        #target_n_ae += (self.n_act_e - n_act_e)/2
        etot = 0.
        #etot = mc.e_tot


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
    mc.fcisolver.runtimeDir = './tmp'
    mc.fcisolver.scratchDirectory = './tmp'
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 8))
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB
    mc.wfnsym='A1g'
    mc.canonicalization = False
    mc.natorb = False
    
    mc.fcisolver.scheduleSweeps = [0, 2]#, 8] #, 12, 16]
    mc.fcisolver.scheduleMaxMs = [250, 250]#, 250] #, 250, 250]
    mc.fcisolver.scheduleTols = [1e-08, 1e-8]#, 1e-8] #, 1e-8, 1e-8]
    mc.fcisolver.scheduleNoises = [0.0001, 0.0001]#, 5e-05] #, 5e-05, 0.0]
    mc.fcisolver.maxIter = 30
    mc.fcisolver.twodot_to_onedot = 20
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




