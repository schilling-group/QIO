"""
Module for DMRG solver, including the function to prepare a DMRGCI object
and some other tools.
"""
from pyscf import mcscf, dmrgscf
import os
# DMRG and RDMs prep block

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''

if dmrgscf.settings.BLOCKEXE == '':
    raise FileNotFoundError("block2main not found")

mc = mcscf.CASCI(self.mf, self.no, self.mf.mol.nelectron)
mc = dmrgci_prep(mc=mc, mol=self.mf.mol, maxM=self.max_M, tol=1e-5)
edmrg = mc.kernel(mo_coeff)[0]

class DMRGMC:
    def __init__(self, mf, max_M):
        """
        Attributes:
            mf (pyscf mf object): mf object
            max_M (int): max bond dimension in DMRG
        """
        self.mf = mf
        self.max_M = max_M

    def kernel(self, mo_coeff=None):
        """
        Run DMRG calculation.

        Args:
            mo_coeff (ndarray): MO coefficients

        Returns:
            e_tot (float): Total DMRG energy
        """
        if mo_coeff is None:
            mo_coeff = self.mf.mo_coeff
        mc = mcscf.CASCI(self.mf, self.mf.mol.nelectron, self.mf.mol.nelectron)
        mc = dmrgci_prep(mc=mc, mol=self.mf.mol, maxM=self.max_M, tol=1e-5)
        return mc.kernel(mo_coeff)[0]

    def make_rdm1(self):
        """Make 1-RDM using DMRG from block2"""
    
    def make_rdm2(self):
        """Make 2-RDM using DMRG from block2"""

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
    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=maxM, tol=tol)
    mc.fcisolver.runtimeDir = './tmp'
    mc.fcisolver.scratchDirectory = './tmp'
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 1))
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