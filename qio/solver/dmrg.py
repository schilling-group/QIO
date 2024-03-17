"""
Module for DMRG solver, including the function to prepare a DMRGCI object
and some other tools.
"""
from pyscf import dmrgscf
import os
# DMRG and RDMs prep block

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''

if dmrgscf.settings.BLOCKEXE == '':
    raise FileNotFoundError("block2main not found")

class DMRG:
    """
    DMRG solver class

    This is a wrapper for the pyscf DMRGCI solver. It is used to prepare the
    solver object and to store the 1- and 2-RDMs.

    See the documentation of block2 and pyscf for more details.
    """

    def __init__(self, mf, mc, max_M=200, tol=1E-12):
        """
        Args:
            mf (pyscf.scf object): mean-field object
            mc: solver for obtaining 1- and 2-RDM
            max_M (int): max bond dimension in DMRG
            tol (float): tolerance for DMRG convergence
        """
        self.mf = mf
        self.no = len(mf.mo_coeff)
        self.max_M = max_M
        self.mc = mc
        self.mc.fcisolver = dmrgscf.DMRGCI(mf.mol, maxM=max_M, tol=tol)
        self.mc.fcisolver.runtimeDir = './'
        self.mc.fcisolver.scratchDirectory = './tmp'
        self.mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 1))
        self.mc.fcisolver.memory = int(mf.mol.max_memory / 1000) # mem in GB
        self.mc.wfnsym='A1g'
        self.mc.canonicalization = False
        self.mc.natorb = False
    
        self.mc.fcisolver.scheduleSweeps = [0, 4]
        self.mc.fcisolver.scheduleMaxMs = [150, 150] 
        self.mc.fcisolver.scheduleTols = [1e-08, 1e-8]
        self.mc.fcisolver.scheduleNoises = [0.0001, 0.0001]
        self.mc.fcisolver.maxIter = 30
        self.mc.fcisolver.twodot_to_onedot = 20
        
        # dm1 and dm2
        self.dm1 = None
        self.dm2 = None

        self.e_tot = None
    
    def kernel(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mf.mo_coeff
        self.e_tot = self.mc.kernel(mo_coeff)[0]
        return self.e_tot
    
    def make_rdm1(self):
        if self.dm1 is None:
            self.dm1, self.dm2 = self.mc.fcisolver.make_rdm12(0, self.no, self.mf.mol.nelectron)
        return self.dm1
    
    def make_rdm2(self):
        if self.dm2 is None:
            self.dm1, self.dm2 = self.mc.fcisolver.make_rdm12(0, self.no, self.mf.mol.nelectron)
        return self.dm2
