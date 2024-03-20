import numpy as np
from pyscf import gto, scf, fci, mcscf
from qio.solver.dmrg import DMRG

'''
This is a test to check whether the DMRG solver works correctly.
'''

def test_dmrg():
    # Build an H2 molecule
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', 
        basis='ccpvdz',spin=0, verbose=0, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'B'
    mol.build()
    
    mf = scf.RHF(mol)
    mf.kernel()
    
    # Solve for the exact ground state via FCI
    cisolver = fci.FCI(mf)
    efci, civec = cisolver.kernel()
    #print('FCI energy:',efci)
    
    # Run DMRG
    mc = mcscf.CASCI(mf, 10,2)
    my_dmrg = DMRG(mf, mc=mc, max_M=150, tol=1e-8)
    edmrg = my_dmrg.kernel()
    #print(edmrg)
    #print('DMRG error:',edmrg-efci)
    assert np.abs(edmrg - efci) < 1e-8

