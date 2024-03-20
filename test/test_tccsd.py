import numpy as np
from pyscf import gto, scf, cc, mcscf
from qio.solver.tccsd import make_tailored_ccsd

def test_energy():
    t1 = t2 = None

    mol = gto.Mole()
    mol.atom = 'N 0 0 0; N 0 0 1.8' 
    mol.basis = 'sto6g'
    mol.verbose = 4
    mol.build()

    # Hartree Fock
    mf = scf.RHF(mol)
    mf.kernel()
    assert mf.converged

    # CCSD
    mycc = cc.CCSD(mf)

    # CASCI(6,6)
    cas = mcscf.CASCI(mf, 6, 6)
    cas.kernel()
    assert cas.converged

    # TCCSD tailored with CASCI(6,6)
    tcc = cc.CCSD(mf)
    tcc, _, _, _ = make_tailored_ccsd(tcc, cas)
    #tcc.max_cycle = 2
    e = tcc.kernel()[0]
    
    assert np.abs(e - -0.4681381077569038) < 1e-8
    print('Passed!')
   

