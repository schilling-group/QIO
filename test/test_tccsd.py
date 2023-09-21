import numpy as np
from pyscf import gto, scf, cc, mcscf
from tccsd import make_tailored_ccsd

def test_energy():
    t1 = t2 = None
    for d in np.arange(2.4, 2.6, 0.2):

        print("d = ", d)
        mol = gto.Mole()
        mol.atom = 'N 0 0 0; N 0 0 %f' % d
        mol.basis = 'sto6g'
        mol.verbose = 4
        mol.build()

        # Hartree Fock
        mf = scf.RHF(mol)
        mf.kernel()
        assert mf.converged

        # CCSD
        mycc = cc.CCSD(mf)
        mycc.kernel(t1=t1, t2=t2)
        t1, t2 = mycc.t1, mycc.t2
        e_ccsd = (mycc.e_tot if mycc.converged else np.nan)

        # CASCI(6,6)
        cas = mcscf.CASCI(mf, 6, 6)
        cas.kernel()
        assert cas.converged

        # TCCSD tailored with CASCI(6,6)
        tcc = cc.CCSD(mf)
        tcc = make_tailored_ccsd(tcc, cas)
        #tcc.max_cycle = 2
        tcc.kernel()
        #assert tcc.converged

        casscf = mcscf.CASSCF(mf, 6, 6)
        casscf.kernel()
        assert casscf.converged

        with open('n2_energies.txt', 'a') as f:
            f.write('%.2f  %.8f  %.8f  %.8f %.8f  %.8f\n' % (d, mf.e_tot, e_ccsd, cas.e_tot, casscf.e_tot, tcc.e_tot))


