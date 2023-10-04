
"""
This file is based on the example from pyscf/examples/cc/04-tailored-ccsd.py
"""
from functools import partial
import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
from pyscf.ci import cisd
import pyscf.mcscf
from pyscf.mp.mp2 import _mo_without_core

einsum = partial(np.einsum, optimize=True)

def make_tailored_ccsd(cc, cas):
    """Create tailored CCSD calculation."""

    nelec_cas = sum(cas.nelecas)
    nocc_cas = nelec_cas//2
    # Determine (MO|CAS) overlap:
    mo_cc = _mo_without_core(cc, cc.mo_coeff)
    nocc_cc = cc.get_nocc()
    mo_cc_occ = mo_cc[:,:nocc_cc]
    mo_cc_vir = mo_cc[:,nocc_cc:]
    mo_cas = cas.mo_coeff[:,cas.ncore:cas.ncore+cas.ncas]
    mo_cas_occ = mo_cas[:,:nocc_cas]
    mo_cas_vir = mo_cas[:,nocc_cas:]
    ovlp = cc._scf.get_ovlp()
    pocc = np.linalg.multi_dot((mo_cc_occ.T, ovlp, mo_cas_occ))
    pvir = np.linalg.multi_dot((mo_cc_vir.T, ovlp, mo_cas_vir))

    def get_cas_t1t2(cas):
        """Get T1 and T2 amplitudes from FCI wave function."""
        cisdvec = pyscf.ci.cisd.from_fcivec(cas.ci, cas.ncas, nelec_cas)
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, cas.ncas, nocc_cas)
        assert (abs(c0) > 1e-8)
        t1 = c1/c0
        t2 = c2/c0 - einsum('ia,jb->ijab', t1, t1)
        return t1, t2

    t1cas_fci, t2cas_fci = get_cas_t1t2(cas)
    t1_init = einsum('ia,Ii,Aa->IA', t1cas_fci, pocc, pvir)
    t2_init = einsum('ijab,Ii,Jj,Aa,Bb->IJAB', t2cas_fci, pocc, pocc, pvir, pvir)

    def callback(kwargs):
        """Tailor CCSD amplitudes within CAS."""
        t1, t2 = kwargs['t1new'], kwargs['t2new']
        # Project CCSD amplitudes onto CAS:
        t1cas_cc = einsum('IA,Ii,Aa->ia', t1, pocc, pvir)
        t2cas_cc = einsum('IJAB,Ii,Jj,Aa,Bb->ijab', t2, pocc, pocc, pvir, pvir)
        #assert np.allclose(t1cas_cc, t1[cas.ncore:cas.ncore+nocc_cas, :cas.ncas-nocc_cas])
        # Take difference FCI-CCSD within CAS:
        dt1 = (t1cas_fci - t1cas_cc)
        dt2 = (t2cas_fci - t2cas_cc)
        # Rotate difference to CCSD space:
        dt1 = einsum('ia,Ii,Aa->IA', dt1, pocc, pvir)
        dt2 = einsum('ijab,Ii,Jj,Aa,Bb->IJAB', dt2, pocc, pocc, pvir, pvir)
        # Add difference:
        t1 += dt1
        t2 += dt2

    def make_rdm1():
        """Make 1-RDM using C1 and C2 from CCSD"""
        t1, t2 = cc.t1, cc.t2
        ind0 = cas.ncore
        ind1 = cas.ncore+nocc_cas
        ind2 = cas.ncas-nocc_cas
        #assert np.allclose(t1cas_fci, t1[ind0:ind1, :ind2])
        #assert np.allclose(t2cas_fci, t2[ind0:ind1, ind0:ind1, :ind2, :ind2])
        c0 = 1.
        c1 = t1.copy()
        c2 = t2 + einsum('ia,jb->ijab', t1, t1) 
        mycisd = cisd.CISD(cc._scf)
        cisdvec = pyscf.ci.cisd.amplitudes_to_cisdvec(c0, c1, c2)
        cisdvec /= np.linalg.norm(cisdvec)
        return mycisd.make_rdm1(cisdvec)

    def make_rdm2():
        """Make 2-RDM using C1 and C2 from CCSD"""
        t1, t2 = cc.t1, cc.t2
        c0 = 1.
        c1 = t1
        c2 = t2 + einsum('ia,jb->ijab', t1, t1) 
        mycisd = cisd.CISD(cc._scf)
        cisdvec = pyscf.ci.cisd.amplitudes_to_cisdvec(c0, c1, c2)
        cisdvec /= np.linalg.norm(cisdvec)
        return mycisd.make_rdm2(cisdvec)

    cc.callback = callback
    cc.make_rdm1 = make_rdm1
    cc.make_rdm2 = make_rdm2
    return cc, t1_init, t2_init 