
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

def set_zero_active_t1t2(t1, t2, nocc, nvir, ncore):
    """
    Set the active amplitudes to zero.
    
    Args:
        t1 (ndarray): T1 amplitudes
        t2 (ndarray): T2 amplitudes
        nocc (int): Number of occupied orbitals
        nvir (int): Number of virtual orbitals
        ncore (int): Number of core orbitals
    
    Returns:
        t1 (ndarray): T1 amplitudes
        t2 (ndarray): T2 amplitudes
    """
    t1[ncore:nocc, :nvir] = 0.
    t2[ncore:nocc, ncore:nocc, :nvir, :nvir] = 0.
    return t1, t2

def make_tailored_ccsd(cc, cas, is_dmrg=False):
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
    is_good_ref = True

    def find_ref_det(cas):
        """
        Identifies the dominant det in the CASCI solution. 
        reorder the mo_coeff so that this det represents the reference
        det for CCSD.

        Args:
            cas: the CASCI/CASSCF object
        
        Returns:
            cas (ndarray): reordered mo coefficients.
        """
        cisdvec = pyscf.ci.cisd.from_fcivec(cas.ci, cas.ncas, nelec_cas)
        ref_det_idx = np.argmax(np.abs(cisdvec))
        curr_str = '1'*cc.nocc*2
        det_str = pyscf.fci.cistring.addr2str(cc.nmo, cc.nocc*2, ref_det_idx)
        det_bin = bin(ref_det_idx)[2:]
        locs = list(set(find_ones(det_bin)).symmetric_difference(set(find_ones(curr_str))))
        act_dm1 = cas.fcisolver.make_rdm1(cas.ci, cas.ncas, cas.nelecas).diagonal()
        sort_idx = list(range(cas.ncore)) + list(np.argsort(act_dm1)[::-1] + cas.ncore) + list(range(cas.ncore+cas.ncas, cc.nmo))
        # sort the orbitals
        cas.mo_coeff = cas.mo_coeff[:,sort_idx]

        #if len(locs) > 0:
        #    locs = np.array(locs)//2
        #    if np.min(locs) > cas.ncore and np.max(locs) < cas.ncore+cas.ncas:
        #        print("Swapping indices in determinant: ", locs)
        #        if len(locs) > 2:
        #            raise ValueError("Error: more than two orbitals to swap. This is not supported yet.")

        #        # swap the orbitals
        #        mo_coeff = cas.mo_coeff.copy()
        #        cas.mo_coeff[:,locs] = mo_coeff[:,[locs[1],locs[0]]]
                #ucas = cas.mo_coeff[:,cas.ncore:cas.ncore+cas.ncas].copy()

                # update the ci vectors
                #cas.ci = cas.fcisolver.transform_ci_for_orbital_rotation(cas.ci, cas.ncas, cas.nelecas, ucas)
        return cas

    def find_ones(s):
        return [i for i, char in enumerate(reversed(s)) if char == '1']

    def get_cas_t1t2(cas):
        """Get T1 and T2 amplitudes from FCI wave function."""
        is_good_ref = True
        cisdvec = pyscf.ci.cisd.from_fcivec(cas.ci, cas.ncas, nelec_cas)
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, cas.ncas, nocc_cas)
        c1_max = np.max(np.abs(c1))
        print("|C0| = %.4e, |C1_max| = %.4e" % (np.abs(c0), c1_max))

        #while np.abs(c0) < c1_max or abs(c0) < 2e-3:
        #    print("Warning: |C0| = %.4e is smaller than |C1_max| = %.4e. Current choice of reference determinant is bad!" % (np.abs(c0), c1_max))
        #    print("Trying to find a better reference determinant in the singles space...")
        #    cas = find_ref_det(cas)
        #    print("Rerun CASCI to get new CI vector")
        #    cas.natorb = True
        #    cas.kernel()
        #    cas.natorb = False 
        #    cisdvec = pyscf.ci.cisd.from_fcivec(cas.ci, cas.ncas, nelec_cas)
        #    c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, cas.ncas, nocc_cas)
        #    c1_max = np.max(np.abs(c1))
        #    print("New CI coeffs:")
        #    print("|C0| = %.4e, |C1_max| = %.4e" % (np.abs(c0), c1_max))

        assert (np.abs(c0) > 1e-8)
        if (np.abs(c0)<c1_max) or (np.abs(c0)<1e-2):
        #if (np.abs(c0) < 1e-1) or (np.abs(c0)<c1_max):
            is_good_ref = False
            print("Warning: |C0| = %.4e is too small for TCCSD. Current orbitals are a bad guess!" % np.abs(c0))
        t1 = c1/c0
        t2 = c2/c0 - einsum('ia,jb->ijab', t1, t1)
        return t1, t2, is_good_ref

    t1cas_fci, t2cas_fci, is_good_ref = get_cas_t1t2(cas)
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
    return cc, t1_init, t2_init, is_good_ref 

def make_no(rdm1, mo_coeff, subspace=None):
    """
    Make natural orbitals from 1-RDM.
    
    Args:
        rdm1 (ndarray): 1-RDM
        mo_coeff (ndarray): MO coefficients
        subspace (list): space indices in which natural orbitals are constructed
    
    Returns:
        no_coeff: Natural orbitals
    """
    if subspace is not None:
        rdm1_sub = rdm1[subspace,:][:,subspace]
        mo_coeff_sub = mo_coeff[:, subspace]
    else:
        rdm1_sub = rdm1
        mo_coeff_sub = mo_coeff
    # diagonalize 1-RDM
    e, v = np.linalg.eigh(rdm1_sub)
    # sort in descending order
    idx = np.argsort(e)[::-1]
    e = e[idx]
    v = v[:,idx]
    # transform to NO basis
    no_coeff_sub = np.dot(mo_coeff_sub, v)
    no_coeff = mo_coeff.copy()
    if subspace is not None:
        no_coeff[:, subspace] = no_coeff_sub
        rdm1_new = rdm1.copy()
        rdm1_new[subspace,:][:,subspace] = np.diag(np.ones(len(subspace))*2)
    else:
        no_coeff = no_coeff_sub
        rdm1_new = np.diag(np.ones(len(rdm1))*2)
    
    return no_coeff, rdm1_new


def semi_canonicalize(mf, mo_coeff, nocc):
    """
    Semi-canonicalize MO coefficients.
    
    Args:
        mf: Mean-field object
        mo_coeff: MO coefficients
    """
    from functools import reduce
    fockao = mf.get_fock()
    fockmo = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
    foo = fockmo[:nocc,:nocc]
    fvv = fockmo[nocc:,nocc:]
    _, v_canon_occ = np.linalg.eigh(foo)
    _, v_canon_vir = np.linalg.eigh(fvv)
    mo_coeff_occ = np.dot(mo_coeff[:,:nocc], v_canon_occ)
    mo_coeff_vir = np.dot(mo_coeff[:,nocc:], v_canon_vir)
    mo_coeff_semican = np.concatenate((mo_coeff_occ, mo_coeff_vir), axis=1)

    return mo_coeff_semican