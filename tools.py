from pyscf import gto, scf, lib, dmrgscf, mcscf
import pyscf
import os
import numpy as np
from orb_rot import *


def dmrgci_prep(mc,mol,maxM,stages=0,hf=None,tol=1E-12):
    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=maxM, tol=tol)
    mc.fcisolver.runtimeDir = lib.param.TMPDIR
    mc.fcisolver.scratchDirectory = lib.param.TMPDIR
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 1))
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB
    mc.wfnsym='A1g'
    mc.canonicalization = False
    mc.natorb = False
    if stages == 1:
        mc.fcisolver.restart = False
        mc.fcisolver.scheduleSweeps =[50] 
        mc.fcisolver.scheduleMaxMs = [maxM] 
        mc.fcisolver.scheduleTols = [1e-6] 
        mc.fcisolver.scheduleNoises = [0] 
    return mc

def dmrgscf_prep(mol,mf,maxM,nactorb,nactelec):
    mc = dmrgscf.DMRGSCF(mf=mf, norb=nactorb, nelec=nactelec, maxM=maxM, tol=1E-10)
    mc.fcisolver.runtimeDir = lib.param.TMPDIR
    mc.fcisolver.scratchDirectory = lib.param.TMPDIR
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 1))
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB
    mc.canonicalization = True
    mc.natorb = True
    return mc

def gamma_Gamma_prep(dm1,dm2):
    no = len(dm1)

    Gamma = np.zeros((no,no,no,no))
    gamma = np.zeros((2*no,2*no))


    for a in range(no):
        for b in range(no):
            gamma[2*a,2*b] = dm1[a,b]/2
            gamma[2*a+1,2*b+1] = dm1[a,b]/2
            for c in range(no):
                for d in range(no):
                    Gamma[a,b,c,d] = dm2[a,c,d,b]/2

    return gamma, Gamma



def qicas(active_indices,inactive_indices,Nrepeat,mf,no,n_cas,orbs,ne,mol,N_cycle,bd):

    emin = 0
    cost_min = 100
    

    for n in range(Nrepeat):
        edmrg = 0
        
        for k in range(1):
            mc = mcscf.CASCI(mf,28,12)
            mc = dmrgci_prep(mc=mc,mol=mol,maxM=bd,stages=1,tol=1e-5)
            edmrg_ = mc.kernel(orbs)[0]
            print(edmrg_)
            if edmrg_ < edmrg:
                dm1, dm2 = mc.fcisolver.make_rdm12(0,no,ne,spin=True)
                print('got rdms...')
                gamma,Gamma = gamma_Gamma_prep(dm1,dm2)
                g = np.zeros((no,no))
                for i in range(no):
                    g[i,i] = gamma[2*i,2*i] + gamma[2*i+1,2*i+1]
                print(np.diag(g))
                edmrg = edmrg_
        
        
        
        for count in range(1):
            rotations,U,gamma_,Gamma_ = minimize_orb_corr_jacobi(gamma,Gamma,active_indices,inactive_indices,N_cycle)
            rotation2, n_closed, V = reorder(gamma_,Gamma_,n_cas,inactive_indices)
            rotations =  rotations + rotation2
            U_ = np.matmul(V,U)
    
            orbs_ = orb_rot_pyscf(orbs,U_)

            nu = int((ne-2*n_closed)/2)
            nd = int((ne-2*n_closed)/2)

            mycas = mcscf.CASCI(mf,n_cas,ne-2*n_closed)
            
            
        
            mycas.fix_spin_(ss=0)
            mycas.canonicalization = True
            mycas.natorb = True
            etot = mycas.kernel(orbs_)[0]

        
        
            if etot < emin:
                egrad = emin-etot
                emin = etot
                orbs_opt = orbs_
                n_closed_opt = n_closed

        if egrad < 1e-6:
            break
        

            

    return emin,etot,n_closed,nu,nd




