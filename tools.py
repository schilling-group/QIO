from pyscf import gto, scf, lib, dmrgscf, mcscf
import pyscf
import os
import numpy as np
from orb_rot import *


def dmrgci_prep(mc,mol,maxM,hf=None,tol=1E-12):

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
    mc.fcisolver.runtimeDir = lib.param.TMPDIR
    mc.fcisolver.scratchDirectory = lib.param.TMPDIR
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 1))
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


def gamma_Gamma_prep(dm1,dm2):

    '''
    Prepare the 1- and 2-RDM. Pyscf returns spin averaged 1RDM
    Separate the spin-up and spin-down 1RDM
    Select the relevant part of 2RDM, namely elements where the first (last) two indices correspond to spin up (down)



    '''

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



def qicas(active_indices,inactive_indices,mf,no,n_cas,orbs,ne,mol,N_cycle,bd):


    mc = mcscf.CASCI(mf,no,ne)
    mc = dmrgci_prep(mc=mc,mol=mol,maxM=bd,tol=1e-5)
    edmrg = mc.kernel(orbs)[0]
    print('DMRG energy:',edmrg)
    dm1, dm2 = mc.fcisolver.make_rdm12(0,no,ne,spin=False)
    print('got rdms...')
    gamma,Gamma = gamma_Gamma_prep(dm1,dm2)
    

        
    rotations,U,gamma_,Gamma_ = minimize_orb_corr_jacobi(gamma,Gamma,active_indices,inactive_indices,N_cycle)
    rotation2, n_closed, V = reorder(gamma_,Gamma_,n_cas,inactive_indices)
    rotations =  rotations + rotation2
    U_ = np.matmul(V,U)

    orbs_ = orb_rot_pyscf(orbs,U_)


    mycas = mcscf.CASCI(mf,n_cas,ne-2*n_closed)
    
    mycas.fix_spin_(ss=0)
    mycas.canonicalization = True
    mycas.natorb = True
    etot = mycas.kernel(orbs_)[0]
        

    return etot,n_closed




