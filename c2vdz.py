from pyscf import gto, scf, lib, dmrgscf, mcscf
import pyscf
import os,sys,copy
import numpy as np
from tools import *
from orb_rot import *
import time


dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''
path = 'nosym_casscf_random_bd200/'

n_core = 0
ne = 12-2*n_core
n_cas = 8
n_should_close = 2
r = [float(sys.argv[-2])]
bd = int(sys.argv[-1])
E = np.zeros((len(r),4))
data = np.zeros((len(r),5))



for i in range(len(r)):
    print('r=',r[i])
    E[i,0] = r[i]

    # create molecule with desired geometry and basis

    mol = gto.M(atom='C 0 0 0; C 0 0 '+"{:.4}".format(r[i]), 
        basis='ccpvdz',spin=0, verbose=1, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'A'

    mf = scf.RHF(mol)
    mf.kernel()
    
    orbs = copy.deepcopy(mf.mo_coeff)
    sym = 0
    print(len(orbs))
    no = len(orbs)-n_core

    t0 = time.time()

    active_indices = list(range(n_should_close,n_cas+n_should_close))
    inactive_indices = list(range(n_should_close))+list(range(n_cas+n_should_close,no))
    
    e_qicas,nclosed = qicas(active_indices = active_indices, inactive_indices = inactive_indices,
        mf=mf,no=no,n_cas=n_cas,orbs=orbs,ne=ne,mol=mol,N_cycle=200,bd=bd)
    E[i,1] = e_qicas
    print(e_qicas)
    
    
    t1 = time.time()
    print('icas time:',t1-t0)

    
    mol = gto.M(atom='C 0 0 0; C 0 0 '+"{:.4}".format(r[i]), 
        basis='ccpvdz',spin=0, verbose=1, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'A'

    mf = scf.RHF(mol)
    mf.kernel()

    
    mycas = mcscf.CASSCF(mf,n_cas,ne-2*n_should_close)
    mycas.frozen = n_core
    mycas.max_cycle_macro = 100
    mycas.max_cycle_micro = 50
    mycas.fix_spin_(ss=0)
    etot = mycas.kernel()[0]
    E[i,2] = etot
    
    t2 = time.time()
    print('CASSCF time:', t2 - t1)
    
    
    mycas = mcscf.CASCI(mf,n_cas,ne-2*n_should_close)
    etot = mycas.kernel(mf.mo_coeff)[0]
    E[i,3] = etot
    

    print(E[i,0],E[i,3],E[i,2],E[i,1])


print('r\tHF-CAS(8,8)\tCASSCF(8,8)\tQICAS-CASCI')
for i in range(len(r)):
    print(E[i,0],E[i,3],E[i,2],E[i,1])




