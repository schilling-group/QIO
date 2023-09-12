from pyscf import gto, scf, dmrgscf, mcscf
import os,sys,copy
import numpy as np
from qicas import *
from orb_rot import *
import time


dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''


n_core = 0                    # number of frozen orbitals
ne = 12-2*n_core              # number of total electrons
n_cas = 8                     # number of active orbitals
n_should_close = 2            # target number of closed orbitals
# r = [float(sys.argv[-2])]     # list of geometry parameters
# bd = int(sys.argv[-1])        # max bond dimension for DMRG
r = [1.2] 
bd = 100
E = np.zeros((len(r),4))      # array of output data



for i in range(len(r)):
    print('r=',r[i])
    E[i,0] = r[i]

    # create molecule with desired geometry and basis

    mol = gto.M(atom='C 0 0 0; C 0 0 '+"{:.4}".format(r[i]), 
        basis='sto6g',spin=0, verbose=1, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'A'

    # Run RHF

    mf = scf.RHF(mol)
    mf.kernel()
    
    mo_coeff = copy.deepcopy(mf.mo_coeff)
    no = len(mo_coeff)-n_core

    t0 = time.time()

    # Run QICAS and output post-QICAS CASCI energy
    
    active_indices = list(range(n_should_close,n_cas+n_should_close))
    inactive_indices = list(range(n_should_close))+list(range(n_cas+n_should_close,no))
    act_space = (n_cas,ne-2*n_should_close)
    my_qicas = QICAS(mf=mf, mc=None, act_space=act_space) 
    my_qicas.max_cycle = 100
    my_qicas.max_M = bd

    e_qicas,nclosed = my_qicas.kernel(active_indices=active_indices, inactive_indices=inactive_indices,
        mo_coeff=mo_coeff)
    E[i,1] = e_qicas
    print(e_qicas)
    
    
    t1 = time.time()
    print('icas time:',t1-t0)



    # The following code runs a HF-CASSCF and HF-CASCI from sratch for comparison
    
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




