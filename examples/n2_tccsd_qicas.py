from pyscf import gto, scf, dmrgscf, mcscf
import os,sys,copy
import numpy as np
from qicas import *
from orb_rot import *
import time


dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''


n_core = 0                    # number of frozen orbitals
ne = 14-2*n_core              # number of total electrons
n_cas = 6                     # number of active orbitals
n_should_close = 4            # target number of closed orbitals
# r = [float(sys.argv[-2])]     # list of geometry parameters
# bd = int(sys.argv[-1])        # max bond dimension for DMRG
r = [2.0]
bd = 100
E = np.zeros((len(r),4))      # array of output data
basis = '321g'               # basis set


for i in range(len(r)):
    print('r=',r[i])
    E[i,0] = r[i]

    # The following code runs a HF-CASSCF and HF-CASCI from sratch for comparison
    
    mol = gto.M(atom='N 0 0 0; N 0 0 '+"{:.4}".format(r[i]), 
        basis=basis,spin=0, verbose=1, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'A'

    mf = scf.RHF(mol)
    mf.kernel()
    
    mycas = mcscf.CASSCF(mf,n_cas,ne-2*n_should_close)
    mycas.verbose = 4
    mycas.frozen = n_core
    mycas.max_cycle_macro = 100
    mycas.natorb = True
    mycas.max_cycle_micro = 50
    mycas.fix_spin_(ss=0)
    etot = mycas.kernel()[0]
    E[i,2] = etot
    
    
    
    mol = gto.M(atom='N 0 0 0; N 0 0 '+"{:.4}".format(r[i]), 
        basis=basis,spin=0, verbose=1, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'A'
    mf = scf.RHF(mol)
    mf.kernel()
    mycasci = mcscf.CASCI(mf,n_cas,ne-2*n_should_close)
    mycasci.verbose = 4
    etot = mycasci.kernel(mf.mo_coeff)[0]
    E[i,3] = etot
    # create molecule with desired geometry and basis

    mol = gto.M(atom='N 0 0 0; N 0 0 '+"{:.4}".format(r[i]), 
        basis=basis,spin=0, verbose=4, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'A'

    # Run RHF

    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.kernel()
    
    mo_coeff = copy.deepcopy(mf.mo_coeff)
    no = len(mo_coeff)-n_core

    t0 = time.time()

    # Run QICAS and output post-QICAS CASCI energy
    active_indices = list(range(n_should_close,n_cas+n_should_close))
    inactive_indices = list(range(n_should_close))+list(range(n_cas+n_should_close,no))
    act_space = (n_cas,ne-2*n_should_close)
    for micro_i in range(2):
        print("mo_energy", mf.mo_energy)
        my_qicas = QICAS(mf=mf, mc=None, act_space=act_space) 
        my_qicas.max_cycle = 100
        my_qicas.max_M = bd

        e_qicas,nclosed = my_qicas.kernel(is_tcc=True,  inactive_indices=inactive_indices,
            mo_coeff=mo_coeff)
        mo_coeff = my_qicas.mo_coeff.copy()
        print("micro_i", micro_i)
        print("e_qicas", e_qicas)
    E[i,1] = e_qicas
    print(e_qicas)
    
    
    t1 = time.time()
    print('icas time:',t1-t0)




    print(E[i,0],E[i,3],E[i,2],E[i,1])


print('r\tHF-CAS(8,8)\tCASSCF(8,8)\tQICAS-CASCI')
for i in range(len(r)):
    print(E[i,0],E[i,3],E[i,2],E[i,1])

np.savetxt('n2_energies_tcc1.txt',E)




