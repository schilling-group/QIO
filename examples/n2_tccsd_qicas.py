from pyscf import gto, scf, dmrgscf, mcscf
import os, copy
import numpy as np
from qicas import QICAS
import time


dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''


n_core = 0                    # number of frozen orbitals
ne = 14-2*n_core              # number of total electrons
n_cas = 8                     # number of active orbitals
n_should_close = 3            # target number of closed orbitals
# r = [float(sys.argv[-2])]     # list of geometry parameters
# bd = int(sys.argv[-1])        # max bond dimension for DMRG
r = np.arange(2.0,3.3,0.2)    # list of geometry parameters


bd = 200
tot_iter = 6
E_casci = np.zeros((len(r), 3+tot_iter))      # array of output data
E_tccsd = np.zeros((len(r), tot_iter))      # array of output data
basis = '321g'               # basis set


for i in range(len(r)):
    print('r=',r[i])
    E_casci[i,0] = r[i]

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
    mycas.max_cycle_macro = 150
    mycas.natorb = True
    mycas.max_cycle_micro = 50
    mycas.fix_spin_(ss=0)
    etot = mycas.kernel()[0]
    # casscf energy
    E_casci[i,1] = etot
    
    
    
    mol = gto.M(atom='N 0 0 0; N 0 0 '+"{:.4}".format(r[i]), 
        basis=basis,spin=0, verbose=1, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'A'
    mf = scf.RHF(mol)
    mf.kernel()
    mycasci = mcscf.CASCI(mf,n_cas,ne-2*n_should_close)
    mycasci.verbose = 4
    etot = mycasci.kernel(mf.mo_coeff)[0]
    # casci energy
    E_casci[i,2] = etot
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
   
    for micro_i in range(tot_iter):
        print("mo_energy", mf.mo_energy)
        my_qicas = QICAS(mf=mf, mc=None, act_space=act_space) 
        my_qicas.max_cycle = 100
        my_qicas.max_M = bd
        my_qicas.step_size = 0.5
        my_qicas.thresh = 5e-5
        my_qicas.dump_flags()
        e_qicas = my_qicas.kernel(is_tcc=True,  inactive_indices=inactive_indices,
            mo_coeff=mo_coeff, method='nr')
        mo_coeff = my_qicas.mo_coeff.copy()
        print("micro_i", micro_i)
        print("e_qicas", e_qicas)
        E_casci[i, micro_i+3] = e_qicas
        E_tccsd[i, micro_i] = my_qicas.tcc_e_tot
    
    t1 = time.time()
    print('icas time:',t1-t0)



print('r\tHF-CAS(6,6)\tCASSCF(6,6)\tQICAS-CASCI')
print(E_casci)
print('r\tTCCSD')
print(E_tccsd)
np.savetxt('n2_casci_energies_tcc_scan_cisd_iter5_321g_test.txt',E_casci)
np.savetxt('n2_tccsd_energies_tcc_scan_cisd_iter5_321g_test.txt',E_tccsd)




