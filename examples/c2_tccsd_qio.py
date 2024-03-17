from pyscf import gto, scf, mcscf
import copy
import numpy as np
from qio import QIO
from qio.entropy import get_cost_fqi
import time
from qio.solver.tccsd import TCCSD


n_core = 0                    # number of frozen orbitals
ne = 12-2*n_core              # number of total electrons
n_cas = 8                     # number of active orbitals
n_should_close = 2            # target number of closed orbitals
n_act_e = ne-2*n_should_close # number of active electrons
r = [1.8]
tot_iter = 5
E_casci = np.zeros((len(r), 3+tot_iter))      # array of output data
E_tccsd = np.zeros((len(r), tot_iter))      # array of output data
entropy_tccsd = np.zeros((len(r), tot_iter))      # array of output data
E_dmrg = np.zeros((len(r), tot_iter))      # array of output data
basis = 'ccpvdz'               # basis set


for i in range(len(r)):
    print('r=',r[i])
    E_casci[i,0] = r[i]

    # The following code runs a HF-CASSCF and HF-CASCI from sratch for comparison
    
    mol = gto.M(atom='C 0 0 0; C 0 0 '+"{:.4}".format(r[i]), 
        basis=basis,spin=0, verbose=4, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'B'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    
    
    mo_coeff = copy.deepcopy(mf.mo_coeff)
    no = len(mo_coeff)-n_core

    t0 = time.time()

    # Use the following indices if you want to optimize the active space 
    # orbitals only
    # active_indices = list(range(n_should_close,n_cas+n_should_close))
    # inactive_indices = list(range(n_should_close))+list(range(n_cas+n_should_close,no))

    # Use the following indices if you want to optimize the whole space
    active_indices = []
    inactive_indices = list(range(no))
    act_space = (n_cas,ne-2*n_should_close)

    # prepare the wavefunction solver for the QIO
    # e.g. TCCSD
    # first prepare for the active space solver CASCI
    mc = mcscf.CASCI(mf, *act_space)
    mc.verbose = mf.verbose
    mc.canonicalization = True
    mc.sorting_mo_energy = False 
    mc.fcisolver.max_cycle = 800
    mc.fix_spin_(ss=0, shift=0.5)
    mc.tol = 1e-8
    mc.fcisolver.level_shift = 0.1
    mc.natorb = False

    my_tcc = TCCSD(mf, mc)

   
    my_qio = QIO(mf=mf, sol=my_tcc, act_space=act_space) 
    my_qio.max_cycle = 4
    my_qio.step_size = 0.2
    my_qio.thresh = 1e-7
    my_qio.dump_flags()
    
    for micro_i in range(tot_iter):
        mo_coeff = my_qio.kernel(inactive_indices=inactive_indices, mo_coeff=mo_coeff, method='nr')
        E_tccsd[i, micro_i] = my_tcc.e_tot

        entropy_tccsd[i, micro_i] = get_cost_fqi(my_qio.gamma, my_qio.Gamma, inactive_indices)
        #entropy_tccsd[i, micro_i] = get_cost_fqi(gamma, Gamma, inactive_indices)
        np.savetxt('c2_test_energy_'+basis+'.8in8.txt',E_tccsd)
        np.savetxt("c2_test_entropy_"+basis+".8in8.txt", np.asarray(entropy_tccsd))


