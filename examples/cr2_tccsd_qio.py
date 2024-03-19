from pyscf import gto, scf, mcscf
import os, copy
import numpy as np
from qio import QIO
from solver.gradient import get_cost_fqi
import time
from qio.solver.tccsd import TCCSD


n_core = 0                    # number of frozen orbitals
ne = 24-2*n_core              # number of total electrons
n_cas = 12                     # number of active orbitals
n_should_close = 6           # target number of closed orbitals
n_act_e = ne-2*n_should_close # number of active electrons
r =[1.8]

tot_iter = 20
E_casci = np.zeros((len(r), 3+tot_iter))      # array of output data
E_tccsd = np.zeros((len(r), tot_iter))      # array of output data
entropy_tccsd = np.zeros((len(r), tot_iter))      # array of output data
E_dmrg = np.zeros((len(r), tot_iter))      # array of output data
basis = 'cc-pVDZ-DK'               # basis set


for i in range(len(r)):
    mol = gto.M(atom='Cr 0 0 0; Cr 0 0 '+"{:.4}".format(r[i]), 
        basis=basis,spin=0, verbose=4, 
        max_memory=50000,symmetry = False, unit='A') # mem in MB
    mol.build()

    mf = scf.RHF(mol).x2c()
    mf.kernel()
    

    mo_coeff = copy.deepcopy(mf.mo_coeff)
    no = len(mo_coeff)-n_core

    t0 = time.time()

    active_indices = []
    inactive_indices = list(range(no))
    act_space = (n_cas,ne-2*n_should_close)
   
    mc = mcscf.CASCI(mf, *act_space)
    mc.verbose = mf.verbose
    mc.canonicalization = True
    mc.sorting_mo_energy = False 
    mc.fcisolver.max_cycle = 800
    # the shift parameter can be very important for converging to
    # the correct solution.
    mc.fix_spin_(ss=0, shift=5.0)
    mc.tol = 1e-8
    mc.fcisolver.level_shift = 0.1
    mc.natorb = False

    my_tcc = TCCSD(mf, mc)

    my_qio = QIO(mf=mf, sol=my_tcc, act_space=act_space) 
    my_qio.max_cycle = 20
    my_qio.step_size = 0.2
    my_qio.thresh = 1e-7
    my_qio.dump_flags()
    
    for micro_i in range(tot_iter):

        if micro_i < 1:
            my_qio.max_cycle = 10
        else:
            my_qio.max_cycle = 30

        mo_coeff = my_qio.kernel(inactive_indices=inactive_indices, mo_coeff=mo_coeff, method='nr')
        E_tccsd[i, micro_i] = my_tcc.e_tot

        entropy_tccsd[i, micro_i] = get_cost_fqi(my_qio.gamma, my_qio.Gamma, inactive_indices)
        np.savetxt('cr2_tccsd_energy_'+basis+'.12in12.txt',E_tccsd)
        np.savetxt("cr2_tccsd_entropy_"+basis+".12in12.txt", np.asarray(entropy_tccsd))
    np.save('cr2_mo_coeff_'+basis+'.npy', mo_coeff)