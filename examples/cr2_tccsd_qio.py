from pyscf import gto, scf, dmrgscf, mcscf, cc, fci
from pyscf.tools import fcidump
import os, copy
import numpy as np
from qio import QIO, prep_rdm12
from solver.gradient import get_cost_fqi
import time
from qio.solver.tccsd import make_no, make_tailored_ccsd
from dmrg_block2 import run_dmrg

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''


n_core = 0                    # number of frozen orbitals
ne = 24-2*n_core              # number of total electrons
n_cas = 10                     # number of active orbitals
n_should_close = 6           # target number of closed orbitals
n_act_e = ne-2*n_should_close # number of active electrons
# r = [float(sys.argv[-2])]     # list of geometry parameters
# bd = int(sys.argv[-1])        # max bond dimension for DMRG
r = np.arange(3.8,4.0,0.2)    # list of geometry parameters
r =[3.2]
#r = [2.4]

bd = 100
tot_iter = 20
E_casci = np.zeros((len(r), 3+tot_iter))      # array of output data
E_tccsd = np.zeros((len(r), tot_iter))      # array of output data
entropy_tccsd = np.zeros((len(r), tot_iter))      # array of output data
E_dmrg = np.zeros((len(r), tot_iter))      # array of output data
basis = '321g'               # basis set


for i in range(len(r)):
    print('r=',r[i])
    E_casci[i,0] = r[i]

    # The following code runs a HF-CASSCF and HF-CASCI from sratch for comparison
    
    mol = gto.M(atom='Cr 0 0 0; Cr 0 0 '+"{:.4}".format(r[i]), 
        basis=basis,spin=0, verbose=4, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'A'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    
    #mycas = mcscf.CASSCF(mf,n_cas,ne-2*n_should_close)
    #mycas.verbose = 4
    #mycas.frozen = n_core
    #mycas.max_cycle_macro = 150
    #mycas.natorb = True
    #mycas.max_cycle_micro = 50
    #mycas.fix_spin_(ss=0)
    #etot = mycas.kernel()[0]
    ## casscf energy
    #E_casci[i,1] = etot
    
    
    
    mol = gto.M(atom='N 0 0 0; N 0 0 '+"{:.4}".format(r[i]), 
        basis=basis,spin=0, verbose=1, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'A'
    mf = scf.RHF(mol)
    mf.kernel()
    mycasci = mcscf.CASCI(mf,n_cas,ne-2*n_should_close)
    mycasci.verbose = 4
    mycasci.natorb = True
    shift = 50
    mycasci.fcisolver.max_cycle = 600
    mycasci.fix_spin_(ss=0, shift=shift)
    etot = mycasci.kernel(mf.mo_coeff)[0]

   
    ## casci energy
    #E_casci[i,2] = etot
    # create molecule with desired geometry and basis

    #mol = gto.M(atom='N 0 0 0; N 0 0 '+"{:.4}".format(r[i]), 
    #    basis=basis,spin=0, verbose=4, 
    #    max_memory=50000,symmetry = False) # mem in MB
    #mol.unit = 'A'

    # Run RHF

    #mf = scf.RHF(mol)
    #mf.verbose = 4
    #mf.kernel()
    
    mo_coeff = copy.deepcopy(mycasci.mo_coeff)
    #mo_coeff = np.load('cr2_mo_coeff_'+basis+'.npy')
    no = len(mo_coeff)-n_core

    t0 = time.time()

    # Run QICAS and output post-QICAS CASCI energy
    #active_indices = list(range(n_should_close,n_cas+n_should_close))
    #inactive_indices = list(range(n_should_close))+list(range(n_cas+n_should_close,no))
    active_indices = []
    inactive_indices = list(range(no))
    act_space = (n_cas,ne-2*n_should_close)
   
    my_qio = QIO(mf=mf, mc=None, act_space=act_space) 
    my_qio.max_cycle = 20
    my_qio.max_M = bd
    my_qio.step_size = 0.2
    my_qio.thresh = 1e-7
    my_qio.casci_natorb = False
    my_qio.casci_ss_shift = 10.0
    my_qio.casci_max_cycle = 400
    my_qio.tcc_level_shift = 1.0
    my_qio.dump_flags()
    
    # get the FCI wave function in the whole space



    for micro_i in range(tot_iter):

        if micro_i < 1:
            my_qio.tcc_max_cycle = 100
            my_qio.max_cycle = 30
            my_qio.casci_natorb = True
        else:
            my_qio.tcc_max_cycle = 100
            my_qio.max_cycle = 30
            my_qio.casci_natorb = False 

        e_qicas = my_qio.kernel(is_tcc=True,  inactive_indices=inactive_indices,
            mo_coeff=mo_coeff, method='nr')
        mo_coeff = my_qio.mo_coeff.copy()

        E_tccsd[i, micro_i] = my_qio.tcc_e_tot


        entropy_tccsd[i, micro_i] = get_cost_fqi(my_qio.gamma, my_qio.Gamma, inactive_indices)
        #entropy_tccsd[i, micro_i] = get_cost_fqi(gamma, Gamma, inactive_indices)
        np.savetxt('cr2_tccsd_energy_'+basis+'.12in12.1.txt',E_tccsd)
        np.savetxt("cr2_tccsd_entropy_"+basis+".12in12.1.txt", np.asarray(entropy_tccsd))
        #np.savetxt("c2_dmrg_tccsd_qiorb_energy_"+basis+".scan.txt", np.asarray(E_dmrg))
    t1 = time.time()
    print('icas time:',t1-t0)
    np.save('cr2_mo_coeff_'+basis+'.npy', mo_coeff)