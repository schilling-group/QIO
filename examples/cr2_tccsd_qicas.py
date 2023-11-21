from pyscf import gto, scf, dmrgscf, mcscf, cc, fci
from pyscf.tools import fcidump
import os, copy
import numpy as np
from qicas import QICAS, prep_rdm12
from solver.gradient import get_cost_fqi
import time
from tccsd import make_no, make_tailored_ccsd
from dmrg_block2 import run_dmrg

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''


n_core = 0                    # number of frozen orbitals
ne = 24-2*n_core              # number of total electrons
n_cas = 12                     # number of active orbitals
n_should_close = 6            # target number of closed orbitals
n_act_e = ne-2*n_should_close # number of active electrons
# r = [float(sys.argv[-2])]     # list of geometry parameters
# bd = int(sys.argv[-1])        # max bond dimension for DMRG
r = np.arange(3.8,4.0,0.2)    # list of geometry parameters
#r =[2.0]
r = [2.4]

bd = 100
tot_iter = 20
E_casci = np.zeros((len(r), 3+tot_iter))      # array of output data
E_tccsd = np.zeros((len(r), tot_iter))      # array of output data
entropy_tccsd = np.zeros((len(r), tot_iter))      # array of output data
E_dmrg = np.zeros((len(r), tot_iter))      # array of output data
basis = '631g'               # basis set


for i in range(len(r)):
    print('r=',r[i])
    E_casci[i,0] = r[i]

    # The following code runs a HF-CASSCF and HF-CASCI from sratch for comparison
    
    mol = gto.M(atom='Cr 0 0 0; Cr 0 0 '+"{:.4}".format(r[i]), 
        basis=basis,spin=0, verbose=1, 
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
    
    
    
    #mol = gto.M(atom='N 0 0 0; N 0 0 '+"{:.4}".format(r[i]), 
    #    basis=basis,spin=0, verbose=1, 
    #    max_memory=50000,symmetry = False) # mem in MB
    #mol.unit = 'A'
    #mf = scf.RHF(mol)
    #mf.kernel()
    mycasci = mcscf.CASCI(mf,n_cas,ne-2*n_should_close)
    mycasci.verbose = 4
    mycasci.natorb = True
    shift = 20
    mycasci.fcisolver.max_cycle = 400
    mycasci.fix_spin_(ss=0, shift=shift)
    etot = mycasci.kernel(mf.mo_coeff)[0]

   
    # casci energy
    E_casci[i,2] = etot
    # create molecule with desired geometry and basis

    #mol = gto.M(atom='N 0 0 0; N 0 0 '+"{:.4}".format(r[i]), 
    #    basis=basis,spin=0, verbose=4, 
    #    max_memory=50000,symmetry = False) # mem in MB
    #mol.unit = 'A'

    # Run RHF

    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.kernel()
    
    mo_coeff = copy.deepcopy(mycasci.mo_coeff)
    no = len(mo_coeff)-n_core

    t0 = time.time()

    # Run QICAS and output post-QICAS CASCI energy
    #active_indices = list(range(n_should_close,n_cas+n_should_close))
    #inactive_indices = list(range(n_should_close))+list(range(n_cas+n_should_close,no))
    active_indices = []
    inactive_indices = list(range(no))
    act_space = (n_cas,ne-2*n_should_close)
   
    my_qicas = QICAS(mf=mf, mc=None, act_space=act_space) 
    my_qicas.max_cycle = 20
    my_qicas.max_M = bd
    my_qicas.step_size = 0.5
    my_qicas.thresh = 1e-6
    my_qicas.casci_natorb = False
    my_qicas.casci_ss_shift = shift
    my_qicas.casci_max_cycle = 400
    my_qicas.tcc_level_shift = 0.5
    my_qicas.dump_flags()
    
    # get the FCI wave function in the whole space



    for micro_i in range(tot_iter):

        if micro_i < 1:
            my_qicas.tcc_max_cycle = 0
            my_qicas.max_cycle = 10
        else:
            my_qicas.tcc_max_cycle = 50
            my_qicas.max_cycle = 30

        e_qicas = my_qicas.kernel(is_tcc=True,  inactive_indices=inactive_indices,
            mo_coeff=mo_coeff, method='nr')
        mo_coeff = my_qicas.mo_coeff.copy()

        # construct TCCSD natural orbitals
        #mc = mcscf.CASCI(mf, n_cas, n_act_e)
        ##mc.frozen = self.n_core
        #mc.verbose = mf.verbose
        #mc.canonicalization = True 
        #mc.sorting_mo_energy = True
        #mc.fix_spin_(ss=0)
        #mc.tol = 1e-8
        ##mc.natorb = self.tcc_casci_natorb
        #mc.kernel(mo_coeff.copy())
        #tcc = cc.CCSD(mf, mo_coeff=mo_coeff.copy())

        #tcc, t1, t2 = make_tailored_ccsd(tcc, mc)
        #tcc.verbose = mf.verbose
        #tcc.kernel()
        
        # dump fcidump
        #fcidump.from_mo(mf.mol, 'fcidump', mo_coeff)
        ## do a fixed-bond dimension DMRG calculation
        #e_dmrg = run_dmrg(fcidump_file='fcidump')
        #E_dmrg[i, micro_i] = e_dmrg

        #dm1 = tcc.make_rdm1()
        #dm2 = tcc.make_rdm2()
        #mo_coeff = make_no(dm1, mo_coeff)

        #E_tccsd[i, micro_i] = tcc.e_tot
        E_tccsd[i, micro_i] = my_qicas.tcc_e_tot


        #gamma, Gamma = prep_rdm12(dm1,dm2)

        entropy_tccsd[i, micro_i] = get_cost_fqi(my_qicas.gamma, my_qicas.Gamma, inactive_indices)
        #entropy_tccsd[i, micro_i] = get_cost_fqi(gamma, Gamma, inactive_indices)
        np.savetxt('cr2_tccsd_energy_'+basis+'.12in12.1.txt',E_tccsd)
        np.savetxt("cr2_tccsd_entropy_"+basis+".12in12.1.txt", np.asarray(entropy_tccsd))
        #np.savetxt("c2_dmrg_tccsd_qiorb_energy_"+basis+".scan.txt", np.asarray(E_dmrg))
    t1 = time.time()
    print('icas time:',t1-t0)



print('r\tHF-CAS(8,8)\tCASSCF(8,8)\tQICAS-CASCI')
print(E_casci)
print('r\tTCCSD')
print(E_tccsd)




