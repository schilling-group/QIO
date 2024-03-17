from pyscf import gto, scf, dmrgscf, mcscf
import os,sys,copy
import numpy as np
from qio import *
import time


dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''

def get_casci_energy(mo_coeff, n_cas, n_act_e, r):
    mol = gto.M(atom='C 0 0 0; C 0 0 '+"{:.4}".format(r), 
        basis=basis,spin=0, verbose=1, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'A'

    mf = scf.RHF(mol)
    mf.kernel()
    
    
    mycas = mcscf.CASCI(mf,n_cas,n_act_e)
    etot = mycas.kernel(mo_coeff)[0]
    return etot

n_core = 0                    # number of frozen orbitals
ne = 12-2*n_core              # number of total electrons
n_cas = 8                     # number of active orbitals
n_should_close = 2            # target number of closed orbitals
# r = [float(sys.argv[-2])]     # list of geometry parameters
# bd = int(sys.argv[-1])        # max bond dimension for DMRG
r = 1.243

#for basis in ['sto6g', 'ccpvdz', 'ccpvtz']:
for basis in ['ccpvdz']:
    print(' basis =', basis)
    #for bd in [50, 100, 200, 400]:

    E = np.zeros((3))      # array of output data
    mol = gto.M(atom='C 0 0 0; C 0 0 '+"{:.4}".format(r), 
        basis=basis,spin=0, verbose=1, 
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
    E[1] = etot

    mycas = mcscf.CASCI(mf,n_cas,ne-2*n_should_close)
    etot = mycas.kernel(mf.mo_coeff)[0]
    E[2] = etot

    for bd in [100]:
        print(' bd =', bd)

        # create molecule with desired geometry and basis

        mol = gto.M(atom='C 0 0 0; C 0 0 '+"{:.4}".format(r), 
            basis=basis,spin=0, verbose=1, 
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

        e_qicas,nclosed = my_qicas.kernel(is_tcc=False, inactive_indices=inactive_indices,
            mo_coeff=mo_coeff)
        E[0] = e_qicas
        print(e_qicas)

        np.save('rdm1_'+basis+'_'+str(bd)+'.npy', my_qicas.gamma)
        np.save('rdm2_'+basis+'_'+str(bd)+'.npy', my_qicas.Gamma)
        np.save('mo_coeff_'+basis+'_'+str(bd)+'.npy', my_qicas.mo_coeff)
        t1 = time.time()
        print('icas time:',t1-t0)

        # sampling random unitaries around the optimized one
        # sample random i and j indices between 0 and no
        cost_list = []
        energy_list = []
        mf_mo_coeff = copy.deepcopy(mf.mo_coeff)
        #for n in range(10):
        #    print("Generating random unitary", n)
        #    X = (np.random.rand(no,no)/2-1)/1/(1+49*np.random.rand())
        #    #X = (np.random.rand(no,no)/2-1)*2*np.pi/50
        #    X = X - X.T
        #    U = expm(X)
        #    U_ = np.kron(U,np.eye(2))
        #    gamma = np.einsum('ia,jb,ab->ij',U_,U_, my_qicas.gamma,optimize='optimal')
        #    Gamma = np.einsum('ia,jb,kc,ld,abcd->ijkl',U,U,U,U,my_qicas.Gamma,optimize='optimal')
        #    cost = jacobi_cost_full(gamma, Gamma, inactive_indices)
        #    cost_list.append(cost)
        #    mo_coeff = mf_mo_coeff @ U.T
        #    ecasci = get_casci_energy(mo_coeff, n_cas, ne-2*n_should_close, r)
        #    energy_list.append(ecasci)
        #    print("Cost:", cost)
        #    print("Energy:", ecasci)
        #np.save('cost_list_'+basis+'_'+str(bd)+'.npy', np.asarray([cost_list, energy_list]))


        # The following code runs a HF-CASSCF and HF-CASCI from sratch for comparison

        np.save('c2_e_'+basis+'_'+str(bd)+'.npy',E)




