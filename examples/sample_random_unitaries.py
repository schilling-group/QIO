from pyscf import gto, scf, dmrgscf, mcscf
import os,sys,copy
import numpy as np
from qicas import *
from orb_rot import *
import time


dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''

def get_casci_energy(mol, mo_coeff, n_cas, n_act_e, r):
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
    mol = gto.M(atom='C 0 0 0; C 0 0 '+"{:.4}".format(r), 
            basis=basis,spin=0, verbose=1, 
            max_memory=50000,symmetry = False)
    #mol = gto.M(
    # atom= "O        0.000000    0.000000    0.117790; H        0.000000    0.755453   -0.471161; H        0.000000   -0.755453   -0.471161",
    # basis=basis,spin=0, verbose=1, 
    # max_memory=50000,symmetry = False) # mem in MB
     # mem in MB
    print(' basis =', basis)
    #for bd in [50, 100, 200, 400]:
    E = np.zeros((3))      # array of output data

    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.kernel()
    mo_coeff = mf.mo_coeff.copy()
    no = len(mo_coeff)-n_core

    active_indices = list(range(n_should_close,n_cas+n_should_close))
    inactive_indices = list(range(n_should_close))+list(range(n_cas+n_should_close,no))


    for bd in [100]:
        print(' bd =', bd)

        # create molecule with desired geometry and basis

        mol = gto.M(atom='C 0 0 0; C 0 0 '+"{:.4}".format(r), 
            basis=basis,spin=0, verbose=1, 
            max_memory=50000,symmetry = False) # mem in MB
        #mol = gto.M(
        #atom= "O        0.000000    0.000000    0.117790; H        0.000000    0.755453   -0.471161; H        0.000000   -0.755453   -0.471161",
        #basis=basis,spin=0, verbose=1, 
        #max_memory=50000,symmetry = False) # mem in MB
        mol.unit = 'A'

        # Run RHF

        mf = scf.RHF(mol)
        mf.kernel()

        mo_coeff = copy.deepcopy(mf.mo_coeff)
        no = len(mo_coeff)-n_core

        t0 = time.time()

        gamma0 = np.load('rdm1_'+basis+'_'+str(bd)+'.npy')
        Gamma0 = np.load('rdm2_'+basis+'_'+str(bd)+'.npy')
        qi_mo_coeff = np.load('mo_coeff_'+basis+'_'+str(bd)+'.npy')

        cost_list=[]
        energy_list = []
        for n in range(500):
            print("Generating random unitary", n)
            #X = (np.random.random((no,no))/2-1)/1/(1+200*np.random.rand())
            X = (np.random.random((no,no))/2-1)*2*np.pi/10
            #X = np.zeros((no,no))
            X = X - X.T
            U = expm(X)
            U_ = np.kron(U,np.eye(2))
            gamma = np.einsum('ia,jb,ab->ij',U_,U_, gamma0,optimize='optimal')
            Gamma = np.einsum('ia,jb,kc,ld,abcd->ijkl',U,U,U,U,Gamma0,optimize='optimal')
            cost = jacobi_cost_full(gamma, Gamma, inactive_indices)
            cost_list.append(cost)
            # rotate from optimized orbitals.
            mo_coeff = qi_mo_coeff @ U.T
            ecasci = get_casci_energy(mol, mo_coeff, n_cas, ne-2*n_should_close, r)
            energy_list.append(ecasci)
            print("Cost:", cost)
            print("Energy:", ecasci)
        np.save('cost_list_'+basis+'_'+str(bd)+'_1.npy', np.asarray([cost_list, energy_list]))


        # The following code runs a HF-CASSCF and HF-CASCI from sratch for comparison

        #np.save('c2_e_'+basis+'_'+str(bd)+'.npy',E)




