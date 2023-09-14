from pyscf import gto, scf
from qicas import *

def test_energy():

    n_core = 0                    # number of frozen orbitals
    ne = 12-2*n_core              # number of total electrons
    n_cas = 8                     # number of active orbitals
    n_should_close = 2            # target number of closed orbitals
    # r = [float(sys.argv[-2])]     # list of geometry parameters
    # bd = int(sys.argv[-1])        # max bond dimension for DMRG
    r = [1.2]
    bd = 100
    basis = 'sto6g'               # basis set

    mol = gto.M(atom='C 0 0 0; C 0 0 '+"{:.4}".format(r[0]), 
        basis=basis,spin=0, verbose=1, 
        max_memory=50000,symmetry = False) # mem in MB
    mol.unit = 'A'

    # Run RHF

    mf = scf.RHF(mol)
    mf.kernel()
    
    mo_coeff = copy.deepcopy(mf.mo_coeff)
    no = len(mo_coeff)-n_core


    # Run QICAS and output post-QICAS CASCI energy
    
    active_indices = list(range(n_should_close,n_cas+n_should_close))
    inactive_indices = list(range(n_should_close))+list(range(n_cas+n_should_close,no))
    act_space = (n_cas,ne-2*n_should_close)
    my_qicas = QICAS(mf=mf, mc=None, act_space=act_space) 
    my_qicas.max_cycle = 100
    my_qicas.max_M = bd

    e_qicas,nclosed = my_qicas.kernel(active_indices=active_indices, inactive_indices=inactive_indices,
        mo_coeff=mo_coeff)
    
    assert np.isclose(e_qicas, -75.427831, atol=1e-5)

