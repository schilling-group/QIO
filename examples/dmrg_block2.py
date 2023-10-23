
import numpy as np
from pyscf.tools import fcidump
from pyscf import gto, scf, mcscf
# prepare fcidump
# fcidump_file = 'FCIDUMP.H2.tc'
# fcidump_file = '../new-fcidump/NEW-FCIDUMP'

# dav_type = "NonHermitian"
# dav_type = "NonHermitianDavidsonPrecond"
# dav_type = "ExactNonHermitian"

def run_dmrg(fcidump_file):
    dav_type = None

    # read fcidump
    with open(fcidump_file, 'r') as f:
        lines = [x.lower().strip() for x in f.readlines()]
        lbrk = [il for il, l in enumerate(lines) if "&end" in l or "/" in l][0]
        k = 'orbsym'
        #orb_sym = [[int(x) for x in (l.split(k)[1].split('=')[1].split(','))] for l in lines if k in l][0]


    with open(fcidump_file, 'r') as f:
        lines = [x.lower().strip() for x in f.readlines()]
        lbrk = [il for il, l in enumerate(lines) if "&end" in l or "/" in l][0]
        keys = {'norb': None, 'nelec': None, 'ms2': None}
        for k in keys:
            keys[k] = [int(l.split(k)[1].split('=')[1].split(',')[0]) for l in lines if k in l][0]
        print(keys)
        n_sites = keys['norb']
        n_elec = keys['nelec']
        spin = keys['ms2']
        h1e = np.zeros((n_sites, n_sites))
        g2e = np.zeros((n_sites, n_sites, n_sites, n_sites))
        ecore = 0
        for l in lines[lbrk + 1:]:
            if len(l.split()) == 0:
                continue
            a, i, j, k, l = l.split()
            i, j, k, l = [int(x) - 1 for x in [i, j, k, l]]
            if i + j + k + l == -4:
                ecore += float(a)
            elif k + l == -2:
                h1e[i, j] = float(a)
                h1e[j, i] = float(a)
            else:
                g2e[i, j, k, l] = float(a)
                g2e[j, i, k, l] = float(a)
                g2e[i, j, l, k] = float(a)
                g2e[j, i, l, k] = float(a)
                g2e[k, l, i, j] = float(a)
                g2e[k, l, j, i] = float(a)
                g2e[l, k, i, j] = float(a)
                g2e[l, k, j, i] = float(a)


    # check hermiticity
    print('h1e non-herm = ', np.linalg.norm(h1e - h1e.transpose(1, 0).conj()))
    print('g2e non-herm = ', np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()))
    print('h1e norm = ', np.linalg.norm(h1e))
    print('g2e norm = ', np.linalg.norm(g2e))

    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    spin = 0

    driver = DMRGDriver(scratch='./tmp', symm_type=SymmetryTypes.SU2,
        n_threads=8, stack_mem=int(40E9))
    driver.initialize_system(n_sites=n_sites, n_elec=n_elec, spin=spin, orb_sym=None)

    idx = driver.orbital_reordering(h1e, g2e)
    print('reordering = ', idx)
    h1e = h1e[idx][:, idx]
    g2e = g2e[idx][:, idx][:, :, idx][:, :, :, idx]

    # build Hamiltonian expression
    b = driver.expr_builder()
    b.add_sum_term("(C+D)0", np.sqrt(2) * h1e)
    b.add_sum_term("((C+(C+D)0)1+D)0", g2e.transpose(0, 2, 3, 1))
    b.add_const(ecore)
    mpo = driver.get_mpo(b.finalize(), iprint=2)

    # run dmrg -- bond dim = 250
    ket = driver.get_random_mps(tag='GS', bond_dim=200, nroots=1)
    bond_dims = [200] * 15 
    noises = [1E-4] * 5 + [1E-4] * 5 + [1E-5] * 5 
    thrds = [1E-7] * 15
    gs_energies = driver.dmrg(mpo, ket, n_sweeps=len(bond_dims) + 10, noises=noises,
        iprint=2, bond_dims=bond_dims, dav_type=dav_type, thrds=thrds)
    if isinstance(gs_energies, float):
        print('DMRG Energies = %20.12f\n' % gs_energies)
    else:
        print('DMRG Energies =', ('%20.12f' * len(gs_energies) + '\n') % tuple(gs_energies))

    driver.finalize()
    return gs_energies

def main():
    r = np.arange(0.8,3.3,0.2)    # list of geometry parameters

    basis = 'ccpvtz'               # basis set


    n_core = 0                    # number of frozen orbitals
    ne = 14-2*n_core              # number of total electrons
    n_cas = 8                     # number of active orbitals
    n_should_close = 3            # target number of closed orbitals

    dmrg_e = []
    for i, r_ in enumerate(r):
        print('r=',r[i])

        # The following code runs a HF-CASSCF and HF-CASCI from sratch for comparison

        mol = gto.M(atom='N 0 0 0; N 0 0 '+"{:.4}".format(r[i]), 
            basis=basis,spin=0, verbose=1, 
            max_memory=50000,symmetry = False) # mem in MB
        mol.unit = 'A'

        mf = scf.RHF(mol)
        mf.kernel()

        mycas = mcscf.CASSCF(mf,n_cas,ne-2*n_should_close)
        mycas.verbose = 4
        mycas.max_cycle_macro = 150
        mycas.natorb = True
        mycas.max_cycle_micro = 50
        mycas.fix_spin_(ss=0)
        etot = mycas.kernel()[0]
        fcidump_file = 'fcidump.'+str(i)
        fcidump.from_mo(mf.mol, fcidump_file, mycas.mo_coeff)
        #e = run_dmrg(fcidump_file, i)
        #dmrg_e.append(e)

if __name__ == "__main__":
    main() 
