# Quantum Information-based Active and General Orbital Optimization (QICAS-QIO)

This project realizes orbital optimization using quantum information tools. It can be used for active space and general orbital optimization.


## Dependencies

'''Python
numpy
scipy
pyscf
block2
'''

## Usage
Please see examples.

Other solvers other than DMRG or TCCSD can be used, as long as you design a wrapper sticking to the standard as in these two, i.e. it should include the following member functions

'''Python
def kernel(mo_coeff)
def make_rdm1()
def make_rdm2()
'''


Please cite the following papers if you use QICAS-QIO 

1. Lexin Ding, Stefan Knecht, Christian Schilling, Quantum Information-Assisted Complete Active Space Optimization (QICAS), J. Phys. Chem. Lett. 14, 49, 11022–11029 (2023)
2. Ke Liao, Lexin Ding, Christian Schilling, Unveiling Intrinsic Many-Body Complexity by Compressing Single-Body Triviality, arXiv preprint arXiv:2402.16841 (2024)

  