# Quantum Information-assisted Active and General Orbital Optimization (QIO)

This project realizes orbital optimization using quantum information tools. It encompasses a specific scheme designed for active space optimization (QICAS) [^1] and general quantum information-assisted orbital optimization (QIO) [^2].

## Installation
First clone or download it. Then use 
```pip install .```
to install. 

### Dependencies
**Python version 3.11 will break some code in dmrgscf and pyscf, so suggested versions to use are Python3.8, Python3.9, Python3.10.**
```Python
numpy
scipy
pyscf
block2
dmrgscf
```

To use the dmrg solver provided by block2, one needs to install dmrgscf manually. See the [documentation](https://block2.readthedocs.io/en/latest/user/dmrg-scf.html) of block2 on how to install it.

## Usage
Import qio as a module to use all its functions and classes
```Python
import qio
```
- For active space orbital optimization, see c2_dmrg_qicas.py under the example directory.
- For general orbital optimization, see other examples.

Solvers other than the provided DMRG or TCCSD can be used, as long as you design a wrapper sticking to the standard as in these two, i.e. it should include the following member functions

```Python
class Your_Solver:
  def kernel(mo_coeff):
    pass
  def make_rdm1():
    pass
  def make_rdm2():
    pass
```


## How to Cite 

[^1]: Lexin Ding, Stefan Knecht, Christian Schilling, Quantum Information-Assisted Complete Active Space Optimization (QICAS), J. Phys. Chem. Lett. 14, 49, 11022–11029 (2023)

[^2]: Ke Liao, Lexin Ding, Christian Schilling, Unveiling Intrinsic Many-Body Complexity by Compressing Single-Body Triviality, arXiv preprint arXiv:2402.16841 (2024)

  
