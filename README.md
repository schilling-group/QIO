# Quantum Information-based Active and General Orbital Optimization (QICAS-QIO)

This project realizes orbital optimization using quantum information tools. It can be used for active space [^1] and general orbital optimization [^2].

## Installation
First clone or download it. Then use 
```pip install .```
to install. 

### Dependencies

```Python
numpy
scipy
pyscf
block2
```

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


Please cite the following papers if you use QICAS-QIO 

[^1]: Lexin Ding, Stefan Knecht, Christian Schilling, Quantum Information-Assisted Complete Active Space Optimization (QICAS), J. Phys. Chem. Lett. 14, 49, 11022–11029 (2023)

[^2]: Ke Liao, Lexin Ding, Christian Schilling, Unveiling Intrinsic Many-Body Complexity by Compressing Single-Body Triviality, arXiv preprint arXiv:2402.16841 (2024)

  
