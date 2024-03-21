# QIO: Quantum Information-assisted Orbital Optimization

![qio_logo](https://github.com/schilling-group/QIO/assets/79213502/6cca564e-4cd3-47f3-9582-330333fe2b92)

This project realizes orbital optimization using quantum information tools. It encompasses a specific scheme designed for active space optimization (QICAS) and general quantum information-assisted orbital optimization.

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


## How to cite

When using this package for your work, please cite the following two primary references:

**Active space orbital optimization:** 
* [Lexin Ding, Stefan Knecht, Christian Schilling. Quantum Information-Assisted Complete Active Space Optimization (QICAS), J. Phys. Chem. Lett. 14, 49, 11022–11029 (2023)](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.3c02536)

**Orbital optimization for treating both dynamic and static correlation:** 
* [Ke Liao, Lexin Ding, Christian Schilling. Unveiling Intrinsic Many-Body Complexity by Compressing Single-Body Triviality, arXiv preprint arXiv:2402.16841 (2024)](https://arxiv.org/abs/2402.16841)

Optional further references, introduicng and explaining quantum information concepts and tools for quantum chemistry:

* [Lexin Ding, Christian Schilling. Correlation Paradox of the Dissociation Limit: A Quantum Information Perspective. J. Chem. Theory Comput. 16, 7, 4159–4175 (2020)](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00054)
* [Lexin Ding, Sam Mardazad, Sreetama Das, Zoltán Zimborás, Szilard Szálay, Ulrich Schollwöck, Christian Schilling. Concept of orbital entanglement and correlation in quantum chemistry. J. Chem. Theory Comput. 2021, 17, 1, 79–95 (2020)](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00559)
* [Lexin Ding, Stefan Knecht, Zoltán Zimborás, Christian Schilling. Quantum correlations in molecules: from quantum resourcing to chemical bonding. Quantum Sci. Technol. 8 015015 (2022)](https://iopscience.iop.org/article/10.1088/2058-9565/aca4ee/meta)

