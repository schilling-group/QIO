# Quantum Information-Assisted Complete Active Space Optimization (QICAS)
** Please always add corresponding test cases when you add new features. **

* orb_rot.py and tool.py contains necessary functions to run QICAS
* c2vdz.py is a demo. To run it, use command line "python3 c2vdz.py [internuclear distance] [bond dimension]".
  
* For S != 0 states: Special modifications to the do_twopdm and save_twopdm_stackblock_format functions in block2main file as well as make_rdm12 in the local dmrgci.py file are needed. The reason is that block2 only output a spin-averaged version of the 2RDM, whereas QICAS requires spin-specific elements of the 2RDM.
* For S = 0 state, no modified block2main and dmrgci.py are needed. Just make sure block2 is running at spin-adapted mode (SU(2) symmetry)
