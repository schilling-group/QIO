
import numpy as np

def gamma_Gamma_prep(dm1,dm2):

    '''
    Prepare the 1- and 2-RDM (splitting 1-RDM into spin parts and fix prefactor of 2-RDM)

    Args:
        dm1 (ndarray): 1RDM from pyscf
        dm2 (ndarray): 2RDM from pyscf

    Returns:
        gamma(ndarray): prepared 1RDM in spin-orbital indices
        Gamma(ndarray): prepared relevant part of the 2RDM in orbital indices and spin (up,down,down,up)
    
    '''

    no = len(dm1)

    Gamma = np.zeros((no,no,no,no))
    gamma = np.zeros((2*no,2*no))


    for a in range(no):
        for b in range(no):
            gamma[2*a,2*b] = dm1[a,b]/2
            gamma[2*a+1,2*b+1] = dm1[a,b]/2
            for c in range(no):
                for d in range(no):
                    Gamma[a,b,c,d] = dm2[a,c,d,b]/2

    return gamma, Gamma