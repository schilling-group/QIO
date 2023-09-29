import numpy as np
from scipy.optimize import Bounds
import os,sys
import math
import itertools as it
import copy
import sys
import scipy
from scipy.linalg import qr
from scipy.linalg import expm
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.stats import unitary_group
from scipy.stats import ortho_group
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.optimize import fsolve, root, least_squares, newton, line_search
import random
np.set_printoptions(threshold=sys.maxsize)


def shannon_entr(spec):
    '''
    Shannon entropy of a probability distribution

    Args:
        spec (ndarray): probability distribution
    
    Returns:
        S (float): Shannon entropy of spec
    '''
    S = 0
    spec = np.array(spec[spec>0])
    
    return -np.dot(spec,np.log(spec))


def gamma_grad_diag(gamma,i,j):
    no = len(gamma)
    # First derivative (d gamma)/(d theta) (diagonal elements) of 1RDM gamma wrt rotational angle theta betwee orbital i and j evaluated at theta = 0
    dg = np.zeros(no)
    dg[2*i] = 2*gamma[2*i,2*j]
    dg[2*j] = -dg[2*i]
    dg[2*i+1] = dg[2*i]
    dg[2*j+1] = -dg[2*i]

    return dg

def gamma_hess_diag(gamma,i,j):
    no = len(gamma)
    ddg = np.zeros(no)
    ddg[2*i] = -2*(gamma[2*i,2*i] - gamma[2*j,2*j])
    ddg[2*j] = 2*(gamma[2*i,2*i] - gamma[2*j,2*j])
    ddg[2*i+1] = ddg[2*i]
    ddg[2*j+1] = ddg[2*j]
    return ddg

def Gamma_grad_diag(Gamma,i,j):
    no = len(Gamma)

    # First derivative (d Gamma)/(d theta) (diagonal elements) of 2RDM Gamma wrt rotational angle theta betwee orbital i and j evaluated at theta = 0
    dG = np.zeros(no)
    dG[i] = Gamma[j,i,i,i] + Gamma[i,j,i,i] + Gamma[i,i,j,i] + Gamma[i,i,i,j]
    dG[j] = -Gamma[i,j,j,j] - Gamma[i,j,j,j] - Gamma[j,j,i,j] - Gamma[j,j,j,i]

    return dG

def Gamma_hess_diag(Gamma,i,j):
    no = len(Gamma)
    ddG = np.zeros(no)
    ddG[i] = -4*Gamma[i,i,i,i] + 2*(Gamma[j,j,i,i]+Gamma[j,i,j,i] +Gamma[j,i,i,j]+Gamma[i,j,j,i] + Gamma[i,j,i,j] + Gamma[i,i,j,j])
    ddG[j] = -4*Gamma[j,j,j,j] + 2*(Gamma[j,j,i,i]+Gamma[j,i,j,i] +Gamma[j,i,i,j]+Gamma[i,j,j,i] + Gamma[i,j,i,j] + Gamma[i,i,j,j])

    return ddG


def orb_entr_grad(gamma,Gamma,i,j,inactive_indices):
    dg = gamma_grad_diag(gamma,i,j)
    dG = Gamma_grad_diag(Gamma,i,j)
    dS = []

    for k in set(inactive_indices).intersection({i,j}):

        
        
        spec = np.array([1-gamma[2*k,2*k]-gamma[2*k+1,2*k+1]+Gamma[k,k,k,k],
            gamma[2*k,2*k]-Gamma[k,k,k,k],
            gamma[2*k+1,2*k+1]-Gamma[k,k,k,k],
            Gamma[k,k,k,k]])

        dspec = np.array([-dg[2*k]-dg[2*k+1]+dG[k],
            dg[2*k]-dG[k],
            dg[2*k+1]-dG[k],
            dG[k]])

        

        #spec = np.array([1-gamma[2*k,2*k],gamma[2*k,2*k],1-gamma[2*k+1,2*k+1],gamma[2*k+1,2*k+1]])
        #dspec = np.array([-dg[2*k],dg[2*k],-dg[2*k+1],dg[2*k+1]])

        dS.append(-np.dot(np.log(spec[spec>0]),dspec[spec>0]))   # vN entropy
        #dS.append(-2*np.dot(spec,dspec))

    
    return np.array(dS)

def orb_entr_hess(gamma,Gamma,i,j,inactive_indices):
    dg = gamma_grad_diag(gamma,i,j)
    dG = Gamma_grad_diag(Gamma,i,j)
    ddg = gamma_hess_diag(gamma,i,j)
    ddG = Gamma_hess_diag(Gamma,i,j)
    ddS = []
    
    for k in set(inactive_indices).intersection({i,j}):
        spec = np.array([1-gamma[2*k,2*k]-gamma[2*k+1,2*k+1]+Gamma[k,k,k,k],
            gamma[2*k,2*k]-Gamma[k,k,k,k],
            gamma[2*k+1,2*k+1]-Gamma[k,k,k,k],
            Gamma[k,k,k,k]])

        dspec = np.array([-dg[2*k]-dg[2*k+1]+dG[k],
            dg[2*k]-dG[k],
            dg[2*k+1]-dG[k],
            dG[k]])

        ddspec = np.array([-ddg[2*k]-ddg[2*k+1]+ddG[k],
            ddg[2*k]-ddG[k],
            ddg[2*k+1]-ddG[k],
            ddG[k]])

        ddS.append(-np.dot(1/spec[spec>0],dspec[spec>0]**2)-np.dot(np.log(spec[spec>0]),ddspec[spec>0]))
        #ddS.append(-2*np.dot(spec,ddspec)-2*np.dot(dspec,dspec))

    
    return np.array(ddS)


def FQI_grad(gamma,Gamma,inactive_indices):
    no = len(Gamma)
    # In shape of X
    dX = np.zeros((no,no))

    for i in range(no):
        for j in range(i):
            dX[i,j] = sum(orb_entr_grad(gamma,Gamma,i,j,inactive_indices))

    return dX - dX.T

def FQI_hess(gamma,Gamma,inactive_indices):

    # In shape of X
    no = len(Gamma)
    ddX = np.zeros((no,no))

    for i in range(no):
        for j in range(i):
            ddX[i,j] = sum(orb_entr_hess(gamma,Gamma,i,j,inactive_indices))
    

    return ddX + ddX.T

def FQI_display(gamma,Gamma,inactive_indices,verbose=True):
    no = len(Gamma)
    S = np.zeros(no)
    for k in inactive_indices:
        spec = np.array([1-gamma[2*k,2*k]-gamma[2*k+1,2*k+1]+Gamma[k,k,k,k],
            gamma[2*k,2*k]-Gamma[k,k,k,k],
            gamma[2*k+1,2*k+1]-Gamma[k,k,k,k],
            Gamma[k,k,k,k]])
        S[k] = shannon_entr(spec)
        #S[i] = 1-np.dot(spec,spec)
    

    if verbose == True:
        print('FQI =',sum(S))
    return sum(S)
            





def minimize_orb_corr_GD(gamma_,Gamma_,inactive_indices):
    
    '''

    Orbital optimization from initial orbitals to QICAS optimized orbitals

    Args:
        gamma (ndarray): initial 1RDM
        Gamma (ndarray): initial 2RDM
        active_indices (list): active orbital indices
        inactive_indices (list): inactive orbital indices
        N_cycle (int): maximal number of cycles of jacobi rotation during orbital optimization

    Returns:
        rotations (list): history of jacobi rotations (orbital_i, orbital_j, rotational_angle)
        U (ndarray): unitary that transform the initial orbitals to the QICAS-optimized orbitals
        gamma (ndarray): transformed 1RDM
        Gamma (ndarray): transformed 2RDM

    '''

    no = len(Gamma_)

    X = np.zeros((no,no))
    X = X - X.T
    U = expm(X)
    U_ = np.kron(U,np.eye(2))
    gamma0 = np.einsum('ia,jb,ab->ij',U_,U_,gamma_,optimize='optimal')
    Gamma0 = np.einsum('ia,jb,kc,ld,abcd->ijkl',U,U,U,U,Gamma_,optimize='optimal')
    FQI_display(gamma0,Gamma0,inactive_indices)

    U_tot = U
    grad = np.ones((no,no))
    
    n=0
    while np.amax(abs(grad)) > 0.0001 or n < 10**2:
        n += 1
        grad = FQI_grad(gamma0,Gamma0,inactive_indices)    
        hess = FQI_hess(gamma0,Gamma0,inactive_indices)
        level_shift = 1e-2
        X = -grad/(hess+level_shift*np.ones((no,no)))

        
        U = expm(X)
        U_tot = np.matmul(U,U_tot)
        U_ = np.kron(U,np.eye(2))
        gamma0 = np.einsum('ia,jb,ab->ij',U_,U_,gamma0,optimize='optimal')
        Gamma0 = np.einsum('ia,jb,kc,ld,abcd->ijkl',U,U,U,U,Gamma0,optimize='optimal')
        if n % 10 == 0:
            FQI_display(gamma0,Gamma0,inactive_indices)

    return U_tot, gamma0, Gamma0




def reorder(gamma,Gamma,N_cas,inactive_indices):

    '''
    After orbital rotations, among the inactive orbitals move the orbitals more than 
    halfly occupied to closed (front of list) and the ones less than halfly occupied 
    to virtual (back of list)

    Args:
        gamma (ndarray): initial 1RDM
        Gamma (ndarray): initial 2RDM
        N_cas (int): number of active orbitals
        inactive_indices (list): inactive orbital indices

    Returns:
        rotations (list): sequence of used rotations
        n_closed (int): number of closed orbitals predicted by QICAS
        V (ndarray): permutation matrix that performs the desired reordering 

    '''

    test = 1
    #S1 = orb_corr(gamma,Gamma)
    S1 = np.zeros(len(Gamma))
    N1 = np.zeros(len(Gamma))
    for i in range(len(Gamma)):
        nu = gamma[2*i,2*i]
        nd = gamma[2*i+1,2*i+1]
        N1[i] = nu + nd
        spec = [1-nu,nu,1-nd,nd]
        S1[i] = shannon_entr(spec)
    #print(S1,N1)
    rotations = []
    no = len(S1)
    V = np.eye(no)
    
    while test == 1:
        test = 0
        for i in range(len(S1)-1):
            if S1[i] < S1[i+1]:
                test = 1

                c = S1[i]
                S1[i] = S1[i+1]
                S1[i+1] = c


                c = N1[i]
                N1[i] = N1[i+1]
                N1[i+1] = c

                rotations = rotations + [[i+1,i+2,0]]
                V_ = np.eye(no)
                V_[i,i] = 0
                V_[i+1,i+1] = 0
                V_[i,i+1] = 1
                V_[i+1,i] = 1
                V = np.matmul(V_,V)
    
    
    n_closed = 0
    for j in range(N_cas,len(S1)):
        if N1[j] > 1:
            n_closed += 1
            for i in range(j):
                #print(i)

                c = S1[j-1-i]
                S1[j-1-i] = S1[j-i]
                S1[j-i] = c


                c = N1[j-1-i]
                N1[j-1-i] = N1[j-i]
                N1[j-i] = c

                rotations = rotations + [[j-i,j-i+1,0]]
                V_ = np.eye(no)
                V_[j-1-i,j-1-i] = 0
                V_[j-i,j-i] = 0
                V_[j-1-i,j-i] = 1
                V_[j-i,j-1-i] = 1
                V = np.matmul(V_,V)


    print(S1,N1)
    print('n_closed =',n_closed)
    return rotations, n_closed, V

