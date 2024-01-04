import numpy as np
import sys
from scipy.linalg import expm

from entropy import get_cost_fqi

np.set_printoptions(threshold=sys.maxsize)


def gamma_grad_diag(gamma,i,j):
    no = len(gamma)
    # First derivative (d gamma)/(d theta) (diagonal elements) of 1RDM gamma wrt rotational angle theta 
    # between orbital i and j evaluated at theta = 0
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

def act_occ_grad(gamma,i,j,active_indices):
    dg = gamma_grad_diag(gamma,i,j)
    docc = []
    for k in set(active_indices).intersection({i,j}):
        docc.append(dg[2*k]+dg[2*k+1])

    return np.array(docc)



def orb_entr_grad(gamma,Gamma,i,j,inactive_indices, active_indices):
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

        dS.append(-np.dot(np.log(spec[spec>0]),dspec[spec>0]))   # vN entropy

    
    return np.array(dS)

def act_occ_hess(gamma,i,j,active_indices):
    dg = gamma_grad_diag(gamma,i,j)
    ddg = gamma_hess_diag(gamma,i,j)
    ddocc = []
    intersection = set(active_indices).intersection({i,j})
    for k in intersection:
        ddocc.append(ddg[2*k]+ddg[2*k+1])

    return np.array(ddocc)

def orb_entr_hess(gamma,Gamma,i,j,inactive_indices,active_indices):
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


def FQI_grad(gamma,Gamma,inactive_indices, active_indices, mu=0.):
    no = len(Gamma)
    # In shape of X
    dX = np.zeros((no,no))

    for i in range(no):
        for j in range(i):
            dX[i,j] = np.sum(orb_entr_grad(gamma,Gamma,i,j,inactive_indices,active_indices))
            #dX[i,j] += mu*np.sum(act_occ_grad(gamma,i,j,active_indices))

    return dX - dX.T

def FQI_hess(gamma,Gamma,inactive_indices,active_indices, mu=0.):

    # In shape of X
    no = len(Gamma)
    ddX = np.zeros((no,no))

    for i in range(no):
        for j in range(i):
            ddX[i,j] = np.sum(orb_entr_hess(gamma,Gamma,i,j,inactive_indices,active_indices))
            #ddX[i,j] += mu*np.sum(act_occ_hess(gamma,i,j,active_indices))
    
    return ddX + ddX.T

def FQI_display(gamma,Gamma,inactive_indices,verbose=True):
    cost_ = get_cost_fqi(gamma, Gamma, inactive_indices)
    print('FQI cost: ', cost_)

    return cost_
            

def minimize_orb_corr_GD(gamma_,Gamma_,inactive_indices,active_indices,step_size=0.1,thresh=1e-4,
                         max_cycle=100, mu=2., mu_rate=1., target_n_ae=None, noise=1e-3, logger=None):
    
    '''

    Orbital optimization from initial orbitals to QICAS optimized orbitals

    Args:
        gamma (ndarray): initial 1RDM
        Gamma (ndarray): initial 2RDM
        inactive_indices (list): inactive orbital indices
        step_size (float): step size of gradient descent
        thresh (float): threshold of max gradient norm to stop optimization
        max_cycle (int): maximal number of cycles of jacobi rotation during orbital optimization
        mu (float): lagrangian multiplier for the electron number constraint
        noise (float): noise added to the gradient to avoid trapping in local minima
        logger (logger): logger

    Returns:
        rotations (list): history of jacobi rotations (orbital_i, orbital_j, rotational_angle)
        U (ndarray): unitary that transform the initial orbitals to the QICAS-optimized orbitals
        gamma (ndarray): transformed 1RDM
        Gamma (ndarray): transformed 2RDM

    '''

    no = len(Gamma_)

    X = np.zeros((no,no))
    X = X - X.T
    U0 = expm(X)
    gamma0 = gamma_.copy()
    Gamma0 = Gamma_.copy()

    U_tot = U0
    grad = np.ones((no,no))
    
    n=0
    cost_old = get_cost_fqi(gamma0,Gamma0,inactive_indices)
    delta_cost = np.inf
    level_shift_ave = 0.
    level_shift = level_shift_ave

    n_ae = np.sum(np.diag(gamma0)[2*active_indices])*2
    while np.abs(delta_cost) > thresh and n < max_cycle:
        n += 1
        grad = FQI_grad(gamma0,Gamma0,inactive_indices, active_indices)
        #print("Max grad", np.max(abs(grad)))    
        hess = FQI_hess(gamma0,Gamma0,inactive_indices, active_indices)
        min_hess = np.min((hess[np.abs(hess)>1e-12]))
        level_shift = min_hess

        denom = hess-level_shift*np.ones((no,no))
        X = -np.divide(grad, denom, out=np.zeros_like(grad), where=np.abs(denom)>1e-12) * step_size 
        
        U = expm(X)
        U_tot = np.matmul(U,U_tot)
        U_ = np.kron(U,np.eye(2))
        gamma0_ = np.einsum('ia,jb,ab->ij',U_,U_,gamma0,optimize='optimal')
        Gamma0_ = np.einsum('ia,jb,kc,ld,abcd->ijkl',U,U,U,U,Gamma0,optimize='optimal')
        cost = get_cost_fqi(gamma0_,Gamma0_,inactive_indices)
        delta_cost = cost_old - cost

        #n_ae = np.sum(np.diag(gamma0_)[2*active_indices])*2
        #mu += (n_ae-target_n_ae) * mu_rate
        #mu_rate = np.max([mu_rate/2., 1.])
        
        gamma0 = gamma0_.copy()
        Gamma0 = Gamma0_.copy()

        # calculate the electron number in the active space


        cost_old = cost
        if n % 1 == 0:
            if logger is not None:
                logger.info("iteration:"+ str(n) + " max |grad| = "+str(np.max(abs(grad))) + " cost = " +str(cost))
                #logger.info("mu = "+str(mu)+", n_ae = "+str(n_ae))
            else:
                print("iteration", n, "max grad", np.max(abs(grad)), "cost", cost)

    return U_tot, gamma0, Gamma0, mu
