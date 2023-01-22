import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def A_stat(dT):
    """Time invariant matrix A(t/T) = A for DGP 1-12. This will generate stationary features z.

    Args:
        dT (int): number of features

    Returns:
        a: matrix A
    """
    a = np.empty((dT,dT))
    for i in range(dT):
        for j in range(dT):
            a[i,j] = 0.4**(np.abs(i-j)+1)
    return a

def A_localStat(dT):
    """Generate matrix A1 and A2 for DGP 13-14. This will generate LOCAL stationary features z. In the DGP, the time varying matrix A(t/T) is defined as
    A = (1-t/T)*A1 + (t/T)*A2

    Args:
        dT (int): number of features

    Returns:
        a: matrix A1 and A2
    """
    a1 = np.empty((dT,dT))
    a2 = np.empty((dT,dT))
    for i in range(dT):
        for j in range(dT):
            a1[i,j] = 0.2**(np.abs(i-j)+1)
            a2[i,j] = 0.4**(np.abs(i-j)+1)
    return a1, a2

def betaRW(T):
    """Generate random walk for Y

    Args:
        T (int): number of sample

    Returns:
        beta: array of beta coefficients
    """
    delta = np.random.multivariate_normal(np.zeros(4), 1/np.sqrt(T)*np.identity(4), size=T)
    beta = np.zeros((T,4))
    beta[0,:] = [1,2,0.5,0.5]
    for t in range(1,T):
        beta[t] = beta[t-1] + delta[t]
    return beta

def betaBreak(t, T_b):
    """Generate a break at specific time T_b for Y

    Args:
        t (int): a certain time point
        T_b (int): Time of break

    Returns:
        beta: array of beta coefficients
    """
    beta = np.array([-1 if t > T_b else 0 for _ in range(4)])
    return beta

def simulateDGP(A, B, T_b = 150, T_sample = 200, dT = 100, rho = 0.6, b = 0.5, lags = 3):
    """Simulate target Y and regressors Z
    Args:
        A (func): matrix A generator
        B (func): matrix Beta generator
        T_b (int, optional): Time of break. Defaults to 150.
        T (int, optional): Sample size. Defaults to 200.
        dT (int, optional): Number of features. Defaults to 100.
        rho (float, optional): value of coefficient for lagged Y. Defaults to 0.6.
        b (float, optional): intercept for Beta. Defaults to 0.5.

    Returns:
        Y: target variable Y
        z: features z
    """
    T = T_sample + lags
    Y = np.empty(T)
    z = np.empty((T, dT))

    epsilon = np.random.standard_normal(T)
    nu = np.random.multivariate_normal(np.zeros(dT), np.identity(dT), size=T)
    
    if A == A_localStat:
        a1,a2 = A(dT)
    else:
        a = A(dT)
        
    if B == betaRW:
        beta = B(T)
    else:
        pass
        
    Y[0] = epsilon[0]
    z[0] = nu[0]
    
    for t in tqdm(range(1,T)):
        if A == A_localStat:
            a = (1 - t/T)*a1 + (t/T)*a2
        if B == betaBreak:
            beta_t = B(t, T_b)
        else:
            beta_t = beta[t]
            
        Y[t] = rho*Y[t-1] + (b+beta_t).T@z[t-1,0:4] + epsilon[t]
        z[t] = a@z[t-1] + nu[t]
    
    return Y, z

Y, z = simulateDGP(A_stat, betaRW)

plt.plot(z[:,1])        
plt.show()