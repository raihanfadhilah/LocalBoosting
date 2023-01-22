import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from simulated_data import *
from scipy.optimize import minimize
import statsmodels.api as sm

def kernel(x, bw, tkernel = "Unif", N = 1):
  x =  x/(N * bw)
  value = np.zeros(len(x))
  if (tkernel == "Gaussian"):
    value = np.exp(-0.5 * x**2)/2.506628274631000241612355239340104162693023681640625 #sqrt(2*pi)
  
  if(tkernel == "Epa"):
    index = (x >= -1) & (x <= 1)
    value[index] = .75 * (1 - x[index]**2) #12/11
    
  if(tkernel == "Unif"):
    index = (x >= -1) & (x <= 1)
    print(index)
    value[index] = 1 #12/11
    
  return value
  

class LLBoost():
    def __init__(self, y, X, b, h, nu, MT) -> None:
        """_summary_

        Args:
            y (_type_): _description_
            X (_type_): Regressors (Yt-1, Yt-2, Yt-3, Zt-1, Zt-2, Zt-3)
            weights (_type_): _description_
            h (_type_): _description_
            P (_type_): _description_
        """
        self.X = X
        self.y = y
        self.h = h
        self.T = np.shape(X)[0]
        self.nreg = np.shape(X)[1] 
        self.b = b
        self.nu = nu
        self.MT = MT
        self.F0 = None
        self.thetaSm = np.zeros((self.nreg, self.MT))

    def estimateThetaSm(self, weights, resid, Z):
        min_wsse = np.inf
        Sm = 0
        thetaSm = 0
        for j in range(self.nreg):
            wls = sm.WLS(resid, Z[:,j], weights = weights)
            wls_model = wls.fit()
            theta_j = wls_model.params[0]
            wsse = wls_model.centered_tss
            if wsse < min_wsse:
                min_wsse = wsse
                Sm = j
                thetaSm = theta_j
        theta_m = np.zeros(self.nreg)
        theta_m[Sm] = thetaSm
        
        return theta_m
        
    def inSampleFit(self,U):
        grid = np.arange(1,self.T+1)/self.T
        Z = np.hstack(self.X, self.X*(grid-u)[:, np.newaxis])
        Zf = Z[:(self.T-self.h),]
        Yf = self.y[1:(self.T-self.h),]       
    
    def fit(self, u, y, Z, kern):

        w = (np.arange(1,len(Z)+1)/len(Z)) - u
        weights = kernel(x = w, bw=self.b, kernel=kern)

        Fm = np.zeros((self.T, self.MT))
        self.F0 = self.T**(-1) * weights@Yf
        Fm[:,0] = np.repeat(self.F0, self.T)
        
        for m in range(self.MT):
            U = self.y - Fm[:,m]
            self.thetaSm[:,m] = estimateThetaSm(weights, U, Z)
         
        
    def predict(self, newdata):
        for m in range(self.MT):
            
        
test = np.array([[1,2,3],[4,5,6]])

test*np.array([10,100])[:, np.newaxis]
T_sample = 200
lags = 3
y, X = simulateDGP(A_stat, betaRW, T_sample=T_sample, lags = lags)

index = np.arange(lags,(T_sample + lags))
Y0 = y[index]
Y1 = y[index-1]
Y2 = y[index-2]
Y3 = y[index-3]
X1 = X[index-1,:]
X2 = X[index-2,:]
X3 = X[index-3,:]
reg = np.column_stack([Y1,Y2,Y3,X1,X2,X3])
w = (np.arange(1,(len(reg[:,0])+1))/len(reg[:,0]))-0.2
weights = kernel(x = w, bw=0.1, tkernel="Gaussian")

plt.plot(weights)
plt.show()

