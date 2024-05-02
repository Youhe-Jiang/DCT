import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from scipy import stats
from scipy.optimize import fminbound
from scipy.stats import mvn
import sympy as sp
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

def choose_Psi(X, j1, j2 ):
    if len(set(X[:,j1])) == 2 and len(set(X[:,j2])) == 2:
        return Psi_vector(X, j1, j2)
    elif len(set(X[:,j1])) > 2 and len(set(X[:,j2])) == 2:
        return Psi_vector_mixed(X, j1, j2)
    elif len(set(X[:,j1])) == 2 and len(set(X[:,j2])) > 2:
        return Psi_vector_mixed(X, j2, j1)
    else:
        return Psi_con(X, j1, j2)


def choose_Psi_grad(X, j1, j2):
    if len(set(X[:,j1])) == 2 and len(set(X[:,j2])) == 2:
        return Psi_grad(X, j1, j2)
    elif len(set(X[:,j1])) > 2 and len(set(X[:,j2])) == 2:
        return Psi_grad_mixed(X, j1, j2)
    elif len(set(X[:,j1])) == 2 and len(set(X[:,j2])) > 2:
        return Psi_grad_mixed(X, j2, j1)
    else:
        return Psi_con(X, j1, j2)


def _trigger_stable(X, j1, j2):
    if len(set(X[:,j1])) == 2 and len(set(X[:,j2])) == 2:
        return False
    else:
        return True


class Estimation():
    def __init__(self, X_obs):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

    def _calc_h_hat(self,j):
        X = self.X
        ratio = 1 - (X[:,j].sum()/self.n)
        h_hat = stats.norm.ppf(ratio)

        return h_hat

    def _calc_pi_hat_j(self,j):
        X = self.X
        ratio = X[:,j].sum()/self.n
        pi_hat_j = ratio

        return pi_hat_j

    def _calc_pi_hat(self,j1,j2):
        X = self.X
        ratio = (X[:,j1]*X[:,j2]).sum()/self.n
        pi_hat = ratio

        return pi_hat

    # where j1 is the index of continuous, j2 is the index of binary
    def _calc_pi_hat_mixed(self, j1, j2):
        X = self.X 
        ratio = sum(1 for xy in X if xy[j1]>=0 and xy[j2]==1)/self.n
        pi_hat = ratio

        return pi_hat


    def calc_pi_hat_func(self, j1,j2, sigma):
        h_hat_j1 = self._calc_h_hat(j1)
        h_hat_j2 = self._calc_h_hat(j2)
        mat = np.array([[1, sigma], [sigma, 1]], dtype=np.double, ndmin=2)
        # pi_hat = 1- stats.multivariate_normal.cdf(x=[h_hat_j1, h_hat_j2], cov=mat, allow_singular=True)
        mean = [0, 0]  # Means for x and y
        # Upper limits are infinity (representing the tail of the distribution)
        upper_limit = np.array([np.inf, np.inf])
        # Lower limits
        lower_limit = np.array([h_hat_j1, h_hat_j2])
        prob, _ = mvn.mvnun(lower_limit, upper_limit, mean, mat)
        pi_hat = prob

        return pi_hat


    # j2 is the index of the binary variable
    def calc_pi_hat_func_mixed(self, j2, sigma):
        h_hat_j1 = 0 
        h_hat_j2 = self._calc_h_hat(j2)
        mat = np.array([[1, sigma], [sigma, 1]], dtype=np.double, ndmin=2)
        mean = [0, 0]  # Means for x and y
        # Upper limits are infinity (representing the tail of the distribution)
        upper_limit = np.array([np.inf, np.inf])
        # Lower limits
        lower_limit = np.array([h_hat_j1, h_hat_j2])
        prob, _ = mvn.mvnun(lower_limit, upper_limit, mean, mat)
        pi_hat = prob

        return pi_hat        
        

    def _calc_sigma_hat(self, j1, j2):
        # determine X_j1 is continous or not
        if len(set(self.X[:,j1])) == 2 and len(set(self.X[:,j2])) == 2:
            pi_hat = self._calc_pi_hat(j1, j2)
            obj = lambda sigma_hat: (self.calc_pi_hat_func(j1,j2, sigma_hat) - pi_hat)**2
            sigma_hat = fminbound(obj, -1, 1)

        elif len(set(self.X[:,j2])) == 2 and len(set(self.X[:,j1])) > 2:
            pi_hat = self._calc_pi_hat_mixed(j1, j2)
            obj = lambda sigma_hat: (self.calc_pi_hat_func_mixed(j2, sigma_hat) - pi_hat)**2
            sigma_hat = fminbound(obj, -1, 1)

        elif len(set(self.X[:,j1])) == 2 and len(set(self.X[:,j2])) > 2:
            pi_hat = self._calc_pi_hat_mixed(j2, j1)
            obj = lambda sigma_hat: (self.calc_pi_hat_func_mixed(j1, sigma_hat) - pi_hat)**2
            sigma_hat = fminbound(obj, -1, 1)

        # covariance of two continous variables
        else: 
            sigma_hat = np.corrcoef(self.X[:,j1], self.X[:,j2])[0,1]

        return sigma_hat


    def _calc_all_sigma(self): # return all sigma_hat as a matrix
        p = self.p
        sigma_hat_mat = np.zeros((p,p))
        # Parallel to find out the sigma_hat_matrix and return as a matrix
        sigma_hat_mat = Parallel(n_jobs=-1)(delayed(self._calc_sigma_hat)(j1,j2) for j1 in range(p) for j2 in range(p))
        sigma_hat_mat = np.array(sigma_hat_mat).reshape(p,p)
        # np.fill_diagonal(sigma_hat_mat, 1)
        
        return sigma_hat_mat

    
    def _calc_sigma_hat_j(self, j):
        sigma_hat_j = np.zeros((self.p-1,1))
        sigma_hat_j = Parallel(n_jobs=-1)(delayed(self._calc_sigma_hat)(j,j1) for j1 in range(self.p) if j1 != j) 

        return np.array(sigma_hat_j).reshape(-1,1)

class Psi_vector():
    def __init__(self, X_obs, j1, j2):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.j1 = j1
        self.j2 = j2
        self.estimation = Estimation(self.X)
        self.pi_hat_12 = self.estimation._calc_pi_hat(j1,j2)
        self.pi_hat1 = self.estimation._calc_pi_hat_j(j1)
        self.pi_hat2 = self.estimation._calc_pi_hat_j(j2)

    def Psi_theta_1(self, i):
        X = self.X
        j1 = self.j1 
        j2 = self.j2
        pi_hat_12_i = X[i,j1] * X[i,j2]

        psi_theta_1 = pi_hat_12_i - self.pi_hat_12

        return psi_theta_1

    def Psi_theta_2(self, i):
        X = self.X
        j1 = self.j1
        pi_hat1  = X[i, j1] - self.pi_hat1

        return pi_hat1

    def Psi_theta_3(self, i):
        X = self.X
        j2 = self.j2
        pi_hat2  = X[i, j2] - self.pi_hat2

        return pi_hat2    
    
    def Psi_theta_vector(self, i):
        psi_theta1 = self.Psi_theta_1(i)
        psi_theta2 = self.Psi_theta_2(i)
        psi_theta3 = self.Psi_theta_3(i)

        return np.array([psi_theta1, psi_theta2, psi_theta3]).reshape(-1,1)

    # calculate the matrix Psi * Psi.T for ith entry
    def Psi_theta_matrix(self, i):
        psi_theta1 = self.Psi_theta_1(i)
        psi_theta2 = self.Psi_theta_2(i)
        psi_theta3 = self.Psi_theta_3(i)

        psi_theta_vector = np.array([psi_theta1, psi_theta2, psi_theta3]).reshape(-1,1)

        psi_theta_matrix = psi_theta_vector @ psi_theta_vector.T

        return psi_theta_matrix
    
    

class Psi_grad():
    def __init__(self, X_obs, j1, j2):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.j1 = j1
        self.j2 = j2
        self.estimation = Estimation(self.X)
        estimation = self.estimation
        self.pi_hat_12 = estimation._calc_pi_hat(j1,j2)
        self.pi_hat1 = estimation._calc_pi_hat_j(j1)
        self.pi_hat2 = estimation._calc_pi_hat_j(j2)
        self.h_hat1 = estimation._calc_h_hat(j1)
        self.h_hat2 = estimation._calc_h_hat(j2)
        self.sigma_hat = estimation._calc_sigma_hat(j1, j2)

    def Psi_11(self):
        h_hat1 = self.h_hat1
        h_hat2 = self.h_hat2
        sigma_hat = self.sigma_hat
        numerator = np.exp(-(h_hat1**2 - 2*sigma_hat*h_hat1 * h_hat2 + h_hat2**2)/(2*(1- sigma_hat**2)))
        denominator = -2*np.pi * np.sqrt(1-sigma_hat**2)

        return numerator/denominator
    
    def Psi_12(self):
        x = sp.symbols('x')
        h_hat1 = self.h_hat1
        h_hat2 = self.h_hat2
        sigma_hat = self.sigma_hat
        numerator = sp.exp(-(h_hat1**2 - 2*sigma_hat*h_hat1 * x + x**2)/(2*(1- sigma_hat**2)))
        denominator = -2*sp.pi * sp.sqrt(1-sigma_hat**2)
        f = numerator/denominator

        definite_integral = sp.integrate(f,(x, h_hat2, sp.oo))
        
        return -definite_integral
        # return 0


    def Psi_13(self):
        x = sp.symbols('x')
        h_hat1 = self.h_hat1
        h_hat2 = self.h_hat2
        sigma_hat = self.sigma_hat
        numerator = sp.exp(-(h_hat2**2 - 2*sigma_hat*h_hat2 * x + x**2)/(2*(1- sigma_hat**2)))
        denominator = -2*sp.pi * sp.sqrt(1-sigma_hat**2)
        f = numerator/denominator

        definite_integral = sp.integrate(f,(x, h_hat1, sp.oo))

        return -definite_integral
        # return 0
        
    def Psi_21(self):

        return 0
    
    def Psi_22(self):
        h_hat1 = self.h_hat1

        return np.exp(- h_hat1**2 / 2)/np.sqrt(2 * np.pi)
    
    def Psi_23(self):

        return 0
    
    def Psi_31(self):

        return 0
    
    def Psi_32(self):

        return 0
    
    def Psi_33(self):
        h_hat2 = self.h_hat2

        return np.exp(- h_hat2**2 / 2)/np.sqrt(2 * np.pi)
    
    def Psi_grad_matrix(self):
        
        return np.array([
            [self.Psi_11(), self.Psi_12(), self.Psi_13()],
            [self.Psi_21(), self.Psi_22(), self.Psi_23()],
            [self.Psi_31(), self.Psi_32(), self.Psi_33()]
        ]).astype(np.float64)

        
# this should only be only only for i, in practice we need to calculate for all i and sum together
# def psi_theta_vector(psi_vector, psi_grad):

#     return np.linalg.inv(psi_grad) @ psi_vector


# j1 is the index of the continous variable j2 is the index of binary variable
class Psi_vector_mixed():
    def __init__(self, X_obs, j1, j2):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.j1 = j1
        self.j2 = j2
        self.estimation = Estimation(self.X)
        self.pi_hat_12 = self.estimation._calc_pi_hat_mixed(j1,j2)
        self.pi_hat2 = self.estimation._calc_pi_hat_j(j2)   

    def Psi_theta_1(self, i):
        X = self.X
        j1 = self.j1
        j2 = self.j2
        # return 1 if X[i,j1] > 0 and X[i,j2] == 1 else 0
        pi_hat_12_i = 1 if X[i,j1] >= 0 and X[i,j2] == 1 else 0
        psi_hat_12_i = pi_hat_12_i - self.pi_hat_12

        return psi_hat_12_i

    def Psi_theta_2(self, i):
        X = self.X
        j2 = self.j2
        pi_hat2  = X[i, j2] - self.pi_hat2

        return pi_hat2

    def Psi_theta_vector(self, i):
        psi_theta1 = self.Psi_theta_1(i)
        psi_theta2 = self.Psi_theta_2(i)

        return np.array([psi_theta1, psi_theta2]).reshape(-1,1)

    # calculate the matrix Psi * Psi.T for ith entry
    def Psi_theta_matrix(self, i):
        psi_theta1 = self.Psi_theta_1(i)
        psi_theta2 = self.Psi_theta_2(i)

        psi_theta_vector = np.array([psi_theta1, psi_theta2]).reshape(-1,1)

        psi_theta_matrix = psi_theta_vector @ psi_theta_vector.T

        return psi_theta_matrix


# j1 is continuous, j2 is binary
class Psi_grad_mixed():
    def __init__(self, X_obs, j1, j2):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.j1 = j1
        self.j2 = j2
        self.estimation = Estimation(self.X)
        estimation = self.estimation
        self.pi_hat_12 = estimation._calc_pi_hat_mixed(j1,j2)
        self.pi_hat2 = estimation._calc_pi_hat_j(j2)
        self.h_hat2 = estimation._calc_h_hat(j2)
        self.sigma_hat = estimation._calc_sigma_hat(j1, j2)

    def Psi_11(self):
        h_hat1 = 0
        h_hat2 = self.h_hat2
        sigma_hat = self.sigma_hat
        numerator = np.exp(-(h_hat1**2 - 2*sigma_hat*h_hat1 * h_hat2 + h_hat2**2)/(2*(1- sigma_hat**2)))
        denominator = -2*np.pi * np.sqrt(1-sigma_hat**2)

        return numerator/denominator
    
    def Psi_12(self):
        x = sp.symbols('x')
        h_hat1 = 0
        h_hat2 = self.h_hat2
        sigma_hat = self.sigma_hat
        numerator = sp.exp(-(h_hat2**2 - 2*sigma_hat*h_hat2 * x + x**2)/(2*(1- sigma_hat**2)))
        denominator = -2*sp.pi * sp.sqrt(1-sigma_hat**2)
        f = numerator/denominator

        definite_integral = sp.integrate(f,(x, h_hat1, sp.oo))

        return -definite_integral
    
    def Psi_21(self):

        return 0
    
    def Psi_22(self):
        h_hat2 = self.h_hat2

        return np.exp(- h_hat2**2/2)/np.sqrt(2 * np.pi)

    def Psi_grad_matrix(self):
        
        return np.array([
            [self.Psi_11(), self.Psi_12()],
            [self.Psi_21(), self.Psi_22()],
        ]).astype(np.float64)
        

# Continuous case
class Psi_con():
    def __init__(self, X_obs, j1, j2):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.j1 = j1
        self.j2 = j2
        self.j1mean = self.X[:,j1].mean()
        self.j2mean = self.X[:,j2].mean()
        self.estimation = Estimation(self.X)
        self.sigma_hat = np.corrcoef(self.X[:,j1], self.X[:,j2])[0,1]


    def Psi_theta_vector(self, i):
        xj1 = self.X[i, self.j1]
        xj2 = self.X[i, self.j2]
        psi_theta_1 = xj1 * xj2 - self.j1mean * self.j2mean - self.sigma_hat
        
        return psi_theta_1.reshape(1,1)


    def Psi_theta_matrix(self, i):
        xj1 = self.X[i, self.j1]
        xj2 = self.X[i, self.j2]
        psi_theta_1 = (xj1 * xj2 - self.j1mean * self.j2mean - self.sigma_hat) **2
        
        return psi_theta_1.reshape(1,1)


    def Psi_grad_matrix(self):
        
        return np.array([-1]).reshape(1,1)

# gradient will be 1
    def Var(self):
        tmp = 0
        for i in range(self.n):
            tmp = tmp + self.Psi_theta_vector(i)**2
            
        return tmp/self.n
        



class disTest():
    def __init__(self, data, **kwargs):
        self.X = data.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.estimation = Estimation(self.X)
        self.sigma_hat_mat = self.estimation._calc_all_sigma()
        self.grad_all, self.grad_all_inv = self._save_all_grad()
        # self.index_mapping = self.create_upper_triangle_matrix(self.p)
        self.index_mapping = np.arange(self.p*self.p).reshape(self.p, self.p)

    def get_varaince(self, X, j1, j2):
        X = X.copy()
        n = X.shape[0]
        psi_matrix_summation = np.zeros((3,3))
        psi_vector = Psi_vector(X, j1, j2)
        for i in range(n):
            psi_matrix = psi_vector.Psi_theta_matrix(i)
            psi_matrix_summation = psi_matrix + psi_matrix_summation
        psi_matrix_average = psi_matrix_summation/n    
        psi_grad = Psi_grad(X, j1, j2)
        psi_grad_matrix = psi_grad.Psi_grad_matrix()
        variance_vector = np.linalg.inv(psi_grad_matrix) @ psi_matrix_average @ np.linalg.inv(psi_grad_matrix.T)

        return variance_vector[0,0]


    # return a p-1xp-1 matrix using parallel computing
    def get_varaince_all(self, j):
        X_tmp = self.X.copy()
        p = X_tmp.shape[1]
        X_tmp = np.delete(X_tmp, j, axis=1) 
        variance_all = Parallel(n_jobs=-1, backend ='multiprocessing')(delayed(self.get_varaince)(X_tmp, j1, j2) for j1 in range(p-1) for j2 in range(p-1))
        variance_all = np.array(variance_all).reshape(p-1, p-1)
        
        return variance_all


    def get_varaince_whole_mat(self):
        X_tmp = self.X.copy()
        p = X_tmp.shape[1]
        variance_mat = Parallel(n_jobs=-1, backend ='multiprocessing')(delayed(self.get_varaince)(X_tmp, j1, j2) for j1 in range(p) for j2 in range(p))
        variance_mat = np.array(variance_mat).reshape(p, p)

        return variance_mat


    # return a p-1 vector using parallel computing
    def get_varaince_vector(self, j):
        X = self.X.copy()
        p = X.shape[1]
        variance_vector = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(self.get_varaince)(X, j1, j) for j1 in range(p) if j1 != j)
        variance_vector = np.array(variance_vector).reshape(p-1,1)

        return variance_vector

    def parallel_process(self, args):
        i, j = args
        # Assuming these methods can be called independently and are picklable
        Xai_minus_j_minus_j_i = self._calc_xai_all_minusj_minusj_i(i, j)
        Xai_minus_j_j_i = self._calc_xai_all_minusj_j_i(i, j)
        Xai_vector = self._rearrange_xai(Xai_minus_j_minus_j_i, Xai_minus_j_j_i)
        return Xai_vector @ Xai_vector.T
    

    def _save_all_grad(self):
        p = self.p 
        number_grad = int(p*p)
        # grad_all = np.zeros([number_grad, 3, 3])
        # grad_all_inv = np.zeros([number_grad, 3, 3])
        # index = 0 
        grad_all = []
        grad_all_inv = []
        index = 0 
        for j1 in range(p):
            for j2 in range(p):
                psi_grad = choose_Psi_grad(self.X, j1, j2)
                if _trigger_stable(self.X, j1, j2):
                    print('triggered,' )
                    grad_all.append(psi_grad.Psi_grad_matrix() + 1e-4* np.eye(len(psi_grad.Psi_grad_matrix())))
                else:
                    # print(index, 'not triggered')
                    grad_all.append(psi_grad.Psi_grad_matrix())
                # print(index)
                grad_all_inv.append(np.linalg.inv(grad_all[index]))
                index += 1

        return grad_all, grad_all_inv


    def _calc_sigma_mat_minus_j(self, j): 
        # return the sigma matrix without the jth row and column
        return np.delete(np.delete(self.sigma_hat_mat, j, 0), j, 1)
    

    def _calc_sigma_mat_minus_j_inv(self,j):

        return np.linalg.inv(self._calc_sigma_mat_minus_j(j))
    
    
    def _calc_sigma_vector_minusj_j(self, j):

        return self.estimation._calc_sigma_hat_j(j)
    

    def _calc_xai_i(self, X, i, j1, j2):
        p = self.p
        psi_vector_ins = choose_Psi(X, j1, j2)
        psi_vector = psi_vector_ins.Psi_theta_vector(i)
        # if j1> j2:
        #     j1, j2 = j2, j1
        psi_grad_inv = self.grad_all_inv[self.index_mapping[j1, j2]]
        # psi_grad_inv = self.grad_all_inv[j1, j2]
        xai_i = psi_grad_inv @ psi_vector

        return xai_i[0] 
    
    
    # return the matrix of (sigma_hat_minusj_minusj - sigma_minusj_minusj)_i
    def _calc_xai_all_minusj_minusj_i(self, i, j):
        X_tmp = self.X.copy()
        # X_tmp = np.delete(X_tmp, j, axis=1)
        # xai_all = Parallel(n_jobs=-1)(delayed(self._calc_xai_i)(X_tmp, i, j1, j2) for j1 in range(self.p-1) for j2 in range(self.p-1))
        xai_all = Parallel(n_jobs=-1)(delayed(self._calc_xai_i)(X_tmp, i, j1, j2) for j1 in range(self.p) for j2 in range(self.p))
        xai_all_minusj = np.array(xai_all).reshape(self.p, self.p)
        xai_all_minusj = np.delete(np.delete(xai_all_minusj, j, axis=1), j, axis=0)
        # xai_all = np.array(xai_all).reshape(self.p-1, self.p-1)

        # return xai_all
        return xai_all_minusj

    
    # return the vector of sigma_hat_minusj_j - sigma_minusj_j
    def _calc_xai_all_minusj_j_i(self, i, j):
        xai_all = Parallel(n_jobs=-1)(delayed(self._calc_xai_i)(self.X, i, j1, j) for j1 in range(self.p))
        xai_all = np.array(xai_all).reshape(self.p,1)
        xai_all = np.delete(xai_all, j, axis=0)

        return xai_all
    
    
    def _true_beta_from_cov(self, cov, j):
        
        cov = np.array(cov)
        XTX = np.delete(np.delete(cov, j , axis=1), j, axis=0)
        XTY = np.delete(cov[:,j], j, axis=0)
        
        return np.linalg.inv(XTX) @ XTY
    
    
    def _rearrange_xai(self, Xai_minus_j_minus_j, Xai_minus_j_j):

        p = self.p 
        tmp_number = int((p-1) ** 2)
        Xai_minus_j_minus_j_vector = np.zeros(int((p-1)** 2 + p-1))
        index = 0
        for v in range(p-1):
            for q in range(p-1):
                Xai_minus_j_minus_j_vector[index] = Xai_minus_j_minus_j[v,q]
                index = index + 1

        for v in range(p-1):
            Xai_minus_j_minus_j_vector[tmp_number + v] = Xai_minus_j_j[v].item()
            
        return Xai_minus_j_minus_j_vector.reshape(-1,1)
    

    def _get_variance_jk(self, Sigma_minus_j_minus_j,  beta_j, j , k):
        # Rearrange the matrix to be shape with px(p+1)/2 - 1 vector as Xai_minus_j_minus_j is symmetric
        p = self.p
        beta_j_tmp = beta_j.copy()
        
        if j < k:
            k = k-1
            beta_j_tmp[k] = 0
            # print('the term of beta_j is', beta_j_tmp)  
        else :
            beta_j_tmp[k] = 0
            # print('the term of beta_j is', beta_j_tmp)  
            
        tmp_number = int((p-1) ** 2)
        weighting_vector = np.zeros(int((p-1)** 2 + p-1))
        index = 0
        for v in range(p-1):
            for q in range(p-1):
                weighting_vector[index] = -(Sigma_minus_j_minus_j[k,v] * beta_j_tmp[q]).item()
                index += 1 

        for v in range(p-1):
            weighting_vector[tmp_number + v] = Sigma_minus_j_minus_j[k, v]
        
        # print(weighting_vector)
        # rearrange the Xai_minus_j_minus_j to be a vector of length px(p+1)/2 - 1
        Xai_mat_summation = np.zeros([int((p-1)** 2 + p-1), int((p-1)** 2 + p-1)])
        
        with Pool() as pool:
            results = pool.map(self.parallel_process, [(i, j) for i in range(self.n)])
            for result in results:
                Xai_mat_summation += result
        # for i in range(self.n):
        #     Xai_minus_j_minus_j_i = self._calc_xai_all_minusj_minusj_i(i, j)
        #     Xai_minus_j_j_i = self._calc_xai_all_minusj_j_i(i, j)
        #     Xai_vector = self._rearrange_xai(Xai_minus_j_minus_j_i, Xai_minus_j_j_i)
        #     Xai_mat_summation += Xai_vector @ Xai_vector.T
        Xai_mat_avg = Xai_mat_summation/self.n    
        # print(Xai_mat_avg)
        variance = weighting_vector.T @ Xai_mat_avg @ weighting_vector

        return variance
    

    def _inference(self, j, k):
        X = self.X.copy()
        theta = self._calc_sigma_mat_minus_j_inv(j)
        beta_j = theta @ self._calc_sigma_vector_minusj_j(j)
        # beta_j_true = self._true_beta_from_cov(self.cov, j)        
        
        var_jk = self._get_variance_jk(theta, beta_j, j , k)
        if k > j:
            beta_jk = beta_j[k-1]
        else:
            beta_jk = beta_j[k]
            
        z_score = beta_jk*np.sqrt(self.n)/np.sqrt(var_jk)
        # z_score = beta_jk * self.n/np.sqrt(var_jk)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        print("Variance:", var_jk)
        # print("Z-score:", z_score)
        print("P-value:", p_value)
        
        return p_value
    

    # def _independence_inference(self, j1, j2):
    #     X = self.X.copy()
    #     n = self.n 
    #     var_j1_j2 = self.get_varaince(X, j1, j2)
    #     sigma_hat = self.estimation._calc_sigma_hat(j1, j2)
    #     z_score = (sigma_hat*np.sqrt(n) - 0)/np.sqrt(var_j1_j2)
    #     p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    #     print("P-value", p_value)
    #     return p_value
    def _independence_inference(self, j1, j2):

        X = self.X
        n = self.n 
        p = self.p     

        estimator = self.estimation
        
        if len(set(self.X[:,j1])) == 2 and len(set(self.X[:,j2])) == 2:
            psi_matrix_summation = np.zeros((3,3))
            psi_vector = Psi_vector(X, j1, j2)
            for i in range(n):
                psi_matrix = psi_vector.Psi_theta_matrix(i)
                psi_matrix_summation = psi_matrix + psi_matrix_summation
            psi_matrix_average = psi_matrix_summation/n    

            # print(psi_matrix_average, '\n')
            psi_grad = Psi_grad(X, j1, j2)
            psi_grad_matrix = psi_grad.Psi_grad_matrix()
            variance_vector = np.linalg.inv(psi_grad_matrix) @ psi_matrix_average @ np.linalg.inv(psi_grad_matrix.T)
                
            
        elif len(set(self.X[:,j1])) > 2 and len(set(self.X[:,j2])) == 2:
            psi_matrix_summation = np.zeros((2,2))
            psi_vector = Psi_vector_mixed(X, j1, j2)
            for i in range(n):
                psi_matrix = psi_vector.Psi_theta_matrix(i)
                psi_matrix_summation = psi_matrix + psi_matrix_summation
            psi_matrix_average = psi_matrix_summation/n    

            # print(psi_matrix_average, '\n')
            psi_grad = Psi_grad_mixed(X, j1, j2)
            psi_grad_matrix = psi_grad.Psi_grad_matrix()
            # print(psi_grad_matrix, '\n')
            variance_vector = np.linalg.inv(psi_grad_matrix) @ psi_matrix_average @ np.linalg.inv(psi_grad_matrix.T)
            
            
        elif len(set(self.X[:,j1])) == 2 and len(set(self.X[:,j2])) > 2:
            psi_matrix_summation = np.zeros((2,2))
            psi_vector = Psi_vector_mixed(X, j2, j1)
            for i in range(n):
                psi_matrix = psi_vector.Psi_theta_matrix(i)
                psi_matrix_summation = psi_matrix + psi_matrix_summation
            psi_matrix_average = psi_matrix_summation/n
            
            psi_grad = Psi_grad_mixed(X, j2, j1)
            psi_grad_matrix = psi_grad.Psi_grad_matrix()
            psi_grad_matrix_inv = np.linalg.inv(psi_grad_matrix)
            # logging.debug(psi_grad_matrix_inv)
            variance_vector = np.linalg.inv(psi_grad_matrix) @ psi_matrix_average @ np.linalg.inv(psi_grad_matrix.T)
        
        else:
            variance_vector = Psi_con(X, j1, j2).Var()
            variance_vector = variance_vector.reshape(1,1)
            
        # print("The estimated covaraince is ", sigma_hat)
        sigma_hat = estimator._calc_sigma_hat(j1, j2)
        z_score = (sigma_hat*np.sqrt(n) - 0)/np.sqrt(variance_vector[0,0])
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Calculate the p-value for a two-tailed test
        # print("Variance:", variance_vector[0,0], variance_vector[1,1], variance_vector[2,2])
        # print("Z-score:", z_score)
        print("Variance:", variance_vector[0,0])
        print("P-value:", p_value)

        return p_value            
    
