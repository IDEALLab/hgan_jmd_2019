"""
Estimates likelihood of generated data using kernel density estimation 

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from utils import mean_err

def precision(X, n_points, feasibility_func):
    n_true = 0
    N = X.shape[0]
    
    for i in range(N):
        X_list = []
        n_points_c = np.cumsum([0,] + n_points)
        for j in range(len(n_points)):
            X_list.append(np.squeeze(X)[i, n_points_c[j]:n_points_c[j+1]])
        n_true += feasibility_func(*X_list)
        
    return float(n_true)/N
    
def ci_prc(n, gen_func, feasibility_func, n_points):
    prcs = np.zeros(n)
    for i in range(n):
        X_gen = gen_func(2000)
        prcs[i] = precision(X_gen, n_points, feasibility_func)
    mean, err = mean_err(prcs)
    return mean, err