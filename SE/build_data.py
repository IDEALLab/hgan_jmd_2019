"""
Creates dataset of SE

Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import numpy as np
from matplotlib import pyplot as plt
#plt.switch_backend('Qt5Agg')
import math
from scipy.interpolate import splev, splprep, interp1d
from scipy.integrate import cumtrapz

import sys
sys.path.append("../")
from utils import visualize


def get_sf_params(variables, alpha, beta):
    '''
    alpha : control nonlinearity
    beta : control number of categories
    '''    
    params = []
    for v in variables:
        # v = [s, t]
        # Set [m, n1, n2, n3]
        params.append([4+math.floor(v[0]+v[1])%beta, alpha*v[0], alpha*(v[0]+v[1]), alpha*(v[0]+v[1])])
    return  np.array(params)

def r(phi, m, n1, n2, n3):
    # a = b = 1, m1 = m2 = m
    return ( abs(math.cos(m * phi / 4)) ** n2 + abs(math.sin(m * phi / 4)) ** n3 ) ** (-1/n1)

def interpolate(Q, N, k, D=20, resolution=1000):
    ''' Interpolate N points whose concentration is based on curvature. '''
    res, fp, ier, msg = splprep(Q.T, u=None, k=k, s=1e-6, per=0, full_output=1)
    tck, u = res    
    uu = np.linspace(u.min(), u.max(), resolution)
    x, y = splev(uu, tck, der=0)
    dx, dy = splev(uu, tck, der=1)
    ddx, ddy = splev(uu, tck, der=2)
    cv = np.abs(ddx*dy - dx*ddy)/(dx*dx + dy*dy)**1.5 + D
    cv_int = cumtrapz(cv, uu, initial=0)
    fcv = interp1d(cv_int, uu)
    cv_int_samples = np.linspace(0, cv_int.max(), N)
    u_new = fcv(cv_int_samples)
    x_new, y_new = splev(u_new, tck, der=0)
    return x_new, y_new, fp, ier

def gen_superformula(m, n1, n2, n3, num_points=64):
    
    phis = np.linspace(0, 2*np.pi, num_points*4)#, endpoint=False)
    S = [(r(phi, m, n1, n2, n3) * math.cos(phi), 
          r(phi, m, n1, n2, n3) * math.sin(phi)) for phi in phis]
    S = np.array(S)
    
    # Scale the heights to 1.0
    mn = np.min(S[:,1])
    mx = np.max(S[:,1])
    h = mx-mn
    S /= h
    
    x_new, y_new, fp, ier = interpolate(S, N=num_points, k=3)
    S = np.vstack((x_new, y_new)).T

    return S

def gen_ellipse(a, b, num_points=16):
    
    phis = np.linspace(0, 2*np.pi, num_points)
    E = [(a * math.cos(phi), 
          b * math.sin(phi)) for phi in phis]
    
    return np.array(E)

def filt(superformula, ellipses):
    
    N = ellipses.shape[0]
    R_sf = np.linalg.norm(superformula, axis=-1)
    Rs_sf = np.tile(np.expand_dims(R_sf, axis=0), (N, 1))
    Rs_el = np.linalg.norm(ellipses, axis=-1)
    
    # The radii of ellipse should be smaller than those of the superformula at the same angle
    feas_ind = np.all(Rs_sf > Rs_el, axis=-1)
    
    return feas_ind

def build_data(s_points=64, e_points=16):
    
    n_s = 1000
    n_e = 11
    
    # Superformulas
    vars_sf = np.random.uniform(1.0, 10.0, size=(n_s, 2))
    params = get_sf_params(vars_sf, 1.0, 1)
    superformulas = []
    for param in params:
        try:
            superformula = gen_superformula(param[0], param[1], param[2], param[3], num_points=s_points)
            superformulas.append(superformula)
        except ValueError:
            print('Unable to interpolate.')
    superformulas = np.array(superformulas)
    
    X = []
    count_s = 0
    for (i, superformula) in enumerate(superformulas):
    
        # Ellipses
        vars_e = np.random.uniform(0.05, 0.5, (1000, 2))
        ellipses = []
        for var_e in vars_e:
            ellipse = gen_ellipse(var_e[0], var_e[1], num_points=s_points) # same points as sf because we need to compare them later
            ellipses.append(ellipse)
        ellipses = np.array(ellipses)
        
        feas_ind = filt(superformula, ellipses)
        count_e = 0
        for j in np.arange(len(feas_ind))[feas_ind]:
            ellipse = gen_ellipse(vars_e[j,0], vars_e[j,1], num_points=e_points) # resample
            se = np.concatenate((superformula, ellipse))
            X.append(se)
            count_e += 1
            if count_e == n_e:
                break
        print('{} : {}'.format(i, count_e))
        if count_e != 0:
            count_s += 1
    print('Total valid superformulas: {}'.format(count_s))
    
    X = np.array(X)
    np.random.shuffle(X)
    
    directory = '../results/SE'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save('%s/SE.npy' % directory, X)
    
    return X
    
if __name__ == "__main__":
    
    s_points = 64
    e_points = 64
    
    X = build_data(s_points, e_points)
    visualize(X[:300, :s_points, :])
    visualize(X[:300, -e_points:, :])
    visualize(X[:300, :, :])
    
    # Plot examples
    ind = np.random.randint(1, X.shape[0], size=5)
    for i in ind:
        plt.figure()
        plt.scatter(X[i,:,0], X[i,:,1], s=20, alpha=.5)
#        plt.xticks([])
#        plt.yticks([])
#        plt.axis('off')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    