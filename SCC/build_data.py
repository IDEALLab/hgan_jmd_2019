"""
Creates dataset of SCC

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

def gen_circle(center, r, num_points=64):
    x = center[0]
    y = center[1]
    C = [(x + math.cos(2*np.pi/(num_points-1)*i)*r, y + math.sin(2*np.pi/(num_points-1)*i)*r) for i in xrange(num_points)]
    return np.array(C)

def random_circle(superformula, num_points, outside=False):
    idx = np.random.choice(np.arange(1, superformula.shape[0]-1))
    t = superformula[idx+1] - superformula[idx-1]
    if outside:
        t = -t
    t = t.reshape(2,1)
    n = np.dot(np.array([[0.,-1.],[1.,0.]]), t)
    n = n.flatten()/np.linalg.norm(n)
    r = np.random.uniform(0.1, 0.2)
    center = superformula[idx] + r * n
    circle = gen_circle(center, r, num_points)
    return circle

def check_feasibility_sc(superformula, circle):
    
    center = np.mean(circle[:-1], axis=0)
    r = np.mean(np.linalg.norm(circle[:-1]-center, axis=1))
    dist = np.min(np.linalg.norm(center-superformula, axis=1))
    
    # The radius of circle should be equal to the distance from center to superformula
    is_feasible = np.abs(r-dist) < 3e-2
    
    return is_feasible

def check_feasibility(superformula, circles_a, circles_b):
    is_feasible_a = check_feasibility_sc(superformula, circles_a)
    is_feasible_b = check_feasibility_sc(superformula, circles_b)
    is_feasible = is_feasible_a and is_feasible_b
    return is_feasible

def build_data(s_points=64, c_points=16):
    
    n_s = 1000
    n_c = 21
    
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
        count_c = 0
        for j in range(n_c):
            circle_a = random_circle(superformula, num_points=c_points)
            circle_b = random_circle(superformula, num_points=c_points, outside=True)
            is_feasible = check_feasibility(superformula, circle_a, circle_b)
            if is_feasible:
                sc = np.concatenate((superformula, circle_a, circle_b))
                X.append(sc)
                count_c += 1
        print('{} : {}'.format(i, count_c))
        if count_c != 0:
            count_s += 1
    print('Total valid superformulas: {}'.format(count_s))
    
    X = np.array(X)
    np.random.shuffle(X)
    
    directory = '../results/SCC'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save('%s/SCC.npy' % directory, X)
    
    return X
    
if __name__ == "__main__":
    
    s_points = 64
    e_points = 64
    
    X = build_data(s_points, e_points)
    visualize(X[:300, :s_points, :])
    visualize(X[:300, s_points:-e_points, :])
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
    
    