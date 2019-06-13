"""
Creates dataset of AH

Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import pyplot as plt
#plt.switch_backend('Qt5Agg')
import math


def points_on_circle(circle, n=100):
    x = circle[0]
    y = circle[1]
    r = circle[2]
    C = [(x + math.cos(2*np.pi/(n-1)*i)*r, y + math.sin(2*np.pi/(n-1)*i)*r) for i in xrange(n)]
    return np.array(C)

def filt(airfoil, circles):
    centers = circles[:,:2]
    radii = circles[:,2]
    # The center should be inside the airfoil
    head = np.argmin(airfoil[:,0])
    ubs = np.interp(centers[:,0], airfoil[range(head,-1,-1),0], airfoil[range(head,-1,-1),1])
    lbs = np.interp(centers[:,0], airfoil[head:,0], airfoil[head:,1])
    ind1 = np.logical_and(centers[:,1]<ubs, centers[:,1]>lbs)
    # All points on the airfoil contour should be outside the circle
    distances = pairwise_distances(airfoil, centers)
    ind2 = np.all(distances-radii.reshape((1,-1))>0.01, axis=0)
    return np.logical_and(ind1, ind2)

def build_data(a_points=64, h_points=16):
    
    # Airfoils
    try:
        A = np.load('airfoil.npy')
    except IOError:
        A = np.load('./AH/airfoil.npy')
    
    # Holes
    lb = np.min(A.reshape((-1,2)),0)
    ub = np.max(A.reshape((-1,2)),0)
    r_min = 0.03
    r_max = (ub[1]-lb[1])*.5
    low = np.append(lb+r_min, r_min)
    high = np.append(ub-r_min, r_max)
    X = []
    count_a = 0
    N = 100000
    for (i, a) in enumerate(A):
        C = np.random.uniform(low, high, size=(N, 3)) # random circles (x0, y0, r)
        ind = filt(a, C)
        count_h = 0
        for j in np.arange(N)[ind]:
            h = points_on_circle(C[j], h_points)
            ah = np.concatenate((a, h))
            X.append(ah)
            count_h += 1
            if count_h == 20:
                break
        print('{} : {}'.format(i, count_h))
        if count_h != 0:
            count_a += 1
    print('Total valid airfoils: {}'.format(count_a))
    
    X = np.array(X)
    np.random.shuffle(X)
    
    directory = '../results/AH'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save('%s/AH.npy' % directory, X)
    
    return X
    
if __name__ == "__main__":
    
    a_points = 64
    h_points = 64
    
    X = build_data(a_points, h_points)
    
    # Visualize
    for i in np.random.randint(1, X.shape[0], size=10):
        plt.figure()
        plt.scatter(X[i,:,0], X[i,:,1], s=20, alpha=.5)
#        plt.xticks([])
#        plt.yticks([])
#        plt.axis('off')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    