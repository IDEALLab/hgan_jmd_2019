"""
Creates dataset of SEiEo

Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import numpy as np
from matplotlib import pyplot as plt
#plt.switch_backend('Qt5Agg')

import sys
sys.path.append("../")
from utils import visualize


def build_data(s_points=64, e_points=16):
    
    X = np.load('../results/SEoEi/SEoEi.npy')
    X_permute = np.zeros_like(X)
    X_permute[:, :s_points] = X[:, :s_points]
    X_permute[:, s_points:s_points+e_points] = X[:, -e_points:]
    X_permute[:, -e_points:] = X[:, s_points:s_points+e_points]
    
    directory = '../results/SEiEo'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save('%s/SEiEo.npy' % directory, X_permute)
    
    return X_permute
    
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
    
    