import numpy as np
from SE.build_data import filt


def param_ellipse(X):
    a = np.max(X[:,0])
    b = np.max(X[:,1])
    ellipse = np.array([a, b])
    return ellipse

def check_feasibility(X0, X1):
    is_feasibe0 = filt(X0, np.expand_dims(X1, axis=0))[0]
    center1 = np.mean(X1[:-1], axis=0)
    is_feasibe1 = np.linalg.norm(center1) < 1e-2
    is_feasibe = is_feasibe0 and is_feasibe1
    return is_feasibe


if __name__ == '__main__':
    
    X = np.load('../results/SE/SE.npy')
    X0 = X[:,:64]
    X1 = X[:,-64:]
    for i in range(X.shape[0]):
        is_feasibe = check_feasibility(X0[i], X1[i])
        print('{}: {}'.format(i, is_feasibe))
        if not is_feasibe:
            break