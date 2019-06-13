import numpy as np
from SEoEi.build_data import filt_se


def param_ellipse(X):
    a = np.max(X[:,0])
    b = np.max(X[:,1])
    ellipse = np.array([a, b])
    return ellipse
    
def filt_ee(X_o, X_i):
    ellipse_o = param_ellipse(X_o)
    ellipse_i = param_ellipse(X_i)
    is_feasibe = np.all(ellipse_o > ellipse_i)
    return is_feasibe

def check_feasibility(X0, X1, X2):
    is_feasibe0 = filt_se(X0, np.expand_dims(X2, axis=0))[0]
    is_feasibe1 = filt_ee(X2, X1)
    center1 = np.mean(X1[:-1], axis=0)
    center2 = np.mean(X2[:-1], axis=0)
    is_feasibe2 = np.linalg.norm(center1) < 1e-2
    is_feasibe3 = np.linalg.norm(center2) < 1e-2
    is_feasibe = is_feasibe0 and is_feasibe1 and is_feasibe2 and is_feasibe3
    return is_feasibe


if __name__ == '__main__':
    
    X = np.load('../results/SEiEo/SEiEo.npy')
    X0 = X[:,:64]
    X1 = X[:,64:128]
    X2 = X[:,-64:]
    for i in range(X.shape[0]):
        is_feasibe = check_feasibility(X0[i], X1[i], X2[i])
        print('{}: {}'.format(i, is_feasibe))
        if not is_feasibe:
            break