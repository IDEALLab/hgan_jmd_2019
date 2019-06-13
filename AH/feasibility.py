import numpy as np
from AH.build_data import filt


def param_circle(X):
    center = np.mean(X[:-1], axis=0)
#    radius = np.linalg.norm(X[0]-center)
    radius = np.max(np.linalg.norm(X-center, axis=1))
    circle = np.append(center, radius)
    return circle

def check_feasibility(X0, X1):
    circles = np.expand_dims(param_circle(X1), axis=0)
    is_feasibe = filt(X0, circles)[0]
    return is_feasibe


if __name__ == '__main__':
    
    X = np.load('../results/AH/AH.npy')
    N = X.shape[0]
    X0 = X[:,:64]
    X1 = X[:,-64:]
    count_infeas = 0
    for i in range(X.shape[0]):
        is_feasibe = check_feasibility(X0[i], X1[i])
        print('{}: {}'.format(i, is_feasibe))
        if not is_feasibe:
            count_infeas += 1
    print('Infeasible: {}%'.format(100*float(count_infeas)/N))