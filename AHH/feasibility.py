import numpy as np
from AHH.build_data import filt


def param_circle(X):
    center = np.mean(X[:-1], axis=0)
#    radius = np.linalg.norm(X[0]-center)
    radius = np.max(np.linalg.norm(X-center, axis=1))
    circle = np.append(center, radius)
    return circle

def check_feasibility(X0, X1, X2):
    circles1 = np.expand_dims(param_circle(X1), axis=0)
    circles2 = np.expand_dims(param_circle(X2), axis=0)
    is_feasibe0 = filt(X0, circles1, circles2)[0]
    y1 = np.mean(X1[:-1,1], axis=0)
    y2 = np.mean(X2[:-1,1], axis=0)
    is_feasibe1 = np.abs(y1-y2) < 1e-2 # two centers should lie on a horizontal line
    is_feasibe = is_feasibe0 and is_feasibe1
    return is_feasibe


if __name__ == '__main__':
    
    X = np.load('../results/AHH/AHH.npy')
    N = X.shape[0]
    X0 = X[:,:64]
    X1 = X[:,64:128]
    X2 = X[:,-64:]
    count_infeas = 0
    for i in range(X.shape[0]):
        is_feasibe = check_feasibility(X0[i], X1[i], X2[i])
        print('{}: {}'.format(i, is_feasibe))
        if not is_feasibe:
            count_infeas += 1
    print('Infeasible: {}%'.format(100*float(count_infeas)/N))