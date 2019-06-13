import numpy as np
from SC.build_data import check_feasibility


if __name__ == '__main__':
    
    X = np.load('../results/SC/SC.npy')
    X0 = X[:,:64]
    X1 = X[:,64:]
    for i in range(X.shape[0]):
        is_feasibe = check_feasibility(X0[i], X1[i])
        print('{}: {}'.format(i, is_feasibe))
        if not is_feasibe:
            break