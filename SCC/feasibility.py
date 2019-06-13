import numpy as np
from SCC.build_data import check_feasibility


if __name__ == '__main__':
    
    X = np.load('../results/SCC/SCC.npy')
    X0 = X[:,:64]
    X1 = X[:,64:128]
    X2 = X[:,-64:]
    for i in range(X.shape[0]):
        is_feasibe = check_feasibility(X0[i], X1[i], X2[i])
        print('{}: {}'.format(i, is_feasibe))
        if not is_feasibe:
            break