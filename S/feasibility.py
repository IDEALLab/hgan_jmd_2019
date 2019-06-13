import numpy as np
from S.build_data import check_feasibility


if __name__ == '__main__':
    
    X = np.load('../results/S/S.npy')
    for i in range(X.shape[0]):
        is_feasibe = check_feasibility(X)
        print('{}: {}'.format(i, is_feasibe))
        if not is_feasibe:
            break