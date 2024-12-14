import numpy as np
from sklearn.datasets import make_spd_matrix



def positive_definite_diagonal_kp_matrix():
    A = np.diag(np.full(6, 0.5))
    X = np.zeros(shape=(6, 6))
    print(A)

    is_pos_def = np.all(np.linalg.eigvals(A) > 0)
    if is_pos_def:
        return A
    return X



def positive_definite_diagonal_ki_matrix():
    A = np.diag(np.full(6, 50))
    X = np.zeros(shape=(6, 6))
    print(A)

    is_pos_def = np.all(np.linalg.eigvals(A) > 0)
    if is_pos_def:
        return A
    return X
