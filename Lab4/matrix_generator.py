import scipy.sparse as scs
import random
import numpy as np


def generate_diagonal_dominance_matrix(k):
    A_k = scs.csr_matrix(([], [], [0] * (k + 1)), shape=(k, k), dtype=np.float64)

    for i in range(k):
        for j in range(k):
            if i != j:
                A_k[i, j] = -1
            else:
                A_k[i, j] = k + 1
    return A_k


def generate_random_matrix(k):
    A = scs.csr_matrix(([], [], [0] * (k + 1)), shape=(k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            A[i, j] = random.randint(-10, 10)
    return A


def generate_vector(A):
    n = A.indptr.size
    x = []
    for i in range(1, n):
        x.append(i)
    return np.dot(A.toarray(), x)


def generate_gilbert(k):
    A = scs.csr_matrix(([], [], [0] * (k + 1)), shape=(k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            A[i, j] = 1 / (i + j + 1)
    return A