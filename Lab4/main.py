import functions as fun
import numpy as np
import scipy.sparse as scs
import matrix_generator as mg
from tabulate import tabulate

'''
size = []
iters = []
cond = []
eps = []

for i in range(2, 11):
    A = mg.generate_diagonal_dominance_matrix(i * 10)
    a, b, c = fun.jacobi(A.toarray(), 0.001)
    size.append(i * 10)
    eps.append(0.001)
    iters.append(c)
    cond.append(fun.count_condition_number(A.toarray()))

table = {"Size": size, "Epsilon": eps, "cond(A)": cond, "Iterations": iters}
print(tabulate(table, headers="keys", tablefmt="pretty"))
'''

A = mg.generate_gilbert(3)
a, b, c = fun.jacobi(A.toarray(), 0.001)
print("Matrix:", A.toarray())
print("\nEigenvalues: ", a)
print("\nEigenvectors:", b)
