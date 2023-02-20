import numpy as np
import cvxpy as cp
from index import Indexer

dimX = 5
dimY = 5
dim = dimX + dimY
l1 = lambda C, x : cp.sum(cp.abs(C @ x))

indexer = Indexer(dimX, dimY)
W, b = indexer.get_Wb()
M = cp.Variable((dim, 1))
A = np.random.rand(dim, dim)

constraints = [W @ M == b]

print(constraints)
prob = cp.Problem(cp.Minimize(l1(A, M)), constraints)
sol = prob.solve()
print(prob.value, M.value)
