import numpy as np
import cvxpy as cp
from index import Indexer

dimX = 3
dimY = 3
dim = dimX*dimY

l1 = lambda C, x : cp.sum(cp.abs(C @ x))

indexer = Indexer(dimX, dimY)
W, b = indexer.get_Wb()
B, zeros = indexer.get_ineq()
M = cp.Variable((dim, 1))

# A := Mobius transform
A = np.random.rand(dim, dim)

constraints = [W @ M == b, B @ M >= zeros]

print(constraints)
prob = cp.Problem(cp.Minimize(l1(A, M)), constraints)
sol = prob.solve()
print(prob.value, M.value)
