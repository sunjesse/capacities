import numpy as np
import cvxpy as cp
import argparse
from index import Indexer
from mobius import mobius

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lp', action='store_true')
args = parser.parse_args()

dimX = 3
dimY = 3
dim = dimX*dimY

# marginal capacities
mu = np.ones((dimX, 1))/dimX
nu = np.ones((dimY, 1))/dimY

indexer = Indexer(dimX, dimY)
W, b = indexer.get_eq(mu, nu)
B, zeros = indexer.get_ineq()
M = cp.Variable((dim, 1))

# A := Mobius transform
A = mobius(dim)
constraints = [B @ M >= zeros, W @ M == b, np.ones((1, dim)) @ M == 1]

if not args.lp:
	print("Minimizing L_1 norm...")
	l1 = lambda C, x : cp.sum(cp.abs(C @ x))

	prob = cp.Problem(cp.Minimize(l1(A, M)), constraints)
	sol = prob.solve()
	print(f"Solution: {M.value} with value {sol}.")

else:
	print("Minimizing LP using L_1 norm trick...")
	c = np.ones((dim, 1))
	t = cp.Variable((dim, 1))
	
	prob = cp.Problem(cp.Minimize(c.T @ t),
					  constraints + [A@M <= t, A@M >= -t])
	sol = prob.solve()
	print(f"Solution: {M.value} with value {sol}.")
