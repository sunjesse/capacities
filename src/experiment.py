import numpy as np
import cvxpy as cp
from index import Indexer
from mobius import mobius

class Experiment():
	def __init__(self, dimX, dimY):
		self.dimX = dimX
		self.dimY = dimY
		self.dim = dimX*dimY
		self.indexer = Indexer(dimX, dimY)
	
	def random(self, lp=True, verbose=False):
		# marginal capacities
		mu = np.random.uniform(size=(self.dimX, 1))
		nu = np.random.uniform(size=(self.dimY, 1))
		mu = mu/sum(mu)
		nu = nu/sum(nu)
		
		if verbose:
			print("Solving for: ")
			print(f"	mu := {mu.T}")
			print(f"	nu := {nu.T}")

		W, b = self.indexer.get_eq(mu, nu)
		B, zeros = self.indexer.get_ineq()
		M = cp.Variable((self.dim, 1))

		# A := Mobius transform
		A = mobius(self.dim)
		constraints = [B @ M >= zeros, W @ M == b, np.ones((1, self.dim)) @ M == 1]

		if not lp:
			if verbose: print("Minimizing L_1 norm...")
			l1 = lambda C, x : cp.sum(cp.abs(C @ x))

			prob = cp.Problem(cp.Minimize(l1(A, M)), constraints)
			sol = prob.solve()
			if verbose: print(f"Solution: {M.value} with value {sol}.")

		else:
			if verbose: print("Minimizing LP using L_1 norm trick...")
			c = np.ones((self.dim, 1))
			t = cp.Variable((self.dim, 1))
			
			prob = cp.Problem(cp.Minimize(c.T @ t),
							  constraints + [A@M <= t, A@M >= -t])
			sol = prob.solve()
			if verbose: print(f"Solution: {M.value} with value {sol}.")
		return mu, nu, M, sol
