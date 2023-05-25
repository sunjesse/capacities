import numpy as np
import cvxpy as cp
from index import Indexer
from mobius import mobius
import cdd

class Experiment():
	def __init__(self, dimX, dimY, N):
		self.dimX = dimX
		self.dimY = dimY
		self.dim = dimX*dimY
		self.indexer = Indexer(dimX, dimY)
		self.N = N
		self.count = 0
	
	def random(self, lp=True, verbose=False, get_poly=False):
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
		if get_poly:
			B_ = []
			for r in B:
				B_ += [r.tolist()]
			B_mat = cdd.Matrix(B_)
			poly = cdd.Polyhedron(B_mat)
			g = poly.get_generators()
			#print(g)

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
		
		'''
		for x in B @ M:
			if abs(x) < 1e-6:
				self.count += 1
		'''
		return mu, nu, M, sol

	def get_percentage_tight(self):
		return self.count/self.N
