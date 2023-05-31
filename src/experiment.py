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

	def gen_normalized_additive(self, dim):
		mu = np.zeros((1 << self.dimX))
		for i in range(self.dimX):
			mu[1<<i] = np.random.uniform()
		for i in range(1, 1 << self.dimX):
			# power of two, skip as we already calculated
			if (i & (i-1) == 0): continue
			for j in range(self.dimX):
				if (1 << j) | i == i: # element j is in i
					mu[i] += mu[1 << j]
		mu = mu/mu[-1] # normalize
		return mu[:, np.newaxis]	
	
	def random(self, lp=True, verbose=False, get_poly=False):
		# marginal capacities
		# normalized additive
		mu = self.gen_normalized_additive(self.dimX)
		nu = self.gen_normalized_additive(self.dimY)
		if verbose:
			print("Solving for: ")
			print(f"	mu := {mu.T}")
			print(f"	nu := {nu.T}")

		W, b = self.indexer.get_eq(mu, nu)
		B, zeros = self.indexer.get_ineq()
		M = cp.Variable((1 << self.dim, 1))

		# A := Mobius transform
		A = mobius(1 << self.dim)
		print(B.shape, W.shape)
		constraints = [B @ M >= zeros, W @ M == b]
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
			sol = prob.solve(solver=cp.SCIPY)
			if verbose: print(f"Solution: {M.value} with value {sol}.")

		else:
			if verbose: print("Minimizing LP using L_1 norm trick...")
			c = np.ones((self.dim, 1))
			t = cp.Variable((self.dim, 1))
			
			prob = cp.Problem(cp.Minimize(c.T @ t),
							  constraints + [A@M <= t, A@M >= -t])
			sol = prob.solve(solver=cp.SCIPY)
			if verbose: print(f"Solution: {M.value} with value {sol}.")
		
		'''
		for x in B @ M:
			if abs(x) < 1e-6:
				self.count += 1
		'''
		return mu, nu, M, sol

	def get_percentage_tight(self):
		return self.count/self.N
