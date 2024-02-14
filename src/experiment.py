import numpy as np
import cvxpy as cp
from index import Indexer
from mobius import mobius
from utils import subset_pairs, disjoint_subset_pairs
import cdd

class Experiment():
    def __init__(self, dimX, dimY, N):
        self.dimX = dimX
        self.dimY = dimY
        self.dim = dimX*dimY
        self.indexer = Indexer(dimX, dimY)
        self.N = N
        self.count = 0

    def get_random_capacity(self, dim, type="na"):
        mu = np.zeros((1 << dim))
        for i in range(dim):
            mu[1<<i] = np.random.uniform()

        # additive
        if type == "na":
            for i in range(1, 1 << dim):
                # power of two, skip as we already calculated
                if (i & (i-1) == 0): continue
                for j in range(dim):
                    if (1 << j) | i == i: # element j is in i
                        mu[i] += mu[1 << j]

        if type == "sa":
            for i in range(1, 1 << dim):
                if (i & (i-1) == 0): continue
                ss = disjoint_subset_pairs(i)
                dp = 0
                for a, b in ss:
                    dp = max(dp, mu[a]+mu[b])
                mu[i] = dp + np.random.uniform()
        
        if type == "sm":
            for i in range(1, 1 << dim):
                if (i & (i-1) == 0): continue
                ss = subset_pairs(i)
                dp = 0
                for a, b in ss:
                    dp = max(dp, mu[a] + mu[b] - mu[a & b])
                mu[i] = dp + np.random.uniform()

        mu = mu/mu[-1] # normalize
        return mu[:, np.newaxis]    
    
    def random(self, lp=True, verbose=False, get_poly=False, type="sm"):
        # marginal capacities
        # normalized additive
        mu = self.get_random_capacity(self.dimX, type=type)
        nu = self.get_random_capacity(self.dimY, type=type)
        if verbose:
            print("Solving for: ")
            print(f"    mu := {mu.T}")
            print(f"    nu := {nu.T}")

        W, b = self.indexer.get_eq(mu, nu)
        B, zeros = self.indexer.get_ineq()
        M = cp.Variable((1 << self.dim, 1))

        # A := Mobius transform
        A = mobius(1 << self.dim)
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
