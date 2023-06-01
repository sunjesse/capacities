import itertools
import numpy as np
from utils import *
import os

class Indexer():
	def __init__(self, nx, ny):
		self.nx = nx
		self.ny = ny
		self.n = self.nx*self.ny
		self.id_eq = f"./cache/x{nx}_y{ny}_eq.npy"
		self.id_ineq = f"./cache/x{nx}_y{ny}_ineq.npy"

	def get_vec(self, a, sz):
		'''
		:type a: int
		:rtype: np.array, np.array
		'''
		assert a < (1 << sz)
		
		ret = []
		while a > 0:
			ret = [a & 1] + ret
			a = a >> 1

		ret = [0]*(sz - len(ret)) + ret 
		return np.array(ret)

	def get_eq(self, mu, nu):
		'''
		:type mu: np.array
		:type nu: np.array
		:rtype: np.array, np.array
		'''
		mu_dim, nu_dim = mu.shape[0], nu.shape[0]
		load_eq = False #os.path.isfile(self.id_eq)
		
		if load_eq: rows = np.load(self.id_eq)
		else: rows = []

		b = []
		if not load_eq:
			# empty set equals zero equality
			r = np.zeros((1 << self.n))
			r[0] = 1
			rows += [r[np.newaxis, :]]
			b += [[0]]

			# normalized to 1 equality
			r = np.zeros((1 << self.n))
			r[-1] = 1
			rows += [r[np.newaxis, :]]
			b += [[1]]
		
		# G x Y
		# here, blocks of size nu_dim are ones or zeros depending if element x in X
		for x in range(mu_dim):
			# bit mask x
			b += [mu[x]]	
			if load_eq: continue
			r = 0
			if x > 0:
				for z in range(0, self.nx):
					if (1 << z) | x == x: # element z in x
						for j in range(self.ny):
							r += 1 << (z*self.ny+j)
			_row = np.zeros((1 << self.n))
			_row[r] = 1
			rows.append(_row[np.newaxis, :])

		# X x F
		for y in range(nu_dim):
			b += [nu[y]]
			if load_eq: continue
			r = 0
			if y > 0:
				for z in range(0, self.ny):
					if (1 << z) | y == y: # element z in y
						s = z
						while (1 << s) < (1 << self.n):
							r += (1 << s)
							s += self.ny
			_row = np.zeros((1 << self.n))
			_row[r] = 1
			rows.append(_row[np.newaxis, :])

		if not load_eq:
			rows = np.concatenate(rows, axis=0)
			np.save(self.id_eq, rows)

		b = np.array(b, dtype=object)
		return rows, b

	def get_ineq(self):
		'''
		:rtype: np.array, np.array
		'''
		load_ineq = False #os.path.isfile(self.id_ineq)
		if load_ineq:
			B = np.load(self.id_ineq)
			return B, np.zeros((B.shape[0], 1))
			
		B = []
		for b in range(1, 1 << self.n):
			for z in range(0, int(np.log(b))+1):
				if (1 << z) | b == b:
					row = np.zeros((1 << self.n))
					row[b] = 1
					row[b-(1<<z)] = -1
					B += [row[:, np.newaxis]]
		_B = np.concatenate(B, axis=-1).T
		np.save(self.id_ineq, _B)
		return _B, np.zeros((len(B), 1))


if __name__ == '__main__':
	idxr = Indexer(3,3)
	W, b = idxr.get_eq(np.ones((3, 1)), np.ones((3, 1)))
	#B, z = idxr.get_ineq()
	print(W)
	print(b)
	print(W.shape, b.shape)
