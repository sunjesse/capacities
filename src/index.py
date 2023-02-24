import itertools
import numpy as np
from utils import *

class Indexer():
	def __init__(self, nx, ny):
		self.nx = nx
		self.ny = ny
		self.n = self.nx*self.ny

	def get_vec(self, a):
		'''
		a - integer
		'''
		assert a < (1 << self.n)
		
		ret = []
		while a > 0:
			ret = [a & 1] + ret
			a = a >> 1

		ret = [0]*(self.n - len(ret)) + ret 
		return np.array(ret)

	def get_Wb(self, nu=None):
		XY = [self.get_vec(i)[:, np.newaxis] for i in range(1 << self.n)]
		# placeholder marginals for now
		# TODO: implement actual computation of
		# marginals for a given mu
		def marginals():
			raise NotImplementedError

		nus = np.zeros((1 << self.n, 1)) 
		return np.concatenate(XY, axis=-1).T, nus

	def get_ineq(self):
		B = []
		seen = set()
		for b in range(1, 1 << self.n):
			b_vec = self.get_vec(b)
			# since mu is linear, positions where a AND b == 1 cancel out.
			# ==> can just do xor
			subsets = [b^a for a in range(b) if is_subset(a, b)]
			for x in subsets:
				if x in seen: continue
				B.append(self.get_vec(x)[:, np.newaxis])
				seen.add(x)
		return np.concatenate(B, axis=-1).T, np.zeros((len(B), 1))


if __name__ == '__main__':
	idxr = Indexer(3,3)
	B, z = idxr.get_ineq()
	print(B)
	print(B.shape, z.shape)
