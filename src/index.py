from collections import defaultdict
from functools import lru_cache
import itertools
import numpy as np

class Indexer():
	def __init__(self, nx, ny):
		self.nx = nx
		self.ny = ny
		self.n = self.nx + self.ny

	@lru_cache(512)
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

	def get_mat_prod(self):
		XY = [self.get_vec(i)[:, np.newaxis] for i in range(1 << self.n)]
		return np.concatenate(XY, axis=-1).T

if __name__ == '__main__':
	idxr = Indexer(10, 10)
	W = idxr.get_mat_prod()
	print(W)
