import itertools
import numpy as np

class Indexer():
	def __init__(self, nx, ny):
		self.nx = nx
		self.ny = ny
		self.n = self.nx + self.ny

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

	def get_Wb(self):
		XY = [self.get_vec(i)[:, np.newaxis] for i in range(1 << self.n)]
		# placeholder marginals for now
		# TODO: implement actual computation of
		# marginals for a given mu
		mus = np.zeros((1 << self.n, 1)) 
		return np.concatenate(XY, axis=-1).T, mus

if __name__ == '__main__':
	idxr = Indexer(5, 5)
	W = idxr.get_Wb()
	print(W)
