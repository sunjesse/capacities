import itertools
import numpy as np
from utils import *

class Indexer():
	def __init__(self, nx, ny):
		self.nx = nx
		self.ny = ny
		self.n = self.nx*self.ny

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

		rows = []
		b = []
		# G x Y
		for i in range(1 << mu_dim):
			v = self.get_vec(i, sz=mu_dim)
			_row = np.zeros((self.n))
			for j in range(self.n):
				if v[j//nu_dim] == 1:
					_row[j] = 1
			b.append(v.T @ mu)
			rows.append(_row[np.newaxis, :])
		
		# X x F
		for i in range(1 << nu_dim):
			v = self.get_vec(i, sz=nu_dim)
			_row = np.zeros((self.n))
			for j in range(self.n):
				if v[j % mu_dim] == 1:
					_row[j] = 1
			b.append([v.T @ nu])
			rows.append(_row[np.newaxis, :])

		
		rows = np.concatenate(rows, axis=0)
		b = np.array(b, dtype=object)
		return rows, b

	def get_ineq(self):
		'''
		:rtype: np.array, np.array
		'''
		B = []
		seen = set()
		for b in range(1, 1 << self.n):
			b_vec = self.get_vec(b, sz=self.n)
			# since mu is linear, positions where a AND b == 1 cancel out.
			# ==> can just do xor
			subsets = [b^a for a in range(b) if is_subset(a, b)]
			for x in subsets:
				if x in seen: continue
				B.append(self.get_vec(x, sz=self.n)[:, np.newaxis])
				seen.add(x)
		return np.concatenate(B, axis=-1).T, np.zeros((len(B), 1))


if __name__ == '__main__':
	idxr = Indexer(3,3)
	W, b = idxr.get_eq(np.ones((3, 1)), np.ones((3, 1)))
	#B, z = idxr.get_ineq()
	print(W)
	print(b)
	print(W.shape, b.shape)

'''
 (mu_2, mu_1)
 (1,2,3), (A,B,C)
 (1,A), (1,B), (1, C), (2,A), (2, B), (2, C), (3,A), (3,B), (3, C)
m_0		m_1		m_2		m_3		m_4		m_5	   m_6	 m_7	m_8

Let G = {1, 3}
{1, 3} x {A,B} = (1, A), (1, B), (3, A), (3, B)  = m_5, m_4, m_1, m_0 = mu_2 + mu_1

Let F = {A, C}, {}
{1,2,3} x {A, C} = (1, A), (1, C), (2, A), (2, C), (3, A), (3, C) = m_0, m_2, m_3, m_5, m_6, m_8
'''
