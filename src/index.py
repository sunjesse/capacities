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
		self.load_eq = os.path.isfile(self.id_eq)
		self.load_ineq = os.path.isfile(self.id_ineq)

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
		
		if self.load_eq: rows = np.load(self.id_eq)
		else: rows = []

		b = []
		# G x Y
		for i in range(1 << mu_dim):
			v = self.get_vec(i, sz=mu_dim)
			b.append(v.T @ mu)
			
			if not self.load_eq:
				_row = np.zeros((self.n))
				for j in range(self.n):
					if v[j//nu_dim] == 1:
						_row[j] = 1
				rows.append(_row[np.newaxis, :])
		
		# X x F
		for i in range(1 << nu_dim):
			v = self.get_vec(i, sz=nu_dim)
			b.append([v.T @ nu])

			if not self.load_eq:
				_row = np.zeros((self.n))
				for j in range(self.n):
					if v[j % nu_dim] == 1:
						_row[j] = 1
				rows.append(_row[np.newaxis, :])

		if not self.load_eq:
			rows = np.concatenate(rows, axis=0)
			np.save(self.id_eq, rows)

		b = np.array(b, dtype=object)
		return rows, b

	def get_ineq(self):
		'''
		:rtype: np.array, np.array
		'''
		if self.load_ineq:
			B = np.load(self.id_ineq)
			return B, np.zeros((B.shape[0], 1))
			
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

'''
 (mu_2, mu_1)
 (1,2,3), (A,B,C)
 (1,A), (1,B), (1, C), (2,A), (2, B), (2, C), (3,A), (3,B), (3, C)
m_0		m_1		m_2		m_3		m_4		m_5	   m_6	 m_7	m_8

Let G = {1, 3}
{1, 3} x {A,B} = (1, A), (1, B), (3, A), (3, B)  = m_5, m_4, m_1, m_0 = mu_2 + mu_1

v = '11'
Let F = {A, C}, {} 111111
{1,2,3} x {A, C} = (1, A), (1, C), (2, A), (2, C), (3, A), (3, C) = m_0, m_2, m_3, m_5, m_6, m_8
'''
