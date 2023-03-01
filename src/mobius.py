import numpy as np
import math
from utils import is_subset
from functools import lru_cache

@lru_cache(128)
def hamming(n):
      """
      :type n: int
      :rtype: int
      """
      n = str(bin(n))
      c = 0
      for i in n:
         if i == "1":
            c += 1
      return c

def mobius(n):
	# compute mobius transform of set function f on a.
	# a is an integer / bitmask.
	A = np.zeros((n, n))
	for i in range(n):
		for j in range(i+1):
			if is_subset(j, i):
				A[i][j] = (-1)**hamming(i^j)
	return A
