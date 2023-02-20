def is_subset(a, b):
	'''
	Given bit rep of a and b,
	returns if a is subset of b.

	a, b - integers
	'''
	return a & b == a
