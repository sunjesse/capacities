def is_subset(a, b):
	"""
	Given a, b, returns True if a iff subset of b.
	:type a: int
	:type b: int
	:rtype: bool
	"""
	return a & b == a
