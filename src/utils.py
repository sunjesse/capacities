def is_subset(a, b):
	"""
	Given a, b, returns True if a iff subset of b.
	:type a: int
	:type b: int
	:rtype: bool
	"""
	return a & b == a

def disjoint_subset_pairs(mask):
	"""
	Generate all pairs of disjoint 
	subsets where the union is mask.
	"""
	ret = set()
	for z in range(1, mask):
		if z | mask == mask:
			if z >= mask ^ z:
				ret.add((z, mask^z))
			else: ret.add((mask^z, z))
	return ret

def subset_pairs(mask):
	"""
	Generate all pairs of subsets
	where the union is mask.
	"""
	ret = set()
	for z in range(1, mask):
		for u in range(1, mask):
			if z | u == mask:
				if z >= u: ret.add((z, u))
				else: ret.add((u, z))
	return ret
