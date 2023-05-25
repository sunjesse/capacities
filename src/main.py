import numpy as np
import argparse
from experiment import Experiment
from mobius import composition
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lp', action='store_true')
parser.add_argument('--N', default=1000, type=int,  help="number of trials (default: 1000)")
parser.add_argument('--dimX', default=3, type=int,  help="dimension of X (default: 3)")
parser.add_argument('--dimY', default=3, type=int,  help="dimension of Y (default: 3)")
parser.add_argument('--verbose', action='store_true', help="output verbose (default: False)")
parser.add_argument('--poly', action='store_true', help="generate poly (default: False)")
args = parser.parse_args()

if __name__ == '__main__':
	test = Experiment(dimX=args.dimX,
					  dimY=args.dimY,
					  N=args.N)
	x, y = [], []
	for i in tqdm(range(args.N)):
		mu, nu, M, sol = test.random(lp=args.lp, verbose=args.verbose, get_poly=args.poly)
		mu_comp = composition(mu)
		nu_comp = composition(nu)
		M_comp = composition(M.value)
		x.append(mu_comp+nu_comp)	
		y.append(M_comp)
	fig, ax = plt.subplots(1, 1, figsize=(10, 5))
	plt.scatter(x, y)
	ax.set_xlabel("comp_norm(mu) + comp_norm(nu)")
	ax.set_ylabel("comp_norm(M)")
	plt.savefig(fname=f"./results/{datetime.now()}_N{args.N}_dimX{args.dimX}_dimY{args.dimY}.png")
	print("Saved figure!")
	print(f"{test.get_percentage_tight()}% of inequalities are tight.")
