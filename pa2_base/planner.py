import numpy as np
import matplotlib.pyplot as plt
import argparse
if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--mdp', type=str)
	parser.add_argument('--algorithm', type=str)
	args = parser.parse_args()
	mdp_file = args.mdp
	algo = args.algorithm
