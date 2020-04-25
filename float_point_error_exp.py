import math
import numpy as np
import networkx as nx
from main import get_exp_graphs

# function to run an experiment idenfitying the smallest and largest distance between any two points
# @param Networkx Graph G
# @param String outFile: the file to write experiment results to
def error_exp(G, outFile):
	with open("output_005_approx/error_exp/"+outFile, "w+") as f:
		minimum = min([v[1]['v'].get_x() for v in list(G.nodes(data=True))])
		maximum = max([v[1]['v'].get_x() for v in list(G.nodes(data=True))])
		width = maximum-minimum

		yVals = np.sort([v[1]['v'].get_y() for v in list(G.nodes(data=True))])
		height = min(np.diff(yVals))

		f.write("n,width,height\n")
		f.write(str(len(G.nodes()))+","+str(width)+","+str(height))



def main():
	data_type = 1 #1 for random, 2 for mpeg7, 3 for mnist, 4 for all data types
	exp_list = get_exp_graphs(data_type)
	print(len(exp_list))
	for e in exp_list:
		error_exp(e["G"], e["output_file"])

if __name__=='__main__':main()
