# FILE FOR MANAGING EXPERIMENTS

from vertex import *
from orth_angle import *
from load_datasets import *
import random
import time
import pickle
import sys
import math
import os
import copy
import numpy as np
import networkx as nx
import experiment
from visualize import *

###############################
####### Global constants ######
###############################

# modify for that approximation type for emnist and mpeg7
# choices include graphs_001_approx and graphs_005_approx
graphs_dir = "graphs_001_approx"
# same as above but specifies where to write results
out_graphs_dir = "output_001_approx"

# NOTE THAT THERE ARE MORE VARIABLES TO SET IN main()

######################################################
##### Functions for experiments ######################
######################################################

######################################################
##### Functions for running different experiments ####
######################################################

# experiment setup for graphs
# @param networkx Graph G: input_graph
# @param string output_file: where to write results
# @param int exp_type: type of experiment to run (specified in main)
# stores the results in outfiles defined below
def exp(G,output_file,exp_type,out_graphs_dir):
	sample_sizes=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
	G, arcs = stratify(G)
	if exp_type == 1:
		stratum_experiment(G,arcs,output_file,out_graphs_dir)
	elif exp_type == 2:
		sample_experiment(G,arcs,sample_sizes,output_file,out_graphs_dir)
	elif exp_type == 3:
		smallest_angle_experiment(G,arcs,output_file,out_graphs_dir)
	elif exp_type == 4:
		uniform_sample_experiment(G,arcs,sample_sizes,output_file,out_graphs_dir)
	elif exp_type == 5:
		stratum_experiment(copy.deepcopy(G),copy.deepcopy(arcs),output_file)
		sample_experiment(copy.deepcopy(G),copy.deepcopy(arcs),sample_sizes,output_file)
		smallest_angle_experiment(copy.deepcopy(G),copy.deepcopy(arcs),output_file)
		uniform_sample_experiment(G,arcs,sample_sizes,output_file)
	#optional
	# draw_graph(G, G.graph["stratum"], output_file)

# a small experiment that prints out the Graph nodes and the stratum for verification
def small_stratum_verification(G, arcs):
	G, arcs = stratify(G)
	for arc in arcs:
		print arc
		print "\n"
	for v in list(G.nodes(data=True)):
		print(str(v[1]['v'].get_id()) + " " +str(v[1]['v'].get_x()) + " "+ str(v[1]['v'].get_y()))

def stratum_order_exp(G):
	verts = list(G.nodes(data=True))
	for i in range(0, len(verts)):
		print(str(i) + " " + str(verts[i][1]['v'].get_id()))
		print(str(i) + " " + str(verts[i]))

# wrapper class for multiprocessing pool map function
def exp_wrapper(args):
   exp(*args)


def get_exp_graphs(data_type,graphs_dir,out_graphs_dir):
	exp_list = []
        manager = experiment.PathManager()

	# random experiment
	if data_type == 1 or data_type == 4:
            exp_list = manager.random_paths()

	# MPEG7 dataset
	if data_type == 2 or data_type == 4:
            exp_list = manager.mpeg_paths()

	# MNIST
	if data_type == 3 or data_type == 4:
            exp_list = manager.mnist_paths()

	# Test experiment
	if data_type == 5:
		#get one random graph
		# G = nx.read_gpickle('graphs/random/RAND_3_1.gpickle')
		# output_file = "TEST_RAND.txt"
		# exp_list.append({"G":G, "output_file":output_file})

		# #get the first MPEG7 file
		# G1 = nx.read_gpickle('graphs/mpeg7/MPEG7_apple-1.gpickle')
		# output_file = "TEST_MPEG7.txt"
		# exp_list.append({"G":G1, "output_file":output_file})

		#get the first MNIST file
		G2 = nx.read_gpickle('graphs_random/RAND_3_1.gpickle')
		output_file = "random/RAND_3_1.txt"
		exp_list.append({"G":G2, "output_file":output_file})
	return exp_list


######################################################
##### Main: for setting exp parameters ###############
######################################################

# main function for setting up and executing experiments
if __name__ == "__main__":
	start = time.time()
	# Set for random experiments only
	random.seed(423652346)
	np.random.seed(423652346)


	# G = get_exp_graphs(5)

	#### exp type is:
	#				1 for stratification experiment (distribution_exp)
	#				2 for random sample experiment (sample_exp)
	#				3 for smallest angle experiment (smallest_angle_exp)
	#				4 for a uniform random sample experiment (uniform_sample_exp)
	#				5 for all four exps
	exp_type = 4
	#### data is:
	#				1 for random
	#				2 for MPEG7 (classes from PHT paper - Turner et al.)
	#				3 for EMNIST
	#				4 for all three
	#				5 for test
	data_type = 4

	exp_list = get_exp_graphs(data_type,graphs_dir,out_graphs_dir)

	# G,arcs = stratify(exp_list[0]["G"])
	# print G.graph["stratum"]
	# for a in arcs:
	# 	print a
	# for n in G.nodes(data=True):
	# 	print str(n[1]['v'].get_id()) + " " +str(n[1]['v'].get_x()) + " " + str(n[1]['v'].get_y())

	# sys.exit(1)

	# stratum_order_exp(exp_list[0]["G"])

	# Run the experiments
	# p.map(exp_wrapper, [(e["G"],e["output_file"],exp_type) for e in exp_list])
	counter = 1
	for e in exp_list:
		print("Graph "+str(counter)+" of "+str(len(exp_list)))
		exp(e["G"], e["output_file"], exp_type, out_graphs_dir)
		counter+=1

	print("Execution time: "+str(time.time() - start)+"(s)")


