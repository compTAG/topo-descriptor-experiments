# FILE FOR MANAGING EXPERIMENTS

from PIL import Image
from utils.vertex import *
from utils.orth_angle import *
from utils.load_datasets import *
import argparse

import subprocess

import time
import pickle
import sys
import math
import os
import copy
import numpy as np
import networkx as nx
from utils.visualize import *
from utils.delta_exp import *
from tqdm import tqdm
from utils.combine_data import rando, mpeg7_mnist, combine_pngs
import rpy2.robjects as robjects
import random
######################################################
##### Functions for experiments ######################
######################################################

def stratify(G):
    fillangmatrix(G.graph["stratum"], len(G.nodes()), list(G.nodes(data=True)))
    arcs = find_arc_lengths(G.graph["stratum"])
    #print(f"\n\n\n{arcs}\n\n\n")
    return G, arcs


# @param networkx Graph G: the graph containing the points
# @param str outFile: string of file name to write the stratum sizes to
# stores the results in designated outfiles
def stratum_experiment(G, arcs, outFile, out_graphs_dir):
    print("Stratum experiment")
    with open(os.path.join(out_graphs_dir, "distribution_exp", outFile), "w+") as f:
        f.write("startv1,startv2,endv1,endv2,length\n")
        f.write("\n".join([(str(arc["start"]["vertex1"]) + "," + str(arc["start"]["vertex2"]) +
                            "," + str(arc["end"]["vertex1"]) + "," + str(arc["end"]["vertex2"]) + "," + str(
                    arc["length"]))
                           for arc in arcs]))


# runs an experiment randomly sampling from the unit sphere and marking off arcs we hit
# @param Graph G: graph to perform experiment on
# @param list arcs: list of arcs for the stratified regions of the sphere for G
# @param list sample_sizes: different numbers of random samples to take
# @param str outFile: string of file name (please include extension) to pickle point clouds to
# stores the results in designated outfiles
def sample_experiment(G, arcs, sample_sizes, outFile, out_graphs_dir):
    print("Sample experiment")
    # open up a file to write the outputs to for this pc size
    with open(os.path.join(out_graphs_dir, "sample_exp", outFile), "w+") as f:
        # we store three values: samples, hits (number of stratum hit), num_stratum (total number of stratum on this graph)
        f.write("n,samples,hits,num_stratum")
        f.write("\n")
        # iterate through each number of samples, each of these loops is an experiment
        for num_samples in sample_sizes:
            for j in range(0, num_samples):
                # take a random sample in radians
                sample = random.uniform(0.0, 2 * math.pi)
                for arc in arcs:
                    # test to see which stratum this sample falls into and update that stratum to designate a hit
                    if ((arc["start"]["location"] < arc["end"]["location"])
                            and (sample >= arc["start"]["location"])
                            and (sample < arc["end"]["location"])):
                        arc["hit"] = 1
                    elif ((arc["start"]["location"] > arc["end"]["location"])
                          and ((sample >= arc["start"]["location"])
                               or (sample < arc["end"]["location"]))):
                        arc["hit"] = 1

            # keep track of total number of hits
            hit_count = sum([arc["hit"] for arc in arcs])
            # reset hits to 0 for next iteration
            for arc in arcs:
                arc["hit"] = 0

            f.write(str(len(G)) + "," + str(num_samples) + "," + str(hit_count) + "," + str(len(arcs)))
            f.write("\n")


# runs an experiment randomly sampling uniformly alon the unit sphere and marking off arcs we hit
# @param Graph G: graph to perform experiment on
# @param list arcs: list of arcs for the stratified regions of the sphere for G
# @param list sample_sizes: different numbers of random samples to take
# @param str outFile: string of file name (please include extension) to pickle point clouds to
# stores the results in designated outfiles
def uniform_sample_experiment(G, arcs, sample_sizes, outFile, out_graphs_dir):
    if len(arcs) < 5000:
        # print("Num arcs: "+str(len(arcs)))
        # open up a file to write the outputs to for this pc size
        # print(os.path.join(out_graphs_dir,"uniform_sample_exp",outFile))
        with open(os.path.join(out_graphs_dir, "uniform_sample_exp", outFile), "w+") as f:
            # we store three values: samples, hits (number of stratum hit), num_stratum (total number of stratum on this graph)
            f.write("n,samples,hits,num_stratum")
            f.write("\n")
            # iterate through each number of samples, each of these loops is an experiment
            for num_samples in sample_sizes:
                increment = (2 * math.pi) / num_samples
                # print("INCREMENT "+str(increment))
                sample = 0
                for j in range(0, num_samples):
                    for arc in arcs:
                        # test to see which stratum this sample falls into and update that stratum to designate a hit
                        if ((arc["start"]["location"] < arc["end"]["location"])
                                and (sample >= arc["start"]["location"])
                                and (sample < arc["end"]["location"])):
                            arc["hit"] = 1
                        elif ((arc["start"]["location"] > arc["end"]["location"])
                              and ((sample >= arc["start"]["location"])
                                   or (sample < arc["end"]["location"]))):
                            arc["hit"] = 1
                    sample += increment

                # keep track of total number of hits
                hit_count = sum([arc["hit"] for arc in arcs])
                # reset hits to 0 for next iteration
                for arc in arcs:
                    arc["hit"] = 0

                # print(str(len(G.nodes()))+","+str(num_samples)+","+str(hit_count)+","+str(len(arcs)))
                f.write(str(len(G.nodes())) + "," + str(num_samples) + "," + str(hit_count) + "," + str(len(arcs)))
                f.write("\n")


# else:
# print("Not running exp, too many arcs: "+str(len(arcs)))

# @param networkx Graph G: graph to run experiments on
# @param list arcs: stratum along the sphere for G
# @param str outFile: string of file name to write results to (see headers in function)
# stores the results in designated outfiles
def smallest_stratum_experiment(G, arcs, outFile, out_graphs_dir):
    with open(os.path.join(out_graphs_dir, "smallest_stratum_exp", outFile), "w+") as f:
        # Add headers to output file
        f.write("n,min_angle,num_stratum,num_needed_stratum,ratio")
        f.write("\r\n")
        min_arc = min([a["length"] for a in arcs])

        num_stratum = math.ceil((2 * math.pi) / min_arc)
        num_needed_stratum = len(arcs)
        num_unneeded_stratum = num_stratum - num_needed_stratum
        ratio = (num_needed_stratum / num_stratum)

        f.write(str(len(G.nodes())) + "," + str(min_arc) + "," + str(num_stratum) + "," + str(
            num_needed_stratum) + "," + str(ratio))
        f.write("\r\n")
	


def overlap_exp(G, arcs, outFile):
    for i in range(0, len(arcs)):
        for j in range(0, len(arcs)):
            if i != j:
                overlap = False
                if ((arcs[i]["start"]["location"] < arcs[i]["end"]["location"])
                        and (arcs[j]["start"]["location"] > arcs[i]["start"]["location"])
                        and (arcs[j]["start"]["location"] < arcs[i]["end"]["location"])):
                    overlap = True
                elif ((arcs[i]["start"]["location"] < arcs[i]["end"]["location"])
                      and (arcs[j]["end"]["location"] > arcs[i]["start"]["location"])
                      and (arcs[j]["end"]["location"] < arcs[i]["end"]["location"])):
                    overlap = True
                elif ((arcs[i]["start"]["location"] > arcs[i]["end"]["location"])
                      and ((arcs[j]["start"]["location"] > arcs[i]["start"]["location"])
                           or (arcs[j]["start"]["location"] < arcs[i]["end"]["location"]))):
                    overlap = True
                elif ((arcs[i]["start"]["location"] > arcs[i]["end"]["location"])
                      and ((arcs[j]["end"]["location"] > arcs[i]["start"]["location"])
                           or (arcs[j]["end"]["location"] < arcs[i]["end"]["location"]))):
                    overlap = True
                if overlap:
                    print("Overlap on graph " + str(outFile))
                    print("i start: " + str(arcs[i]["start"]["location"]) + " end " + str(arcs[i]["end"]["location"]))
                    print("j start: " + str(arcs[j]["start"]["location"]) + " end " + str(arcs[j]["end"]["location"]))
                    sys.exit(1)


######################################################
##### Functions for running different experiments ####
######################################################

# experiment setup for graphs
# @param networkx Graph G: input_graph
# @param string output_file: where to write results
# @param int exp_type: type of experiment to run (specified in main)
# stores the results in outfiles defined below
def exp(G, output_file, exp_type, out_graphs_dir):
    sample_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    G, arcs = stratify(G)
    if exp_type == 1:
        smallest_stratum_experiment(G, arcs, output_file, out_graphs_dir)
    elif exp_type == 2:
        uniform_sample_experiment(G, arcs, sample_sizes, output_file, out_graphs_dir)
    # elif exp_type == 3:
    # Too few goes here
    elif exp_type == 4:
        delta_exp(G, output_file, out_graphs_dir)

    elif exp_type == 5:
        stratum_experiment(copy.deepcopy(G), copy.deepcopy(arcs), output_file)
        sample_experiment(copy.deepcopy(G), copy.deepcopy(arcs), sample_sizes, output_file)
        smallest_stratum_experiment(copy.deepcopy(G), copy.deepcopy(arcs), output_file)
        uniform_sample_experiment(G, arcs, sample_sizes, output_file)


# optional
# draw_graph(G, G.graph["stratum"], output_file)

# a small experiment that prints out the Graph nodes and the stratum for verification
def small_stratum_verification(G, arcs):
    G, arcs = stratify(G)
    for arc in arcs:
        print(arc)
        print("\n")
    for v in list(G.nodes(data=True)):
        print(str(v[1]['v'].get_id()) + " " + str(v[1]['v'].get_x()) + " " + str(v[1]['v'].get_y()))


def stratum_order_exp(G):
    verts = list(G.nodes(data=True))
    for i in range(0, len(verts)):
        print(str(i) + " " + str(verts[i][1]['v'].get_id()))
        print(str(i) + " " + str(verts[i]))


# wrapper class for multiprocessing pool map function
def exp_wrapper(args):
    exp(*args)


def get_exp_graphs(data_type, graphs_dir, out_graphs_dir):
    exp_list = []
    # random experiment
    if data_type == 1 or data_type == 4:
        for filename in os.listdir(os.path.join('graphs', 'random')):
            G = read_graph_from_pickle(os.path.join('graphs', 'random', filename))
            output_file = os.path.join("random", filename[:-8] + ".txt")
            exp_list.append({"G": G, "output_file": output_file})
    # MPEG7 dataset
    if data_type == 2 or data_type == 4:
        for filename in os.listdir(os.path.join(graphs_dir, 'mpeg7')):
            G = read_graph_from_pickle(os.path.join(graphs_dir, 'mpeg7', filename))
            output_file = os.path.join("mpeg7", filename[:-8] + ".txt")
            exp_list.append({"G": G, "output_file": output_file})
    # MNIST
    if data_type == 3 or data_type == 4:
        for filename in os.listdir(os.path.join(graphs_dir, 'mnist')):
            # test_output_file = "mnist/"+filename[:-8]+".txt"
            # if not os.path.exists(out_graphs_dir+"/uniform_sample_exp/"+test_output_file):
            G = read_graph_from_pickle(os.path.join(graphs_dir, 'mnist', filename))
            output_file = os.path.join("mnist", filename[:-8] + ".txt")
            exp_list.append({"G": G, "output_file": output_file})

    # Test experiment
    if data_type == 5:
        # get one random graph
        # G = nx.read_gpickle('graphs/random/RAND_3_1.gpickle')
        # output_file = "TEST_RAND.txt"
        # exp_list.append({"G":G, "output_file":output_file})

        # #get the first MPEG7 file
        # G1 = nx.read_gpickle('graphs/mpeg7/MPEG7_apple-1.gpickle')
        # output_file = "TEST_MPEG7.txt"
        # exp_list.append({"G":G1, "output_file":output_file})

        # get the first MNIST file
        G2 = read_graph_from_pickle('graphs_random/RAND_3_1.gpickle')
        output_file = "random/RAND_3_1.txt"
        exp_list.append({"G": G2, "output_file": output_file})
    return exp_list

def plot_exps(data_type, exp_type, graph_dir, eps):
    if exp_type == 1:
        approx = eps

        rando("smallest_stratum_exp",approx,"angle_stats")
        mpeg7_mnist("mpeg7","smallest_stratum_exp", approx, "angle_stats")
        mpeg7_mnist("mnist","smallest_stratum_exp", approx, "angle_stats")
        
        # Source the R script
        robjects.r['source']("utils/analysis.R")

        # Assign the Python variable to the R variable 'approx'
        robjects.r.assign("approx", approx)

        # Call the R functions
        get_exp_files = robjects.r['get_exp_files']
        perform_smallest_stratum_exp_analysis = robjects.r['perform_smallest_stratum_exp_analysis']

        random = get_exp_files(approx, "smallest_stratum_exp", "random")
        mnist = get_exp_files(approx, "smallest_stratum_exp", "mnist")
        mpeg7 = get_exp_files(approx, "smallest_stratum_exp", "mpeg7")

        perform_smallest_stratum_exp_analysis(random, mnist, mpeg7, approx)
		
		
        combine_pngs('smallest_stratum_exp', approx)
        img = Image.open("figs/smallest_stratum_exp/smallest_stratum_exp_"+approx+".png")
        img.show()

    elif exp_type == 2:
        approx = eps

        rando("uniform_sample_exp",approx,"angle_stats")
        mpeg7_mnist("mpeg7","uniform_sample_exp", approx, "angle_stats")
        mpeg7_mnist("mnist","uniform_sample_exp", approx, "angle_stats")

        # Source the R script
        robjects.r['source']("utils/analysis.R")

        # Assign the Python variable to the R variable 'approx'
        robjects.r.assign("approx", approx)

        # Call the R functions
        get_exp_files = robjects.r['get_exp_files']
        perform_uniform_sample_analysis = robjects.r['perform_uniform_sample_analysis']

        random = get_exp_files(approx, "uniform_sample_exp", "random")
        mnist = get_exp_files(approx, "uniform_sample_exp", "mnist")
        mpeg7 = get_exp_files(approx, "uniform_sample_exp", "mpeg7")

        perform_uniform_sample_analysis(random, mnist, mpeg7, approx)


        combine_pngs('uniform_sample_exp', approx)
        img = Image.open("figs/uniform_sample_exp/uniform_sample_exp_"+approx+".png")
        img.show()
    elif exp_type == 4: 
        r_code = """
        mnist_file <- read.table("output_001_approx/delta_exp/mnist/deltas.txt", header=TRUE, sep=",")
        mpeg7_file <- read.table("output_001_approx/delta_exp/mpeg7/deltas.txt", header=TRUE, sep=",")

        pdf("figs/delta_exp_figs/mnist/mnist_delta_exp_001.pdf")
        par(mar=c(5, 5, 5, 5))
        hist(mnist_file$delta, ylab="Number of Graphs", xlab="size (radians)", main="", family="serif", cex.lab=3, cex.main=2, cex.sub=2, cex.axis=2)
        print("Min delta for MNIST")
        print(min(mnist_file$delta))
        print("Max delta for MNIST")
        print(max(mnist_file$delta))
        print("Total number of graphs")
        print(length(mnist_file$n))
        print(summary(mnist_file$delta))
        dev.off()

        pdf("figs/delta_exp_figs/mnist/mnist_delta_exp_plot_001.pdf")
        par(mar=c(5, 5, 5, 5))
        plot(mnist_file$n, mnist_file$delta, ylab="Delta (radians)",xlab="Vertices",
            main="",
            family="serif",cex.lab=3, cex.main=2,
        cex.sub=2,cex.axis=2)
        dev.off()

        pdf("figs/delta_exp_figs/mpeg7/mpeg7_delta_exp_001.pdf")
        par(mar=c(5, 5, 5, 5))
        hist(mpeg7_file$delta, ylab="Number of Graphs", xlab="size (radians)", main="", family="serif", cex.lab=3, cex.main=2, cex.sub=2, cex.axis=2)
        print("Min delta for MPEG7")
        print(min(mpeg7_file$delta))
        print("Max delta for MPEG7")
        print(max(mpeg7_file$delta))
        print("Total number of graphs")
        print(length(mpeg7_file$n))
        print(summary(mpeg7_file$delta))
        dev.off()

        pdf("figs/delta_exp_figs/mpeg7/mpeg7_delta_exp_plot_001.pdf")
        par(mar=c(5, 5, 5, 5))
        plot(mpeg7_file$n, mpeg7_file$delta, ylab="Delta (radians)",xlab="Vertices",
            main="",
            family="serif",cex.lab=3, cex.main=2,
        cex.sub=2,cex.axis=2)
        dev.off()
        """

        # Execute the R code
        robjects.r(r_code)
        
        

def read_graph_from_pickle(filename):
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    return graph
		

######################################################
##### Main: for setting exp parameters ###############
######################################################

# main function for setting up and executing experiments
if __name__ == "__main__":

	parser = argparse.ArgumentParser(prog='Exp Handler - TMTF',
		description = 'Implements experiments from Too Many, Too Few')
	parser.add_argument('--epsilon' , '-eps', action = 'store',
		choices = ['001','005'],
		dest= 'eps',
		type = str,
		nargs=1,
		required=True)
	parser.add_argument('--experiment' , '-exp', action = 'store',
		choices = [1,2,4,5],
		dest= 'exp',
		type = int,
		nargs=1,
		required=True)
	parser.add_argument('--data' , '-d', action = 'store',
		choices = [1,2,3,4],
		dest= 'data',
		type = int,
		nargs=1,
		required=True)
	args = parser.parse_args()
	eps = args.eps[0]

	# choices include graphs_001_approx and graphs_005_approx
	graphs_dir = "graphs_"+eps+"_approx"
	# same as above but specifies where to write results
	out_graphs_dir = "output_"+eps+"_approx"

	# main function for setting up and executing experiment
	# Set for random experiments only
	random.seed(423652346)
	np.random.seed(423652346)

	#### exp type is:
	#       1 for smallest stratum experiment (smallest_stratum_exp)
	#       2 for a uniform random sample experiment (uniform_sample_exp)
	#       3 for small graphs experiment
	exp_type = args.exp[0]

	#### data is:
	#       1 for random
	#       2 for MPEG7 (classes from PHT paper - Turner et al.)
	#       3 for EMNIST
	#       4 for all three
	data_type = args.data[0]

	exp_list = get_exp_graphs(data_type, graphs_dir, out_graphs_dir)

	# Run the experiments
	for e in tqdm(exp_list, desc="Experiment Progress"):
		exp(e["G"], e["output_file"], exp_type, out_graphs_dir)

	plot_exps(data_type, exp_type, out_graphs_dir, eps)
	print("Complete")


