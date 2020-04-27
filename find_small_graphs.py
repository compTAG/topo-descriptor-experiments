import networkx as nx
import os
from orth_angle import *
import math

graphs_dir = "graphs_005_approx"
output_dir = "output_005_approx"

def get_mnist():
	exp_list=[]
	for filename in os.listdir(graphs_dir+'/mnist/'):
		G = nx.read_gpickle(graphs_dir+'/mnist/' + filename)
		output_file = "mnist/"+filename[:-8]+".txt"
		exp_list.append({"G":G, "output_file":output_file})
	return exp_list

def get_mpeg7():
	exp_list=[]
	for filename in os.listdir(graphs_dir+'/mpeg7/'):
		G = nx.read_gpickle(graphs_dir+'/mpeg7/' + filename)
		output_file = "mpeg7/"+filename[:-8]+".txt"
		exp_list.append({"G":G, "output_file":output_file})
	return exp_list

def small_n_exp(exp_list):
	small_n_list = []
	for e in exp_list:
		G = e["G"]
		if len(G.nodes()) < 5:
			small_n_list.append((e["output_file"], len(G.nodes())))
	return small_n_list

def large_n_exp(exp_list):
	n_list = []
	for e in exp_list:
		G = e["G"]
		if len(G.nodes()) < 72:
			n_list.append((e["output_file"], len(G.nodes())))
	return n_list

def main():
	exp_list_mpeg7 = get_mpeg7()
	print(len(exp_list_mpeg7))
	# mpeg7_small_n = small_n_exp(exp_list_mpeg7)
	print("MPEG7 Results")
	# for n in mpeg7_small_n:
	# print n
	print(len(large_n_exp(exp_list_mpeg7)))

	exp_list_mnist = get_mnist()
	print(len(exp_list_mnist))
	# mnist_small_n = small_n_exp(exp_list_mnist)
	print("MNIST Results")
	# for n in mnist_small_n:
	# 	print n
	print(len(large_n_exp(exp_list_mnist)))

if __name__ == '__main__':main()