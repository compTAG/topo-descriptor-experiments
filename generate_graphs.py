from load_datasets import *
from visualize import *
import networkx as nx
import numpy as np
import time
import os


def generate():
	##### For random graph experiments
	# number of pt clouds to generate of each size
	n = 100
	# different point cloud sizes, typically = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
	k = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
	#### size of box (l by m) to generate points in
	l = 10.0
	m = 10.0

	t = time.time()

	#original eps is .005, change according to how close we want the approx
	eps = .001
	graphs_dir = "graphs_no_approx"

	# Random point clouds
	# for g_size in k:
	# 	pcs = generate_point_clouds(n,g_size,l,m)
	# 	for pc in pcs:
	# 		G = pc["pc"]
	# 		index = pc["id"]
	# 		output_file = "RAND_"+str(len(G.nodes()))+"_"+str(index)
	# 		draw_graph(G, G.graph['stratum'], "graphs/random_imgs/"+output_file)
	# 		nx.write_gpickle(G, "graphs/random/"+str(output_file)+".gpickle")
	# 		print(output_file+ ": " +str(time.time() - t)+ "(s)")
	# 		t = time.time()

	# MPEG7 data
	for f in os.listdir('data/mpeg7/'):
		if f.endswith(".gif"):
			# original eps is .005
			# G = get_img_data_approx(get_mpegSeven_img(f),eps)
			G = get_img_data(get_mpegSeven_img(f))
			output_file = "MPEG7_"+str(f[:-4])
			if G !=-1:
				draw_graph(G, G.graph['stratum'], graphs_dir+"/mpeg7_imgs/"+output_file)
				nx.write_gpickle(G, graphs_dir+"/mpeg7/"+str(output_file)+".gpickle")
			else:
				print(output_file + " was too large")
			print(output_file+ ": " +str(time.time() - t)+ "(s)")
			t = time.time()


	# MNIST data
	####
	# classes 0 -> 9 are integers 0->9
	# classes 10 -> 35 are uppercase letters A->Z
	# classes 36 -> 61 are lowercase letters a->z
	classes = range(0,62) #gives us a list [0,1,...,61] whic covers all classes
	samples = 100 #get first 100 samples from each class

	for c in classes:
		images = get_mnist_img(c, samples)
		samp_count = 0
		for img in images:
			# original eps is .005
			# G = get_img_data_approx(img,eps)
			G = get_img_data(img)
			output_file = "MNIST_C"+str(c)+"_S"+str(samp_count)
			if G != -1:
				draw_graph(G, G.graph['stratum'], graphs_dir+"/mnist_imgs/"+output_file)
				nx.write_gpickle(G, graphs_dir+"/mnist/"+str(output_file)+".gpickle")
			else:
				print(output_file + " was too large")
			samp_count+=1
			print(output_file+ ": " +str(time.time() - t)+ "(s)")
			t = time.time()


def main():
	# make sure we have the same seeds as main
	random.seed(423652346)
	np.random.seed(423652346)

	# generate all random graphs
	generate()
if __name__=='__main__':main()



