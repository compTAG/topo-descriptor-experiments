from utils.load_datasets import *
from utils.visualize import *
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
	eps= .001
	graphs_dir = "graphs_001_approx" 

	#Random point clouds
	for g_size in k:
	 pcs = generate_point_clouds(n,g_size,l,m)
	 for pc in pcs:
	 	G = pc["pc"]
	 	index = pc["id"]
	 	output_file = "RAND_"+str(len(G.nodes()))+"_"+str(index)
	 	draw_graph(G, G.graph['stratum'], "graphs/random_imgs/"+output_file)
	 	nx.write_gpickle(G, "graphs/random/"+str(output_file)+".gpickle")
	 	print(output_file+ ": " +str(time.time() - t)+ "(s)")
	 	t = time.time()


	#### FOR MPEG7, MAKE SURE RAT-09 IS NOT IN THE DATA SET
	unused_mpeg7 = []
	# MPEG7 data
	for f in os.listdir('data/mpeg7/'):
		if f.endswith(".gif"):
			# original eps is .005
			print("Starting on graph: "+str(f))
			G, ret = get_img_data_approx(get_mpegSeven_img(f),eps,0)
			# G = get_img_data(get_mpegSeven_img(f))
			output_file = "MPEG7_"+str(f[:-4])
			draw_graph(G, G.graph['stratum'], graphs_dir+"/mpeg7_imgs/"+output_file)
			if ret != -2 and ret !=-1 and ret != 0:
				nx.write_gpickle(G, graphs_dir+"/mpeg7/"+str(output_file)+".gpickle")
			else:
				unused_mpeg7.append((output_file, ret))
				print(output_file + " was not used")
			print(output_file+ ": " +str(time.time() - t)+ "(s)")
			t = time.time()
	with open(graphs_dir+"/unused_mpeg7.txt","w+") as f:
		f.write(str(len(unused_mpeg7))+"\n")
		for u in unused_mpeg7:
			f.write(str(u)+"\n")

	unused_mnist = []

	# MNIST data
	####
	# classes 0 -> 9 are integers 0->9
	# classes 10 -> 35 are uppercase letters A->Z
	# classes 36 -> 61 are lowercase letters a->z
	classes = range(0,62) #gives us a list [0,1,...,61] which covers all classes
	samples = 100 #get first 100 samples from each class

	#[95.0, 97.0, 110.0, 100.0, 106.0, 114.0, 111.0, 100.0, 113.0, 104.0, 96.0, 100.0, 96.0, 107.0, 100.0, 106.0, 95.0, 108.0, 90.0, 95.0, 114.0, 84.0, 95.0, 100.0, 105.0, 99.0, 106.0, 104.0, 109.0, 100.0, 97.0, 109.0, 115.0, 109.0, 114.0, 115.0, 102.0, 98.0, 100.0, 107.0, 115.0, 107.0, 104.0, 112.0, 96.0, 96.0, 107.0, 110.0, 95.0, 115.0, 101.0, 95.0, 103.0, 100.0, 102.0, 82.0, 108.0, 95.0, 96.0, 115.0, 108.0, 96.0]
	#Average: 102.951612903 Std: 7.631487022378845
	threshold = 102.951612903
	for c in classes:
		images = get_mnist_img(c, samples)
		samp_count = 0
		for img in images:
			# original eps is .005
			G, ret = get_img_data_approx(img,eps,102.951612903)
			# G = get_img_data(img)
			output_file = "MNIST_C"+str(c)+"_S"+str(samp_count)
			draw_graph(G, G.graph['stratum'], graphs_dir+"/mnist_imgs/"+output_file)
			if ret != -2 and ret != -1 and ret != 0:
				nx.write_gpickle(G, graphs_dir+"/mnist/"+str(output_file)+".gpickle")
			else:
				unused_mnist.append((output_file, ret))
				print(output_file + " graph was not used")
			samp_count+=1
			print(output_file+ ": " +str(time.time() - t)+ "(s)")
			t = time.time()

	with open(graphs_dir+"/unused_mnist.txt","w+") as f:
		f.write(str(len(unused_mnist)) + "\n")
		for u in unused_mnist:
			f.write(str(u)+"\n")

def main():
	# make sure we have the same seeds as main
	random.seed(423652346)
	np.random.seed(423652346)
	
	# generate all random graphs
	generate()
if __name__=='__main__':main()



