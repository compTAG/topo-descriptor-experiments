import networkx as nx
import os
from vertex import *
def find_max(G):
	x = max([v[1]['v'].get_x() for v in G.nodes(data=True)])
	y = max([v[1]['v'].get_y() for v in G.nodes(data=True)])
	return x,y

# MNIST
x_list_mnist = []
y_list_mnist = []
for filename in os.listdir(os.path.join('graphs', 'mnist')):
	G = nx.read_gpickle(os.path.join('graphs', 'mnist', filename))
	x,y = find_max(G)
	x_list_mnist.append(x)
	y_list_mnist.append(y)

print("MNIST")
print(max(x_list_mnist))
print(max(y_list_mnist))

#MPEG7
x_list_mpeg7 = []
y_list_mpeg7 = []
for filename in os.listdir(os.path.join('graphs','mpeg7')):
	G = nx.read_gpickle(os.path.join('graphs','mpeg7', filename))
	x,y = find_max(G)
	if x == 1108.0836907535127 or y == 1108.0836907535127:
		print (filename)
	x_list_mpeg7.append(x)
	y_list_mpeg7.append(y)

print("MPEG7")
print(max(x_list_mpeg7))
print(max(y_list_mpeg7))

#Rand
x_list_rand = []
y_list_rand = []
for filename in os.listdir(os.path.join('graphs_random')):
	G = nx.read_gpickle(os.path.join('graphs_random' , filename))
	x,y = find_max(G)
	x_list_rand.append(x)
	y_list_rand.append(y)
print("RAND")
print(max(x_list_rand))
print(max(y_list_rand))
