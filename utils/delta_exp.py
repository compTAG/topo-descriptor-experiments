import networkx as nx
import os
from utils.orth_angle import *
import math

#graphs_dir = "graphs_001_approx"
#output_dir = "output_001_approx"
# Counterclockwise angle in degrees by turning from a to c around b
# Returns a float between 0.0 and 2pi
# found at https://python-forum.io/Thread-finding-angle-between-three-points-on-a-2d-graph
def angle(a, b, c):
    ang = math.atan2(c.get_y()-b.get_y(), c.get_x()-b.get_x()) - math.atan2(a.get_y()-b.get_y(), a.get_x()-b.get_x())
    return ang + (2*math.pi) if ang < 0 else ang

def get_mnist():
	exp_list=[]
	for filename in os.listdir(os.path.join(graphs_dir,'mnist')):
		G = nx.read_gpickle(os.path.join(graphs_dir,'mnist' , filename))
		output_file = os.path.join("mnist", filename[:-8]+".txt")
		exp_list.append({"G":G, "output_file":output_file})
	return exp_list

def get_mpeg7():
	exp_list=[]
	for filename in os.listdir(os.path.join(graphs_dir,'mpeg7')):
		G = nx.read_gpickle(os.path.join(graphs_dir,'mpeg7' , filename))
		output_file = os.path.join("mpeg7", filename[:-8]+".txt")
		exp_list.append({"G":G, "output_file":output_file})
	return exp_list

def print_G(G):
	print("VERTICES\n")
	for v in G.nodes(data=True):
		print(str(v[0]) + " " +str(v[1]['v'].get_x()) + " " + str(v[1]['v'].get_y()))
	print("\nEDGES\n")
	for e in G.edges(data=True):
		print(str(e))
		print(str(G.node[e[0]]['v'].get_x()) + " " +str(G.node[e[0]]['v'].get_y()))
		print(str(G.node[e[1]]['v'].get_x()) + " " +str(G.node[e[1]]['v'].get_y()))

def delta_exp(G,output_file,output_dir):
	with open(os.path.join(output_dir,"delta_exp", output_file.split('/')[0],"deltas.txt"), 'a') as f:
			
			### we get our deltas in R^2 from example 7.4 of Curry et al. 2018
		delta_list = []
		for i in range(0,len(G.nodes())):
			neighbors = list(G.neighbors(i))
			if len(neighbors) != 2:
				print("ERROR, neighbors list is not size 2!")
				print(neighbors)
				print(i)
				print_G(G)
				print(output_file)
				sys.exit(1)

			n1 = neighbors[0]
			n2 = neighbors[1]
			delta = math.pi - min(angle(G.nodes[n1]['v'], G.nodes[i]['v'], G.nodes[n2]['v']),
				angle(G.nodes[n2]['v'], G.nodes[i]['v'], G.nodes[n1]['v']))
			delta_list.append(delta)
		delta = min(delta_list)
		#print("Smallest delta: "+str(delta))
		f.write(str(len(G.nodes()))+","+str(delta)+","+output_file+"\n")
		f.close()

def test_angle_func():
	origin = Vertex(-1, 0.0, 0.0)

	q1 = Vertex(0, 1.0, 1.0)
	q2 = Vertex(1, -1.0, 1.0)
	q3 = Vertex(2, -1.0, -1.0)
	q4 = Vertex(3, 1.0, -1.0)
	print("Testing q1 and q2")
	print("Should be 1/2pi")
	print(angle(q1,origin,q2))
	print("Should be 3/2pi")
	print(angle(q2,origin,q1))

	print("Testing q1 and q3")
	print("Should be pi")
	print(angle(q1,origin,q3))
	print("Should be pi")
	print(angle(q3,origin,q1))

	print("Testing q1 and q4")
	print("Should be 3/2pi")
	print(angle(q1,origin,q4))
	print("Should be 1/2pi")
	print(angle(q4,origin,q1))

	print("Testing q2 and q3")
	print("Should be 1/2pi")
	print(angle(q2,origin,q3))
	print("Should be 3/2pi")
	print(angle(q3,origin,q2))

	print("Testing q2 and q4")
	print("Should be pi")
	print(angle(q2,origin,q4))
	print("Should be pi")
	print(angle(q4,origin,q2))

	print("Testing q3 and q4")
	print("Should be 1/2pi")
	print(angle(q3,origin,q4))
	print("Should be 3/2pi")
	print(angle(q4,origin,q3))

	x = Vertex(4, 1.0, 0.0)
	print("Testing x, should be 0.0")
	print(angle(x, origin, x))

	# x = Vertex(0,0.0,1.0)
	# y = Vertex(1,1.0,0.0)
	# z = Vertex(2,0.0,0.0)
	# a = Vertex(3,-1.0,0.0)
	# b = Vertex(4,1.0,-.001)
	# print(angle(x,z,y)) # should be 3/2 pi
	# print(angle(y,z,x)) # should be 1/2 pi
	# print(angle(a,z,b))
	# print(angle(b,z,a))
	# print(math.pi - min(angle(a,z,b), angle(b,z,a)))
	# print(angle(x,z,b))
	# print(angle(b,z,x))

def main():
	exp_list_mpeg7 = get_mpeg7()
	print(len(exp_list_mpeg7))
	delta_exp(exp_list_mpeg7,"mpeg7")
	exp_list_mnist = get_mnist()
	print(len(exp_list_mnist))
	delta_exp(exp_list_mnist,"mnist")
	# test_angle_func()

if __name__ == '__main__':main()

