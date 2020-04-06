import networkx as nx
import os
from orth_angle import *
import math

graphs_dir = "graphs_005_approx"
output_dir = "output_005_approx"
# Counterclockwise angle in degrees by turning from a to c around b
# Returns a float between 0.0 and 2pi
# found at https://python-forum.io/Thread-finding-angle-between-three-points-on-a-2d-graph
def angle(a, b, c):
    ang = math.atan2(c.get_y()-b.get_y(), c.get_x()-b.get_x()) - math.atan2(a.get_y()-b.get_y(), a.get_x()-b.get_x())
    return ang + (2*math.pi) if ang < 0 else ang

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

def print_G(G):
	print("VERTICES\n")
	for v in G.nodes(data=True):
		print(str(v[0]) + " " +str(v[1]['v'].get_x()) + " " + str(v[1]['v'].get_y()))
	print("\nEDGES\n")
	for e in G.edges(data=True):
		print(str(e))
		print(str(G.node[e[0]]['v'].get_x()) + " " +str(G.node[e[0]]['v'].get_y()))
		print(str(G.node[e[1]]['v'].get_x()) + " " +str(G.node[e[1]]['v'].get_y()))
	print(G.has_edge(30, 31))
	print(G.has_edge(31, 30))
	print(G.has_edge(23, 30))

def delta_exp(exp_list,exp_type):
	with open(output_dir+"/delta_exp/"+exp_type+"/deltas.txt", "w+") as f:
		f.write("n,delta,outFile\n")
		for e in exp_list:
			G = e['G']
			output_file = e['output_file']
			# print_G(G)
			# print(list(nx.cycle_basis(G)))

			### we get our deltas in R^2 from example 7.4 of Curry et al. 2018
			delta_list = []
			for c in nx.cycle_basis(G):

				#for situations where a contour might just have a single point
				if len(c) < 3:
					continue
				# print(c)
				for i in range(0, len(c)-2):
					# print(str(c[i])+","+str(c[i+1])+","+str(c[i+2]))
					# angle1 = angle(G.node[c[i]]['v'], G.node[c[i+1]]['v'], G.node[c[i+2]]['v'])
					# angle2 = angle(G.node[c[i+2]]['v'], G.node[c[i+1]]['v'], G.node[c[i]]['v'])
					# print(angle1)
					# print(angle2)
					delta = math.pi - min(angle(G.node[c[i]]['v'], G.node[c[i+1]]['v'], G.node[c[i+2]]['v']),
						angle(G.node[c[i+2]]['v'], G.node[c[i+1]]['v'], G.node[c[i]]['v']))
					# print(math.degrees(delta))
					delta_list.append(delta)
				# print(str(c[len(c)-2])+","+str(c[len(c)-1])+","+str(c[0]))
				# angle1 = angle(G.node[c[len(c)-2]]['v'], G.node[c[len(c)-1]]['v'], G.node[c[0]]['v'])
				# angle2 = angle(G.node[c[0]]['v'], G.node[c[len(c)-1]]['v'], G.node[c[len(c)-2]]['v'])
				# print(angle1)
				# print(angle2)
				delta = math.pi - min(angle(G.node[c[len(c)-2]]['v'], G.node[c[len(c)-1]]['v'], G.node[c[0]]['v']),
					angle(G.node[c[0]]['v'], G.node[c[len(c)-1]]['v'], G.node[c[len(c)-2]]['v']))
				# print(math.degrees(delta))
				delta_list.append(delta)


				# print(str(c[len(c)-1])+","+str(c[0])+","+str(c[1]))
				# angle1 = angle(G.node[c[len(c)-1]]['v'], G.node[c[0]]['v'], G.node[c[1]]['v'])
				# angle2 = angle(G.node[c[1]]['v'], G.node[c[0]]['v'], G.node[c[len(c)-1]]['v'])
				# print(angle1)
				# print(angle2)
				delta = math.pi - min(angle(G.node[c[len(c)-1]]['v'], G.node[c[0]]['v'], G.node[c[1]]['v']),
					angle(G.node[c[1]]['v'], G.node[c[0]]['v'], G.node[c[len(c)-1]]['v']))
				# print(math.degrees(delta))
				delta_list.append(delta)
			delta = min(delta_list)
			# print("Smallest delta: "+str(delta))
			f.write(str(len(G.nodes()))+","+str(delta)+","+output_file+"\n")

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
	# exp_list_mnist = get_mnist()
	# delta_exp(exp_list_mnist,"mnist")
	# test_angle_func()

if __name__ == '__main__':main()

