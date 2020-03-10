from vertex import *
from loadEMNIST import *
from importTurner import *
import networkx as nx
import numpy as np




# convert arrays into networkx graph
# this method is definitely not optimal in the slightest, but it works
def array_to_nx(vert_array, edge_array):
    G = nx.Graph()
    id_list = []
    x_list = []
    y_list = []
    index = 0
    for i in vert_array:
        for j in i:
            if index == 0:
                id_list.append(j)
            if index == 1:
                x_list.append(j)
            if index == 2:
                y_list.append(j)
        index += 1

    for i in range(0, len(x_list)-1):
        G.add_node(Vertex(id_list[i], x_list[i], y_list[i]))

    edge_id_list = []
    edge_x_list = []
    edge_y_list = []
    index = 0
    for i in edge_array:
        for j in i:
            if index == 0:
                edge_id_list.append(int(j))
            if index == 1:
                edge_x_list.append(int(j))
            if index == 2:
                edge_y_list.append(int(j))
        index += 1

    print(list(G.nodes))

    for i in range(0, len(edge_id_list) - 1):
        G.add_edge(edge_x_list[i], edge_y_list[i])

    print(list(G.edges))

    return G




with open("ShapeBoundaries/C1S1.txt") as fp:

    vert_array, edge_array = gen_turner_arrays(fp)

    array_to_nx(vert_array, edge_array)
