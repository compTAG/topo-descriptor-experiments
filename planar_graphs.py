import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import os
import planarity as p
import itertools
from shapely.geometry import LineString, Point
import pickle
import glob
from itertools import permutations
import random
import topology
import dionysus as d

def create_graph(verts, edges):
    g = nx.Graph()
    g.add_nodes_from(range(len(verts)))
    for i, coords in enumerate(verts):
        g.nodes[i]['pos'] = coords

    g.add_edges_from(edges)

    return g

def get_city_map(city,state, country):
    try:
    # Create target Directory
        dirname = os.path.join("graphs","maps",city)
        os.mkdir(dirname)
        print("Directory" , dirname ,  "Created ")
    except FileExistsError:
        print("Directory" , dirname ,  "already exists")


    G = ox.graph_from_place(','.join([city, state, country]), simplify=False, network_type='drive')
    #G = G.to_undirected()
    G_proj = ox.project_graph(G)
    nodes_proj, gdf_edges = ox.graph_to_gdfs(G_proj, edges=True, nodes = True)

    nx.write_gpickle(G, os.path.join("graphs","maps",city,city)+".pkl")
    
def request_graph(location_point, dist):
  G = ox.graph_from_point(location_point, dist=dist,simplify=False)
  G = ox.get_undirected(G)

  return G

#Pull a graph from osmnx by location
def get_source_graph(graph):
  #fig, ax = ox.plot_graph(graph,node_color='r',show=False, close=False)
  #plt.show()
  G_relable = nx.convert_node_labels_to_integers(graph)
  G_proj = ox.project_graph(G_relable)

  nodes_proj, gdf_edges = ox.graph_to_gdfs(G_proj, edges=True, nodes = True)
  
  #Get node positions
  x = nodes_proj['x'].tolist()
  y = nodes_proj['y'].tolist()
  verts = list(zip(x, y))

  return G_relable,verts

def is_intersection(vertices, edges,intersect = False):
    check_vertices = []
    intersection_points = []
    for vertex in vertices:
        check_vertices.append(Point(vertex))
    for e_1,e_2 in itertools.combinations(edges, 2):
        line1 = LineString([vertices[e_1[0]], vertices[e_1[1]]])
        line2 = LineString([vertices[e_2[0]], vertices[e_2[1]]])
        int_pt = line1.intersection(line2)
        if line1.intersects(line2):
            intersection_points.append(int_pt)
    for i_p in intersection_points:
        if any((i_p == v) for v in check_vertices):
            continue
        else:
            intersect = True
    return intersect 

def find_planar_graphs(vertices, edges):
    G = []
    if len(edges) != 0:
        G = G + find_planar_graphs(vertices, edges[:-1])
        for graph in G:
            if not is_intersection(vertices, graph + [edges[-1]]):
                G = G + [graph + [edges[-1]]]
        return G
    else:
        return [G]        

def plot_graphs(graphs, figsize=14, dotsize=20):
    n = len(graphs)
    fig = plt.figure(figsize=(figsize,figsize))
    fig.patch.set_facecolor('white') 
    k = int(np.sqrt(n))
    for i in range(n):
        plt.subplot(k+1,k+1,i+1)
        g = nx.Graph()
        for e in graphs[i]:            
            g.add_edge(e[0],e[1])
        nx.draw_kamada_kawai(g, node_size=dotsize)
        #print(nx.check_planarity(g))
        print('.', end='')
    plt.show()                                             


def circle_disc(d,n):
    directions = []
    points = np.linspace(d, d + 2*np.pi, n, endpoint=False)
    for point in points:
        x = np.cos(point)
        y = np.sin(point)
        directions.append((x, y))
    return directions



'''
# Testing code

graph = create_graph(verts,edges)
pos = { i : verts[i] for i in range(0, len(verts) ) }
print(pos)
nx.draw(graph,pos, with_labels = True)
plt.show()


source_dir = os.path.join("graphs","maps","Bozeman","graphs","4_graphs", "4_nodes.pickle")
in_graph = open(source_dir,"rb")
graphs = pickle.load(in_graph)
G,verts = get_source_graph(graphs[3])

test = check_planar(verts,list(itertools.combinations(G.nodes(), 2)))
print(len(test))

test_1 = clean_planar(verts,test)
print(test_1)
print(len(test_1))

verts = [(492476.236455919, 5059797.971598339), 
        (492499.0750844237, 5059799.543931472), 
        (492516.71305272356, 5059801.522552364), 
        (492535.561073114, 5059812.687826383)]
edges = [(0, 2), (1, 3), (2, 3)]
#edges = [(0, 1), (1, 2), (2, 3)]
test = is_intersection(verts,edges)
print(test)
'''

'''
verts = [(492476.236455919, 5059797.971598339), 
        (492499.0750844237, 5059799.543931472), 
        (492516.71305272356, 5059801.522552364), 
        (492535.561073114, 5059812.687826383)]
e = []
unch = [(0, 1), (1, 2), (2, 3)]

def find_planar(edges, unchecked):
    G = []
    if len(unchecked) == 0:
        return edges
    for e in unchecked:
        G = edges +[e]
        #G = G + find_planar(edges, unchecked[:-1])
        #G = G + find_planar(edges + [unchecked[:-1]], unchecked[:-1])
    return G

test= find_planar(e,unch)
print(test)
'''

'''
verts = [(492476.236455919, 5059797.971598339), 
        (492499.0750844237, 5059799.543931472), 
        (492516.71305272356, 5059801.522552364), 
        (492535.561073114, 5059812.687826383)]

edges = [(0, 2), (2,3), (1, 3)]

#test = is_intersection(verts,edges)
#print(test)

#test = test_planar(verts, edges)
test = find_all_planar_graphs(edges)
print(test)
'''
'''
source_dir = os.path.join("graphs","maps","Bozeman","graphs","4_graphs", "4_nodes.pickle")
in_graph = open(source_dir,"rb")
graphs = pickle.load(in_graph)

graph = graphs[3]
G,verts = get_source_graph(graph)
test = test_planar(verts,list(itertools.combinations(G.nodes(), 2)))
print(test)
'''
'''
source_dir = os.path.join("graphs","maps","Bozeman","graphs","4_graphs", "4_nodes.pickle")
in_graph = open(source_dir,"rb")
graphs = pickle.load(in_graph)
print(len(graphs))

for G in graphs[400:-1]:
    fig, ax = ox.plot_graph(G,node_color='r',show=False, close=False)
    plt.show()
'''









