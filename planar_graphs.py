import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import os
import planarity as p
import itertools

def create_graph(verts, edges):
    g = nx.Graph()
    g.add_nodes_from(range(len(verts)))
    for i, coords in enumerate(verts):
        g.nodes[i]['pos'] = coords

    g.add_edges_from(edges)

    return g

#Pull a graph from osmnx by location
def get_source_graph(location_point, dist):
  G = ox.graph_from_point(location_point, dist=dist,simplify=False)
  G = ox.get_undirected(G)

  #fig, ax = ox.plot_graph(G,node_color='r',show=False, close=False)
  #plt.show()
  G_relable = nx.convert_node_labels_to_integers(G)
  G_proj = ox.project_graph(G_relable)
  nodes_proj, gdf_edges = ox.graph_to_gdfs(G_proj, edges=True, nodes = True)
  
  #Get node positions
  x = nodes_proj['x'].tolist()
  y = nodes_proj['y'].tolist()
  verts = list(zip(x, y))

  return G_relable,verts


def find_all_planar_graphs(edges):
    G = []
    if len(edges) != 0:
        G = G + find_all_planar_graphs(edges[:-1])
        for graph in G:
            #print(graph + [edges[-1]])
            if p.is_planar(graph + [edges[-1]]):
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











'''
#direction = (0,1)

verts = (
            (0,0),
            (2,2),
            (1,3),
            (4,1),
            (5,2),
            (6,1),
        )

edges = (
            (0, 1),
            (1, 2),
            (2, 0),
            (1, 3),
            (3, 5),
            (5, 4),
            (4, 3),
        )

graph = create_graph(verts, edges)
pos = { i : verts[i] for i in range(0, len(verts) ) }
nx.draw(graph,pos, labels={node:node for node in graph.nodes()},edge_labels=True)
nx.draw_networkx_edge_labels(graph,pos,edge_labels={(0,1):'a',
(1,2):'b',(2,0):'c',(1,3):'d',(3,5):'e',(5,4):'f',(4,3):'g'},font_color='red')
xcoords = [0, 1, 2,4,5,6]
colors = ['k','k','k','k','k','k']
for xc,c in zip(xcoords,colors):
    plt.axvline(x=xc, label='line at x = {}'.format(xc), c=c)
plt.show()
direction = (1,0)
filtr = topology.LowerStarFiltrationFactory(direction).create(graph)
m = d.homology_persistence(filtr)
diag = d.init_diagrams(m, filtr)
print(diag)
print(type(diag[0]))
d.plot.plot_diagram(diag[0])

'''

'''
ox.config(log_console=True, use_cache=True)
location_point = (45.67930061221573, -111.03874239039452)
distance = 55
G,verts = get_source_graph(location_point,distance)

#print(verts)


unch = list(itertools.combinations(G.nodes(), 2))

test = find_all_planar_graphs(unch)
#test.pop(0)
test.pop(0)
print(test)
#plot_graphs(test)
'''

'''
# Example of the complete graph of 5 nodes, K5
# K5 is not planar
# any of the following formats can bed used for representing the graph

edgelist = [(0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2),(1, 3),(1, 4),
            (2, 3), (2, 4),
            (3, 4)]
P=p.PGraph(edgelist)
#print(P.nodes()) # indexed from 1..n
print(P.mapping()) # the node mapping
#print(P.edges()) # edges
print(P.is_planar())  # False
#print(P.kuratowski_edges())

edgelist.remove((0,1))
P=p.PGraph(edgelist)
print(P.is_planar())  
'''

