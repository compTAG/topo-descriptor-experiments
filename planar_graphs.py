import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import os



def get_graph(location_point, dist):
  G = ox.graph_from_point(location_point, dist=dist,simplify=False)
  G = ox.get_undirected(G)

  fig, ax = ox.plot_graph(G,node_color='r',show=False, close=False)
  plt.show()
  G_relable = nx.convert_node_labels_to_integers(G)
  G_proj = ox.project_graph(G_relable)
  nodes_proj, gdf_edges = ox.graph_to_gdfs(G_proj, edges=True, nodes = True)
  
  #Get node positions
  x = nodes_proj['x'].tolist()
  y = nodes_proj['y'].tolist()
  verts = list(zip(x, y))

  return G_relable,verts


def build_graphs(n=2, i=None, j=None):
    """Make a graph recursively, by either including, or skipping each edge.
    Edges are given in lexicographical order by construction."""
    out = []
    if i is None: # First call
        out  = [[(0,1)]+r for r in build_graphs(n=n, i=0, j=1)]
    elif j<n-1:
        out += [[(i,j+1)]+r for r in build_graphs(n=n, i=i, j=j+1)]
        out += [          r for r in build_graphs(n=n, i=i, j=j+1)]
    elif i<n-1:
        out = build_graphs(n=n, i=i+1, j=i+1)
    else:
        out = [[]]
    return out

def plot_graphs(graphs, figsize=14, dotsize=20):
    """Utility to plot a lot of graphs from an array of graphs. 
    Each graphs is a list of edges; each edge is a tuple."""
    n = len(graphs)
    fig = plt.figure(figsize=(figsize,figsize))
    fig.patch.set_facecolor('white') # To make copying possible (no transparent background)
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

def perm(n, s=None):
    """All permutations of n elements."""
    if s is None: return perm(n, tuple(range(n)))
    if not s: return [[]]
    return [[i]+p for i in s for p in perm(n, tuple([k for k in s if k!=i]))]



def permute(g, n):
    """Create a set of all possible isomorphic codes for a graph, 
    as nice hashable tuples. All edges are i<j, and sorted lexicographically."""
    ps = perm(n)
    out = set([])
    for p in ps:
        out.add(tuple(sorted([(p[i],p[j]) if p[i]<p[j] else (p[j],p[i]) for i,j in g])))
    return list(out)


def connected(g):
    """Check if the graph is fully connected, with Union-Find."""
    nodes = set([i for e in g for i in e])
    roots = {node: node for node in nodes}
    
    def _root(node, depth=0):
        if node==roots[node]: return (node, depth)
        else: return _root(roots[node], depth+1)
    
    for i,j in g:
        ri,di = _root(i)
        rj,dj = _root(j)
        if ri==rj: continue
        if di<=dj: roots[ri] = rj
        else:      roots[rj] = ri
    return len(set([_root(node)[0] for node in nodes]))==1


def filter(gs, target_nv):
    """Filter all improper graphs: those with not enough nodes, 
    those not fully connected, and those isomorphic to previously considered."""
    mem = set({})
    gs2 = []
    for g in gs:
        nv = len(set([i for e in g for i in e]))
        if nv != target_nv:
            continue
        if not connected(g):
            continue
        if tuple(g) not in mem:
            gs2.append(g)
            mem |= set(permute(g, target_nv))
        #print('\n'.join([str(a) for a in mem]))
    return gs2





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


