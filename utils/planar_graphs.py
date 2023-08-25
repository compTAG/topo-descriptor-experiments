import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import os
import itertools
from shapely.geometry import LineString, Point
import pickle
import glob
from itertools import permutations
from itertools import combinations
import random
import topology
import dionysus as d
import time

def create_graph(verts, edges):
    g = nx.Graph()
    g.add_nodes_from(range(len(verts)))
    for i, coords in enumerate(verts):
        g.nodes[i]['pos'] = coords

    g.add_edges_from(edges)

    return g
    
def save_city_map(city, state, country): 
  G = ox.graph_from_place(','.join([city, state, country]), simplify=False, network_type='drive')
  fig = plt.figure()    
  fig, ax = ox.plot_graph(G,node_color='r',show=False, close=False, bgcolor="white")
  fig.savefig(os.path.join("graphs","maps",city, city + "_map.pdf"), bbox_inches='tight')

def get_city_map(city,state, country):

    dirname = os.path.join("graphs","maps",city,"experiments")
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Directory %s created" % dirname)
    else:    
        print("Directory %s already exists" % dirname)

    G = ox.graph_from_place(','.join([city, state, country]), simplify=False, network_type='drive')
    #G = G.to_undirected()
    G_proj = ox.project_graph(G)
    nodes_proj, gdf_edges = ox.graph_to_gdfs(G_proj, edges=True, nodes = True)

    nx.write_gpickle(G, os.path.join("graphs","maps",city,city + "_" + state+".pkl"))

    return G, nodes_proj.head(200)
    
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

# Cache for previously checked pairs of edges
intersection_memo = {}

def bounding_boxes_intersect(line1_coords, line2_coords):
    x1, y1 = line1_coords[0]
    x2, y2 = line1_coords[1]
    x3, y3 = line2_coords[0]
    x4, y4 = line2_coords[1]

    # Compute bounding boxes
    box1 = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    box2 = (min(x3, x4), min(y3, y4), max(x3, x4), max(y3, y4))

    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

def is_intersection(vertices, edges):
    check_vertices = {Point(vertex) for vertex in vertices}

    for e_1, e_2 in itertools.combinations(edges, 2):
        # If this combination was checked before, use the cached result
        if (e_1, e_2) in intersection_memo:
            if intersection_memo[(e_1, e_2)]:  # If they were found intersecting before
                return True
            continue

        # Quick bounding box check
        if not bounding_boxes_intersect([vertices[e_1[0]], vertices[e_1[1]]], [vertices[e_2[0]], vertices[e_2[1]]]):
            continue

        line1 = LineString([vertices[e_1[0]], vertices[e_1[1]]])
        line2 = LineString([vertices[e_2[0]], vertices[e_2[1]]])

        if line1.intersects(line2):
            int_pt = line1.intersection(line2)
            if int_pt not in check_vertices:
                intersection_memo[(e_1, e_2)] = True
                return True

        intersection_memo[(e_1, e_2)] = False

    return False

# Memoization for find_planar_graphs
memo = {}

def find_planar_graphs(vertices, edges):

    n = len(edges)
    
    # Check if the result for this input is already computed
    if (tuple(vertices), n) in memo:
        return memo[(tuple(vertices), n)]

    G = []

    # Generate all subgraphs with varying sizes
    for size in range(1, n+1):
        for sub_edges in combinations(edges, size):
            if not any(is_intersection(vertices, list(sub_edges[:i+1])) for i in range(len(sub_edges))):
                G.append(list(sub_edges))

    # Store the result in the memoization table
    memo[(tuple(vertices), n)] = G


    return G
'''
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
    if len(edges) != 1:
        G = G + find_planar_graphs(vertices, edges[:-1])
        for graph in G:
            if not is_intersection(vertices, graph + [edges[-1]]):
                G = G + [graph + [edges[-1]]]

        return G
    else:
        return [G]        
'''
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

def collinear(vertices):
    d = np.array([[1,1,1], 
        [vertices[0][0], vertices[1][0], vertices[2][0]], 
        [vertices[0][1], vertices[1][1], vertices[2][1]]])
    return 0.5*abs(np.linalg.det(d))

def make_bounding_box(vertex, meters):
    lat = vertex[0]
    lon = vertex[1]

    #Radius of the earth
    r = 6371
    #meters in kilometers
    m = meters/1000

    north = lat + (m/r)*(180/np.pi)
    south = lat - (m/r)*(180/np.pi)
    east = lon + (m/r)*(180/np.pi)/np.cos(lat*np.pi/180)
    west = lon - (m/r)*(180/np.pi)/np.cos(lat*np.pi/180)

    return [north,south, east, west]


def duplicate_graphs(g, graphs, duplicate = False):
    for graph in graphs:
        if set(g.nodes()) == set(graph.nodes()) and set(g.edges()) == set(graph.edges()):
            print("Duplicate Found")
            duplicate = True
            break
    return duplicate



def find_subgraphs(G, nodes_proj, source_dir, bbox):
    graphs4 = []
    graphs5 = []
    graphs6 = []
   #Get node positions
    x = nodes_proj['lat'].tolist()
    y = nodes_proj['lon'].tolist()
    vertices = list(zip(x, y))
    k = 0
    for vertex in vertices:
        try:
            bb = make_bounding_box(vertex, bbox)
            g = ox.truncate.truncate_graph_bbox(G, bb[0], bb[1], bb[2], bb[3])
            if len(g.nodes()) == 4:
                if len(graphs4) == 0:
                    graphs4.append(g)
                else:
                 if duplicate_graphs(g,graphs4) == False:
                    graphs4.append(g)
                print("4 graphs " + str(len(graphs4)))    
            elif len(g.nodes()) == 5:
                if len(graphs5) == 0:
                    graphs5.append(g)
                else:
                 if duplicate_graphs(g,graphs5) == False:
                    graphs5.append(g)
                print("5 graphs " + str(len(graphs5)))
            elif len(g.nodes()) == 6:
                if len(graphs6) == 0:
                    graphs6.append(g)
                else:
                 if duplicate_graphs(g,graphs6)==False:
                    graphs6.append(g)
                print("6 graphs " + str(len(graphs6)))
            k = k+1
            print(k)
            pass      
        except Exception as e:
            print(e.__class__, "occurred.")
            print("Next entry.") 
    
    g4 = "Bozeman_4graphs_from_source_"+ str(bbox) +".pickle"
    g5 = "Bozeman_5graphs_from_source_"+ str(bbox) +".pickle"
    g6 = "Bozeman_6graphs_from_source_"+ str(bbox) +".pickle"
    with open(os.path.join(source_dir,g4), "wb") as f:
        pickle.dump(graphs4, f)
    with open(os.path.join(source_dir,g5), "wb") as f:
        pickle.dump(graphs5, f)
    with open(os.path.join(source_dir,g6), "wb") as f:
        pickle.dump(graphs6, f)


