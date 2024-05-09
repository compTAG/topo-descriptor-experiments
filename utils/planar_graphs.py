import osmnx as ox
import warnings
import json
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
import random
import topology
import dionysus as d

warnings.filterwarnings("ignore", message="invalid value encountered in intersection")

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

    """
    Retrieves and saves a map of a specified city.

    Args:
        city (str): Name of the city.
        state (str): Name of the state or province.
        country (str): Name of the country.

    Returns:
        tuple: A tuple containing the graph representation of the city and its projected nodes.

    """

    # Create directory for storing maps if it does not exist
    dirname = os.path.join("graphs","maps",city,"experiments")
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Directory %s created" % dirname)
    else:    
        print("Directory %s already exists" % dirname)

    # Retrieve graph data for the specified location
    G = ox.graph_from_place(','.join([city, state, country]), simplify=False, network_type='drive')
    G_proj = ox.project_graph(G)
    nodes_proj, gdf_edges = ox.graph_to_gdfs(G_proj, edges=True, nodes = True)

    # Save graph data to JSON file
    with open(os.path.join("graphs", "maps", city, city + "_" + state + ".json"), "w") as f:
        data = nx.readwrite.json_graph.node_link_data(G)
        json.dump(data, f)

    return G, nodes_proj
    
def request_graph(location_point, dist):
  G = ox.graph_from_point(location_point, dist=dist,simplify=False)
  G = ox.get_undirected(G)

  return G

#Pull a graph from osmnx by location
def get_source_graph(graph):
  """
  Converts the node labels to integers, projects the graph, and retrieves node positions.

    Args:
        graph (networkx.Graph): The input graph.

    Returns:
        tuple: A tuple containing the undirected graph and its vertex positions.
  """
  # Convert node labels to integers and project the graph
  G_relable = nx.convert_node_labels_to_integers(graph)
  G_proj = ox.project_graph(G_relable)

  # Retrieve node positions
  nodes_proj, gdf_edges = ox.graph_to_gdfs(G_proj, edges=True, nodes = True)
  x = nodes_proj['x'].tolist()
  y = nodes_proj['y'].tolist()
  verts = list(zip(x, y))

  #Convert to undirected graph
  G_undirected = G_relable.to_undirected()

  return G_undirected,verts

def is_intersection(vertices, edges,intersect = False):
    """
    Check if there is any intersection between edges in the given graph.

    This function checks if there is any intersection between edges in the graph formed by the
    given vertices and edges.

    Parameters:
        vertices (list): List of vertices in the graph.
        edges (list): List of edges in the graph.
        intersect (bool): Flag indicating if there is an intersection. Default is False.

    Returns:
        bool: True if there is an intersection, False otherwise.
    """
    check_vertices = [] # List to store vertices as Point objects for intersection checks
    intersection_points = [] # List to store intersection points

    # Convert vertices to Point objects
    for vertex in vertices:
        check_vertices.append(Point(vertex))
    
    # Check for intersections between all pairs of edges
    for e_1,e_2 in itertools.combinations(edges, 2):
        line1 = LineString([vertices[e_1[0]], vertices[e_1[1]]]) # Create LineString for edge 1
        line2 = LineString([vertices[e_2[0]], vertices[e_2[1]]]) # Create LineString for edge 2
        if line1.intersects(line2): # Check for intersection between the two lines
            int_pt = line1.intersection(line2) # Find the intersection point
            intersection_points.append(int_pt) # Add the intersection point to the list

    # Check if any intersection points are also vertices
    for i_p in intersection_points:
        if any((i_p == v) for v in check_vertices):
            continue
        else:
            intersect = True
    return intersect

def find_planar_graphs(vertices, edges):
    """
    Find all possible planar graphs given a set of vertices and edges.

    This function recursively generates all possible combinations of edges that form planar graphs
    using a divide-and-conquer approach.

    Parameters:
        vertices (list): List of vertices in the graph.
        edges (list): List of edges in the graph.

    Returns:
        list: List of lists representing all possible planar graphs.
    """

    G = [] # Initialize list to store planar graphs
    if len(edges) != 0:
        # Recursively call find_planar_graphs with one less edge
        G = G + find_planar_graphs(vertices, edges[:-1])
        for graph in G:
            # Check if adding the current edge forms a valid planar graph
            if not is_intersection(vertices, graph + [edges[-1]]):
                if edges[-1] not in graph:
                    G = G + [graph + [edges[-1]]] # Add the edge to the planar graph
        return G
    else:
        return [G] # Return an empty graph as the base case     

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
        print('.', end='')
    plt.show()                                             


def circle_disc(arcs):
    """
    Randomly selects a direction within a circle sector defined by arcs.

    Parameters:
        arcs (list of dict): List of arcs defining circle sectors. Each arc is represented by a dictionary 
                             containing 'start' and 'end' points and a 'hit' flag.

    Returns:
        tuple or None: A tuple (x, y) representing a random direction within a circle sector. Returns None if 
                       all arcs have been hit.
    """

    # Select a random arc that has not been hit
    random_arc = random.randint(0, len(arcs)-1)
    while (arcs[random_arc]["hit"] == 1):
        random_arc = random.randint(0, len(arcs)-1)
        # Check if all arcs have been hit, if so, return None
        if all(arc["hit"] == 1 for arc in arcs):
            print("All arcs have been hit.")
            return None
        
    # Mark the selected arc as hit
    arcs[random_arc]["hit"] = 1

    random_point = random.uniform(arcs[random_arc]["start"]["location"],arcs[random_arc]["end"]["location"])
    x = np.cos(random_point) # Calculate x-coordinate of the point on the circle
    y = np.sin(random_point) # Calculate y-coordinate of the point on the circle
    direction = ((x, y)) # Construct a tuple representing the direction
    return direction

def collinear(vertices):
    d = np.array([[1,1,1], 
        [vertices[0][0], vertices[1][0], vertices[2][0]], 
        [vertices[0][1], vertices[1][1], vertices[2][1]]])
    return 0.5*abs(np.linalg.det(d))

def make_bounding_box(vertex, meters):
    """
    Creates a bounding box around a given vertex with a specified distance in meters.

    Args:
        vertex (tuple): Latitude and longitude of the vertex.
        meters (float): Distance in meters for the bounding box.

    Returns:
        list: Bounding box coordinates [north, south, east, west].

    """
    lat = vertex[0]
    lon = vertex[1]

    #Radius of the earth
    r = 6371

    #meters in kilometers
    m = meters/1000

    # Calculate bounding box coordinates
    north = lat + (m/r)*(180/np.pi)
    south = lat - (m/r)*(180/np.pi)
    east = lon + (m/r)*(180/np.pi)/np.cos(lat*np.pi/180)
    west = lon - (m/r)*(180/np.pi)/np.cos(lat*np.pi/180)

    return [north,south, east, west]


def duplicate_graphs(g, graphs, duplicate = False):
    """
    Checks if a graph is a duplicate of any graph in a list of graphs.

    Args:
        g (networkx.Graph): The graph to check for duplicates.
        graphs (list): List of graphs to compare against.
        duplicate (bool): Flag indicating whether a duplicate was found (default is False).

    Returns:
        bool: True if a duplicate is found, False otherwise.

    """
    for graph in graphs:
        if set(g.nodes()) == set(graph.nodes()) and set(g.edges()) == set(graph.edges()):
            print("Duplicate Found")
            duplicate = True
            break
    return duplicate



def find_subgraphs(G, nodes_proj, source_dir, bbox):
    """
    Finds and saves subgraphs of specified sizes from a given graph.

    Args:
        G (networkx.Graph): The input graph.
        nodes_proj (DataFrame): DataFrame containing projected node positions.
        source_dir (str): Directory to save subgraphs.
        bbox (tuple): Bounding box coordinates.

    """

    graphs4 = []
    graphs5 = []
    graphs6 = []

    # Get node positions
    x = nodes_proj['lat'].tolist()
    y = nodes_proj['lon'].tolist()
    vertices = list(zip(x, y))
    k = 0

    for vertex in vertices:
        try:
            # Create bounding box around vertex
            bb = make_bounding_box(vertex, bbox)

            # Create bounding box around vertex
            g = ox.truncate.truncate_graph_bbox(G, bbox=bb)

            # Check size of truncated graph and add to appropriate list
            if len(g.nodes()) == 4:
                if len(graphs4) == 0:
                    graphs4.append(g)
                else:
                    if not duplicate_graphs(g, graphs4):
                        graphs4.append(g)
                print("4 graphs " + str(len(graphs4)))    
            elif len(g.nodes()) == 5:
                if len(graphs5) == 0:
                    graphs5.append(g)
                else:
                    if not duplicate_graphs(g, graphs5):
                        graphs5.append(g)
                print("5 graphs " + str(len(graphs5)))
            elif len(g.nodes()) == 6:
                if len(graphs6) == 0:
                    graphs6.append(g)
                else:
                    if not duplicate_graphs(g, graphs6):
                        graphs6.append(g)
                print("6 graphs " + str(len(graphs6)))

            k += 1
            print(k)
            pass      
        except Exception as e:
            print(e.__class__, "occurred.")
            print("Next entry.") 
    
    # Save subgraphs to JSON files
    g4 = "Bozeman_4graphs_from_source_"+ str(bbox) +".json"
    g5 = "Bozeman_5graphs_from_source_"+ str(bbox) +".json"
    g6 = "Bozeman_6graphs_from_source_"+ str(bbox) +".json"
    
    save_graph_to_json(graphs4, os.path.join(source_dir, g4))
    save_graph_to_json(graphs5, os.path.join(source_dir, g5))
    save_graph_to_json(graphs6, os.path.join(source_dir, g6))


def save_graph_to_json(graphs, filename):
    """
    Saves a list of graphs to a JSON file.

    Args:
        graphs (list): List of networkx.Graph objects to be serialized.
        filename (str): Path to the output JSON file.

    """
    # Serialize each graph in the list
    serialized_graphs = []
    for graph in graphs:
        serialized_graph = nx.readwrite.json_graph.node_link_data(graph)
        serialized_graphs.append(serialized_graph)

    # Write the serialized graphs to the JSON file
    with open(filename, 'w') as f:
        json.dump(serialized_graphs, f)










