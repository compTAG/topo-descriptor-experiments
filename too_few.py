import argparse
import sys
import json
import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import os
import topology
import dionysus as d
import matplotlib.pyplot as plt
import matplotlib
from utils.planar_graphs import get_city_map, get_source_graph, plot_graphs, create_graph, circle_disc, find_planar_graphs, collinear, find_subgraphs
import itertools
import pickle
import glob
import functools
import math
import random
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
from PyPDF2 import PdfFileMerger, PdfFileReader

class DirectionalExp(object):
  """
  Class for conducting directional exploration on a graph.

    This class facilitates the exploration of a graph in a specific direction.

    Attributes:
        graph (networkx.Graph): The input graph.
        directional_diagram: The directional diagram generated during exploration.
        verts (list): List of vertices in the graph.
        graphs (list): List of all possible planar graphs derived from the input graph.
        planar_graphs (list): List of planar graphs identified during exploration.
        equal_graphs (list): List of graphs with equal elements.
        alpha (int): A parameter used in exploration.
        pos (dict): Dictionary containing positions of vertices.
        test_collinearity (dict): Dictionary for testing collinearity during exploration.

    Methods:
        __init__: Initialize the DirectionalExp object.
        collinear_points: Identify sets of collinear points in the graph.
        plot_planar: Plot planar graphs derived from the input graph.
        create_planar: Create planar graphs from the input graph.
        clean: Reset count attribute for each planar graph.
        create_diagrams: Create directional diagrams based on a given direction.
        find_graphs: Find graphs that match a given diagram.
        fill: Populate diagrams with matching graphs.
        find_equals: Identify graphs with equal elements.
        graph_plots: Plot equal graphs corresponding to a given diagram.
        plot_equal_graphs: Plot all equal graphs.
        plot_num_directions: Plot the number of directions vs. the size of equivalent classes.
        planar_exp: Perform planar exploration using a given direction.
        bottlenecks: Calculate bottleneck distances between two diagrams.
  """
  def __init__(self, Graph, verts):
      """
      Initialize the DirectionalExp object.

        Parameters:
            Graph (networkx.Graph): The input graph.
            verts (list): List of vertices in the graph.

        Returns:
            None
      """
      self.graph = Graph
      self.directional_diagram = None
      self.verts = verts
      self.graphs = find_planar_graphs(self.verts,list(itertools.combinations(Graph.nodes(), 2)))
      self.planar_graphs = []
      self.equal_graphs = []
      self.alpha=0
      self.pos = { i : verts[i] for i in range(0, len(verts) ) }
      self.test_collinearity = {}

    
  def collinear_points(self):
    """
    Identify sets of collinear points in the graph.

        Returns:
            None
    """
    vertcombinations = set(itertools.combinations(self.verts, 3))
    for vertices in vertcombinations:
      self.test_collinearity[collinear(vertices)] = vertices

  def plot_planar(self):
    """
    Plot planar graphs derived from the input graph.

        Returns:
            None
    """
    plot_graphs(self.graphs[1:-1])

  def create_planar(self):
    """
    Create planar graphs from the input graph.

        Returns:
            None
    """

    self.graphs.pop(0)

    for graph in self.graphs:
      # Create an ExpGraph object from the current planar graph
      g = ExpGraph(create_graph(self.verts, graph))
      # Check if the graph is planar
      if (nx.check_planarity(g.graph)):
        # If planar, append it to the list of planar graphs
        self.planar_graphs.append(g)
  
  def clean(self):
    """
    Reset count attribute for each planar graph.

        Returns:
            None
    """
    for graph in self.planar_graphs:
       graph.count = 0

  def create_diagrams(self, direction):
    """
    Create directional diagrams based on a given direction.

        Parameters:
            direction (tuple): A tuple representing the direction.

        Returns:
            None
    """
    self.directional_diagram = topology.DirectionalDiagram(self.graph, direction)


  def find_graphs(self,diagram):
    """
    Find graphs that match a given diagram.

        Parameters:
            diagram: The diagram to match.

        Returns:
            None
    """
    for graph in self.planar_graphs:
      dgm = topology.DirectionalDiagram(graph.graph, diagram.dir)
      if (diagram == (dgm)):
        diagram.equal_diagrams.append(dgm)
        diagram.equal_graphs.append(graph)
        graph.count += 1

  def fill(self):
    """
    Populate diagrams with matching graphs.

        Returns:
            None
    """
    self.find_graphs(self.directional_diagram) 
  
  def find_equals(self):
    """
    Identify graphs with equal elements.

        Returns:
            None
    """
    self.equal_graphs = []
    for graph in self.planar_graphs:
      if graph.count > 0:
         self.equal_graphs.append(graph)
         

  def graph_plots(self,diagram,figsize=14, dotsize=40):
    """
    Plot equal graphs corresponding to a given diagram.

        Parameters:
            diagram: The diagram to plot.
            figsize (int): Size of the figure.
            dotsize (int): Size of the nodes in the graph.

        Returns:
            None
    """
    n = len(diagram.equal_graphs)
    fig = plt.figure(figsize=(figsize,figsize))
    fig.patch.set_facecolor('white')
    k = int(np.sqrt(n))
    i=0
    for g in diagram.equal_graphs:
        plt.subplot(k+1,k+1,i+1).title.set_text(diagram.equal_diagrams[i].dir)
        nx.draw(g.graph,self.pos, edge_labels=True,node_size=dotsize)
        i = i +1
        print('.', end='')
    plt.show()

  def plot_equal_graphs(self):
    """
    Plot all equal graphs.

        Returns:
            None
    """
    for diagram in self.directional_diagrams:
      self.graph_plots(diagram)

    
  def plot_num_directions(self):
    """
    Plot the number of directions vs. the size of equivalent classes.

        Returns:
            None
    """
    max_count = max(map(lambda i: i.count, self.planar_graphs))
    x = np.arange(0, max_count)  
    y = np.array(self.num_directions)

    plt.title("Equiv Class vs. Directions")  
    plt.xlabel("# of Directions")  
    plt.ylabel("Size of Equiv Class")  
    plt.plot(x, y, color ="black")  
    plt.show()
    

  def planar_exp(self,direction):
    """
    Perform a planar exploration using a given direction.

    This method orchestrates the steps required for planar exploration:
    1. Create a planar structure.
    2. Create diagrams based on the given direction.
    3. Fill the created diagrams.
    4. Find and handle equal elements.

    Parameters:
        direction (tuple): A tuple representing the direction of exploration.

    Returns:
        None
    """

    # Step 1: Create a planar structure
    self.create_planar()

    # Step 2: Create diagrams based on the given direction
    self.create_diagrams(direction)

    # Step 3: Fill the created diagrams
    self.fill()

    # Step 4: Find and handle equal elements
    self.find_equals()
    

  def bottlenecks(self):
    """
    Calculate bottleneck distances between two diagrams.

        Returns:
            None
    """
    dgm_1 = self.directional_diagrams[0]
    dgm_2 = self.directional_diagrams[1]
    b = d.bottleneck_distance(dgm_1._dgms[0], dgm_2._dgms[0])

class ExpGraph(object):
    def __init__(self, graph):
      self.graph = graph
      self.count = 0

def coars_stratification (verts, edges):
  """
  Constructs a coarse stratification from a set of vertices and edges.

    Args:
        verts (list): List of vertex positions.
        edges (list): List of graph edges.

    Returns:
        tuple: A tuple containing the graph representation of the stratification and the arc lengths.

  """
  # Create graph from vertices and edges
  graph = create_graph(verts, edges)

  # Initialize stratum attribute for the graph
  graph.graph["stratum"] = np.zeros((len(graph.nodes()),len(graph.nodes())))

  # Fill angular matrix for the stratum
  fillangmatrix(graph.graph["stratum"], len(graph.nodes()), list(graph.nodes(data=True)))

  # Find arc lengths
  arcs = find_arc_lengths(graph.graph["stratum"])

  return graph, arcs

def fillangmatrix(angmatrix, n, vertlist):
    """
    Fills the angular matrix with angles between edges connecting vertices.

    Args:
        ang_matrix (numpy.ndarray): Angular matrix to be filled.
        n (int): Number of vertices.
        vert_list (list): List of vertices with their positions.

    """

    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                angmatrix[i][j] = None  # fills diagonal of matrix with null, as no edge between vertex and itself
            else:
                angmatrix[i][j] = findorthangle(vertlist[i][1], vertlist[j][1])  # sets [i][j] to orthangle of line through i and j. [j][i] = [i][j] +- pi

def findorthangle(a, b):
    """
    Finds the orthogonal angle between two points.

    Args:
        a (dict): Dictionary containing position information for point a.
        b (dict): Dictionary containing position information for point b.

    Returns:
        float: Orthogonal angle between the two points, in radians.
    """
    # Get the difference in x and y coordinates
    tempx = a['pos'][0] - b['pos'][0]
    tempy = a['pos'][1] - b['pos'][1]

    # Calculate the slope of the perpendicular line to the line segment (a, b)
    orthx = -tempy
    orthy = tempx

    # Use atan2 to find the angle in [-pi, pi] that the orthogonal slope makes with the x-axis
    orthangle = math.atan2(orthy, orthx)

    # If the angle is negative, adjust it to be in [0, 2pi)
    if orthangle < 0.0:
        orthangle = 2*math.pi + orthangle

    return orthangle  # Will be in radians

def find_arc_lengths(m):
    """
    Finds arc lengths between stratum boundaries.

    Args:
        m (numpy.ndarray): Matrix representing stratum boundaries.

    Returns:
        list: List of arcs with their start, end, length, and hit properties.
    """

    stratum_boundaries = []
    # Iterate over the matrix to find stratum boundaries
    for i in range (0, len(m)):
        for j in range (0, len(m)):
            if i != j:
                stratum_boundaries.append({"location":m[i][j], "vertex1":i,
                    "vertex2":j})

    # Sort the boundaries by their locations on the sphere (stored in radians)
    stratum_boundaries = sorted(stratum_boundaries, key=lambda i: i['location'])

    arcs = []

    # Iterate over the sorted boundaries to calculate arc lengths
    for i in range(0, len(stratum_boundaries)-1):
        arc_length = 0.0
        start = stratum_boundaries[i]
        end = stratum_boundaries[i+1]
        arcs.append({"start":start,
            "end":end,
            "length":abs(start["location"]-end["location"]),
            "hit":0})
    
    # Calculate arc length for the last boundary to the first boundary
    arcs.append({"start":stratum_boundaries[len(stratum_boundaries)-1],
            "end":stratum_boundaries[0],
            "length":abs((2*math.pi -
                stratum_boundaries[len(stratum_boundaries)-1]["location"]) +
                stratum_boundaries[0]["location"]),
                "hit":0})
    return arcs


def test_gen_pos(G):
    """
    Function to test if the given graph layout is valid.

    Parameters:
        G (networkx.Graph): The graph to be tested.

    Returns:
        bool: True if the graph layout is valid, False otherwise.
    """

    # Test to make sure no two points share an x- or y-coord
    for c in itertools.combinations(list(G.nodes(data=True)), 2):
        if (c[0][1]['x'] == c[1][1]['x'] or
            c[0][1]['y'] == c[1][1]['y']):
            print("Shared x- and y-coords")
            return False
        
    # Test to make sure no three points are colinear
    for c in itertools.combinations(list(G.nodes(data=True)), 3):
        if colin(c[0][1],c[1][1],c[2][1]):
            print("3 points colin")
            return False

    return True

def colin(x,y,z):
  """
  Check if three points are collinear within a certain tolerance.

    Parameters:
        x (dict): Dictionary representing the first point with 'x' and 'y' coordinates.
        y (dict): Dictionary representing the second point with 'x' and 'y' coordinates.
        z (dict): Dictionary representing the third point with 'x' and 'y' coordinates.
        tolerance (float): Tolerance value for considering points collinear. Default is 1e-13.

    Returns:
        bool: True if the points are collinear within the given tolerance, False otherwise.
  """
  cross_product = abs((y['x'] - x['x']) * (z['y'] - x['y']) - (y['y'] - x['y']) * (z['x'] - x['x']))

  return cross_product == 0

def recenter_and_rescale (V_set):
  """
  Recenter and rescale a set of vertex positions to fit within the range [0, 1].

    Args:
        V_set (list): List of vertex positions.

    Returns:
        list: Recentered and rescaled vertex positions.
  """
  # Extract x and y coordinates from vertex positions
  x_values = [coord[0] for coord in V_set]
  y_values = [coord[1] for coord in V_set]

  # Calculate minimum and maximum values for x and y coordinates
  min_x = min(x_values)
  max_x = max(x_values)
  min_y = min(y_values)
  max_y = max(y_values)

  # Calculate scaling factors
  scale_x = 1 / (max_x - min_x)
  scale_y = 1 / (max_y - min_y)

  # Recenter and rescale vertex positions
  recentered_scaled_positions = []
  for node in V_set:
      recentered_x = (node[0] - min_x) * scale_x
      recentered_y = (node[1] - min_y) * scale_y
      recentered_scaled_positions.append((recentered_x, recentered_y))

  return recentered_scaled_positions

def bar_charts(graphs_file, alphas, num_verts):
  """
  Generate bar charts based on the number of directions.

    This function creates a bar chart to visualize the distribution of the number of directions
    across different subgraphs.

    Parameters:
        graphs_file (str): Path to the file containing graphs data.
        alphas (list): List of numbers representing the number of directions for each subgraph.
        num_verts (int): Number of vertices in the subgraphs.

    Returns:
        None
  """

  #Set bar chart properties
  font = FontProperties()
  font.set_family('serif')
  font.set_name('Times New Roman')

  f = plt.figure()
  labels, counts = np.unique(alphas, return_counts=True)
  plt.bar(labels, counts, align='center', color = "black")
  plt.gca().set_xticks(range(max(alphas)+1))
  plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
  plt.xlabel('Number of Directions',fontsize=24, fontproperties=font)
  plt.ylabel('Number of Subgraphs',fontsize=24, fontproperties=font)
  plt.xticks(fontsize=20, fontproperties=font)
  plt.yticks(fontsize=20, fontproperties=font)

  file_path = os.path.join(
     "graphs","maps","Bozeman","experiments", 
     str(num_verts)+"_graphs",str(num_verts)+"_vert_exp",
     "exp-small-graph-"+ str(num_verts) +"-vertex-directions_" + str(bbox)+ ".pdf")
  
  # Create directories if they do not exist
  os.makedirs(os.path.dirname(file_path), exist_ok=True)

  f.savefig(file_path, bbox_inches='tight')


def run_experiment(graphs_file, num_verts, bbox):

  """
    Runs an experiment for each graph loaded from JSON files.

    Args:
        graphs_file (str): Path to the JSON file containing graph data.
        num_verts (int): Number of vertices.
        bbox (tuple): Bounding box coordinates.

    """

  #Load data from JSON files
  random.seed(41822)
  in_file = graphs_file
  with open(in_file, "r") as f:
        graphs_data = json.load(f)
  graphs = []
  for graph_data in graphs_data:
      graph = nx.readwrite.json_graph.node_link_graph(graph_data)
      graphs.append(graph)
  print(len(graphs))
  k = 0
  alphas = []
  experiments = []

  #Run experiment for each graph
  for graph in graphs:
    directions = []
    print(k)
    try:
        
        #Get vertices and graph from loaded data
        G,verts = get_source_graph(graph)

        #Rescale data into [0,1]^2 box
        verts = recenter_and_rescale(verts)

        #Calculate coarse stratification for data
        G,arcs = coars_stratification(verts,G.edges())

        #Check if general position assumption is met.
        if (test_gen_pos(graph)==False):
          k += 1
          pass
        else:

          #Get first direction
          directions.append(circle_disc(arcs))
          print(G.nodes())

          #Create experiment
          exp = DirectionalExp(G,verts)
          exp.planar_exp(directions[-1])
          exp.planar_graphs = exp.equal_graphs

          #Continue sampling directions until we ca identify graph amoung set of all posible plane graphs
          while len(exp.planar_graphs) > 1:
            exp.clean()     
            direction = circle_disc(arcs)                         
            directions.append(direction)
            exp.create_diagrams(directions[-1])
            exp.fill()
            exp.find_equals()
            exp.planar_graphs = exp.equal_graphs
          exp.alpha = len(directions)
          exp.collinear_points()
          experiments.append(exp)
          alphas.append(exp.alpha)
          k += 1
          pass
    except Exception as e:
            k += 1
            print(e.__class__, "occurred.")
            print("Next entry.")

  #Make Bar Chart
  bar_charts(graphs_file,alphas,num_verts)

  #Save Experiment       
  exp_file = os.path.join("graphs","maps","Bozeman","experiments", str(num_verts)+"_graphs", str(num_verts)+"_vert_exp", str(num_verts) +'_nodes_experiment_' + str(bbox) +'.pickle')
  with open(exp_file, "wb") as f:
      pickle.dump(experiments, f)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Run the Too Few experiment')
  parser.add_argument('--bbox', type=int, required=True, help='Bounding box value')
  parser.add_argument('--number_of_vertices', type=int, required=True, help='Number of vertices')

  args = parser.parse_args()

  bbox = args.bbox
  vertices = args.number_of_vertices

  city = "Bozeman"
  state = "MT"
  country = "USA"
  source_dir = os.path.join("graphs","maps",city,"experiments")
  
  #Get subgraphs
  #G, project_nodes = get_city_map(city,state, country)
  #find_subgraphs(G, project_nodes, source_dir, bbox)


  #In graphs location
  graphs_file = "Bozeman_" + str(vertices) + f"graphs_from_source_{bbox}.json"
  run_experiment(os.path.join(source_dir,graphs_file),vertices,bbox)
  
  #Plot graphs with the number of directions greater or equal to seven
  in_file = os.path.join("graphs","maps","Bozeman","experiments", str(vertices)+"_graphs", str(vertices)+"_vert_exp", str(vertices) +'_nodes_experiment_' + str(bbox) + '.pickle')
  in_exps = open(in_file,"rb")
  exps = pickle.load(in_exps)

  # Iterate over each experiment
  for e in exps:
    # Check if alpha value is greater than or equal to 7
    if e.alpha >= 6:
      val_map = {}
      # Find the minimum collinear value
      collinear_val = min(e.test_collinearity.keys())
      # Map nodes with collinear values to white color
      for node in e.test_collinearity.get(collinear_val):
        val_map[list(e.pos.keys())[list(e.pos.values()).index(node)]] = 'white'
    
      # Assign colors to nodes based on collinearity
      values = [val_map.get(node, 'red') for node in e.graph.nodes()]
      font = FontProperties()
      font.set_family('serif')
      font.set_name('Times New Roman')
      f = plt.figure()
      plt.title(str(e.alpha) + " Directions ")
      # Draw the graph with nodes colored according to collinearity
      nx.draw(e.graph, e.pos, edgecolors='black', node_size = 50, cmap=plt.get_cmap('hot'), node_color=values)
      # Save the figure
      file_path = os.path.join("graphs","maps","Bozeman","experiments",
                                "collinear_graphs", str(vertices)+"_nodes",str(collinear_val)+".pdf")
      
      os.makedirs(os.path.dirname(file_path), exist_ok=True)
      
      f.savefig(file_path, bbox_inches='tight')
  



