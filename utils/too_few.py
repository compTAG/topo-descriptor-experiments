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
from planar_graphs import get_city_map, get_source_graph, plot_graphs, create_graph, circle_disc, is_intersection, find_planar_graphs, collinear,make_bounding_box, find_subgraphs,duplicate_graphs
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
  def __init__(self, Graph, verts):
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
    vertcombinations = set(itertools.combinations(self.verts, 3))
    for vertices in vertcombinations:
      self.test_collinearity[collinear(vertices)] = vertices

  def plot_planar(self):
    plot_graphs(self.graphs[1:-1])

  def create_planar(self):
    self.graphs.pop(0)
    for graph in self.graphs:
      g = ExpGraph(create_graph(self.verts, graph))
      if (nx.check_planarity(g.graph)):
        self.planar_graphs.append(g)
  
  def clean(self):
    for graph in self.planar_graphs:
       graph.count = 0

  def create_diagrams(self, direction):
    self.directional_diagram = topology.DirectionalDiagram(self.graph, direction)


  def find_graphs(self,diagram):
    for graph in self.planar_graphs:
      dgm = topology.DirectionalDiagram(graph.graph, diagram.dir)
      if (diagram == (dgm)):
        diagram.equal_diagrams.append(dgm)
        diagram.equal_graphs.append(graph)
        graph.count += 1

  def fill(self):
    self.find_graphs(self.directional_diagram) 
  
  def find_equals(self):
    self.equal_graphs = []
    for graph in self.planar_graphs:
      if graph.count > 0:
         self.equal_graphs.append(graph)
         

  def graph_plots(self,diagram,figsize=14, dotsize=40):
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
    for diagram in self.directional_diagrams:
      self.graph_plots(diagram)

    
  def plot_num_directions(self):
    max_count = max(map(lambda i: i.count, self.planar_graphs))
    x = np.arange(0, max_count)  
    y = np.array(self.num_directions)

    plt.title("Equiv Class vs. Directions")  
    plt.xlabel("# of Directions")  
    plt.ylabel("Size of Equiv Class")  
    plt.plot(x, y, color ="black")  
    plt.show()
    

  def planar_exp(self,direction):
    self.create_planar()
    self.create_diagrams(direction)
    self.fill()
    self.find_equals()
    

  def bottlenecks(self):
    dgm_1 = self.directional_diagrams[0]
    dgm_2 = self.directional_diagrams[1]
    b = d.bottleneck_distance(dgm_1._dgms[0], dgm_2._dgms[0])

class ExpGraph(object):
    def __init__(self, graph):
      self.graph = graph
      self.count = 0

def coars_stratification (verts, edges):
  graph = create_graph(verts, edges)
  graph.graph["stratum"] = np.zeros((len(graph.nodes()),len(graph.nodes())))
  fillangmatrix(graph.graph["stratum"], len(graph.nodes()), list(graph.nodes(data=True)))
  arcs = find_arc_lengths(graph.graph["stratum"])
  return graph, arcs

def fillangmatrix(angmatrix, n, vertlist):
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                angmatrix[i][j] = None  # fills diagonal of matrix with null, as no edge between vertex and itself
            else:
                angmatrix[i][j] = findorthangle(vertlist[i][1], vertlist[j][1])  # sets [i][j] to orthangle of line through i and j. [j][i] = [i][j] +- pi

def findorthangle(a, b):
    # get slope
    tempx = a['pos'][0] - b['pos'][0]
    tempy = a['pos'][1] - b['pos'][1]

    # get slope of perpindicular line to line segment (a,b)
    orthx = -tempy
    orthy = tempx

    # atan2 gives back the angle in [-pi, pi] that the orthogonal slope makes with the x-axis
    orthangle = math.atan2(orthy, orthx)
    # if it is negative, then we need to subtract it from 2pi so it is in [0, 2pi)
    if orthangle < 0.0:
        orthangle = 2*math.pi + orthangle
    return orthangle  # will be in radians

def find_arc_lengths(m):
    stratum_boundaries = []
    for i in range (0, len(m)):
        for j in range (0, len(m)):
            if i != j:
                stratum_boundaries.append({"location":m[i][j], "vertex1":i,
                    "vertex2":j})
    # sort by the boundary locations on the sphere (stored in radians)
    stratum_boundaries = sorted(stratum_boundaries, key=lambda i: i['location'])
    arcs = []
    for i in range(0, len(stratum_boundaries)-1):
        arc_length = 0.0
        start = stratum_boundaries[i]
        end = stratum_boundaries[i+1]
        arcs.append({"start":start,
            "end":end,
            "length":abs(start["location"]-end["location"]),
            "hit":0})
    arcs.append({"start":stratum_boundaries[len(stratum_boundaries)-1],
            "end":stratum_boundaries[0],
            "length":abs((2*math.pi -
                stratum_boundaries[len(stratum_boundaries)-1]["location"]) +
                stratum_boundaries[0]["location"]),
                "hit":0})
    return arcs


def test_gen_pos(G):
    # Test to make sure no two points share an x- or y-coord
    for c in itertools.combinations(list(G.nodes(data=True)), 2):
        if (c[0][1]['x'] == c[1][1]['x'] or
            c[0][1]['y'] == c[1][1]['y']):
            print("Shared x- and y-coords")
            return False
    # Test to make sure no three points are colinear
    for c in itertools.combinations(list(G.nodes(data=True)), 3):
        # print(str(c[0][1]['v'].get_id()) + " " + str(c[1][1]['v'].get_id()) + " "+ str(c[2][1]['v'].get_id()))
        if colin(c[0][1],c[1][1],c[2][1]):
            print("3 points colin")
            return False
    return True

def colin(x,y,z, tolerance=1e-13):
  cross_product = abs((y['x'] - x['x']) * (z['y'] - x['y']) - (y['y'] - x['y']) * (z['x'] - x['x']))
  return cross_product < tolerance

def recenter_and_rescale (V_set):
  x_values = [coord[0] for coord in V_set]
  y_values = [coord[1] for coord in V_set]
  min_x = min(x_values)
  max_x = max(x_values)
  min_y = min(y_values)
  max_y = max(y_values)

  scale_x = 1 / (max_x - min_x)
  scale_y = 1 / (max_y - min_y)

  recentered_scaled_positions = []
  for node in V_set:
      recentered_x = (node[0] - min_x) * scale_x
      recentered_y = (node[1] - min_y) * scale_y
      recentered_scaled_positions.append((recentered_x, recentered_y))

  return recentered_scaled_positions

def bar_charts(graphs_file, alphas, num_verts):
  #Set bar chart properties
  font = FontProperties()
  font.set_family('serif')
  font.set_name('Times New Roman')
  #font.set_weight('bold')

  f = plt.figure()
  labels, counts = np.unique(alphas, return_counts=True)
  plt.bar(labels, counts, align='center', color = "black")
  plt.gca().set_xticks(range(max(alphas)+1))
  plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
  plt.title('Directions on ' + str(num_verts) + ' Vertex Subgraphs',fontsize=26, fontproperties=font)
  plt.xlabel('Number of Directions',fontsize=24, fontproperties=font)
  plt.ylabel('Number of Subgraphs',fontsize=24, fontproperties=font)

  f.savefig(os.path.join("graphs","maps","Bozeman","experiments", str(num_verts)+"_graphs", str(num_verts)+"_vert_exp","exp-small-graph-"+ str(num_verts) +"-vertex-directions_" + str(bbox)+ ".pdf"), bbox_inches='tight')


def run_experiment(graphs_file, num_verts, bbox):
  random.seed(41822)
  in_file = graphs_file
  in_graphs = open(in_file,"rb")

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
  missed = 0
  with open("logfile.txt", "a") as f:
    for graph in graphs:
      directions = []
      print(k)

      #Show the graph
      #fig, ax = ox.plot_graph(graph,node_color='r',show=False, close=False)
      #plt.show()

      try:
        G,verts = get_source_graph(graph)
        verts = recenter_and_rescale(verts)
        G,arcs = coars_stratification(verts,G.edges())
        

        if (test_gen_pos(graph)==False):
          k += 1
          pass
        
        else:
          directions.append(circle_disc(arcs))
          print(G.nodes())
          exp = DirectionalExp(G,verts)
          exp.planar_exp(directions[-1])
          exp.planar_graphs = exp.equal_graphs
          while len(exp.planar_graphs) > 1:
            exp.clean()     
            direction = circle_disc(arcs)
            if direction is not None:
              
              
              directions.append(direction)
              exp.create_diagrams(directions[-1])
              exp.fill()
              exp.find_equals()
              exp.planar_graphs = exp.equal_graphs


            else:
              fig, ax = ox.plot_graph(graph,node_color='r',show=False, close=False)
              plt.show()
              missed += 1
              f.write(f'Something went wrong! Missed data count: {missed}\n')
              break
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

  print(f" Graphs with a lot of directions needed {missed}")

  #Make Bar Chart
  bar_charts(graphs_file,alphas,num_verts)

  #Save Experiment       
  exp_file = os.path.join("graphs","maps","Bozeman","experiments", str(num_verts)+"_graphs", str(num_verts)+"_vert_exp", str(num_verts) +'_nodes_experiment_' + str(bbox) +'.pickle')
  with open(exp_file, "wb") as f:
      pickle.dump(experiments, f)


if __name__ == "__main__":

  city = "Bozeman"
  state = "MT"
  country = "USA"
  vertices = 6
  bbox = 30
  source_dir = os.path.join("graphs","maps",city,"experiments")
  
  #Get subgraphs
  G, project_nodes = get_city_map(city,state, country)
  find_subgraphs(G, project_nodes, source_dir, bbox)


  #In graphs location
  graphs_file = "Bozeman_" + str(vertices) + "graphs_from_source_60.json"
  run_experiment(os.path.join(source_dir,graphs_file),vertices,bbox)
  

  
  city = "Bozeman"
  in_file = os.path.join("graphs","maps","Bozeman","experiments", str(vertices)+"_graphs", str(vertices)+"_vert_exp", str(vertices) +'_nodes_experiment_' + str(bbox) + '.pickle')
  in_exps = open(in_file,"rb")
  exps = pickle.load(in_exps)

  
  for e in exps:
    if e.alpha >= 12:
      val_map = {}
      collinear_val = min(e.test_collinearity.keys())
      for node in e.test_collinearity.get(collinear_val):
        val_map[list(e.pos.keys())[list(e.pos.values()).index(node)]] = 'white'
    
      
      values = [val_map.get(node, 'red') for node in e.graph.nodes()]
      font = FontProperties()
      font.set_family('serif')
      font.set_name('Times New Roman')
      f = plt.figure()
      plt.title(str(e.alpha) + " Directions ")
      nx.draw(e.graph, e.pos, edgecolors='black', node_size = 50, cmap=plt.get_cmap('hot'), node_color=values)
      #plt.show()
      f.savefig(os.path.join("graphs","maps","Bozeman","experiments", "collinear_graphs", str(vertices)+"_nodes",str(collinear_val)+".pdf"), bbox_inches='tight')
  



