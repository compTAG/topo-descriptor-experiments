import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import os
import topology
import dionysus as d
import planarity as p
import matplotlib.pyplot as plt
from planar_graphs import get_source_graph, plot_graphs, create_graph, circle_disc, is_intersection, clean_planar,find_all_planar_graphs, make_graphs
import itertools
import pickle
import glob
import functools
import math
import random

class DirectionalExp(object):
  def __init__(self, G, verts, directions):
      self.graph = create_graph(verts, G.edges())
      self.main_diagram = topology.DirectionalDiagram(self.graph, (0,1))
      self.directional_diagrams = []
      self.verts = verts
      self.planar = find_all_planar_graphs(list(itertools.combinations(G.nodes(), 2)))
      self.graphs = clean_planar(self.verts, self.planar)
      #self.graphs = make_graphs(len(G.nodes()))
      self.planar_graphs = []
      self.pos = { i : verts[i] for i in range(0, len(verts) ) }
      self.directions = directions
      self.num_directions = []
      self.alpha = 0

  def plot_planar(self):
    plot_graphs(self.graphs[1:-1])
  #and (g.edges() != self.graph.edges()
  def clean(self):
    self.graphs.pop(0)
    for graph in self.graphs:
      g = ExpGraph(create_graph(self.verts, graph))
      if (nx.check_planarity(g.graph)):
        self.planar_graphs.append(g)

  def create_diagrams(self):
    for direction in self.directions:
      self.directional_diagrams.append(topology.DirectionalDiagram(self.graph, direction))


  def find_graphs(self,diagram):
    for graph in self.planar_graphs:
      for direction in self.directions:
        diagrams = []
        graphs = []
        dgm = topology.DirectionalDiagram(graph.graph, direction)
        if (diagram == (dgm)):
          diagram.equal_diagrams.append(dgm)
          diagram.equal_graphs.append(graph)
          graph.count += 1

  def fill(self):
    for diagram in self.directional_diagrams:
      self.find_graphs(diagram) 

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

  def find_num_directions(self):
    max_count = max(map(lambda i: i.count, self.planar_graphs))
    #print(max_count)
    for n in range(max_count):
      self.num_directions.append(sum(map(lambda i: i.count >= n, self.planar_graphs)))
    
    self.alpha = len(self.num_directions)-1
    #if len(self.directions) > 1:
    #self.alpha = self.num_directions[-1]
    #print(self.num_directions)
      #self.num_directions.index(1)
    
  def plot_num_directions(self):
    max_count = max(map(lambda i: i.count, self.planar_graphs))
    x = np.arange(0, max_count)  
    y = np.array(self.num_directions)

    plt.title("Equiv Class vs. Directions")  
    plt.xlabel("# of Directions")  
    plt.ylabel("Size of Equiv Class")  
    plt.plot(x, y, color ="black")  
    plt.show()
    

  def planar_exp(self):
    #self.plot_planar()
    self.clean()
    self.create_diagrams()
    self.fill()
    #self.find_num_directions()

  def bottlenecks(self):
    dgm_1 = self.directional_diagrams[0]
    dgm_2 = self.directional_diagrams[1]
    b = d.bottleneck_distance(dgm_1._dgms[0], dgm_2._dgms[0])
    print(b)

class PlanarGraph(object):
  def __init__(self, graph, directions):
    self.graph = graph.graph
    self.diagrams = []
    self.directions = directions

  def build_diagrams(self):
    for direction in self.directions:
      dgm = topology.DirectionalDiagram(self.graph, direction)
      self.diagrams.append(dgm)

class ExpGraph(object):
    def __init__(self, graph):
      self.graph = graph
      self.count = 0


if __name__ == "__main__":
  '''
  source_dir = os.path.join("graphs","maps","Bozeman","graphs","5_graphs", "5_nodes.pickle")
  in_graph = open(source_dir,"rb")
  graphs = pickle.load(in_graph)
  
  experiment = []
  directions = circle_disc(10)
  for graph in graphs:
    try:
      G,verts = get_source_graph(graph)
      exp = DirectionalExp(G,verts,directions)
      exp.planar_exp()
      #exp.find_num_directions()
      experiment.append(exp.alpha)
      pass
    except Exception as e:
          print(e.__class__, "occurred.")
          print(exp.num_directions)
          #fig, ax = ox.plot_graph(G,node_color='r',show=False, close=False)
          #plt.show()
          print("Next entry.")


  bins = np.linspace(math.ceil(min(experiment)), 
                   math.floor(max(experiment)),
                   20) # fixed number of bins

  plt.xlim([min(experiment), max(experiment)+2])

  plt.hist(experiment, bins=bins, alpha=0.5)
  plt.title('Alpha Values on 5 Node Subgraphs (10 Directions on Unit Circle)')
  plt.xlabel('Alpha')
  plt.ylabel('count')

  plt.show()
  '''

  
  source_dir = os.path.join("graphs","maps","Bozeman","graphs","5_graphs", "5_nodes.pickle")
  in_graph = open(source_dir,"rb")
  graphs = pickle.load(in_graph)
  
  alphas = []
  experiments = []
  for graph in graphs:
    i= 1
    try:
      directions = circle_disc(i)
      G,verts = get_source_graph(graph)
      exp = DirectionalExp(G,verts,directions)
      exp.planar_exp()
      exp.find_num_directions()
      #print(exp.num_directions[-1])
      while exp.num_directions[-1] > 1:
        #print(i)
        i += 1
        directions = circle_disc(i)
        exp = DirectionalExp(G,verts,directions)
        exp.planar_exp()
        exp.find_num_directions()
      experiments.append(exp)
      alphas.append(exp.alpha)
      pass
    except Exception as e:
          print(e.__class__, "occurred.")
          print(exp.num_directions)
          #fig, ax = ox.plot_graph(G,node_color='r',show=False, close=False)
          #plt.show()
          print("Next entry.")


  bins = np.linspace(math.ceil(min(alphas)), 
                   math.floor(max(alphas)),
                   20) # fixed number of bins

  plt.xlim([min(alphas), max(alphas)+2])

  plt.hist(alphas, bins=bins, alpha=0.5)
  plt.title('Alpha Values on 5 Node Subgraphs (Equally Spaced Directions on Unit Circle)')
  plt.xlabel('Alpha (Number of Directions)')
  plt.ylabel('count')

  plt.show()

  exp_file = os.path.join("graphs","maps","Bozeman","graphs","5_graphs", "5_nodes_exp.pickle")
  with open(exp_file, "wb") as f:
      pickle.dump(experiments, f)
  
  
  '''
  source_dir = os.path.join("graphs","maps","Bozeman","graphs","4_graphs", "4_nodes.pickle")
  in_graph = open(source_dir,"rb")
  graphs = pickle.load(in_graph)
  planar_dgms = []
  directions = circle_disc(4)
  #print(directions)
  
  
  directions = [(1.0,0.0),(np.sqrt(3)/2, 1/2), (np.sqrt(2)/2,np.sqrt(2)/2), 
                (1/2,np.sqrt(3)/2),(0,1),(-1/2,np.sqrt(3)/2),(-np.sqrt(2)/2,np.sqrt(2)/2),
                (-np.sqrt(3)/2, 1/2), (-1,0),(-np.sqrt(3)/2, -1/2),(-np.sqrt(2)/2,-np.sqrt(2)/2),
                (-1/2,-np.sqrt(3)/2), (0,-1),(1/2,-np.sqrt(3)/2),(-np.sqrt(2)/2,-np.sqrt(2)/2),
                (np.sqrt(3)/2, -1/2)]
  
  direction = [(1.0,0.0)]
  print(direction)
  
  G,verts = get_source_graph(graphs[3])
  exp = DirectionalExp(G,verts,directions)
  exp.planar_exp()
  exp.find_num_directions()
  #exp.plot_equal_graphs()
  #exp.plot_num_directions()
  print(exp.alpha)
  print(exp.num_directions)

  
  for pg in exp.planar_graphs:
    p = PlanarGraph(pg,directions)
    p.build_diagrams()
    planar_dgms.append(p)

  print(len(planar_dgms))
  for p_d in planar_dgms:
    if functools.reduce(lambda i, j : i and j, map(lambda m, k: m == k, p_d.diagrams, exp.directional_diagrams), True):  
      print ("The lists are identical") 
    else: 
      print ("The lists are not identical") 
  '''


  



  


  
