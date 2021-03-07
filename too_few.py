import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import os
import topology
import dionysus as d
import planarity as p
import matplotlib.pyplot as plt
from planar_graphs import get_source_graph, plot_graphs, create_graph, circle_disc, is_intersection, find_planar_graphs
import itertools
import pickle
import glob
import functools
import math
import random

class DirectionalExp(object):
  def __init__(self, G, verts, directions):
      self.graph = create_graph(verts, G.edges())
      self.directional_diagrams = []
      self.verts = verts
      self.graphs = find_planar_graphs(self.verts,list(itertools.combinations(G.nodes(), 2)))
      self.planar_graphs = []
      self.pos = { i : verts[i] for i in range(0, len(verts) ) }
      self.directions = directions
      self.num_directions = []
      self.alpha = 0

  def plot_planar(self):
    plot_graphs(self.graphs[1:-1])


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

    for n in range(max_count):
      self.num_directions.append(sum(map(lambda i: i.count >= n, self.planar_graphs)))
    
    self.alpha = len(self.num_directions)-1
    
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
    self.clean()
    self.create_diagrams()
    self.fill()

  def bottlenecks(self):
    dgm_1 = self.directional_diagrams[0]
    dgm_2 = self.directional_diagrams[1]
    b = d.bottleneck_distance(dgm_1._dgms[0], dgm_2._dgms[0])

class ExpGraph(object):
    def __init__(self, graph):
      self.graph = graph
      self.count = 0


if __name__ == "__main__":
  
  source_dir = os.path.join("graphs","maps","Bozeman","graphs","4_graphs", "4_nodes.pickle")
  in_graph = open(source_dir,"rb")
  graphs = pickle.load(in_graph)
  print(len(graphs))
  
  
  d = random.uniform(0, 2*np.pi)
  print(d)
  
  alphas = []
  experiments = []
  for graph in graphs:
    i = 1
    try:
      directions = circle_disc(d,i)
      G,verts = get_source_graph(graph)
      exp = DirectionalExp(G,verts,directions)
      exp.planar_exp()
      exp.find_num_directions()
      
      while exp.num_directions[-1] > 1:
        i += 1
        directions = circle_disc(d,i)
        exp = DirectionalExp(G,verts,directions)
        exp.planar_exp()
        exp.find_num_directions()
      experiments.append(exp)
      alphas.append(exp.alpha)
      pass
    except Exception as e:
          print(e.__class__, "occurred.")
          print(directions)
          #fig, ax = ox.plot_graph(G,node_color='r',show=False, close=False)
          #plt.show()
          print("Next entry.")


  bins = np.linspace(math.ceil(min(alphas)), 
                   math.floor(max(alphas)),
                   20) # fixed number of bins

  plt.xlim([min(alphas)-2, max(alphas)+2])

  plt.hist(alphas, bins=bins, alpha=0.5)
  plt.title('Alpha Values on 4 Node Subgraphs (Random Equally Spaced Directions on Unit Circle)')
  plt.xlabel('Alpha (Number of Directions)')
  plt.ylabel('count')

  plt.show()

  exp_file = os.path.join("graphs","maps","Bozeman","graphs","4_graphs", "4_nodes_exp_rand_4.pickle")
  with open(exp_file, "wb") as f:
      pickle.dump(experiments, f)

  
  '''
  r = random.uniform(0, 2*np.pi)
  print(r)

  direc = disc(r, 1)
  print(direc)

  source_dir = os.path.join("graphs","maps","Bozeman","graphs","4_graphs", "4_nodes.pickle")
  in_graph = open(source_dir,"rb")
  graphs = pickle.load(in_graph)


  G,verts = get_source_graph(graphs[3])
  exp = DirectionalExp(G,verts,[(-0.6141074713467367, -0.7892224107538488)])
  exp.planar_exp()
  exp.find_num_directions()
  '''
  
  




  


  
