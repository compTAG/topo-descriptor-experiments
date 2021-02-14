import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import os
import topology
import dionysus as d
import planarity as p
import matplotlib.pyplot as plt
from planar_graphs import get_source_graph, find_all_planar_graphs, plot_graphs, create_graph
import itertools

class DirectionalExp(object):
  def __init__(self, G, verts, directions):
      self.graph = create_graph(verts, G.edges())
      self.main_diagram = topology.DirectionalDiagram(self.graph, (0,1))
      self.directional_diagrams = []
      self.verts = verts
      self.graphs = find_all_planar_graphs(list(itertools.combinations(G.nodes(), 2)))
      self.planar_graphs = []
      self.pos = { i : verts[i] for i in range(0, len(verts) ) }
      self.directions = directions

  def plot_planar(self):
    plot_graphs(self.graphs[1:-1])

  def clean(self):
    self.graphs.pop(0)
    for graph in self.graphs:
      g = create_graph(self.verts, graph)
      if (nx.check_planarity(g) and (g.edges() != self.graph.edges())):
        self.planar_graphs.append(g)

  def create_diagrams(self):
    for direction in directions:
      self.directional_diagrams.append(topology.DirectionalDiagram(self.graph, direction))


  def find_graphs(self,diagram):
    for graph in self.planar_graphs:
      for direction in self.directions:
        diagrams = []
        graphs = []
        dgm = topology.DirectionalDiagram(graph, direction)
        if (diagram.__eq__(dgm)):
          diagram.equal_diagrams.append(dgm)
          diagram.equal_graphs.append(graph)

  def fill(self):
    for diagram in self.directional_diagrams:
      self.find_graphs(diagram) 

  def build_graphs(self,diagram,figsize=14, dotsize=40):
    n = len(diagram.equal_graphs)
    fig = plt.figure(figsize=(figsize,figsize))
    fig.patch.set_facecolor('white')
    k = int(np.sqrt(n))
    i=0
    for g in diagram.equal_graphs:
        plt.subplot(k+1,k+1,i+1).title.set_text(diagram.equal_diagrams[i].dir)
        nx.draw(g,self.pos, edge_labels=True,node_size=dotsize)
        i = i +1
        print('.', end='')
    plt.show()

  def show_graphs(self):
    self.plot_planar()
    self.clean()
    self.create_diagrams()
    self.fill()
    for diagram in self.directional_diagrams:
      self.build_graphs(diagram)

  def bottlenecks(self):
    dgm_1 = self.directional_diagrams[0]
    dgm_2 = self.directional_diagrams[1]
    b = d.bottleneck_distance(dgm_1._dgms[0], dgm_2._dgms[0])
    print(b)



if __name__ == "__main__":
  ox.config(log_console=True, use_cache=True)
  location_point = (45.67930061221573, -111.03874239039452)
  distance = 55
  G,verts = get_source_graph(location_point,distance)

  directions = [(0,1), (1,0), (0,-1), (-1,0), (np.sqrt(2)/2,np.sqrt(2)/2)]
  test = DirectionalExp(G,verts,directions)
  test.show_graphs()
   






