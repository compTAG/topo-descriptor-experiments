import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import os
import topology
import dionysus as d
import matplotlib.pyplot as plt
import matplotlib
#from osmnx_requests import make_bounding_box, find_subgraphs,duplicate_graphs
from planar_graphs import get_city_map, get_source_graph, plot_graphs, create_graph, circle_disc, is_intersection, find_planar_graphs, collinear,make_bounding_box, find_subgraphs,duplicate_graphs
import itertools
import pickle
import glob
import functools
import math
import random
from matplotlib.font_manager import FontProperties
from PyPDF2 import PdfFileMerger, PdfFileReader

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
      self.test_collinearity = {}

    
  def collinear_points(self):
    vertcombinations = set(itertools.combinations(self.verts, 3))
    for vertices in vertcombinations:
      self.test_collinearity[collinear(vertices)] = vertices

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
  plt.title('Directions on ' + str(num_verts) + ' Vertex Subgraphs',fontsize=26, fontproperties=font)
  plt.xlabel('Number of Directions',fontsize=24, fontproperties=font)
  plt.ylabel('Number of Subgraphs',fontsize=24, fontproperties=font)

  f.savefig(os.path.join("graphs","maps","Bozeman","experiments", str(num_verts)+"_graphs", str(num_verts)+"_vert_exp","exp-small-graph-"+ str(num_verts) +"-vertex-directions_" + str(bbox)+ ".pdf"), bbox_inches='tight')


def run_experiment(graphs_file, num_verts, bbox):
  random.seed(41821)
  #in_graphs = glob.glob(graph_files + "/*.pickle")
  in_file = graphs_file
  in_graphs = open(in_file,"rb")
  graphs = pickle.load(in_graphs)
  print(len(graphs))
  d = random.uniform(0, 2*np.pi)
  print(d)
  k = 0
  alphas = []
  experiments = []
  for graph in graphs:
    i = 1
    print(k)
    try:
      #in_graph = open(file,"rb")
      #graph = pickle.load(in_graph)
      directions = circle_disc(d,i)
      G,verts = get_source_graph(graph)
      print(G.nodes())
      #fig, ax = ox.plot_graph(G,node_color='r',show=False, close=False)
      #plt.show()
      exp = DirectionalExp(G,verts,directions)
      exp.planar_exp()
      exp.find_num_directions()
      while exp.num_directions[-1] > 1:
        i += 1
        directions = circle_disc(d,i)
        exp = DirectionalExp(G,verts,directions)
        exp.planar_exp()
        exp.find_num_directions()
      exp.collinear_points()
      experiments.append(exp)
      alphas.append(exp.alpha)
      k += 1
      pass
    except Exception as e:
          print(e.__class__, "occurred.")
          print("Next entry.")

  #Make Bar Chart
  bar_charts(graphs_file,alphas,num_verts,bbox)

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
  #G, project_nodes = get_city_map(city,state, country)
  #find_subgraphs(G, project_nodes, source_dir, bbox)


  #In graphs location
  graphs_file = "Bozeman_" + str(vertices) + "graphs_from_source_30.pickle"
  run_experiment(os.path.join(source_dir,graphs_file),vertices,bbox)
  

  
  city = "Bozeman"
  in_file = os.path.join("graphs","maps","Bozeman","experiments", str(vertices)+"_graphs", str(vertices)+"_vert_exp", str(vertices) +'_nodes_experiment_' + str(bbox) + '.pickle')
  in_exps = open(in_file,"rb")
  exps = pickle.load(in_exps)

  
  for e in exps:
    if e.alpha >= 5:
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
  



