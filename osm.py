import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import requests
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import math
import planar
from planar.line import LineSegment
from planar.vector import Vec2
from shapely.geometry import Polygon
import pickle
from operator import itemgetter, attrgetter, methodcaller

class Node:
    def __init__(self, osmid, x, y, edges):
        self.osmid = osmid
        self.x = x
        self.y = y
        self.edges = edges
        self.theta = 0
        self.angles = []
        self.distances = []

    def get_osmid(self):
        return self.osmid

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_edges(self):
        return self.edges

    def get_theta(self):
        return self.theta

    def delete_edges(self):
        #Delete duplicate edges if they exist
        for e_1, e_2 in zip(self.edges, self.edges[1:]):
            if(e_1.osmid == e_2.osmid):
                e_2.osmid = -1

        bad_edges = []
        for i in range(len(self.edges)):
            if (self.edges[i].get_osmid() == -1):
                bad_edges.append("e"+str(i))
            
        self.edges.drop(labels=bad_edges,inplace=True)


    def find_thetas(self):
        for edge in self.edges:
            x_1 = edge.x.values[0]
            y_1 = edge.y.values[0]

            x_2 = x_1 - self.x.values[0]
            y_2 = y_1 - self.y.values[0]
            edge.theta = cart2pol(x_2,y_2)
        self.edges = sorted(self.edges, key=lambda x: x.theta, reverse=True)

    def get_angles(self):
        for e_1, e_2 in zip(self.edges, self.edges[1:]):
                vert = [self.x.values[0], self.y.values[0]]
                p_1 = [e_1.x.values[0], e_1.y.values[0]]
                p_2 = [e_2.x.values[0], e_2.y.values[0]]

                #print(np.subtract(p_1, vert))
                
                a = np.subtract(p_1, vert)
                b = np.subtract(p_2, vert)

                dot = np.dot(a,b)
                mag_a = np.linalg.norm(a)
                mag_b = np.linalg.norm(b)
                radians = math.acos((dot)/(mag_a*mag_b))
                degrees = math.degrees(radians)
                self.angles.append(degrees)
        for e_1, e_2 in zip(self.edges, self.edges[-1:]):
                vert = [self.x.values[0], self.y.values[0]]
                p_1 = [e_1.x.values[0], e_1.y.values[0]]
                p_2 = [e_2.x.values[0], e_2.y.values[0]]

                #print(np.subtract(p_1, vert))
                
                a = np.subtract(p_1, vert)
                b = np.subtract(p_2, vert)

                dot = np.dot(a,b)
                mag_a = np.linalg.norm(a)
                mag_b = np.linalg.norm(b)
                radians = math.acos((dot)/(mag_a*mag_b))
                degrees = math.degrees(radians)
                if(math.isnan(degrees)):
                    print(self.osmid)
                self.angles.append(degrees)

        print(self.angles)          
      
    def get_distances(self):
        for e_1, e_2 in zip(self.edges, self.edges[1:]):
            p_1 = planar.Vec2(self.x.values[0], self.y.values[0])
            p_2 = planar.Vec2(e_1.x.values[0], e_1.y.values[0])
            p_3 = planar.Vec2(e_2.x.values[0], e_2.y.values[0])

            anchor = p_2
            vector = p_3 - p_2
            segment = planar.line.LineSegment(anchor,vector)
            distance = segment.distance_to(p_1)
            if(math.isnan(distance)):
                print(self.osmid)
            self.distances.append(distance)

        for e_1, e_2 in zip(self.edges, self.edges[-1:]):
            p_1 = planar.Vec2(self.x.values[0], self.y.values[0])
            p_2 = planar.Vec2(e_1.x.values[0], e_1.y.values[0])
            p_3 = planar.Vec2(e_2.x.values[0], e_2.y.values[0])

            anchor = p_2
            vector = p_3 - p_2
            segment = planar.line.LineSegment(anchor,vector)
            distance = segment.distance_to(p_1)
            self.distances.append(distance)
  
        print(self.distances) 
       
            

def find_triangles(G):
    G_proj = ox.project_graph(G)
    nodes_proj = ox.graph_to_gdfs(G_proj, edges=False)
    all_cliques= nx.enumerate_all_cliques(G)
    triad_cliques = [x for x in all_cliques if len(x)==3 ]

    return triad_cliques, nodes_proj

#check if any matching triangles
def check_duplicates(G):
    G_proj = ox.project_graph(G)
    nodes_proj = ox.graph_to_gdfs(G_proj, edges=False)
    all_cliques= nx.enumerate_all_cliques(G)
    triad_cliques = [set(x) for x in all_cliques if len(x)==3 ]
    for i in range(len(triad_cliques)):
        for j in range(i+1,len(triad_cliques)):
            if triad_cliques[i] == triad_cliques[j]:
                print("Duplicate")




def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi)

#Find all edges for each node
def build_df(city_map):
    G = nx.read_gpickle(city_map)
    G = G.to_undirected()
    node_edges = {}
    G_proj = ox.project_graph(G)
    nodes_proj, gdf_edges = ox.graph_to_gdfs(G_proj, edges=True, nodes = True)

    df = nodes_proj[['osmid', 'x', 'y']]

    all_nodes = list(nodes_proj['osmid'])
    node_edges = {}
    for i in range(len(all_nodes)):
        for edge in G.edges(all_nodes[i]):
            if (edge == all_nodes[i]):
                break
        else: 
            node_edges[i] = G.edges(all_nodes[i])
            continue


    df_2 = pd.DataFrame(node_edges.values())

    df_2[['osmid','edge_1']] = pd.DataFrame(df_2[0].tolist(), index= df_2.index)
    df_2 = df_2.drop(['edge_1'], axis=1)

    df = pd.merge(df, df_2, on="osmid")

    df = df.stack().unstack(fill_value=(-1, -1))

    edge_df = df[['osmid', 'x', 'y']].copy()
    for i in range(len(df.columns) - 3):
        edge_df[['test','e' + str(i)]] = pd.DataFrame(df[i].tolist(), index= df.index)
        edge_df = edge_df.drop(['test'], axis=1)


    print(edge_df.head())
    edge_df.to_pickle("nodes_edges.pkl")

#test = pd.read_pickle("nodes_edges.pkl")
#print(test.head())

#city_map = "graphs/maps/BerlinGermany.gpickle"
#build_df(city_map)



def create_edge_node(df, osmid, edges):
    x = df.loc[df['osmid'] == osmid]['x']
    y = df.loc[df['osmid'] == osmid]['y']
    node = Node(osmid,x,y,edges)
    return node

'''
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df_2 = pd.read_pickle("nodes_edges.pkl")
df = df_2[df_2['e1'] != -1].copy()

for column in df.columns[3:]:
    df[column] = df.apply(lambda row: create_edge_node(df_2, row[column], []), axis =1)

df['vertex'] = df.apply(lambda row: create_edge_node(df_2, row['osmid'], row[3:]), axis =1)


df.to_pickle("final_df.pkl")
'''


'''
df = pd.read_pickle("final_df.pkl")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



df['vertex'].apply(lambda row: row.delete_edges())
df['vertex'].apply(lambda row: row.find_thetas())
df['vertex'].apply(lambda row: row.get_angles())
df['vertex'].apply(lambda row: row.get_distances())
df.to_pickle("angles_distances.pkl")
'''


'''
#df = pd.read_pickle("angles_distances.pkl")
#data = df['vertex']

#print(data.head())

result_df = pd.DataFrame(columns = ["angle", "distance"])
angles = pd.DataFrame(columns = ["angle"])
distances = pd.DataFrame(columns = ["distance"])

def results(data,result_df, angles, distances):
    for row in data.iterrows():
        for angle in row[1]['vertex'].angles:
            angles= angles.append({'angle': angle},ignore_index=True)
        for distance in row[1]['vertex'].distances:
            distances = distances.append({'distance': distance}, ignore_index=True)

    result_df = angles.join(distances, how='outer')
    return result_df

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

results = results(df,result_df,angles,distances)
print(results.head())
results.to_pickle("final_results.pkl")
'''

df = pd.read_pickle("final_results.pkl")

#df= df[df['distance'] < 1000]

df.plot(kind='scatter',x='angle',y='distance',color='black')
plt.show()















