import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import requests
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import math
from shapely.geometry import Polygon


def find_triangles(G):
    G_proj = ox.project_graph(G)
    nodes_proj = ox.graph_to_gdfs(G_proj, edges=False)
    all_cliques= nx.enumerate_all_cliques(G)
    triad_cliques = [x for x in all_cliques if len(x)==3 ]

    return triad_cliques, nodes_proj

def get_angles(triad_cliques, nodes_proj):
    triangles = {}
    for i in range(len(triad_cliques)):
        angles = []
        triangles[i] = {}
        triangles[i]['nodes'] = triad_cliques[i]
        node_1 = nodes_proj.loc[nodes_proj['osmid'] == triad_cliques[i][0]]
        node_2 = nodes_proj.loc[nodes_proj['osmid'] == triad_cliques[i][1]]
        node_3 = nodes_proj.loc[nodes_proj['osmid'] == triad_cliques[i][2]]

        p_1 = [node_1.iloc[0]['x'], node_1.iloc[0]['y']]
        p_2 = [node_2.iloc[0]['x'], node_2.iloc[0]['y']]
        p_3 = [node_3.iloc[0]['x'], node_3.iloc[0]['y']]

        triangles[i]['positions'] = [p_1,p_2,p_3]

        vectors = [[np.subtract(p_2, p_1),np.subtract(p_3, p_1)],
                [np.subtract(p_1, p_2),np.subtract(p_3, p_2)],
                [np.subtract(p_1, p_3),np.subtract(p_2, p_3)]]
        for vector in vectors: 
            a= vector[0]
            b= vector[1]
            dot = np.dot(a,b)
            mag_a = np.linalg.norm(a)
            mag_b = np.linalg.norm(b)
            radians = math.acos((dot)/(mag_a*mag_b))
            degrees = math.degrees(radians)
            angles.append(degrees)
        triangles[i]['angles'] = angles
    print(triangles)
    return triangles


def get_distance(data,output):
    max_angle = np.max(data['angles'])
    print(max_angle)
    output[i] = {}
    output[i]['max_angle'] = max_angle
    angle_position = data['angles'].index(max_angle)
    vertex = data['nodes'][angle_position]
    p_1 = [data['positions'][angle_position][0],data['positions'][angle_position][1]]

    
    data['positions'].pop(angle_position)
   

    p_2 = [data['positions'][0][0],data['positions'][0][1]]
    p_3 = [data['positions'][1][0],data['positions'][1][1]]
    

    distance = abs((p_3[1]-p_2[1])*p_1[0] - (p_3[0]-p_2[0])*p_1[1] + (p_3[0]*p_2[1]) 
        - (p_3[1]*p_2[0]))/math.sqrt(((p_3[1]-p_2[1])**2) + ((p_3[0]-p_2[0])**2) )
    

    output[i]['distance'] = distance
    return output

'''
G = nx.read_gpickle("graphs/maps/BerlinGermany.gpickle")
G_1 = G.to_undirected()

triangles, nodes_proj = find_triangles(G_1)

data = get_angles(triangles, nodes_proj)


output = {}
for i in range(len(data)):
    get_distance(data[i],output)
'''
'''
df = pd.DataFrame.from_dict(output, orient='index')
print(df.head())

plt.scatter(df['max_angle'], df['distance'])
plt.xlabel("Max Angle from each Triangle")
plt.ylabel("Distance to Edge")
plt.show()
'''



'''
Plot nodes from all triangles vs actual graph of Berlin
G_proj = ox.project_graph(G_1)
nodes_proj = ox.graph_to_gdfs(G_proj, edges=False)
all_cliques= nx.enumerate_all_cliques(G_1)
triad_cliques = [x for x in all_cliques if len(x)==3 ]
nodes = []
for sublist in triad_cliques:
    for item in sublist:
        nodes.append(item)
print(nodes)
test = G_1.subgraph(nodes)
fig, ax = ox.plot_graph(test)
fig, ax = ox.plot_graph(G)
'''


