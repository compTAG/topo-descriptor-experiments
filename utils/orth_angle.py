import math
import random
import numpy as np
import utils.vertex
from utils.load_datasets import *
# 4/12/19

# potentially visualize with turtle graphics if I get bored/want to check if it looks correct visually
# finds angle orthogonal to line passing through two vertices, a and b
# @param Vertex a: first vertex to find orthogonal angle to
# @param Vertex b: second vertex to find orthogonal angle to
# @return float orthangle: orthgonal angle to line intersecting a and b in range [0, 2pi)
def findorthangle(a, b):
    # get slope
    tempx = a.get_x() - b.get_x()
    tempy = a.get_y() - b.get_y()

    # get slope of perpindicular line to line segment (a,b)
    orthx = -tempy
    orthy = tempx

    # atan2 gives back the angle in [-pi, pi] that the orthogonal slope makes with the x-axis
    orthangle = math.atan2(orthy, orthx)
    # if it is negative, then we need to subtract it from 2pi so it is in [0, 2pi)
    if orthangle < 0.0:
        orthangle = 2*math.pi + orthangle
    return orthangle  # will be in radians

# fill the angmatrix with appropriate orthangles
# @param [][] angmatrix: nxn matrix for storing orthogonal angles in
# @param int n: dimensions on nxn matrix angmatrix
# @param [] vertlist: set of nodes in networkx graph
# sets [i][j] of angmatrix to orthangle of line through i and j. [j][i] = [i][j] +- pi
def fillangmatrix(angmatrix, n, vertlist):
    for i in range(0, n):
        for j in range(0, n):
            # print("i: "+str(i) + " j: "+str(j)+" vi id: "+str(vertlist[i][1]['v'].get_id()) + " vj id: " + str(vertlist[j][1]['v'].get_id()))
            if i == j:
                angmatrix[i][j] = None  # fills diagonal of matrix with null, as no edge between vertex and itself
            else:
                angmatrix[i][j] = findorthangle(vertlist[i][1]['v'], vertlist[j][1]['v'])  # sets [i][j] to orthangle of line through i and j. [j][i] = [i][j] +- pi


# computes arc lengths of stratum on the unit sphere
# @param matrix m: an nxn matrix which stores the orthogonal angle to the line intersecting each pair of vertices
# @return [] arcs: a list of "arcs" defined by a start radian, end radian, and length
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