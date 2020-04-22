#MPEG7

from vertex import *
from orth_angle import *
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import os
import cv2
import zipfile
import wget
import scipy.io as sio
import networkx as nx
from sklearn import datasets
import sys
import copy
from visualize import draw_graph
from itertools import combinations
from planar import Polygon


# takes a sample from each emnist class and determines the threshold we should use
def determine_emnist_threshold():
    data = sio.loadmat('data/emnist/emnist-byclass.mat')
    images = data['dataset'][0][0][0][0][0][0]
    labels = data['dataset'][0][0][0][0][0][1]
    imgs = []
    for c in range(0,62):
        indexes = [i for i, label in enumerate(x[0] for x in labels) if label == c][:1]
        # images are returned as an array with 784 elements, so we reshape to be a 2d array
        imgs.append([(images[i]).reshape(28, 28) for i in indexes][0])

    thresholds = []
    for img in imgs:
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresholds.append(ret)
    print thresholds
    print("Average: "+str(sum(thresholds) / len(thresholds)) + " Std: "+str(np.std(thresholds)))

# generates and returns n point clouds of size k as a list
# @param int n: number of point clouds to generate of each size
# @param list k: list of integers which denote the size of the different sets of point clouds
# @param float l: width of the bounding box to generate points in
# @param float m: height of the bounding box to generate points in
# of dictionaries
def generate_point_clouds(n,k,l,m):
    pcs = []
    for i in range(0,n):
        pc = nx.Graph()
        pc.graph["stratum"] = np.zeros((k,k))
        for j in range(0,k):
            pc.add_node(j, v=Vertex(j,
                            random.uniform(0.0, l),
                            random.uniform(0.0, m)))
        if not test_gen_pos(pc):
            print("The following point cloud is not in general position:")
            for pt in pc.nodes(data=True):
                print(pt[1]['v'].get_x())
                print(pt[1]['v'].get_y())
                print("")
            print("\n")
            continue
        pcs.append({"id":i,
            "k":k,
            "pc":pc
            })
    return pcs


# tests to see if three points are colinar
# @param Vertex x: first vertex
# @param Vertex y: second vertex
# @param Vertex z: third vertex
# @return: True if three pts are colinear, False otherwise
def colin(x,y,z):
    # Uses determinant method from: http://mathworld.wolfram.com/Collinear.html
    if (x.get_x()*(y.get_y() - z.get_y()) + y.get_x()*(z.get_y() - x.get_y()) +
        z.get_x()*(x.get_y() - y.get_y())) == 0:
        return True
    return False

# tests to see if the contour defines a simple polygon
# @param G: networkx graph
# returns True if the polygon is simple, False otherwise
def simple_polygon(G):
    # generate a list of the coordinates
    coords = []
    if len(nx.cycle_basis(G,root=0)) != 1:
        print("The graph is not a single contour...")
        return False
        # print(nx.cycle_basis(G,root=0))
        # sys.exit(1)
    # our contours are a single closed curve so we can just
    # use the cycle basis to generate the polygon
    for i in nx.cycle_basis(G,root=0)[0]:
        coords.append((G.node[i]['v'].get_x(), G.node[i]['v'].get_y()))

    poly = Polygon(coords)
    return poly.is_simple

# tests to make sure gen position assumptions are met for a point cloud
# @param networkx Graph G: point cloud object
# @return: True if general position assumptions are met, False otherwise
def test_gen_pos(G):
    # Test to make sure no two points share an x- or y-coord
    for c in combinations(list(G.nodes(data=True)), 2):
        if (c[0][1]['v'].get_x() == c[1][1]['v'].get_x() or
            c[0][1]['v'].get_y() == c[1][1]['v'].get_y()):
            print("Shared x- and y-coords")
            return False
    # Test to make sure no three points are colinear
    for c in combinations(list(G.nodes(data=True)), 3):
        # print(str(c[0][1]['v'].get_id()) + " " + str(c[1][1]['v'].get_id()) + " "+ str(c[2][1]['v'].get_id()))
        if colin(c[0][1]['v'],c[1][1]['v'],c[2][1]['v']):
            print("3 points colin")
            return False
    return True

# takes a graph (not in gen position) and perturbs the vertices
# until they are in general position
# @param networkx Graph G
def perturb(G):
    for v in G.nodes(data=True):
        # mean, std dev, number of samples
        perturbations = np.random.uniform(low=-0.01, high=0.01, size=2)
        v[1]['v'].set_x(v[1]['v'].get_x() + perturbations[0])
        v[1]['v'].set_y(v[1]['v'].get_y() + perturbations[1])
    gen_pos = test_gen_pos(G)
    if not gen_pos:
        print("Perturbation did not work on graph")
        return -1
    return G


# takes a filepath to Turner et al's data and returns a networkx graph with the
# perimeter data
# @param string file_path: the relative file path to the desired file
def get_pht_mpegSeven_data(file_path):
    G = nx.Graph()

    # get all the vertices
    id_num = 0
    with open(file_path) as f:
        for line in f.readlines():
            ######
            # Need to come through and only add "relevant points"
            ######
            split = line.split("\t", 2)
            G.add_node(id_num, v = Vertex(id_num, float(split[0]), float(split[1])))
            id_num+=1

    # add the edges
    for id_num in range(0, len(G.nodes())):
        if id_num == len(G.nodes()) - 1:
            G.add_edge(id_num, 0)
        else:
            G.add_edge(id_num, id_num+1)
    G.graph["stratum"]=np.zeros((len(G.nodes()),len(G.nodes())))
    # Make sure we meet gen pos assumption
    G = perturb(G)
    return G


# Takes an input filename from the mpeg7 dataset and loads it in
# @param String filemane: the name of the mpeg7 file (.gif)
# returns a img from the input gif file
def get_mpegSeven_img(filename):
    return np.array(Image.open("data/mpeg7/" + filename))

#EMNIST
# load a data set and extract a list of images from emnist
# @param int c: the class, ranges from 0->61
# @param int n: the number of images to sample from the class (takes the first n)
# returns a list of the first n images from class c
def get_mnist_img(c, n):
    data = sio.loadmat('data/emnist/emnist-byclass.mat')
    images = data['dataset'][0][0][0][0][0][0]
    labels = data['dataset'][0][0][0][0][0][1]
    indexes = [i for i, label in enumerate(x[0] for x in labels) if label == c][:n]
    # images are returned as an array with 784 elements, so we reshape to be a 2d array
    return [(images[i]).reshape(28, 28) for i in indexes]


# takes a filepath to Turner et al's data and returns a networkx graph with the
# perimeter data
# @param thresh: the thresholding done on the original img
# @param contours: the contours generated on the threshold
# @param img: a deep copy of the original img
# @param string fileman: name of file to save image in
# save three pdfs showing the original img (orig), the thresholded img (before)
# and the original image with contours (after)
def save_contour_img(thresh, contours, img, filename):
    plt.imshow(img)
    plt.savefig("output/imgs/orig_"+str(filename)+".pdf")
    plt.clf()

    plt.imshow(thresh)
    plt.savefig("output/imgs/thresh_"+str(filename)+".pdf")
    plt.clf()

    for contour in contours:
        cv2.drawContours(img, contour, -1, (0,255,0), 1)

    plt.imshow(img)
    plt.savefig("output/imgs/contours_"+str(filename)+".pdf")
    plt.clf()

# takes an img from the MNIST data set and returns a networkx graph with the
# perimeter data
# @param img: the image
# returns a networkx graph with vertices on the perimeter and edges along the
# contour. Note that the vertices are a SIMPLE approx of the actual contour
# data.
def get_img_data(img):
    G = nx.Graph()

    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #### TODO approximate contours with polygons: https://datacarpentry.org/image-processing/09-contours/

    node_id = 0
    sep = 0
    for contour in contours:
        for pt in contour:
            G.add_node(node_id, v=Vertex(node_id,
                                        float(pt[0][0]),
                                        float(pt[0][1]))) #negate because their y-axis goes from k->0
            node_id+=1
        # add in the appropriate edges for this contour
        for i in range(sep, node_id-1):
            G.add_edge(i, i+1)
        G.add_edge(node_id-1, sep)
        sep = node_id

    # visualization functions for debugging
    # save_contour_img(thresh, contours, copy.deepcopy(img), "test")
    # draw_graph(G)
    G.graph["stratum"] = np.zeros((len(G.nodes()),len(G.nodes())))
    # 2(71 choose 2) = 4970, anything larger will be too big
    if len(G.nodes()) > 71:
        print("G has too many vertices: "+str(len(G.nodes())))
        return -1
    # Make sure we meet gen pos assumption
    G = perturb(G)
    return G

# Wrapper to get the contour length for sorting
def contour_length(c):
    return cv2.arcLength(c,closed=True)


# takes a x,y coordinates
# @ param x: integer
# @ param y: integer
# @ param G: networkx undirected graph
# returns the vertex id of the coordinates if it exists, -1 otherwise
def get_node_index(x,y,G):
    for v in G.nodes(data=True):
        if v[1]['v'].get_x() == x and v[1]['v'].get_y()==y:
            return v[1]['v'].get_id()
    return -1

# takes an img from the MNIST data set and returns a networkx graph with the
# perimeter data
# @param img: the image
# @param eps: the epsilon value used in contour approximation
# @param threshed: the threshold used on converting to binary (MPEG7 should have 0, MNIST should have another fixed param)
# returns a networkx graph with vertices on the perimeter and edges along the
# contour. Note that the vertices are a SIMPLE approx of the actual contour
# data. The second return value is pertaining to general position. -2 means it was not
# a simple polygon, -1 means it did not mean gen pos, 0 means it has duplicate
# vertices in the contour  and 1 means success
# Note that original eps is .005
def get_img_data_approx(img, eps, threshold):
    G = nx.Graph()
    # get the thresholded image (thresh)
    ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort the contours by length (in descending order)
    cntrs = sorted(contours, key=contour_length, reverse=True)

    # store the longest contour
    c_temp = cntrs[0]
    epsilon = eps * cv2.arcLength(curve = c_temp, closed = True)
    #get an approximate contour from c_temp
    c = cv2.approxPolyDP(curve = c_temp,
            epsilon = epsilon,
            closed = True)
    # print c

    # add the vertices to a networkx graph
    node_id = 0
    for pt in c:
        index = get_node_index(pt[0][0], pt[0][1], G)
        # check to make sure we haven't already added this vertex
        if index == -1:
            G.add_node(node_id, v=Vertex(node_id,
                                        float(pt[0][0]),
                                        float(pt[0][1])))
            node_id+=1
        # vertices have to be unique, so if it isn't, we exit
        else:
            print("There is a duplicate vertex")
            print(pt)
            # return G, 0
    # add in the appropriate edges for the contour
    for i in range(0, len(c)-1):
        v1 = c[i]
        v2 = c[i+1]
        G.add_edge(get_node_index(v1[0][0], v1[0][1], G),
            get_node_index(v2[0][0], v2[0][1], G))
    # add edge from last to first vertex in contour to make closed curve
    v1 = c[len(c)-1]
    v2 = c[0]
    G.add_edge(get_node_index(v1[0][0], v1[0][1], G),
        get_node_index(v2[0][0], v2[0][1], G))

    # visualization functions for debugging
    # save_contour_img(thresh, contours, copy.deepcopy(img), "test")
    # draw_graph(G)
    G.graph["stratum"] = np.zeros((len(G.nodes()),len(G.nodes())))

    # Make sure we meet gen pos assumption
    G = perturb(G)
    if G == -1:
        print("Does not meet gen pos")
        return G, -1

    # Test to make sure the polygon is simple
    if not simple_polygon(G):
        print("Not simple polygon!")
        return G, -2
    return G, 1



#### for testing purposes only
def main():
    # determine_emnist_threshold()

    # G = get_img_data(get_mpegSeven_img("cattle-3.gif"))
    # draw_graph(G, G.graph['stratum'], "graphs/test_data/cattle-3")
    # print len(G.nodes())

    # G = get_img_data_approx(get_mpegSeven_img("spring-4.gif"),.005, 0)
    # draw_graph(G, G.graph['stratum'], "graphs_005_approx/test_data/spring-4-approx")
    # print(G.nodes())

    #test against old deer 3
    # deer = nx.read_gpickle('graphs_005_approx/mpeg7/MPEG7_spring-4.gpickle')
    # print(deer.nodes())

    c = 1
    n = 40
    image = get_mnist_img(c,n)
    G,ret = get_img_data_approx(image[32],.005, 102.951612903)
    draw_graph(G, G.graph['stratum'], "graphs_005_approx/test_data/MNIST_C1_S21_test")
    print len(G.nodes())
    # #test against old C3 S0
    # c8_s0 = nx.read_gpickle('graphs_005_approx/mnist/MNIST_C2_S20.gpickle')
    # print(len(c8_s0.nodes()))


    # G = get_img_data_approx(get_mnist_img(10))
    # draw_graph(G, G.graph['stratum'], "graphs/test_data/mnist10-approx")
    # print len(G.nodes())





if __name__=='__main__':main()
