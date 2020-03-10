import sys
import numpy as np
import math
from itertools import permutations
import argparse
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import time
import itertools
from scipy.spatial import Delaunay
import csv

# import reconstruction;

def generate_points(N, square_size, seed):
    np.random.seed(seed)
    x = np.random.uniform(0, square_size, N)
    y = np.random.uniform(0, square_size, N)
    verts = [(i,j) for (i,j) in zip(x,y)]
    return(verts)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def subtract(p1, p2):
    return((p1[0]-p2[0], p1[1]-p2[1]))

def min_angle(points):
    min_angle = 90
    for triple in permutations(points, 3):
        p1, p2, p3 = triple
        vec1 = subtract(p2, p1)
        vec2 = subtract(p3, p1)
        angle = angle_between(vec1, vec2)
        min_angle = min(min_angle, angle)
    return(math.log10(min_angle))

def min_angle2(points): #use this one, much faster
    coords = [c for p in points for c in p]
    return reconstruction.min_angle(coords)

def make_boxplot(data, labels, num_trials):
    fig, ax = plt.subplots()
    ax.set_title('Minimum angle between randomly placed vertices')
    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Log10 of minimum angle for {} trials".format(num_trials))
    plt.axhline(y=-6)
    plt.savefig("min_angle.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_start") # Used 10
    parser.add_argument("num_points") # Used 8
    parser.add_argument("n_by") # Used 10
    trials = 1000
    args = parser.parse_args()
    n_start = int(args.n_start)
    n_by = int(args.n_by)
    num_points = int(args.num_points)
    n_list = [n_start + n_by*i for i in range(num_points)]
    print("Testing the following n values: {}".format(n_list))
    square_size = 100
    data = []
    for n in n_list:
        print("Testing n=", n)
        min_angles = []
        for trial in range(trials):
            random_seed = n + trial
            points = generate_points(n, square_size, random_seed)
            # min_angles.append(min_angle(points))
            min_angles.append(min_angle2(points))
        data.append(min_angles)
    with open('./min_angle_results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for i in range(len(n_list)):
            for j in range(trials):
                writer.writerow([(data[i])[j], n_list[i]])
    labels = ["n=" + str(n) for n in n_list]
    print(labels)
    make_boxplot(data, labels, trials)
