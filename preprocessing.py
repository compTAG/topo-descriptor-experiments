import numpy as np
from utils.get_data import preprocess_data, dir_list
from utils.build_graphs import randpts_graphs,mpeg7_graphs,mnist_graphs
import random


preprocess_data(dir_list)

# make sure we have the same seeds as main experiments
random.seed(423652346)
np.random.seed(423652346)

randpts_graphs()

#Change eps according to how close we want the approx
eps = .001
#Change to corresponding eps 
graphs_dir = "graphs_001_approx" 
mpeg7_graphs(eps, graphs_dir)



eps = .005
graphs_dir = "graphs_005_approx" 
mpeg7_graphs(eps, graphs_dir)


#Change eps according to how close we want the approx
eps = .001

#Change to corresponding eps 
graphs_dir = "graphs_001_approx" 

mnist_graphs(eps, graphs_dir)

#Change eps according to how close we want the approx
eps = .005
#Change to corresponding eps 
graphs_dir = "graphs_005_approx" 

mnist_graphs(eps, graphs_dir)
