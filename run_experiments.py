from main import *

#generate_graphs creates random point clouds and stores them in graphs/random
#when running experiments, get_exp_graphs looks for gpickle files in graphs_random and cannot run properly


# modify for that approximation type for emnist and mpeg7
# choices include graphs_001_approx and graphs_005_approx
graphs_dir = "graphs_001_approx"
# same as above but specifies where to write results
out_graphs_dir = "output_001_approx"

#### exp type is:
#       1 for stratification experiment (distribution_exp)
#       2 for random sample experiment (sample_exp)
#       3 for smallest angle experiment (smallest_angle_exp)
#       4 for a uniform random sample experiment (uniform_sample_exp)
#       5 for all four exps
exp_type = 1

#### data is:
#       1 for random
#       2 for MPEG7 (classes from PHT paper - Turner et al.)
#       3 for EMNIST
#       4 for all three
#       5 for test
data_type = 3


exp_list = get_exp_graphs(data_type)

start = time.time()
counter = 1
for e in exp_list:
  print("Graph "+str(counter)+" of "+str(len(exp_list)))
  exp(e["G"], e["output_file"], exp_type)
  counter+=1

print("Execution time: "+str(time.time() - start)+"(s)")