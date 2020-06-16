from generate_graphs import *

#Take downloaded data and make graphs for experiments 
#One issue is that generate() does not take a parameter to change eps from .005 to .001. 
#When adding a parameter, this function runs much slower for mpeg7 and emnist. 
#What are the random seeds needed for? They are the same in main.py and generate_graphs.py

#original eps is .005, store graphs in graphs_005_approx 
#main()



#Change eps so that we generate graphs for eps= .001 approx and store in graphs_001_approx
#eps = .001
#graphs_dir = "graphs_001_approx"

# We want to be able to generate graphs for eps = .005 and .001, but have not
# figured out a good way to do this yet.
main()