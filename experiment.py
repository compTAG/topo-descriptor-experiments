import math
import os
import copy

import networkx as nx

import orth_angle


def stratify(G):
	orth_angle.fillangmatrix(G.graph["stratum"], len(G.nodes()), list(G.nodes(data=True)))
	arcs = orth_angle.find_arc_lengths(G.graph["stratum"])
	return G, arcs

# @param networkx Graph G: the graph containing the points
# @param str outFile: string of file name to write the stratum sizes to
# stores the results in designated outfiles
def stratum_experiment(G,arcs,outFile,out_graphs_dir):
	print("Stratum experiment")
	with open(os.path.join(out_graphs_dir,"distribution_exp", outFile), "w+") as f:
		f.write("startv1,startv2,endv1,endv2,length\n")
		f.write("\n".join([(str(arc["start"]["vertex1"])+","+str(arc["start"]["vertex2"]) +
			","+str(arc["end"]["vertex1"])+","+str(arc["end"]["vertex2"])+","+str(arc["length"]))
			for arc in arcs]))

# runs an experiment randomly sampling from the unit sphere and marking off arcs we hit
# @param Graph G: graph to perform experiment on
# @param list arcs: list of arcs for the stratified regions of the sphere for G
# @param list sample_sizes: different numbers of random samples to take
# @param str outFile: string of file name (please include extension) to pickle point clouds to
# stores the results in designated outfiles
def sample_experiment(G,arcs,sample_sizes,outFile,out_graphs_dir):
	print("Sample experiment")
	# open up a file to write the outputs to for this pc size
	with open(os.path.join(out_graphs_dir,"sample_exp",outFile), "w+") as f:
		# we store three values: samples, hits (number of stratum hit), num_stratum (total number of stratum on this graph)
		f.write("n,samples,hits,num_stratum")
		f.write("\n")
		# iterate through each number of samples, each of these loops is an experiment
		for num_samples in sample_sizes:
			for j in range(0,num_samples):
				# take a random sample in radians
				sample = random.uniform(0.0, 2*math.pi)
				for arc in arcs:
					# test to see which stratum this sample falls into and update that stratum to designate a hit
					if ((arc["start"]["location"] < arc["end"]["location"])
						and (sample >= arc["start"]["location"])
						and (sample < arc["end"]["location"])):
						arc["hit"] = 1
					elif ((arc["start"]["location"] > arc["end"]["location"])
						and ((sample >= arc["start"]["location"])
							or (sample < arc["end"]["location"]))):
						arc["hit"] = 1

			# keep track of total number of hits
			hit_count = sum([arc["hit"] for arc in arcs])
			# reset hits to 0 for next iteration
			for arc in arcs:
				arc["hit"] = 0

			f.write(str(len(G))+","+str(num_samples)+","+str(hit_count)+","+str(len(arcs)))
			f.write("\n")

# runs an experiment randomly sampling uniformly alon the unit sphere and marking off arcs we hit
# @param Graph G: graph to perform experiment on
# @param list arcs: list of arcs for the stratified regions of the sphere for G
# @param list sample_sizes: different numbers of random samples to take
# @param str outFile: string of file name (please include extension) to pickle point clouds to
# stores the results in designated outfiles
def uniform_sample_experiment(G,arcs,sample_sizes,outFile,out_graphs_dir):
	if len(arcs) < 5000:
		print("Num arcs: "+str(len(arcs)))
		# open up a file to write the outputs to for this pc size
		print(os.path.join(out_graphs_dir,"uniform_sample_exp",outFile))
		with open(os.path.join(out_graphs_dir,"uniform_sample_exp",outFile), "w+") as f:
			# we store three values: samples, hits (number of stratum hit), num_stratum (total number of stratum on this graph)
			f.write("n,samples,hits,num_stratum")
			f.write("\n")
			# iterate through each number of samples, each of these loops is an experiment
			for num_samples in sample_sizes:
				increment = (2*math.pi) / num_samples
				# print("INCREMENT "+str(increment))
				sample = 0
				for j in range(0,num_samples):
					for arc in arcs:
						# test to see which stratum this sample falls into and update that stratum to designate a hit
						if ((arc["start"]["location"] < arc["end"]["location"])
							and (sample >= arc["start"]["location"])
							and (sample < arc["end"]["location"])):
							arc["hit"] = 1
						elif ((arc["start"]["location"] > arc["end"]["location"])
							and ((sample >= arc["start"]["location"])
								or (sample < arc["end"]["location"]))):
							arc["hit"] = 1
					sample += increment

				# keep track of total number of hits
				hit_count = sum([arc["hit"] for arc in arcs])
				# reset hits to 0 for next iteration
				for arc in arcs:
					arc["hit"] = 0

				# print(str(len(G.nodes()))+","+str(num_samples)+","+str(hit_count)+","+str(len(arcs)))
				f.write(str(len(G.nodes()))+","+str(num_samples)+","+str(hit_count)+","+str(len(arcs)))
				f.write("\n")
	else:
		print("Not running exp, too many arcs: "+str(len(arcs)))

# @param networkx Graph G: graph to run experiments on
# @param list arcs: stratum along the sphere for G
# @param str outFile: string of file name to write results to (see headers in function)
# stores the results in designated outfiles
#### NOTE, THIS EXPERIMENT IS ACTUALLY THE SMALLEST STRATUM SIZE, NOT ANGLE
def smallest_angle_experiment(G,arcs,outFile,out_graphs_dir):
	print("Smallest angle experiment")
	with open(os.path.join(out_graphs_dir,"smallest_angle_exp",outFile), "w+") as f:
		# Add headers to output file
		f.write("n,min_angle,num_stratum,num_needed_stratum,ratio")
		f.write("\r\n")
		min_arc = min([a["length"] for a in arcs])

		num_stratum = math.ceil((2*math.pi)/min_arc)
		num_needed_stratum = len(arcs)
		num_unneeded_stratum = num_stratum - num_needed_stratum
		ratio = (num_needed_stratum / num_stratum)

		f.write(str(len(G.nodes()))+","+str(min_arc)+","+str(num_stratum)+","+str(num_needed_stratum)+","+str(ratio))
		f.write("\r\n")


def overlap_exp(G,arcs,outFile):
	for i in range(0, len(arcs)):
		for j in range(0, len(arcs)):
			if i != j:
				overlap = False
				if ((arcs[i]["start"]["location"] < arcs[i]["end"]["location"])
					and (arcs[j]["start"]["location"] > arcs[i]["start"]["location"])
					and (arcs[j]["start"]["location"] < arcs[i]["end"]["location"])):
						overlap=True
				elif ((arcs[i]["start"]["location"] < arcs[i]["end"]["location"])
					and (arcs[j]["end"]["location"] > arcs[i]["start"]["location"])
					and (arcs[j]["end"]["location"] < arcs[i]["end"]["location"])):
						overlap=True
				elif ((arcs[i]["start"]["location"] > arcs[i]["end"]["location"])
					and ((arcs[j]["start"]["location"] > arcs[i]["start"]["location"])
					or (arcs[j]["start"]["location"] < arcs[i]["end"]["location"]))):
						overlap=True
				elif ((arcs[i]["start"]["location"] > arcs[i]["end"]["location"])
					and ((arcs[j]["end"]["location"] > arcs[i]["start"]["location"])
					or (arcs[j]["end"]["location"] < arcs[i]["end"]["location"]))):
						overlap=True
				if overlap:
					print "Overlap on graph "+str(outFile)
					print("i start: "+str(arcs[i]["start"]["location"]) + " end " +str(arcs[i]["end"]["location"]))
					print("j start: "+str(arcs[j]["start"]["location"]) + " end " +str(arcs[j]["end"]["location"]))
					sys.exit(1)





class Runner(object):


    STRAT = 1
    RANDOM = 2
    SMALL_ANGLE = 3
    UNIF = 4
    ALL = 5

    def run(self, G,output_file,exp_type,out_graphs_dir):
        """
        experiment setup for graphs
        @param networkx Graph G: input_graph
        @param string output_file: where to write results
        @param int exp_type: type of experiment to run (specified in main)
        stores the results in outfiles defined below
        """
        sample_sizes=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        G, arcs = stratify(G)
        if exp_type == self.STRAT:
            stratum_experiment(G,arcs,output_file,out_graphs_dir)
        elif exp_type == self.RANDOM:
            sample_experiment(G,arcs,sample_sizes,output_file,out_graphs_dir)
        elif exp_type == self.SMALL_ANGLE:
            smallest_angle_experiment(G,arcs,output_file,out_graphs_dir)
        elif exp_type == self.UNIF:
            uniform_sample_experiment(G,arcs,sample_sizes,output_file,out_graphs_dir)
        elif exp_type == self.ALL:
            self.run(G, output_file, self.STRAT, out_graphs_dir)
            self.run(G, output_file, self.RANDOM, out_graphs_dir)
            self.run(G, output_file, self.SMALL_ANGLE, out_graphs_dir)
            self.run(G, output_file, self.UNIF, out_graphs_dir)

        #optional
        # draw_graph(G, G.graph["stratum"], output_file)


