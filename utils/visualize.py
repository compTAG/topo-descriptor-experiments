import networkx as nx
import matplotlib.pyplot as plt
def draw_graph(G, stratum, outfile):
	loc = {}
	lab={}
	for v in G.nodes(data=True):
		loc[v[1]['v'].get_id()] = (v[1]['v'].get_x(), v[1]['v'].get_y())
		lab[v[1]['v'].get_id()] = str(v[1]['v'].get_x())+" "+str(v[1]['v'].get_y())
	nx.draw_networkx(G, pos=loc, with_labels=False, node_size=10)
	## uncomment to verify that perturbations have been added to points
	# nx.draw_networkx_labels(G, pos=loc, with_labels=False, labels=lab, node_size=25)


	# plt.axis('off')
	plt.savefig(outfile+"_perim_graph.pdf")
	plt.clf()