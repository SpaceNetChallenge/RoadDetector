import pickle
import sys

my_map = pickle.load( open( sys.argv[1], "rb" ) )
nodes = my_map[0]
edges = my_map[1]

with open(sys.argv[2], 'w') as f:
	nodemap = {}
	counter = 0
	for node_id, node in nodes.items():
		nodemap[node_id] = counter
		counter += 1
		f.write("{} {}\n".format(node[1], node[0]))
	f.write("\n")
	for edge in edges.values():
		f.write("{} {}\n".format(nodemap[edge[0]], nodemap[edge[1]]))
		f.write("{} {}\n".format(nodemap[edge[1]], nodemap[edge[0]]))
