'''Virtual Network Embedding Algorithm'''

import networkx as nx
import weighted_graph_test as wg
import sp_dijkstra as sp

WSN = nx.Graph()  #represents the substrate network resources
VNR = set()     #virtual network requests
M = None       #comitted mappings

#a conflict graph representing interference is required
CG = nx.Graph()

#def embed_vnrs():