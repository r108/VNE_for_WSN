'''Virtual Network Embedding Algorithm'''

import networkx as nx
import weighted_graph_test as wg
import sp_dijkstra as sp

WSN = nx.Graph()  #represents the substrate network resources
VNR = set()     #virtual network requests
M = None       #comitted mappings

#a conflict graph representing interference is required
CG = nx.Graph()

def get_vnrs():
    vnr1 = (1000, (51, {'load': 17}, 1, {'load': 17}), {'load': 17, 'plr': 40})
    vnr2 = (1000, (14, {'load': 12}, 1, {'load': 12}), {'load': 12, 'plr': 40})
    vnr3 = (1000, (56, {'load': 10}, 1, {'load': 10}), {'load': 10, 'plr': 40})
    vnr4 = (1000, (52, {'load': 8}, 1, {'load': 8}), {'load': 8, 'plr': 40})
    vnr5 = (1000, (37, {'load': 7}, 1, {'load': 7}), {'load': 7, 'plr': 40})
    vnr6 = (1000, (6, {'load': 6}, 1, {'load': 6}), {'load': 6, 'plr': 40})
    vnr7 = (1000, (19, {'load': 11}, 1, {'load': 11}), {'load': 11, 'plr': 40})
    vnr8 = (1000, (32, {'load': 5}, 1, {'load': 5}), {'load': 5, 'plr': 40})
    vnr9 = (1000, (38, {'load': 3}, 1, {'load': 3}), {'load': 3, 'plr': 40})



    #return [vnr1,vnr2,vnr3,vnr4,vnr5,vnr6,vnr7,vnr8, vnr9]
    return [vnr1,vnr2,vnr3,vnr4,vnr5]
#    return [vnr5,vnr4,vnr3,vnr2,vnr1]
#    return [vnr9, vnr8, vnr6, vnr5, vnr4]
#    return [vnr4, vnr5, vnr6, vnr8, vnr9]
