import networkx as nx
import copy

class WSN():

    __WSN_Substrate = nx.DiGraph()
    __link_weights = dict()
    __adj_list = dict()
    __two_hops_list = dict()
    __conflicting_links = dict()


    def __init__(self):
        self.init_wsn_substrate(self.get_adjacency_list())
        self.init_two_hop_neighborhood(self.get_adjacency_list())
        self.init_conflicting_links(self.get_adjacency_list())


    def get_wsn_substrate(self):
        return self.__WSN_Substrate

    def get_conflicting_links(self):
        return self.__conflicting_links

    def get_link_weights(self):
        return self.__link_weights

    def get_two_hops_list(self):
        return self.__two_hops_list

    def init_two_hop_neighborhood(self, adj_list):
        for n in adj_list:
            items = adj_list.get(n)
            if_list = []
            if_list.extend(items)
            for i in items:
                if_list.extend(adj_list.get(i))
            self.__two_hops_list[n] = list(set([x for x in if_list if x != n]))
        #print("Interferences ",self.__two_hops_list)

    def calculate_conflicting_links(self, path_nodes):
        self.__adj_list
        tx_nodes = copy.deepcopy(path_nodes)
        tx_nodes.pop()
        #    #print("tx_nodes",tx_nodes,"\npath_nodes",path_nodes)
        effected_edges = []
        #    #print("initialize effected_edges",effected_edges)
        for i, tx in enumerate(tx_nodes):
            visited_nodes = []
            #        #print(i,"visit tx", tx)
            visited_nodes.append(tx)
            #        #print(tx,"appended to visited nodes", visited_nodes)
            for n in self.__adj_list[tx]:
                #            #print(tx,"-> visit n", n)
                if n not in visited_nodes:
                    #                #print("add if n not in []",visited_nodes)
                    effected_edges.append((tx, n))
                    effected_edges.append((n, tx))
                for nn in self.__adj_list[n]:
                    #config.total_operations += 1
                    #                #print(n, "->-> visit nn", nn)
                    if nn not in visited_nodes:
                        #                    #print("add if nn not in []", visited_nodes)
                        effected_edges.append((n, nn))
                        effected_edges.append((nn, n))
                visited_nodes.append(n)
                #            #print(n, "appended to visited nodes in tx", visited_nodes)
                #            #print(i," in length",len(path_nodes))
            rx = path_nodes[i + 1]
            #        #print(i, "visit rx", rx)
            for n in self.__adj_list[rx]:
                #            #print(rx, "-> visit n", n)
                if n not in visited_nodes:
                    for nn in self.__adj_list[n]:
                        #config.total_operations += 1
                        #                    #print(n, "->-> visit nn", nn)
                        if nn not in visited_nodes:
                            #                        #print("add if nn not in []", visited_nodes)
                            effected_edges.append((n, nn))
                    visited_nodes.append(n)
                    #                #print(n, "appended to visited nodes in rx", visited_nodes)
            effected_edges_set = list(set(effected_edges))
            return effected_edges, effected_edges_set


    def init_conflicting_links(self, adj_list):
        for node in adj_list:
            neighbors = adj_list.get(node)
            neighbor_conflict = {}
            for neighbor in neighbors:
                path = [node,neighbor]
                e_list, e_set = self.calculate_conflicting_links(path)
                neighbor_conflict.update({neighbor:e_set})
                #print(neighbor,"e_lis",sorted(e_list))
                #print(neighbor,"e_set",sorted(e_set))
            self.__conflicting_links.update({node:neighbor_conflict})
        print("__conflicting_links",self.__conflicting_links)

    def init_wsn_substrate(self, links):
        adj_list = links
        for n in adj_list:
            self.__WSN_Substrate.add_node(n, {'rank':1, 'load':1})
            items = adj_list.get(n)
            for i in items:
                self.__WSN_Substrate.add_edge(n,i, {'plr':1, 'load':1, 'weight':1})
                self.__link_weights[(n,i)] = 1


    def update_adj_list(self, adj_list):
        self.__adj_list = adj_list

    def get_adjacency_list(self):
        self.__adj_list = {1: [2, 9],
         2: [1, 3, 10],
         3: [2, 4, 11],
         4: [3, 5, 12],
         5: [4, 6, 13],
         6: [5, 7, 14],
         7: [6, 8, 15],
         8:[7, 16],
         9:[1, 10, 17],
        10:[2, 9, 11, 18],
        11:[3, 10, 12, 19],
        12:[4, 11, 13, 20],
        13:[5, 12, 14, 21],
        14:[6, 13, 15, 22],
        15:[7, 14, 16, 23],
        16:[8, 15, 24],
        17:[9, 18, 25],
        18:[10, 17, 19, 26],
        19:[11, 18, 20, 27],
        20:[12, 19, 21, 28],
        21:[13, 20, 22, 29],
        22:[14, 21, 23, 30],
        23:[15, 22, 24, 31],
        24:[16, 23, 32],
        25:[17, 26, 33],
        26:[18, 25, 27, 34],
        27:[19, 26, 28, 35],
        28:[20, 27, 29, 36],
        29:[21, 28, 30, 37],
        30:[22, 29, 31, 38],
        31:[23, 30, 32, 39],
        32:[24, 31, 40],
        33:[25, 34, 41],
        34:[26, 33, 35, 42],
        35:[27, 34, 36, 43],
        36:[28, 35, 37, 44],
        37:[29, 36, 38, 45],
        38:[30, 37, 39, 46],
        39:[31, 38, 40, 47],
        40:[32, 39, 48],
        41:[33, 42, 49],
        42:[34, 41, 43, 50],
        43:[35, 42, 44, 51],
        44:[36, 43, 45, 52],
        45:[37, 44, 46, 53],
        46:[38, 45, 47, 54],
        47:[39, 46, 48, 55],
        48:[40, 47, 56],
        49:[41, 50],
        50:[42, 49, 51],
        51:[43, 50, 52],
        52:[44, 51, 53],
        53:[45, 52, 54],
        54:[46, 53, 55],
        55:[47, 54, 56],
        56:[48, 55],}
        return self.__adj_list


    def get_nodes_position(self):

        position = {1:(0,0),
        2:(0,0.5),
        3:(0,1),
        4:(0,1.5),
        5:(0,2),
        6:(0,2.5),
        7:(0,3),
        8:(0,3.5),
        9:(0.5,0),
        10:(0.5,0.5),
        11:(0.5,1),
        12:(0.5,1.5),
        13:(0.5,2),
        14:(0.5,2.5),
        15:(0.5,3),
        16:(0.5,3.5),
        17:(1,0),
        18:(1,0.5),
        19:(1,1),
        20:(1,1.5),
        21:(1,2),
        22:(1,2.5),
        23:(1,3),
        24:(1,3.5),
        25:(1.5,0),
        26:(1.50,0.5),
        27:(1.50,1),
        28:(1.50,1.5),
        29:(1.50,2),
        30:(1.50,2.5),
        31:(1.50,3),
        32:(1.50,3.5),
        33:(2,0),
        34:(2,0.5),
        35:(2,1),
        36:(2,1.5),
        37:(2,2),
        38:(2,2.5),
        39:(2,3),
        40:(2,3.5),
        41:(2.5,0),
        42:(2.5,0.5),
        43:(2.5,1),
        44:(2.5,1.5),
        45:(2.5,2),
        46:(2.5,2.5),
        47:(2.5,3),
        48:(2.5,3.5),
        49:(3,0),
        50:(3,0.5),
        51:(3,1),
        52:(3,1.5),
        53:(3,2),
        54:(3,2.5),
        55:(3,3),
        56:(3,3.5)}

        return position

if __name__ == '__main__':
    wsn = WSN()
    print()
    network = wsn.get_wsn_substrate()
    print(network.edge[1][2]['load'])
    print(network[1][9]['weight'])
    print(network.node[1])
    print(network.node[2]['load'])
    print(network[9])
    print(network[26][34])


    print("wsn.get_wsn_substrate().nodes(data=True):",wsn.get_wsn_substrate().nodes(data=True))
    print("wsn.get_wsn_substrate().edges(data=True):",wsn.get_wsn_substrate().edges(data=True))



    for n,d in wsn.get_wsn_substrate().nodes(data=True):
        print(n,d)

    print("===")
    for u,v,d in wsn.get_wsn_substrate().edges(data=True):
        print(u,v,d['weight'])

    print("TEST")