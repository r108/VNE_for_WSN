import networkx as nx

class WSN():

    __WSN_Substrate = nx.DiGraph()
    __link_weights = dict()
    __adj_list = dict()
    __two_hops_list = dict()

    def __init__(self):
        self.init_wsn_substrate(self.get_adjacency_list())
        self.init_two_hop_neighborhood(self.get_adjacency_list())

    def get_wsn_substrate(self):
        return self.__WSN_Substrate

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
        39:[31, 38, 39, 47],
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

        position = {1:(1.5,-1.6),
        2:(1.5,-1.2),
        3:(1.5,-0.8),
        4:(1.5,-0.4),
        5:(1.5,0.4),
        6:(1.5,0.8),
        7:(1.5,1.2),
        8:(1.5,1.6),
        9:(1,-1.6),
        10:(1,-1.2),
        11:(1,-0.8),
        12:(1,-0.4),
        13:(1,0.4),
        14:(1,.8),
        15:(1,1.2),
        16:(1,1.6),
        17:(0.5,-1.6),
        18:(0.5,-1.2),
        19:(0.5,-0.8),
        20:(0.5,-0.4),
        21:(0.5,0.4),
        22:(0.5,0.8),
        23:(0.5,1.2),
        24:(0.5,1.6),
        25:(0,-1.6),
        26:(0,-1.2),
        27:(0,-0.8),
        28:(0,-0.4),
        29:(0,0.4),
        30:(0,0.8),
        31:(0,1.2),
        32:(0,1.6),
        33:(-0.5,-1.6),
        34:(-0.5,-1.2),
        35:(-0.5,-0.8),
        36:(-0.5,-0.4),
        37:(-0.5,0.4),
        38:(-0.5,0.8),
        39:(-0.5,1.2),
        40:(-0.5,1.6),
        41:(-1,-1.6),
        42:(-1,-1.2),
        43:(-1,-0.8),
        44:(-1,-0.4),
        45:(-1,0.4),
        46:(-1,0.8),
        47:(-1,1.2),
        48:(-1,1.6),
        49:(-1.5,-1.6),
        50:(-1.5,-1.2),
        51:(-1.5,-0.8),
        52:(-1.5,-0.4),
        53:(-1.5,0.4),
        54:(-1.5,0.8),
        55:(-1.5,1.2),
        56:(-1.5,1.6)}

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