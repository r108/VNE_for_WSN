import networkx as nx
import matplotlib.pyplot as plt
from wsn_substrate import WSN
import config
import copy

CVIOLETBG = "\33[45m"
CBLUEBG = "\33[44m"
CGREYBG = "\33[100m"
CGREEN = "\33[32m"
CRED = "\33[31m"
CEND = "\33[0m"



def display_edge_attr(G):
    for u,v,d in G.edges_iter(data=True):
        if 'weight' in d:
            if 'plr' in d:
                if 'load' in d:
                    print "Edge", CGREEN, u, "<->", v, CEND, "has",CBLUEBG, "weight", d['weight'], CEND,CVIOLETBG, "plr", format(G[u][v]['plr']), CEND,CGREYBG, "load", format(G[u][v]['load']), CEND
                else:
                    print(CRED,"Missing",CGREYBG,"load",CEND,CRED,"attribute in",CEND,CGREEN,u,"<->",v,CEND)
            else:
                print(CRED,"Missing",CVIOLETBG,"plr",CEND,CRED,"attribute in",CEND,CGREEN,u,"<->",v,CEND)
        else:
            print(CRED,"Missing",CBLUEBG,"weight",CEND,CRED,"attribute in",CEND,CGREEN,u,"<->",v,CEND)
    #print(G[u][v].keys())
    print("")

def display_node_attr(G):
    print("")
    for n, d in G.nodes_iter(data=True):
        if 'rank' in d:
            if 'load' in d:
                print "Node",CGREEN,n,CEND,"has",CBLUEBG,"rank",d['rank'],CEND,CGREYBG,"load",d['load'],CEND
            else:
                print CRED,"Missing",CGREYBG,"load",CRED,"attribute in",CEND,CGREEN,n,CEND
        else:
            print CRED,"Missing",CBLUEBG,"rank",CRED,"attribute in",CEND,CGREEN,n,CEND
    print ""

def display_vn_node_allocation(G):
    print("")
    for n, d in G.nodes_iter(data=True):
        if 'load' in d:
            print("Node",CGREEN,n,CEND,"has",CGREYBG,"load",d['load'],CEND,"allocated")
        else:
            print(CRED, "Missing", CGREYBG, "load", CRED, "attribute in", CEND, CGREEN, n, CEND)
    print("")

def display_vn_edge_allocation(G):
    for u,v,d in G.edges_iter(data=True):
        if 'load' in d:
            print "Edge", CGREEN, u, "<->", v, CEND, "has",CGREYBG, "load", format(G[u][v]['load']), CEND,"allocated"
        else:
            print CRED,"Missing",CGREYBG,"load",CEND,CRED,"attribute in",CEND,CGREEN,u,"<->",v,CEND
    print("")

#def show_graph_plot(G,shortest_path, path):
#    plt.show() # display

def generate_plot(G,shortest_path, path,plt,weight_flag,position):

    print shortest_path, path
    print(G.nodes(data=True))
    print(G.edges(data=True))
    embeding_positions = list(map(int, path))
    colors = []
    G_source = nx.Graph()
    G_sink = nx.Graph()
    if path:
        G_source.add_node(path[0], {'rank':1, 'load':1})
        G_sink.add_node(path[len(path)-1], {'rank': 1, 'load': 1})

    for n in G.nodes():
        if n in embeding_positions:
            colors.append('r')
        else:
            colors.append('g')

    positions = position # WSN.get_nodes_position()
    print(positions)
    fixed_positions = dict()
    for n in G.nodes(data=False):
        fixed_positions.update({n: positions[n]})
    #fixed_positions = dict((n,d) for n,d in fixed_positions if n in G.nodes(data=False))

    fixed_nodes = fixed_positions.keys()
    # elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >1200]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True)]  # if d['weight'] <=1200]
    if weight_flag == True:
        edge_labels = dict([((u, v), d['weight']) for u, v, d in G.edges(data=True)])
    else:
        edge_labels = dict([((u, v), d['load']) for u, v, d in G.edges(data=True)])

    eembed = shortest_path
    # eembed.append((1,2))
    pos = nx.spring_layout(G, pos=fixed_positions, fixed=fixed_nodes)  # positions for all nodes
    # nodes

    nx.draw_networkx_nodes(G, pos, ax=plt, node_size=300, node_color=colors)
    #print("G_source.nodes",G_source.nodes(data=True))
    nx.draw_networkx_nodes(G_source, pos, ax=plt, node_size=500, node_color='y')
    nx.draw_networkx_nodes(G_sink, pos, ax=plt, node_size=600, node_color='b')
    # edges
    # nx.draw_networkx_edges(G,pos,edgelist=elarge,width=2)

    nx.draw_networkx_edges(G, pos, ax=plt, edgelist=esmall,
                           width=1, edge_color='b', style='dashed')
    if len(eembed)<0:
        nx.draw_networkx_edges(G, pos, ax=plt, edgelist=eembed, width=4, edge_color='r', style='solid')

    # labels
    nx.draw_networkx_labels(G, pos, ax=plt, font_size=15, font_family='sans-serif')
    nx.draw_networkx_edge_labels(G, pos, ax=plt, edge_labels=edge_labels,label_pos=0.3, font_size=12)

def draw_graph(wsn_substrate):
    mappings = config.current_test_best['mappings']
    position = wsn_substrate.get_nodes_position()
    #print mappings

    for vn in mappings:
        VN_nodes = vn[3]
        VN_links = vn[4]

        shortest_p = vn[5]
        path_n = vn[6]
        plotit(VN_links, shortest_p, path_n, config.current_test_best['permutation'],position)
        #print "call plotit"


def draw_graph_():
    print "lllllllllll"
    for k, v in config.best_embeddings.items():
        index = v['permutation']
        print("index", index)
        for vn in config.active_vns:
            VN_links = vn[1]
            shortest_p = vn[2]
            path_n = vn[3]
            plotit(VN_links, shortest_p, path_n, index)
            print "call plotit"

#        for perm in config.all_embeddings[index]:
#            print("perm", perm)
#            VN_links = perm[1]
#            shortest_p = perm[2]
#            path_n = perm[3]
#            plotit(VN_links, shortest_p, path_n, index)


def plotit(VN_links, shortest_path, path_nodes, index, position):
    #print "+",VN_links.edges(data=True), shortest_path, path_nodes, index
    fig = plt.figure(figsize=(30, 15), dpi=100)
    config.sp1 = copy.deepcopy(path_nodes)
    if config.online_flag:
        plt1 = fig.add_subplot(1, 3, 1)
        generate_plot(VN_links, shortest_path, path_nodes, plt1, False,position)
        plt1 = fig.add_subplot(1, 3, 2)
        generate_plot(config.committed_wsn, shortest_path, path_nodes, plt1, False,position)
        plt1 = fig.add_subplot(1, 3, 3)
        generate_plot(config.committed_wsn, shortest_path, [], plt1, True,position)
    else:
        plt1 = fig.add_subplot(1, 1, 1)
        generate_plot(VN_links, shortest_path, path_nodes, plt1, False,position)
        #plt1 = fig.add_subplot(1, 3, 2)
        #generate_plot(wsn, shortest_path, path_nodes, plt1, False,position)
        #plt1 = fig.add_subplot(1, 3, 3)
        #generate_plot(wsn, shortest_path, [], plt1, True,position)
    config.plot_counter += 1
    plt.axis('on')
    plt.savefig(str(index)+"graph_" + str(config.plot_counter) + ".png")  # save as png
    #plt.savefig(str(index)+"graph_" + str(config.plot_counter) + ".svg", format='svg')

def draw_substrate(wsn_substrate):
    fig2 = plt.figure(figsize=(30, 15), dpi=100)
    colors = []
    sizes = []
    for n in config.wsn.nodes():
        if n == 0:
            colors.append('b')
            sizes.append(700)
        else:
            colors.append('r')
            sizes.append(300)
    plt1 = fig2.add_subplot(1, 1, 1)
    position = wsn_substrate.get_nodes_position()

    VN_links = wsn_substrate.get_wsn_substrate()

    shortest_path = []
    path_nodes = []
    #generate_plot(VN_links, shortest_path, path_nodes, plt1, False, position)

    #e_labels = dict([((u, v), d['load']) for u, v, d in config.committed_wsn.edges(data=True)])
    e_labels = dict([((u, v), d['load']) for u, v, d in config.wsn.edges(data=True)])

    # nx.draw_networkx_nodes(config.wsn, wsn_substrate.get_nodes_position(), node_size=300, node_color='r')
    nx.draw(config.wsn, wsn_substrate.get_nodes_position(), nodelist=config.wsn.nodes(), node_size=sizes, alpha=0.8,node_color=colors)
    nx.draw_networkx_labels(config.wsn, wsn_substrate.get_nodes_position(), font_size=15, font_family='sans-serif')
    nx.draw_networkx_edge_labels(config.wsn,wsn_substrate.get_nodes_position(), edge_labels=e_labels, label_pos=0.3, font_size=12)
    plt.savefig(str(56) + "graph_" + str(config.plot_counter) + ".png")  # save as png
    plt.show()