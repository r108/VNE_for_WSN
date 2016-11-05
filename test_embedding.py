try:
    import matplotlib.pyplot as plt
except:
    raise

import copy
from wsn_substrate import WSN
import networkx as nx
from link_weight import LinkCost
import sp_dijkstra as sp
import config

CVIOLETBG = '\33[45m'
CBLUEBG = '\33[44m'
CGREYBG = '\33[100m'
CGREEN = '\33[32m'
CRED = '\33[31m'
CEND = '\33[0m'

wsn_substrate = WSN()
exit_flag = True

def update_all_links_attributes(plr, load):
    for u,v,d  in wsn.edges_iter(data=True):
        wsn[u][v]['plr'] = plr
        wsn[u][v]['load'] = load
        link_weight = LinkCost(wsn[u][v]['plr'], wsn[u][v]['load'])
        wsn[u][v]['weight'] = link_weight.get_weight(link_weight)
        link_weights[(u, v)] = link_weight.get_weight(link_weight)
     #   print(link_weight.get_weight(link_weight))
    print(wsn.edges(data=True))
    print("")

def update_node_attributes(Nodes, node, load):
    for n, d in Nodes.nodes_iter(data=True):
        #print("n is",type(n),"and node is",type(node))
        if n == int(node):
            d['load'] = d['load']+int(load)
            #d['rank'] = len(adj[n])

def update_link_attributes(u, v, plr, load):

    if plr is -1:
        wsn[u][v]['plr'] = wsn[u][v]['plr']
    else:
        wsn[u][v]['plr'] += plr
    if load is -1:
        wsn[u][v]['load'] = wsn[u][v]['load']
    else:
        wsn[u][v]['load'] += load

    link_weight = LinkCost(wsn[u][v]['plr'], wsn[u][v]['load'])
    wsn[u][v]['weight'] = link_weight.get_weight(link_weight)
    #links[(u, v)] = link_weight.get_weight(link_weight)


def display_edge_attr(G):
    for u,v,d in G.edges_iter(data=True):
        if 'weight' in d:
            if 'plr' in d:
                if 'load' in d:
                    print("Edge", CGREEN, u, "<->", v, CEND, "has",
                          CBLUEBG, "weight", d['weight'], CEND,
                          CVIOLETBG, "plr", format(G[u][v]['plr']), CEND,
                          CGREYBG, "load", format(G[u][v]['load']), CEND)
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
                print("Node",CGREEN,n,CEND,"has",CBLUEBG,"rank",d['rank'],CEND,CGREYBG,"load",d['load'],CEND)
            else:
                print(CRED,"Missing",CGREYBG,"load",CRED,"attribute in",CEND,CGREEN,n,CEND)
        else:
            print(CRED,"Missing",CBLUEBG,"rank",CRED,"attribute in",CEND,CGREEN,n,CEND)
    print("")


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
            print("Edge", CGREEN, u, "<->", v, CEND, "has",
                  CGREYBG, "load", format(G[u][v]['load']), CEND,"allocated")
        else:
            print(CRED,"Missing",CGREYBG,"load",CEND,CRED,"attribute in",CEND,CGREEN,u,"<->",v,CEND)
    print("")

def show_graph_plot(G,shortest_path, path):

    plt.show() # display

def generate_plot(G,shortest_path, path,plt,weight_flag):

    print("G",type(G),"|",G)
    print("G.nodes type",type(G.nodes()),"|",G.nodes())
    print("G.edges type",type(G.edges()),"|",G.edges())





    embeding_positions = list(map(int, path))
    colors = []

    for n in G.nodes():
        if n in embeding_positions:
            colors.append('r')
        else:
            colors.append('g')

    positions = wsn_substrate.get_nodes_position()
    fixed_positions = dict()
    for n in G.nodes(data=False):
        fixed_positions.update({n: positions[n]})
    # fixed_positions = dict(x,d for x,d in fixed_positions if x  in G.nodes(data=False))

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
    # edges
    # nx.draw_networkx_edges(G,pos,edgelist=elarge,width=2)
    nx.draw_networkx_edges(G, pos, ax=plt, edgelist=eembed, width=10, edge_color='r')

    nx.draw_networkx_edges(G, pos, ax=plt, edgelist=esmall,
                           width=4, alpha=0.5, edge_color='b', style='dashed')
    # labels
    nx.draw_networkx_labels(G, pos, ax=plt, font_size=15, font_family='sans-serif')
    nx.draw_networkx_edge_labels(G, pos, ax=plt, edge_labels=edge_labels, label_pos=0.3, font_size=12)


def display_data_structs():
    print("Nodes - ", wsn.nodes(data=True))
    print("Edges - ", wsn.edges(data=True))
    print("Adjacency list - ", wsn.adjacency_list())
    print("adjacencies- ", adjacencies)
    print("two_hops_list - ", two_hops_list)
    print("link_weights- ", link_weights)


######################################################################################################

def map_links(e_list, e_list2, required_load):
    print("@@",e_list2)
    for u,v in e_list2:
        update_link_attributes(int(u), int(v), -1, (e_list.count((u,v)) * required_load))
        #if (e_fr,e_to) not in config.allocated_links_weight:
        config.allocated_links_weight.update({(u,v): wsn[u][v]['weight']})
        config.allocated_links_load.update({(u, v): wsn[u][v]['load']})
#        print(e_fr,e_to,"occur ", e_list.count((e_fr,e_to)),"times")
    print("config.allocated_links_weight.",config.allocated_links_weight)
    print("config.allocated_links_load.", config.allocated_links_load)
    #print("get reduced adj",type(config.allocated_links_weight)

def map_nodes(all_path_nodes, required_load):
    for idx,pn in enumerate(all_path_nodes):
        if idx == 0:
            update_node_attributes(wsn, pn, required_load)
        elif idx == (len(all_path_nodes) - 1):
            update_node_attributes(wsn, pn, required_load)
        else:
            update_node_attributes(wsn, pn, required_load)

def commit_vn(VN_nodes, VN_links, required_load,e_list, e_list2, path_nodes, shortest_path):
    map_nodes(VN_nodes.nodes(), required_load)
    map_links(e_list, e_list2, required_load)
    vn = (VN_nodes, VN_links, shortest_path, path_nodes)
    config.VWSNs.append(vn)

    print("config.link_weights after commit= ", config.link_weights)
    print("original link_weights after commit", link_weights)

    display_edge_attr(wsn)
    display_node_attr(wsn)
    display_vn_node_allocation(VN_nodes)
    display_vn_edge_allocation(VN_links)

    fig = plt.figure(figsize=(30,15),dpi=150)

    plt1 = fig.add_subplot(1, 3, 1)

    generate_plot(VN_links, shortest_path, path_nodes, plt1, False)
    plt1 = fig.add_subplot(1, 3, 2)
    generate_plot(WSN_Links, shortest_path, path_nodes, plt1, False)
    plt1 = fig.add_subplot(1, 3, 3)
    generate_plot(WSN_Links, shortest_path, path_nodes, plt1, True)
    config.plot_counter += 1

    plt.axis('on')
    plt.savefig("graph_" + str(config.plot_counter) + ".png")  # save as png


def check_link_constraints(e_list, e_list2, required_load, required_plr, shortest_path):
    VN_links = nx.DiGraph()
    print("check_link_constraints")

    for u,v in e_list2:
        if wsn.edge[u][v]['load'] + (required_load * e_list.count((u,v))) > 100:
            print("Link",u, v,"requires",wsn.edge[u][v]['load']," + ",(required_load * e_list.count((u, v))), "but have not got enough")
            return (u,v),VN_links
        else:
            allocated_load = (required_load * e_list.count((u, v)))
            VN_links.add_edge(u,v, {'load':allocated_load})
            return_value = (0,0)
    print("RETURN LINK ",return_value)
    return return_value,VN_links


def check_node_constraints(nodes_in_path, required_load):
    VN_nodes = nx.DiGraph()
    #print("checking nodes -----",nodes_in_path)
    print("check_node_constraints")
    for idx,n in enumerate(nodes_in_path):
        VN_nodes.add_node(n, {'load': required_load})
        if wsn.node[n]['load'] + required_load > 100:
            if idx == 0:
                    print("Source node",n," has - ",wsn.node[n]['load'],"but require",+ required_load )
                    return n, VN_nodes
            elif idx == (len(nodes_in_path) - 1):
                    print("Sink node",n,"has - ",wsn.node[n]['load'],"but require",+ required_load )
                    return n, VN_nodes
            else:
                    print("Relay node",n,"has - ",wsn.node[n]['load'],"but require",+ required_load )
                    return n, VN_nodes
    return 0, VN_nodes

def verify(link_reqiurement, frm, to, node_requirement):
    config.counter_value = config.counter_value +1
    print("verify ----------------------------",config.counter_value,"counter ")
    node_check, VN_nodes = check_node_constraints([frm,to], node_requirement)
    shortest_path, path_nodes = sp.get_shortest_path(config.reduced_adj, config.link_weights, frm, to)
    if node_check != 0:
        print("node ", node_check, "does not have enough resource\nEMBEDDING FAILED!")
        return
    else:
        print("source and sink nodes have enough resource")

    if config.has_embedding == True:
        if (0,0) in config.avoid: config.avoid.remove((0,0))

        #shortest_path, path_nodes = sp.get_shortest_path(config.reduced_adj, config.link_weights, frm, to)
        e_list, e_list2 = get_conflicting_links(path_nodes)
        print("cconfig.failed_links_list",config.failed_links_list)
        for u,v in config.avoid:
            config.avoid.remove((u, v))
            print("ccconfig.avoid after", config.avoid)
            #print("sp",shortest_path,"\nn", path_nodes )

            print("u,v",u,v)

            for idx, n in enumerate(path_nodes):
                print("!n",n,"!!!",idx,"path_nodes",path_nodes)
                config.link_weights[(u,v)] = 1000000

                if (n in config.two_hops.get(u)):
                    print(u,"config.two_hops.get(u)",config.two_hops.get(u))
                    if n < path_nodes[idx+1]:
                        print("!1", idx, "!path_nodes fails here", path_nodes)
                        config.link_weights[(n,path_nodes[idx+1])] = 1000000
                    else:
                        config.link_weights[(path_nodes[idx + 1],n)] = 1000000

                    if path_nodes[idx+1] < u:
                        config.link_weights[(path_nodes[idx + 1], u)] = 1000000
                    else:
                        config.link_weights[(u, path_nodes[idx + 1])] = 1000000


                elif (n in config.two_hops.get(v)):
                    print(v,"config.two_hops.get(v)",config.two_hops.get(v))
                    print("!!path_nodes",idx," fails here", path_nodes)
                    if n < path_nodes[idx + 1]:
                        print("!",idx,"!path_nodes fails here", path_nodes)
                        config.link_weights[(n, path_nodes[idx + 1])] = 1000000
                    else:
                        config.link_weights[(path_nodes[idx + 1], n)] = 1000000

                    if path_nodes[idx + 1] < v:
                        config.link_weights[(path_nodes[idx + 1], v)] = 1000000
                    else:
                        config.link_weights[(v, path_nodes[idx + 1])] = 1000000
            print("path enumeration has finished")
            print("n is ", n," and idx is ",idx)



        #shortest_path, path_nodes = sp.get_shortest_path(config.reduced_adj, config.link_weights, frm, to)
        if shortest_path is None:
            print("EMBEDDING HAS FAILED!")
            return
    else:
        #shortest_path, path_nodes = sp.get_shortest_path(adjacencies, link_weights, frm, to)
        e_list, e_list2 = get_conflicting_links(path_nodes)



    # get list of unique nodes from conflicting link list
    nodes_in_path = []
    for e1, e2 in e_list2:
        if e1 not in nodes_in_path:
            nodes_in_path.append(e1)
        if e2 not in nodes_in_path:
            nodes_in_path.append(e2)

    node_check, VN_nodes = check_node_constraints(nodes_in_path, node_requirement)

    if node_check != 0:
        print("node ", node_check, "does not have enough resource\nEMBEDDING FAILED!")
        return
    else:
        print("all nodes in path have enough resource")

    link_check, VN_links = check_link_constraints(e_list, e_list2, link_reqiurement['load'], link_reqiurement['plr'],shortest_path)
    #print("link_check",link_check)
    if link_check != (0,0):
        print("link_check is", link_check)
        if config.counter_value > 50:
            print(link_check, "do not have enough resource")
            print("EMBEDDING HAS FAILED!!")
            return
        if link_check not in config.avoid:
            print("was not in link_check ", link_check)
            config.avoid.append(link_check)
        else:
            print("already in link_check ", link_check)

        if link_check not in config.failed_links_list:
            print(link_check,"was added to config.failed_links_list")
            config.failed_links_list.append(link_check)

        print(link_check,"need more")
        verify(link_reqiurement, frm, to, node_requirement)

 #       reduce_feasible_edges(link_reqiurement, link_check_u, link_check_v, node_requirement)
    else:
        print("edges have enough resource")
        print("++SUCCESSFUL EMBEDDING++")
        config.feasible = True
        config.has_embedding = True
        commit_vn(VN_nodes,VN_links,node_requirement,e_list, e_list2, path_nodes, shortest_path)


def embed_vn(VN):

    config.reduced_adj = copy.deepcopy(wsn_substrate.get_adjacency_list())
    config.link_weights = copy.deepcopy(link_weights)
    config.two_hops = copy.deepcopy(two_hops_list)

    print("VN embedding: ", VN)
    print("@links->config.link_weights",config.link_weights)


    vwsn_nodes = VN[1]
    link_reqiurement = VN[2]

    frm = list(vwsn_nodes)[0]
    to = list(vwsn_nodes)[2]

    node_requirement = vwsn_nodes[1]['load']
    config.avoid = [(0,0)]
    verify(link_reqiurement, frm, to, node_requirement)

def get_conflicting_links(path_nodes):
    p_nodes = []
    p_nodes.extend(path_nodes)
    counter1 = 0
    elist = []
    all_path_nodes = []
    all_path_nodes.extend(p_nodes)
    p_nodes.pop()
    for pn in p_nodes:
        counter1 =+ 1
        already_added_list = []
        already_added_list.append(pn)
        for n in config.reduced_adj.get(pn):
            counter1 =+ 1
            elist.append((pn, n))
            elist.append((n, pn))
            already_added_list.append(n)
            for nn in config.reduced_adj.get(n):
                counter1 =+ 1
                elist.append((n, nn))
                already_added_list.append(nn)
                print("already_added_list after adding nn", nn, "|", already_added_list)

    elist2 = []
    elist2 = list(set(elist))

    print("get conflict elist: ", elist)
    print("get conflict elist2: ", elist2)
    print("ccounter1 ",counter1)

    return elist, elist2

if __name__ == '__main__':

    link_weights = wsn_substrate.get_link_weights()

    adjacencies = wsn_substrate.get_adjacency_list()
    wsn = wsn_substrate.get_wsn_substrate()
    two_hops_list = wsn_substrate.get_two_hops_list()

    WSN_Links = nx.DiGraph(wsn.edges(data=True))
    print("@@@WSN_Links",WSN_Links.edges())
    update_all_links_attributes(1, 1)

    shortest_path, path_nodes = [],[] #sp.display_shortest_path(get_shortest_path(adj,links,1,6))

    display_edge_attr(wsn)
    display_node_attr(wsn)
    display_data_structs()

    while exit_flag is True:
        print("\n---->\n0 - Exit\n1 - Embed\n2 - Draw")
        user_input = input(': ')

        if user_input is '1':
            source = input(" source node: ")
            if source != "":
                sink = input(" sink node: ")
            else:
                continue
            if sink != "":
                quota = input(" quota: ")
            else:
                continue

            VWSN_nodes = (int(source), {'load': int(quota)},
                          int(sink), {'load': int(quota)})

            link_reqiurement = {'load': int(quota), 'plr': 40}
            VN = (1000, VWSN_nodes, link_reqiurement)

            embed_vn(VN)
            # print_conflicting_lnks(int(conflict_node1),int(conflict_node2))

        elif user_input is '2':
            # draw_graph()
            display_edge_attr(wsn)
            display_node_attr(wsn)

            for idx, vwsn in enumerate(config.VWSNs):
                print("vwsn", idx, "nodes", vwsn[0].nodes(data=True))
                print("vwsn", idx, "links", vwsn[1].edges(data=True))
                print("VWSN ", idx, "allocations:")
                display_vn_edge_allocation(vwsn[1])
                display_vn_node_allocation(vwsn[0])
    #            show_graph_plot(vwsn[1], vwsn[2], vwsn[3])
                print("wsn.get_wsn_links()", vwsn[1].edges(data=True), "\n shortest_path", vwsn[2],
                      "\n path_nodes", vwsn[3])
                print("VN nodes", vwsn[1].nodes(data=False))

