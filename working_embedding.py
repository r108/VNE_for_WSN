try:
    import matplotlib.pyplot as plt
except:
    raise

import copy
import itertools as itool
from wsn_substrate import WSN
import networkx as nx
from link_weight import LinkCost
import config
import vne
import time

CVIOLETBG = '\33[45m'
CBLUEBG = '\33[44m'
CGREYBG = '\33[100m'
CGREEN = '\33[32m'
CRED = '\33[31m'
CEND = '\33[0m'

wsn_substrate = WSN()
exit_flag = True

def update_all_links_attributes(wsn,plr, load):
    for u,v,d  in wsn.edges_iter(data=True):
        wsn[u][v]['plr'] = plr
        wsn[u][v]['load'] = load
        link_weight = LinkCost(wsn[u][v]['plr'], wsn[u][v]['load'])
        wsn[u][v]['weight'] = link_weight.get_weight(link_weight)
        #link_weights[(u, v)] = link_weight.get_weight(link_weight)
     #  # print(link_weight.get_weight(link_weight))
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
        config.wsn_for_this_perm[u][v]['plr'] = config.wsn_for_this_perm[u][v]['plr']
    else:
        config.wsn_for_this_perm[u][v]['plr'] += plr
    if load is -1:
        config.wsn_for_this_perm[u][v]['load'] = config.wsn_for_this_perm[u][v]['load']
    else:
        config.wsn_for_this_perm[u][v]['load'] += load

    link_weight = LinkCost(config.wsn_for_this_perm[u][v]['plr'], config.wsn_for_this_perm[u][v]['load'])
    config.wsn_for_this_perm[u][v]['weight'] = link_weight.get_weight(link_weight)
    config.link_weights_for_this_perm[(u, v)] = link_weight.get_weight(link_weight)

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

    positions = wsn_substrate.get_nodes_position()
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

def draw_graph():
    for k, v in config.best_embeddings.items():
        index = v['permutation']
        print("index", index)
        print("config.all_embeddings[index]", config.all_embeddings[index])

        for perm in config.all_embeddings[index]:
            print("perm", perm)
            VN_links = perm[1]
            shortest_p = perm[2]
            path_n = perm[3]
            plotit(VN_links, shortest_p, path_n, index)

def display_data_structs():
    print("Nodes - ", wsn.nodes(data=True))
    print("Edges - ", wsn.edges(data=True))
    print("Adjacency list - ", wsn.adjacency_list())
    print("adjacencies- ", adjacencies)
    print("two_hops_list - ", two_hops_list)
    print("link_weights- ", link_weights)


######################################################################################################

def on_line_vn_request():
    source = input(" source node: ")
    if source != "":
        sink = input(" sink node: ")
    else:
        return
    if sink != "":
        quota = input(" quota: ")
    else:
        return
    if quota != "":
        VWSN_nodes = (int(source), {'load': int(quota)},
                      int(sink), {'load': int(quota)})
        link_reqiurement = {'load': int(quota), 'plr': 40}
        VN = (1000, VWSN_nodes, link_reqiurement)
        embed_vn(VN)
    else:
        return

def map_links(e_list, e_list2, required_load):
    for u,v in e_list2:
        update_link_attributes(int(u), int(v), -1, (e_list.count((u,v)) * required_load))
        config.allocated_links_weight.update({(u,v): config.wsn_for_this_perm[u][v]['weight']})
        config.allocated_links_load.update({(u, v): config.wsn_for_this_perm[u][v]['load']})
#        print(e_fr,e_to,"occur ", e_list.count((e_fr,e_to)),"times")
#    print("config.allocated_links_weight.",config.allocated_links_weight)
#    print("config.allocated_links_load.", config.allocated_links_load)

def map_links_cost(e_list, e_list2, required_load):
    link_embedding_cost = 0
    for u,v in e_list2:
        required = (e_list.count((u,v)) * required_load)
        link_embedding_cost +=required
        update_link_attributes(int(u), int(v), -1, required)
        config.allocated_links_weight.update({(u,v): config.wsn_for_this_perm[u][v]['weight']})
        config.allocated_links_load.update({(u, v): config.wsn_for_this_perm[u][v]['load']})
#    print("config.allocated_links_weight.",config.allocated_links_weight)
#    print("config.allocated_links_load.", config.allocated_links_load)
#    print("get reduced adj",type(config.allocated_links_weight)
    return link_embedding_cost

def map_nodes(all_path_nodes, required_load):
    for idx,pn in enumerate(all_path_nodes):
        if idx == 0:
            update_node_attributes(config.wsn_for_this_perm, pn, required_load)
        elif idx == (len(all_path_nodes) - 1):
            update_node_attributes(config.wsn_for_this_perm, pn, required_load)
        else:
            update_node_attributes(config.wsn_for_this_perm, pn, required_load)

def map_nodes_cost(all_path_nodes, required_load):
    node_embedding_cost = 0
    for idx, pn in enumerate(all_path_nodes):
        if idx == 0:
            update_node_attributes(config.wsn_for_this_perm, pn, required_load)
            node_embedding_cost += (required_load)
        elif idx == (len(all_path_nodes) - 1):
            update_node_attributes(config.wsn_for_this_perm, pn, required_load)
            node_embedding_cost += (required_load)
        else:
            update_node_attributes(config.wsn_for_this_perm, pn, required_load)
            node_embedding_cost += (required_load)
    return  node_embedding_cost

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
#    if len(config.VWSNs) == 1:
    config.VN_l1 = copy.deepcopy(VN_links)
    config.sp1 = copy.deepcopy(path_nodes)
 #   elif len(config.VWSNs) == 2:
    plt1 = fig.add_subplot(1, 3, 1)
    generate_plot(VN_links, shortest_path, path_nodes, plt1, False)
    plt1 = fig.add_subplot(1, 3, 2)
    generate_plot(wsn, shortest_path, path_nodes, plt1, False)
    plt1 = fig.add_subplot(1, 3, 3)
    generate_plot(wsn, shortest_path, [], plt1, True)
    config.plot_counter += 1

    plt.axis('on')
    plt.savefig("graph_" + str(config.plot_counter) + ".png")  # save as png

def commit(VN_nodes, VN_links, required_load,e_list, e_list2, path_nodes, shortest_path):
    n_cost = map_nodes_cost(VN_nodes.nodes(), required_load)
    l_cost = map_links_cost(e_list, e_list2, required_load)
    vn = (VN_nodes, VN_links, shortest_path, path_nodes, (n_cost+l_cost))
    config.VWSNs.append(vn)
    config.current_emb_costs.update({vn[3][0]: vn[4]})
    config.overall_cost += vn[4]
#    print(vn[3])
#    print(vn[4])
#    print(config.VWSNs)
#    print("config.current_emb_costs", config.current_emb_costs)
def check_link_constraints(e_list, e_list2, load, required_plr, shortest_path):
    VN_links = nx.DiGraph()
    for u,v in e_list2:
        #config.total_operations +=  1
        required_load = load * e_list.count((u,v))
        if config.wsn_for_this_perm.edge[u][v]['load'] + required_load > 100:
 #           print("Link",u, v,"requires",wsn.edge[u][v]['load']," + ",required_load, "but have not got enough")
            return (u,v),VN_links
        else:
            VN_links.add_edge(u,v, {'load':required_load})
            return_value = (0,0)
    print("RETURN LINK ",return_value)
    return return_value,VN_links

def check_node_constraints(nodes_in_path, required_load):
    VN_nodes = nx.DiGraph()
    for idx,n in enumerate(nodes_in_path):
        #config.total_operations +=  1
        VN_nodes.add_node(n, {'load': required_load})
        if config.wsn_for_this_perm.node[n]['load'] + required_load > 100:
            if idx == 0:
                    #print("Source node",n," has - ",wsn.node[n]['load'],"but require",+ required_load )
                    return n, VN_nodes
            elif idx == (len(nodes_in_path) - 1):
                    #print("Sink node",n,"has - ",wsn.node[n]['load'],"but require",+ required_load )
                    return n, VN_nodes
            else:
                    #print("Relay node",n,"has - ",wsn.node[n]['load'],"but require",+ required_load )
                    return n, VN_nodes
    return 0, VN_nodes

def get_shortest_path(graph, frm, to):
    #modified the below file to return 2 parameters (path, length) instead of 1 path
    #/home/roland/anaconda3/lib/python3.5/site-packages/networkx/algorithms/shortest_paths/weighted.py
    path, length = nx.dijkstra_path(graph, source=frm, target=to, weight='weight')
#    length = nx.dijkstra_path_length(graph, source=frm, target=to, weight='weight')
    print('Shortest path weight is ',length)
    if (path is None) or (length > 10000000):
        return None, None
    s_path = []
    print('Shortest path is ', end="")
    for idx, p in enumerate(path):
        if idx == 0:
            # shortest_path.append((p, p+1))
            print(CGREEN, p, "->", end="")
        elif idx == (len(path) - 1):
            print(p, CEND)
        else:
            print(p, "->", end="")
        if idx != len(path) - 1:
            s_path.append((path[idx], path[idx + 1]))
#    print("Shortest path links  ", shortest_path,"\n")
    return s_path, path

def verify_feasibility(link_reqiurement, frm, to, node_requirement):
    #config.total_operations +=  1
    config.counter_value = config.counter_value+1
    print("verify ----------------------------",config.counter_value,"counter ")
    node_check, VN_nodes = check_node_constraints([frm,to], node_requirement)
    if node_check != 0:
        print("node ", node_check, "does not have enough resource\nEMBEDDING FAILED!")
        return
    else:
        #print("source and sink nodes have enough resource")
        pass
    shortest_path, path_nodes = get_shortest_path(config.current_wsn, frm, to)
    if shortest_path is None:
        print("No feasible path!\nEMBEDDING HAS FAILED!")
        return
    #else:
        #print("Feasible path exists |", shortest_path)
    e_list, e_set = get_conflicting_links(path_nodes)
    # get list of unique nodes from conflicting link list
    effected_nodes = []
    for u, v in e_set:
        if u not in effected_nodes:
            effected_nodes.append(u)
        if v not in effected_nodes:
            effected_nodes.append(v)
    node_check, VN_nodes = check_node_constraints(effected_nodes, node_requirement)
    if node_check != 0:
        print("node ", node_check, "does not have enough resource\nEMBEDDING FAILED!")
        return
    link_check, VN_links = check_link_constraints(e_list, e_set, link_reqiurement['load'], link_reqiurement['plr'],shortest_path)
    if link_check != (0,0):
#        print("link_check is", link_check)
        if link_check not in config.avoid:
            #print("was not in link_check ", link_check)
            config.avoid.append(link_check)
        if link_check not in config.failed_links_list:
            #print(link_check,"was added to config.failed_links_list", config.failed_links_list)
            config.failed_links_list.append(link_check)
#        verify(link_reqiurement, frm, to, node_requirement)
        if recalculate_path_weights(frm,to,path_nodes,shortest_path):
            verify_feasibility(link_reqiurement, frm, to, node_requirement)
        else:
            return
    else:
        #print("edges have enough resource")
        print("++SUCCESSFUL EMBEDDING++")
        config.feasible = True
        config.has_embedding = True
        commit(VN_nodes, VN_links, node_requirement, e_list, e_set, path_nodes, shortest_path)

def verify(link_reqiurement, frm, to, node_requirement):
    config.counter_value =config.counter_value+1
    print("verify ----------------------------",config.counter_value,"counter ")
    node_check, VN_nodes = check_node_constraints([frm,to], node_requirement)

    if node_check != 0:
        print("node ", node_check, "does not have enough resource\nEMBEDDING FAILED!")
        return
    else:
        print("source and sink nodes have enough resource")
    print("call sp",config.link_weights)
    shortest_path, path_nodes = get_shortest_path(config.current_wsn, frm, to)
    print(shortest_path)
    print(path_nodes)
    if shortest_path is None:
        print("No feasible path!\nEMBEDDING HAS FAILED!")
        return
    else:
        print("Feasible path exists |", shortest_path)
    e_list, e_list2 = get_conflicting_links(path_nodes)
    # get list of unique nodes from conflicting link list
    effected_nodes = []
    for u, v in e_list2:
        if u not in effected_nodes:
            effected_nodes.append(u)
        if v not in effected_nodes:
            effected_nodes.append(v)
    node_check, VN_nodes = check_node_constraints(effected_nodes, node_requirement)
    if node_check != 0:
        print("node ", node_check, "does not have enough resource\nEMBEDDING FAILED!")
        return
    else:
        print("all nodes in path have enough resource")
    link_check, VN_links = check_link_constraints(e_list, e_list2, link_reqiurement['load'], link_reqiurement['plr'],shortest_path)
    if link_check != (0,0):
        print("link_check is", link_check)
        if link_check not in config.avoid:
            print("was not in link_check ", link_check)
            config.avoid.append(link_check)
        else:
            print("already in link_check ", link_check)
        if link_check not in config.failed_links_list:
            print(link_check,"was added to config.failed_links_list", config.failed_links_list)
            config.failed_links_list.append(link_check)
        print(link_check,"need more")
#        verify(link_reqiurement, frm, to, node_requirement)
        if recalculate_path_weights(frm,to,path_nodes,shortest_path):
            verify(link_reqiurement, frm, to, node_requirement)
        else:
            return
 #       reduce_feasible_edges(link_reqiurement, link_check_u, link_check_v, node_requirement)
    else:
        print("edges have enough resource")
        print("++SUCCESSFUL EMBEDDING++")
        config.feasible = True
        config.has_embedding = True
        commit_vn(VN_nodes,VN_links,node_requirement,e_list, e_list2, path_nodes, shortest_path)

def show_penalized_links():
    print("&&&",config.avoid)
    config.penalized_list = list(set(config.penalized_list))
    for u, v in config.penalized_list:
        print(u,v,config.current_wsn[u][v]['weight'])

def recalculate_path_weights(frm,to,path_nodes,shortest_path):
    for (u, v) in config.avoid:
        ##config.total_operations += 1
        config.avoid.remove((u, v))
        config.current_wsn[u][v]['weight'] = 10000000
        config.penalized_list.append((u, v))
        if u == frm:
            print("Source node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
            return False
        if v == frm:
            print("Source node v", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
            return False
        elif u == to:
            print("Sink node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
            return False
#        two_hops_of_u = [x for x in config.two_hops.get(u) if x not in config.reduced_adj.get(u)]
#        two_hops_of_v = [x for x in config.two_hops.get(v) if x not in config.reduced_adj.get(v)]
        shared_neighbors_u = list(set(path_nodes) & set(config.reduced_adj.get(u)))
        shared_neighbors_v = list(set(path_nodes) & set(config.reduced_adj.get(v)))
        common_neighbors = list(set(shared_neighbors_u) | set(shared_neighbors_v))
        if (u, v) in shortest_path:
           print(u,v,"u-v link in path does not have enough resource!")
           if v == to:
               if len(path_nodes)>2:
                   print("path length is 2 nodes")
               else:
                   print("Sink node v", v, "does not have enough resource!\nEMBEDDING HAS FAILED!")
                   return False
        elif (v, u) in shortest_path:
            print(v, u,"v-u link in path does not have enough resource!")
            config.current_wsn[v][u]['weight'] = 10000000
            config.penalized_list.append((v, u))
            if v == to:
                if len(path_nodes) > 2:
                    print("path length is 2 nodes")
                else:
                    print("SSink node v", v, "does not have enough resource!\nEMBEDDING HAS FAILED!")
                    return False
        elif u in path_nodes:
            i = path_nodes.index(u)
            # print(u,"index is",i)
            config.current_wsn[path_nodes[i - 1]][path_nodes[i]]['weight'] = 10000000
            config.penalized_list.append((path_nodes[i - 1], path_nodes[i]))
            if u == frm:
#                   if len(path_nodes) > 2:
                print("SSource node u", u, "does not have enough resource!\nEMBEDDING HAS FAILED!")
                return False
        elif v in path_nodes:
            i = path_nodes.index(v)
            config.current_wsn[path_nodes[i - 1]][path_nodes[i]]['weight'] = 10000000
            config.penalized_list.append((path_nodes[i - 1], path_nodes[i]))
            if v == frm:
#                   if len(path_nodes) > 2:
                print("SSource node v", v, "does not have enough resource!\nEMBEDDING HAS FAILED!")
                return False
        else:
            highest = 0
            for nbr in common_neighbors:
                if nbr in path_nodes:
                    i = path_nodes.index(nbr)
                    if i > highest:
                        highest = i
            if highest > 0:
                config.current_wsn[path_nodes[highest-1]][path_nodes[highest]]['weight'] = 10000000
                config.penalized_list.append((path_nodes[highest-1], path_nodes[highest]))
#    show_penalized_links()
    return True

def embed_vn(VN):
    config.current_wsn = copy.deepcopy(config.wsn_for_this_perm)
    config.reduced_adj = copy.deepcopy(config.adjacencies_for_this_perm)
    config.link_weights = copy.deepcopy(config.link_weights_for_this_perm)
    config.two_hops = copy.deepcopy(two_hops_list)
    config.penalized_list = []
    vwsn_nodes = VN[1]
    link_reqiurement = VN[2]
    frm = list(vwsn_nodes)[0]
    to = list(vwsn_nodes)[2]
    node_requirement = vwsn_nodes[1]['load']
    config.avoid = []
    verify(link_reqiurement, frm, to, node_requirement)

def get_conflicting_links(path_nodes):
    tx_nodes = copy.deepcopy(path_nodes)
    tx_nodes.pop()
#    print("tx_nodes",tx_nodes,"\npath_nodes",path_nodes)
    effected_edges = []
#    print("initialize effected_edges",effected_edges)
    for i,tx in enumerate(tx_nodes):
        visited_nodes = []
#        print(i,"visit tx", tx)
        visited_nodes.append(tx)
#        print(tx,"appended to visited nodes", visited_nodes)
        for n in config.reduced_adj.get(tx):
#            print(tx,"-> visit n", n)
            if n not in visited_nodes:
#                print("add if n not in []",visited_nodes)
                effected_edges.append((tx, n))
                effected_edges.append((n, tx))
            for nn in config.reduced_adj.get(n):
                ##config.total_operations += 1
#                print(n, "->-> visit nn", nn)
                if nn not in visited_nodes:
#                    print("add if nn not in []", visited_nodes)
                    effected_edges.append((n, nn))
                    effected_edges.append((nn, n))
            visited_nodes.append(n)
#            print(n, "appended to visited nodes in tx", visited_nodes)
#            print(i," in length",len(path_nodes))
        rx = path_nodes[i+1]
#        print(i, "visit rx", rx)
        for n in config.reduced_adj.get(rx):
#            print(rx, "-> visit n", n)
            if n not in visited_nodes:
                for nn in config.reduced_adj.get(n):
                    ##config.total_operations += 1
#                    print(n, "->-> visit nn", nn)
                    if nn not in visited_nodes:
#                        print("add if nn not in []", visited_nodes)
                        effected_edges.append((n, nn))
                visited_nodes.append(n)
#                print(n, "appended to visited nodes in rx", visited_nodes)
    effected_edges_set = list(set(effected_edges))
    return effected_edges, effected_edges_set

def embed(vnr):
    print("BEGIN VNR EMBEDDING", vnr)
    config.current_wsn = copy.deepcopy(config.wsn_for_this_perm)
    config.reduced_adj = copy.deepcopy(config.adjacencies_for_this_perm)
    config.link_weights = copy.deepcopy(config.link_weights_for_this_perm)
    config.two_hops = copy.deepcopy(two_hops_list)
    config.penalized_list = []
    vwsn_nodes = vnr[1]
    link_reqiurement = vnr[2]
    frm = list(vwsn_nodes)[0]
    to = list(vwsn_nodes)[2]
    node_requirement = vwsn_nodes[1]['load']
    config.avoid = []
    verify_feasibility(link_reqiurement, frm, to, node_requirement)

def plotit(VN_links, shortest_p, path_n, index):
    fig = plt.figure(figsize=(30, 15), dpi=150)
    #    if len(config.VWSNs) == 1:
    config.VN_l1 = copy.deepcopy(VN_links)
    config.sp1 = copy.deepcopy(path_nodes)
    #   elif len(config.VWSNs) == 2:
    plt1 = fig.add_subplot(1, 1, 1)
    generate_plot(VN_links, shortest_p, path_n, plt1, False)
    #plt1 = fig.add_subplot(1, 3, 2)
    #generate_plot(wsn, shortest_path, path_nodes, plt1, False)
    #plt1 = fig.add_subplot(1, 3, 3)
    #generate_plot(wsn, shortest_path, [], plt1, True)
    config.plot_counter += 1
    plt.axis('on')
    plt.savefig(str(index)+"graph_" + str(config.plot_counter) + ".png")  # save as png

def evaluate_perms(current_perm):
    keys = [k for k in current_perm]
    source_nodes = []
    overall_cost = current_perm[keys[0]]['overall_cost']
    for k, v in current_perm[keys[0]]['embeddings'].items():
        source_nodes.append(k)
    if len(config.best_embeddings) != 0:
        if config.max_accepted_vnrs <= len(source_nodes):
            config.max_accepted_vnrs = len(source_nodes)
            if str(source_nodes) in config.best_embeddings:
                cost = config.best_embeddings[str(source_nodes)]['overall_cost']
                if cost > overall_cost:
                    config.best_embeddings.update({str(source_nodes): {'overall_cost': overall_cost, 'permutation': keys[0]}})
            else:
                current_key = list(config.best_embeddings.keys())
                cost = config.best_embeddings[current_key[0]]['overall_cost']
                if cost > overall_cost:
                    config.best_embeddings.pop(current_key[0],0)
                    config.best_embeddings.update(
                        {str(source_nodes): {'overall_cost': overall_cost, 'permutation': keys[0]}})
    else:
        config.best_embeddings.update({str(source_nodes): {'overall_cost': overall_cost, 'permutation': keys[0]}})
        config.max_accepted_vnrs = len(source_nodes)
    print("Optimal solution is:", config.best_embeddings)
    end = time.time()
    print(end - config.start)

def run_permutations():
    config.start = time.time()
    perms = itool.permutations(vne.get_vnrs(), r=None)
    config.best_embeddings = {}
    config.max_accepted_vnrs = 0
    for i, per in enumerate(perms):
        config.link_weights_for_this_perm = copy.deepcopy(link_weights)
        config.adjacencies_for_this_perm = copy.deepcopy(adjacencies)
        config.wsn_for_this_perm = copy.deepcopy(wsn)
        config.VWSNs = []
        config.current_emb_costs = {}
        config.overall_cost = 0
        for vnr in per:
            #config.total_operations +=  1
            embed(vnr)
        current_perm = {i: {'embeddings': config.current_emb_costs, 'overall_cost': config.overall_cost}}
        config.embedding_costs.update(current_perm)
        config.all_embeddings.append(config.VWSNs)
        print()
        evaluate_perms(current_perm)
    print("total # operation:",config.total_operations)

if __name__ == '__main__':
    link_weights = wsn_substrate.get_link_weights()
    adjacencies = wsn_substrate.get_adjacency_list()
    wsn = wsn_substrate.get_wsn_substrate()
    two_hops_list = wsn_substrate.get_two_hops_list()
    update_all_links_attributes(1, 1)
    shortest_path, path_nodes = [],[]
    display_edge_attr(wsn)
    display_node_attr(wsn)
    display_data_structs()
    config.link_weights_for_this_perm = copy.deepcopy(link_weights)
    config.adjacencies_for_this_perm = copy.deepcopy(adjacencies)
    config.wsn_for_this_perm = copy.deepcopy(wsn)
    while exit_flag is True:
        print("\n---->\n0 - Exit\n1 - Embed\n2 - Plot")
        user_input = input(': ')
        if user_input is '0':
            run_permutations()
        if user_input is '1':
            on_line_vn_request()
        elif user_input is '2':
            draw_graph()
