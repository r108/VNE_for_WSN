try:
    import matplotlib.pyplot as plt
except:
    raise

import copy
import itertools as itool
from wsn_substrate import WSN
import networkx as nx
from link_weight import LinkCost
import sp_dijkstra as sp
import config
import vne
from combinationIterator import CombinationIterator


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


def display_data_structs():
    print("Nodes - ", wsn.nodes(data=True))
    print("Edges - ", wsn.edges(data=True))
    print("Adjacency list - ", wsn.adjacency_list())
    print("adjacencies- ", adjacencies)
    print("two_hops_list - ", two_hops_list)
    print("link_weights- ", link_weights)


######################################################################################################

def map_links(e_list, e_list2, required_load):
#    print("@@",e_list2)
    for u,v in e_list2:
        update_link_attributes(int(u), int(v), -1, (e_list.count((u,v)) * required_load))
        #if (e_fr,e_to) not in config.allocated_links_weight:
        config.allocated_links_weight.update({(u,v): config.wsn_for_this_perm[u][v]['weight']})
        config.allocated_links_load.update({(u, v): config.wsn_for_this_perm[u][v]['load']})
#        print(e_fr,e_to,"occur ", e_list.count((e_fr,e_to)),"times")
    print("config.allocated_links_weight.",config.allocated_links_weight)
    print("config.allocated_links_load.", config.allocated_links_load)
    #print("get reduced adj",type(config.allocated_links_weight)

def map_links_cost(e_list, e_list2, required_load):
    link_embedding_cost = 0
    for u,v in e_list2:
        required = (e_list.count((u,v)) * required_load)
        link_embedding_cost +=required
        update_link_attributes(int(u), int(v), -1, required)
        #if (e_fr,e_to) not in config.allocated_links_weight:
        config.allocated_links_weight.update({(u,v): config.wsn_for_this_perm[u][v]['weight']})
        config.allocated_links_load.update({(u, v): config.wsn_for_this_perm[u][v]['load']})
#        print(e_fr,e_to,"occur ", e_list.count((e_fr,e_to)),"times")
    print("config.allocated_links_weight.",config.allocated_links_weight)
    print("config.allocated_links_load.", config.allocated_links_load)
    #print("get reduced adj",type(config.allocated_links_weight)
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

#def calculate_embedding_cost(VN_nodes, VN_links, required_load,e_list, e_list2, path_nodes, shortest_path):




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
    print(vn[3])
    print(vn[4])
    print(config.VWSNs)

#    str = input("commit: ")
#    if str != "":
#        pass

#    print("config.link_weights after commit= ", config.link_weights)
#    print("original link_weights after commit", link_weights)

#    display_edge_attr(wsn)
#    display_node_attr(wsn)
#    display_vn_node_allocation(VN_nodes)
#    display_vn_edge_allocation(VN_links)

def check_link_constraints(e_list, e_list2, load, required_plr, shortest_path):
    VN_links = nx.DiGraph()
#    print("check_link_constraints for load",load)
    required_load = []
    for u,v in e_list2:
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
    #print("checking nodes -----",nodes_in_path)
    #print("check_node_constraints")
    for idx,n in enumerate(nodes_in_path):
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

    path = nx.dijkstra_path(graph, source=frm, target=to, weight='weight')
    length = nx.dijkstra_path_length(graph, source=frm, target=to, weight='weight')
    print(length)
    #path = find_sp(graph, weight, frm, to)
    if (path is None) or (length > 10000000):
        return None, None
    shortest_path = []
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
            shortest_path.append((path[idx], path[idx + 1]))
#    print("Shortest path links  ", shortest_path,"\n")
    return shortest_path, path


def verify_feasibility(link_reqiurement, frm, to, node_requirement):
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
        #print("all nodes in path have enough resource")
        pass

    link_check, VN_links = check_link_constraints(e_list, e_list2, link_reqiurement['load'], link_reqiurement['plr'],shortest_path)

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
        commit(VN_nodes, VN_links, node_requirement, e_list, e_list2, path_nodes, shortest_path)



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
        if config.counter_value > 50:
            print(link_check, "do not have enough resource")
            print("EMBEDDING HAS FAILED!!")
            #return

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
#        if (1, 2) in shortest_path:
#            str = input("continue:")
#            if str != "":
#                pass
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
        two_hops_of_u = [x for x in config.two_hops.get(u) if x not in config.reduced_adj.get(u)]
        two_hops_of_v = [x for x in config.two_hops.get(v) if x not in config.reduced_adj.get(v)]

        shared_neighbors_u = list(set(path_nodes) & set(config.reduced_adj.get(u)))
        shared_neighbors_v = list(set(path_nodes) & set(config.reduced_adj.get(v)))
        common_neighbors = list(set(shared_neighbors_u) | set(shared_neighbors_v))

        for idx, n in enumerate(path_nodes):
            if (u, v) in shortest_path:
               print(u,v,"u-v link in path does not have enough resource!???????")
               if v == to:
                   if len(path_nodes)>2:
                       print("path length is 2 nodes")
                   else:
                       print("Sink node v", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                       return False
            elif (v, u) in shortest_path:
                print(v, u,"v-u link in path does not have enough resource!??????")
                config.current_wsn[v][u]['weight'] = 10000000
                config.penalized_list.append((v, u))
                if v == to:
                    if len(path_nodes) > 2:
                        print("path length is 2 nodes")
                    else:
                        print("SSink node v", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                        return False
            elif u in path_nodes:
                i = path_nodes.index(u)
                # print(u,"index is",i)
                config.current_wsn[path_nodes[i - 1]][path_nodes[i]]['weight'] = 10000000
                config.penalized_list.append((path_nodes[i - 1], path_nodes[i]))

                if u == frm:
#                   if len(path_nodes) > 2:
                    print("SSource node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                    return False

            elif v in path_nodes:
                i = path_nodes.index(v)
                config.current_wsn[path_nodes[i - 1]][path_nodes[i]]['weight'] = 10000000
                config.penalized_list.append((path_nodes[i - 1], path_nodes[i]))
                if v == frm:
 #                   if len(path_nodes) > 2:
                    print("SSource node v", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
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
    print("**", config.current_wsn)
    show_penalized_links()
    return True



def recalculate_path_weights0(frm,to,path_nodes,shortest_path):
    show_penalized_links()
    print("config.avoid", config.avoid, "\n", path_nodes, "\n", config.link_weights)
    str = input("continue:")
    if str != "":
        print("OK")

    for (u, v) in config.avoid:
        config.avoid.remove((u, v))
        if u == frm:
            print("source node", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
            return
        elif u == to:
            print("sink node", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
            return
        two_hops_of_u = [x for x in config.two_hops.get(u) if x not in config.reduced_adj.get(u)]
        two_hops_of_v = [x for x in config.two_hops.get(v) if x not in config.reduced_adj.get(v)]

        print("ccconfig.avoid after", config.avoid)
        config.link_weights[(u, v)] = 10000000
        config.penalized_list.append((u, v))


        for idx, n in enumerate(path_nodes):
            if (u, v) in shortest_path:
               print(u,v,"link in path does not have enough resource!")
               if v == to:
                   if len(path_nodes)>2:
                       print("path length is 2 nodes")
                   else:
                       print("sink node", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                       return
            elif n == u:
                for in_n in config.reduced_adj.get(u):
                    config.link_weights[(in_n, u)] = 10000000
                    config.penalized_list.append((in_n, u))
            elif n == v:
                for in_n in config.reduced_adj.get(v):
                    config.link_weights[(in_n, v)] = 10000000
                    config.penalized_list.append((in_n,v))
                if v == to:
                    if len(path_nodes) > 2:
                        shared_neighbor = list(set(config.reduced_adj.get(to))&set(config.reduced_adj.get(path_nodes[idx-2])))
                    if shared_neighbor:
                        config.link_weights[(path_nodes[idx-2], path_nodes[idx-1])] = 10000000
                        config.penalized_list.append((path_nodes[idx-2], path_nodes[idx-1]))
                        for sn in shared_neighbor:
                            config.link_weights[(path_nodes[idx - 2], sn)] = 10000000
                            config.penalized_list.append((path_nodes[idx - 2], sn))
                    else:
                        print("sink node", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                        return
            elif n in two_hops_of_u:
                if n == frm:
                    print("two hop neighbor",u,"of source node",frm," does not have enough resource.\nEMBEDDING HAS FAILED!")
                    return
                if n == to:
                    print("two hop neighbor",u,"of sink node",to," does not have enough resource.\nEMBEDDING HAS FAILED!")
                    return
                config.link_weights[(n, path_nodes[idx+1])] = 10000000
                config.penalized_list.append((n, path_nodes[idx+1]))

                for in_n in config.reduced_adj.get(u):
                    config.link_weights[(in_n, u)] = 10000000
                    config.penalized_list.append((in_n,u))

                shared_neighbor = list(set(config.reduced_adj.get(n)) & set(config.reduced_adj.get(u)))
                for sn in shared_neighbor:
                    config.link_weights[(n, sn)] = 10000000
                    config.penalized_list.append((n,sn))

            elif n in two_hops_of_v:
                if n == frm:
                    print("two hop neighbor", v, "of source node", frm,
                          " does not have enough resource.\nEMBEDDING HAS FAILED!")
                    return

                if n == to:
                    print("sink node", n, "does not have enough resource.")
                    str = input("continue:")
                    if str != "":
                        continue
                else:
                    config.link_weights[(n, path_nodes[idx + 1])] = 10000000
                    config.penalized_list.append((n,path_nodes[idx + 1]))

                for in_n in config.reduced_adj.get(v):
                    config.link_weights[(in_n, v)] = 10000000
                    config.penalized_list.append((in_n,v))

                shared_neighbor = list(set(config.reduced_adj.get(n)) & set(config.reduced_adj.get(v)))
                for sn in shared_neighbor:
                    config.link_weights[(n, sn)] = 10000000
                    config.penalized_list.append((n,sn))
    print("**",config.link_weights)



def recalculate_path_weights1(frm,to,path_nodes,shortest_path):
    for (u,v) in config.avoid:
        config.avoid.remove((u, v))
        print("ccconfig.avoid after", config.avoid)
        config.link_weights[(u, v)] = 10000000

        if (u, v) in shortest_path:
            print(u, v, "is in path", shortest_path)
            adjacent_path_nodes = list(set(config.reduced_adj.get(u)) & set(path_nodes))
            print(u,"u adjacent_path_nodes",adjacent_path_nodes)
            for apn in adjacent_path_nodes:
                config.link_weights[(apn, u)] = 10000000
            adjacent_path_nodes = list(set(config.reduced_adj.get(v)) & set(path_nodes))
            print(v, "v adjacent_path_nodes", adjacent_path_nodes)
            for apn in adjacent_path_nodes:
                config.link_weights[(apn, v)] = 10000000

        elif (u in path_nodes):
            adjacent_path_nodes = list(set(config.reduced_adj.get(u)) & set(path_nodes))
            for apn in adjacent_path_nodes:
                config.link_weights[(apn, u)] = 10000000
        elif (v in path_nodes):
            adjacent_path_nodes = list(set(config.reduced_adj.get(v)) & set(path_nodes))
            for apn in adjacent_path_nodes:
                config.link_weights[(apn, v)] = 10000000
        else:


            two_hop_neighbors = [x for x in config.two_hops.get(u) if x not in config.reduced_adj.get(u)]
            adjacent_path_nodes = list(set(config.reduced_adj.get(u)) & set(path_nodes))
            adjacent_nodes = config.reduced_adj.get(u)

            if (set(two_hop_neighbors) & set(path_nodes)):
                for thn in two_hop_neighbors:
                    for an in config.reduced_adj.get(u):
                        if (thn,an) in config.link_weights:
                            config.link_weights[(thn, an)] = 10000000

            two_hop_neighbors = [x for x in config.two_hops.get(v) if x not in config.reduced_adj.get(v)]
            if (set(two_hop_neighbors) & set(path_nodes)):
                for thn in two_hop_neighbors:
                    for an in config.reduced_adj.get(v):
                        if (thn, an) in config.link_weights:
                            config.link_weights[(thn, an)] = 10000000



def recalculate_path_weights2(link_check,path_nodes,shortest_path):

    #e_list, e_list2 = get_conflicting_links(path_nodes)

    for (u,v) in config.avoid:
        config.avoid.remove((u, v))

        print("ccconfig.avoid after", config.avoid)
        config.link_weights[(u, v)] = 10000000
        if (u,v) in shortest_path:
            print(u, v, "is in path")
            for n in config.reduced_adj.get(u):
                config.link_weights[(n, u)] = 10000000
            for n in config.reduced_adj.get(v):
                config.link_weights[(n, v)] = 10000000
        elif (u in path_nodes):
            print(u,"is in path")
            for n in config.reduced_adj.get(u):
                config.link_weights[(n, u)] = 10000000
            for idx, n in enumerate(path_nodes):
                shared_neighbors = list(set(config.reduced_adj.get(v)) & set(config.reduced_adj.get(n)))
                for sn in shared_neighbors:
                    config.link_weights[(sn, v)] = 10000000
                    config.link_weights[(n, sn)] = 10000000
        elif (v in path_nodes):
            print(v,"is in path")
            for n in config.reduced_adj.get(v):
                config.link_weights[(n, v)] = 10000000
            for idx, n in enumerate(path_nodes):
                shared_neighbors = list(set(config.reduced_adj.get(u)) & set(config.reduced_adj.get(n)))
                for sn in shared_neighbors:
                    config.link_weights[(sn, u)] = 10000000
                    config.link_weights[(n, sn)] = 10000000
        else:
            for idx, n in enumerate(path_nodes):
                if n in config.two_hops.get(u):
                    print(n,"is a 2 hops neghbor of",u)
                    if n not in config.reduced_adj.get(u):
                        if idx < len(path_nodes)-1:
                            print("and", n, u, " not neighbors")
                            print(n, u, v, "in u", )
                            config.link_weights[(n, path_nodes[idx+1])] = 10000000
                    else:
                        print("and", n, u, " neighbors")
                        config.link_weights[(n, u)] = 10000000
                    shared_neighbors = list(set(config.reduced_adj.get(u)) & set(config.reduced_adj.get(n)))
                    for sn in shared_neighbors:
                        config.link_weights[(sn, u)] = 10000000
                        config.link_weights[(n, sn)] = 10000000
                elif n in config.two_hops.get(v):
                    print(n, "is a 2 hops neghbor of", v)
                    if n not in config.reduced_adj.get(v):
                        if idx < len(path_nodes)-1:
                            print("and",n,v," not neighbors")
                            print(n,u,v,"in v",)
                            config.link_weights[(n, path_nodes[idx+1])] = 10000000
                    else:
                        print("and", n, v, " neighbors")
                        config.link_weights[(n, v)] = 10000000
                    shared_neighbors = list(set(config.reduced_adj.get(v)) & set(config.reduced_adj.get(n)))
                    for sn in shared_neighbors:
                        config.link_weights[(sn, v)] = 10000000
                        config.link_weights[(n, sn)] = 10000000







def embed_vn(VN):
    config.current_wsn = copy.deepcopy(config.wsn_for_this_perm)
    config.reduced_adj = copy.deepcopy(config.adjacencies_for_this_perm)
    config.link_weights = copy.deepcopy(config.link_weights_for_this_perm)
    config.two_hops = copy.deepcopy(two_hops_list)
    config.penalized_list = []

#    config.current_wsn = copy.deepcopy(wsn)
#    config.reduced_adj = copy.deepcopy(wsn_substrate.get_adjacency_list())
#    config.link_weights = copy.deepcopy(link_weights)
#    config.two_hops = copy.deepcopy(two_hops_list)

    print("VN embedding: ", VN)
    print("@links->config.link_weights",config.link_weights)


    vwsn_nodes = VN[1]
    link_reqiurement = VN[2]

    frm = list(vwsn_nodes)[0]
    to = list(vwsn_nodes)[2]

    node_requirement = vwsn_nodes[1]['load']
    config.avoid = []

    #paths = nx.all_shortest_paths(wsn, source=frm, target=to, weight='weight')
#    length, paths = nx.single_source_dijkstra(wsn,1,weight='weight')
#    print (length)
#    print(paths)
#    for i,p in enumerate(paths):
#        print(",,,",p)

    verify(link_reqiurement, frm, to, node_requirement)

#    paths = nx.all_shortest_paths(wsn, source=frm, target=to, weight='weight')
#    for p in paths:
#        print(p)






def get_conflicting_links_old(path_nodes):
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
#                    print(n, "->-> visit nn", nn)
                    if nn not in visited_nodes:
#                        print("add if nn not in []", visited_nodes)
                        effected_edges.append((n, nn))
                visited_nodes.append(n)
#                print(n, "appended to visited nodes in rx", visited_nodes)

    effected_edges_set = list(set(effected_edges))
#    print("return effected_edges",effected_edges)
#    print("return effected_edges_set",effected_edges_set)

    return effected_edges, effected_edges_set

def embed(vnr):

    config.current_wsn = copy.deepcopy(config.wsn_for_this_perm)
    config.reduced_adj = copy.deepcopy(config.adjacencies_for_this_perm)
    config.link_weights = copy.deepcopy(config.link_weights_for_this_perm)
    config.two_hops = copy.deepcopy(two_hops_list)
    config.penalized_list = []


#    str = input("embed(vnr):")
#    if str != "":
#        pass

    print("VNR embedding: ", vnr)
    print("@links->config.link_weights", config.link_weights)

    vwsn_nodes = vnr[1]
    link_reqiurement = vnr[2]

    frm = list(vwsn_nodes)[0]
    to = list(vwsn_nodes)[2]

    node_requirement = vwsn_nodes[1]['load']
    config.avoid = []
    verify_feasibility(link_reqiurement, frm, to, node_requirement)


def get_conflicting_links2(path_nodes):
    tx_nodes = copy.deepcopy(path_nodes)
    tx_nodes.pop()
#    print("tx_nodes",tx_nodes,"\npath_nodes",path_nodes)
    effected_edges = []
    print("initialize effected_edges",effected_edges)

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
###            for nn in config.reduced_adj.get(n):
###                print(n, "->-> visit nn", nn)
###                if nn not in visited_nodes:
###                    print("add if nn not in []", visited_nodes)
###                    effected_edges.append((n, nn))
###                    effected_edges.append((nn, n))


#            visited_nodes.append(n)
#            print(n, "appended to visited nodes in tx", visited_nodes)
#            print(i," in length",len(path_nodes))

        rx = path_nodes[i+1]
#        print(i, "visit rx", rx)
        for n in config.reduced_adj.get(rx):
#            print(rx, "-> visit n", n)
            if n not in visited_nodes:
                effected_edges.append((rx, n))
                effected_edges.append((n, rx))
                for nn in config.reduced_adj.get(n):
#                    print(n, "->-> visit nn", nn)
                    if nn not in visited_nodes:
#                        print("add if nn not in []", visited_nodes)
                        effected_edges.append((n, nn))
                visited_nodes.append(n)
#                print(n, "appended to visited nodes in rx", visited_nodes)

    effected_edges_set = list(set(effected_edges))
    print("return effected_edges",effected_edges)
    print("return effected_edges_set",effected_edges_set)

    return effected_edges, effected_edges_set

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

def evaluate_perms():
    for i, embedings in enumerate(config.all_embeddings):
        overall_cost = 0
        config.current_mappings = {}
        for idx, vn in enumerate(embedings):
            config.current_mappings.update({vn[3][0]: vn[4]})
            overall_cost += vn[4]
        config.successful_embeddings.update({i: {'embeddings': config.current_mappings, 'overall_cost': overall_cost}})
        #print("config.successful_embeddings",config.successful_embeddings)
        combos = []
        config.best_embeddings = {}
        optimal_perm = {}
        max_accepted_vn = 0
        for key, val in config.successful_embeddings.items():
            source_nodes = []
            overall_cost = val['overall_cost']

            for k, v in val['embeddings'].items():
                source_nodes.append(k)

            if len(config.best_embeddings) != 0:
                if max_accepted_vn <= len(source_nodes):
                    max_accepted_vn = len(source_nodes)
                    print(source_nodes)
                    if str(source_nodes) in config.best_embeddings:
                        cost = config.best_embeddings[str(source_nodes)]['overall_cost']
                        if cost > overall_cost:
                            config.best_embeddings.update({str(source_nodes): {'overall_cost': overall_cost, 'permutation': key}})
                            last_index = config.best_embeddings[str(source_nodes)]['permutation']
                    else:
                        current_key = list(config.best_embeddings.keys())
                        print("current_key",current_key[0])
                        cost = config.best_embeddings[current_key[0]]['overall_cost']
                        if cost > overall_cost:
                            config.best_embeddings.pop(current_key[0],0)
                            config.best_embeddings.update(
                                {str(source_nodes): {'overall_cost': overall_cost, 'permutation': key}})
                            if source_nodes not in combos:
                                combos.append(source_nodes)
            else:
                config.best_embeddings.update({str(source_nodes): {'overall_cost': overall_cost, 'permutation': key}})
                max_accepted_vn = len(source_nodes)

    print("combos",combos)
    print("emb", config.best_embeddings)

def run_permutations():
    perms = itool.permutations(vne.get_vnrs(), r=None)
    for i, per in enumerate(perms):
        print("permutation", i, " is ", per)
        #                str = input(":")
        #                if str != "":
        #                    pass
        config.link_weights_for_this_perm = copy.deepcopy(link_weights)
        config.adjacencies_for_this_perm = copy.deepcopy(adjacencies)
        config.wsn_for_this_perm = copy.deepcopy(wsn)
        config.VWSNs = []
        for vnr in per:
            print(per.index(vnr), vnr)
            embed(vnr)
        vn_embeddings_for_this_perm = config.VWSNs
        config.all_embeddings.append(vn_embeddings_for_this_perm)

        #              for idx, vwsn in enumerate(vn_embeddings_for_this_perm):
        #                  print("vwsn", idx, "path", vwsn[3])
        #                  print("vwsn", idx, "nodes", vwsn[0].nodes(data=True))
        #                  print("vwsn", idx, "links", vwsn[1].edges(data=True))
        print()
    # print("**config.all_embeddings length", len(config.all_embeddings))
    evaluate_perms()


if __name__ == '__main__':
    link_weights = wsn_substrate.get_link_weights()
    adjacencies = wsn_substrate.get_adjacency_list()
    wsn = wsn_substrate.get_wsn_substrate()
    two_hops_list = wsn_substrate.get_two_hops_list()
    update_all_links_attributes(1, 1)
    shortest_path, path_nodes = [],[] #sp.display_shortest_path(get_shortest_path(adj,links,1,6))
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
            #display_edge_attr(wsn)
            #display_node_attr(wsn)


                    #print("vwsn", idx, "path", vwsn[3],"cost",vwsn[4])
                    #print("vwsn", idx, "nodes", vwsn[0].nodes(data=True))
                    #print("vwsn", idx, "links", vwsn[1].edges(data=True))
        #            print("VWSN ", idx, "allocations:")
        #            display_vn_edge_allocation(vwsn[1])
        #            display_vn_node_allocation(vwsn[0])
        #            show_graph_plot(vwsn[1], vwsn[2], vwsn[3])
                    #print("wsn.get_wsn_links()", vwsn[1].edges(data=True), "\n shortest_path", vwsn[2],"\n path_nodes", vwsn[3])


            for k,v in config.best_embeddings.items():
                index = v['permutation']
                print("index",index)
                print("config.all_embeddings[index]",config.all_embeddings[index])

                for perm in config.all_embeddings[index]:
                    print("perm",perm)
                    VN_links = perm[1]
                    shortest_p = perm[2]
                    path_n = perm[3]
                    plotit(VN_links, shortest_p, path_n, index)

