

import copy
import itertools as itool
from wsn_substrate import WSN
import networkx as nx
from link_weight import LinkCost
import config
import vne
import time
import visualize as vis
from itertools import islice
import cProfile
import re

wsn_substrate = WSN()
exit_flag = True

def display_data_structs():
    print("Nodes - ", config.wsn.nodes(data=True))
    print("Edges - ", config.wsn.edges(data=True))
    print("Adjacency list - ", config.wsn.adjacency_list())
    print("adjacencies- ", adjacencies)
    print("two_hops_list - ", two_hops_list)
    print("link_weights- ", link_weights)

def update_all_links_attributes(wsn,plr, load):
    for u,v,d  in config.wsn.edges_iter(data=True):
        wsn[u][v]['plr'] = plr
        wsn[u][v]['load'] = load
        link_weight = LinkCost(wsn[u][v]['plr'], wsn[u][v]['load'])
        wsn[u][v]['weight'] = link_weight.get_weight(link_weight)
        link_weights[(u, v)] = link_weight.get_weight(link_weight)
    #print(config.wsn.edges(data=True))
    #print("")

def update_node_attribs(nodes, node, load):
    for n, d in nodes.nodes_iter(data=True):
        if n == int(node):
            d['load'] = d['load']+int(load)
            #d['rank'] = len(adj[n])

def update_link_attribs(wsn,u, v, plr, load):
    current_link_weight = LinkCost(wsn[u][v]['plr'], wsn[u][v]['load'])
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
    return current_link_weight.get_weight(current_link_weight)

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
        vnr = (1000, VWSN_nodes, link_reqiurement)
        config.online_flag = True
        embed(vnr)
    else:
        return

def map_links_cost(e_list, e_set, required_load):
    link_embedding_cost = 0
    current_weight = 0
    for u,v in e_set:
        required = (e_list.count((u,v)) * required_load)
        if config.online_flag:
            current_weight = update_link_attribs(config.committed_wsn, int(u), int(v), -1, required)
        else:
            current_weight = update_link_attribs(config.wsn_for_this_perm, int(u), int(v), -1, required)
        link_embedding_cost +=(required * current_weight) #weighted cost
#        link_embedding_cost += (required)
    return link_embedding_cost

def map_nodes_cost(all_path_nodes, required_load):
    node_embedding_cost = 0
    for idx, pn in enumerate(all_path_nodes):
        if config.online_flag:
            update_node_attribs(config.committed_wsn, pn, required_load)
        else:
            update_node_attribs(config.wsn_for_this_perm, pn, required_load)
        node_embedding_cost += (required_load)
    return  node_embedding_cost

def commit(VN_nodes, VN_links, required_load,e_list, e_list2, path_nodes, shortest_path):
    n_cost = map_nodes_cost(VN_nodes.nodes(), required_load)
    l_cost = map_links_cost(e_list, e_list2, required_load)
    cost = (n_cost+l_cost)
    vn = (VN_nodes, VN_links, shortest_path, path_nodes, cost, path_nodes[0])
    config.VWSNs.append(vn)
    if config.online_flag:
        vis.display_edge_attr(config.committed_wsn)
        vis.display_node_attr(config.committed_wsn)
        vis.display_vn_node_allocation(VN_nodes)
        vis.display_vn_edge_allocation(VN_links)
        vis.plotit(VN_links, shortest_path, path_nodes, 0)
        config.active_vns.append(vn)

    config.current_emb_costs.update({path_nodes[0]: cost})
    config.overall_cost += cost

#    #print(vn[3])
#    #print(vn[4])
#    #print(config.VWSNs)
#    #print("config.current_emb_costs", config.current_emb_costs)

def check_link_constraints(e_list, e_list2, load, required_plr, wsn):
    VN_links = nx.DiGraph()
    for u,v in e_list2:
        config.total_operations +=  1
        required_load = load * e_list.count((u,v))
        if wsn.edge[u][v]['load'] + required_load > 100:
 #           #print("Link",u, v,"requires",wsn.edge[u][v]['load']," + ",required_load, "but have not got enough")
            return (u,v),VN_links
        else:
            VN_links.add_edge(u,v, {'load':required_load})
            return_value = (0,0)
    #print("RETURN LINK ",return_value)
    return return_value,VN_links

def check_node_constraints(nodes_in_path, required_load, wsn):
    VN_nodes = nx.DiGraph()
    for idx,n in enumerate(nodes_in_path):
        config.total_operations +=  1
        VN_nodes.add_node(n, {'load': required_load})
        if wsn.node[n]['load'] + required_load > 100:
            if idx == 0:
                    ##print("Source node",n," has - ",wsn.node[n]['load'],"but require",+ required_load )
                    return n, VN_nodes
            elif idx == (len(nodes_in_path) - 1):
                    ##print("Sink node",n,"has - ",wsn.node[n]['load'],"but require",+ required_load )
                    return n, VN_nodes
            else:
                    ##print("Relay node",n,"has - ",wsn.node[n]['load'],"but require",+ required_load )
                    return n, VN_nodes
    return 0, VN_nodes

def get_shortest_path(graph, frm, to):
    #modified the below file to return 2 parameters (path, length) instead of 1 path
    #/home/roland/anaconda3/lib/python3.5/site-packages/networkx/algorithms/shortest_paths/weighted.py
#    path, length = nx.dijkstra_path(graph, source=frm, target=to, weight='weight')
    length,path = nx.bidirectional_dijkstra(graph, source=frm, target=to, weight='weight')
#    length = nx.dijkstra_path_length(graph, source=frm, target=to, weight='weight')
    #print('Shortest path weight is ',length)
    if (path is None) or (length >= 10000000):
        return None, None
    s_path = []
#    print('Shortest path is ', end="")
    for idx, p in enumerate(path):
        if idx == 0:
            shortest_path.append((p, p+1))
#            print(vis.CGREEN, p, "->", end="")
        elif idx == (len(path) - 1):
#            print(p, vis.CEND)
            pass
        else:
#            print(p, "->", end="")
            pass
        if idx != len(path) - 1:
            s_path.append((path[idx], path[idx + 1]))
#    #print("Shortest path links  ", shortest_path,"\n")
    return s_path, path

def get_max_edge_load(wsn,node, is_source):
    max_load = 0
    for n in config.reduced_adj[node - 1]:
        link_load = wsn[node][n]['load']
        if link_load > max_load:
            max_load = link_load
        if not is_source:
            link_load = wsn[n][node]['load']
            if link_load > max_load:
                max_load = link_load
    return max_load



def verify_feasibility(link_reqiurement, frm, to, node_requirement):
    config.total_operations +=  1
    config.counter_value = config.counter_value+1
    hops = len(nx.shortest_path(config.wsn, source=frm, target=to)) - 1
    max_load = 0

    if config.online_flag:
        node_check, VN_nodes = check_node_constraints([frm, to], node_requirement, config.committed_wsn)
        max_load = max([get_max_edge_load(config.committed_wsn, frm, True), get_max_edge_load(config.committed_wsn, to, False)])
    else:
        node_check, VN_nodes = check_node_constraints([frm, to], node_requirement, config.wsn_for_this_perm)
        max_load = max([get_max_edge_load(config.wsn_for_this_perm, frm, True),get_max_edge_load(config.wsn_for_this_perm, to, False)])

    if node_check != 0:
        #print("node ", node_check, "does not have enough resource\nEMBEDDING FAILED!")
        return False

    if(hops > 2):
        if (link_reqiurement['load']*2) > (100 - max_load):
            #print("Failed!",frm,to," cannot support request")
            return False

    #print("verify feasibility----------------------",config.counter_value,"counter ")
    shortest_path, path_nodes = get_shortest_path(config.current_wsn, frm, to)
    if shortest_path is None:
        #print("No feasible path!\nEMBEDDING HAS FAILED!")
        return False
    e_list, e_set = get_conflicting_links(path_nodes)
#   e_list, e_set =get_conflictng_edges(path_nodes)
    # get list of unique nodes from conflicting link list
    effected_nodes = []
    for u, v in e_set:
        if u not in effected_nodes:
            effected_nodes.append(u)
        if v not in effected_nodes:
            effected_nodes.append(v)
    if config.online_flag:
        node_check, VN_nodes = check_node_constraints(effected_nodes, node_requirement, config.committed_wsn)
    else:
        node_check, VN_nodes = check_node_constraints(effected_nodes, node_requirement, config.wsn_for_this_perm)
    if node_check != 0:
        #print("node ", node_check, "does not have enough resource\nEMBEDDING FAILED!")
        return False
    elif config.online_flag:
        link_check, VN_links = check_link_constraints(e_list, e_set, link_reqiurement['load'], link_reqiurement['plr'],config.committed_wsn)
    else:
        link_check, VN_links = check_link_constraints(e_list, e_set, link_reqiurement['load'], link_reqiurement['plr'],config.wsn_for_this_perm)
    if link_check == (0,0):
        #print("++SUCCESSFUL EMBEDDING++")
        config.has_embedding = True
        commit(VN_nodes, VN_links, node_requirement, e_list, e_set, path_nodes, shortest_path)
        config.feasible = True
#        print("commit-",path_nodes[0])
        return False
    else:
        #print("link_check is", link_check)
        if link_check not in config.avoid:
            config.avoid.append(link_check)
        if link_check not in config.failed_links_list:
            config.failed_links_list.append(link_check)
        if recalculate_path_weights(frm, to, path_nodes, shortest_path) == True:
            #print("Something went wrong!", config.recursion_counter)
            return False
    check_again(link_reqiurement, frm, to, node_requirement)
        #verify_feasibility(link_reqiurement, frm, to, node_requirement)

def check_again(link_reqiurement, frm, to, node_requirement):
    if config.feasible == False:
        is_failed = verify_feasibility(link_reqiurement, frm, to, node_requirement)
        if is_failed:
            print("Verify failed!!! ")
            pass

##This is simpler but very inefficient
def get_conflictng_edges(path_nodes):
    e_list2 = []
    e_set2 = []
    for i, tx in enumerate(path_nodes):
#       config.total_operations += 1
        if i < (len(path_nodes) - 1):
            del e_set2
            # get all edges of Tx
            e_set2 = wsn_substrate.get_wsn_substrate().in_edges(tx)
            e_set2.extend(wsn_substrate.get_wsn_substrate().out_edges(tx))
            # get all edges for all neighbors of Tx
            for txn in nx.neighbors(wsn_substrate.get_wsn_substrate(), tx):
#               config.total_operations += 1
                e_set2.extend(wsn_substrate.get_wsn_substrate().in_edges(txn))
                e_set2.extend(wsn_substrate.get_wsn_substrate().out_edges(txn))
            # get all out edges of Rx
            for rxn in nx.neighbors(wsn_substrate.get_wsn_substrate(), path_nodes[i + 1]):
 #              config.total_operations += 1
                e_set2.extend(wsn_substrate.get_wsn_substrate().out_edges(rxn))
            e_set2 = list(set(e_set2))
            e_list2.extend(e_set2)
    e_set2 = list(set(e_list2))
    return e_list2, e_set2

def get_conflicting_links(path_nodes):
    tx_nodes = copy.deepcopy(path_nodes)
    tx_nodes.pop()
#    #print("tx_nodes",tx_nodes,"\npath_nodes",path_nodes)
    effected_edges = []
#    #print("initialize effected_edges",effected_edges)
    for i,tx in enumerate(tx_nodes):
        visited_nodes = []
#        #print(i,"visit tx", tx)
        visited_nodes.append(tx)
#        #print(tx,"appended to visited nodes", visited_nodes)
        for n in config.reduced_adj[tx-1]:
#            #print(tx,"-> visit n", n)
            if n not in visited_nodes:
#                #print("add if n not in []",visited_nodes)
                effected_edges.append((tx, n))
                effected_edges.append((n, tx))
            for nn in config.reduced_adj[n-1]:
                config.total_operations += 1
#                #print(n, "->-> visit nn", nn)
                if nn not in visited_nodes:
#                    #print("add if nn not in []", visited_nodes)
                    effected_edges.append((n, nn))
                    effected_edges.append((nn, n))
            visited_nodes.append(n)
#            #print(n, "appended to visited nodes in tx", visited_nodes)
#            #print(i," in length",len(path_nodes))
        rx = path_nodes[i+1]
#        #print(i, "visit rx", rx)
        for n in config.reduced_adj[rx-1]:
#            #print(rx, "-> visit n", n)
            if n not in visited_nodes:
                for nn in config.reduced_adj[n-1]:
                    config.total_operations += 1
#                    #print(n, "->-> visit nn", nn)
                    if nn not in visited_nodes:
#                        #print("add if nn not in []", visited_nodes)
                        effected_edges.append((n, nn))
                visited_nodes.append(n)
#                #print(n, "appended to visited nodes in rx", visited_nodes)
    effected_edges_set = list(set(effected_edges))
    return effected_edges, effected_edges_set

def embed(vnr):
    #print("BEGIN VNR EMBEDDING", vnr)
    del config.two_hops
    config.two_hops = copy.deepcopy(two_hops_list)
    del config.penalized_list
    config.penalized_list = []
    vwsn_nodes = vnr[1]
    link_reqiurement = vnr[2]
    frm = list(vwsn_nodes)[0]
    to = list(vwsn_nodes)[2]
    node_requirement = vwsn_nodes[1]['load']
    del config.avoid
    config.avoid = []
    config.feasible = False
    if config.online_flag:
        config.VWSNs = []
        config.current_emb_costs = {}
        if config.has_embedding == False:
            config.committed_wsn = copy.deepcopy(config.wsn)
        config.current_wsn = copy.deepcopy(config.committed_wsn)
        config.reduced_adj = copy.deepcopy(config.committed_wsn.adjacency_list())
    else:
        #print(config.current_wsn.edges(data=True))
        #print(config.wsn_for_this_perm.edges(data=True))
        del config.current_wsn
        config.current_wsn = copy.deepcopy(config.wsn_for_this_perm)
        #print(config.current_wsn.edges(data=True))
        #print(config.wsn_for_this_perm.edges(data=True))
        del config.reduced_adj
        config.reduced_adj = copy.deepcopy(config.adjacencies_for_this_perm)
    verify_feasibility(link_reqiurement, frm, to, node_requirement)

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
                if cost >= overall_cost:
                    config.best_embeddings.update({str(source_nodes): {'overall_cost': overall_cost, 'permutation': keys[0]}})
                    del config.committed_wsn
                    config.committed_wsn = copy.deepcopy(config.wsn_for_this_perm)
                    del config.active_vns
                    config.active_vns = copy.deepcopy(config.VWSNs)
            else:
                current_key = list(config.best_embeddings.keys())
                cost = config.best_embeddings[current_key[0]]['overall_cost']
                if cost >= overall_cost:
                    config.best_embeddings.pop(current_key[0],0)
                    config.best_embeddings.update(
                        {str(source_nodes): {'overall_cost': overall_cost, 'permutation': keys[0]}})
                    del config.committed_wsn
                    config.committed_wsn = copy.deepcopy(config.wsn_for_this_perm)
                    del config.active_vns
                    config.active_vns = copy.deepcopy(config.VWSNs)
    else:
        config.best_embeddings.update({str(source_nodes): {'overall_cost': overall_cost, 'permutation': keys[0]}})
        config.max_accepted_vnrs = len(source_nodes)
        del config.committed_wsn
        config.committed_wsn = copy.deepcopy(config.wsn_for_this_perm)
        del config.active_vns
        config.active_vns = copy.deepcopy(config.VWSNs)

def run_permutations():
    config.online_flag = False
    config.start = time.time()
    perms = itool.permutations(vne.get_vnrs(), r=None)
    config.best_embeddings = {}
    config.max_accepted_vnrs = 0
    vns_per_perm = []
    for i, per in enumerate(perms):
        #print("config.recursion_counter",config.recursion_counter)
        config.recursion_counter = 0
        #print("Permutation ",i)
        #if i == 29760 or i == 29761:
        #    user_input = input('perm 29760-1: ')
        #    if user_input is '':
        #        pass
        #config.link_weights_for_this_perm = copy.deepcopy(link_weights)
        del config.adjacencies_for_this_perm
        config.adjacencies_for_this_perm = copy.deepcopy(adjacencies)
        del config.wsn_for_this_perm
        config.wsn_for_this_perm = copy.deepcopy(config.wsn)
        del config.VWSNs
        config.VWSNs = []
        config.current_emb_costs = {}
        del config.overall_cost
        config.overall_cost = 0
        del vns_per_perm
        vns_per_perm = []
        config.feasible = False
        for idx, vnr in enumerate(per):
            config.total_operations +=  1
            embed(vnr)
#            print("--", (idx, list(vnr[1])[0]))
            vns_per_perm.append({idx: (list(vnr[1])[0], config.feasible)})
            config.feasible = False
        config.perms_list.extend((i,vns_per_perm))
#        print("---",(i,vns_per_perm))
        current_perm = {i: {'embeddings': config.current_emb_costs, 'overall_cost': config.overall_cost}}
#        config.embedding_costs.update(current_perm)
#        config.all_embeddings.append(config.VWSNs)
        evaluate_perms(current_perm)

    vis.display_edge_attr(config.committed_wsn)
    vis.display_node_attr(config.committed_wsn)
#    display_data_structs()
    #print()
#    #print("All feasible embddings:", config.all_embeddings)
#    #print("All embedding costs:", config.embedding_costs)
    print("Optimal solution is:", config.best_embeddings)
    end = time.time()
    print(end - config.start)
    print("total # operation:", config.total_operations)

def show_penalized_links():
    #print("&&&",config.avoid)
    config.penalized_list = list(set(config.penalized_list))
    for u, v in config.penalized_list:
        #print(u,v,config.current_wsn[u][v]['weight'])
        pass

def recalculate_path_weights(frm,to,path_n,shortest_path):
    if config.recursion_counter > 100:
        user_input = input('recalculate: ')
        if user_input is '':
            return True
    for (u, v) in config.avoid:
        #print("recalculate",u,v)
        path_nodes = copy.deepcopy(path_n)
        path_n.reverse()
#        #print(path_nodes)
#        #print(path_n)
        config.total_operations += 1
        config.avoid.remove((u, v))
        config.current_wsn[u][v]['weight'] = 10000000 #penalize link
        config.penalized_list.append((u, v))
#        two_hops_of_frm = [x for x in config.two_hops.get(path_nodes[0]) if x not in config.reduced_adj[frm-1]]
#        two_hops_of_u = [x for x in config.two_hops.get(u) if x not in config.reduced_adj[u-1]]
#        two_hops_of_v = [x for x in config.two_hops.get(v) if x not in config.reduced_adj[v-1]]
#        shared_neighbors_u = list(set(path_nodes) & set(config.reduced_adj[u - 1]))
#        shared_neighbors_v = list(set(path_nodes) & set(config.reduced_adj[v-1]))
#        shared_neighbors_frm = list(set(path_nodes) & set(config.reduced_adj[frm-1]))
#        shared_neighbors_to = list(set(path_nodes) & set(config.reduced_adj[to-1]))
#        common_neighbors = list(set(shared_neighbors_u) | set(shared_neighbors_v))

        if len(path_nodes) == 2:
            #print("Source node", frm, "does not have enough resource.\nEMBEDDING HAS FAILED!")
            for n in config.reduced_adj[frm - 1]:
                config.current_wsn[frm][n]['weight'] = 10000000  # make path cost unfeasible
            return False
        ##changed to elif
        elif (v, u) in shortest_path:
            #print(v, u, "v-u link in path does not have enough resource!")
            #print(config.current_wsn[v][u]['weight'])
            config.current_wsn[v][u]['weight'] = 10000000
            return False
        elif (u, v) in shortest_path:
            #print(u, v, "u-v link in path does not have enough resource!")
            config.current_wsn[u][v]['weight'] = 10000000
            #           config.current_wsn[shortest_path[shortest_path.index(u)-1]][u]['weight'] = 10000000  #about this I'm not sure!!
            return False
        ##changed to elif
        elif u == frm:
            if (len(path_nodes) <= 3) or (v != path_nodes[1]):
                #print("Source node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                for n in config.reduced_adj[frm-1]:
                    config.current_wsn[frm][n]['weight'] = 10000000 #make path cost unfeasible
            elif v == path_nodes[1]:
                for n in path_n:
                    if n in config.reduced_adj[frm-1]:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                        #print("Source node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                        return False
            else:
                #print("Source node u", u, "does not have enough resource. This case has not been handled yet!\nEMBEDDING HAS FAILED!")
                user_input = input(': ')
                if user_input is '':
                    return False
            return False
        elif v == frm:
            if (len(path_nodes) <= 3) or (u != path_nodes[1]):
                #print("Source node v", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                for n in config.reduced_adj[frm-1]:
                    config.current_wsn[frm][n]['weight'] = 10000000 #make path cost unfeasible
            elif u == path_nodes[1]:
                for n in path_n:
                    if n in config.reduced_adj[frm-1]:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                        #print("Source node v", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                        return False
            else:
                #print("Source node v", v, "does not have enough resource. This case has not been handled yet!\nEMBEDDING HAS FAILED!")
                user_input = input(': ')
                if user_input is '':
                    return False
            return False
        elif u == to:
            #print("Sink node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
            for n in config.reduced_adj[frm - 1]:
                config.current_wsn[frm][n]['weight'] = 10000000  # make path cost unfeasible
            return False
        elif v == to:
            if (u != path_n[1]) and (path_n[2] not in config.reduced_adj[u-1]):
                #print("Sink node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                for n in config.reduced_adj[to-1]:
                    config.current_wsn[frm][n]['weight'] = 10000000 #make path cost unfeasible
            elif u == path_n[1]:
                #print(u, v, "u-v link is in path! Penalize link (predecessor of u)->u!")
                config.current_wsn[path_n[2]][u]['weight'] = 10000000 #make path cost unfeasible
            elif path_n[2] in config.reduced_adj[u-1]:
                #print(u, v, "u-v link is in right angle to path! Penalize link [path_n[2]][path_n[1]]!")
                config.current_wsn[path_n[2]][path_n[1]]['weight'] = 10000000 #make path cost unfeasible
                config.current_wsn[path_n[3]][path_n[2]]['weight'] = 10000000 #make path cost unfeasible##############################
            else:
                #print(v," is source. This case has not been handled yet!")
                user_input = input(': ')
                if user_input is '':
                    return False
            return False

        ##double check this section!
        ##changed from if to elif
        elif u in path_nodes:
            #print(u, "u in path does not have enough resource!")
            config.current_wsn[path_nodes[path_nodes.index(u) - 1]][u]['weight'] = 10000000
            return False
        elif v in path_nodes:
            #print(v, "v in path does not have enough resource!")
            config.current_wsn[path_nodes[path_nodes.index(v) - 1]][v]['weight'] = 10000000
            return False
        else:
            for n in path_n:
                nbr = config.reduced_adj[n-1]
                ##print("n",n)
                ##print("nbr",nbr)
                if u in nbr:
                    #print(u, "u is a neighbor of",n,"and does not have enough resource!")
                    config.current_wsn[n][u]['weight'] = 10000000
                    if n != frm:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                    else:
                        config.current_wsn[frm][path_nodes[1]]['weight'] = 10000000
                    return False
                elif v in nbr:
                    #print(v, "v is a neighbor of",n," and does not have enough resource!!!")
                    config.current_wsn[n][v]['weight'] = 10000000
                    if n != frm:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                    else:
                        config.current_wsn[frm][path_nodes[1]]['weight'] = 10000000
                    return False
                    #    show_penalized_links()
                #else:
                #    #print("Unkown case x")
                #    user_input = input(': ')
                #    if user_input is '':
                #        pass

    return True

'''
        elif v in config.reduced_adj[(frm-1)]:
            #if u in two_hops_of_frm:
            for n in path_n:
                if u in config.reduced_adj[n]:
                    config.current_wsn[path_n[path_n.index(n)+1]][n]['weight'] = 10000000
                else:
                    #print("Source node v-", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                    return False


            #elif u in config.reduced_adj[n]:


        elif u in config.reduced_adj[(frm - 1)]:
            if v in two_hops_of_frm:
                for n in path_n:
                    if v in config.reduced_adj[n]:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                    else:
                        #print("Source node v-", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                        return False
        #elif u in two_hops_of_frm:



        #elif u in two_hops_of_frm:

#        shared_neighbors_u = list(set(path_nodes) & set(config.reduced_adj.get(u)))
#        shared_neighbors_v = list(set(path_nodes) & set(config.reduced_adj.get(v)))

        #shared_neighbors_u = list(set(path_nodes) & set(config.reduced_adj[u-1]))
        #common_neighbors = list(set(shared_neighbors_u) | set(shared_neighbors_v))

        elif (v, u) in shortest_path:
            #print(v, u,"v-u link in path does not have enough resource!")
            config.current_wsn[v][u]['weight'] = 10000000
            config.penalized_list.append((v, u))
            if v == to:
                if len(path_nodes) > 2:
                    #print("path length is 2 nodes")
                else:
                    #print("SSink node v", v, "does not have enough resource!\nEMBEDDING HAS FAILED!")
                    return False
        elif u in path_nodes:
            i = path_nodes.index(u)
            #print(u,"index is",i)
            config.current_wsn[path_nodes[i - 1]][path_nodes[i]]['weight'] = 10000000
            config.penalized_list.append((path_nodes[i - 1], path_nodes[i]))
            if u == frm:
#                   if len(path_nodes) > 2:
                #print("SSource node u", u, "does not have enough resource!\nEMBEDDING HAS FAILED!")
                return False
            else:
                #print("DO SOMETHING u")
        elif v in path_nodes:
            i = path_nodes.index(v)
            config.current_wsn[path_nodes[i - 1]][path_nodes[i]]['weight'] = 10000000
            config.penalized_list.append((path_nodes[i - 1], path_nodes[i]))
            if v == frm:
#                   if len(path_nodes) > 2:
                #print("SSource node v", v, "does not have enough resource!\nEMBEDDING HAS FAILED!")
                return False
            else:
                #print("DO SOMETHING v")
        else:
            #print("DO SOMETHING with highest")
            highest = 0
            for nbr in common_neighbors:
                if nbr in path_nodes:
                    i = path_nodes.index(nbr)
                    if i > highest:
                        highest = i
            if highest > 0:
                #print(" highest > 0")
                config.current_wsn[path_nodes[highest-1]][path_nodes[highest]]['weight'] = 10000000
                config.penalized_list.append((path_nodes[highest-1], path_nodes[highest]))
'''

def remove_vn(vn):
    for u,v in vn[1].edges():
        update_link_attribs(config.committed_wsn, int(u), int(v), -1, -(vn[1][u][v]['load']))
    for n in vn[0].nodes():
        update_node_attribs(config.committed_wsn,n,-(vn[0].node[n]['load']))
    return True

def get_k_shortest_paths(wsn,source,sink,k,weight=None):
    k_paths = nx.shortest_simple_paths(wsn, source=source, target=sink, weight=weight)
    k_paths = islice(k_paths, k)
    #for p in k_paths:
        #print(p)
    return k_paths

def get_min_hops():
    config.reduced_adj = copy.deepcopy(adjacencies)
    #k_paths = nx.shortest_simple_paths(config.wsn,source=34,target=1)
    #k_paths = islice(k_paths,1000)
    #for p in k_paths:
        #print(p)

    print(len(nx.shortest_path(config.wsn,source=(56),target=1))-1)
    for n in config.wsn.nodes():
        if n !=1:
            min_h = nx.shortest_path(config.wsn,source=(n),target=1)
            print((n),"to 1 is", (len(min_h)-1),"hops via",min_h)
'''
    user_inp = input('source:')
    if user_inp is not '':

        all_min_hops = nx.shortest_path(config.wsn)
        ##print(all_min_hops)
        src = all_min_hops[int(user_inp)][1]
        ##print(two_hops_list.get(src[0]))
        #print(src)

        e_list2, e_set2 = get_conflictng_edges(src)
        e_list, e_set = get_conflicting_links(src)
        #print(sorted(e_list2))
        #print(sorted(e_list))
        e_set2 = list(set(e_list2))
        #print(sorted(e_set2))
        #print(sorted(e_set))
'''

if __name__ == '__main__':
    link_weights = wsn_substrate.get_link_weights()
    #adjacencies = wsn_substrate.get_adjacency_list()
    adjacencies = wsn_substrate.get_wsn_substrate().adjacency_list()
    config.wsn = wsn_substrate.get_wsn_substrate()
    two_hops_list = wsn_substrate.get_two_hops_list()
    update_all_links_attributes(config.wsn,1, 1)
    shortest_path, path_nodes = [],[]
    vis.display_edge_attr(config.wsn)
    vis.display_node_attr(config.wsn)
    display_data_structs()
    #config.link_weights_for_this_perm = copy.deepcopy(link_weights)
    #config.adjacencies_for_this_perm = copy.deepcopy(adjacencies)
    #config.wsn_for_this_perm = copy.deepcopy(wsn)

    #run_permutations()

    while exit_flag is True:
        print("\n---->\n0 - Run permutations\n1 - Embed single VNR\n2 - Plot\n3 - Show wsn resources\n4 - Show/Remove active VNs\n5 - Min hops")
        user_input = input(':')
        if user_input is '0':
            run_permutations()
        elif user_input is '1':
            on_line_vn_request()
        elif user_input is '2':
            vis.draw_graph()
        elif user_input is '3':
            vis.display_edge_attr(config.committed_wsn)
            vis.display_node_attr(config.committed_wsn)
            display_data_structs()
            #print(config.committed_wsn.edge)
            #print(config.current_wsn.edge)
        elif user_input is '4':
            active_vns = {}
            for i,vn in enumerate(config.active_vns):
                 active_vns.update({vn[3][0]:i})
            #print("Active VNs:", list(active_vns))

            choice = input('Enter VN# to remove: ')
            removed = False
            if choice !='':
                to_remove = active_vns.get(int(choice))
                if to_remove is not None:
                    vn = config.active_vns[to_remove]
                    removed = remove_vn(vn)
                else:
                    #print("vn_to_remove", to_remove)
                    pass

                if removed:
                    config.active_vns.pop(to_remove)
                    #print("VN",to_remove,"was removed." )
            else:
                pass
        elif user_input is '5':
            get_min_hops()