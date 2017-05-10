## the original working parallel version
#from __future__ import#print_function
#import ast
#import cProfile
#import re
#import copy

import pp
import config
import itertools
from wsn_substrate import WSN
import networkx as nx
from link_weight import LinkCost

import vne
import time
import visualize as vis
from itertools import islice
import pickle
import copy
import multiprocessing
import math

def display_data_structs():
   print("Nodes - ", config.wsn.nodes(data=True))
   print("Edges - ", config.wsn.edges(data=True))
   print("Adjacency list - ", config.wsn.adjacency_list())
   print("adjacencies- ", adjacencies)
   print("two_hops_list - ", two_hops_list)
 #  #print("link_weights- ", link_weights)

def show_dataStructs(position):
    print "\n @@",position
    print "config.VWSNs",config.VWSNs
    print "config.perms_list",config.perms_list
    print "config.vns_per_perm",config.vns_per_perm
    print "config.perm_prefix",config.perm_prefix
    print "config.current_perm_emb_costs",config.current_perm_emb_costs
    print "config.overall_cost",config.overall_cost
    print "config.best_embeddings",config.best_embeddings
    print "config.max_accepted_vnrs",config.max_accepted_vnrs
    print "config.active_vns",config.active_vns
    print "config.already_mapped_vnrs",config.already_mapped_vnrs
    print "config.current_key_prefix",config.current_key_prefix
    print "config.current_perm",config.current_perm
    print "\n"

def update_all_links_attributes(wsn,plr, load):
    for u,v,d  in config.wsn.edges_iter(data=True):
        wsn[u][v]['plr'] = plr
        wsn[u][v]['load'] = load
        link_weight = LinkCost(wsn[u][v]['plr'], wsn[u][v]['load'])
        wsn[u][v]['weight'] = link_weight.get_weight(link_weight)
        #link_weights[(u, v)] = link_weight.get_weight(link_weight)
    #print(config.wsn.edges(data=True))
    #print("")

def update_node_attribs(nodes, node, load):
    for n, d in nodes.nodes_iter(data=True):
        config.total_operations += 1
        if n == int(node):
            d['load'] = d['load']+int(load)
            #d['rank'] = len(adj[n])

def update_link_attribs(wsn,u, v, plr, load):
    config.total_operations += 1
    current_link_weight = LinkCost(wsn[u][v]['plr'], wsn[u][v]['load'])
#    if plr is -1:
#        wsn[u][v]['plr'] = wsn[u][v]['plr']
#    else:
#        wsn[u][v]['plr'] += plr
    if load is -1:
        wsn[u][v]['load'] = wsn[u][v]['load']
    else:
        wsn[u][v]['load'] += load
    link_weight = LinkCost(wsn[u][v]['plr'], wsn[u][v]['load'])
    wsn[u][v]['weight'] = link_weight.get_weight(link_weight)
    return current_link_weight.get_weight(current_link_weight)

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
        embed(vnr,0,True)
    else:
        return

def map_links_cost(e_list, e_set, link_requirement,wsn):
    link_embedding_cost = 0
    current_weight = 0
    for u,v in e_set:
        config.total_operations += 1
        required = (e_list.count((u,v)) * link_requirement['load'])
        current_weight = update_link_attribs(wsn, int(u), int(v), link_requirement['plr'], required)
#        link_embedding_cost +=(required * current_weight) #weighted cost
        link_embedding_cost += (required  * wsn[u][v]['plr'])
    return link_embedding_cost

def map_nodes_cost(all_path_nodes, required_load,wsn):
    node_embedding_cost = 0
    for idx, pn in enumerate(all_path_nodes):
        config.total_operations += 1
        update_node_attribs(wsn, pn, required_load)
        node_embedding_cost += (required_load)
    return node_embedding_cost

def check_link_reliability_constraints(shortest_path, required_reliability,wsn):
    link_list = []
    if shortest_path is not None:
        for u,v in shortest_path:
            link_list.append(1- wsn[u][v]['plr']/100.0)
        path_reliability = reduce(lambda x, y: x * y, link_list)
        if path_reliability <= 1 - required_reliability/100:
            return False
        else:
            return True

def check_link_constraints(shortest_path,e_list, e_list2, load, required_plr, wsn):
    VN_links = nx.DiGraph()
    if not check_link_reliability_constraints(shortest_path, required_plr, wsn):  # need to fix 1 hop paths
        worst_link = (0,0)
        worst_plr = 0.0
        #print(shortest_path)
        for u, v in shortest_path:
            if wsn.edge[u][v]['plr'] >= worst_plr:
                #print(u,v,wsn.edge[u][v]['plr'],worst_plr)
                worst_plr = wsn.edge[u][v]['plr']
                worst_link = (u, v)
        #print ("worst_link",worst_link)
        return worst_link,VN_links
    for u,v in e_list2:
        config.verify_operations += 1
        required_load = load * e_list.count((u,v))
        if wsn.edge[u][v]['load'] + required_load > 100:
            #print("Link",u, v,"requires",wsn.edge[u][v]['load']," + ",required_load, "but have not got enough")
            return (u,v),VN_links
        else:
            VN_links.add_edge(u,v, {'load':required_load})
            return_value = (0,0)
    return return_value,VN_links

def check_node_constraints(nodes_in_path, required_load, wsn):
    VN_nodes = nx.DiGraph()
    for idx,n in enumerate(nodes_in_path):
        config.verify_operations += 1
        VN_nodes.add_node(n, {'load': required_load})
        if wsn.node[n]['load'] + required_load > 100:
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
    #path, length = nx.dijkstra_path(graph, source=frm, target=to, weight='weight')
#    length,path = nx.bidirectional_dijkstra(graph, source=frm, target=to, weight='weight')
    #length,path = nx.astar_path(graph, source=frm, target=to, heuristic=None, weight='weight')
    if config.sp_alg_str == "Dijkstra":
        length, path = nx.bidirectional_dijkstra(graph, source=frm, target=to, weight='weight')
    else:
        length, path = nx.astar_path_length(graph, source=frm, target=to, heuristic=None, weight='weight')

    #length = nx.dijkstra_path_length(graph, source=frm, target=to, weight='weight')
    #print('Shortest path weight is ',length)
    config.verify_operations += 1
    if (path is None) or (length >= 10000000):
        return None, None
    s_path = []
    #print('Shortest path is ', end="")
    for idx, p in enumerate(path):
        config.verify_operations += 1
        if idx == 0:
            #shortest_path.append((p, p+1))
            #print(vis.CGREEN, p, "->", end="")
            pass
        elif idx == (len(path) - 1):
            #print(p, vis.CEND)
            pass
        else:
            #print(p, "->", end="")
            pass
        if idx != len(path) - 1:
            s_path.append((path[idx], path[idx + 1]))
    #print("Shortest path links  ", s_path,"\n")
    return s_path, path

def get_max_edge_load(wsn,node, is_source):
    max_load = 0
    for n in config.reduced_adj[node]:
        config.verify_operations += 1
        link_load = wsn[node][n]['load']
        if link_load > max_load:
            max_load = link_load
        if not is_source:
            link_load = wsn[n][node]['load']
            if link_load > max_load:
                max_load = link_load
    return max_load

def check_frm_to_links(wsn,node,link_requirement):
    for n in config.reduced_adj[node]:
        config.verify_operations += 1
        if (100 - wsn[node][n]['load'] < link_requirement['load']) or (100 - wsn[n][node]['load'] < link_requirement['load']):
            return False
    return True

def remove_vn(vn):
    for u,v in vn[1].edges():
        update_link_attribs(config.committed_wsn, int(u), int(v), -1, -(vn[1][u][v]['load']))
    for n in vn[0].nodes():
        update_node_attribs(config.committed_wsn,n,-(vn[0].node[n]['load']))
    return True

def get_min_hops(sink):
    config.reduced_adj = list(adjacencies)
    min_hops = {}
    #print(len(nx.shortest_path(config.wsn,source=(56),target=sink))-1)
    for n in config.wsn.nodes():
        if n !=1:
            min_h = nx.shortest_path(config.wsn,source=(n),target=sink)
            min_hops.update({(n,sink):(len(min_h)-1)})
            #print(n,"to",sink,"is", (len(min_h)-1),"hops via",min_h)
            #print(min_hops)
    return min_hops

def generate_output(best):
    acceptance_rate = float(best['acceptance']) / float(config.numvn)
    output_dict = {
        # First three copied from input vector
        'nwksize': config.nwksize,
        'numvn': config.numvn,
        'iteration': config.iteration,
        # Following result from algorithm execution
        'proc_time': config.proc_time,
        'acceptance': acceptance_rate,
        'mapping': best['mappings'],
        'objective': best['overall_cost']
    }
    config.result_vectors.append(output_dict)

def write_to_File():
    suffix = '_with_plr_4'
    try:
        with open(dir_path + 'results/' + input_file_name + suffix, 'w') as handle:
            pickle.dump(config.result_vectors, handle)
    except Exception as e:
        print (e)
        return -1
    return 0

def commit(VN_nodes, VN_links, node_requirement, link_requirement,e_list, e_list2, path_nodes, shortest_path,wsn):
   #print "\nCOMMIT",path_nodes[0],"\n"
    config.total_operations += 1
#    n_cost = map_nodes_cost(VN_nodes.nodes(), node_requirement,wsn)
    l_cost = map_links_cost(e_list, e_list2, link_requirement,wsn)
#    cost = (n_cost+l_cost) #node costs are not used by Victor
    cost = l_cost
    current_vn = (nx.DiGraph(VN_nodes), nx.DiGraph(VN_links), list(shortest_path), list(path_nodes), float(cost), int(path_nodes[0]))
    config.VWSNs.append(current_vn)
    if config.online_flag:
        vis.display_edge_attr(config.committed_wsn)
        vis.display_node_attr(config.committed_wsn)
        vis.display_vn_node_allocation(VN_nodes)
        vis.display_vn_edge_allocation(VN_links)
        vis.plotit(VN_links, shortest_path, path_nodes, 0)
        config.active_vns.append(current_vn)

    config.current_perm_emb_costs.update({path_nodes[0]: cost})
    config.overall_cost += cost
    #show_dataStructs("COMMITTED")

def verify_feasibility(link_requirement, frm, to, node_requirement):
    #print("verify")
    config.verify_operations += 1
    hops = min_hops_dict.get((frm,to))
    max_load = 0
    wsn = nx.DiGraph()
    if config.online_flag:
        wsn = config.committed_wsn
    else:
        wsn = config.wsn_for_this_perm
    node_check, VN_nodes = check_node_constraints([frm, to], node_requirement, wsn)
    max_load = max([get_max_edge_load(wsn, frm, True),get_max_edge_load(wsn, to, False)])
    if not check_frm_to_links(wsn,to, link_requirement):
        #print("Sink node ", to, "does not have enough link resource\nEMBEDDING FAILED!")
        return False
    if not check_frm_to_links(wsn,frm, link_requirement):
        #print("Source node ", frm, "does not have enough link resource\nEMBEDDING FAILED!")
        return False
    # This may need to be disabled as Victor is not using it
    if node_check != 0:
        #print("node ", node_check, "does not have enough resource\nEMBEDDING FAILED!")
        return False
    if hops > 2:
        if (link_requirement['load']*2) > (100 - max_load):
            #print("Failed!",frm,to," cannot support request\nEMBEDDING FAILED!")
            return False
    shortest_path, path_nodes = get_shortest_path(config.current_wsn, frm, to)
#    shortest_path, path_nodes = get_shortest_path(config.current_wsn_removed_edges, frm, to)

    if shortest_path is None:
        #print("No feasible path!\nEMBEDDING HAS FAILED!")
        return False
    e_list, e_set = get_conflicting_links(path_nodes)
    ##get list of unique nodes from conflicting link list
    effected_nodes = []
    for u, v in e_set:
        config.verify_operations += 1
        if u not in effected_nodes:
            effected_nodes.append(u)
        if v not in effected_nodes:
            effected_nodes.append(v)
    node_check, VN_nodes = check_node_constraints(effected_nodes, node_requirement, wsn)
    #This may need to be disabled as Victor is not using it
    if node_check != 0:
       #print("node ", node_check, "does not have enough resource\nEMBEDDING FAILED!")
        return False
    else:
        link_check, VN_links = check_link_constraints(shortest_path,e_list, e_set, link_requirement['load'], link_requirement['plr'],wsn)
    if link_check == (0,0):
        #print("++SUCCESSFUL EMBEDDING++")
        config.has_embedding = True
        commit(VN_nodes, VN_links, node_requirement, link_requirement, e_list, e_set, path_nodes, shortest_path, wsn)
        config.feasible = True
        return False
    else:
        if link_check not in config.avoid:
            config.avoid.append(link_check)
        if recalculate_path_weights(frm, to, path_nodes, shortest_path):
            #print("recalculate_path_weights returned TRUE!!!\nEMBEDDING FAILED! ")
            return False
        #else:
            #print("recalculate_path_weights returned OK")
    check_again(link_requirement, frm, to, node_requirement)
        #verify_feasibility(link_reqiurement, frm, to, node_requirement)

def check_again(link_reqiurement, frm, to, node_requirement):
    config.verify_operations += 1
    if config.feasible == False:
        is_failed = verify_feasibility(link_reqiurement, frm, to, node_requirement)
        if is_failed:
           #print("Verify failed!!! ")
            pass

def get_conflicting_links(path_nodes):
    config.verify_operations += 1
    tx_nodes = list(path_nodes)
    tx_nodes.pop()
    effected_edges = []
    for i, rx in enumerate(path_nodes):
        config.verify_operations += 1
        if i != 0:
            effected_edges.extend(conflicting_links_dict[path_nodes[i-1]][rx])
    effected_edges_set = list(set(effected_edges))
    return effected_edges, effected_edges_set

'''
##This was re-implemented in wsn_substrate.py to pre-compute it before the algorithm runs
def get_conflicting_links__(path_nodes):
#    tx_nodes = copy.deepcopy(path_nodes)
    tx_nodes = list(path_nodes)
    tx_nodes.pop()
#    #print("tx_nodes",tx_nodes,"\npath_nodes",path_nodes)
    effected_edges = []
#    #print("initialize effected_edges",effected_edges)
    for i,tx in enumerate(tx_nodes):
        ##config.total_operations += 1+= 1
        visited_nodes = []
#        #print(i,"visit tx", tx)
        visited_nodes.append(tx)
#        #print(tx,"appended to visited nodes", visited_nodes)
        for n in config.reduced_adj[tx-1]:
            ##config.total_operations += 1+= 1
#            #print(tx,"-> visit n", n)
            if n not in visited_nodes:
#                #print("add if n not in []",visited_nodes)
                effected_edges.append((tx, n))
                effected_edges.append((n, tx))
            for nn in config.reduced_adj[n-1]:
                ##config.total_operations += 1+= 1
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
            ##config.total_operations += 1+= 1
#            #print(rx, "-> visit n", n)
            if n not in visited_nodes:
                ##config.total_operations += 1+= 1
                for nn in config.reduced_adj[n-1]:
                    ##config.total_operations += 1+= 1
#                    #print(n, "->-> visit nn", nn)
                    if nn not in visited_nodes:
#                        #print("add if nn not in []", visited_nodes)
                        effected_edges.append((n, nn))
                visited_nodes.append(n)
#                #print(n, "appended to visited nodes in rx", visited_nodes)
    effected_edges_set = list(set(effected_edges))
    return effected_edges, effected_edges_set
'''

def embed(vnr,idx,prev_success):
    vwsn_nodes = vnr[1]
    link_reqiurement = vnr[2]
    frm = list(vwsn_nodes)[0]
    to = list(vwsn_nodes)[2]
    node_requirement = vwsn_nodes[1]['load']
    del config.avoid
    config.avoid = []
    if config.online_flag:
        print("ONLINE EMBEDDING")
        config.VWSNs = []
        config.current_perm_emb_costs = {}
        if config.has_embedding == False:
            config.committed_wsn = nx.DiGraph(config.wsn)
        config.current_wsn = nx.DiGraph(config.committed_wsn)
        config.reduced_adj = list(config.committed_wsn.adjacency_list())
    else:
        #check if same request has failed in previous perm at a higher position
        if not prev_success:
            #print("--if not prev_success return\n")
            return False
        #check if current sequence has been memoized
        if str(config.current_key_prefix) in config.already_mapped_vnrs:
            config.wsn_for_this_perm = nx.DiGraph(config.already_mapped_vnrs[str(config.current_key_prefix)]['graph'])
            config.current_perm_emb_costs = dict(config.already_mapped_vnrs[str(config.current_key_prefix)]['embeddings'])    #.update({path_nodes[0]: cost})
            config.overall_cost = float(config.already_mapped_vnrs[str(config.current_key_prefix)]['overall_cost'])
            config.best_embeddings = dict(config.already_mapped_vnrs[str(config.current_key_prefix)]['best_embeddings'])
            config.VWSNs = list(config.already_mapped_vnrs[str(config.current_key_prefix)]['vwsns'])
            return False
        #else:
            #show_dataStructs("ELSE, NOT IN MEMOIZED")
        del config.current_wsn
        config.current_wsn = nx.DiGraph(config.wsn_for_this_perm)
        del config.reduced_adj
        config.reduced_adj = copy.deepcopy(adjacencies)
        remove_insufficient_links(vnr[2]['plr'])
    config.perm_counter += 1
    config.feasible = False
    verify_feasibility(link_reqiurement, frm, to, node_requirement)
    return True

def remove_insufficient_links(required_link_quality):
    #del config.current_wsn_removed_edges
    #config.current_wsn_removed_edges = copy.deepcopy(config.current_wsn)
    for item in wsn_substrate.get_link_quality():
        if item[1] >= required_link_quality:
            #print (item[0][0],item[0][1],"-",100.0-item[1])
            #print (item[0][0], "-", item[0][1])
            #config.reduced_adj[item[0][0]].remove(item[0][1])
            #config.current_wsn_removed_edges.remove_edge(item[0][0],item[0][1])
            config.current_wsn[item[0][0]][item[0][1]]['weight'] = 10000000
        #else:
            #print ("+",item[0][0], item[0][1],100.0-item[1])

def evaluate_perms(current_perm):

   #print ("current_perm",current_perm)
    keys = []
    for k in current_perm:
        ##print(k)
        config.total_operations += 1
        keys.append(k)
    source_nodes = []
    overall_cost = current_perm[keys[0]]['overall_cost']
    for k, v in current_perm[keys[0]]['embeddings'].items():
        config.total_operations += 1
        source_nodes.append(k)
    if len(config.best_embeddings) != 0:
        config.total_operations += 1
        if config.max_accepted_vnrs < len(source_nodes):
            config.max_accepted_vnrs = len(source_nodes)
            current_key = list(config.best_embeddings.keys())
            config.best_embeddings.pop(current_key[0], 0)
            config.best_embeddings.update(
                {str(source_nodes): {'overall_cost': overall_cost, 'permutation': keys[0]}})
            del config.committed_wsn
            config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
            del config.active_vns
            config.active_vns = list(config.VWSNs)
        elif config.max_accepted_vnrs == len(source_nodes):
            current_key = list(config.best_embeddings.keys())
            best_cost = config.best_embeddings[current_key[0]]['overall_cost']
            if best_cost > overall_cost:
                config.best_embeddings.pop(current_key[0], 0)
                config.best_embeddings.update(
                    {str(source_nodes): {'overall_cost': overall_cost, 'permutation': keys[0]}})
                del config.committed_wsn
                config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
                del config.active_vns
                config.active_vns = list(config.VWSNs)
    else:
        config.total_operations += 1
        config.best_embeddings.update({str(source_nodes): {'overall_cost': overall_cost, 'permutation': keys[0]}})
        config.max_accepted_vnrs = len(source_nodes)
        del config.committed_wsn
        config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
        del config.active_vns
        config.active_vns = list(config.VWSNs)
        config.acceptance = config.max_accepted_vnrs

#evaluate the quality of the embeddings based on the objective function [min (Cost(max (VNRs())))]
def evaluate_perms_(current_perm):
   #print ("current_perm", current_perm)
    keys = []
    for k in current_perm:
        #print("keys-",k)
        config.total_operations += 1
        keys.append(k)
    source_nodes = []
    overall_cost = float(current_perm[keys[0]]['overall_cost'])

    for k, v in current_perm[keys[0]]['embeddings'].items():
        config.total_operations += 1
        #print("source_nodes-",k,"-v-",v)
        source_nodes.append(k)

    if len(source_nodes) > config.max_accepted_vnrs:
        config.best_embeddings.update({str(source_nodes): {'overall_cost': float(overall_cost), 'permutation': keys[0]}})
        current_key = list(config.best_embeddings.keys())
        config.max_accepted_vnrs = len(source_nodes)
        del config.committed_wsn
        config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
        del config.active_vns
        config.active_vns = list(config.VWSNs)
    elif len(source_nodes) == config.max_accepted_vnrs and len(source_nodes) != 0 :
        current_key = list(config.best_embeddings.keys())
        best_cost = float(config.best_embeddings[current_key[0]]['overall_cost'])
        if best_cost > overall_cost:
            config.best_embeddings.pop(current_key[0], 0)
            config.best_embeddings.update({str(source_nodes): {'overall_cost': float(overall_cost), 'permutation': keys[0]}})
            del config.committed_wsn
            config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
            del config.active_vns
            config.active_vns = list(config.VWSNs)

#memoize and use already calculated sequences to eliminate duplicate work
def memoize_perms():
    #show_dataStructs("\nMEMOIZE_PERMS")
    if len(config.prefix_length) < len(config.current_key_prefix):
        config.prefix_length.append(list(config.current_key_prefix))
        config.already_mapped_vnrs.update({str(config.current_key_prefix): {
            'graph': nx.DiGraph(config.wsn_for_this_perm), 'embeddings': dict(config.current_perm_emb_costs),
            'overall_cost': float(config.overall_cost), 'best_embeddings': dict(config.best_embeddings), 'vwsns': list(config.VWSNs)}})
    elif config.prefix_length[len(config.current_key_prefix) - 1] != config.current_key_prefix:
        config.already_mapped_vnrs.pop(str(config.prefix_length[len(config.current_key_prefix) - 1]))
        config.prefix_length[len(config.current_key_prefix) - 1] = config.current_key_prefix
        config.already_mapped_vnrs.update({str(config.current_key_prefix): {
            'graph': nx.DiGraph(config.wsn_for_this_perm), 'embeddings': dict(config.current_perm_emb_costs),
            'overall_cost': float(config.overall_cost), 'best_embeddings': dict(config.best_embeddings), 'vwsns': list(config.VWSNs)}})

#uses no suffix for output file
def recalculate_path_weights_old(frm,to,path_n,shortest_path):
    for (u, v) in config.avoid:
        config.link_penalize_operations += 1
        #print("recalculate",u,v)
        path_nodes = list(path_n)
        path_n.reverse()
        config.avoid.remove((u, v))
        config.current_wsn[u][v]['weight'] = 10000000 #penalize link
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
            for n in config.reduced_adj[frm]:
                config.link_penalize_operations += 1
                config.current_wsn[frm][n]['weight'] = 10000000  # make path cost unfeasible
            return False
        elif (v, u) in shortest_path:
            #print(v, u, "v-u link in path does not have enough resource!")
            #print(config.current_wsn[v][u]['weight'])
            config.current_wsn[v][u]['weight'] = 10000000
            return False
        elif (u, v) in shortest_path:
            #print(u, v, "u-v link in path does not have enough resource!")
            config.current_wsn[u][v]['weight'] = 10000000
            #config.current_wsn[shortest_path[shortest_path.index(u)-1]][u]['weight'] = 10000000  #about this I'm not sure!!
            return False
        elif u == frm:
            if (len(path_nodes) <= 3) or (v != path_nodes[1]):
                #print("Source node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                for n in config.reduced_adj[frm]:
                    config.link_penalize_operations += 1
                    config.current_wsn[frm][n]['weight'] = 10000000 #make path cost unfeasible
            elif v == path_nodes[1]:
                for n in path_n:
                    config.link_penalize_operations += 1
                    if n in config.reduced_adj[frm]:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                        #print("Source node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                        return False
            else:
                #print("Source node u", u, "does not have enough resource. This case has not been handled yet!\nEMBEDDING HAS FAILED!")
                user_input = input('?: ')
                if user_input is '':
                    return False
            return False
        elif v == frm:
            if (len(path_nodes) <= 3) or (u != path_nodes[1]):
                #print("Source node v", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                for n in config.reduced_adj[frm]:
                    config.link_penalize_operations += 1
                    config.current_wsn[frm][n]['weight'] = 10000000 #make path cost unfeasible
            elif u == path_nodes[1]:
                for n in path_n:
                    config.link_penalize_operations += 1
                    if n in config.reduced_adj[frm]:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                        #print("Source node v", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                        return False
            else:
                #print("Source node v", v, "does not have enough resource. This case has not been handled yet!\nEMBEDDING HAS FAILED!")
                user_input = input('??: ')
                if user_input is '':
                    return False
            return False
        elif u == to:
            #print("Sink node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
            for n in config.reduced_adj[frm]:
                config.link_penalize_operations += 1
                config.current_wsn[frm][n]['weight'] = 10000000  # make path cost unfeasible
            return False
        elif v == to:
            #print(u,v)
            #print("path_n", path_n)
            #print("path_n[1]",path_n[1])
            #print("path_n[2]", path_n[2])
            #print("config.reduced_adj[u-1]",config.reduced_adj[u-1])
            #print("config.reduced_adj[u]", config.reduced_adj[u])
            if (u != path_n[1]) and (path_n[2] not in config.reduced_adj[u]):
                #print("Sink node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
 #               for n in config.reduced_adj[to-1]:
                for n in config.reduced_adj[frm]:
                    config.link_penalize_operations += 1
                    config.current_wsn[frm][n]['weight'] = 10000000 #make path cost unfeasible
            elif u == path_n[1]:
                #print(u, v, "u-v link is in path! Penalize link (predecessor of u)->u!")
                config.current_wsn[path_n[2]][u]['weight'] = 10000000 #make path cost unfeasible
            elif path_n[2] in config.reduced_adj[u]:
                #print(u, v, "u-v link is in right angle to path! Penalize link [path_n[2]][path_n[1]]!")
                config.current_wsn[path_n[2]][path_n[1]]['weight'] = 10000000 #make path cost unfeasible
                config.current_wsn[path_n[3]][path_n[2]]['weight'] = 10000000 #make path cost unfeasible##############################
            else:
                #print(v," is source. This case has not been handled yet!")
                user_input = input('???: ')
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
                config.link_penalize_operations += 1
                nbr = config.reduced_adj[n]
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
                    #show_penalized_links()
    return True

#uses suffix _ for output file
def recalculate_path_weights(frm,to,path_n,shortest_path):
    for (u, v) in config.avoid:
        config.link_penalize_operations += 1
        #print("recalculate",u,v)
        path_nodes = list(path_n)
        path_n.reverse()
        config.avoid.remove((u, v))
        config.current_wsn[u][v]['weight'] = 10000000 #penalize link

        if len(path_nodes) == 2:
            #print("Source node", frm, "does not have enough resource in a single-hop path.\nEMBEDDING HAS FAILED!")
            #for n in config.reduced_adj[frm]:
                #config.link_penalize_operations += 1
                #config.current_wsn[frm][n]['weight'] = 10000000  # make path cost unfeasible
            return True
        elif (v, u) in shortest_path:
            #print(v, u, "v-u link in path does not have enough resource!")
            #print(config.current_wsn[v][u]['weight'])
            config.current_wsn[v][u]['weight'] = 10000000
            return False
        elif (u, v) in shortest_path:
            #print(u, v, "u-v link in path does not have enough resource!")
            config.current_wsn[u][v]['weight'] = 10000000
            #config.current_wsn[shortest_path[shortest_path.index(u)-1]][u]['weight'] = 10000000  #about this I'm not sure!!
            return False
        elif u == frm:
            #if (len(path_nodes) <= 3) or (v != path_nodes[1]):
            if (len(path_nodes) <= 3):
                #print("Source node u+", u, "does not have enough resource in 2-hops path.\nEMBEDDING HAS FAILED!")
                #for n in config.reduced_adj[frm]:
                    #config.link_penalize_operations += 1
                    #print(config.reduced_adj[frm])
                    #print(frm,n)
                    #config.current_wsn[frm][n]['weight'] = 10000000 #make path cost unfeasible
                return True
            #this is probably redundant
            elif v == path_nodes[1]:
                for n in path_n:
                    config.link_penalize_operations += 1
                    if n in config.reduced_adj[frm]:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                        #print("Source node u-", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                        return False
            else:
                #print("Source node u", u, "does not have enough resource. This case has not been properlyy handled yet!\nEMBEDDING HAS FAILED!")
                config.current_wsn[path_n[1]][path_n[0]]['weight'] = 10000000
                #user_input = input('?: ')
                #if user_input is 0:
                    #return False
            return False
        elif v == frm:
            #if (len(path_nodes) <= 3) or (u != path_nodes[1]):
            if len(path_nodes) <= 3:
                #print("Source node v+", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                #for n in config.reduced_adj[frm]:
                    #config.link_penalize_operations += 1
                    #config.current_wsn[frm][n]['weight'] = 10000000 #make path cost unfeasible
                return True
            # both cases below are probably redundant here
            elif u == path_nodes[1]:
                for n in path_n:
                    config.link_penalize_operations += 1
                    if n in config.reduced_adj[frm]:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                        #print("Source node v-", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                        return False
            else:
                #print("Source node v", v, "does not have enough resource. This case has not been handled yet!\nEMBEDDING HAS FAILED!")
                config.current_wsn[path_n[1]][path_n[0]]['weight'] = 10000000
                #user_input = input('??: ')
                #if user_input is '':
                    #return False
            return False
        elif u == to:
            #print("Sink node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
            #for n in config.reduced_adj[frm]:
                #config.link_penalize_operations += 1
                #config.current_wsn[frm][n]['weight'] = 10000000  # make path cost unfeasible
            return True
        elif v == to:
            #print(u,v)
            #print("path_n", path_n)
            #print("path_n[1]",path_n[1])
            #print("path_n[2]", path_n[2])
            #print("config.reduced_adj[u-1]",config.reduced_adj[u-1])
            #print("config.reduced_adj[u]", config.reduced_adj[u])
            #print("config.reduced_adj[v]", config.reduced_adj[v])
            if (u != path_n[1]) and (path_n[2] not in config.reduced_adj[u]):
                config.current_wsn[path_n[1]][path_n[0]]['weight'] = 10000000
                #print("Sink node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
 #               for n in config.reduced_adj[to-1]:
                #for n in config.reduced_adj[frm]:
                    #config.link_penalize_operations += 1
                    #config.current_wsn[frm][n]['weight'] = 10000000 #make path cost unfeasible
                #return True
            elif (u != path_n[1]) and (path_n[2] in config.reduced_adj[u]):
                #print(u, v, "u-v link is in right angle to path!! Penalize link [path_n[2]][path_n[1]]!")

                config.current_wsn[path_n[2]][path_n[1]]['weight'] = 10000000  # make path cost unfeasible
            elif u == path_n[1]:
              #print(u, v, "u-v link is in path! Penalize link (predecessor of u)->u!")
                config.current_wsn[path_n[2]][u]['weight'] = 10000000 #make path cost unfeasible
            elif path_n[2] in config.reduced_adj[u]:
                #This may be redundant here!!
                #print(u, v, "u-v link is in right angle to path!! Penalize link [path_n[2]][path_n[1]]!")
                config.current_wsn[path_n[2]][path_n[1]]['weight'] = 10000000 #make path cost unfeasible
                if len(path_n) > 3:
                    config.current_wsn[path_n[3]][path_n[2]]['weight'] = 10000000 #make path cost unfeasible##############################
            else:
                #print(v," is source. This case has not been handled yet!")
                user_input = input('???: ')
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
                config.link_penalize_operations += 1
                nbr = config.reduced_adj[n]
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
                    #show_penalized_links()
    return True

def reinitialize():
   #print "\nREINIT ALL\n"
    # VNR specific fields (used in embed())
    config.current_wsn = {}  # copy of wsn_for_this_perm represents the substrate
    config.reduced_adj = dict()  # copy of original adjacency list of substrate
    config.avoid = []  # holds the links that need to be penalised
    config.has_embedding = False  # indicates if there are existing appings (used for ONLINE)
    config.feasible = False  # controls the termination of re-tries (used for verify_feasibility)
    config.online_flag = False  # controls embedding mode

    # permutation specific fields
    config.wsn_for_this_perm = {}  # copy of wsn represents the substrate
    config.VWSNs = []  # feasible embeddings for each/current permutation
    config.current_perm_emb_costs = {}  # embeddings and their costs for current permutation
    config.overall_cost = 0  # total cost for all embeddings in each/current permutation
    config.committed_wsn = {}  # copy of wsn_for_this_perm after identifying final/best embedding for current perm
    config.max_accepted_vnrs = 0  # highest number of vnrs in current perm

    config.already_mapped_vnrs = {}  # memoizes results of previous embeddings (used in independent perm blocks)
    config.best_embeddings = {}  # best embeddings for each combination (ultimately the optimal solution) [('[8, 4, 45]', {'overall_cost': 56371.374759009785, 'permutation': 0})]

    # test case specific fields
    config.wsn = {}  # the original instance of the initialized substrate graph
    config.active_vns = []  # copy of VWSNs, stores the list of embedded vns (used by generate_output to update mapping_dictionary)

    #config.sp_alg_str = "Dijkstra"  # identifies path finder algorithm used for test
    config.main_sink = 0  # identifies sink node id used for test
    config.X = 0  # used to set graph dimensions manually
    config.Y = 0  # used to set graph dimensions manually

    # Output, first three copied from input vector (used in pickle file)
    config.nwksize = 0
    config.numvn = 0
    config.iteration = 0
    # Following result from algorithm execution
    config.proc_time = 0.0
    # acceptance = config.max_accepted_vnr / config.numvn
    config.mapping = dict()  #: dictionary vlink:[slinks],
    config.overall_cost = 0.0
    config.acceptance = 0
#    config.result_vectors = []  # stores test results until written into pickle file

    # performance metrics /fields
    config.start = 0.0
    config.perm_counter = 0
    config.total_operations = 0
    config.dijkstra_operations = 0
    config.link_penalize_operations = 0
    config.verify_operations = 0
    config.plot_counter = 0

    config.current_perm = []
    config.previous_perm = []
    config.perm_prefix = []
    config.vns_per_perm = {}  # success/fail of each vnrs per perms
    config.perms_list = {}  # store success/fail of vnrs for all perms

    config.prefix_length = []  # used to identify already calculated sequences to eliminate duplicate work
    config.current_key_prefix = []


    config.current_perm_block_results = {}
    config.current_perm_results = {}
    config.current_test_best = {'acceptance':0,'overall_cost':0.0,'committed':None,'mappings':None}


def generate_independent_perm_blocks(vnrs_list, result_q):
    perms = tuple(itertools.permutations(vne.get_vnrs(vnrs_list), r=None))
    vnrs_size = len(vne.get_vnrs(vnrs_list))
    start_indx = 0
    idx_increm = math.factorial(vnrs_size - 1)
    for v in range(0,1):# vnrs_size):
        config.total_operations += 1
        end_indx = start_indx + idx_increm
        independent_perm_block = itertools.islice(perms, start_indx, end_indx)
        start_indx = end_indx
        jobs.append(multiprocessing.Process(target=process_independent_perm_block, args=(independent_perm_block,result_q)) )
#    jobs.append(mp.Process(target=process_independent_perm_block, args=(perms,result_q)) )

def process_independent_perm_block(independent_perm_block,result_q):
    #show_dataStructs("\nPROCESS INDEPENDENT PERM BLOCK")
    config.online_flag = False
    perms = independent_perm_block
    config.current_perm_block_best = {'acceptance':0,'overall_cost':0.0,'committed':None,'mappings':None}
    config.current_perm_block_results = {}  # stores final results for current perm block
    config.current_perm_block_acceptance = 0  # max accepted vnrs for current perm block
    config.current_perm_block_cost = 0.0  # max embedding cost for current perm block
    config.best_embeddings = {}
    config.max_accepted_vnrs = 0
    config.active_vns = []  # ?

    for i, per in enumerate(perms):
        #print i, "th permutation", multiprocessing.current_process()
        config.current_perm_results = {}  # stores final results for current perm (goes into current_perm_block_results)
        del config.wsn_for_this_perm
        config.wsn_for_this_perm = nx.DiGraph(config.wsn)
        del config.VWSNs
        config.VWSNs = []
        del config.current_perm_emb_costs  # ?
        config.current_perm_emb_costs = dict()  # ?
        del config.overall_cost
        config.overall_cost = 0.0
        del config.vns_per_perm
        config.vns_per_perm = dict()
        config.feasible = False
        config.current_key_prefix = []

        for idx, vnr in enumerate(per):
            current_success = True
            # concatenate current source to previous sequence
            config.current_key_prefix = config.current_key_prefix + [(idx, vnr[1][0])]
            # optmize work effort by avoiding to process known unfeasible sequence of requests
            # check the state of the same request in the previous perm and if it has failed at a
            # higher index/position then skip it
            # (cases when same source node requested multiple times in a perm needs to be handled yet)
            # condition makes sense since there is no previous perm of perm 0 nor vnr 0
            if i > 0 and idx > 0:
                # get the position of current vnr in previous perm
                previous_position = list(config.perms_list[i - 1][vnr[1][0]].keys())[0]
                # get the result of current vnr in previous perm
                success = config.perms_list[i - 1][vnr[1][0]].get(previous_position)
                # check if current vnr was in a higher position in the previous perm and if it Failed
                if previous_position < idx and success is False:
                    # set flag to False so it indicates that it must be skipped
                    current_success = False
            if embed(vnr, idx, current_success):
                current_success = config.feasible
                #               if current_success == False:
                #                   config.current_perm_emb_costs.update({vnr[1][0]: 0.0})
            else:
                # get the position of current vnr in previous perm
                previous_position = list(config.perms_list[i - 1][vnr[1][0]].keys())[0]
                # get the result of current vnr in previous perm
                current_success = config.perms_list[i - 1][vnr[1][0]].get(previous_position)
            # memoize success/fail of current vnr
            config.vns_per_perm.update({vnr[1][0]: {idx: current_success}})
            # memoize the ordered subsets for each sequence up to the n-2 left most positions
            # last 2 items must be calculated anyways
            if idx < len(per) - 2:
                memoize_perms()
        # if current perm has at least 2 previous
        if i > 1:
            config.perms_list.pop(i - 2)

        config.current_perm_results.update({str(config.current_key_prefix): {'acceptance': int(len(config.VWSNs)),
                                                                             'overall_cost': float(config.overall_cost),
                                                                             'vwsns': list(config.VWSNs)}})

        config.perms_list.update({i: config.vns_per_perm})
        current_perm = {i: {'embeddings': config.current_perm_emb_costs, 'overall_cost': config.overall_cost}}
        evaluate()
        #evaluate_perms(current_perm)
#    config.current_perm_block_results.update({i:dict(config.current_perm_results)})
#   #print "config.current_perm_block_results",config.current_perm_block_results
    perm_block_result = (dict(config.best_embeddings), int(config.max_accepted_vnrs), nx.DiGraph(config.committed_wsn), list(config.active_vns))

    result_q.put(config.current_perm_block_best)


def evaluate():
    perm_result = config.current_perm_results[config.current_perm_results.keys()[0]]
    if config.current_perm_block_acceptance < perm_result['acceptance']:
        config.current_perm_block_acceptance = perm_result['acceptance']
        config.current_perm_block_cost = perm_result['overall_cost']
        config.current_perm_block_best = {
            'acceptance': perm_result['acceptance'],'overall_cost': perm_result['overall_cost'],
            'committed': nx.DiGraph(config.wsn_for_this_perm), 'mappings':perm_result}
        config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
    elif config.current_perm_block_acceptance == perm_result['acceptance']:
        if config.current_perm_block_cost > perm_result['overall_cost']:
            config.current_perm_block_cost = perm_result['overall_cost']
            config.current_perm_block_best = {
                'acceptance': perm_result['acceptance'], 'overall_cost': perm_result['overall_cost'],
                'committed': nx.DiGraph(config.wsn_for_this_perm), 'mappings': perm_result}
            config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
   #print "config.current_perm_block_best",config.current_perm_block_best

def select_best__(overall_results):
    best_cost = -1
    max_acccepted = -1
    best_result = ()
    while overall_results.empty() is False:
        config.total_operations += 1
        perm_block_result = overall_results.get()
       #print "perm_block_result",perm_block_result
        if max_acccepted == -1:
            best_result = config.best_embeddings = dict(perm_block_result[0])
            max_acccepted = config.max_accepted_vnrs = int(perm_block_result[1])
            config.committed_wsn = nx.DiGraph(perm_block_result[2])
            config.active_vns = list(perm_block_result[3])
            best_cost = config.overall_cost = float(perm_block_result[0][perm_block_result[0].keys()[0]]['overall_cost'])
        elif perm_block_result[1] > max_acccepted:
            best_result = config.best_embeddings = dict(perm_block_result[0])
            max_acccepted = config.max_accepted_vnrs = int(perm_block_result[1])
            config.committed_wsn = nx.DiGraph(perm_block_result[2])
            config.active_vns = list(perm_block_result[3])
            best_cost = config.overall_cost = float(perm_block_result[0][perm_block_result[0].keys()[0]]['overall_cost'])
        elif perm_block_result[1] == max_acccepted:
            if perm_block_result[0][perm_block_result[0].keys()[0]]['overall_cost'] < best_cost:
                best_result = config.best_embeddings = dict(perm_block_result[0])
                max_acccepted = config.max_accepted_vnrs = int(perm_block_result[1])
                config.committed_wsn = nx.DiGraph(perm_block_result[2])
                config.active_vns = list(perm_block_result[3])
                best_cost = config.overall_cost = float(perm_block_result[0][perm_block_result[0].keys()[0]]['overall_cost'])
    return best_result

def select_best_(overall_results):
    best_cost = -1
    max_acccepted = -1
    best_result = ()
    while overall_results.empty() is False:
        config.total_operations += 1
        perm_block_result = overall_results.get()
       #print "perm_block_result",perm_block_result
        if int(perm_block_result[1]) != 0:
            if max_acccepted == -1:
                config.best_embeddings = dict(perm_block_result[0])
                best_result = dict(perm_block_result[0])
                config.max_accepted_vnrs = int(perm_block_result[1])
                max_acccepted = int(perm_block_result[1])
                config.committed_wsn = nx.DiGraph(perm_block_result[2])
                config.active_vns = list(perm_block_result[3])
                config.overall_cost = float(perm_block_result[0][perm_block_result[0].keys()[0]]['overall_cost'])
                best_cost = float(perm_block_result[0][perm_block_result[0].keys()[0]]['overall_cost'])
            elif perm_block_result[1] > max_acccepted:
                config.best_embeddings = dict(perm_block_result[0])
                best_result = dict(perm_block_result[0])
                config.max_accepted_vnrs = int(perm_block_result[1])
                max_acccepted = int(perm_block_result[1])
                config.committed_wsn = nx.DiGraph(perm_block_result[2])
                config.active_vns = list(perm_block_result[3])
                config.overall_cost = float(perm_block_result[0][perm_block_result[0].keys()[0]]['overall_cost'])
                best_cost = float(perm_block_result[0][perm_block_result[0].keys()[0]]['overall_cost'])
            elif perm_block_result[1] == max_acccepted:
                if perm_block_result[0][perm_block_result[0].keys()[0]]['overall_cost'] < best_cost:
                    config.best_embeddings = dict(perm_block_result[0])
                    best_result = dict(perm_block_result[0])
                    config.max_accepted_vnrs = int(perm_block_result[1])
                    max_acccepted = int(perm_block_result[1])
                    config.committed_wsn = nx.DiGraph(perm_block_result[2])
                    config.active_vns = list(perm_block_result[3])
                    config.overall_cost = float(perm_block_result[0][perm_block_result[0].keys()[0]]['overall_cost'])
                    best_cost = float(perm_block_result[0][perm_block_result[0].keys()[0]]['overall_cost'])

    return best_result


def select_best(perm_block_results):

    while perm_block_results.empty() is False:
        perm_block_result = overall_results.get()
       #print "perm_block_result", perm_block_result

        if config.current_test_best['acceptance'] < perm_block_result['acceptance']:
            config.current_test_best = perm_block_result
        elif config.current_test_best['acceptance'] == perm_block_result['acceptance']:
            if config.current_test_best['overall_cost'] > perm_block_result['overall_cost']:
                config.current_test_best = perm_block_result
       #print "config.current_test_best",config.current_test_best

    return config.current_test_best


if __name__ == '__main__':

    ppservers = ("*",)
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(0,ppservers=ppservers)

    result_q = multiprocessing.Queue()
    jobs = []
    dir_path = 'tests/50/parallel/'
    input_file_name = 'input_vector_50_7.pickle'
    test_vectors = pickle.load(open(dir_path+input_file_name, 'rb'))
    for test_case in test_vectors:
        #if test_case['iteration'] < 3:
        reinitialize()
        config.nwksize = test_case['nwksize']
        config.numvn = test_case['numvn']
        #if config.numvn == 5:
        config.iteration = test_case['iteration']
        #print ("TEST CASE --nwksize",config.nwksize,"numvn", config.numvn,"iter", config.iteration)
        #if config.iteration == 67:
        vnrs_list = []
        for vnrs in test_case['vnlist']:
            vnrs_list.append(vnrs.convert_to_heuristic())
        generated_wsn = test_case['substrate'].output_for_heuristic()
        wsn_substrate = WSN(config.X,config.Y,generated_wsn)
        adjacencies = wsn_substrate.get_wsn_substrate().adjacency_list()
        config.wsn = wsn_substrate.get_wsn_substrate()
        two_hops_list = wsn_substrate.get_two_hops_list()
        conflicting_links_dict = wsn_substrate.get_conflicting_links()
        min_hops_dict = get_min_hops(config.main_sink)
        shortest_path, path_nodes = [],[]
        config.committed_wsn = nx.DiGraph(config.wsn)
        #config.sp_alg_str = "A*"
        config.sp_alg_str = "Dijkstra"
        config.start = time.time()
        generate_independent_perm_blocks(vnrs_list,result_q)


        for j in jobs:
            j.start()
        overall_results = multiprocessing.Queue()
        for j in jobs:
            res = result_q.get()
            #print "res",res.items()
            overall_results.put(res)
        for j in jobs:
            j.join()
        best = select_best(overall_results)
        if best is not None:
            #print "best", best
            jobs = []
            end = time.time()
            config.proc_time = (end - config.start)
            print("Optimal solution is:","acceptance-",best['acceptance'],"overall_cost-",best['overall_cost'],"processing_time-",config.proc_time)
            config.acceptance = config.max_accepted_vnrs
            generate_output(best)
    write_to_File()
'''
    while exit_flag is True:
       #print("\n---->\n0 - Run permutations\n1 - Embed single VNR\n2 - Plot\n3 - Show wsn resources\n4 - Show/Remove active VNs\n5 - Min hops")
        user_input = input(':')

           #print("1 - use Dijkstra\n2 - use A*")
            if input(":") == 1:
                config.sp_alg_str = "Dijkstra"
            else:
                config.sp_alg_str = "A*"
            run_permutations(vnrs_list)
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
                if removed:
                    config.active_vns.pop(to_remove)
                   #print("VN",to_remove,"was removed." )
                else:
                   #print("VN", choice, "is not in active.")
            else:
                pass
        elif user_input is '5':
            get_min_hops(1)
        elif user_input is '6':
            frm = input('Enter src : ')
            if frm != '':
                #path, length = nx.dijkstra_path(config.wsn, source=int(frm), target=1, weight='weight')
                #print(length)
                #print(path)
                grid = nx.grid_2d_graph(int(frm),int(frm))
                adj_grid = {}
                #for i,n,
               #print(grid.adjacency_list())
                d_nodes = {}
                d_adj = {}
               #print(grid.nodes())
               #print(grid.edges())
                for idx,(r,c) in enumerate(grid.nodes()):
                    d_nodes.update({str((r,c)):idx+1})
               #print(d_nodes.items())
                for idx, ajd in enumerate(grid.adjacency_list()):
                   #print(idx,ajd)
                    n_adj = []
                    for r,c in ajd:
                        n = d_nodes[str((r,c))]
                        n_adj.append(n)
                    d_adj.update({idx:n_adj})
               #print(d_adj.items())
                #print(wsn_substrate.get_adjacency_list().items())
'''
