## Implements the cost function used to compare standalone WSNs to Virtual WSNs
## Includes CAPEX, OPEX  and INEX

import math
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def read_in_files(input_file_l):
    res_dict = {}
    dicts_list = []
    for i, input_file in enumerate(input_file_l):
        # read python dict back from the file
        pkl_file = open(input_file, 'rb')
        print "open-", input_file
        # if choice == 6:
        # result_vectors += pickle.load(pkl_file)
        res_file = pickle.load(pkl_file)
        dicts_list.append(res_file)

        if choice == 0:
            for i, item in enumerate(res_file):
                #print item.keys()
                #mapping_list.append(item['mapping'])
                result_list.append({'iteration':item['iteration'],'committed':item['committed'],'mapping':item['mapping']})
        elif choice == 1:
            for i, item in enumerate(res_file):
                mapping_list.append(item)
        elif choice == 2:
            result_list.extend(res_file)
        elif choice == 3:
            calculate_avg(res_file,i)
        elif choice == 4:
            result_list.extend(res_file)
        #elif choice == 5:
            #plotit(res_file)




def calculate_avg(mydict,idx):

    vn_lease_costs = []
    vn_stndalone_cost = []
    vn_allocations = []
    vn_interferences = []

    for i, item in enumerate(mydict):
        vns = item[5]
        ##print item[4]
        vn_l_c = []
        vn_allocs = []
        vn_ints = []
        for vn in vns:
            vn_l_c.append(vn[4])
            vn_allocs.append(vn[5]/float(substrate_nwk_size))
            vn_ints.append(vn[6]/float(substrate_nwk_size))
        vn_stndalone_cost.append(sum(vn_l_c))
        vn_allocations.append(sum(vn_allocs))
        vn_interferences.append(sum(vn_ints))
        vn_lease_costs.append(item[4])

    results_list[idx] += vn_stndalone_cost
    results_list2[idx] += vn_lease_costs
    results_list3[idx] += vn_allocations
    results_list4[idx] += vn_interferences


def plot_3_subplot(results_list,results_list2,results_list3,results_list4):

    means = []
    stds = []
    for i, rslt in enumerate(results_list):
        # lngth = len(rslt)
        # print 'lngth', lngth
        # for j in range(lngth, 1000):
        #    rslt.append(0.0)
        m = np.mean(rslt)
        s = np.std(rslt, ddof=0)
        means.append(m)
        stds.append(s)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array(means)  # Effectively y = x**2
    e = np.array(stds)
    means2 = []
    stds2 = []
    for i, rslt in enumerate(results_list2):
        print rslt
        # lngth = len(rslt)
        # print 'lngth', lngth
        # for j in range(lngth, 1000):
        #    rslt.append(0.0)

        m = np.mean(rslt)
        s = np.std(rslt, ddof=0)
        means2.append(m)
        stds2.append(s)
    y2 = np.array(means2)  # Effectively y = x**2
    e2 = np.array(stds2)
    # plt.errorbar(x, y, e, linestyle='-', marker='.', ecolor='y', label='avg standalone cost')
    # plt.errorbar(x, y2, e2, linestyle='-', marker='.', ecolor='r', label='avg leasing cost')
    # plt.xlim([0, 9])
    # plt.ylim([-0.1, 1.1])
    # plt.yscale('log')
    # plt.legend(loc='best')
    # plt.show()

    #  pad acceptance rate lists with zeros
    means3 = []
    stds3 = []
    for i, rslt in enumerate(results_list3):
        # lngth = len(rslt)
        # print 'lngth', lngth
        # for j in range(lngth, 1000):
        #    rslt.append(0.0)
        m = np.mean(rslt)
        s = np.std(rslt, ddof=0)
        means3.append(m)
        stds3.append(s)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y3 = np.array(means3)  # Effectively y = x**2
    e3 = np.array(stds3)
    means4 = []
    stds4 = []
    for i, rslt in enumerate(results_list4):
        print rslt
        # lngth = len(rslt)
        # print 'lngth', lngth
        # for j in range(lngth, 1000):
        #    rslt.append(0.0)

        m = np.mean(rslt)
        s = np.std(rslt, ddof=0)
        means4.append(m)
        stds4.append(s)
    y4 = np.array(means4)  # Effectively y = x**2
    e4 = np.array(stds4)

    #x1 = np.linspace(0.0, 5.0)
    #x2 = np.linspace(0.0, 2.0)

    #y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
   # y2 = np.cos(2 * np.pi * x2)
###########################
    plt.subplot(2, 1, 1)
    plt.errorbar(x, y, e, linestyle='-', marker='^', color='r', ecolor='r', label='leasing cost')#.plot(x, y, 'ko-')
    plt.errorbar(x, y2, e2, linestyle='-', marker='^', color='g', ecolor='g', label='standalone cost')
    plt.title('Network utilization and cost for 150 nodes substrate', fontsize=17)
    plt.ylabel('Cost', fontsize=15)
    plt.legend(loc='upper left')
    plt.subplots_adjust(left=0.05, bottom=0.07)
    plt.xlim([0, 9])
    plt.subplot(2, 1, 2)
   # plt.plot(x, y2, 'r.-')
    plt.errorbar(x, y3, e3, linestyle='-.', marker='.', color='b', ecolor='b', label='active elements')  # .plot(x, y, 'ko-')
    plt.errorbar(x, y4, e4, linestyle='-.', marker='.', color='y',ecolor='y', label='interference')
    plt.xlabel('Number of virtual network request(s)', fontsize=15)
    plt.ylabel('Network utilization', fontsize=15)
    plt.legend(loc='upper left')
    plt.xlim([0, 9])
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.07)
    plt.show()

def plot_2_y(results_list,results_list2,results_list3,results_list4):
    means = []
    stds = []
    for i, rslt in enumerate(results_list):
        # lngth = len(rslt)
        # print 'lngth', lngth
        # for j in range(lngth, 1000):
        #    rslt.append(0.0)
        m = np.mean(rslt)
        s = np.std(rslt, ddof=0)
        means.append(m)
        stds.append(s)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array(means)  # Effectively y = x**2
    e = np.array(stds)
    means2 = []
    stds2 = []
    for i, rslt in enumerate(results_list2):
        print rslt
        # lngth = len(rslt)
        # print 'lngth', lngth
        # for j in range(lngth, 1000):
        #    rslt.append(0.0)

        m = np.mean(rslt)
        s = np.std(rslt, ddof=0)
        means2.append(m)
        stds2.append(s)
    y2 = np.array(means2)  # Effectively y = x**2
    e2 = np.array(stds2)
    #plt.errorbar(x, y, e, linestyle='-', marker='.', ecolor='y', label='avg standalone cost')
    #plt.errorbar(x, y2, e2, linestyle='-', marker='.', ecolor='r', label='avg leasing cost')
    #plt.xlim([0, 9])
    # plt.ylim([-0.1, 1.1])
    # plt.yscale('log')
    #plt.legend(loc='best')
    #plt.show()

    #  pad acceptance rate lists with zeros
    means3 = []
    stds3 = []
    for i, rslt in enumerate(results_list3):
        # lngth = len(rslt)
        # print 'lngth', lngth
        # for j in range(lngth, 1000):
        #    rslt.append(0.0)
        m = np.mean(rslt)
        s = np.std(rslt, ddof=0)
        means3.append(m)
        stds3.append(s)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y3 = np.array(means3)  # Effectively y = x**2
    e3 = np.array(stds3)
    means4 = []
    stds4 = []
    for i, rslt in enumerate(results_list4):
        print rslt
        # lngth = len(rslt)
        # print 'lngth', lngth
        # for j in range(lngth, 1000):
        #    rslt.append(0.0)

        m = np.mean(rslt)
        s = np.std(rslt, ddof=0)
        means4.append(m)
        stds4.append(s)
    y4 = np.array(means4)  # Effectively y = x**2
    e4 = np.array(stds4)
    #plt.errorbar(x, y, e, linestyle='-', marker='.', ecolor='y', label='avg allocation')
    #plt.errorbar(x, y2, e2, linestyle='-', marker='.', ecolor='r', label='avg interference')
    #plt.xlim([0, 9])
    # plt.ylim([-0.1, 1.1])
    # plt.yscale('log')
    #plt.legend(loc='best')
    #plt.show()


##################################

    lines = ['b*--','b*-','r*--','r*-']
    labels = ('standalone cost', 'leasing cost', 'allocations','interferences')
    fig, ax1 = plt.subplots()
    lines1 = ax1.errorbar(x, y, e , linestyle='--',marker='.',color='y')#.plot(x, y, 'b*--')
    ax1.set_xlabel('number of virtual network request(s)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('cost', color='b')
    ax1.tick_params('y', colors='b')
    lines2 = ax1.errorbar(x, y2, e2 )#.plot(x, y2, 'b*-')


    ax2 = ax1.twinx()
    lines3, = ax2.plot(x, y3, 'r*--')
    lines4, = ax2.plot(x, y4, 'r*-')
    ax2.set_ylabel('utilization', color='r')
    ax2.tick_params('y', colors='r')
    ax1.legend((lines1, lines2, lines3, lines4), labels, loc='upper left')
    #ax2.legend(loc='upper left')
    #plt.figlegend( , labels, loc = 'lower center', ncol=5, labelspacing=0. )
    fig.tight_layout()
    plt.show()


def plot_utilization(results_list3,results_list4):
    #  pad acceptance rate lists with zeros
    means = []
    stds = []
    for i, rslt in enumerate(results_list3):
        #lngth = len(rslt)
        #print 'lngth', lngth
        #for j in range(lngth, 1000):
        #    rslt.append(0.0)
        m = np.mean(rslt)
        s = np.std(rslt, ddof=0)
        means.append(m)
        stds.append(s)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array(means)  # Effectively y = x**2
    e = np.array(stds)
    means2 = []
    stds2 = []
    for i, rslt in enumerate(results_list4):
        print rslt
        #lngth = len(rslt)
        #print 'lngth', lngth
        #for j in range(lngth, 1000):
        #    rslt.append(0.0)

        m = np.mean(rslt)
        s = np.std(rslt, ddof=0)
        means2.append(m)
        stds2.append(s)
    y2 = np.array(means2)  # Effectively y = x**2
    e2 = np.array(stds2)
    plt.errorbar(x, y, e, linestyle='-', marker='.', ecolor='y', label='avg allocation')
    plt.errorbar(x, y2, e2, linestyle='-', marker='.', ecolor='r', label='avg interference')
    plt.xlim([0, 9])
    # plt.ylim([-0.1, 1.1])
    # plt.yscale('log')
    plt.legend(loc='best')
    plt.show()



    fig, ax1 = plt.subplots()
    t = np.arange(0.01, 10.0, 0.01)
    s1 = np.exp(t)
    ax1.plot(x, y, 'b-')
    ax1.set_xlabel('time (s)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('exp', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    s2 = np.sin(2 * np.pi * t)
    ax2.plot(x, y2, 'r.')
    ax2.set_ylabel('sin', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()


def plot_mean_costs(results_list,results_list2):
    #  pad acceptance rate lists with zeros
    means = []
    stds = []
    for i, rslt in enumerate(results_list):
        #lngth = len(rslt)
        #print 'lngth', lngth
        #for j in range(lngth, 1000):
        #    rslt.append(0.0)
        m = np.mean(rslt)
        s = np.std(rslt, ddof=0)
        means.append(m)
        stds.append(s)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array(means)  # Effectively y = x**2
    e = np.array(stds)
    means2 = []
    stds2 = []
    for i, rslt in enumerate(results_list2):
        print rslt
        #lngth = len(rslt)
        #print 'lngth', lngth
        #for j in range(lngth, 1000):
        #    rslt.append(0.0)

        m = np.mean(rslt)
        s = np.std(rslt, ddof=0)
        means2.append(m)
        stds2.append(s)
    y2 = np.array(means2)  # Effectively y = x**2
    e2 = np.array(stds2)
    plt.errorbar(x, y, e, linestyle='-', marker='.', ecolor='y', label='avg standalone cost')
    plt.errorbar(x, y2, e2, linestyle='-', marker='.', ecolor='r', label='avg leasing cost')
    plt.xlim([0, 9])
    # plt.ylim([-0.1, 1.1])
    # plt.yscale('log')
    plt.legend(loc='best')
    plt.show()



def write_out_file():
    try:
        with open(output_file_name, 'w') as handle:
            print 'Writing result_vectors of length ', len(result_list)
            # print result_vectors
            pickle.dump(result_list, handle)
    except Exception as e:
        print (e)

def get_discount_rate(quantity):

    base_discount = 0.03
    base_quantity = 10
    discount = base_discount

    for i in range(0,(quantity/base_quantity)):
        base_discount = base_discount * 0.9
        discount += base_discount
        #print discount

    return discount

def get_qos(current_vnr):

    if current_vnr[3]['latency'] >= 666:
        return 1
    if current_vnr[3]['latency'] <= 333:
        return 1.2
    else:
        return 1.1

def get_qoi(current_vnr):

    if current_vnr[1][1] >= .66:
        return 1
    if current_vnr[1][1] <= .33:
        return 1.2
    else:
        return 1.1

# calculates the maintenance cost per substrate network per hour
def get_maintenance_cost(nwk_size):
    maintenance_cost = (nwk_size * ( deployment_hourly_rate / 2.0 + battery_charging_price))  / maintenance_period
    print "mntnc per hour per SN", maintenance_cost #(nwk_size * ( deployment_hourly_rate / 2 + battery_charging_price))  / maintenance_period
    return maintenance_cost


def calculate_capex(nwk_size):
    num_bs_required = int(math.ceil(float(nwk_size)/50))  # 1 bs per 50 nodes required
    size_increment = int(math.ceil(float(nwk_size)/10))
    hw_price = (nwk_size * (node_unit_price + (battery_unit_price * 2))) + (base_station_unit_price * num_bs_required)
    equipment_price = equipment_unit_price * size_increment
    development_price = development_hours * development_hourly_rate
    deployment_price = deployment_hourly_rate * nwk_size
    capex = (hw_price + equipment_price + deployment_price + development_price) * (1.0 - get_discount_rate(nwk_size))
    print hw_price, equipment_price, deployment_price, development_price
    return capex

def calculate_vn_capex(nwk_size):
    num_bs_required = int(math.ceil(float(nwk_size)/50))  # 1 bs per 50 nodes required
    size_increment = int(math.ceil(float(nwk_size)/10))
    hw_price = (nwk_size * (node_unit_price + (battery_unit_price * 2))) + (base_station_unit_price * num_bs_required)
    equipment_price = equipment_unit_price * size_increment
    development_price = (development_hours * development_hourly_rate) / 3
    deployment_price = deployment_hourly_rate * nwk_size
    capex = (hw_price + equipment_price + deployment_price + development_price) * (1.0 - get_discount_rate(nwk_size))
    print capex / num_lifecycles, hw_price, equipment_price, deployment_price, development_price
    return capex

def calculate_opex2(i, item):
        opex = 0.0
    #print "vnr_list is",type(vnr_list)
    #for item in vnr_list:
        iteration = item['iteration']
        committed = item['committed']
        mapping = item['mapping']
        sp_list = []
        total_vn_allocation = 0.0
        for vn in mapping:
            quota = vn['load']
            path = vn['shortest_path']
            nodes = vn['path_nodes']
            sp_list.append(path)
            links = vn['links']
            total_vn_allocation += vn_interference_ratio(links,path,nodes) / substrate_nwk_size * maintenance_cost
            vnr = vn['vnr']
            opex += get_sensing_cost(quota,vnr)
            opex += get_radio_cost(links, path,vnr)
        print "OPEX for iteration",iteration,"is",opex
        print total_vn_allocation
        #unit_cost = calculate_unit_cost(CAPEX, opex, committed)
        #print "unit cost is", unit_cost
        return opex

def calculate_opex(i, cycle):
    opex = 0.0
    iteration = cycle['iteration']
    committed = cycle['committed']
    mapping = cycle['mapping']
    sp_list = []
    total_vn_allocation = 0.0
    total_interference = 0.0
    for vn in mapping:
        quota = vn['load']
        path = vn['shortest_path']
        nodes = vn['path_nodes']
        links = vn['links']
        vnr = vn['vnr']
        current_allocation, current_interference,total_capacity,active_nodes = vn_interference_ratio(links, path, nodes)
        total_vn_allocation += current_allocation
        total_interference += current_interference
        opex += current_allocation / substrate_nwk_size * maintenance_cost_per_hour * avg_vn_lifetime * get_qos(vnr) * get_qoi(vnr)
        #vnr = vn['vnr']
        #opex += get_sensing_cost(quota, vnr)
        #opex += get_radio_cost(links, path, vnr)
    #opex += total_vn_allocation
    print "OPEX for iteration", iteration, "is", opex
    print total_interference, total_interference/substrate_nwk_size,total_vn_allocation, total_vn_allocation/substrate_nwk_size
    # unit_cost = calculate_unit_cost(CAPEX, opex, committed)
    # print "unit cost is", unit_cost
    return opex, total_interference, total_vn_allocation


def vn_interference_ratio2(links,path,nodes):

    total_capacity = 0.0
    total_allocation = 0.0
    interfering_links = 0.0
    active_links = 0.0
    link_count = 0
    #print "get interference ratio"
    for u, v, d in links.edges_iter(data=True):
        #print u,v,d['load']
        #print links.edges(u)
        link_count += 1
        total_capacity += 1.0

        total_allocation += links[u][v]['load'] / 100.0

        if (u,v) in path or (u,v) in path:
            #print path
            #print (u,v),d['load']/100.0
            active_links += links[u][v]['load'] / 100.0
        else:
            interfering_links  += links[u][v]['load'] / 100.0

    print active_links,interfering_links,total_allocation,total_capacity
    print ""

def vn_interference_ratio(links,path,nodes):
    total_capacity = 0.0
    total_allocation = 0.0
    interference = 0.0
    active_nodes = 0
    link_count = 0
    node_count = len(links)
    #print node_count
    #print "get interference ratio"
   # print nodes
    for n,d in links.nodes_iter(data=True):
        total_capacity += 1.0
        edges = links.edges(n)

        if n in nodes:
            active_nodes += 1.0
            max_capacity = 0.0
            for l in edges:
                #print l[0],l[1],"pathn",links[l[0]][l[1]]['load']
                current_capacity = links[l[0]][l[1]]['load'] / 100.0
                if max_capacity < current_capacity:
                    max_capacity = current_capacity
            #print "-cc",current_capacity
            total_allocation += max_capacity
        else:
            max_capacity = 0.0
            for l in edges:
                #print l[0],l[1],"npath", links[l[0]][l[1]]['load']
                current_capacity = links[l[0]][l[1]]['load'] / 100.0
                if max_capacity < current_capacity:
                    max_capacity = current_capacity
            interference += max_capacity
    #print active_nodes,total_capacity,total_allocation,interference
    return total_allocation,interference,total_capacity,active_nodes

def get_radio_cost(links,path,vnr):
    radio_cost = 0
    for l in path:
        #print l,links[l[0]][l[1]]['load']/100.0
        radio_cost += (links[l[0]][l[1]]['load']/100.0) * get_qos(vnr) * maintenance_cost #get_maintenance_cost(substrate_nwk_size,utilization)
        radio_cost += (links[l[1]][l[0]]['load']/100.0) * get_qos(vnr) * maintenance_cost #get_maintenance_cost(substrate_nwk_size,utilization)
    return radio_cost

def get_sensing_cost(quota,vnr):
    sensing_cost = (quota/100.0) * get_qoi(vnr) * maintenance_cost #get_maintenance_cost(substrate_nwk_size,utilization)
    return sensing_cost

def calculate_unit_cost(capex,opex,committed):

    unit_cost = (capex + opex) / (utilization)


def calculate_vn_cost(price, cycle, total_usage):
    iteration = cycle['iteration']
    mapping = cycle['mapping']
    total_vn_allocation = 0.0
    total_interference = 0.0
    total_vn_price = 0.0
    current_vn_tuple = ()
    current_vn_list = []
    for vn in mapping:
        current_vn_price = 0.0
        path = vn['shortest_path']
        nodes = vn['path_nodes']
        links = vn['links']
        vnr = vn['vnr']
        current_allocation, current_interference, total_capacity,active_nodes = vn_interference_ratio(links, path, nodes)
        total_vn_allocation += current_allocation
        total_interference += current_interference

        current_vn_price = current_allocation / total_usage * price # leasing price
        total_vn_price += current_vn_price
        print ";;;;;;;;;;;;;;;;;;;;"
        print "current_vn_price",current_vn_price
        print ""
        capex = calculate_vn_capex(len(nodes))
        print "vn capex",capex
        opex = current_allocation / len(nodes) * maintenance_cost_per_hour * avg_vn_lifetime  # * get_qos(vnr) * get_qoi(vnr)
        print "vn opex", opex

        vn_total_cost = opex + capex
        vn_cycle_cost = opex + capex / num_lifecycles
        print "standalone total cost",vn_total_cost
        print "standalone cost per cycle",vn_cycle_cost
        print ""
        current_vn_list.append((current_vn_price,capex,opex,vn_total_cost,vn_cycle_cost,current_allocation, current_interference, total_capacity,active_nodes))
    result_list.append((CAPEX, OPEX, CAPOPEX, interference_cost, current_price,current_vn_list))
    #opex += total_vn_allocation
    print "total vn price for iteration", iteration, "is", total_vn_price
    #print total_interference, total_interference/substrate_nwk_size,total_vn_allocation, total_vn_allocation/substrate_nwk_size
    # unit_cost = calculate_unit_cost(CAPEX, opex, committed)
    # print "unit cost is", unit_cost
    #return opex, total_interference

if __name__ == '__main__':

        choice = 1

    #for vnr_num in range(1, 8):
        substrate_nwk_size = 150
        #utilization = 0.3

        vnr_num = 8
        current_vn_num = vnr_num
        input_file_list = []
        vnr_list = []
        mapping_list = []
        result_list = []
        begin = 0
        stop = 1000
        increm = 200

        results_list = [[], [], [], [], [], [], [], []]
        results_list2 = [[], [], [], [], [], [], [], []]
        results_list3 = [[], [], [], [], [], [], [], []]
        results_list4 = [[], [], [], [], [], [], [], []]
        results_list5 = [[], [], [], [], [], [], [], []]
        results_list6 = [[], [], [], [], [], [], [], []]
        results_list7 = [[], [], [], [], [], [], [], []]
        results_list8 = [[], [], [], [], [], [], [], []]

        dir_path = '/media/roland/Docker/ftp/results/test2/' + str(vnr_num) + '/'
        if choice == 0:
            if vnr_num == 9:
                for i in range(begin, stop, increm):
                    print i, stop, increm
                    input_filename = dir_path + str(begin) + '_' + str(begin+increm) + '_input_vector_' + str(substrate_nwk_size) + '_' + '8.pickle-'
                    print input_filename
                    input_file_list.append(input_filename)
                    begin += increm
                    #read_in_files(input_file_list)
            else:
                #input_filename = dir_path + '0_1000_input_vector_' + str(substrate_nwk_size) + '_' + str(vnr_num) + '.pickle-'

                input_filename = dir_path + str(begin) + '_' + str(begin + increm) + '_input_vector_' + str(substrate_nwk_size) + '_' + '8.pickle-'
                print input_filename
                input_file_list.append(input_filename)
                #output_file_name = './test/' + str(nwk_size) + '_early_solution_' + str(vnr_num) + '.pickle_' + str(begin) + '_' + str(begin + increm) + '-'
            output_file_name = input_filename + 'cost'
            read_in_files(input_file_list)
            write_out_file()
        elif choice == 1:
            input_filename = dir_path + '800_1000_input_vector_' + str(substrate_nwk_size) + '_' + str(vnr_num) + '.pickle-cost'
            #input_filename = dir_path + '0_1000_input_vector_' + str(substrate_nwk_size) + '_' + str(vnr_num) + '.pickle-'
            print input_filename
            input_file_list.append(input_filename)
            read_in_files(input_file_list)
            output_file_name = input_filename + '-resultsss'
        elif choice == 2:
            for i in range(begin, stop, increm):
                print i, stop, increm
                input_filename = dir_path + str(begin) + '_' + str(begin + increm) + '_input_vector_' + str(substrate_nwk_size) + '_' + '8.pickle-cost-results'
                input_file_list.append(input_filename)
                begin += increm
            output_file_name = dir_path + '0_1000_input_vector_' + str(substrate_nwk_size) + '_' + '8.pickle-cost-results'
            read_in_files(input_file_list)
            write_out_file()

        elif choice == 3:
            for vnr_num in range(1, 9):
                current_vn_num = vnr_num
                dir_path = '/media/roland/Docker/ftp/results/test2/' + str(vnr_num) + '/'
                if vnr_num < 8:
                    input_filename = dir_path + '0_1000_input_vector_' + str(substrate_nwk_size) + '_' + str(vnr_num) + '.pickle--results'
                else:
                    input_filename = dir_path + '0_1000_input_vector_' + str(substrate_nwk_size) + '_' + str(vnr_num) + '.pickle-cost-results'
                print input_filename
                input_file_list.append(input_filename)
            output_file_name = dir_path + str(vnr_num) + '-avg.pickle'
            read_in_files(input_file_list)
            #plot_mean_costs(results_list, results_list2)
            #plot_utilization(results_list3, results_list4)
            #plot_2_y(results_list, results_list2, results_list3, results_list4)
            plot_3_subplot(results_list, results_list2, results_list3, results_list4)
            result_list.append(results_list)
            result_list.append(results_list2)
            result_list.append(results_list3)
            result_list.append(results_list4)
            #write_out_file()
        elif choice == 4:
            for i in range(1,4):
                print i, stop, increm
                input_filename = dir_path + str(i) + '-avg.pickle'
                input_file_list.append(input_filename)
            output_file_name = dir_path + '-avg.pickle'
            read_in_files(input_file_list)
            write_out_file()
        elif choice == 5:
            input_filename = dir_path + '-avg.pickle'
            input_file_list.append(input_filename)
            #output_file_name = dir_path + '-avg.pickle'
            read_in_files(input_file_list)
            #write_out_file()

        discount_rate = get_discount_rate(substrate_nwk_size)
        substrate_nwk_size_increment =  substrate_nwk_size / 10
        profit_margin = 0.25

        # CAPEX params ##################################################
        depreciation_period = 17520.0 # hours or 730 days or 2 years
        node_unit_price = 100.0
        battery_unit_price = 10.0
        battery_charging_price = 5.0
        base_station_unit_price = 1000.0
        equipment_unit_price = 150.0
        deployment_hourly_rate = 25.0
        development_hourly_rate = 30.0
        development_hours = 504.0 / 3 # 3 months


        total_hw_price = (substrate_nwk_size * (node_unit_price + (battery_unit_price * 2))) + (base_station_unit_price * (substrate_nwk_size / 50))
        total_equipment_price = equipment_unit_price * substrate_nwk_size_increment
        total_development_price = development_hours * development_hourly_rate
        total_deployment_price = deployment_hourly_rate * substrate_nwk_size

        # OPEX params ##################################################
        # maintenance params
        energy = 2500.0
        consumption = 30.0
        XFACTOR = 0.75 # adjustment to account for external environmental factors (i.e. inefficiency)
        maintenance_period = energy / consumption * XFACTOR
        STBRATE = 1.1 # standby amortization / energy consumption
        maintenance_cost_per_hour = get_maintenance_cost(substrate_nwk_size)

        avg_vn_lifetime = 720.0 # hours or 30 days
        num_lifecycles = depreciation_period / avg_vn_lifetime

        # INEX params ##################################################
        interference_waste = 0.0



        print "discount",get_discount_rate(substrate_nwk_size)
        CAPEX = calculate_capex(substrate_nwk_size)
        CCAPEX = CAPEX / num_lifecycles # proportional CAPEX Straight Line Depreciation with 24 30 days cycles for 2 years
        print "CAPEX", CAPEX, "CCAPEX",CCAPEX
        print "QoS weight",get_qos((1000, (76, 0.40122662225566352), 0, {'load': 1, 'latency': 332, 'plr': 50.0}))
        print "QoI weight",get_qoi((1000, (76, 0.140122662225566352), 0, {'load': 1, 'latency': 332, 'plr': 50.0}))
        #print "maintenance", maintenance_cost


        for i,vnr_list in enumerate(mapping_list):
            print ""
            OPEX,interference_waste,total_usage = calculate_opex(i,vnr_list)
            CAPOPEX = ((OPEX + CCAPEX * (1.0 - get_discount_rate(substrate_nwk_size))) + CCAPEX)
            print "Total CAPOPEX per VN lifecycle (30 days)",CAPOPEX

            profit = CAPOPEX * profit_margin
            print "Total profit with 25% margin", profit
            #print "Hourly CAPOPEX", CAPOPEX / substrate_nwk_size
            interference_cost = interference_waste/substrate_nwk_size * profit
            print "interference cost",interference_cost
            current_price = interference_cost + profit + CAPOPEX
            print "current total price", current_price

            print "-----------------------------------"
            calculate_vn_cost(current_price, vnr_list, total_usage)
            print "-----------------------------------"
#        write_out_file()
        #for i, vnr_list in enumerate(mapping_list):

