import networkx as nx
from combinationIterator import CombinationIterator as ci
VN_l1 = nx.DiGraph()
counter_value = 0
plot_counter = 0
avoid = []
penalized_list= []
penalize = dict()
failed_links_list = []
feasible = False
has_embedding = False
VWSNs = []
all_embeddings = []
successful_embeddings = {}
current_mappings = {}
best_embeddings = {}
vnr_list = []
allocated_links_load = dict()
allocated_links_weight = dict()
reduced_adj = dict()
link_weights = dict()
two_hops = dict()

current_wsn = nx.DiGraph()
wsn_for_this_perm = nx.DiGraph()


