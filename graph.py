import random, math, json
from collections import Counter
import networkx as nx
import community
import numpy as np

class Graph():
    def __init__(self, 
                 num_nodes=1000,
                 num_cores=1,
                 intra_core_connectivity=0.5,
                 core_connectivity=0.3,
                 add_nodes_random=0.0,
                 add_nodes_popularity=1.0,
                 connect_cores_directly=0.8,
                 connect_second_neighbours=1.0,
                 connect_random=1.0):
        self.num_nodes = num_nodes
        self.num_cores = num_cores
        self.intra_core_connectivity = intra_core_connectivity
        self.core_connectivity = core_connectivity
        self.add_nodes_random = add_nodes_random
        self.add_nodes_popularity = add_nodes_popularity
        self.connect_cores_directly = connect_cores_directly
        self.connect_second_neighbours = connect_second_neighbours
        self.connect_random = connect_random
        self.nodeids = set()
        self.interactions = {}
        self.reverse_interactions = {}
        self.mapping = []
        self.nx_graph = None
        self.neighbours = {}
        self.in_degree = Counter()
        self.out_degree = Counter()
        self.communities = None
        self.clusters = {}
        self.node_attrs = {}
        self.attr_names = []
        self.attr_types = []
        self.make_graph()

    def exists_nodeid(self, nodeid):
        if nodeid in self.nodeids:
            return True
        return False

    def get_incoming_weights(self, nodeid):
        ret = []
        if nodeid not in self.reverse_interactions:
            return ret
        for source, weight in self.reverse_interactions[nodeid].items():
            ret.append([source, weight])
        return ret

    def get_outgoing_weights(self, nodeid):
        ret = []
        if nodeid not in self.interactions:
            return ret
        for target, weight, in self.interactions[nodeid].items():
            ret.append([target, weight])
        return ret

    def get_neighbours(self, nodeid):
        return self.neigbours[nodeid]

    def get_second_neighbours(self, nodeid):
        second_neighbours = set()
        for nodeid in self.neighbours[nodeid]:
            second_neighbours.update(self.neighbours[nodeid])
        return second_neighbours

    def add_edge(self, source, target, weight=1):
        # Register nodeids
        self.nodeids.add(source)
        self.nodeids.add(target)
        # Add item to interactions
        if source not in self.interactions:
            self.interactions[source] = Counter()
        self.interactions[source][target] += weight
        # Add item to reverse interactions
        if target not in self.reverse_interactions:
            self.reverse_interactions[target] = Counter()
        self.reverse_interactions[target][source] += weight
        # Record in-degree, out-degree
        self.in_degree[target] += weight
        self.out_degree[source] += weight
        # Record neigbours
        if source not in self.neighbours:
            self.neighbours[source] = set()
        self.neighbours[source].add(target)
        if target not in self.neighbours:
            self.neighbours[target] = set()
        self.neighbours[target].add(source)
        # Add mapping item
        self.mapping.append((source, target, weight))

    def save_graph_csv(self, filename):
        with open(filename, "w") as f:
            f.write("Source,Target,Weight\n")
            for source, targets in self.interactions.items():
                for target, count in targets.items():
                    f.write(str(source)+","+str(target)+","+str(count)+"\n")

    def make_mapping(self):
        mapping = []
        for source, targets in self.interactions.items():
            for target, count in targets.items():
                mapping.append((source, target, count))
        self.mapping = mapping

    def make_graph(self):
        # Create a starting core network
        print("Creating core network with " + str(self.num_cores) + " cores.")
        cores = {}
        ci = 0
        for cn in range(self.num_cores):
            cores[cn] = []
            subset = int(self.num_nodes/(20*self.num_cores))
            sn = int(subset + random.randint(1, subset))
            for source in range(ci, sn+ci):
                cores[cn].append(source)
                max_connections = int(sn * self.intra_core_connectivity)
                if max_connections > 0:
                    num_connections = random.randint(1, max_connections)
                    connections = random.sample(range(ci, sn+ci), num_connections)
                    for target in connections:
                        self.add_edge(source, target)
            ci += sn

        # Create some connections between cores
        if self.num_cores > 1:
            num_connections = int(np.mean([len(x) for l, x in cores.items()]) * self.core_connectivity)
            for _ in range(num_connections):
                core_selection = random.sample([l for l, x in cores.items()], 2)
                if random.random() < self.connect_cores_directly:
                    source = random.choice(cores[core_selection[0]])
                    target = random.choice(cores[core_selection[1]])
                    self.add_edge(source, target)
                else:
                    source = ci
                    target = random.choice(cores[core_selection[0]])
                    self.add_edge(source, target)
                    target = random.choice(cores[core_selection[1]])
                    self.add_edge(source, target)
                    ci += 1

        # Add further nodes to existing graph
        print("Adding extra nodes by popularity")
        for i in range(int(self.num_nodes * self.add_nodes_popularity)):
            ci = len(self.interactions)
            categorical = []
            for item, count in self.in_degree.items():
                categorical.extend([item]*count)
            target = random.choice(categorical)
            self.add_edge(ci, target)

        print("Adding extra nodes by random selection")
        for i in range(int(self.num_nodes * self.add_nodes_random)):
            ci = len(self.interactions)
            target = random.choice(range(ci))
            self.add_edge(ci, target)

        # Add extra random connections between nearby existing nodes
        print("Adding extra connections fron existing nodes to their second neighbours")
        for _ in range(int(self.num_nodes * self.connect_second_neighbours)):
            source = random.randint(1, self.num_nodes)
            first_neighbours = self.neighbours[source]
            second_neighbours = self.get_second_neighbours(source)
            second_neighbours.difference_update(first_neighbours)
            if len(second_neighbours) > 0:
                target = random.choice(list(second_neighbours))
                self.add_edge(source, target)

        print("Adding extra connections fron existing nodes random other nodes")
        for _ in range(int(self.num_nodes * self.connect_random)):
            source, target = random.sample(range(self.num_nodes), 2)
            self.add_edge(source, target)

        print("Making nx graph")
        self.make_nx_graph()
        print("Getting communities")
        self.make_communities()

        for nodeid in self.nodeids:
            self.node_attrs[nodeid] = []

        for label, names in self.clusters.items():
            for name in names:
                self.node_attrs[name].append(label)
        self.attr_names.append("community")
        self.attr_types.append("integer")

    def make_nx_graph(self):
        self.nx_graph = nx.Graph()
        self.nx_graph.add_weighted_edges_from(self.mapping)

    def make_communities(self):
        self.communities = community.best_partition(self.nx_graph)
        for node, mod in self.communities.items():
            if mod not in self.clusters:
                self.clusters[mod] = []
            self.clusters[mod].append(node)

    def print_community_stats(self, num=5):
        for cluster_index, cluster_nodes in self.clusters.items():
            msg = ""
            msg += "Community: " + str(cluster_index)
            msg += " members: " + str(len(cluster_nodes))
            msg += " "
            cluster_indegree = Counter()
            for x in cluster_nodes:
                cluster_indegree[x] = self.in_degree[x]
            msg += str(cluster_indegree.most_common(num))
            print(msg)

    def write_gexf(self, filename):
        nodes = sorted(list(set([m[0] for m in self.mapping]).union(set([m[1] for m in self.mapping]))))
        vocab = {}
        vocab_inv = {}
        for index, node in enumerate(nodes):
            label = "n" + str(index)
            vocab[node] = label
            vocab_inv[label] = node

        with open(filename, "w") as f:
            header = ""
            with open("gexf_header.txt", "r") as g:
                for line in g:
                    header += line
            f.write(header + "\n")

            if len(self.attr_names) > 0:
                f.write("\t\t<attributes class=\"node\">\n")
                for index, name in enumerate(self.attr_names):
                    f.write("\t\t\t<attribute id=\"" + str(index) + "\" title=\"" + str(name) + "\" type=\"" + str(self.attr_types[index]) + "\"/>\n")
                f.write("\t\t</attributes>\n")


            f.write("\t\t<nodes>\n")
            indent = '\t\t\t'
            for index, node in enumerate(nodes):
                label = vocab[node]
                entry = indent+ "<node id=\"" + str(label) + "\" label=\"" + str(node) + "\">\n"
                if len(self.attr_names) > 0:
                    entry += indent + "\t<attvalues>\n"
                    for index, name in enumerate(self.attr_names):
                        a = self.node_attrs[node]
                        entry += indent + "\t\t<attvalue for=\"" + str(index) + "\" value=\"" + str(a[index]) + "\"/>\n"
                    entry += indent + "\t</attvalues>\n"
                entry += indent + "</node>\n"
                f.write(entry)
            f.write("\t\t</nodes>\n")
            f.write("\t\t<edges>\n")
            for m in self.mapping:
                sid = vocab[m[0]]
                tid = vocab[m[1]]
                w = m[2]
                entry = indent + "<edge source=\"" + str(sid) + "\" target=\"" + str(tid) + "\" weight=\"" + str(w) + "\"/>\n"
                f.write(entry)
            f.write("\t\t</edges>\n")
            f.write("\t</graph>\n")
            f.write("</gexf>\n")

    def rebuild_graph(self, interactions):
        self.nodeids = set()
        self.interactions = {}
        self.reverse_interactions = {}
        self.mapping = []
        self.nx_graph = None
        self.neighbours = {}
        self.in_degree = Counter()
        self.out_degree = Counter()
        self.communities = None
        self.clusters = {}
        for source, targets in interactions.items():
            for target, count in targets.items():
                self.add_edge(source, target, count)
        self.make_nx_graph()
        self.make_communities()

    def move_node(self, nodeid):
        first_neighbours = self.neighbours[nodeid]
        second_neighbours = self.get_second_neighbours(nodeid)
        num_to_change = min(1, int(len(first_neighbours)*0.1))
        to_break = random.sample(first_neighbours, num_to_change)
        to_make = random.sample(second_neighbours, num_to_change)
        ret = {}
        for source, targets in self.interactions.items():
            for target, count in targets.items():
                if source==nodeid and target in to_break:
                    continue
                if target==nodeid and source in to_break:
                    continue
                if source not in ret:
                    ret[source] = Counter()
                ret[source][target] = count
        for target in to_make:
            if source not in ret:
                ret[source] = Counter()
            ret[source][target] += 1
        self.rebuild_graph(ret)



