import random, math, json, sys
from collections import Counter
import networkx as nx
import community
import numpy as np

class Graph():
    def __init__(self, 
                 num_nodes=1000,
                 num_cores=1,
                 intra_core_connectivity=0.3,
                 core_connectivity=0.7,
                 add_nodes_random=0.4,
                 add_nodes_popularity=1.4,
                 popularity_cutoff=1.0,
                 connect_cores_directly=0.2,
                 connect_second_neighbours=1.5,
                 move_nodes_second_neighbour=0.5,
                 connect_random=0.4,
                 config=None):
        self.num_nodes = num_nodes
        self.num_cores = num_cores
        self.intra_core_connectivity = intra_core_connectivity
        self.core_connectivity = core_connectivity
        self.add_nodes_random = add_nodes_random
        self.add_nodes_popularity = add_nodes_popularity
        self.popularity_cutoff = popularity_cutoff
        self.connect_cores_directly = connect_cores_directly
        self.connect_second_neighbours = connect_second_neighbours
        self.move_nodes_second_neighbour = move_nodes_second_neighbour
        self.connect_random = connect_random
        self.config = None
        if config is not None:
            valid = self.validate_config(config)
            if valid == False:
                sys.exit(0)
            self.config = config
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
        self.node_desc = {}
        self.node_attrs = {}
        self.attr_names = []
        self.attr_types = []
        if self.config is not None:
            self.make_graph_from_config()
        else:
            self.make_new_graph_default()
        self.make_nx_graph()
        self.make_communities()

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
        if nodeid in self.neighbours:
            return self.neighbours[nodeid]
        else:
            return []

    def get_second_neighbours(self, nodeid):
        second_neighbours = set()
        for nodeid in self.get_neighbours(nodeid):
            second_neighbours.update(self.get_neighbours(nodeid))
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

    def oversample(self, data, num):
        sample = []
        if num > len(data):
            for _ in range(num):
                sample.append(random.choice(data))
        else:
            sample = random.sample(data, num)
        return sample

    def make_categorical(self, nodeids, cutoff):
        categorical = []
        num_entries = int(max(1, abs(len(nodeids) * cutoff)))
        indeg = Counter()
        for n in nodeids:
            indeg[n] = self.in_degree[n]
        if cutoff > 0:
            for x, c in indeg.most_common(num_entries):
                categorical.extend([x]*c)
        else:
            lowest = [x for x, c in indeg.most_common()]
            if num_entries > len(lowest):
                lowest = lowest[-num_entries:]
            for x in lowest:
                categorical.extend([x]*indeg[x])
        return categorical

    def connect_node_popularity_nodelist(self, nodeid, nodelist):
        categorical = self.make_categorical(nodelist, 1.0)
        target = random.choice(categorical)
        self.add_edge(nodeid, target)

    def connect_node_popularity(self, nodeid):
        ci = len(self.interactions)
        categorical = []
        nc = int(len(self.in_degree) * self.popularity_cutoff)
        for item, count in self.in_degree.most_common(nc):
            categorical.extend([item]*count)
        target = random.choice(categorical)
        self.add_edge(nodeid, target)

    def connect_node_random_nodelist(self, nodeid, nodelist):
        target = random.choice(nodelist)
        self.add_edge(nodeid, target)

    def connect_node_random(self, nodeid):
        ci = len(self.interactions)
        target = random.choice(range(ci))
        self.add_edge(nodeid, target)

    def connect_node_second_neighbour(self, nodeid):
        first_neighbours = self.get_neighbours(nodeid)
        second_neighbours = self.get_second_neighbours(nodeid)
        second_neighbours.difference_update(first_neighbours)
        if len(second_neighbours) > 0:
            target = random.choice(list(second_neighbours))
            self.add_edge(nodeid, target)

    def validate_config(self, config):
        valid = True
        sl_fields =   [["cores",
                        "extra_nodes_random",
                        "extra_nodes_popularity",
                        "popularity_cutoff",
                        "extra_edges_second_neighbour",
                        "extra_edges_random"],
                       ["core_core_connections",
                        "core_core_connection_type"]]
        for item in sl_fields:
            base = len(config[item[0]])
            for name in item[1:]:
                if len(config[name]) != base:
                    print("Field: " + str(name) + " should contain " + str(base) + " items.")
                    valid = False
        return valid

    def make_graph_from_config(self):
        cores = {}
        ci = 0
        # Create initial cores
        for index, core_size in enumerate(self.config["cores"]):
            cores[index] = []
            cores[index].append(ci)
            ci += 1
            self.add_edge(ci, cores[index][0])
            cores[index].append(ci)
            for _ in range(core_size):
                ci += 1
                self.connect_node_random_nodelist(ci, cores[index])
                cores[index].append(ci)

        if len(cores) > 1:
            for index, item in enumerate(self.config["core_core_connections"]):
                source_core, target_core, num_connections = item
                target1 = self.oversample(cores[source_core], num_connections)
                target2 = self.oversample(cores[target_core], num_connections)
                for ind in range(num_connections):
                    if self.config["core_core_connection_type"][index] == "direct":
                        self.add_edge(target1[ind], target2[ind])
                    else:
                        # Consider directionality
                        self.add_edge(target1[ind], ci)
                        self.add_edge(ci, target2[ind])
                        ci += 1

        for index, item in enumerate(self.config["extra_nodes_random"]):
            targets = self.oversample(cores[index], item)
            for ind in range(item):
                self.add_edge(ci, targets[ind])
                cores[index].append(ci)
                ci += 1

        for index, item in enumerate(self.config["extra_nodes_popularity"]):
            cutoff = self.config["popularity_cutoff"][index]
            categorical = self.make_categorical(cores[index], cutoff)
            targets = self.oversample(categorical, item)
            for ind in range(item):
                self.add_edge(ci, targets[ind])
                cores[index].append(ci)
                ci += 1

        for index, item in enumerate(self.config["extra_edges_second_neighbour"]):
            for _ in range(item):
                nodeid = random.choice(cores[index])
                self.connect_node_second_neighbour(nodeid)

        for index, item in enumerate(self.config["extra_edges_random"]):
            for _ in range(item):
                nodeid = random.choice(cores[index])
                self.connect_node_random(nodeid)

    def make_new_graph_default(self):
        # Create a starting core network
        cores = {}
        ci = 0
        for cn in range(self.num_cores):
            cores[cn] = []
            subset = max(1, int(self.num_nodes/(20*self.num_cores)))
            sn = int(subset + random.randint(1, subset))
            for source in range(ci, sn+ci):
                cores[cn].append(source)
                max_connections = int(sn * self.intra_core_connectivity)
                if max_connections > 0:
                    num_connections = random.randint(1, max_connections)
                    if sn > 0 and sn >= num_connections:
                        connections = random.sample(range(ci, sn+ci), num_connections)
                    else:
                        connections = [ci]
                    for target in connections:
                        self.add_edge(source, target)
            ci += sn

        # Create some connections between cores
        if self.num_cores > 1:
            num_connections = max(1, int(np.mean([len(x) for l, x in cores.items()]) * self.core_connectivity))
            for _ in range(num_connections):
                cs = self.oversample(range(self.num_cores), 2)
                target1 = random.choice(cores[cs[0]])
                target2 = random.choice(cores[cs[1]])
                if random.random() < self.connect_cores_directly:
                    self.add_edge(target1, target2)
                else:
                    self.add_edge(ci, target1)
                    self.add_edge(ci, target2)
                    ci += 1

        # Add further nodes to existing graph
        for nodeid in range(int(self.num_nodes * self.add_nodes_popularity)):
            self.connect_node_popularity(ci)
            ci += 1

        for nodeid in range(int(self.num_nodes * self.add_nodes_random)):
            self.connect_node_random(ci)
            ci += 1

        # Add extra connections between nearby existing nodes
        for _ in range(int(self.num_nodes * self.connect_second_neighbours)):
            nodeid = random.choice(range(len(self.nodeids)))
            self.connect_node_second_neighbour(nodeid)

        # Add extra connections between random nodes
        for _ in range(int(self.num_nodes * self.connect_random)):
            nodeid = random.choice(range(len(self.nodeids)))
            self.connect_node_random(nodeid)

    # These are used when writing the gexf file
    # Values should be a dict of nodeid:val entries
    def set_node_desc(self, name, atype, values):
        self.node_desc[name] = {}
        self.node_desc[name]['type'] = atype
        self.node_desc[name]['values'] = values

    def make_nx_graph(self):
        self.nx_graph = nx.Graph()
        self.nx_graph.add_weighted_edges_from(self.mapping)

    def make_communities(self):
        self.communities = community.best_partition(self.nx_graph)
        nd = dict(self.communities)
        for node, mod in self.communities.items():
            if mod not in self.clusters:
                self.clusters[mod] = []
            self.clusters[mod].append(node)
        # Set community labels for gexf output
        self.set_node_desc("community", "integer", nd)

    def get_community_dist(self):
        dist = Counter()
        for ci, cn in self.clusters.items():
            dist[len(cn)] += 1
        return dist

    def print_basic_stats(self):
        msg = ""
        msg += "Nodes: " + str(len(self.nodeids))
        msg += " Edges: " + str(len(self.mapping))
        print(msg)

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
        anames = []
        atypes = []
        nattrs = {}
        for aname, data in self.node_desc.items():
            anames.append(aname)
            atypes.append(data["type"])
            for n, v in data["values"].items():
                if n not in nattrs:
                    nattrs[n] = []
                nattrs[n].append(v)
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

            if len(anames) > 0:
                f.write("\t\t<attributes class=\"node\">\n")
                for index, name in enumerate(anames):
                    f.write("\t\t\t<attribute id=\"" + str(index) + "\" title=\"" + str(name) + "\" type=\"" + str(atypes[index]) + "\"/>\n")
                f.write("\t\t</attributes>\n")


            f.write("\t\t<nodes>\n")
            indent = '\t\t\t'
            for index, node in enumerate(nodes):
                label = vocab[node]
                entry = indent+ "<node id=\"" + str(label) + "\" label=\"" + str(node) + "\">\n"
                if len(anames) > 0:
                    entry += indent + "\t<attvalues>\n"
                    for index, name in enumerate(anames):
                        a = nattrs[node]
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
        self.old_nodeids = set(self.nodeids)
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
        d = self.old_nodeids.difference(self.nodeids)
        # Reconnect orphaned nodes
        if len(d) > 0:
            for nodeid in d:
                self.connect_node_random(nodeid)
        self.make_nx_graph()
        self.make_communities()

    # Increases weights of outgoing edges from each node by 1
    # The proportion of edges selected is specified by num
    # At least one edge is always chosen
    def increase_weights(self, nodeids, num=0.1):
        ret = {}
        for source, targets in self.interactions.items():
            if source in nodeids:
                ret[source] = Counter()
                up_targets = [x for x, c in targets.items()]
                num_changes = max(1, int(len(targets)*num))
                up_targets = self.oversample(up_targets, num_changes)
                for target, weight in targets.items():
                    if target in up_targets:
                        weight += 1
                    ret[source][target] = weight
            else:
                ret[source] = targets
        self.rebuild_graph(ret)

    # Decreases weights of outgoing edges from each node by 1
    # The proportion of edges selected is specified by num
    # At least one edge is always chosen
    # If the edge weight falls below zero, the edge disappears
    def decrease_weights(self, nodeids, num=0.1):
        ret = {}
        for source, targets in self.interactions.items():
            if source in nodeids:
                # Do not orphan a node in this process
                down_targets = [x for x, c in targets.items() if len(self.get_incoming_weights(x)) > 1]
                if len(down_targets) > 0:
                    num_changes = max(1, int(len(targets)*num))
                    down_targets = self.oversample(down_targets, num_changes)
                for target, weight in targets.items():
                    if target in down_targets:
                        weight -= 1
                    if weight > 0:
                        if source not in ret:
                            ret[source] = Counter()
                        ret[source][target] = weight
            else:
                ret[source] = targets
        self.rebuild_graph(ret)

    def move_nodes(self, nodeids, num=0.1):
        if random.random() < self.move_nodes_second_neighbour:
            return self.move_nodes_sn(nodeids, num)
        else:
            return self.move_nodes_random(nodeids, num)

    # Moves nodes, breaking existing edges and forming new edges with weight of 1
    # The proportion of edges to break and make are specified by num
    # New edges are always formed to second neighbour nodes
    def move_nodes_sn(self, nodeids, num=0.1):
        to_break = {}
        to_make = {}
        for nodeid in nodeids:
            # Do not orphan a node in this process
            first_neighbours = [x for x in self.get_neighbours(nodeid) if len(self.get_incoming_weights(x)) > 1]
            if len(first_neighbours) > 0:
                second_neighbours = self.get_second_neighbours(nodeid)
                num_to_change = max(1, int(len(first_neighbours)*num))
                tb = self.oversample(first_neighbours, num_to_change)
                tm = self.oversample(second_neighbours, num_to_change)
                to_break[nodeid] = tb
                to_make[nodeid] = tm
        ret = {}
        for source, targets in self.interactions.items():
            for target, count in targets.items():
                if source in to_break.keys() and target in to_break[source]:
                    continue
                if target in to_break.keys() and source in to_break[target]:
                    continue
                if source not in ret:
                    ret[source] = Counter()
                ret[source][target] = count
        for source, targets in to_make.items():
            for target in targets:
                if source not in ret:
                    ret[source] = Counter()
                ret[source][target] += 1
        self.rebuild_graph(ret)

    # Moves nodes, breaking existing edges and forming new edges with weight of 1
    # The proportion of edges to break and make are specified by num
    # New edges are always formed to second neighbour nodes
    def move_nodes_random(self, nodeids, num=0.1):
        to_break = {}
        to_make = {}
        for nodeid in nodeids:
            # Do not orphan a node in this process
            first_neighbours = [x for x in self.get_neighbours(nodeid) if len(self.get_incoming_weights(x)) > 1]
            if len(first_neighbours) > 0:
                second_neighbours = range(len(self.nodeids))
                num_to_change = max(1, int(len(first_neighbours)*num))
                tb = self.oversample(first_neighbours, num_to_change)
                tm = self.oversample(second_neighbours, num_to_change)
                to_break[nodeid] = tb
                to_make[nodeid] = tm
        ret = {}
        for source, targets in self.interactions.items():
            for target, count in targets.items():
                if source in to_break.keys() and target in to_break[source]:
                    continue
                if target in to_break.keys() and source in to_break[target]:
                    continue
                if source not in ret:
                    ret[source] = Counter()
                ret[source][target] = count
        for source, targets in to_make.items():
            for target in targets:
                if source not in ret:
                    ret[source] = Counter()
                ret[source][target] += 1
        self.rebuild_graph(ret)
