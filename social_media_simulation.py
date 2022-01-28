from graph import *
from graphviz import *
from collections import Counter
import os, time, random, sys, json

def make_categorical(vals):
    cat = []
    for x, c in vals.items():
        n = max(1, int(c * 100))
        cat.extend([x]*n)
    return cat

def save_graphviz(g, fn):
    gv = GraphViz(from_dict=g.interactions,
                  mag_factor=1.0,
                  scaling=20,
                  gravity=5,
                  min_font_size=1,
                  max_font_size=55,
                  min_node_size=5,
                  max_node_size=20,
                  min_edge_size=1,
                  max_edge_size=1,
                  label_font="Arial",
                  background_mode="black",
                  font_scaling="pow2.0",
                  auto_zoom=False,
                  expand=1.5)
    im = gv.make_graphviz()
    im.save(fn)

# Make population interaction graph
g = Graph(num_nodes=1000,
          num_cores=2,
          intra_core_connectivity=0.3,
          core_connectivity=0.2,
          add_nodes_random=0.2,
          add_nodes_popularity=2.5,
          popularity_cutoff=0.5,
          connect_cores_directly=0.0,
          connect_second_neighbours=1.0,
          connect_random=0.0)
g.print_basic_stats()
g.print_community_stats()
#save_graphviz(g, "social_simulation.png")

# Assign nodes as influencer or consumer
cluster_in_deg = {}
for mod, items in g.clusters.items():
    if mod not in cluster_in_deg:
        cluster_in_deg[mod] = Counter()
    for nodeid in items:
        in_deg = g.in_degree[nodeid]
        cluster_in_deg[mod][nodeid] = in_deg

influencers_per_community = 5
mod_influencers = {}
for mod, in_deg in cluster_in_deg.items():
    mod_influencers[mod] = [x for x, c in in_deg.most_common(influencers_per_community)]

influencers = []
for mod, infs in mod_influencers.items():
    influencers.extend(infs)

# Assign chance of publishing original content
node_originality = {}
for nodeid in g.nodeids:
    if nodeid in influencers:
        node_originality[nodeid] = random.uniform(0.8, 1.0)
    else:
        node_originality[nodeid] = random.uniform(0.01, 0.5)
g.set_node_desc("originality", "float", node_originality)
originality_categorical = make_categorical(node_originality)

# Assign chance of sharing content
node_verbosity = {}
for nodeid in g.nodeids:
    node_verbosity[nodeid] = random.uniform(0.01, 0.3)
g.set_node_desc("verbosity", "float", node_verbosity)
verbosity_categorical = make_categorical(node_verbosity)

def new_post(post_id, nodeid, ts):
    entry = {}
    entry["pid"] = post_id
    entry["poster"] = nodeid
    entry["ts"] = step
    entry["shared_pid"] = None
    entry["shared_by"] = []
    entry["last_seen"] = ts
    return entry

def share_post(post_id, nodeid, ts, shared_pid):
    entry = {}
    entry["pid"] = post_id
    entry["poster"] = nodeid
    entry["ts"] = step
    entry["shared_pid"] = shared_pid
    entry["shared_by"] = []
    entry["last_seen"] = ts
    return entry

savefile = "generated_data.json"
if os.path.exists(savefile):
    os.remove(savefile)
f = open(savefile, "a")

post_verbosity = 0.00017
share_verbosity = 0.00019
post_id = 0
expire_after = 5000
delete_after = 50000
recently_published = {}
last_seen = {}
step = 0
while post_id < 100000:
    if step % 10000 == 0:
        post_verbosity = 0.00017 + random.uniform(-0.00002, 0.00003)
        share_verbosity = 0.00019 + random.uniform(-0.00002, 0.00003)
        print("**VERBOSITY CHANGED**", post_verbosity, share_verbosity)
    if random.random() < post_verbosity:
        nodeid = random.choice(originality_categorical)
        e = new_post(post_id, nodeid, step)
        recently_published[post_id] = e
        print("Original", e)
        f.write(json.dumps(e)+"\n")
        post_id += 1
    to_add = []
    expired = []
    shared = {}
    for pid, entry in recently_published.items():
        ls = entry["last_seen"]
        if (step - ls) > delete_after:
            expired.append(pid)
        ts = entry["ts"]
        if (step - ts) < expire_after:
            if random.random() < share_verbosity:
                poster = entry["poster"]
                neighbours = g.get_neighbours(poster)
                shared_by = entry["shared_by"]
                diff = set()
                for x in neighbours:
                    if x not in shared_by:
                        diff.add(x)
                if len(diff) > 0:
                    v = {}
                    for n in diff:
                        v[n] = node_verbosity[n]
                    c = make_categorical(v)
                    sharer = random.choice(c)
                    p = entry["pid"]
                    if entry["shared_pid"] is not None:
                        p = entry["shared_pid"]
                    e = share_post(post_id, sharer, step, p)
                    if p not in shared:
                        shared[p] = []
                    shared[p].append(sharer)
                    to_add.append(e)
                    print("Share", poster, e)
                    f.write(json.dumps(e)+"\n")
                    post_id += 1
    if len(to_add) > 0:
        for entry in to_add:
            pid = entry["pid"]
            recently_published[pid] = entry
    if len(shared) > 0:
        for x, c in shared.items():
            if x not in recently_published:
                continue
            recently_published[x]["shared_by"].extend(c)
            recently_published[x]["shared_by"] = list(set(recently_published[x]["shared_by"]))
            recently_published[x]["last_seen"] = step
    if len(expired) > 0:
        print("Expiring pids: ", expired)
        temp = {}
        for pid, entry in recently_published.items():
            if pid not in expired:
                temp[pid] = entry
        recently_published = dict(temp)
    step += 1

