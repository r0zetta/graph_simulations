import json, os, random
from collections import Counter
from graphviz import *
import networkx as nx
import community as louvain
import moviepy.video.io.ImageSequenceClip

def get_point_in_cluster(nodeid, inter, positions):
    mapping = []
    for source, targets in inter.items():
        for target, count in targets.items():
            mapping.append((source, target, count))

    G = nx.Graph()
    G.add_weighted_edges_from(mapping)
    clusters = louvain.best_partition(G)
    communities = {}
    for name, mod in clusters.items():
        if mod not in communities:
            communities[mod] = []
        communities[mod].append(name)

    xval, yval = random.sample(range(50, 1500), 2)
    if nodeid not in clusters:
        return xval, yval

    mod = clusters[nodeid]
    members = communities[mod]
    if len(members) < 1:
        return xval, yval

    xpos = []
    ypos = []
    for member in members:
        if member in positions:
            x, y = positions[member]
            xpos.append(x)
            ypos.append(y)
    if len(xpos) < 1 or len(ypos) < 1:
        return xval, yval
    xmin = min(xpos)
    xmax = max(xpos)
    ymin = min(ypos)
    ymax = max(ypos)
    xval = (xmax-xmin) / 2
    yval = (ymax-ymin) / 2
    return xval, yval


fn = "generated_data.json"
raw = []
with open(fn, "r") as f:
    for line in f:
        entry = json.loads(line.strip())
        raw.append(entry)

slice_len = 10000
ind_inc = 100
num_slices = 400
current_ind = 0
glist = []
pos = None
for n in range(num_slices):
    inter = {}
    nodeids = set()
    for entry in raw[current_ind:current_ind+slice_len]:
        if entry["shared_pid"] is not None:
            poster = entry["poster"]
            original_poster = entry["original_poster"]
            nodeids.add(poster)
            nodeids.add(original_poster)
            if poster not in inter:
                inter[poster] = Counter()
            inter[poster][original_poster] += 1

    # Set initial node positions from previous visualization
    # Assign positions for new nodes based on modularity
    if pos is not None:
        new_pos = {}
        for nodeid in nodeids:
            if nodeid in pos:
                new_pos[nodeid] = pos[nodeid]
            else:
                x, y = get_point_in_cluster(nodeid, inter, pos)
                new_pos[nodeid] = [x, y]
        pos = dict(new_pos)

    current_ind += ind_inc
    print("Getting slice " + str(n) + " / " + str(num_slices))
    gv = GraphViz(from_dict=inter,
                  initial_pos=pos,
                  mag_factor=1.0,
                  graph_style="glowy",
                  scaling=40,
                  gravity=5,
                  min_font_size=1,
                  max_font_size=55,
                  min_node_size=10,
                  max_node_size=20,
                  min_edge_size=2,
                  max_edge_size=5,
                  size_by="in_degree",
                  #font_scaling="pow2.0",
                  font_scaling="root2.5",
                  auto_zoom=False,
                  expand=0.3)
    pos = gv.positions
    #im = gv.make_graphviz()
    #im.save("test.png")
    #sys.exit(0)
    glist.append(gv)

savedir = "frames"
if not os.path.exists(savedir):
    os.makedirs(savedir)
num_steps = 5
gv0 = glist[0]
gv0.interpolate_multiple(glist, savedir, num_steps=num_steps)

num_frames = min(1800, num_slices*num_steps)
image_files = [savedir + "/frame" + "%05d"%x + ".png" for x in range(num_frames)]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=20)
clip.write_videofile("timelapse.mp4")
