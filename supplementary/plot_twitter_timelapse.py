from gather_analysis_helper import *
from graphviz import *
import networkx as nx
import community as louvain
import moviepy.video.io.ImageSequenceClip

def load_slice(start_index, length, filename):
    ret = []
    end_index = start_index + length
    index = 0
    with io.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            index += 1
            if index >= start_index and index < end_index:
                d = json.loads(line)
                ret.append(d)
            if index > end_index:
                break
    return ret

def get_stats(current_ind, slice_len, filename, view):
    raw = load_slice(current_ind, slice_len, filename)
    full = get_counters_and_interactions2(raw)
    inter = full[view]
    return inter

def get_points_in_cluster(nodeids, inter, positions):
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

    cpos = {}
    for nodeid in nodeids:
        xval, yval = random.sample(range(50, 1500), 2)
        if nodeid in clusters:
            mod = clusters[nodeid]
            members = communities[mod]
            if len(members) > 0:
                xpos = []
                ypos = []
                for member in members:
                    if member in positions:
                        x, y = positions[member]
                        xpos.append(x)
                        ypos.append(y)
                if len(xpos) > 0 and len(ypos) > 0:
                    xmin = min(xpos)
                    xmax = max(xpos)
                    ymin = min(ypos)
                    ymax = max(ypos)
                    xval = (xmax-xmin) / 2
                    yval = (ymax-ymin) / 2
        cpos[nodeid] = [xval, yval]
    return cpos


target = "QPR"
filename = "../twitter_analysis/" + target + "/data/raw.json"
savedir = "frames" + target
if not os.path.exists(savedir):
    os.makedirs(savedir)

view = "sn_rsn"
num_steps = 5

slice_len = 2000
ind_inc = 500
num_slices = 10

current_ind = 0

glist = []
pos = None

for n in range(num_slices):
    print("Getting slice " + str(n) + " / " + str(num_slices))
    inter = get_stats(current_ind, slice_len, filename, view)
    nodeids = set()
    for source, targets in inter.items():
        nodeids.add(source)
        for tar, weight in targets.items():
            nodeids.add(tar)

    # Set initial node positions from previous visualization
    # Assign random positions for new nodes
    if pos is not None:
        new_pos = {}
        to_get = []
        for nodeid in nodeids:
            if nodeid in pos:
                new_pos[nodeid] = pos[nodeid]
            else:
                to_get.append(nodeid)
        if len(to_get) > 0:
            cpos = get_points_in_cluster(to_get, inter, pos)
            for x, c in cpos.items():
                new_pos[x] = c
        pos = dict(new_pos)

    current_ind += ind_inc
    #print("Calculating layout")
    gv = GraphViz(from_dict=inter,
                  initial_pos=pos,
                  mag_factor=1.0,
                  graph_style="glowy",
                  scaling=40,
                  gravity=5,
                  min_font_size=1,
                  max_font_size=60,
                  min_node_size=10,
                  max_node_size=20,
                  min_edge_size=1,
                  max_edge_size=5,
                  size_by="in_degree",
                  font_scaling="root2.5",
                  auto_zoom=False,
                  expand=0.5)
    pos = gv.positions
    #print("Making image")
    #im = gv.make_graphviz()
    #im.save("test.png")
    #sys.exit(0)
    glist.append(gv)

gv0 = glist[0]
gv0.interpolate_multiple(glist, savedir, num_steps=num_steps)
num_frames = min(1800, num_slices*num_steps)
image_files = [savedir + "/frame" + "%05d"%x + ".png" for x in range(num_frames)]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=20)
clip.write_videofile("timelapse_" + target + ".mp4")
