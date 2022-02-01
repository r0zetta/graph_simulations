import json, os, random
from collections import Counter
from graphviz import *
import moviepy.video.io.ImageSequenceClip

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
    # Assign random positions for new nodes
    if pos is not None:
        new_pos = {}
        for nodeid in nodeids:
            if nodeid in pos:
                new_pos[nodeid] = pos[nodeid]
            else:
                x, y = random.sample(range(1200), 2)
                new_pos[nodeid] = [x, y]
                #print("New node, pos:", x, y)
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

image_files = [savedir + "/frame" + "%05d"%x + ".png" for x in range(1800)]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=20)
clip.write_videofile("timelapse.mp4")
