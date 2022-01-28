import json, os
from collections import Counter
from graphviz import *
import moviepy.video.io.ImageSequenceClip

fn = "social_data.json"
raw = []
with open(fn, "r") as f:
    for line in f:
        entry = json.loads(line.strip())
        raw.append(entry)

pid_poster = {}
for entry in raw:
    if entry["shared_pid"] == None:
        pid_poster[entry["pid"]] = entry["poster"]

slice_len = 10000
ind_inc = 50
num_slices = 100
current_ind = 0
glist = []
for n in range(num_slices):
    inter = {}
    for entry in raw[current_ind:current_ind+slice_len]:
        if entry["shared_pid"] is not None:
            poster = entry["poster"]
            shared = entry["shared_pid"]
            if poster not in inter:
                inter[poster] = Counter()
            inter[poster][pid_poster[shared]] += 1

    current_ind += ind_inc
    print("Getting slice " + str(n) + " / " + str(num_slices))
    gv = GraphViz(inter,
                  size_by="in_degree",
                  font_scaling="root2.5")
    glist.append(gv)

savedir = "frames"
if not os.path.exists(savedir):
    os.makedirs(savedir)
num_steps = 10
gv0 = glist[0]
gv0.interpolate_multiple(glist, savedir, num_steps=num_steps)

image_files = [savedir + "/frame" + "%05d"%x + ".png" for x in range(num_steps*num_slices)]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=20)
clip.write_videofile("timelapse.mp4")
