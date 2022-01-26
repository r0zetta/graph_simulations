from gather_analysis_helper import *
from graphviz import *
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

def make_position(sn, sn_details):
    xpos = 0
    ypos = 0
    if sn_details is not None:
        if sn in sn_details:
            dets = sn_details[sn]
            id_str = dets['id_str']
            xpos = int(id_str[:4])
            ypos = int(id_str[-4:])
    return (xpos, ypos)

def get_stats(current_ind, slice_len, filename, view):
    raw = load_slice(current_ind, slice_len, filename)
    full = get_counters_and_interactions2(raw)
    inter = full[view]
    sn_details = full['sn_details']
    pos = {}
    for source, targets in inter.items():
        pos[source] = make_position(source, sn_details)
        for target, count in targets.items():
            pos[target] = make_position(target, sn_details)
    return inter, pos

target = "UK_trolls_2021"
filename = "../twitter_analysis/" + target + "/data/raw.json"
savedir = "figs_" + target
if not os.path.exists(savedir):
    os.makedirs(savedir)

view = "sn_rep"
num_steps = 20

slice_len = 10000
ind_inc = 200
num_slices = 90

slice_ind = 0
current_ind = 0

glist = []

for n in range(num_slices):
    slice_ind += 1
    inter, pos = get_stats(current_ind, slice_len, filename, view)
    current_ind += ind_inc
    print("Getting slice " + str(slice_ind) + " / " + str(num_slices))
    gv = GraphViz(inter, initial_pos=pos, size_by="in_degree",
                  font_scaling="root2.5")
    glist.append(gv)

gv0 = glist[0]
gv0.interpolate_multiple(glist, savedir, num_steps=num_steps)

image_files = [savedir + "/frame" + "%05d"%x + ".png" for x in range(num_steps*num_slices)]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=20)
clip.write_videofile("timelapse_" + target + ".mp4")
