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
num_steps = 10

slice_len = 10000
ind_inc = 100
num_slices = 1000

slice_ind = 0
current_ind = 0
fig_index = 0

inter, pos = get_stats(current_ind, slice_len, filename, view)
current_ind += ind_inc
print("Getting slice " + str(slice_ind))
gv1 = GraphViz(inter, initial_pos=pos, size_by="in_degree",
               font_scaling="root2.5", interpolation="acc")
im = gv1.make_graphviz()
fn = savedir + "/fig" + "%05d"%fig_index + ".png"
print("Saving graphviz: " + fn)
im.save(fn)
fig_index += 1

for n in range(num_slices):
    slice_ind += 1
    inter, pos = get_stats(current_ind, slice_len, filename, view)
    current_ind += ind_inc
    print("Getting slice " + str(slice_ind))
    gv2 = GraphViz(inter, initial_pos=pos, size_by="in_degree",
                   font_scaling="root2.5", interpolation="acc")
    iml = gv1.interpolate(gv2, num_steps = num_steps)
    for im in iml:
        fn = savedir + "/fig" + "%05d"%fig_index + ".png"
        print("Saving graphviz: " + fn)
        im.save(fn)
        fig_index += 1
    gv1 = gv2

image_files = [savedir + "/fig" + "%05d"%x + ".png" for x in range(fig_index)]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=20)
clip.write_videofile("timelapse_" + target + ".mp4")
# Change stuff in graph simulations to match this
