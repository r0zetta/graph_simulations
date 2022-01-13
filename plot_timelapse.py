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

def get_stats(current_ind, slice_len, filename):
    raw = load_slice(current_ind, slice_len, filename)
    full = get_counters_and_interactions2(raw)
    inter = full[view]
    sn_details = full['sn_details']
    pos = {}
    in_degree = Counter()
    out_degree = Counter()
    for source, targets in inter.items():
        pos[source] = make_position(source, sn_details)
        for target, count in targets.items():
            pos[target] = make_position(target, sn_details)
            in_degree[target] += count
            out_degree[source] += count
    degree = in_degree
    if sizes == "out_degree":
        degree = out_degree
    return inter, degree, pos

target = "UK_trolls_2021"
filename = "../twitter_analysis/" + target + "/data/raw.json"
savedir = "figs_" + target
if not os.path.exists(savedir):
    os.makedirs(savedir)

sizes = "in_degree"
view = "sn_rep"
scaling = 5
gravity = 20
iterations = 100
expand = 0.3
eadjust = 0.4
auto_zoom = True
font_scaling = "lin"
interpolation = "acc"

slice_len = 10000
ind_inc = 500
num_slices = 100

slice_ind = 0
current_ind = 0
fig_index = 0

inter, degree, pos = get_stats(current_ind, slice_len, filename)
current_ind += ind_inc
print("Getting slice " + str(slice_ind))
gv1 = GraphViz(inter, degree, pos,
               scaling=scaling, iterations=iterations, gravity=gravity,
               expand=expand, eadjust=eadjust, auto_zoom=auto_zoom,
               font_scaling=font_scaling, interpolation=interpolation)
im = gv1.make_graphviz()
fn = savedir + "/fig" + "%05d"%fig_index + ".png"
print("Saving graphviz: " + fn)
im.save(fn)
fig_index += 1

for n in range(num_slices):
    slice_ind += 1
    inter, degree, pos = get_stats(current_ind, slice_len, filename)
    current_ind += ind_inc
    print("Getting slice " + str(slice_ind))
    gv2 = GraphViz(inter, degree, pos,
                   scaling=scaling, iterations=iterations, gravity=gravity,
                   expand=expand, eadjust=eadjust, auto_zoom=auto_zoom,
                   font_scaling=font_scaling, interpolation=interpolation)
    iml = gv1.interpolate(gv2)
    for im in iml:
        fn = savedir + "/fig" + "%05d"%fig_index + ".png"
        print("Saving graphviz: " + fn)
        im.save(fn)
        fig_index += 1
    gv1 = gv2

image_files = [savedir + "/fig" + "%05d"%x + ".png" for x in range(fig_index)]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=30)
clip.write_videofile("timelapse_" + target + ".mp4")
# Change stuff in graph simulations to match this
