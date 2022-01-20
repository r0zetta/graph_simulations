from gather_analysis_helper import *
from graphviz import *

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

target = "UK_trolls_2021"
filename = "../twitter_analysis/" + target + "/data/raw.json"
savedir = "figs_" + target
if not os.path.exists(savedir):
    os.makedirs(savedir)

print("Loading and parsing data")
view = "sn_rep"
slice_len = 10000
current_ind = 0
inter = get_stats(current_ind, slice_len, filename, view)

other = {}
for source, targets in inter.items():
    if source not in other:
        val = random.uniform(-1, 1)
        other[source] = val
    for target, count in targets.items():
        if target not in other:
            val = random.uniform(-1, 1)
            other[target] = val

extra_vars={"other":other}
mag_factor=1.0
scaling=1.0
gravity=1.0
iterations=100
strong_gravity=False
dissuade_hubs=False
edge_weight_influence=1.0
eadjust=0.3
expand=0.2
auto_zoom=True
label_font="Arial Bold"
min_font_size=5
max_font_size=40
font_scaling="root2.5"
min_node_size=5
max_node_size=20
node_scaling=font_scaling
min_edge_size=1
max_edge_size=5
edge_scaling="lin"
background_mode="black"
edge_style="curved"
palette="intense"
color_by="modularity"
size_by="in_degree"
labels="nodeid"
label_type="short"

print("Making graph")
gv = GraphViz(from_dict=inter,
              extra_vars = extra_vars,
              mag_factor = mag_factor,
              scaling = scaling,
              gravity = gravity,
              iterations = iterations,
              strong_gravity = strong_gravity,
              dissuade_hubs = dissuade_hubs,
              edge_weight_influence = edge_weight_influence,
              eadjust = eadjust,
              expand = expand,
              auto_zoom = auto_zoom,
              label_font = label_font,
              min_font_size = min_font_size,
              max_font_size = max_font_size,
              font_scaling = font_scaling,
              min_node_size = min_node_size,
              max_node_size = max_node_size,
              node_scaling = node_scaling,
              min_edge_size = min_edge_size,
              max_edge_size = max_edge_size,
              edge_scaling = edge_scaling,
              background_mode = background_mode,
              edge_style = edge_style,
              palette = palette,
              color_by = color_by,
              size_by = size_by,
              labels = labels,
              label_type = label_type)

im = gv.make_graphviz()
fn = "test.png"
print("Saving graphviz: " + fn)
im.save(fn)
