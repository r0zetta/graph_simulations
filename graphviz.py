import networkx as nx
import community as louvain
from fa2 import ForceAtlas2
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import numpy as np
import math, sys


class GraphViz:
    def __init__(self, from_dict=None, from_mapping=None, from_nx=None,
                 initial_pos=None, extra_vars=None, info=None,
                 mag_factor=1.0, scaling=5.0, gravity=20.0, iterations=100,
                 strong_gravity=False, dissuade_hubs=True, edge_weight_influence=1.0,
                 eadjust=0.5, expand=0.3, zoom=[[0.0,0.0],[1.0,1.0]], auto_zoom=False,
                 modularity_legend=None, label_font="Arial Bold", alt_font="Arial",
                 min_font_size=4, max_font_size=24, font_scaling="lin",
                 min_node_size=5, max_node_size=20, node_scaling="lin",
                 min_edge_size=1, max_edge_size=5, edge_scaling="lin",
                 background_mode="black", edge_style="curved", graph_style="normal",
                 palette="intense", color_by="modularity", size_by="out_degree",
                 labels="nodeid", max_label_len=50, interpolation="lin"):
        self.extra_vars = extra_vars
        if self.extra_vars is None:
            self.extra_vars = {}
        self.info = info
        self.inter = None
        if from_dict is not None:
            self.inter = from_dict
        elif from_mapping is not None:
            self.inter = self.inter_from_mapping(from_mapping)
        elif from_nx is not None:
            self.inter = self.inter_from_nx(from_nx)
        else:
            print("Please provide a valid graph using:")
            print("from_dict=dict")
            print("from_mapping=mapping")
            print("or")
            print("from_nx=nx")
            sys.exit(0)
        self.initial_pos = initial_pos
        self.mag_factor = mag_factor
        self.scaling = scaling
        self.gravity = gravity
        self.strong_gravity = strong_gravity
        self.dissuade_hubs = dissuade_hubs
        self.edge_weight_influence = edge_weight_influence
        self.iterations = iterations
        self.eadjust = eadjust
        self.expand_by = expand
        self.zoom = zoom
        self.auto_zoom = auto_zoom
        self.modularity_legend = modularity_legend
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.font_scaling = font_scaling # lin, pow, root, fixed
        self.label_font = label_font
        self.alt_font = alt_font
        self.min_node_size = min_node_size
        self.max_node_size = max_node_size
        self.node_scaling = node_scaling # lin, pow, root, fixed
        self.min_edge_size = min_edge_size
        self.max_edge_size = max_edge_size
        self.edge_scaling = edge_scaling # lin, pow, root, fixed
        self.edge_style = edge_style # curved, straight
        self.graph_style = graph_style # curved, straight
        self.background_mode = background_mode # white, black
        if self.background_mode == "black":
            self.background_color = (0,0,0)
            self.node_outline = (255,255,255)
            self.font_color = (255,255,255)
        else:
            self.background_color = (255,255,255)
            self.node_outline = (0,0,0)
            self.font_color = (0,0,0)
        self.palette = palette
        self.labels = labels
        self.max_label_len = max_label_len
        self.color_by = color_by
        self.size_by = size_by
        self.interpolation = interpolation # lin, dec, acc
        self.canvas_width = int(1200 * self.mag_factor)
        self.canvas_height = int(1200 * self.mag_factor)
        self.palette_intense = ((0, 131, 182), (255, 75, 0), (32, 198, 0), (255, 84, 255),
                               (111, 50, 25), (0, 204, 134), (255, 0, 138), (219, 168, 0),
                               (150, 97, 180), (8, 91, 0), (0, 222, 255), (120, 191, 143),
                               (0, 173, 255), (255, 126, 86), (34, 71, 68), (208, 164, 79),
                               (243, 0, 176), (0, 97, 87), (0, 201, 255), (163, 120, 195),
                               (0, 188, 0), (255, 86, 0), (0, 195, 246), (0, 97, 59),
                               (177, 75, 75), (0, 152, 254), (204, 130, 0), (0, 174, 90),
                               (221, 111, 255), (255, 52, 150), (129, 113, 26), (135, 157, 0),
                               (0, 126, 147), (255, 48, 86), (66, 142, 22), (0, 177, 163),
                               (50, 85, 124), (0, 175, 255), (199, 96, 31), (108, 134, 255),
                               (253, 73, 214), (65, 162, 202), (169, 79, 128), (63, 172, 130),
                               (202, 135, 186), (17, 181, 0), (0, 198, 255), (255, 78, 0),
                               (0, 114, 88), (150, 100, 27), (204, 114, 255), (38, 100, 180),
                              )
        self.palette_gradient = ((215, 25, 28),
                                 (223, 73, 62),
                                 (223, 130, 102),
                                 (244, 196, 149),
                                 (254, 250, 187),
                                 (212, 228, 189),
                                 (150, 189, 186),
                                 (85, 148, 183),
                                 (52, 128, 182),
                                 (36, 101, 150)
                                )
        self.color_palette = self.palette_intense
        if palette == "gradient":
            self.color_palette = self.palette_gradient
        self.pascal_memo = {}
        self.get_stats()

    def inter_from_mapping(self, mapping):
        inter = {}
        edge_labels = {}
        for item in mapping:
            if len(item) == 3:
                source, target, weight = item
            elif len(item) == 4:
                source, target, weight, label = item
                edge_labels[str(source)+":"+str(target)] = label
            if source not in inter:
                inter[source] = Counter()
            inter[source][target] += weight
        if len(edge_labels) > 0:
            self.extra_vars["edge_labels"] = edge_labels
        return inter

    def inter_from_nx(self, nx):
        inter = {}
        mapping = list(nx.edges.data('weight'))
        return self.inter_from_mapping(mapping)

    def hex_to_rgb(self, h):
        h = h.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        return rgb

    def adjust(self, rgb, val):
        return tuple(min(255, int(x*val)) for x in rgb)

    def angle(self, p1, p2):
        return math.atan2(p2[1]-p1[1], p2[0]-p1[0])

    def distance(self, p1, p2):
        return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

    def move_point(self, p, distance, angle):
        new_x = p[0] + distance * math.cos(angle)
        new_y = p[1] + distance * math.sin(angle)
        return (new_x, new_y)

    def get_midpoint(self, p1, p2):
        if self.edge_style == "curved":
            return self.get_midpoint_bezier(p1, p2)
        else:
            return self.get_midpoint_straight(p1, p2)

    def get_midpoint_straight(self, p1, p2):
        angle = self.angle(p1, p2)
        distance = self.distance(p1, p2) * 0.5
        start_point = p1
        new_point = self.move_point(start_point, distance, angle)
        return new_point

    def get_midpoint_bezier(self, p1, p2):
        control_points = self.get_control_points(p1, p2)
        points = self.make_bezier(control_points, 100)
        steps = len(points)
        mstep = int(steps/2)
        return points[mstep]

    def get_control_points(self, p1, p2):
        num_points = 2
        angle_offset = 0.8
        cp = []
        cp.append(p1)
        start_angle = self.angle(p1, p2)
        start_distance = self.distance(p1, p2)
        new_angle = start_angle + angle_offset
        new_distance = (start_distance / (num_points+1))
        c = p1
        for n in range(num_points):
            c = self.move_point(c, new_distance, new_angle)
            cp.append(c)
            new_angle = new_angle - (angle_offset/(num_points-1))
        cp.append(p2)
        return cp

    def make_bezier(self, control_points, num_steps):
        n = len(control_points)
        combinations = self.pascal_row(n-1)
        ts = [t/num_steps for t in range(num_steps+1)]
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*control_points)))
        return result

    def pascal_row(self, n):
        if n in self.pascal_memo:
            return self.pascal_memo[n]
        result = [1]
        x, numerator = 1, n
        for denominator in range(1, n//2+1):
            x *= numerator
            x /= denominator
            result.append(x)
            numerator -= 1
        if n&1 == 0:
            result.extend(reversed(result[:-1]))
        else:
            result.extend(reversed(result))
        self.pascal_memo[n] = result
        return result

    def convert_coords(self, x, y, pos):
        zxmin, zymin = self.zoom[0]
        zxmax, zymax = self.zoom[1]
        zwidth = zxmax - zxmin
        zheight = zymax - zymin
        max_x = max([c[0] for x, c in pos.items()])
        min_x = min([c[0] for x, c in pos.items()])
        max_y = max([c[1] for x, c in pos.items()])
        min_y = min([c[1] for x, c in pos.items()])
        coords_width = (max_x - min_x) * zwidth
        coords_height = (max_y - min_y) * zheight
        width_ratio = self.canvas_width/coords_width
        height_ratio = self.canvas_height/coords_height
        new_x = (x - min_x)*width_ratio - (self.canvas_width * (zxmin/zwidth))
        new_y = (y - min_y)*height_ratio - (self.canvas_height * (zymin/zheight))
        return new_x, new_y

    def expand_graph(self):
        if self.expand_by == 0:
            return
        shift_x = self.canvas_width * self.expand_by
        shift_y = self.canvas_height * self.expand_by
        new_coords = {}
        for sn, coords in self.positions.items():
            x, y = coords
            x += shift_x
            y += shift_y
            new_coords[sn] = (x, y)
        self.positions = new_coords
        self.canvas_width += int(self.canvas_width * (self.expand_by*2))
        self.canvas_height += int(self.canvas_height * (self.expand_by*2))
        #self.mag_factor += int(self.mag_factor * self.expand_by)
        mid_x = int(self.canvas_width/2)
        mid_y = int(self.canvas_height/2)
        mid_p = (mid_x, mid_y)
        for sn, coords in self.positions.items():
            dist = self.distance(mid_p, coords)
            angle = self.angle(mid_p, coords)
            new_coords = self.move_point(coords, dist * self.expand_by, angle)
            self.positions[sn] = new_coords

    def coords_in_rect(self, coords, xl, yl, xh, yh):
        cx, cy = coords
        if cx >= xl and cx <= xh and cy >= yl and cy <= yh:
            return True
        return False

    def get_bounds(self, var):
        steps = len(var)
        le = 0
        re = 0
        is_dense = False
        for i in range(steps):
            if is_dense == False:
                re += 1
                if var[i] < 10:
                    le += 1
                else:
                    is_dense = True
            else:
                if var[i] > 10:
                    re += 1
        return le, re

    def autozoom(self, pos):
        max_x = int(max([c[0] for x, c in pos.items()]))
        min_x = int(min([c[0] for x, c in pos.items()]))
        max_y = int(max([c[1] for x, c in pos.items()]))
        min_y = int(min([c[1] for x, c in pos.items()]))
        steps = 10
        sw = int((max_x - min_x)/steps)
        sh = int((max_y - min_y)/steps)
        # Left to right sweep
        xsweep = Counter()
        for i, x in enumerate(range(min_x, max_x, sw)):
            xsweep[i] = 0
            for sn, coords in pos.items():
                xl = x
                xh = x + sw
                yl = min_y
                yh = max_y
                if self.coords_in_rect(coords, xl, yl, xh, yh) is True:
                    xsweep[i] += 1
        # Top to bottom sweep
        ysweep = Counter()
        for i, y in enumerate(range(min_y, max_y, sh)):
            ysweep[i] = 0
            for sn, coords in pos.items():
                xl = min_x
                xh = max_x
                yl = y
                yh = y + sh
                if self.coords_in_rect(coords, xl, yl, xh, yh) is True:
                    ysweep[i] += 1
        lx, ux = self.get_bounds(xsweep)
        ly, uy = self.get_bounds(ysweep)
        self.zoom = [[lx*0.1, ly*0.1], [ux*0.1, uy*0.1]]

    def get_stats(self):
        mapping = []
        self.max_weight = 0
        in_degree = Counter()
        out_degree = Counter()
        for source, targets in self.inter.items():
            for target, count in targets.items():
                if count > self.max_weight:
                    self.max_weight = count
                mapping.append((source, target, count))
                in_degree[target] += count
                out_degree[source] += count

        self.extra_vars["in_degree"] = in_degree
        self.extra_vars["out_degree"] = out_degree

        self.G = nx.Graph()
        self.G.add_weighted_edges_from(mapping)
        self.communities = louvain.best_partition(self.G)

        clusters = {}
        for node, mod in self.communities.items():
            if mod not in clusters:
                clusters[mod] = []
            clusters[mod].append(node)

        FA2 = ForceAtlas2(self.G,
                          #outboundAttractionDistribution=self.dissuade_hubs,
                          edgeWeightInfluence=self.edge_weight_influence,
                          strongGravityMode=self.strong_gravity,
                          scalingRatio=self.scaling,
                          gravity=self.gravity,
                          verbose=False)
        pos = FA2.forceatlas2_networkx_layout(self.G,
                                              pos=self.initial_pos,
                                              iterations=self.iterations)
        if self.auto_zoom == True:
            self.autozoom(pos)

        self.positions = {}
        for sn, coords in pos.items():
            x, y = coords
            newx, newy = self.convert_coords(x, y, pos)
            self.positions[sn] = [newx, newy]

        self.expand_graph()

        modularity_class = {}
        communities = []
        for community_number, community in clusters.items():
            communities.append(community_number)
            for name in community: 
                modularity_class[name] = community_number
        self.extra_vars["modularity"] = modularity_class
        self.extra_vars["communities"] = communities

    def set_from_graph(self, g):
        self.inter = g.inter
        self.positions = g.positions
        self.extra_vars = g.extra_vars

    def set_size(self, s, max_v, min_s, max_s, style):
        sf = min_s
        lf = max_s - min_s
        if style == "lin":
            return sf + int(lf * (s/max_v))
        elif "pow" in style:
            p = 2
            if len(style) > 3:
                p = float(style[3:])
            return sf + int(lf * ((s ** p)/(max_v ** p)))
        elif "root" in style:
            p = 2
            if len(style) > 4:
                p = float(style[4:])
            s = sf + int(lf * ((s ** (1.0/p))/(max_v ** (1.0/p))))
            return s
        elif "fixed" in style:
            if len(style) > 5:
                s = int(style[5:])
                return s
            return max_s

    def set_font_size(self, s):
        max_v = max([c for x, c in self.extra_vars[self.size_by].items()])
        min_s = self.min_font_size
        max_s = self.max_font_size
        scaling = self.font_scaling
        return self.set_size(s, max_v, min_s, max_s, scaling)

    def set_node_size(self, s):
        max_v = max([c for x, c in self.extra_vars[self.size_by].items()])
        min_s = self.min_node_size
        max_s = self.max_node_size
        scaling = self.node_scaling
        return self.set_size(s, max_v, min_s, max_s, scaling)

    def set_edge_size(self, s):
        max_v = self.max_weight
        min_s = self.min_edge_size
        max_s = self.max_edge_size
        scaling = self.edge_scaling
        return self.set_size(s, max_v, min_s, max_s, scaling)

    def get_gradient_index(self, mod):
        v = self.extra_vars[self.color_by]
        maxv = max([c for x, c in v.items()])
        minv = min([c for x, c in v.items()])
        num_steps = len(self.color_palette)
        step_size = (maxv - minv)/num_steps
        cv = minv
        for x in range(num_steps):
            if mod >= cv and mod <= cv+step_size:
                return x
            cv += step_size

    def draw_edge_curved(self, p1, p2, w, color):
        control_points = self.get_control_points(p1, p2)
        points = self.make_bezier(control_points, 100)
        steps = len(points)
        for index in range(steps-1):
            x1 = points[index][0]
            y1 = points[index][1]
            x2 = points[index+1][0]
            y2 = points[index+1][1]
            if self.graph_style == "glowy":
                adjust = 1.0
                if index <=(steps/2):
                    adjust = (index+1)/(steps/2)
                else:
                    adjust = (steps-index+1)*2/steps
                adjust = max(0.3, adjust)
                c = tuple([min(255, int(x * adjust)) for x in color])
            else:
                c = color
            self.draw.line((x1, y1, x2, y2), fill=c, width=w)

    def draw_glowy_line(self, p1, p2, width, color, steps):
        sd = distance(p1, p2) * (1/steps)
        ang = angle(p1, p2)
        p = p1
        for n in range(steps):
            new_p = move_point(p, sd, ang)
            adjust = 1.0
            if n <=(steps/2):
                adjust = (n+1)/(steps/2)
            else:
                adjust = (steps-n+1)*2/steps
            adjust = max(0.3, adjust)
            c = tuple([min(255, int(x * adjust)) for x in color])
            x1, y1 = p
            x2, y2 = new_p
            draw.line((x1, y1, x2, y2), fill=c, width=width)
            p = new_p

    def draw_edge_straight(self, p1, p2, w, c):
        if self.graph_style == "glowy":
            self.draw_glowy_line(p1, p2, w, c, 20)
        else:
            x1, y1 = p1
            x2, y2 = p2
            self.draw.line((x1, y1, x2, y2), fill=c, width=w)

    def draw_edge(self, p1, p2, w, c):
        if self.edge_style == "curved":
            self.draw_edge_curved(p1, p2, w, c)
        else:
            self.draw_edge_straight(p1, p2, w, c)

    def draw_node_glowy(self, x, y, radius, color):
        for n in range(10):
            r = radius * ((11-n)/10)
            c = tuple([min(255, int(x * (n+1)/5)) for x in color])
            self.draw.ellipse((x-r, y-r, x+r, y+r), fill=c)

    def draw_node(self, x, y, s, c):
        s = int(s * self.mag_factor)
        if self.graph_style == "glowy":
            self.draw_node_glowy(x, y, s, c)
        else:
            self.draw.ellipse((x-s, y-s, x+s, y+s), fill=c, outline=self.node_outline)

    def draw_label(self, x, y, label, s, color=None, fnt=None):
        fill = self.font_color
        if color is not None:
            fill = color
        label = str(label)
        s = int(s * self.mag_factor)
        font = ImageFont.truetype(self.label_font + ".ttf", s)
        if fnt is not None:
            font = ImageFont.truetype(fnt + ".ttf", s)
        llen = len(label)
        xoff = x - (llen * 0.25 * s)
        yoff = y - (s * 0.25)
        self.draw.text((xoff, yoff), label, fill=fill, font=font)

    def draw_multiline_label(self, x, y, text, s, color=None, fnt=None):
        fill = self.font_color
        if color is not None:
            fill = color
        tl = len(text)
        ml = 30
        words = text.split()
        lines = []
        line = ""
        for word in words:
            line += word + " "
            if len(line) > ml:
                lines.append(line.strip())
                line = ""
        if len(line.strip()) > 0:
            lines.append(line.strip())
        nl = len(lines)
        llen = max([len(x) for x in lines])
        font = ImageFont.truetype(self.label_font + ".ttf", s)
        if fnt is not None:
            font = ImageFont.truetype(fnt + ".ttf", s)
        lh = x - (llen * 0.25 * s) - (s * 0.5)
        rh = x + (llen * 0.25 * s) - (s)
        up = y - (s * 0.5)
        dn = y - (s * 0.25) + (s * len(lines)) + (s * 0.25)
        self.draw.rectangle([lh, up, rh, dn],
                             fill=self.background_color,
                             outline=fill,
                             width=1)
        xoff = x - (llen * 0.25 * s)
        for index, line in enumerate(lines):
            yoff = (y - (s * 0.25)) + (s*index)
            self.draw.text((xoff, yoff), line, fill=self.font_color, font=font)

    def draw_modularity_legend(self):
        position = self.modularity_legend
        num_communities = min(15, len(self.extra_vars["communities"]))
        box_height = 25 * num_communities
        box_width = 75
        up = 0
        if "top" in position:
            up = 20
        else:
            up = self.canvas_height - box_height - 20
        lh = 0
        if "left" in position:
            lh = 20
        else:
            lh = self.canvas_width - box_width - 20
        dn = up + box_height + 5
        rh = lh + box_width
        # Draw background box
        self.draw.rectangle([lh, up-5, rh, dn+5],
                             fill=self.background_color,
                             outline=(128,128,128),
                             width=1)
        # Draw legend
        for index in range(num_communities):
            cnum = self.extra_vars["communities"][index]
            ccolor = self.color_palette[cnum]
            xpos = lh + 20
            ypos = up + (25*(index+1)) - 10
            self.draw_node(xpos, ypos, 10, ccolor)
            xpos = lh + 50
            ypos = ypos - 7
            self.draw_label(xpos, ypos, str(cnum), 24, fnt=self.alt_font)

    def draw_info_box(self, val, position, size):
        font_size = size
        box_height = font_size
        box_width = len(val) * (int(font_size/2) + 2)
        up = 0
        if "top" in position:
            up = 20
        else:
            up = self.canvas_height - box_height - 20
        lh = 0
        if "left" in position:
            lh = 20
        else:
            lh = self.canvas_width - box_width - 20
        pad = int(font_size/6)
        dn = up + box_height + pad
        rh = lh + box_width
        # Draw background box
        self.draw.rectangle([lh, up-pad, rh+pad, dn+pad],
                             fill=self.background_color,
                             outline=self.background_color,
                             width=1)
        xpos = lh + int(box_width/2) - pad
        ypos = up + (pad * 2)
        self.draw_label(xpos, ypos, str(val), font_size, fnt=self.alt_font)

    def draw_image(self):
        self.im = Image.new('RGBA',
                            (self.canvas_width, self.canvas_height),
                            self.background_color)
        self.draw = ImageDraw.Draw(self.im)
        self.draw.rectangle([(0,0),(self.canvas_width,self.canvas_height)],
                            fill=self.background_color)

        # Draw edges
        for source, targets in self.inter.items():
            for target, weight in targets.items():
                w = self.set_edge_size(weight)
                mod = self.extra_vars[self.color_by][source]
                color = (128, 128, 128)
                if self.palette == "intense":
                    if mod < len(self.color_palette):
                        color = self.color_palette[mod]
                else:
                    gi = self.get_gradient_index(mod)
                    color = self.color_palette[gi]
                df = 0.25 + (self.eadjust * w)
                color = self.adjust(color, df)
                sp = self.positions[source]
                tp = self.positions[target]
                self.draw_edge(sp, tp, w, color)

        # Draw nodes
        for sn, coords in self.positions.items():
            xpos, ypos = coords
            s = self.extra_vars[self.size_by][sn]
            if s < 1:
                s = 1
            mod = self.extra_vars[self.color_by][sn]
            color = (128, 128, 128)
            if self.palette == "intense":
                if mod < len(self.color_palette):
                    color = self.color_palette[mod]
            else:
                gi = self.get_gradient_index(mod)
                color = self.color_palette[gi]
            node_size = self.set_node_size(s)
            self.draw_node(xpos, ypos, node_size, color)

        # Draw node labels
        for sn, coords in self.positions.items():
            xpos, ypos = coords
            s = self.extra_vars[self.size_by][sn]
            if s < 1:
                s = 1
            text = sn
            if self.labels != "nodeid":
                if sn in self.extra_vars[self.labels]:
                    text = self.extra_vars[self.labels][sn]
            if len(text) > 0:
                font_size = self.set_font_size(s)
                if len(text) <= self.max_label_len:
                    self.draw_label(xpos, ypos, text, font_size)
                else:
                    self.draw_multiline_label(xpos, ypos, text, font_size)

        # Draw edge labels
        if 'edge_labels' in self.extra_vars:
            for source, targets in self.inter.items():
                for target, weight in targets.items():
                    l = str(source) + ":" + str(target)
                    if l in self.extra_vars['edge_labels']:
                        text = self.extra_vars['edge_labels'][l]
                        if len(text) > 0:
                            font_size = self.set_font_size(weight)
                            sp = self.positions[source]
                            tp = self.positions[target]
                            #cp = self.get_control_points(sp, tp)
                            #xpos, ypos = cp[1]
                            xpos, ypos = self.get_midpoint(sp, tp)
                            if len(text) <= self.max_label_len:
                                self.draw_label(xpos, ypos, text, font_size,
                                                color=(128,128,128), fnt=self.alt_font)
                            else:
                                self.draw_multiline_label(xpos, ypos, text, font_size,
                                                          color=(128,128,128), fnt=self.alt_font)

        # Draw info boxes
        if self.modularity_legend is not None:
            self.draw_modularity_legend()

        if self.info is not None:
            for item in self.info:
                val, position, size = item
                self.draw_info_box(val, position, size)

        return self.im

    def interpolate_next(self, g2, num_steps=10):
        iml = []
        g1_sns = set([x for x, c in self.positions.items()])
        g2_sns = set([x for x, c in g2.positions.items()])
        to_move = g1_sns.intersection(g2_sns)
        dists = {}
        angles = {}
        for sn in to_move:
            p1 = self.positions[sn]
            p2 = g2.positions[sn]
            d = self.distance(p1, p2)
            dists[sn] = d
            a = self.angle(p1, p2)
            angles[sn] = a
        max_d = max([c for x, c in dists.items()])
        mean_d = np.mean([c for x, c in dists.items()])
        if mean_d < 10:
            iml.append(g2.make_graphviz())
            return iml
        if num_steps == 0:
            num_steps = int(mean_d * 0.1)
        for step in range(num_steps):
            new_pos = {}
            for sn in to_move:
                p = self.positions[sn]
                if self.interpolation == "lin":
                    d = dists[sn]/num_steps
                elif self.interpolation == "dec":
                    d = dists[sn] * (0.5 ** (step+1))
                else:
                    d = dists[sn] * (0.5 ** (num_steps - step))
                a = angles[sn]
                new_p = self.move_point(p, d, a)
                self.positions[sn] = new_p
            print("Interpolating " + str(step+1) + "/" + str(num_steps))
            iml.append(self.make_graphviz())
        return iml

    def map_community_labels(self, g1, g2):
        mod1 = g1.extra_vars["modularity"]
        mod2 = g2.extra_vars["modularity"]
        id1 = g1.extra_vars["in_degree"]
        clusters = {}
        for x, c in mod1.items():
            if c not in clusters:
                clusters[c] = []
            clusters[c].append(x)
        cluster_in_deg = {}
        mod_map = {}
        for x, c in clusters.items():
            cluster_in_deg[x] = Counter()
            for sn in c:
                cluster_in_deg[x][sn] = id1[sn]
            key_sns = [x for x, c in cluster_in_deg[x].most_common(20)]
            for key in key_sns:
                if key in mod2:
                    mod_map[mod2[key]] = x
                    break
        return mod_map

    def interpolate_multiple(self, glist, savedir, num_steps=10):
        sn_cp = {}
        sn_start = {}
        for index, g in enumerate(glist):
            for sn, pos in g.positions.items():
                if sn not in sn_start:
                    sn_start[sn] = index
                if sn not in sn_cp:
                    sn_cp[sn] = []
                sn_cp[sn].append(pos)
        sn_points = {}
        for sn, cp in sn_cp.items():
            steps = num_steps*(len(cp))
            if steps > 0:
                points = self.make_bezier(cp, steps)
                sn_points[sn] = points
        total_frames = num_steps * len(glist)
        for index, g in enumerate(glist):
            mod_map = None
            if index > 0:
                mod_map = self.map_community_labels(glist[0], glist[index])
                for sn, m in g.extra_vars["modularity"].items():
                    if m in mod_map:
                        g.extra_vars["modularity"][sn] = mod_map[m]
            sns = [x for x, c in g.positions.items()]
            for step in range(num_steps):
                frame = (index * num_steps) + step
                print("Making frame: " + str(frame) + " / " + str(total_frames))
                for sn in sns:
                    if sn in sn_start and sn in sn_points:
                        start_ind = sn_start[sn]
                        points_index = frame - (start_ind*num_steps)
                        if len(sn_points[sn]) > points_index:
                            node_pos = sn_points[sn][points_index]
                            g.positions[sn] = [node_pos[0], node_pos[1]]
                im = g.make_graphviz()
                im.save(savedir + "/frame" + "%05d"%frame + ".png")

    def make_graphviz(self):
        self.draw_image()
        return self.im



# Add support for info box pointing to node


