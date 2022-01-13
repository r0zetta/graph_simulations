import networkx as nx
import community as louvain
from fa2 import ForceAtlas2
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import numpy as np
import math


class GraphViz:
    def __init__(self, inter, degree, initial_pos=None,
                 mag_factor=1.0, scaling=5.0, gravity=20.0, iterations=200,
                 eadjust=0.5, expand=0.3, zoom=[[0.0,0.0],[1.0,1.0]], auto_zoom=True,
                 font_scaling="lin"):
        self.inter = inter
        self.degree = degree
        self.initial_pos = initial_pos
        self.mag_factor = mag_factor
        self.scaling = scaling
        self.gravity = gravity
        self.iterations = iterations
        self.eadjust = eadjust
        self.expand = expand
        self.zoom = zoom
        self.auto_zoom = auto_zoom
        self.font_scaling = font_scaling # lin, pow, sqrt
        self.canvas_width = int(1200 * self.mag_factor)
        self.canvas_height = int(1200 * self.mag_factor)
        self.color_palette = ((0, 131, 182), (255, 75, 0), (32, 198, 0), (255, 84, 255),
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
        self.get_stats()

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

    def make_bezier(self, xys):
        n = len(xys)
        combinations = self.pascal_row(n-1)
        def bezier(ts):
            result = []
            for t in ts:
                tpowers = (t**i for i in range(n))
                upowers = reversed([(1-t)**i for i in range(n)])
                coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
                result.append(
                    tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
            return result
        return bezier

    def pascal_row(self, n, memo={}):
        if n in memo:
            return memo[n]
        result = [1]
        x, numerator = 1, n
        for denominator in range(1, n//2+1):
            x *= numerator
            x /= denominator
            result.append(x)
            numerator -= 1
        if n&1 == 0:
            # n is even
            result.extend(reversed(result[:-1]))
        else:
            result.extend(reversed(result))
        memo[n] = result
        return result

    def draw_edge(self, p1, p2, w, c):
        control_points = self.get_control_points(p1, p2)
        bez = self.make_bezier(control_points)
        ts = [t/100.0 for t in range(101)]
        points = bez(ts)
        for index in range(len(points)-1):
            x1 = points[index][0]
            y1 = points[index][1]
            x2 = points[index+1][0]
            y2 = points[index+1][1]
            self.draw.line((x1, y1, x2, y2), fill=c, width=w)

    def draw_node(self, x, y, s, c):
        s = int(s * self.mag_factor)
        self.draw.ellipse((x-s, y-s, x+s, y+s), fill=c, outline=(255, 255, 255))

    def draw_label(self, x, y, label, s):
        label = str(label)
        s = int(s * self.mag_factor)
        font = ImageFont.truetype("Arial Bold.ttf", s)
        llen = len(label)
        xoff = x - (llen * 0.25 * s)
        yoff = y - (s * 0.25)
        self.draw.text((xoff, yoff), label, fill=(255,255,255), font=font)

    def convert_coords(self, x, y):
        zxmin, zymin = self.zoom[0]
        zxmax, zymax = self.zoom[1]
        zwidth = zxmax - zxmin
        zheight = zymax - zymin
        coords_width = (self.max_x - self.min_x) * zwidth
        coords_height = (self.max_y - self.min_y) * zheight
        width_ratio = self.canvas_width/coords_width
        height_ratio = self.canvas_height/coords_height
        new_x = (x - self.min_x)*width_ratio - (self.canvas_width * (zxmin/zwidth))
        new_y = (y - self.min_y)*height_ratio - (self.canvas_height * (zymin/zheight))
        return new_x, new_y

    def expand_graph(self, pos):
        max_x = max([c[0] for x, c in pos.items()])
        min_x = min([c[0] for x, c in pos.items()])
        max_y = max([c[1] for x, c in pos.items()])
        min_y = min([c[1] for x, c in pos.items()])
        mid_x = (max_x - min_x)/2
        mid_y = (max_y - min_y)/2
        mid_p = (mid_x, mid_y)
        new_pos = {}
        for sn, coords in pos.items():
            dist = self.distance(mid_p, coords)
            angle = self.angle(mid_p, coords)
            new_coords = self.move_point(coords, dist * self.expand, angle)
            new_pos[sn] = new_coords
        return new_pos

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
        for source, targets in self.inter.items():
            for target, count in targets.items():
                if count > self.max_weight:
                    self.max_weight = count
                mapping.append((source, target, count))

        self.G = nx.Graph()
        self.G.add_weighted_edges_from(mapping)
        self.communities = louvain.best_partition(self.G)

        self.clusters = {}
        for node, mod in self.communities.items():
            if mod not in self.clusters:
                self.clusters[mod] = []
            self.clusters[mod].append(node)

        FA2 = ForceAtlas2(self.G, scalingRatio=self.scaling, gravity=self.gravity, verbose=False)
        pos = FA2.forceatlas2_networkx_layout(self.G, pos=self.initial_pos, iterations=self.iterations)
        if self.expand > 0:
            pos = self.expand_graph(pos)
        if self.auto_zoom == True:
            self.autozoom(pos)
        self.max_x = max([c[0] for x, c in pos.items()])
        self.min_x = min([c[0] for x, c in pos.items()])
        self.max_y = max([c[1] for x, c in pos.items()])
        self.min_y = min([c[1] for x, c in pos.items()])

        self.positions = {}
        for sn, coords in pos.items():
            x, y = coords
            newx, newy = self.convert_coords(x, y)
            self.positions[sn] = [newx, newy]

        self.max_degree = max([c for x, c in self.degree.items()])
        self.min_degree = min([c for x, c in self.degree.items()])
        self.node_sizes = dict([(node, 5+(5*(degree/self.max_degree))) for node, degree in self.degree.items()])
        self.max_node_size = max([c for x, c in self.node_sizes.items()])
        self.min_node_size = min([c for x, c in self.node_sizes.items()])

        self.modularity_class = {}
        for community_number, community in self.clusters.items():
            for name in community: 
                self.modularity_class[name] = community_number


    def set_font_size(self, s):
        if self.font_scaling == "lin":
            return 1 + int(23 * (s/self.max_node_size))
        elif self.font_scaling == "pow":
            return 1 + int(23 * ((s ** 2)/(self.max_node_size ** 2)))
        else:
            return 1 + int(23 * (math.sqrt(s)/math.sqrt(self.max_node_size)))

    def draw_image(self):
        self.im = Image.new('RGB', (self.canvas_width, self.canvas_height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.im)

        # Draw edges
        for source, targets in self.inter.items():
            for target, weight in targets.items():
                w = int((weight*3)/self.max_weight)
                mod = self.modularity_class[source]
                color = (128, 128, 128)
                if mod < len(self.color_palette):
                    color = self.color_palette[mod]
                df = 0.25 + (self.eadjust * w)
                color = self.adjust(color, df)
                sp = self.positions[source]
                tp = self.positions[target]
                self.draw_edge(sp, tp, w, color)

        # Draw nodes and labels
        for sn, coords in self.positions.items():
            xpos, ypos = coords
            s = 1
            if sn in self.node_sizes:
                s = self.node_sizes[sn]
            mod = 1
            if sn in self.modularity_class:
                mod = self.modularity_class[sn]
            color = (128, 128, 128)
            if mod < len(self.color_palette):
                color = self.color_palette[mod]
            font_size = self.set_font_size(s)
            self.draw_node(xpos, ypos, s, color)
            self.draw_label(xpos, ypos, sn, font_size)

        return self.im

    def interpolate(self, g2):
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
        num_steps = 10
        for step in range(num_steps):
            new_pos = {}
            for sn in to_move:
                p = self.positions[sn]
                d = dists[sn] * 0.5
                dists[sn] = d
                a = angles[sn]
                new_p = self.move_point(p, d, a)
                self.positions[sn] = new_p
            print("Interpolating " + str(step+1) + "/" + str(num_steps))
            iml.append(self.make_graphviz())
        return iml

    def make_graphviz(self):
        self.draw_image()
        return self.im

