from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
import numpy as np
from matplotlib.pylab import plt
import cv2
from other_tools import sknw
import os
import pandas as pd
from functools import partial
from itertools import tee
from scipy.spatial.distance import pdist, squareform
from scipy import ndimage as ndi
from collections import defaultdict, OrderedDict
import sys

from multiprocessing import Pool

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def remove_sequential_duplicates(seq):
    #todo
    res = [seq[0]]
    for elem in seq[1:]:
        if elem == res[-1]:
            continue
        res.append(elem)
    return res

def remove_duplicate_segments(seq):
    seq = remove_sequential_duplicates(seq)
    segments = set()
    split_seg = []
    res = []
    for idx, (s, e) in enumerate(pairwise(seq)):
        if (s, e) not in segments and (e, s) not in segments:
            segments.add((s, e))
            segments.add((e, s))
        else:
            split_seg.append(idx+1)
    for idx, v in enumerate(split_seg):
        if idx == 0:
            res.append(seq[:v])
        if idx == len(split_seg) - 1:
            res.append(seq[v:])
        else:
            s = seq[split_seg[idx-1]:v]
            if len(s) > 1:
                res.append(s)
    if not len(split_seg):
        res.append(seq)
    return res


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_angle(p0, p1=np.array([0,0]), p2=None):
    """ compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    """
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def preprocess(img, thresh):
    img = (img > (255 * thresh)).astype(np.bool)
    remove_small_objects(img, 300, in_place=True)
    remove_small_holes(img, 300, in_place=True)
    # img = cv2.dilate(img.astype(np.uint8), np.ones((7, 7)))
    return img

def graph2lines(G):
    node_lines = []
    edges = list(G.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines


def visualize(img, G, vertices):
    plt.imshow(img, cmap='gray')

    # draw edges by pts
    for (s, e) in G.edges():
        vals = flatten([[v] for v in G[s][e].values()])
        for val in vals:
            ps = val.get('pts', [])
            plt.plot(ps[:, 1], ps[:, 0], 'green')

    # draw node by o
    node, nodes = G.node(), G.nodes
    # deg = G.degree
    # ps = np.array([node[i]['o'] for i in nodes])
    ps = np.array(vertices)
    plt.plot(ps[:, 1], ps[:, 0], 'r.')

    # title and show
    plt.title('Build Graph')
    plt.show()

def line_points_dist(line1, pts):
    return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(line1[1] - line1[0])

def remove_small_terminal(G):
    deg = G.degree()
    terminal_points = [i for i, d in deg.items() if d == 1]
    edges = list(G.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            vals = flatten([[v] for v in G[s][s].values()])
            for ix, val in enumerate(vals):
                sum_len += len(val['pts'])
            if sum_len < 3:
                G.remove_edge(s, e)
                continue
        vals = flatten([[v] for v in G[s][e].values()])
        for ix, val in enumerate(vals):
            if s in terminal_points and val.get('weight', 0) < 10:
                G.remove_node(s)
            if e in terminal_points and val.get('weight', 0) < 10:
                G.remove_node(e)
    return


linestring = "LINESTRING {}"
def make_skeleton(root, fn, debug, threshes, fix_borders):
    replicate = 5
    clip = 2
    rec = replicate + clip
    # open and skeletonize
    img = cv2.imread(os.path.join(root, fn), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (1300, 1300)
    if fix_borders:
        img = cv2.copyMakeBorder(img, replicate, replicate, replicate, replicate, cv2.BORDER_REPLICATE)
    img_copy = None
    if debug:
        if fix_borders:
            img_copy = np.copy(img[replicate:-replicate,replicate:-replicate])
        else:
            img_copy = np.copy(img)
    thresh = threshes[fn[4]]
    img = preprocess(img, thresh)
    if not np.any(img):
        return None, None
    ske = skeletonize(img).astype(np.uint16)
    if fix_borders:
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)
    return img_copy, ske


def add_small_segments(G, terminal_points, terminal_lines):
    node = G.node
    term = [node[t]['o'] for t in terminal_points]
    dists = squareform(pdist(term))
    possible = np.argwhere((dists > 0) & (dists < 20))
    good_pairs = []
    for s, e in possible:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]

        if G.has_edge(s, e):
            continue
        good_pairs.append((s, e))

    possible2 = np.argwhere((dists > 20) & (dists < 100))
    for s, e in possible2:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]
        if G.has_edge(s, e):
            continue
        l1 = terminal_lines[s]
        l2 = terminal_lines[e]
        d = line_points_dist(l1, l2[0])

        if abs(d) > 20:
            continue
        angle = get_angle(l1[1] - l1[0], np.array((0, 0)), l2[1] - l2[0])
        if -20 < angle < 20 or angle < -160 or angle > 160:
            good_pairs.append((s, e))

    dists = {}
    for s, e in good_pairs:
        s_d, e_d = [G.node[s]['o'], G.node[e]['o']]
        dists[(s, e)] = np.linalg.norm(s_d - e_d)

    dists = OrderedDict(sorted(dists.items(), key=lambda x: x[1]))

    wkt = []
    added = set()
    for s, e in dists.keys():
        if s not in added and e not in added:
            added.add(s)
            added.add(e)
            s_d, e_d = G.node[s]['o'], G.node[e]['o']
            line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in [s_d, e_d]]
            line = '(' + ", ".join(line_strings) + ')'
            wkt.append(linestring.format(line))
    return wkt


def add_direction_change_nodes(pts, s, e, s_coord, e_coord):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 2
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps


def build_graph(root, fn, debug=False, threshes={'2': .3, '3': .3, '4': .3, '5': .2}, add_small=True, fix_borders=True):
    city = os.path.splitext(fn)[0][5:]
    img_copy, ske = make_skeleton(root, fn, debug, threshes, fix_borders)
    if ske is None:
        return city, [linestring.format("EMPTY")]
    G = sknw.build_sknw(ske, multi=True)
    remove_small_terminal(G)
    node_lines = graph2lines(G)
    if not node_lines:
        return city, [linestring.format("EMPTY")]
    node = G.node
    deg = G.degree()
    wkt = []
    terminal_points = [i for i, d in deg.items() if d == 1]

    terminal_lines = {}
    vertices = []
    for w in node_lines:
        coord_list = []
        additional_paths = []
        for s, e in pairwise(w):
            vals = flatten([[v] for v in G[s][e].values()])
            for ix, val in enumerate(vals):

                s_coord, e_coord = node[s]['o'], node[e]['o']
                pts = val.get('pts', [])
                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)

                ps = add_direction_change_nodes(pts, s, e, s_coord, e_coord)

                if len(ps.shape) < 2 or len(ps) < 2:
                    continue

                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue

                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
                if ix == 0:
                    coord_list.extend(line_strings)
                else:
                    additional_paths.append(line_strings)

                vertices.append(ps)

        if not len(coord_list):
            continue
        segments = remove_duplicate_segments(coord_list)
        for coord_list in segments:
            if len(coord_list) > 1:
                line = '(' + ", ".join(coord_list) + ')'
                wkt.append(linestring.format(line))
        for line_strings in additional_paths:
            line = ", ".join(line_strings)
            line_rev = ", ".join(reversed(line_strings))
            for s in wkt:
                if line in s or line_rev in s:
                    break
            else:
                wkt.append(linestring.format('(' + line + ')'))

    if add_small and len(terminal_points) > 1:
        wkt.extend(add_small_segments(G, terminal_points, terminal_lines))

    if debug:
        vertices = flatten(vertices)
        visualize(img_copy, G, vertices)

    if not wkt:
        return city, [linestring.format("EMPTY")]
    return city, wkt

if __name__ == "__main__":
    prefix = 'AOI'
    results_root = r'/results/results'
    # results_root = r'd:\tmp\roads\albu\results\results'
    # root = os.path.join(results_root, r'results\2m_4fold_512_30e_d0.2_g0.2')
    root = os.path.join(results_root, r'2m_4fold_512_30e_d0.2_g0.2_test', 'merged')
    f = partial(build_graph, root)
    l = [v for v in os.listdir(root) if prefix in v]
    l = list(sorted(l))
    with Pool() as p:
        data = p.map(f, l)
    all_data = []
    for k, v in data:
        for val in v:
            all_data.append((k, val))
    df = pd.DataFrame(all_data, columns=['ImageId', 'WKT_Pix'])
    df.to_csv(sys.argv[1] + '.txt', index=False)
