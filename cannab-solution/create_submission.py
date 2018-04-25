# -*- coding: utf-8 -*-
from os import path, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import pandas as pd
import timeit
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import sys
from shapely.geometry.linestring import LineString
from skimage.morphology import skeletonize_3d, square, erosion, dilation, medial_axis
from skimage.measure import label, regionprops, approximate_polygon
from math import hypot, sin, cos, asin, acos, radians
from sklearn.neighbors import KDTree
from shapely.wkt import dumps

pred_folder = '/wdata/predictions'
test_folders = []
cities = ['Vegas', 'Paris', 'Shanghai', 'Khartoum']

def get_ordered_coords(lbl, l, coords):
    res = []
    
    nxt = coords[0]
    for i in range(coords.shape[0]):
        y, x = coords[i]
        cnt = 0
        for y0 in range(max(0, y-1), min(1300, y+2)):
            for x0 in range(max(0, x-1), min(1300, x+2)):
                if lbl[y0, x0] == l:
                    cnt += 1
        if cnt == 2:
            nxt = coords[i]
            lbl[y, x] = 0
            res.append(nxt)
            break
    while nxt is not None:
        y, x = nxt
        fl = False
        for y0 in range(max(0, y-1), min(1300, y+2)):
            for x0 in range(max(0, x-1), min(1300, x+2)):
                if lbl[y0, x0] == l:
                    fl = True
                    nxt = np.asarray([y0, x0])
                    lbl[y0, x0] = 0
                    res.append(nxt)
                    break
            if fl:
                break
        if not fl:
            nxt = None
    return np.asarray(res)
    
def point_hash(x, y):
    x_int = int(x)
    y_int = int(y)
    h = x_int * 10000 + y_int
    return h

def pair_hash(p1, p2):
    h1 = point_hash(p1[0], p1[1])
    h2 = point_hash(p2[0], p2[1])
    return np.int64(h1 * 1e8 + h2)

def get_next_point(p1, p2, add_dist):
    l = hypot(p1[0] - p2[0], p1[1] - p2[1])
    dx = (p2[1] - p1[1]) / l
    dy = (p2[0] - p1[0]) / l
    x3 = int(round(p1[1] + (l + add_dist) * dx))
    y3 = int(round(p1[0] + (l + add_dist) * dy))
    if x3 < 0:
        x3 = 0
    if y3 < 0:
        y3 = 0
    if x3 > 1299:
        x3 = 1299
    if y3 > 1299:
        y3 = 1299
    return np.asarray([y3, x3])
   
def try_connect(p1, p2, a, max_dist, road_msk, min_prob, msk, roads, r=3):
    prob = []
    r_id = road_msk[p2[0], p2[1]]
    hashes = set([point_hash(p2[1], p2[0])])
    l = hypot(p1[0] - p2[0], p1[1] - p2[1])
    dx = (p2[1] - p1[1]) / l
    dy = (p2[0] - p1[0]) / l
    if a != 0:
        l = 0
        p1 = p2
        _a = asin(dy) + a
        dy = sin(_a)
        _a = acos(dx) + a
        dx = cos(_a)
    step = 0.5
    d = step
    while d < max_dist:
        x3 = int(round(p1[1] + (l + d) * dx))
        y3 = int(round(p1[0] + (l + d) * dy))
        if x3 < 0 or y3 < 0 or x3 > 1299 or y3 > 1299:
            return None
        h = point_hash(x3, y3)
        if h not in hashes:
            prob.append(msk[y3, x3])
            hashes.add(h)
            for x0 in range(max(0, x3 - r), min(1299, x3 + r + 1)):
                for y0 in range(max(0, y3 - r), min(1299, y3 + r + 1)):
                    if road_msk[y0, x0] > 0 and road_msk[y0, x0] != r_id:
                        p3 = np.asarray([y0, x0])
                        r2_id = road_msk[y0, x0] - 1
                        t = KDTree(roads[r2_id])
                        clst = t.query(p3[np.newaxis, :])
                        if clst[0][0][0] < 10:
                            p3 = roads[r2_id][clst[1][0][0]]
                        if np.asarray(prob).mean() > min_prob:
                            return p3
                        else:
                            return None
        d += step
    return None

def inject_point(road, p):
    for i in range(road.shape[0]):
        if road[i, 0] == p[0] and road[i, 1] == p[1]:
            return road, []
    new_road = []
    to_add = True
    new_hashes = []
    for i in range(road.shape[0] - 1):
        new_road.append(road[i])
        if (to_add and min(road[i, 0], road[i+1, 0]) <= p[0] 
            and max(road[i, 0], road[i+1, 0]) >= p[0]
            and min(road[i, 1], road[i+1, 1]) <= p[1]
            and max(road[i, 1], road[i+1, 1]) >= p[1]):
            to_add = False
            new_road.append(p)
            new_hashes.append(pair_hash(p, road[i]))
            new_hashes.append(pair_hash(road[i], p))
            new_hashes.append(pair_hash(p, road[i+1]))
            new_hashes.append(pair_hash(road[i+1], p))
    new_road.append(road[-1])
    return np.asarray(new_road), new_hashes

    
def process_file(img_id, par, par2, vgg_big_path, vgg_small_path, linknet_small_path, small_res_file_path, inc_file_path, 
                 vgg_smallest_file_path, inc_smallest_file_path, res_smallest_file_path, inc3_520_file_path, inc_v2_520_file_path,
                 linknet_big_file_path, linknet_520_file_path,
                 vgg_big_path_1, vgg_smallest_file_path_1, 
                 inc_smallest_file_path_1, res_smallest_file_path_1, inc3_520_file_path_1, inc_v2_520_file_path_1, 
                  linknet_big_file_path_1, linknet_520_file_path_1, save_to=None):
    res_rows = []
    
    if vgg_small_path is None:
        msk = np.zeros((1300, 1300))
    else:
        msk = cv2.imread(vgg_small_path, cv2.IMREAD_UNCHANGED)
        msk = cv2.resize(msk, (1300, 1300))
    if linknet_small_path is None:
        msk2 = np.zeros((1300, 1300))
    else:
        msk2 = cv2.imread(linknet_small_path, cv2.IMREAD_UNCHANGED)
        msk2 = cv2.resize(msk2, (1300, 1300))
    if vgg_big_path is None:
        msk3 = np.zeros((1300, 1300))
        msk3_1 = np.zeros((1300, 1300))
    else:
        msk3 =  cv2.imread(vgg_big_path, cv2.IMREAD_UNCHANGED)
        msk3_1 =  cv2.imread(vgg_big_path_1, cv2.IMREAD_UNCHANGED)
    if small_res_file_path is None:
        res_msk = np.zeros((1300, 1300))
    else:
        res_msk = cv2.imread(small_res_file_path, cv2.IMREAD_UNCHANGED)
        res_msk = cv2.resize(res_msk, (1300, 1300))
    if inc_file_path is None:
        inc_msk = np.zeros((1300, 1300))
    else:
        inc_msk = cv2.imread(inc_file_path, cv2.IMREAD_UNCHANGED)
        inc_msk = cv2.resize(inc_msk, (1300, 1300))
    if vgg_smallest_file_path is None:
        vgg_smlst_msk = np.zeros((1300, 1300))
        vgg_smlst_msk_1 = np.zeros((1300, 1300))
    else:
        vgg_smlst_msk = cv2.imread(vgg_smallest_file_path, cv2.IMREAD_UNCHANGED)
        vgg_smlst_msk = cv2.resize(vgg_smlst_msk, (1300, 1300))
        vgg_smlst_msk_1 = cv2.imread(vgg_smallest_file_path_1, cv2.IMREAD_UNCHANGED)
        vgg_smlst_msk_1 = cv2.resize(vgg_smlst_msk_1, (1300, 1300))
    if inc_smallest_file_path is None:
        inc_smlst_msk = np.zeros((1300, 1300))
        inc_smlst_msk_1 = np.zeros((1300, 1300))
    else:
        inc_smlst_msk = cv2.imread(inc_smallest_file_path, cv2.IMREAD_UNCHANGED)
        inc_smlst_msk = cv2.resize(inc_smlst_msk, (1300, 1300))
        inc_smlst_msk_1 = cv2.imread(inc_smallest_file_path_1, cv2.IMREAD_UNCHANGED)
        inc_smlst_msk_1 = cv2.resize(inc_smlst_msk_1, (1300, 1300))
    if res_smallest_file_path is None:
        res_smlst_msk = np.zeros((1300, 1300))
        res_smlst_msk_1 = np.zeros((1300, 1300))
    else:
        res_smlst_msk = cv2.imread(res_smallest_file_path, cv2.IMREAD_UNCHANGED)
        res_smlst_msk = cv2.resize(res_smlst_msk, (1300, 1300))
        res_smlst_msk_1 = cv2.imread(res_smallest_file_path_1, cv2.IMREAD_UNCHANGED)
        res_smlst_msk_1 = cv2.resize(res_smlst_msk_1, (1300, 1300))
    if inc3_520_file_path is None:
        inc3_520_msk = np.zeros((1300, 1300))
        inc3_520_msk_1 = np.zeros((1300, 1300))
    else:
        inc3_520_msk = cv2.imread(inc3_520_file_path, cv2.IMREAD_UNCHANGED)
        inc3_520_msk = cv2.resize(inc3_520_msk, (1300, 1300))
        inc3_520_msk_1 = cv2.imread(inc3_520_file_path_1, cv2.IMREAD_UNCHANGED)
        inc3_520_msk_1 = cv2.resize(inc3_520_msk_1, (1300, 1300))
    if inc_v2_520_file_path is None:
        inc_v2_520_msk = np.zeros((1300, 1300))
        inc_v2_520_msk_1 = np.zeros((1300, 1300))
    else:
        inc_v2_520_msk = cv2.imread(inc_v2_520_file_path, cv2.IMREAD_UNCHANGED)
        inc_v2_520_msk = cv2.resize(inc_v2_520_msk, (1300, 1300))
        inc_v2_520_msk_1 = cv2.imread(inc_v2_520_file_path_1, cv2.IMREAD_UNCHANGED)
        inc_v2_520_msk_1 = cv2.resize(inc_v2_520_msk_1, (1300, 1300))
    if linknet_big_file_path is None:
        link_big_msk = np.zeros((1300, 1300))
        link_big_msk_1 = np.zeros((1300, 1300))
    else:
        link_big_msk = cv2.imread(linknet_big_file_path, cv2.IMREAD_UNCHANGED)
        link_big_msk_1 = cv2.imread(linknet_big_file_path_1, cv2.IMREAD_UNCHANGED)
    if linknet_520_file_path is None:
        link_520_msk = np.zeros((1300, 1300))
        link_520_msk_1 = np.zeros((1300, 1300))
    else:
        link_520_msk = cv2.imread(linknet_520_file_path, cv2.IMREAD_UNCHANGED)
        link_520_msk = cv2.resize(link_520_msk, (1300, 1300))
        link_520_msk_1 = cv2.imread(linknet_520_file_path_1, cv2.IMREAD_UNCHANGED)
        link_520_msk_1 = cv2.resize(link_520_msk_1, (1300, 1300))
    
    msk3 = (msk3 * 0.5 + msk3_1 * 0.5)
    inc_smlst_msk = (inc_smlst_msk * 0.5 + inc_smlst_msk_1 * 0.5)
    vgg_smlst_msk = (vgg_smlst_msk * 0.5 + vgg_smlst_msk_1 * 0.5)
    res_smlst_msk = (res_smlst_msk * 0.5 + res_smlst_msk_1 * 0.5)
    inc3_520_msk = (inc3_520_msk * 0.5 + inc3_520_msk_1 * 0.5)
    inc_v2_520_msk = (inc_v2_520_msk * 0.5 + inc_v2_520_msk_1 * 0.5)
    link_big_msk = (link_big_msk * 0.5 + link_big_msk_1 * 0.5)
    link_520_msk = (link_520_msk * 0.5 + link_520_msk_1 * 0.5)
    
    coef = []
    tot_sum = par[:12].sum()
    for i in range(12):
        coef.append(par[i] / tot_sum)
    msk = (msk * coef[0] + msk2 * coef[1] + msk3 * coef[2] + res_msk * coef[3] + inc_msk * coef[4]
             + vgg_smlst_msk * coef[5]  + inc_smlst_msk * coef[6] + res_smlst_msk * coef[7] 
             + inc3_520_msk * coef[8] + inc_v2_520_msk * coef[9] + link_big_msk * coef[10] + link_520_msk * coef[11])
    msk = msk.astype('uint8')
    if save_to is not None:
        cv2.imwrite(save_to, msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    msk2 = np.lib.pad(msk, ((22, 22), (22, 22)), 'symmetric')
    
    thr = par[12]
        
    msk2 = 1 * (msk2 > thr)
    msk2 = msk2.astype(np.uint8)
    
    if par2[0] > 0:
        msk2 = dilation(msk2, square(par2[0]))
    if par2[1] > 0:
        msk2 = erosion(msk2, square(par2[1]))
        
    if 'Shanghai' in img_id:
        skeleton = medial_axis(msk2)
    else:
        skeleton = skeletonize_3d(msk2)
    skeleton = skeleton[22:1322, 22:1322]
    
    lbl0 = label(skeleton)
    props0 = regionprops(lbl0)
    
    cnt = 0
    crosses = []
    for x in range(1300):
        for y in range(1300):
            if skeleton[y, x] == 1:
                if skeleton[max(0, y-1):min(1300, y+2), max(0, x-1):min(1300, x+2)].sum() > 3:
                    cnt += 1
                    crss = []
                    crss.append((x, y))
                    for y0 in range(max(0, y-1), min(1300, y+2)):
                        for x0 in range(max(0, x-1), min(1300, x+2)):
                            if x == x0 and y == y0:
                                continue
                            if skeleton[max(0, y0-1):min(1300, y0+2), max(0, x0-1):min(1300, x0+2)].sum() > 3:
                                crss.append((x0, y0))
                    crosses.append(crss)
    cross_hashes = []
    for crss in crosses:
        crss_hash = set([])
        for x0, y0 in crss:
            crss_hash.add(point_hash(x0, y0))
            skeleton[y0, x0] = 0
        cross_hashes.append(crss_hash)
 
    new_crosses = []
    i = 0
    while i < len(crosses):
        new_hashes = set([])
        new_hashes.update(cross_hashes[i])
        new_crss = crosses[i][:]
        fl = True
        while fl:
            fl = False
            j = i + 1
            while j < len(crosses):
                if len(new_hashes.intersection(cross_hashes[j])) > 0:
                    new_hashes.update(cross_hashes[j])
                    new_crss.extend(crosses[j])
                    cross_hashes.pop(j)
                    crosses.pop(j)
                    fl = True
                    break
                j += 1
        mean_p = np.asarray(new_crss).mean(axis=0).astype('int')
        if len(new_crss) > 1:
            t = KDTree(new_crss)
            mean_p = new_crss[t.query(mean_p[np.newaxis, :])[1][0][0]]
        new_crosses.append([(mean_p[0], mean_p[1])] + new_crss)
        i += 1
    crosses = new_crosses
    
    lbl = label(skeleton)
    props = regionprops(lbl)
    
    connected_roads = []
    connected_crosses = [set([]) for p in props]
    for i in range(len(crosses)):
        rds = set([])
        for j in range(len(crosses[i])):
            x, y = crosses[i][j]
            for y0 in range(max(0, y-1), min(1300, y+2)):
                for x0 in range(max(0, x-1), min(1300, x+2)):
                    if lbl[y0, x0] > 0:
                        rds.add(lbl[y0, x0])
                        connected_crosses[lbl[y0, x0]-1].add(i)
        connected_roads.append(rds)
    
    res_roads = []
    
    tot_dist_min = par2[2]
    coords_min = par2[3]
        
    for i in range(len(props)):
        coords = props[i].coords
        crss = list(connected_crosses[i])
        tot_dist = props0[lbl0[coords[0][0], coords[0][1]]-1].area

        if (tot_dist < tot_dist_min) or (coords.shape[0] < coords_min and len(crss) < 2):
            continue
        if coords.shape[0] == 1:
            coords = np.asarray([coords[0], coords[0]])
        else:
            coords = get_ordered_coords(lbl, i+1, coords)
        for j in range(len(crss)):
            x, y = crosses[crss[j]][0]
            d1 = abs(coords[0][0] - y) + abs(coords[0][1] - x)
            d2 = abs(coords[-1][0] - y) + abs(coords[-1][1] - x)
            if d1 < d2:
                coords[0][0] = y
                coords[0][1] = x
            else:
                coords[-1][0] = y
                coords[-1][1] = x
        coords_approx = approximate_polygon(coords, 1.5)
        res_roads.append(coords_approx)
        
    hashes = set([])
    final_res_roads = []
    for r in res_roads:
        if r.shape[0] > 2:
            final_res_roads.append(r)
            for i in range(1, r.shape[0]):
                p1 = r[i-1]
                p2 = r[i]
                h1 = pair_hash(p1, p2)
                h2 = pair_hash(p2, p1)
                hashes.add(h1)
                hashes.add(h2)
                            
    for r in res_roads:
        if r.shape[0] == 2:
            p1 = r[0]
            p2 = r[1]
            h1 = pair_hash(p1, p2)
            h2 = pair_hash(p2, p1)
            if not (h1 in hashes or h2 in hashes):
                final_res_roads.append(r)
                hashes.add(h1)
                hashes.add(h2)
        
    end_points = {}
    for r in res_roads:
        h = point_hash(r[0, 0], r[0, 1])
        if not (h in end_points.keys()):
            end_points[h] = 0
        end_points[h] = end_points[h] + 1
        h = point_hash(r[-1, 0], r[-1, 1])
        if not (h in end_points.keys()):
            end_points[h] = 0
        end_points[h] = end_points[h] + 1
    
    road_msk = np.zeros((1300, 1300), dtype=np.int32)
    road_msk = road_msk.copy()
    thickness = 1
    for j in range(len(final_res_roads)):
        l = final_res_roads[j]
        for i in range(len(l) - 1):
            cv2.line(road_msk, (int(l[i, 1]), int(l[i, 0])), (int(l[i+1, 1]), int(l[i+1, 0])), j+1, thickness)
            
    connect_dist = par2[4]

    min_prob = par2[5]
    angles_to_check = [0, radians(5), radians(-5), radians(10), radians(-10), radians(15), radians(-15)]
    if 'Paris' in img_id or 'Vegas' in img_id:
        angles_to_check += [radians(20), radians(-20), radians(25), radians(-25)]
    
    add_dist = par2[6]
    add_dist2 = par2[7]
    
    con_r = par2[8]

    for i in range(len(final_res_roads)):
        h = point_hash(final_res_roads[i][0, 0], final_res_roads[i][0, 1])
        if end_points[h] == 1:
            p1 = final_res_roads[i][1]
            p2 = final_res_roads[i][0]            
            p3 = try_connect(p1, p2, 0, connect_dist, road_msk, min_prob, msk, final_res_roads, con_r)
            if p3 is not None:
                h1 = pair_hash(p2, p3)
                h2 = pair_hash(p3, p2)
                if not (h1 in hashes or h2 in hashes):
                    r_id = road_msk[p3[0], p3[1]] - 1
                    final_res_roads[r_id], new_hashes = inject_point(final_res_roads[r_id], p3)
                    hashes.update(new_hashes)          
                    tmp_road_msk = np.zeros((1300, 1300), dtype=np.int32)
                    tmp_road_msk = tmp_road_msk.copy()
                    cv2.line(tmp_road_msk, (p2[1], p2[0]), (p3[1], p3[0]), i+1, thickness)
                    road_msk[road_msk == 0] = tmp_road_msk[road_msk == 0]
                    road_msk = road_msk.copy()
                    final_res_roads[i] = np.vstack((p3, final_res_roads[i]))
                    hashes.add(h1)
                    hashes.add(h2)
                    end_points[point_hash(p3[0], p3[1])] = 2
        h = point_hash(final_res_roads[i][-1, 0], final_res_roads[i][-1, 1])
        if end_points[h] == 1:
            p1 = final_res_roads[i][-2]
            p2 = final_res_roads[i][-1]
            p3 = try_connect(p1, p2, 0, connect_dist, road_msk, min_prob, msk, final_res_roads, con_r)
            if p3 is not None:
                h1 = pair_hash(p2, p3)
                h2 = pair_hash(p3, p2)
                if not (h1 in hashes or h2 in hashes):
                    r_id = road_msk[p3[0], p3[1]] - 1
                    final_res_roads[r_id], new_hashes = inject_point(final_res_roads[r_id], p3)
                    hashes.update(new_hashes)
                    tmp_road_msk = np.zeros((1300, 1300), dtype=np.int32)
                    tmp_road_msk = tmp_road_msk.copy()
                    cv2.line(tmp_road_msk, (p2[1], p2[0]), (p3[1], p3[0]), i+1, thickness)
                    road_msk[road_msk == 0] = tmp_road_msk[road_msk == 0]
                    road_msk = road_msk.copy()
                    final_res_roads[i] = np.vstack((final_res_roads[i], p3))
                    hashes.add(h1)
                    hashes.add(h2)
                    end_points[point_hash(p3[0], p3[1])] = 2
                        
    for i in range(len(final_res_roads)):
        h = point_hash(final_res_roads[i][0, 0], final_res_roads[i][0, 1])
        if end_points[h] == 1:
            p1 = final_res_roads[i][1]
            p2 = final_res_roads[i][0]
            p3 = None
            for a in angles_to_check:
                p3 = try_connect(p1, p2, a, connect_dist, road_msk, min_prob, msk, final_res_roads, con_r)
                if p3 is not None:
                    break
            if p3 is not None:
                h1 = pair_hash(p2, p3)
                h2 = pair_hash(p3, p2)
                if not (h1 in hashes or h2 in hashes):
                    r_id = road_msk[p3[0], p3[1]] - 1
                    final_res_roads[r_id], new_hashes = inject_point(final_res_roads[r_id], p3)
                    hashes.update(new_hashes)          
                    tmp_road_msk = np.zeros((1300, 1300), dtype=np.int32)
                    tmp_road_msk = tmp_road_msk.copy()
                    cv2.line(tmp_road_msk, (p2[1], p2[0]), (p3[1], p3[0]), i+1, thickness)
                    road_msk[road_msk == 0] = tmp_road_msk[road_msk == 0]
                    road_msk = road_msk.copy()
                    final_res_roads[i] = np.vstack((p3, final_res_roads[i]))
                    hashes.add(h1)
                    hashes.add(h2)
                    end_points[point_hash(p3[0], p3[1])] = 2
            else:
                p3 = get_next_point(p1, p2, add_dist)
                if not (p3[0] < 2 or p3[1] < 2 or p3[0] > 1297 or p3[1] > 1297):
                    p3 = get_next_point(p1, p2, add_dist2)
                if (p3[0] != p2[0] or p3[1] != p2[1]) and (road_msk[p3[0], p3[1]] == 0):
                    h1 = pair_hash(p2, p3)
                    h2 = pair_hash(p3, p2)
                    if not (h1 in hashes or h2 in hashes):
                        final_res_roads[i] = np.vstack((p3, final_res_roads[i]))
                        hashes.add(h1)
                        hashes.add(h2)
                        tmp_road_msk = np.zeros((1300, 1300), dtype=np.int32)
                        tmp_road_msk = tmp_road_msk.copy()
                        cv2.line(tmp_road_msk, (p2[1], p2[0]), (p3[1], p3[0]), i+1, thickness)
                        road_msk[road_msk == 0] = tmp_road_msk[road_msk == 0]
                        road_msk = road_msk.copy()
                        end_points[point_hash(p3[0], p3[1])] = 2
                        
        h = point_hash(final_res_roads[i][-1, 0], final_res_roads[i][-1, 1])
        if end_points[h] == 1:
            p1 = final_res_roads[i][-2]
            p2 = final_res_roads[i][-1]
            p3 = None
            for a in angles_to_check:
                p3 = try_connect(p1, p2, a, connect_dist, road_msk, min_prob, msk, final_res_roads, con_r)
                if p3 is not None:
                    break
            if p3 is not None:
                h1 = pair_hash(p2, p3)
                h2 = pair_hash(p3, p2)
                if not (h1 in hashes or h2 in hashes):
                    r_id = road_msk[p3[0], p3[1]] - 1
                    final_res_roads[r_id], new_hashes = inject_point(final_res_roads[r_id], p3)
                    hashes.update(new_hashes)
                    tmp_road_msk = np.zeros((1300, 1300), dtype=np.int32)
                    tmp_road_msk = tmp_road_msk.copy()
                    cv2.line(tmp_road_msk, (p2[1], p2[0]), (p3[1], p3[0]), i+1, thickness)
                    road_msk[road_msk == 0] = tmp_road_msk[road_msk == 0]
                    road_msk = road_msk.copy()
                    final_res_roads[i] = np.vstack((final_res_roads[i], p3))
                    hashes.add(h1)
                    hashes.add(h2)
                    end_points[point_hash(p3[0], p3[1])] = 2
            else:
                p3 = get_next_point(p1, p2, add_dist)
                if not (p3[0] < 2 or p3[1] < 2 or p3[0] > 1297 or p3[1] > 1297):
                    p3 = get_next_point(p1, p2, add_dist2)
                if (p3[0] != p2[0] or p3[1] != p2[1]) and (road_msk[p3[0], p3[1]] == 0):
                    h1 = pair_hash(p2, p3)
                    h2 = pair_hash(p3, p2)
                    if not (h1 in hashes or h2 in hashes):
                        final_res_roads[i] = np.vstack((final_res_roads[i], p3))
                        hashes.add(h1)
                        hashes.add(h2)
                        tmp_road_msk = np.zeros((1300, 1300), dtype=np.int32)
                        tmp_road_msk = tmp_road_msk.copy()
                        cv2.line(tmp_road_msk, (p2[1], p2[0]), (p3[1], p3[0]), i+1, thickness)
                        road_msk[road_msk == 0] = tmp_road_msk[road_msk == 0]
                        road_msk = road_msk.copy()
                        end_points[point_hash(p3[0], p3[1])] = 2
            
    lines = [LineString(r[:, ::-1]) for r in final_res_roads]

    if len(lines) == 0:
        res_rows.append({'ImageId': img_id, 'WKT_Pix': 'LINESTRING EMPTY'})
    else:
        for l in lines:
            res_rows.append({'ImageId': img_id, 'WKT_Pix': dumps(l, rounding_precision=0)})   
    return res_rows


if __name__ == '__main__':
    t0 = timeit.default_timer()
    
    for i in range(1, len(sys.argv) - 1):
        test_folders.append(sys.argv[i])
        
    out_file = sys.argv[-1]
    if '.txt' not in out_file:
        out_file = out_file + '.txt'
        
    res_rows = []
    paramss = []
    for d in test_folders:  
        for f in tqdm(sorted(listdir(path.join(d, 'MUL-PanSharpen')))):
            if path.isfile(path.join(d, 'MUL-PanSharpen', f)) and '.tif' in f:
                img_id = f.split('PanSharpen_')[1].split('.')[0]
                cid = cities.index(img_id.split('_')[2])
                
                if cid in [0]:
                    vgg_big_path = None
                    vgg_big_path_1 = None
                else:
                    vgg_big_path = path.join(pred_folder, 'vgg_big', '0', cities[cid], '{0}.png'.format(img_id))
                    vgg_big_path_1 = path.join(pred_folder, 'vgg_big', '1', cities[cid], '{0}.png'.format(img_id))
                if cid in [1]:
                    vgg_small_path = None
                else:
                    vgg_small_path = path.join(pred_folder, 'vgg_small', '0', cities[cid], '{0}.png'.format(img_id))
                if cid in []:
                    linknet_small_path = None
                else:
                    linknet_small_path = path.join(pred_folder, 'linknet_small', '0', cities[cid], '{0}.png'.format(img_id))
                if cid in [0, 1]:
                    resnet_small_path = None
                else:
                    resnet_small_path = path.join(pred_folder, 'resnet_small', '0', cities[cid], '{0}.png'.format(img_id))
                if cid in [1]:
                    inception_small_path = None
                else:
                    inception_small_path = path.join(pred_folder, 'inception_small', '0', cities[cid], '{0}.png'.format(img_id))
                if cid in [1]:
                    vgg_smallest_file_path = None
                    vgg_smallest_file_path_1 = None
                else:
                    vgg_smallest_file_path = path.join(pred_folder, 'vgg_smallest', '0', cities[cid], '{0}.png'.format(img_id))
                    vgg_smallest_file_path_1 = path.join(pred_folder, 'vgg_smallest', '1', cities[cid], '{0}.png'.format(img_id))
                if cid in [2]:
                    inc_smallest_file_path = None
                    inc_smallest_file_path_1 = None
                else:
                    inc_smallest_file_path = path.join(pred_folder, 'inception_smallest', '0', cities[cid], '{0}.png'.format(img_id))
                    inc_smallest_file_path_1 = path.join(pred_folder, 'inception_smallest', '1', cities[cid], '{0}.png'.format(img_id))
                if cid in [0, 3]:
                    res_smallest_file_path = None
                    res_smallest_file_path_1 = None
                else:
                    res_smallest_file_path = path.join(pred_folder, 'resnet_smallest', '0', cities[cid], '{0}.png'.format(img_id))
                    res_smallest_file_path_1 = path.join(pred_folder, 'resnet_smallest', '1', cities[cid], '{0}.png'.format(img_id))
                if cid in [3]:
                    inc3_520_file_path = None
                    inc3_520_file_path_1 = None
                else:
                    inc3_520_file_path = path.join(pred_folder, 'inception_v3_520', '0', cities[cid], '{0}.png'.format(img_id))
                    inc3_520_file_path_1 = path.join(pred_folder, 'inception_v3_520', '1', cities[cid], '{0}.png'.format(img_id))
                if cid in [3]:
                    inc_v2_520_file_path = None
                    inc_v2_520_file_path_1 = None
                else:
                    inc_v2_520_file_path = path.join(pred_folder, 'inception_520', '0', cities[cid], '{0}.png'.format(img_id))
                    inc_v2_520_file_path_1 = path.join(pred_folder, 'inception_520', '1', cities[cid], '{0}.png'.format(img_id))
                if cid in [1, 2]:
                    linknet_big_file_path = None
                    linknet_big_file_path_1 = None
                else:
                    linknet_big_file_path = path.join(pred_folder, 'linknet_big', '0', cities[cid], '{0}.png'.format(img_id))
                    linknet_big_file_path_1 = path.join(pred_folder, 'linknet_big', '1', cities[cid], '{0}.png'.format(img_id))
                if cid in []:
                    linknet_520_file_path = None
                    linknet_520_file_path_1 = None
                else:
                    linknet_520_file_path = path.join(pred_folder, 'linknet_520', '0', cities[cid], '{0}.png'.format(img_id))
                    linknet_520_file_path_1 = path.join(pred_folder, 'linknet_520', '1', cities[cid], '{0}.png'.format(img_id))
                
                if 'Vegas' in img_id:
                    par2 = np.asarray([  3,   0,   5,  10, 120, 120,  24,   4,   9])
                    par = np.asarray([  1,   2,   0,   0,   2,   1,   1,   0,   1,   5,   2,   2, 168])
                elif 'Paris' in img_id:
                    par2 = np.asarray([ 0,  0, 96, 14, 56, 30, 31,  6,  6])
                    par = np.asarray([  0,   1,   4,   0,   0,   0,   4,   3,   5,   5,   0,   2, 127])
                elif 'Shanghai' in img_id:
                    par2 = np.asarray([  0,   5,   5,  10, 120,  90,  24,   7,   5])
                    par = np.asarray([  3,   1,   4,   1,   4,   2,   0,   2,   1,   2,   0,   1, 127])
                elif 'Khartoum' in img_id:
                    par2 = np.asarray([ 0,  0,  5, 10, 80, 30, 24,  4,  5])
                    par = np.asarray([  3,   6,   6,   2,   6,   6,   3,   0,   0,   0,   2,   2, 100])  
                    
                paramss.append((img_id, par, par2, vgg_big_path, vgg_small_path, linknet_small_path, resnet_small_path, inception_small_path, vgg_smallest_file_path, 
                                inc_smallest_file_path, res_smallest_file_path, inc3_520_file_path, inc_v2_520_file_path, linknet_big_file_path, linknet_520_file_path,
                                vgg_big_path_1, vgg_smallest_file_path_1, 
                                inc_smallest_file_path_1, res_smallest_file_path_1, inc3_520_file_path_1, inc_v2_520_file_path_1,  linknet_big_file_path_1, linknet_520_file_path_1))

    with Pool(processes=4) as pool:
        results = pool.starmap(process_file, paramss)
    for i in range(len(results)):
        res_rows.extend(results[i])
        
    sub = pd.DataFrame(res_rows, columns=('ImageId', 'WKT_Pix'))
    sub.to_csv(out_file, index=False, header=False)
                
    elapsed = timeit.default_timer() - t0
    print('Submission file created! Time: {:.3f} min'.format(elapsed / 60))