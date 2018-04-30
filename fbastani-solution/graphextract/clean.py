from discoverlib import geom, graph

import numpy
import math
from multiprocessing import Pool
import os.path
from PIL import Image
import random
import scipy.ndimage
import sys
import time

def graph_filter_edges(g, bad_edges):
	print 'filtering {} edges'.format(len(bad_edges))
	ng = graph.Graph()
	vertex_map = {}
	for vertex in g.vertices:
		vertex_map[vertex] = ng.add_vertex(vertex.point)
	for edge in g.edges:
		if edge not in bad_edges:
			nedge = ng.add_edge(vertex_map[edge.src], vertex_map[edge.dst])
			if hasattr(edge, 'prob'):
				nedge.prob = edge.prob
	return ng

def get_reachable_points(im, point, value_threshold, distance_threshold):
	points = set()
	search = set()
	r = geom.Rectangle(geom.Point(0, 0), geom.Point(im.shape[0]-1, im.shape[1]-1))
	search.add(point)
	for _ in xrange(distance_threshold):
		next_search = set()
		for point in search:
			for offset in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
				adj_point = point.add(geom.Point(offset[0], offset[1]))
				if r.contains(adj_point) and adj_point not in points and im[adj_point.x, adj_point.y] >= value_threshold:
					points.add(adj_point)
					next_search.add(adj_point)
		search = next_search
	return points

def count_adjacent(skeleton, point):
	r = geom.Rectangle(geom.Point(0, 0), geom.Point(skeleton.shape[0], skeleton.shape[1]))
	count = 0
	for offset in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
		adj_point = point.add(geom.Point(offset[0], offset[1]))
		if skeleton[adj_point.x, adj_point.y] > 0:
			count += 1
	return count

def distance_from_value(value):
	return 1.1**max(30-value, 0)

def get_shortest_path(im, src, max_distance):
	r = geom.Rectangle(geom.Point(0, 0), geom.Point(im.shape[0], im.shape[1]))
	in_r = r.add_tol(-1)
	seen_points = set()
	distances = {}
	prev = {}
	dst = None

	distances[src] = 0
	while len(distances) > 0:
		closest_point = None
		closest_distance = None
		for point, distance in distances.items():
			if closest_point is None or distance < closest_distance:
				closest_point = point
				closest_distance = distance

		del distances[closest_point]
		seen_points.add(closest_point)
		if closest_distance > max_distance:
			break
		elif not in_r.contains(closest_point):
			dst = closest_point
			break

		for offset in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
			adj_point = closest_point.add(geom.Point(offset[0], offset[1]))
			if r.contains(adj_point) and adj_point not in seen_points:
				distance = closest_distance + distance_from_value(im[adj_point.x, adj_point.y])
				if adj_point not in distances or distance < distances[adj_point]:
					distances[adj_point] = distance
					prev[adj_point] = closest_point

	if dst is None:
		return

	return dst

def get_segment_confidence(segment, im):
	def get_value(p):
		p = p.scale(0.5)
		sx = max(0, p.x-1)
		sy = max(0, p.y-1)
		ex = min(im.shape[0], p.x+2)
		ey = min(im.shape[1], p.y+2)
		return im[sx:ex, sy:ey].max()

	values = []
	for i in xrange(0, int(segment.length()), 2):
		p = segment.point_at_factor(i)
		values.append(get_value(p))
	return numpy.mean(values)

def get_rs_confidence(rs, im):
	def get_value(p):
		p = p.scale(0.5)
		sx = max(0, p.x-1)
		sy = max(0, p.y-1)
		ex = min(im.shape[0], p.x+2)
		ey = min(im.shape[1], p.y+2)
		return im[sx:ex, sy:ey].max()

	values = []
	for i in xrange(0, int(rs.length()), 2):
		p = rs.point_at_factor(i)
		values.append(get_value(p))
	return numpy.mean(values)

def connect_up(g, im, threshold=40.0):
	# connect road segments to projection
	bad_edges = set()
	updated_vertices = set()
	road_segments, edge_to_rs = graph.get_graph_road_segments(g)
	edgeIdx = g.edgeIndex()
	add_points = []
	for rs in road_segments:
		for vertex in [rs.src(), rs.dst()]:
			if len(vertex.out_edges) > 1 or vertex in updated_vertices:
				continue
			vector = vertex.in_edges[0].segment().vector()
			vector = vector.scale(threshold / vector.magnitude())
			best_edge = None
			best_point = None
			best_distance = None
			for edge in edgeIdx.search(vertex.point.bounds().add_tol(threshold)):
				if edge in rs.edges or edge in rs.get_opposite_rs(edge_to_rs).edges:
					continue
				s1 = edge.segment()
				s2 = geom.Segment(vertex.point, vertex.point.add(vector))
				p = s1.intersection(s2)
				if p is None:
					# maybe still connect if both edges are roughly the same angle, and vector connecting them would also be similar angle
					p = edge.src.point
					if vertex.point.distance(p) >= threshold:
						continue
					v1 = s1.vector()
					v2 = p.sub(vertex.point)
					if abs(v1.signed_angle(vector)) > math.pi / 4 or abs(v2.signed_angle(vector)) > math.pi / 4:
						continue
					elif get_segment_confidence(geom.Segment(vertex.point, p), im) < 55:
						continue
				if p is not None and (best_edge is None or vertex.point.distance(p) < best_distance):
					best_edge = edge
					best_point = p
					best_distance = vertex.point.distance(p)

			if best_edge is not None:
				#print '*** insert new vertex at {} from {} with {}'.format(best_point, vertex.point, best_edge.segment())
				bad_edges.add(best_edge)
				add_points.append((best_point, [best_edge.src, best_edge.dst, vertex]))
				updated_vertices.add(vertex)
	for t in add_points:
		nv = g.add_vertex(t[0])
		for v in t[1]:
			g.add_bidirectional_edge(nv, v)

	return graph_filter_edges(g, bad_edges)

def cleanup_all(graph_fname, im_fname, cleaned_fname):
	g = graph.read_graph(graph_fname)
	im = numpy.swapaxes(scipy.ndimage.imread(im_fname), 0, 1)

	r = geom.Rectangle(geom.Point(0, 0), geom.Point(1300, 1300))
	small_r = r.add_tol(-20)

	# filter lousy road segments
	road_segments, _ = graph.get_graph_road_segments(g)
	bad_edges = set()
	for rs in road_segments:
		if rs.length() < 80 and (len(rs.src().out_edges) < 2 or len(rs.dst().out_edges) < 2) and small_r.contains(rs.src().point) and small_r.contains(rs.dst().point):
			bad_edges.update(rs.edges)
		elif rs.length() < 400 and len(rs.src().out_edges) < 2 and len(rs.dst().out_edges) < 2 and small_r.contains(rs.src().point) and small_r.contains(rs.dst().point):
			bad_edges.update(rs.edges)
	ng = graph_filter_edges(g, bad_edges)

	# connect road segments to the image edge
	road_segments, _ = graph.get_graph_road_segments(ng)
	segments = [
		geom.Segment(geom.Point(0, 0), geom.Point(1300, 0)),
		geom.Segment(geom.Point(0, 0), geom.Point(0, 1300)),
		geom.Segment(geom.Point(1300, 1300), geom.Point(1300, 0)),
		geom.Segment(geom.Point(1300, 1300), geom.Point(0, 1300)),
	]
	big_r = r.add_tol(-2)
	small_r = r.add_tol(-40)
	for rs in road_segments:
		for vertex in [rs.src(), rs.dst()]:
			if len(vertex.out_edges) == 1 and big_r.contains(vertex.point) and not small_r.contains(vertex.point):
				'''d = min([segment.distance(vertex.point) for segment in segments])
				dst = get_shortest_path(im, vertex.point.scale(0.5), max_distance=d*9)
				if dst is None:
					break
				if dst is not None:
					nv = ng.add_vertex(dst.scale(2))
					ng.add_bidirectional_edge(vertex, nv)
					print '*** add edge {} to {}'.format(vertex.point, nv.point)'''
				'''closest_segment = None
				closest_distance = None
				for segment in segments:
					d = segment.distance(vertex.point)
					if closest_segment is None or d < closest_distance:
						closest_segment = segment
						closest_distance = d'''
				for closest_segment in segments:
					vector = vertex.in_edges[0].segment().vector()
					vector = vector.scale(40.0 / vector.magnitude())
					s = geom.Segment(vertex.point, vertex.point.add(vector))
					p = s.intersection(closest_segment)
					if p is not None:
						nv = ng.add_vertex(p)
						ng.add_bidirectional_edge(vertex, nv)
						break

	ng = connect_up(ng, im)

	ng.save(cleaned_fname)

if __name__ == '__main__':
	in_dir = sys.argv[1]
	tile_dir = sys.argv[2]
	out_dir = sys.argv[3]

	fnames = [fname.split('.pix.graph')[0] for fname in os.listdir(in_dir) if '.pix.graph' in fname]
	for fname in fnames:
		cleanup_all('{}/{}.pix.graph'.format(in_dir, fname), '{}/{}.png'.format(tile_dir, fname), '{}/{}.graph'.format(out_dir, fname))
