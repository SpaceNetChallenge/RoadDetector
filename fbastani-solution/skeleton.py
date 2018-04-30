from discoverlib import geom
from discoverlib import graph

import math
import numpy
import os
import scipy.ndimage

PATH = '/data/spacenet2017/favyen/segmentation_model4d3/outputs'
OUT_PATH = '/data/spacenet2017/favyen/segmentation_model4d3_newskeleton/graphs'
TOL = 10
THRESHOLD = 20

circle_mask = numpy.ones((2*TOL+1, 2*TOL+1), dtype='uint8')
for i in xrange(-TOL, TOL+1):
	for j in xrange(-TOL, TOL+1):
		d = math.sqrt(i * i + j * j)
		if d <= TOL:
			circle_mask[i+TOL, j+TOL] = 0

def get_reachable_points(im, point, iterations):
	points = set()
	search = set()
	r = geom.Rectangle(geom.Point(0, 0), geom.Point(im.shape[0]-1, im.shape[1]-1))
	search.add(point)
	for _ in xrange(iterations):
		next_search = set()
		for point in search:
			for offset in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
				adj_point = point.add(geom.Point(offset[0], offset[1]))
				if r.contains(adj_point) and adj_point not in points and im[adj_point.x, adj_point.y] > 0:
					points.add(adj_point)
					next_search.add(adj_point)
		search = next_search
	return points

fnames = [fname for fname in os.listdir(PATH) if '.png' in fname]
for fname in fnames:
	region = fname.split('.png')[0]
	im = numpy.swapaxes(scipy.ndimage.imread(os.path.join(PATH, fname)), 0, 1)
	im = (im > THRESHOLD).astype('uint8')

	g = graph.Graph()
	im_copy = numpy.zeros((im.shape[0], im.shape[1]), dtype='uint8')
	im_copy[:, :] = im[:, :]
	point_to_vertex = {}

	while im_copy.max() > 0:
		p = numpy.unravel_index(im_copy.argmax(), im_copy.shape)
		vertex = g.add_vertex(geom.Point(p[0]*2, p[1]*2))
		point_to_vertex[geom.Point(p[0], p[1])] = vertex

		# coordinates below are start/end between -TOL and TOL
		sx = max(-TOL, -p[0])
		ex = min(TOL + 1, im_copy.shape[0] - p[0])
		sy = max(-TOL, -p[1])
		ey = min(TOL + 1, im_copy.shape[1] - p[1])
		im_copy[sx+p[0]:ex+p[0], sy+p[1]:ey+p[1]] *= circle_mask[sx+TOL:ex+TOL, sy+TOL:ey+TOL]

	for vertex in g.vertices:
		for point in get_reachable_points(im, vertex.point.scale(0.5), 2*TOL):
			if point in point_to_vertex and point_to_vertex[point] != vertex:
				g.add_bidirectional_edge(vertex, point_to_vertex[point])

	g.save(os.path.join(OUT_PATH, region + '.graph'))
