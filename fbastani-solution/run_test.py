#!/usr/bin/python

import model4u_big as model
from graphextract.discoverlib import graph

import georasters
from multiprocessing import Pool
import numpy
import os
from PIL import Image
import random
import scipy.ndimage
import shutil
import skimage.morphology
import subprocess
import sys
import tensorflow as tf
import time

from run_lib import *

basedirs = sys.argv[1:-1]
for i in xrange(len(basedirs)):
	if basedirs[i][-1] == '/':
		basedirs[i] = basedirs[i][:-1]

print 'initializing tensorflow computation graph'
m = model.Model(in_channels=9)
session = tf.Session()

cleaned_path_to_regions = {}

for basedir in basedirs:
	d = basedir.split('/')[-1]
	parts = d.split('_')
	city = '{}_{}_{}'.format(parts[0], parts[1], parts[2])

	pdir = '/wdata/outputs/{}/'.format(city)
	shutil.rmtree(pdir, ignore_errors=True)
	os.makedirs(pdir)

	# load tiles
	print '[{}] 1/5 loading tiles'.format(basedir)
	tiles = []
	path = '{}/MUL-PanSharpen/'.format(basedir)
	fnames = [fname for fname in os.listdir(path) if '.tif' in fname]
	ids = [fname.split('_')[-1].split('.')[0].split('img')[1] for fname in fnames]
	for counter in xrange(len(ids)):
		if counter % 10 == 0:
			print '... {}/{}'.format(counter, len(ids))
		region = '{}.{}.{}'.format(d, city, ids[counter])
		tile = load_tile(basedirs, region, skip_truth=True)
		tiles.append((region, tile))

	# run models
	for i in xrange(4):
		print '[{}] 2/5 running model mem{}/4'.format(basedir, i)
		best_path = 'models/{}/mem{}/model_best/model'.format(city, i)
		m.saver.restore(session, best_path)

		out_path = '{}/mem{}/'.format(pdir, i)
		os.mkdir(out_path)
		do_test2(m, session, tiles, out_path)

	# take median output and add padding
	print '[{}] 3/5 take median output'.format(basedir)
	mem_paths = ['{}/mem{}/'.format(pdir, i) for i in xrange(4)]
	avg_path = pdir + 'avg/'
	os.mkdir(avg_path)
	for region, _ in tiles:
		ims = [scipy.ndimage.imread('{}/{}.png'.format(path, region)) for path in mem_paths]
		im = numpy.stack(ims)
		im = numpy.median(im, axis=0).astype('uint8')

		big_im = numpy.zeros((732, 732), dtype='uint8')
		im = scipy.ndimage.filters.gaussian_filter(im, sigma=1, mode='nearest')
		big_im[:41, 41:41+650] = numpy.tile(im[0:1, :], [41, 1])
		big_im[41:41+650, :41] = numpy.tile(im[:, 0:1], [1, 41])
		big_im[41+650:, 41:41+650] = numpy.tile(im[-2:-1, :], [41, 1])
		big_im[41:41+650, 41+650:] = numpy.tile(im[:, -2:-1], [1, 41])
		big_im[41:41+650, 41:41+650] = im

		Image.fromarray(big_im).save('{}/{}.png'.format(avg_path, region))

	# post-processing
	print '[{}] 4/5 extract graphs'.format(basedir)
	skeleton_path = pdir + 'skeleton/'
	graph_path = pdir + 'graphs/'
	sqlite_path = pdir + 'sqlite/'
	os.mkdir(skeleton_path)
	os.mkdir(graph_path)
	os.mkdir(sqlite_path)
	for counter in xrange(len(tiles)):
		if counter % 10 == 0:
			print '... {}/{}'.format(counter, len(tiles))
		region, _ = tiles[counter]

		in_fname = '{}/{}.png'.format(avg_path, region)
		skeleton_fname = '{}/{}.png'.format(skeleton_path, region)
		sqlite_fname = '{}/{}'.format(sqlite_path, region)
		output_fname = '{}/{}.p'.format(graph_path, region)
		final_fname = '{}/{}.graph'.format(graph_path, region)
		pix_fname = '{}/{}.pix.graph'.format(graph_path, region)

		fnull = open(os.devnull, 'w')
		subprocess.call(['python', 'skeleton.py', in_fname, skeleton_fname], stdout=fnull, cwd='/app/graphextract/')
		fnull.close()
		im = scipy.ndimage.imread(skeleton_fname)
		im[:40, :] = 0
		im[:, :40] = 0
		im[40+652:, :] = 0
		im[:, 40+652:] = 0
		Image.fromarray(im).save(skeleton_fname)

		fnull = open(os.devnull, 'w')
		subprocess.call(['python', 'graph_extract_good.py', skeleton_fname, '/app/graphextract/bounding_boxes/spacenet.txt', sqlite_fname, output_fname], stdout=fnull, cwd='/app/graphextract/')
		fnull.close()
		subprocess.call(['python', 'map2go.py', output_fname, final_fname], cwd='/app/graphextract/')

		with open(final_fname, 'r') as f:
			lines = f.readlines()
		min_lat_ = 36.1666923624
		min_lon_ = -115.272632937
		max_lat_ = 36.1706330372
		max_lon_ = -115.268692263
		min_lat = min_lat_ + (max_lat_ - min_lat_) * 10 / 183
		min_lon = min_lon_ + (max_lon_ - min_lon_) * 10 / 183
		max_lat = max_lat_ - (max_lat_ - min_lat_) * 10 / 183
		max_lon = max_lon_ - (max_lon_ - min_lon_) * 10 / 183
		with open(pix_fname, 'w') as f:
			state = 0
			for line in lines:
				line = line.strip()
				if state == 0:
					if line:
						parts = line.split(' ')
						lon = float(parts[0])
						lat = float(parts[1])
						x = (lon - min_lon) * 1300 / (max_lon - min_lon)
						y = (max_lat - lat) * 1300 / (max_lat - min_lat)
						f.write('{} {}\n'.format(int(x), int(y)))
					else:
						f.write('\n')
						state = 1
				elif state == 1:
					f.write(line + '\n')

	print '[{}] 5/5 clean graphs'.format(basedir)
	cleaned_path = pdir + 'cleaned_graphs/'
	os.mkdir(cleaned_path)
	subprocess.call(['python', 'clean.py', graph_path, avg_path, cleaned_path], cwd='/app/graphextract/')

	cleaned_path_to_regions[cleaned_path] = [region for region, _ in tiles]

# produce csv
print 'writing combined CSV'
all_linestrings = ['ImageId,WKT_Pix']
for cleaned_path, regions in cleaned_path_to_regions.items():
	for region in regions:
		g = graph.read_graph('{}/{}.graph'.format(cleaned_path, region))

		d, city, id = region.split('.')
		image_id = '{}_img{}'.format(city, id)

		linestrings = []
		explored_pairs = set()
		def is_explored(edge):
			return (edge.src.id, edge.dst.id) in explored_pairs or (edge.dst.id, edge.src.id) in explored_pairs or edge.src == edge.dst
		def mark_explored(edge):
			explored_pairs.add((edge.src.id, edge.dst.id))
		for edge in g.edges:
			seq = []
			while not is_explored(edge):
				seq.append(edge)
				mark_explored(edge)
				for candidate in edge.dst.out_edges:
					if not is_explored(candidate):
						edge = candidate
						break
			if len(seq) > 0:
				points = [seq[0].src.point] + [edge.dst.point for edge in seq]
				points_str = ', '.join(['{} {}'.format(point.x, point.y) for point in points])
				s = '"LINESTRING ({})"'.format(points_str)
				linestrings.append(s)
		if len(linestrings) > 0:
			for s in linestrings:
				all_linestrings.append('{},{}'.format(image_id, s))
		else:
			all_linestrings.append('{},LINESTRING EMPTY'.format(image_id))
with open(sys.argv[-1], 'w') as f:
	f.write("\n".join(all_linestrings) + "\n")
