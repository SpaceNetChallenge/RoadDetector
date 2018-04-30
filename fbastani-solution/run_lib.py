import georasters
import numpy
import os
from PIL import Image
import random
import scipy.ndimage
import subprocess
import sys
import tensorflow as tf
import time

SIZE = 256
OUTPUT_SCALE = 2
OUTPUT_CHANNELS = 1
TARGETS = '/wdata/spacenet2017/favyen/truth/'

def load_tile(basedirs, region, skip_truth=False):
	d, city, id = region.split('.')
	BASEDIR = [x for x in basedirs if city in x][0]

	fname = '{}/MUL-PanSharpen/MUL-PanSharpen_{}_img{}.tif'.format(BASEDIR, city, id)
	data1 = georasters.from_file(fname).raster
	data1 = data1.filled(0)
	data1 = numpy.transpose(data1, (2, 1, 0))

	fname = '{}/PAN/PAN_{}_img{}.tif'.format(BASEDIR, city, id)
	data2 = georasters.from_file(fname).raster
	data2 = data2.filled(0)
	data2 = numpy.transpose(data2, (1, 0))

	input_im = numpy.zeros((1300, 1300, 9), dtype='uint8')
	for i in xrange(8):
		input_im[:, :, i] = (data1[:, :, i] / 8).astype('uint8')
	input_im[:, :, 8] = (data2 / 8).astype('uint8')

	if not skip_truth:
		fname = '{}/{}.png'.format(TARGETS, region)
		if not os.path.isfile(fname):
			return None
		output_im = scipy.ndimage.imread(fname)
		if len(output_im.shape) == 3:
			output_im = 255 - output_im[:, :, 0:1]
			output_im = (output_im > 1).astype('uint8') * 255
		else:
			output_im = numpy.expand_dims(output_im, axis=2)

	else:
		output_im = numpy.zeros((1300 / OUTPUT_SCALE, 1300 / OUTPUT_SCALE, OUTPUT_CHANNELS), dtype='uint8')

	return input_im, numpy.swapaxes(output_im, 0, 1)

def load_tiles_new(basedirs, city):
	all_tiles = []
	regions = [fname.split('.png')[0] for fname in os.listdir(TARGETS) if '.png' in fname]
	regions = [region for region in regions if city == region.split('.')[1]]
	counter = 0
	for region in regions:
		counter += 1
		if counter % 20 == 0:
			print '... {}/{}'.format(counter, len(regions))
		all_tiles.append((region, load_tile(basedirs, region)))
	random.shuffle(all_tiles)
	val_tiles = all_tiles[:20]
	train_tiles = all_tiles[20:]
	return val_tiles, train_tiles

def extract(tile):
	input_im, output_im = tile[1]

	i = random.randint(-64, 1300 / OUTPUT_SCALE + 64 - SIZE / OUTPUT_SCALE - 1)
	j = random.randint(-64, 1300 / OUTPUT_SCALE + 64 - SIZE / OUTPUT_SCALE - 1)
	if i < 0:
		i = 0
	elif i >= 1300 / OUTPUT_SCALE - SIZE / OUTPUT_SCALE:
		i = 1300 / OUTPUT_SCALE - SIZE / OUTPUT_SCALE - 1
	if j < 0:
		j = 0
	elif j >= 1300 / OUTPUT_SCALE - SIZE / OUTPUT_SCALE:
		j = 1300 / OUTPUT_SCALE - SIZE / OUTPUT_SCALE - 1

	input_rect = input_im[i*OUTPUT_SCALE:i*OUTPUT_SCALE+SIZE, j*OUTPUT_SCALE:j*OUTPUT_SCALE+SIZE, :]
	output_rect = output_im[i:i+SIZE/OUTPUT_SCALE, j:j+SIZE/OUTPUT_SCALE, :]

	rotations = random.randint(0, 3)
	if rotations > 0:
		input_rect = numpy.rot90(input_rect, k=rotations)
		output_rect = numpy.rot90(output_rect, k=rotations)

	return input_rect, output_rect

def do_test2(m, session, tiles, out_path):
	for counter in xrange(len(tiles)):
		if counter % 10 == 0:
			print '... {}/{}'.format(counter, len(tiles))
		region, tile = tiles[counter]

		inputs = numpy.zeros((1, 1408, 1408, 9), dtype='float32')
		inputs[0, 54:54+1300, 54:54+1300, :] = tile[0].astype('float32') / 255.0
		outputs = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: inputs,
		})
		tile_output = (outputs[0, 27:27+650, 27:27+650] * 255.0).astype('uint8')
		Image.fromarray(numpy.swapaxes(tile_output, 0, 1)).save('{}/{}.png'.format(out_path, region))
