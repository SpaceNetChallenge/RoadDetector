#!/usr/bin/python

import model4u as model

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

from run_lib import *

basedirs = sys.argv[3:]
city = sys.argv[1]
memid = int(sys.argv[2])
PATH = 'models/{}/mem{}/'.format(city, memid)

os.makedirs(PATH)
os.mkdir(PATH + '/model_best')
os.mkdir(PATH + '/model_latest')

m = model.Model(in_channels=9)

print 'loading tiles'
start_loading_time = time.time()
val_tiles, train_tiles = load_tiles_new(basedirs, city)
print 'loaded {} tiles in {} seconds'.format(len(train_tiles), int(time.time() - start_loading_time))

start_training_time = time.time()

def epoch_to_learning_rate(epoch):
	num_tiles = len(train_tiles)
	if epoch < 50000 / num_tiles:
		return 0.0004
	elif epoch < 300000 / num_tiles:
		return 0.0001
	elif epoch < 700000 / num_tiles:
		return 0.00001
	elif epoch < 1000000 / num_tiles:
		return 0.000001
	else:
		return 0.0000001

print 'begin training'
session = tf.Session()
session.run(m.init_op)
latest_path = '{}/model_latest/model'.format(PATH)
best_path = '{}/model_best/model'.format(PATH)
best_loss = None

val_rects = []
for tile in val_tiles:
	for _ in xrange(1300*1300/SIZE/SIZE/2):
		val_rects.append(extract(tile))

for epoch in xrange(2000000 / len(train_tiles)):
	epoch_time = time.time()
	random.shuffle(train_tiles)
	train_losses = []
	for i in xrange(0, len(train_tiles), model.BATCH_SIZE):
		batch_tiles = [extract(t) for t in train_tiles[i:i+model.BATCH_SIZE]]
		_, loss = session.run([m.optimizer, m.loss], feed_dict={
			m.is_training: True,
			m.inputs: [tile[0].astype('float32') / 255.0 for tile in batch_tiles],
			m.targets: [tile[1].astype('float32') / 255.0 for tile in batch_tiles],
			m.learning_rate: epoch_to_learning_rate(epoch),
		})
		train_losses.append(loss)
	train_loss = numpy.mean(train_losses)
	train_time = time.time()

	val_losses = []
	for i in xrange(0, len(val_rects), model.BATCH_SIZE):
		batch_tiles = val_rects[i:i+model.BATCH_SIZE]
		batch_targets = numpy.array([tile[1].astype('float32') / 255.0 for tile in batch_tiles], dtype='float32')
		outputs, loss = session.run([m.outputs, m.loss], feed_dict={
			m.is_training: False,
			m.inputs: [tile[0].astype('float32') / 255.0 for tile in batch_tiles],
			m.targets: batch_targets,
		})
		val_losses.append(loss)

	val_loss = numpy.mean(val_losses)
	val_time = time.time()
	elapsed = time.time() - start_training_time

	print 'iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}, elapsed={}'.format(epoch, int(train_time - epoch_time), int(val_time - train_time), train_loss, val_loss, best_loss, int(elapsed))

	if epoch % 10 == 0:
		m.saver.save(session, latest_path)
	if best_loss is None or val_loss < best_loss:
		best_loss = val_loss
		m.saver.save(session, best_path)

	if elapsed > 3600 * 40:
		break
