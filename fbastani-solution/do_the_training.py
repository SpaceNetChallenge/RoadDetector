import multiprocessing
import os
import subprocess
import sys

def runtrain(params):
	i, cities, basedirs = params
	gpu_env = os.environ.copy()
	gpu_env['CUDA_VISIBLE_DEVICES'] = str(i)
	for city in cities:
		print 'starting member {}/4 for city {}'.format(i, city)
		subprocess.call(['python', 'run_train.py', city, str(i)] + basedirs, env=gpu_env)

basedirs = sys.argv[1:]
for i in xrange(len(basedirs)):
	if basedirs[i][-1] == '/':
		basedirs[i] = basedirs[i][:-1]

cities = []
for basedir in basedirs:
	d = basedir.split('/')[-1]
	parts = d.split('_')
	city = '{}_{}_{}'.format(parts[0], parts[1], parts[2])
	cities.append(city)

todo = [(i, cities, basedirs) for i in xrange(4)]
p = multiprocessing.Pool(4)
p.map(runtrain, todo)
p.close()
