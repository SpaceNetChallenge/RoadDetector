import pyximport; pyximport.install()
import subiterations
import numpy as np
import scipy.ndimage as nd
from scipy.ndimage import imread
from scipy.misc import imsave, toimage
from itertools import izip
from multiprocessing import Process, Manager, cpu_count
from threading import Thread
from scipy.ndimage.morphology import grey_closing
import math

skeleton_images_path = "skeleton_images/"

class GrayscaleSkeleton:
    def __init__(self):
        pass

    def skeletonize(self, image):
        #image = grey_closing(image, footprint=circle(2), mode='constant', cval=0.0)
        image = add_zero_mat(image)
        prev_binary_image = np.zeros_like(image)

        image_bit_depth = (image.dtype.itemsize * 8)# / 2
        print "image_bit_depth: " + str(image_bit_depth)

        #image_thresholds = range(2**image_bit_depth,-1,-16)
        #image_thresholds = [2**x for x in range(image_bit_depth, 3, -1)] + range(15, 0, -1)
        image_thresholds = [60]
        print "image_thresholds: " + str(image_thresholds)

        for curr_threshold in image_thresholds:
            print "curr_threshold: " + str(curr_threshold)

            curr_binary_image = (image >= curr_threshold).astype(np.bool).astype(np.int)
            #imsave(skeleton_images_path + "binary_" + str(curr_threshold) + ".png", curr_binary_image)

            curr_sum_image = (prev_binary_image + curr_binary_image)
            curr_skeleton_image = self.thin_pixels(curr_sum_image)
            #imsave(skeleton_images_path + "skeleton_" + str(curr_threshold) + ".png", curr_skeleton_image)
            print "curr_skeleton max: " + str(curr_skeleton_image.max())

            prev_binary_image = curr_skeleton_image

        return remove_zero_mat(prev_binary_image)

    def thin_pixels(self, image):
        pixel_removed = True

        neighbors = nd.convolve((image>0).astype(np.int),[[1,1,1],[1,0,1],[1,1,1]],mode='constant',cval=0.0)
        fg_pixels = np.where((image==1) & (neighbors >= 2) & (neighbors <= 6))
	check_pixels = zip(fg_pixels[0], fg_pixels[1])

        while len(check_pixels)>0:
            print len(check_pixels)
            (image, sub1_check_pixels) = self.parallel_sub(subiterations.first_subiteration, image, check_pixels)
            (image, sub2_check_pixels) = self.parallel_sub(subiterations.second_subiteration, image, list(set(check_pixels+sub1_check_pixels)))

            check_pixels=list(set(sub1_check_pixels+sub2_check_pixels))

	neighbors = nd.convolve(image>0,[[1,1,1],[1,0,1],[1,1,1]],mode='constant',cval=0.0)
	fg_pixels = np.where(image==1)
	check_pixels = zip(fg_pixels[0],fg_pixels[1])
        (image, _) = self.parallel_sub(self.empty_pools, image, check_pixels)
        return image

    def parallel_sub(self, sub_function, image, fg_pixels):
        manager = Manager()
        queue = manager.Queue()
        next_queue = manager.Queue()

        num_procs = int(math.ceil(float(cpu_count()) * 0.75))
        workload_size = int(math.ceil(float(len(fg_pixels)) / float(num_procs)))

        process_list = []

	if len(fg_pixels) == 0:
            return (image, [])

	(zero_pixels, next_pixels) = sub_function(image,fg_pixels)
	for (x,y) in zero_pixels:
            image[x][y]=0;

        return (image, next_pixels)

    def PRE_first_subiteration(self, curr_image, fg_pixels):
        zero_pixels = {}
	next_pixels = {}

        for (i, j) in fg_pixels:
            if curr_image[i][j] != 1: continue

            p2 = curr_image[i - 1][j]
            p3 = curr_image[i - 1][j + 1]
            p4 = curr_image[i][j + 1]
            p5 = curr_image[i + 1][j + 1]
            p6 = curr_image[i + 1][j]
            p7 = curr_image[i + 1][j - 1]
            p8 = curr_image[i][j - 1]
            p9 = curr_image[i - 1][j - 1]

            if (2 <= (bool(p2) + bool(p3) + bool(p4) + bool(p5) + bool(p6) + bool(p7) + bool(p8) + bool(p9)) <= 6 and
                (p2 * p4 * p6 == 0) and
		(p4 * p6 * p8 == 0)):
                if ((not p2 and p3) + (not p3 and p4) + (not p4 and p5) + (not p5 and p6) + (not p6 and p7) + (not p7 and p8) + (not p8 and p9) + (not p9 and p2) == 1):
                    zero_pixels[(i,j)] = 0
                    if p2 == 1:
			next_pixels[(i-1,j)]=0
                    if p3 == 1:
			next_pixels[(i-1,j+1)]=0
                    if p4 == 1:
			next_pixels[(i,j+1)]=0
                    if p5 == 1:
			next_pixels[(i+1,j+1)]=0
                    if p6 == 1:
			next_pixels[(i+1,j)]=0
                    if p7 == 1:
			next_pixels[(i+1,j-1)]=0
                    if p8 == 1:
			next_pixels[(i,j-1)]=0
                    if p9 == 1:
			next_pixels[(i-1,j-1)]=0

        return zero_pixels.keys(), next_pixels.keys()

    def PRE_second_subiteration(self, curr_image, fg_pixels):
        zero_pixels = {}
	next_pixels = {}

        for (i, j) in fg_pixels:
            if curr_image[i][j] != 1: continue

            p2 = curr_image[i - 1][j]
            p3 = curr_image[i - 1][j + 1]
            p4 = curr_image[i][j + 1]
            p5 = curr_image[i + 1][j + 1]
            p6 = curr_image[i + 1][j]
            p7 = curr_image[i + 1][j - 1]
            p8 = curr_image[i][j - 1]
            p9 = curr_image[i - 1][j - 1]

            if (2 <= (bool(p2) + bool(p3) + bool(p4) + bool(p5) + bool(p6) + bool(p7) + bool(p8) + bool(p9)) <= 6 and
                (p2 * p4 * p8 == 0) and
		(p2 * p6 * p8 == 0)):
                if ((not p2 and p3) + (not p3 and p4) + (not p4 and p5) + (not p5 and p6) + (not p6 and p7) + (not p7 and p8) + (not p8 and p9) + (not p9 and p2) == 1):
                    zero_pixels[(i,j)] = 0
                    if p2 == 1:
			next_pixels[(i-1,j)]=0
                    if p3 == 1:
			next_pixels[(i-1,j+1)]=0
                    if p4 == 1:
			next_pixels[(i,j+1)]=0
                    if p5 == 1:
			next_pixels[(i+1,j+1)]=0
                    if p6 == 1:
			next_pixels[(i+1,j)]=0
                    if p7 == 1:
			next_pixels[(i+1,j-1)]=0
                    if p8 == 1:
			next_pixels[(i,j-1)]=0
                    if p9 == 1:
			next_pixels[(i-1,j-1)]=0

        return zero_pixels.keys(), next_pixels.keys()

    def empty_pools(self, curr_image, fg_pixels):
        zero_pixels = {}

        for (i, j) in fg_pixels:
            p2 = curr_image[i - 1][j]
            p3 = curr_image[i - 1][j + 1]
            p4 = curr_image[i][j + 1]
            p5 = curr_image[i + 1][j + 1]
            p6 = curr_image[i + 1][j]
            p7 = curr_image[i + 1][j - 1]
            p8 = curr_image[i][j - 1]
            p9 = curr_image[i - 1][j - 1]

            if (bool(p2) + bool(p3) + bool(p4) + bool(p5) + bool(p6) + bool(p7) + bool(p8) + bool(p9) > 6):
                zero_pixels[(i,j)] = 0

        return zero_pixels,[]

#
# helper functions
#
def add_zero_mat(image):
    num_rows, num_cols = image.shape

    image = np.insert(image, num_rows, np.zeros(num_cols, dtype=np.int), 0)
    image = np.insert(image, 0, np.zeros(num_cols, dtype=np.int), 0)

    num_rows, num_cols = image.shape

    image = np.insert(image, num_cols, np.zeros(num_rows, dtype=np.int), 1)
    image = np.insert(image, 0, np.zeros(num_rows, dtype=np.int), 1)

    return image

def remove_zero_mat(image):
    num_rows, num_cols = image.shape

    image = np.delete(image, num_rows - 1, 0)
    image = np.delete(image, 0, 0)
    image = np.delete(image, num_cols - 1, 1)
    image = np.delete(image, 0, 1)

    return image

def circle(radius):
    x, y = np.mgrid[:(2 * radius) + 1, :(2 * radius) + 1]
    circle = (x - radius) ** 2 + (y - radius) ** 2
    return (circle <= (radius ** 2)).astype(np.int)

import sys, time
if __name__ == '__main__':
    input_filename = str(sys.argv[1])
    output_filename = str(sys.argv[2])

    print "input filename: " + str(input_filename)
    print "output filename: " + str(output_filename)

    input_kde = imread(input_filename)

    s = GrayscaleSkeleton()

    start_time = time.time()
    skeleton = s.skeletonize(input_kde)
    print "total elapsed time: " + str(time.time() - start_time) + " seconds"

    toimage(skeleton, cmin=0, cmax=255).save(output_filename)
