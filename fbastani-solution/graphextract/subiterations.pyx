import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def first_subiteration(np.ndarray[DTYPE_t, ndim=2] curr_image, fg_pixels):
    cdef int i,j, p2, p3, p4, p5, p6, p7, p8, p9

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
            if (bool(not p2 and p3) + bool(not p3 and p4) + bool(not p4 and p5) + bool(not p5 and p6) + bool(not p6 and p7) + bool(not p7 and p8) + bool(not p8 and p9) + bool(not p9 and p2) == 1):      
                zero_pixels[(i,j)] = 0
                if p2 == 1: next_pixels[(i-1,j)]=0
                if p3 == 1: next_pixels[(i-1,j+1)]=0
                if p4 == 1: next_pixels[(i,j+1)]=0
                if p5 == 1: next_pixels[(i+1,j+1)]=0
                if p6 == 1: next_pixels[(i+1,j)]=0
                if p7 == 1: next_pixels[(i+1,j-1)]=0
                if p8 == 1: next_pixels[(i,j-1)]=0
                if p9 == 1: next_pixels[(i-1,j-1)]=0

    return zero_pixels.keys(), next_pixels.keys()
    
def second_subiteration(np.ndarray[DTYPE_t, ndim=2] curr_image, fg_pixels):
    cdef int i,j, p2, p3, p4, p5, p6, p7, p8, p9

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
            if (bool(not p2 and p3) + bool(not p3 and p4) + bool(not p4 and p5) + bool(not p5 and p6) + bool(not p6 and p7) + bool(not p7 and p8) + bool(not p8 and p9) + bool(not p9 and p2) == 1):      
                zero_pixels[(i,j)] = 0
                if p2 == 1: next_pixels[(i-1,j)]=0
                if p3 == 1: next_pixels[(i-1,j+1)]=0
                if p4 == 1: next_pixels[(i,j+1)]=0
                if p5 == 1: next_pixels[(i+1,j+1)]=0
                if p6 == 1: next_pixels[(i+1,j)]=0
                if p7 == 1: next_pixels[(i+1,j-1)]=0
                if p8 == 1: next_pixels[(i,j-1)]=0
                if p9 == 1: next_pixels[(i-1,j-1)]=0

    return zero_pixels.keys(), next_pixels.keys()
    
