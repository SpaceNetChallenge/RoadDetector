import random
import numpy as np
import matplotlib.pyplot as plt

class ImageCropper:
    def __init__(self, img_rows, img_cols, target_rows, target_cols, pad):
        self.image_rows = img_rows
        self.image_cols = img_cols
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.pad = pad
        self.use_crop = (img_rows != target_rows) or (img_cols != target_cols)
        self.starts_y = self.sequential_starts(axis=0) if self.use_crop else [0]
        self.starts_x = self.sequential_starts(axis=1) if self.use_crop else [0]
        self.positions = [(x, y) for x in self.starts_x for y in self.starts_y]
        # self.lock = threading.Lock()

    def random_crop_coords(self):
        x = random.randint(0, self.image_cols - self.target_cols)
        y = random.randint(0, self.image_rows - self.target_rows)
        return x, y

    def crop_image(self, image, x, y):
        return image[y: y+self.target_rows, x: x+self.target_cols,...] if self.use_crop else image

    def sequential_starts(self, axis=0):
        #dumb thing
        best_dist = float('inf')
        best_starts = None
        big_segment = self.image_cols if axis else self.image_rows
        small_segment = self.target_cols if axis else self.target_rows
        opt_val = len(np.arange(0, big_segment, small_segment - self.pad)) - 1
        for i in range(small_segment - self.pad):
            r = np.arange(0, big_segment, small_segment - self.pad - i)
            minval = abs(big_segment - small_segment - r[opt_val])
            if minval < best_dist:
                best_dist = minval
                best_starts = r
            else:
                starts = best_starts[:opt_val].tolist() + [big_segment - small_segment]
                return starts

    def sequential_crops(self, img):
        for startx in self.starts_x:
            for starty in self.starts_y:
                yield self.crop_image(img, startx, starty)

#dbg functions
def starts_to_mpl(starts, t):
    ends = np.array(starts) + t
    data = []
    prev_e = None
    for idx, (s, e) in enumerate(zip(starts, ends)):
        # if prev_e is not None:
        #     data.append((prev_e, s))
        #     data.append((idx-1, idx-1))
        #     data.append('b')
        #     data.append((prev_e, s))
        #     data.append((idx, idx))
        #     data.append('b')
        data.append((s, e))
        data.append((idx, idx))
        data.append('r')

        prev_e = e
        if idx > 0:
            data.append((s, s))
            data.append((idx-1, idx))
            data.append('g--')
        if idx < len(starts) - 1:
            data.append((e, e))
            data.append((idx, idx+1))
            data.append('g--')

    return data

def calc_starts_and_visualize(c, tr, tc):
    starts_rows = c.sequential_starts(axis=0)
    data_rows = starts_to_mpl(starts_rows, tr)
    starts_cols = c.sequential_starts(axis=1)
    data_cols = starts_to_mpl(starts_cols, tc)

    f, axarr = plt.subplots(1, 2, sharey=True)
    axarr[0].plot(*data_rows)
    axarr[0].set_title('rows')
    axarr[1].plot(*data_cols)
    axarr[1].set_title('cols')
    plt.show()


if __name__ == '__main__':
    opts = 2072, 2072, 1024, 1024, 0
    c = ImageCropper(*opts)
    calc_starts_and_visualize(c, opts[2], opts[3])
