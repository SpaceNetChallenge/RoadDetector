from osgeo import gdal
import os
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, watershed
import tqdm

def rlencode(x, dropna=False):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle
    function from R.

    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.

    Returns
    -------
    start positions, run lengths, run values

    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]

    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]

    return starts, lengths, values

def rldecode(starts, lengths, values, minlength=None):
    """
    Decode a run-length encoding of a 1D array.

    Parameters
    ----------
    starts, lengths, values : 1D array_like
        The run-length encoding.
    minlength : int, optional
        Minimum length of the output array.

    Returns
    -------
    1D array. Missing data will be filled with NaNs.

    """
    starts, lengths, values = map(np.asarray, (starts, lengths, values))
    ends = starts + lengths
    n = ends[-1]
    if minlength is not None:
        n = max(minlength, n)
    x = np.full(n, np.nan)
    for lo, hi, val in zip(starts, ends, values):
        x[lo:hi] = val
    return x

def rle_to_string(rle):
    (starts, lengths, values) = rle
    items = []
    for i in range(len(starts)):
        items.append(str(values[i]))
        items.append(str(lengths[i]))
    return ",".join(items)


def my_watershed(mask1, mask2):
    markers = ndi.label(mask2, output=np.uint32)[0]
    labels = watershed(mask1, markers, mask=mask1, watershed_line=True)
    return labels


def make_submission(prediction_dir, data_dir, submission_file):
    # 8881 - 0.3 / +0.4 / 100 / 120 test 8935
    threshold = 0.3
    f_submit = open(submission_file, "w")
    strings = []
    predictions = list(sorted(os.listdir(prediction_dir)))

    for f in tqdm.tqdm(predictions):
        if 'xml' in f:
            continue
        dsm_ds = gdal.Open(os.path.join(data_dir, f.replace('RGB', 'DSM')), gdal.GA_ReadOnly)
        band_dsm = dsm_ds.GetRasterBand(1)
        nodata = band_dsm.GetNoDataValue()
        dsm = band_dsm.ReadAsArray()
        tile_id = f.split('_RGB.tif')[0]
        mask_ds = gdal.Open(os.path.join(prediction_dir, f))
        mask_img = mask_ds.ReadAsArray()
        mask_img[dsm==nodata] = 0

        img_copy = np.copy(mask_img)
        img_copy[mask_img <= threshold + 0.4] = 0
        img_copy[mask_img > threshold + 0.4] = 1
        img_copy = img_copy.astype(np.bool)
        img_copy = remove_small_objects(img_copy, 100).astype(np.uint8)

        mask_img[mask_img <= threshold] = 0
        mask_img[mask_img > threshold] = 1
        mask_img = mask_img.astype(np.bool)
        mask_img = remove_small_objects(mask_img, 120).astype(np.uint8)

        labeled_array = my_watershed(mask_img, img_copy)

        # labeled_array = remove_on_boundary(labeled_array)
        rle_str = rle_to_string(rlencode(labeled_array.flatten()))
        s = "{tile_id}\n2048,2048\n{rle}\n".format(tile_id=tile_id, rle=rle_str)
        strings.append(s)

    f_submit.writelines(strings)
    f_submit.close()


