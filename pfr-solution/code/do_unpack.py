from __future__ import division, print_function
from osgeo import gdal
from glob import glob
import os
from tqdm import tqdm

def handle_file(path, meta_f):
    assert path.endswith('.tif')
    assert '\\' not in path
    key = path.split('/', 1)[1][:-4]
    ds = gdal.Open(path)
    bands = ds.RasterCount
    assert bands in (1,3,8)
    geo = ds.GetGeoTransform()
    arr_shape = (ds.RasterCount, ds.RasterYSize, ds.RasterXSize)
    meta = arr_shape + geo
    print(','.join(str(x) for x in (key,) + meta), file=meta_f)
    meta_f.flush()

def scan_zone(zone):
    try:
        os.makedirs('unpacked/%s' % zone)
    except os.error:
        pass
    with open('unpacked/%s/geo_transform.csv.tmp'%zone, 'w') as meta_f:
        patterns = [
                'spacenet/%s/PAN/*.tif' % zone,
                'spacenet/%s/MUL/*.tif' % zone,
                #'spacenet/%s/RGB-PanSharpen/*.tif' % zone,
                #'spacenet/%s/MUL-PanSharpen/*.tif' % zone,
                ]
        print('key,bands,height,width,lon,dx_lon,dy_lon,lat,dx_lat,dy_lat', file=meta_f)
        for pattern in patterns:
            for path in sorted(glob(pattern)):
                path = path.replace(r'\\', '/')
                handle_file(path, meta_f)
    os.rename('unpacked/%s/geo_transform.csv.tmp'%zone,
              'unpacked/%s/geo_transform.csv'%zone)


def scan_all():
    zones = sorted(path[9:] for path in glob('spacenet/AOI_*'))
    for zone in tqdm(zones, desc="extract TIFF metadata"):
        scan_zone(zone)

if __name__ == '__main__':
    scan_all()
