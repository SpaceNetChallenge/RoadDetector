import numpy as np


def generate_tiles(img, tile_size=512):
    height = img.shape[0]
    width = img.shape[1]
    rows = 2 * height // tile_size - 1
    columns = 2 * width // tile_size - 1
    overlap = tile_size // 2

    images = np.zeros((rows * columns, tile_size, tile_size, img.shape[-1]), dtype="float32")
    i = 0
    for tile_y in range(int(rows)):
        for tile_x in range(int(columns)):
            x1 = tile_x * tile_size - tile_x * overlap
            x2 = x1 + tile_size
            y1 = tile_y * tile_size - tile_y * overlap
            y2 = y1 + tile_size
            slice = img[y1:y2, x1:x2, :]
            images[i, 0:slice.shape[0], 0:slice.shape[1]] = slice[:]
            i += 1
    return images


def combine_tiles(predicted_tiles, tile_size=1024, height=2048, width=2048):
    img = np.zeros((height, width, 1), dtype="float32")
    rows = 2 * height // tile_size - 1
    columns = 2 * width // tile_size - 1
    overlap = tile_size // 2

    offset = overlap // 2
    for i in range(len(predicted_tiles)):
        tile_x = i % columns
        tile_y = i // columns
        start_x = tile_x * tile_size - tile_x * overlap
        start_y = tile_y * tile_size - tile_y * overlap
        offset_x_start = offset
        offset_y_start = offset
        offset_x_end = offset
        offset_y_end = offset

        if tile_x == 0:
            offset_x_start = 0
        if tile_x == columns - 1:
            offset_x_end = 0
        if tile_y == 0:
            offset_y_start = 0
        if tile_y == rows - 1:
            offset_y_end = 0
        tile = np.expand_dims(predicted_tiles[i][offset_y_start:tile_size - offset_y_end, offset_x_start:tile_size - offset_x_end, 0], -1)
        img[start_y + offset_y_start: start_y + tile_size - offset_y_end, start_x + offset_x_start: start_x + tile_size - offset_x_end, :] = tile
    return img