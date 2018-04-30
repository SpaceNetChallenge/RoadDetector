import numpy as np

def stretch_8bit(bands, lower_percent=0, higher_percent=100):
    out = np.zeros_like(bands).astype(np.float32)
    for i in range(bands.shape[-1]):
        a = 0
        b = 1
        band = bands[:, :, i].flatten()
        filtered = band[band > 0]
        if len(filtered) == 0:
            continue
        c = np.percentile(filtered, lower_percent)
        d = np.percentile(filtered, higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.float32)