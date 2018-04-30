# clahe
# stretch
# caffe

def encode_params(clahe, mode, stretch):
    result = ""
    if clahe:
        result += "1"
    else:
        result += "0"
    if stretch:
        result += "1"
    else:
        result += "0"

    if mode == 'caffe':
        result += "1"
    else:
        result += "0"
    return result


def decode_params(encoded):
    clahe = encoded[0] == "1"
    caffe_mode = encoded[2] == "1"
    if caffe_mode:
        mode = 'caffe'
    else:
        mode = 'tf'
    stretch = encoded[1] == "1"
    return clahe, mode, stretch
