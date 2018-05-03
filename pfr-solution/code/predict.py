import torch
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import os
from tqdm import tqdm
import yaml
from PIL import Image
import itertools

import rd

from PIL import PILLOW_VERSION; assert PILLOW_VERSION=="4.3.0.post0"

n_gpu = 1

def save_prediction(model, model_root, model_desc, batch_size):
    model.train(False)
    folds = 'val test'.split()
    for fold in folds:
        iids = rd.dict_metadata_by_fold(rd.get_val_fold(model_root))[fold].index
        if len(iids) == 0:
            continue
        for iid in tqdm(iids, desc="predict %r on %r" % (model_root, fold), unit="image"):
            output_opt = dict(classes=3, stride=4, atom_size=8, size=40)
            coarsest_step_size = 32
            label_stride = output_opt['stride']
            label_size = output_opt['size']
            label_span = label_stride * label_size
            full_image_size = 1300
            n_classes = output_opt['classes']
            crop_size = 352

            increment_size = coarsest_step_size
            if increment_size % label_stride != 0:
                increment_size *= 3
            assert increment_size % label_stride == 0
            extra_margin = 96
            while 2 * (label_span-extra_margin) < full_image_size:
                label_size += increment_size // label_stride
                label_span = label_stride * label_size
                crop_size += increment_size

            img_extra = (crop_size - label_span)//2
            img_padded = np.array(Image.open(rd.iid_path(iid, 'rgb_aniso_jpg')).crop((-img_extra, -img_extra, full_image_size+img_extra, full_image_size+img_extra)))
            crop_step = full_image_size + 2*img_extra - crop_size
            n_blocks = 2
            img_padded = transforms.ToTensor()(img_padded)
            img_padded = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_padded)
            img_padded = img_padded.cpu()
            crops = (img_padded[:,ij//n_blocks*crop_step:,ij%n_blocks*crop_step:][:,:crop_size,:crop_size] for ij in range(n_blocks**2))
            img_outputs = []
            for _,g in itertools.groupby(enumerate(crops), key=lambda iv: iv[0]//batch_size):
                batch = torch.stack([iv[1] for iv in g]).cuda()
                batch = Variable(batch)
                outputs = model(batch)
                if len(outputs.shape)==4:
                    # batch x chan x size x size -> batch x size x size x chan
                    outputs = outputs.permute(0, 2, 3, 1)
                else:
                    assert len(outputs.shape)==2
                img_outputs.append(outputs.data.cpu().numpy())
                del batch
                del outputs
            img_outputs = np.concatenate(img_outputs).reshape(n_blocks*n_blocks, label_size, label_size, n_classes)

            rem = (label_span - crop_step)//2
            i1 = label_size - rem // label_stride
            i2 = label_size + i1 - (label_span + crop_step) // label_stride
            assert i2>=0
            img_outputs = np.concatenate([
                            np.concatenate([
                             img_outputs[0][:i1,:i1],
                             img_outputs[1][:i1,i2:]], axis=1),
                            np.concatenate([
                             img_outputs[2][i2:,:i1],
                             img_outputs[3][i2:,i2:]], axis=1)], axis=0)
            img_outputs = img_outputs[...,::-1] # BGR for classes 0,1,2
            img_outputs = np.exp(img_outputs)
            img_outputs /= img_outputs.sum(axis=-1)[...,None]
            out = Image.fromarray((255*img_outputs).astype('u1'))
            path = rd.iid_path(iid, 'rgb_pred_%s_png' % model_root)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            out.save(path)

if __name__=='__main__':
    import sys
    args = sys.argv[1:]
    if not args:
        sys.exit("no model specified")

    for yaml_path in args:
        with open(yaml_path) as f:
            model_desc = yaml.load(f)

        model_path = 'trained_models/' + os.path.basename(yaml_path[:-5]) + '.pth'
        model_root = os.path.basename(yaml_path)[:-5]
        batch_size = 2

        model = torch.load(model_path).cuda()
        model.device_ids = list(range(torch.cuda.device_count()))
        save_prediction(model, model_root, model_desc, batch_size)
