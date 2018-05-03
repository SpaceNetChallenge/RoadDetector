from torch.utils.data import Dataset
from torch.nn.modules.module import Module
from torch.nn import functional as F
from PIL import Image
import numpy as np
import random

import rd

class RasterDataset(Dataset):
    def __init__(self, fold, val_fold, augment, model_desc, crop_size, samples_per_epoch, output_opt, data_transform):
        self.fold = fold
        self.val_fold = val_fold
        self.image_metadata = rd.dict_metadata_by_fold(val_fold)[self.fold]
        self.augment = augment
        self.augment_rotation_prob = 0.4
        self.augment_crop_period = model_desc['augment_crop_period']
        self.automatic_clipping = model_desc['automatic_clipping']
        self.drop_margin = model_desc['drop_margin']
        self.samples_per_epoch = samples_per_epoch
        self.output_opt = output_opt
        self.data_transform = data_transform
        self.label_span = output_opt['stride'] * output_opt['size']
        self.image_size = 1300
        self.crop_size = crop_size
        if self.fold == 'val':
            self.seed = 0
        else:
            self.seed = None

    def is_valid(self):
        return len(self.image_metadata) > 0

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        rng = random if self.seed is None else random.Random((self.seed, index))
        iid = rng.choice(self.image_metadata.index)
        interval = self.image_size - self.label_span + 1
        i = rng.randrange(interval)
        j = rng.randrange(interval)
        img_extra = (self.crop_size - self.label_span)//2
        img = np.array(Image.open(rd.iid_path(iid, 'rgb_aniso_jpg')).crop((j-img_extra, i-img_extra, j-img_extra+self.crop_size, i-img_extra+self.crop_size)))

        if self.augment:
            lab_extra = self.label_span//2
            assert lab_extra % self.output_opt['stride'] == 0
            lab_divisor = self.output_opt['stride']
        else:
            lab_extra = 0

        # We reserve the value 255 for unknown labels
        dist_png = np.asarray(Image.open(rd.iid_path(iid, 'dist_aniso_png'))).clip(0,254)
        if self.automatic_clipping >= 2:
            drop_margin = self.drop_margin
            if drop_margin:
                dist_png[:drop_margin] = 255
                dist_png[-drop_margin:] = 255
                dist_png[:,:drop_margin] = 255
                dist_png[:,-drop_margin:] = 255
            dist_png[np.asarray(Image.open(rd.iid_path(iid, 'mask_aniso_png')))==0] = 255
        lab = 255 - np.asarray(Image.fromarray(np.uint8(255) - dist_png).crop((j-lab_extra, i-lab_extra, j+self.label_span+lab_extra, i+self.label_span+lab_extra)))

        lab = lab[self.output_opt['stride']//2::self.output_opt['stride']]
        lab = lab[:,self.output_opt['stride']//2::self.output_opt['stride']]
        if self.augment:
            if rng.random() < .5:
                img = img[::-1]
                lab = lab[::-1]
            if rng.random() < self.augment_rotation_prob:
                angle = rng.random() * 360
                img = np.array(Image.fromarray(img).rotate(angle, Image.BICUBIC))
                lab = np.uint8(255)-lab
                lab = np.array(Image.fromarray(lab).rotate(angle, Image.BILINEAR))
                lab = np.uint8(255)-lab
            else:
                if rng.random() < .5:
                    img = img[:,::-1]
                    lab = lab[:,::-1]
                if rng.random() < .5:
                    img = img.swapaxes(0,1)
                    lab = lab.T
                # Pytorch doesn't support negative strides currently
                img = img.copy()
                lab = lab.copy()
            if self.augment_crop_period is not None:
                for axis in (0,1):
                    crop_pos = rng.randrange(self.augment_crop_period)
                    if crop_pos < img.shape[0]:
                        lab_crop_pos = (crop_pos-img_extra+lab_extra + lab_divisor//2)//lab_divisor
                        if crop_pos < img.shape[0]//2:
                            img[:crop_pos] = 0
                            if 0 <= lab_crop_pos < len(lab):
                                lab[:lab_crop_pos] = 255
                        else:
                            img[crop_pos:] = 0
                            if 0 <= lab_crop_pos < len(lab):
                                lab[lab_crop_pos:] = 255
                    img = img.swapaxes(0,1)
                    lab = lab.T
        if lab_extra:
            lab = lab[lab_extra//lab_divisor:-lab_extra//lab_divisor, lab_extra//lab_divisor:-lab_extra//lab_divisor]
        img = self.data_transform(img)
        lab = lab.ravel()
        label_dtype = np.int64
        old_lab = lab
        assert self.output_opt['classes'] == 3
        lab = (lab < rd.DIST_FACTOR * 12).astype(label_dtype) + (lab < rd.DIST_FACTOR * 6)
        assert lab.dtype==label_dtype

        if self.automatic_clipping:
            lab[old_lab==255] = -100 # standard value for ignore_index

        return (img, lab)


class MultiCrossEntropyLoss(Module):
    def __init__(self, n_classes, size_average=True):
        super(MultiCrossEntropyLoss, self).__init__()
        self.n_classes = n_classes
        self.size_average = size_average

    def forward(self, input, target):
        assert not target.requires_grad
        if len(input.shape) == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
        input = input.view(-1, self.n_classes)
        target = target.view(-1)
        assert input.shape[:1]==target.shape
        if not self.size_average:
            return F.cross_entropy(input, target, size_average=False).mul_(1.0 / target.size(0))
        else:
            return F.cross_entropy(input, target, size_average=True)

