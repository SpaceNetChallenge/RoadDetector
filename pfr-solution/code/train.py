
# Includes code from:
#
#   Name: PyTorch Transfer Learning tutorial
#   License: BSD
#   Author: Sasank Chilamkurthy <https://chsasank.github.io>

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import time
import os
from collections import OrderedDict
from tqdm import tqdm
import sys
import yaml
import pandas as pd

import pytorch_utils
import rd

from PIL import PILLOW_VERSION; assert PILLOW_VERSION=="4.3.0.post0"

data_transforms = {
    'basic_v1': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

n_gpu = torch.cuda.device_count()

base_model = None
yaml_path, = sys.argv[1:]
model_root = os.path.basename(yaml_path)[:-5]
with open(yaml_path) as f:
    model_desc = yaml.load(f)

# Reduce the epoch size if only a subset of the training data is available.
original_samples_per_epoch = 200000
samples_per_epoch = \
    original_samples_per_epoch * len(rd.all_train_iid()) // 2780
validation_samples_per_epoch = samples_per_epoch // 10

crop_size = 352
output_opt = dict(classes=3, stride=4, atom_size=8, size=40)

batch_size_per_gpu = 12
lr_gamma = 0.2

batch_size = n_gpu * batch_size_per_gpu
epoch_milestones = model_desc['epoch_milestones']
n_epochs = epoch_milestones[-1]

data_transform = data_transforms['basic_v1']
val_fold = rd.get_val_fold(model_root)
image_datasets = {
        'train': pytorch_utils.RasterDataset('train', val_fold, 'dihedral', model_desc, crop_size, samples_per_epoch, output_opt, data_transform),
        'val': pytorch_utils.RasterDataset('val', val_fold, False, model_desc, crop_size, validation_samples_per_epoch, output_opt, data_transform),
        }

n_classes = output_opt['classes']
n_atom_outputs = output_opt['atom_size']**2
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=(x=='train'), num_workers=4*n_gpu)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()


######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, first_epoch=0, model_root=None):
    assert model_root is not None

    print("Training %r with %d GPUs" % (model_root, n_gpu))
    if first_epoch != 0:
        print("First epoch will be %d" % first_epoch)

    os.makedirs('training_logs', exist_ok=True)
    with open('training_logs/%s.csv' % model_root, 'w') as log_f:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                if not image_datasets[phase].is_valid():
                    continue

                if epoch < first_epoch:
                    continue

                running_loss = 0.0
                running_corrects = 0
                t0 = time.time()
                last_layer = list(model.modules())[-1]
                last_layer_grad_scale = n_atom_outputs

                # Iterate over data.
                with tqdm(desc="epoch %d/%d: %s" % (epoch, num_epochs, phase),
                          total=len(dataloaders[phase]) * batch_size, unit="images", unit_scale=True) as progress:
                    for data in dataloaders[phase]:
                        # get the inputs
                        inputs, labels = data
                        if epoch==0 and phase=='train' and progress.n==0:
                            from torchvision.utils import save_image
                            save_image(inputs, "/tmp/last_sample_batch_X.jpg", normalize=True)
                            size = output_opt['size']
                            assert labels.dim() == 2
                            pretty_labels = labels.view(labels.size(0), size, size, 1).permute(0, 3, 1, 2).contiguous()
                            save_image(pretty_labels.float(), "/tmp/last_sample_batch_Y.png", range=(0,n_classes-1), normalize=True)
                            pad_size = (inputs.size(2)//output_opt['stride'] - pretty_labels.size(2))//2
                            assert pad_size >= 0
                            save_image(torch.Tensor(np.pad(pretty_labels.numpy().clip(-1,999) + 1, ((0,0), (0,0), (pad_size,pad_size), (pad_size,pad_size)), 'constant', constant_values=0)).float(), "/tmp/last_sample_batch_Y_pad.png", range=(0,n_classes+1-1), normalize=True)

                        # wrap them in Variable
                        if use_gpu:
                            inputs = Variable(inputs.cuda())
                            labels = Variable(labels.cuda())
                        else:
                            inputs, labels = Variable(inputs), Variable(labels)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        outputs = model(inputs)
                        if len(outputs.shape) == 4:
                            if 0:
                                # Used for <raster60
                                _, preds = torch.max(outputs.data.permute(0, 2, 3, 1).contiguous().view(inputs.size(0), -1, n_classes), 2)
                            else:
                                _, preds = None, torch.max(outputs.data, 1)[1].view(inputs.size(0), outputs.size(2)*outputs.size(3))
                        else:
                            _, preds = torch.max(outputs.data.view(inputs.size(0), -1, n_classes), 2)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            for p in last_layer.parameters():
                                p.grad.mul_(last_layer_grad_scale)
                            optimizer.step()

                        # statistics
                        running_loss += loss.data[0] * inputs.size(0)#/32
                        running_corrects += torch.sum((preds == labels.data) | (labels.data == -100)) / labels.size(1)
                        progress.update(len(inputs))
                        del _
                        del outputs, preds, loss, inputs, labels

                stats = OrderedDict()
                stats['epoch'] = epoch
                stats['phase'] = phase
                stats['loss'] = running_loss / dataset_sizes[phase]
                stats['acc'] = running_corrects / dataset_sizes[phase]
                stats['time'] = time.time() - t0

                print('{}  {}'.format(
                    phase, '  '.join('%s: %.4f'%(k,v) for k,v in stats.items() if k not in ('phase', 'epoch'))))

                pd.DataFrame([pd.Series(stats)]).to_csv(log_f, index=False, header=(epoch==first_epoch and phase=='train'))

            print()

    return model


def main_training(base_model=None, model_root=None, load_checkpoint=None):
    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrained model and reset final fully connected layer.
    #

    if load_checkpoint:
        print("Resuming from checkpoint %r" % load_checkpoint)
        model_ft, optimizer_ft, first_epoch = torch.load('checkpoints/%s.pth' % load_checkpoint)
    else:
        first_epoch = 0
        if base_model is None:
            from pytorch_dpn import dpn
            model_ft = dpn.dpn92(pretrained=True)

            conv5_params = model_desc['conv5_params']
            if conv5_params is not None:
                conv5_k, conv5_inc = conv5_params
                original_state_full = model_ft.state_dict()
                original_state_partial = {}
                for k in list(original_state_full.keys()):
                    if k.startswith('features.conv5_1.') or (
                       not k.startswith('features.conv5_') and not k.startswith('classifier.')):
                        original_state_partial[k] = original_state_full[k]
                del model_ft

                model_ft = dpn.DPN(
                    num_init_features=64, k_r=96, groups=32,
                    k_sec=(3, 4, 20, conv5_k), inc_sec=(16, 32, 24, 128),
                    conv5_inc=conv5_inc,
                    num_classes=1000, test_time_pool=True)

                model_ft.load_state_dict(original_state_partial, strict=False)

            model_ft.pooling_kernel_size = 1
            model_ft.pooling_dropped = 3
            model_ft.raster_size = output_opt['atom_size']
            coarsest_step_size = 32
            assert output_opt['size'] * output_opt['stride'] == crop_size - (model_ft.pooling_kernel_size + 2*model_ft.pooling_dropped - 1) * coarsest_step_size
        else:
            model_ft = torch.load(base_model)

        fc_layer_name = list(model_ft.named_modules())[-1][0]

        assert isinstance(getattr(model_ft, fc_layer_name), nn.Conv2d)
        num_ftrs = getattr(model_ft, fc_layer_name).in_channels
        setattr(model_ft, fc_layer_name, nn.Conv2d(num_ftrs, n_classes*n_atom_outputs, kernel_size=1, bias=True))

        if use_gpu:
            model_ft = model_ft.cuda()

        model_ft = nn.DataParallel(model_ft)

        # Observe that all parameters are being optimized
        weight_decay = 1e-7
        lr = 1e-3*batch_size/128
        param_groups = [{ 'params': model_ft.parameters() }]
        optimizer_ft = optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)

    criterion = pytorch_utils.MultiCrossEntropyLoss(n_classes, size_average=model_desc['loss_size_average'])

    step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, epoch_milestones, gamma=lr_gamma)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    model_ft = train_model(model_ft, criterion, optimizer_ft, step_lr_scheduler,
                           num_epochs=n_epochs, first_epoch=first_epoch, model_root=model_root)

    ######################################################################
    #

    os.makedirs('trained_models', exist_ok=True)
    torch.save(model_ft, 'trained_models/%s.pth' % model_root)

if __name__ == '__main__':
    main_training(base_model, model_root)
