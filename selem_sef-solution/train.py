import gc
import os

import numpy as np

from model_name_encoder import encode_params
from params import args
from transormer import RandomTransformer

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from datasets.spacenet import generate_ids, MULSpacenetDataset, get_groundtruth
from tools.clr import CyclicLR

from sklearn.model_selection._split import KFold

from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import RMSprop, Adam, SGD

from losses import make_loss, dice_coef_clipped, binary_crossentropy, dice_coef, ceneterline_loss
from models import make_model
from params import args
import keras.backend as K

def freeze_model(model, freeze_before_layer):
    if freeze_before_layer == "ALL":
        for l in model.layers:
            l.trainable = False
    else:
        freeze_before_layer_index = -1
        for i, l in enumerate(model.layers):
            if l.name == freeze_before_layer:
                freeze_before_layer_index = i
        for l in model.layers[:freeze_before_layer_index]:
            l.trainable = False


def main():
    if args.crop_size:
        print('Using crops of shape ({}, {})'.format(args.crop_size, args.crop_size))
    else:
        print('Using full size images')

    all_ids = np.array(generate_ids(args.data_dirs, args.clahe))
    np.random.seed(args.seed)
    kfold = KFold(n_splits=args.n_folds, shuffle=True)

    splits = [s for s in kfold.split(all_ids)]
    folds = [int(f) for f in args.fold.split(",")]
    for fold in folds:
        encoded_alias = encode_params(args.clahe, args.preprocessing_function, args.stretch_and_mean)
        city = "all"
        if args.city:
            city = args.city.lower()
        best_model_file = '{}/{}_{}_{}.h5'.format(args.models_dir, encoded_alias, city, args.network)
        channels = 8
        if args.ohe_city:
            channels = 12
        model = make_model(args.network, (None, None, channels))

        if args.weights is None:
            print('No weights passed, training from scratch')
        else:
            print('Loading weights from {}'.format(args.weights))
            model.load_weights(args.weights, by_name=True)
        freeze_model(model, args.freeze_till_layer)

        optimizer = RMSprop(lr=args.learning_rate)
        if args.optimizer:
            if args.optimizer == 'rmsprop':
                optimizer = RMSprop(lr=args.learning_rate)
            elif args.optimizer == 'adam':
                optimizer = Adam(lr=args.learning_rate)
            elif args.optimizer == 'sgd':
                optimizer = SGD(lr=args.learning_rate, momentum=0.9, nesterov=True)

        train_ind, test_ind = splits[fold]
        train_ids = all_ids[train_ind]
        val_ids = all_ids[test_ind]
        if args.city:
            val_ids = [id for id in val_ids if args.city in id[0]]
            train_ids = [id for id in train_ids if args.city in id[0]]
        print('Training fold #{}, {} in train_ids, {} in val_ids'.format(fold, len(train_ids), len(val_ids)))
        masks_gt = get_groundtruth(args.data_dirs)
        if args.clahe:
            template = 'CLAHE-MUL-PanSharpen/MUL-PanSharpen_{id}.tif'
        else:
            template = 'MUL-PanSharpen/MUL-PanSharpen_{id}.tif'

        train_generator = MULSpacenetDataset(
            data_dirs=args.data_dirs,
            wdata_dir=args.wdata_dir,
            clahe=args.clahe,
            batch_size=args.batch_size,
            image_ids=train_ids,
            masks_dict=masks_gt,
            image_name_template=template,
            seed=args.seed,
            ohe_city=args.ohe_city,
            stretch_and_mean=args.stretch_and_mean,
            preprocessing_function=args.preprocessing_function,
            crops_per_image=args.crops_per_image,
            crop_shape=(args.crop_size, args.crop_size),
            random_transformer=RandomTransformer(horizontal_flip=True, vertical_flip=True),
        )

        val_generator = MULSpacenetDataset(
            data_dirs=args.data_dirs,
            wdata_dir=args.wdata_dir,
            clahe=args.clahe,
            batch_size=1,
            image_ids=val_ids,
            image_name_template=template,
            masks_dict=masks_gt,
            seed=args.seed,
            ohe_city=args.ohe_city,
            stretch_and_mean=args.stretch_and_mean,
            preprocessing_function=args.preprocessing_function,
            shuffle=False,
            crops_per_image=1,
            crop_shape=(1280, 1280),
            random_transformer=None
        )
        best_model = ModelCheckpoint(filepath=best_model_file, monitor='val_dice_coef_clipped',
                                     verbose=1,
                                     mode='max',
                                     save_best_only=False,
                                     save_weights_only=True)
        model.compile(loss=make_loss(args.loss_function),
                      optimizer=optimizer,
                      metrics=[dice_coef, binary_crossentropy, ceneterline_loss, dice_coef_clipped])

        def schedule_steps(epoch, steps):
            for step in steps:
                if step[1] > epoch:
                    print("Setting learning rate to {}".format(step[0]))
                    return step[0]
            print("Setting learning rate to {}".format(steps[-1][0]))
            return steps[-1][0]

        callbacks = [best_model, EarlyStopping(patience=20, verbose=1, monitor='val_dice_coef_clipped', mode='max')]

        if args.schedule is not None:
            steps = [(float(step.split(":")[0]), int(step.split(":")[1])) for step in args.schedule.split(",")]
            lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, steps))
            callbacks.insert(0, lrSchedule)

        if args.clr is not None:
            clr_params = args.clr.split(',')
            base_lr = float(clr_params[0])
            max_lr = float(clr_params[1])
            step = int(clr_params[2])
            mode = clr_params[3]
            clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step, mode=mode)
            callbacks.append(clr)

        steps_per_epoch = len(all_ids) / args.batch_size + 1
        if args.steps_per_epoch:
            steps_per_epoch = args.steps_per_epoch

        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            validation_data=val_generator,
            validation_steps=len(val_ids),
            callbacks=callbacks,
            max_queue_size=30,
            verbose=1,
            workers=args.num_workers)

        del model
        K.clear_session()
        gc.collect()


if __name__ == '__main__':
    main()
