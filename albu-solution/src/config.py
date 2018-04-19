from collections import namedtuple

Config = namedtuple("Config", [
    "dataset_path",
    "iter_size",
    "folder",
    "img_rows",
    "img_cols",
    "target_rows",
    "target_cols",
    "num_channels",
    "network",
    "dice_weight",
    "optimizer",
    "lr",
    "lr_steps",
    "lr_gamma",
    "batch_size",
    "epoch_size",
    "nb_epoch",
    "predict_batch_size",
    "dbg",
    "save_images",
    "test_pad",
    "train_pad",
    "results_dir"
])


