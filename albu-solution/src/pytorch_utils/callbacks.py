import torch
from copy import deepcopy
import os
from tensorboardX import SummaryWriter

class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        self.trainer = None
        self.estimator = None
        self.metrics_collection = None

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.metrics_collection = trainer.metrics_collection
        self.estimator = trainer.estimator

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class Callbacks(Callback):
    def __init__(self, callbacks):
        super().__init__()
        if isinstance(callbacks, Callbacks):
            callbacks = callbacks.callbacks
        self.callbacks = callbacks
        if callbacks is None:
            self.callbacks = []

    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch):
        for callback in self.callbacks:
            callback.on_batch_end(batch)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


class StepLR(Callback):
    def __init__(self, init_lr=0.1, num_epochs_per_decay=10, lr_decay_factor=0.1):
        super().__init__()
        self.init_lr = init_lr
        self.num_epochs_per_decay = num_epochs_per_decay
        self.lr_decay_factor = lr_decay_factor

    def on_epoch_begin(self, epoch):
        lr = self.init_lr * (self.lr_decay_factor ** (epoch // self.num_epochs_per_decay))
        for param_group in self.estimator.optimizer.param_groups:
            param_group['lr'] = lr


class MultiStepLR(Callback):
    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch):
        for param_group in self.estimator.optimizer.param_groups:
            param_group['lr'] = min(self.schedule[epoch], param_group['lr'])


# @deprecated, todo rewrite as lr_scheduler
class CyclicLR(Callback):
    def __init__(self, save_name, init_lr=1e-4, num_epochs_per_cycle=5, cycle_epochs_decay=2, lr_decay_factor=0.5):
        super().__init__()
        self.init_lr = init_lr
        self.num_epochs_per_cycle = num_epochs_per_cycle
        self.cycle_epochs_decay = cycle_epochs_decay
        self.lr_decay_factor = lr_decay_factor
        self.weights_name = save_name
        self.best_loss = float('inf')

    def on_epoch_begin(self, epoch):
        ep = epoch - self.estimator.start_epoch
        epoch_in_cycle = ep % self.num_epochs_per_cycle
        lr = self.init_lr * (self.lr_decay_factor ** (epoch_in_cycle // self.cycle_epochs_decay))
        for param_group in self.estimator.optimizer.param_groups:
            param_group['lr'] = lr


    # def on_epoch_end(self, epoch):
    #     ep = epoch - self.estimator.start_epoch
    #     epoch_in_cycle = ep % self.num_epochs_per_cycle
    #     cycle_num = int(ep / self.num_epochs_per_cycle)
    #     loss = float(self.metrics_collection.val_metrics['loss'])
    #     if epoch_in_cycle == 0:
    #         self.best_loss = float('inf')
    #     if loss < self.best_loss:
    #         self.best_loss = loss

            # torch.save(deepcopy(self.estimator.model.module),
            #            os.path.join(self.estimator.save_path, self.weights_name)
            #            .format(cycle=cycle_num))


class ModelSaver(Callback):
    def __init__(self, save_every, save_name, best_only=True):
        super().__init__()
        self.save_every = save_every
        self.save_name = save_name
        self.best_only = best_only

    def on_epoch_end(self, epoch):
        loss = float(self.metrics_collection.val_metrics['loss'])
        if epoch % self.save_every == 0:
            if (not self.best_only) or loss < self.metrics_collection.best_loss:
                self.metrics_collection.best_loss = loss
                self.metrics_collection.best_epoch = epoch

                torch.save(deepcopy(self.estimator.model.module),
                           os.path.join(self.estimator.save_path, self.save_name)
                           .format(epoch=epoch, loss="{:.2}".format(loss)))

def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, path):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model_state_dict,
        'optimizer': optimizer_state_dict,
    }, path)



class CheckpointSaver(Callback):
    def __init__(self, save_every, save_name):
        super().__init__()
        self.save_every = save_every
        self.save_name = save_name

    def on_epoch_end(self, epoch):
        loss = float(self.metrics_collection.val_metrics['loss'])
        if epoch % self.save_every == 0:
            save_checkpoint(epoch,
                            self.estimator.model.module.state_dict(),
                            self.estimator.optimizer.state_dict(),
                            os.path.join(self.estimator.save_path, self.save_name).format(epoch=epoch, loss="{:.2}".format(loss)))


class EarlyStopper(Callback):
    def __init__(self, patience):
        super().__init__()
        self.patience = patience

    def on_epoch_end(self, epoch):
        loss = float(self.metrics_collection.val_metrics['loss'])
        if loss < self.metrics_collection.best_loss:
            self.metrics_collection.best_loss = loss
            self.metrics_collection.best_epoch = epoch
        if epoch - self.metrics_collection.best_epoch >= self.patience:
            self.metrics_collection.stop_training = True

class TensorBoard(Callback):
    def __init__(self, logdir):
        super().__init__()
        self.logdir = logdir
        self.writer = None

    def on_train_begin(self):
        os.makedirs(self.logdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

    def on_epoch_end(self, epoch):
        for k, v in self.metrics_collection.train_metrics.items():
            self.writer.add_scalar('train/{}'.format(k), float(v), global_step=epoch)

        for k, v in self.metrics_collection.val_metrics.items():
            self.writer.add_scalar('val/{}'.format(k), float(v), global_step=epoch)

        for idx, param_group in enumerate(self.estimator.optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=epoch)

    def on_train_end(self):
        self.writer.close()

class TelegramSender(Callback):
    def on_train_end(self):
        from telegram_send import send as send_telegram
        message = "Finished on {} with best loss {} on epoch {}".format(
            self.trainer.devices,
            self.trainer.metrics_collection.best_loss or self.metrics_collection.val_metrics['loss'],
            self.trainer.metrics_collection.best_epoch or 'last')
        try:
            send_telegram(messages=message, conf='tg_config.conf')
        except:
            pass
