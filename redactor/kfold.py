import warnings
import json

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Subset, DataLoader

from .dataset import build_dataset
from .trainer import Trainer


class KFoldCrossValidation(Trainer):

    def __init__(self, config):
        self.n_folds = config.kfold.n_folds
        self.master_dataset = None
        super().__init__(config)

        dummy_input = (1, config.data.images.stack, 100, 100)
        self.model.forward(torch.zeros(*dummy_input))
        print('Save original model state')
        torch.save(self.model.state_dict(), f'{self.output_dir}/original_state.pth')
        self.batch_size = config.training.batch_size

    def build_loaders(self, config):
        _ = config
        # don't build loaders in init; loaders are built in the validate step later on.
        # however, DO build a dataset
        self.master_dataset = build_dataset(config)
        return None, None

    def cross_validate(self):
        n = orig_n = len(self.master_dataset)
        inv_test_frac = 5

        indices = np.arange(n)
        np.random.shuffle(indices)
        max_n = (n - (n % inv_test_frac))
        indices = indices[:max_n]

        test_indices = indices[:max_n//inv_test_frac]
        train_and_valid_indices = indices[max_n//inv_test_frac:]
        n = len(train_and_valid_indices)
        max_n = (n - (n % self.n_folds))
        train_and_valid_indices = indices[:max_n]
        dataloader_kws = dict(batch_size=self.batch_size, shuffle=True)
        test_set = Subset(self.master_dataset, test_indices)
        self.test_dl = DataLoader(test_set, **dataloader_kws)
        self.should_test = True
        self.save_dataset_contents(self.test_dl, 'test')

        if orig_n != max_n:
            warnings.warn(f'number of folds {self.n_folds} and test fraction 1/{inv_test_frac} does not divide into dataset size {orig_n}; {orig_n - max_n} data points will be excluded at random.')
        folds = np.split(train_and_valid_indices, self.n_folds)
        fold_results = [None]*self.n_folds
        for i in range(self.n_folds):
            train_indices = np.array(folds[1:]).flatten()
            valid_indices = folds[0]
            folds = np.roll(folds, 1)

            train_set = Subset(self.master_dataset, train_indices)
            valid_set = Subset(self.master_dataset, valid_indices)
            train_set.sizes = valid_set.sizes = self.master_dataset.sizes

            self.train_dl = DataLoader(train_set, **dataloader_kws)
            self.valid_dl = DataLoader(valid_set, **dataloader_kws)

            print('Reset to original model state')
            self.model.load_state_dict(torch.load(f'{self.output_dir}/original_state.pth'))

            self.prefix = f'fold{i+1}_'
            fold_results[i] = self.train()

        with open(f'{self.output_dir}/all_folds_data.json', 'w') as f:
            json.dump(fold_results, f)

        self.plot_compare_folds(fold_results)
        self.plot_compare_lr(fold_results)

    def plot_compare_folds(self, fold_results):
        x = np.arange(self.n_folds) + 1
        for k in fold_results[0].keys():
            y = [fr[k]['y'][-1] for fr in fold_results]

            plt.figure()
            plt.bar(x, y)
            plt.xlabel('Fold number [#]')
            plt.ylabel(k)
            plt.ylim(bottom=(0.0 if min(y) >= 0.0 else -0.1))
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/fig_kfolds_{k}_comparison.pdf')
            plt.close()

    def plot_compare_lr(self, fold_results):
        for logscaling in [True, False]:
            plt.figure()
            for i, res in enumerate(fold_results):
                x_train, y_train = res['loss.per_epoch.train']['x'], res['loss.per_epoch.train']['y']
                x_valid, y_valid = res['loss.per_epoch.valid']['x'], res['loss.per_epoch.valid']['y']
                plt.plot(x_train, y_train, f'C{i}', label=f'fold{i+1} (train)')
                plt.plot([xi*self.validate_every for xi in x_valid], y_valid, f'C{i}--', alpha=0.5, label=f'fold{i+1} (valid)')
            log_or_lin = 'log' if logscaling else 'lin'
            if logscaling:
                plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/fig_kfolds_lr_valid_v_train_{log_or_lin}.pdf')
        plt.figure()
        
# cb disabled 4/4/2022 - errors out if validate_every is not 1.
#         r = -1000
#         t, b = 0, 1e5
#         for i, res in enumerate(fold_results):
#             train_loss = moving_average(res['loss.per_epoch.train']['y'])
#             valid_loss = moving_average(res['loss.per_epoch.valid']['y'][1:])
#             diff = train_loss - valid_loss
#             r = max(r, len(diff) - 1)
#             t = max([t, *diff[50:]])
#             b = min([b, *diff])
#             plt.plot(diff, f'C{i}', label=f'fold{i + 1}')
#         plt.ylim(top=t, bottom=b)
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss Difference (Train - Valid)')
#         plt.axhline(0.0, color='k', ls='--')
#         plt.text(r, 0.0, 'Learning', ha='right', va='bottom')
#         plt.text(r, 0.0, 'Not Learning', ha='right', va='top')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(f'{self.output_dir}/fig_kfolds_lr_ratio.pdf')


def moving_average(v, w=5, avf=np.mean):
    hw = int(w/2)
    n = len(v)
    rv = np.zeros(n)
    for i in range(len(v)):
        lb = max(0, i-hw)
        ub = min(n, i+hw+1)
        win = v[lb:ub]
        rv[i] = avf(win)
    return rv
