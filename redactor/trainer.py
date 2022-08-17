import os
from collections import defaultdict
import json

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

from torchinfo import summary

from .progress_bar import progressbar
from .storage import DataStore
from .util import to_float
from .config import CfgNode
from .model import build_model
from .dataset import build_loaders
from .metrics import build_metrics
from .plotter import build_plotter
from .optim import build_optim
from .sched import build_sched


class Trainer:

    def __init__(self, config: CfgNode):
        self.model = build_model(config)
        if config.debug_mode:
            print(self.model)
            print(summary(self.model, (1, config.data.images.stack, 256, 256), device='cpu'))
        self.train_dl, self.valid_dl = self.build_loaders(config)
        self.test_dl = None
        self.should_test = False
        self.metrics = build_metrics(config)
        self.plotter = build_plotter(self)
        self.opt_t, self.opt_kws = build_optim(config)
        self.sched_t, self.sched_kws = build_sched(config, len(self.train_dl))

        with open(f'{config.output_dir}/config.yaml', 'w') as f:
            f.write(config.dump())

        self.n_epochs = config.training.n_epochs
        self.device = torch.device(config.training.device)
        self.output_dir = config.output_dir
        self.plot_every = config.training.plot_every
        self.checkpoint_every = config.training.checkpoint_every
        self.validate_every = config.training.validate_every
        self.i = self.tbn = self.vbn = self.total_valid_loss = self.total_train_loss = self.total_test_loss = 0
        self.min_valid_loss = np.inf
        self.bar = self.last_checkpoint = None

        # used by sub_classes
        self.prefix = ''

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.checkpoint()

    def build_loaders(self, config):
        return build_loaders(config)

    @property
    def should_plot(self) -> bool:
        return (self.i % self.plot_every == 0) or (self.i == self.n_epochs - 1)

    @property
    def should_validate(self) -> bool:
        return (self.i % self.validate_every == 0) or self.should_plot or self.validate_every == 1

    @property
    def should_checkpoint(self) -> bool:
        return self.i % self.checkpoint_every == 0

    @staticmethod
    def init_store_metadata(store):
        store.add_metadata_re('loss.*', logy=True)
        store.add_metadata('learning_rate', xlabel='Batch [#]')
        store.add_metadata_re(r'.*\.per_batch\..*', xlabel='Training Batch [#]')
        store.add_metadata_re(r'.*\.per_batch\.valid', xlabel='Valid Batch [#]')
        store.add_metadata_re(r'.*\.per_epoch\..*', xlabel='Epoch [#]')
        store.add_metadata_re(r'metrics\..*', xlabel='Epoch [#]')
        store.add_metadata_re(r'metrics\.by_class\..*', plot_timeseries=False)

    def do_train(self, store, opt, scheduler):
        self.model.train()
        for inp, tgt in self.train_dl:
            opt.zero_grad()
            out = self.model(inp, tgt)
            loss = sum(out.values())

            train_loss = loss.item()
            self.total_train_loss += train_loss
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
                store.add_scalar('learning_rate', self.tbn, scheduler.get_last_lr()[0])
            else:
                store.add_scalar('learning_rate', self.tbn, self.opt_kws['lr'])
            self.tbn += 1
            store.add_scalar('loss.per_batch.train', self.tbn, train_loss / len(inp))

        store.add_scalar('loss.per_epoch.train', self.i, self.total_train_loss / len(self.train_dl))

    def validate_or_test(self, dataloader, store, is_test: bool):

        metrics = defaultdict(list)
        valid_or_test = 'test' if is_test else 'valid'

        # if self.should_plot:
        #     self.plotter.init()

        with torch.no_grad():
            for i, (inp, tgt) in enumerate(dataloader):
                # inp = inp.to(self.device)
                # tgt = tgt.to(self.device)
                self.model.train()
                losses = self.model(inp, tgt)
                _loss = float(sum(losses.values()).item())
                if is_test:
                    self.total_test_loss += _loss
                else:
                    self.total_valid_loss += _loss
                    self.vbn += 1

                store.add_scalar(f'loss.per_batch.{valid_or_test}', self.vbn, _loss / len(inp))

                if i == 0:
                    self.model.eval()
                    out = self.model(inp)

                    # Qualitative appraisal
                    for j, (img, res, goal) in enumerate(zip(inp, out, tgt)):

                        if j >= 1:
                            break

                        img = (img.detach().cpu().numpy()*255).astype('uint8')[0]
                        h, w = img.shape
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                        for box in res['boxes']:
                            box = box.detach().cpu().numpy().astype(int)
                            x0, y0, x1, y1 = box

                            x0 = min(w-1, x0)
                            x1 = min(w-1, x1)
                            y0 = min(h-1, y0)
                            y1 = min(h-1, y1)

                            x0 = max(0, x0)
                            x1 = max(0, x1)
                            y0 = max(0, y0)
                            y1 = max(0, y1)

                            img = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 3)

                        for box in goal['boxes']:
                            box = box.detach().cpu().numpy().astype('uint8')
                            x0, y0, x1, y1 = box
                            img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)

                        cv2.imwrite(f'{self.output_dir}/{self.prefix}qual_{j}_epoch={self.i}.jpg', img)


                # for metric_name, metric_func in self.metrics.items():
                #     for o, t, tag in zip(out, tgt):
                #         metric_value = metric_func(o, t)
                #         try:
                #             metric_value = to_float(metric_value)
                #         except TypeError:
                #             print(metric_name)
                #             raise
                #         metrics[f'metrics.{valid_or_test}.{metric_name}'].append(metric_value)

                # self.plot_valid_batch(axes, tgt, out)
                # if self.should_plot:
                #     self.plotter.plot_targets(batch, out)

            # for key, values in metrics.items():
            #     store.add_scalar(key, self.i, np.mean(values))

            # self.plot_finalise(axes, is_test=is_test)
            # if self.should_plot:
            #     self.plotter.finalise()
        store.add_scalar(f'loss.per_epoch.{valid_or_test}', self.i,
                         (self.total_test_loss if is_test else self.total_valid_loss) / len(dataloader))

    def do_validation(self, store):
        self.validate_or_test(self.valid_dl, store=store, is_test=False)

    def do_test(self, store):
        assert self.should_test
        assert self.test_dl is not None
        self.validate_or_test(self.test_dl, store=store, is_test=True)

    def train(self):
        self.i = self.tbn = self.vbn = self.total_valid_loss = self.total_train_loss = 0
        self.min_valid_loss = np.inf

        if os.path.exists(f'{self.output_dir}/training_extent_state.json'):
            with open(f'{self.output_dir}/training_extent_state.json') as f:
                training_extent_state = json.load(f)
            self.i = training_extent_state['i']
            self.tbn = training_extent_state['tbn']
            self.vbn = training_extent_state['vbn']

        self.device = torch.device(self.device)
        print(f'Running on {self.device}')
        self.model = self.model.to(self.device)

        with open(f'{self.output_dir}/model.txt', 'w') as f:
            f.write(str(self.model))

        opt = self.opt_t(self.model.parameters(), **self.opt_kws)
        if os.path.exists(f'{self.output_dir}/opt_state.pth'):
            opt.load_state_dict(torch.load(f'{self.output_dir}/opt_state.pth'))
        if self.sched_t is not None:
            scheduler = self.sched_t(opt, **self.sched_kws)
        else:
            scheduler = None
        self.bar = progressbar(
            range(self.n_epochs),
            unit='epoch',
            total=self.i + self.n_epochs,
            initial=self.i
        )
        with DataStore(self.output_dir, prefix=self.prefix) as store:
            self.init_store_metadata(store)
            self.do_validation(store)
            for _ in self.bar:
                self.i += 1
                self.total_train_loss = 0
                self.do_train(store, opt, scheduler)
                if self.should_validate:
                    self.total_valid_loss = 0.0
                    self.do_validation(store)

                if self.should_plot:
                    store.plot()

                if self.should_checkpoint:
                    self.checkpoint()
                    self.last_checkpoint = self.i

                if self.total_valid_loss / len(self.valid_dl) < self.min_valid_loss:
                    self.min_valid_loss = self.total_valid_loss / len(self.valid_dl)
                    torch.save(self.model.state_dict(), f'{self.output_dir}/{self.prefix}model_min_loss.pth')

                self.update_progress()
                torch.save(opt.state_dict(), f'{self.output_dir}/opt_state.pth')
                store.save()
            self.checkpoint()
            if self.should_test:
                assert self.test_dl
                print('Loading best model state (decided based on validation set results)')
                self.model.load_state_dict(torch.load(f'{self.output_dir}/{self.prefix}model_min_loss.pth'))
                print('Running on test dataset')
                self.do_test(store)
            return store.get_data()

    def checkpoint(self):
        torch.save(self.model.state_dict(), f'{self.output_dir}/{self.prefix}model_state_final.pth')
        torch.save(self.model.state_dict(), f'{self.output_dir}/{self.prefix}model_state_at_epoch={self.i}.pth')

    def update_progress(self):
        desc = f'{self.prefix}t:{self.total_train_loss / len(self.train_dl):.2e}|v:{self.total_valid_loss / len(self.valid_dl):.2e}|'
        if self.last_checkpoint is None:
            desc += '!*'
        else:
            desc += f'c:{self.last_checkpoint}'
            if self.i > self.last_checkpoint:
                desc += '*'
        self.bar.set_description(desc, False)

        training_extent_state = dict()
        training_extent_state['i'] = self.i
        training_extent_state['tbn'] = self.tbn
        training_extent_state['vbn'] = self.vbn

        with open(f'{self.output_dir}/training_extent_state.json', 'w') as f:
            json.dump(training_extent_state, f)
