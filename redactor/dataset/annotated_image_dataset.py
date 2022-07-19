import functools
import json
from glob import glob
import os.path
from typing import Tuple, List
import string

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import cv2

from ..config import CfgNode
from .augmentation import build_augmentations


@functools.lru_cache()
def cached_imread(fn):
    im: np.ndarray = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(f'Error reading image "{fn}"!')
    return im


class AutoAnnotatedImagesDataset(Dataset):

    def __init__(self, tfms, filenames: List[str], image_size: Tuple[int, int]):
        self.transforms = tfms
        self.filenames = filenames
        self.image_size = image_size
        assert len(filenames), 'No filenames supplied to dataset!'

    @classmethod
    def from_config(cls, cfg: CfgNode):
        filenames = []
        for pattern in cfg.data.pattern:
            filenames.extend(glob(pattern))
        tfms = cls.init_transforms(cfg)
        sz = cfg.data.images.size
        return cls(tfms, filenames, (sz, sz))

    @staticmethod
    def init_transforms(cfg: CfgNode):
        augmentations = build_augmentations(cfg.data.images.augmentations)
        init_tfms = [
            transforms.ToPILImage(),
        ]
        final_tfms = [
            transforms.ToTensor()
        ]
        tfms = [*init_tfms, *augmentations, *final_tfms]
        return transforms.Compose(tfms)

    @staticmethod
    def get_random_string():
        return ''.join([np.random.choice(list(string.printable[:-6])) for _ in range(np.random.randint(3, 10))])

    @classmethod
    def add_random_annotation(cls, im, w, h):
        font_kws = dict(
            text=cls.get_random_string(),
            fontFace=np.random.choice([
                cv2.FONT_HERSHEY_SIMPLEX,
                cv2.FONT_HERSHEY_PLAIN,
                cv2.FONT_HERSHEY_DUPLEX,
                cv2.FONT_HERSHEY_COMPLEX,
                cv2.FONT_HERSHEY_TRIPLEX,
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                cv2.FONT_ITALIC,
            ]),
            fontScale=np.random.uniform(1, 5),
            thickness=np.random.randint(1, 4)
        )

        (sw, sh), _ = cv2.getTextSize(**font_kws)

        x, y = (np.random.uniform(0, 1, 2) * [w, h]).astype(int)

        if x + sw > w:
            x = w - sw

        if y + sh > h:
            y = h - sh

        mx, my = 10, 15
        if np.random.uniform() < 0.5 or True:
            c = np.random.randint(0, 255)
            ci = 255 - c
            cv2.rectangle(im, (x - mx, y + my), (x + sw + mx, y - sh - my), thickness=-1, color=ci)
        else:
            mnc = im[y:y - sh, x:x + sw].mean()
            c = 255 if mnc < 127 else 0

        cv2.putText(im, org=(x, y), color=c, **font_kws)
        bbox = x0, y0, x1, y1 = (x - mx), (y - sh - my), (x + sw + mx), (y + my)
        bw, bh = abs(x1 - x0), abs(y1 - y0)
        area = bw * bh
        target = dict(
            boxes=bbox,
            labels=1,
            area=area,
            iscrowd=0,
        )
        return target

    @classmethod
    def get_annotated(cls, fn: str, sz):
        im = cached_imread(fn)
        h, w = im.shape
        if sz is None:
            sz = w, h

        boxes = list()
        labels = list()
        image_id = list()
        area = list()
        iscrowd = list()
        n = np.random.randint(1, 5)
        for i in range(n):
            a_target = cls.add_random_annotation(im, w, h)
            boxes.append(a_target['boxes'])
            labels.append(a_target['labels'])
            image_id.append(123)
            area.append(a_target['area'])
            iscrowd.append(a_target['iscrowd'])

        xf, yf = sz[0]/w, sz[1]/h

        boxes = [
            [
                int(v*(yf if i % 2 else xf))
                for i, v in enumerate(box)
            ]
            for box in boxes
        ]

        target = dict(
            boxes=torch.tensor(boxes).float(),
            labels=torch.tensor(labels).long(),
            image_id=torch.tensor(image_id).long(),
            area=torch.tensor(area),
            iscrowd=torch.tensor(iscrowd).long(),
        )

        im = cv2.resize(im, sz)
        return torch.tensor(np.reshape(im, (1, sz[1], sz[0]))) / 255., target

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i: int):
        return self.get_annotated(self.filenames[i], self.image_size)

    @classmethod
    def build_loaders(cls, config):
        dataset = cls.from_config(config)
        n = len(dataset)
        ntrain = int(n * config.data.frac_train)
        nvalid = n - ntrain
        train, valid = random_split(dataset, [ntrain, nvalid])

        print(f'{len(dataset)} stacks (train: {len(train)}, valid: {len(valid)})')

        def collate_fn(args):
            img = [r[0].to(config.training.device) for r in args]
            tgt = [{k: v.to(config.training.device) for k, v in r[1].items()} for r in args]
            return img, tgt

        kws = dict(batch_size=config.training.batch_size, shuffle=config.training.shuffle_every_epoch, collate_fn=collate_fn)
        return DataLoader(train, **kws), DataLoader(valid, **kws)
