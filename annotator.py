from glob import glob
import string

import numpy as np
import cv2

import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader


def get_random_string():
    return ''.join([np.random.choice(list(string.printable[:-6])) for _ in range(np.random.randint(3, 10))])


def get_annotated(fn: str):
    im = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape

    font_kws = dict(
        text=get_random_string(),
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

    x, y = (np.random.uniform(0, 1, 2)*[w, h]).astype(int)

    if x + sw > w:
        x = w - sw

    if y + sh > h:
        y = h - sh

    mx, my = 10, 15
    if np.random.uniform() < 0.5:
        c = np.random.randint(0, 255)
        ci = 255 - c
        cv2.rectangle(im, (x-mx, y+my), (x+sw+mx, y-sh-my), thickness=-1, color=ci)
    else:
        mnc = im[y:y-sh, x:x+sw].mean()
        c = 255 if mnc < 127 else 0

    cv2.putText(im, org=(x, y), color=c, **font_kws)
    bbox = x0, y0, x1, y1 = x-mx, y-sh-my, x+sw+mx, y+my
    bw, bh = abs(x1 - x0), abs(y1 - y0)
    area = bw*bh
    target = dict(
        boxes=torch.tensor([bbox]).float(),
        labels=torch.tensor([1]).long(),
        image_id=torch.tensor([123]).long(),
        area=torch.tensor([area]),
        iscrowd=torch.tensor([0]).long(),
    )
    return torch.tensor([im])/255., target


class Dataset(TorchDataset):

    def __init__(self):
        self.images = glob('E:/Data/NR121214-01/*.bmp')

    def __len__(self):
        return 100

    def __getitem__(self, item):
        return get_annotated(np.random.choice(self.images))


def collate_fn(args):
    img = [r[0] for r in args]
    tgt = [dict(**r[1]) for r in args]
    return img, tgt


if __name__ == '__main__':
    images = glob('E:/Data/NR121214-01/*.bmp')

    dataset = Dataset()
    dl = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    for img, tgt in dl:
        # print(img, tgt)

        out = model(img, tgt)
        print(out)
        break

