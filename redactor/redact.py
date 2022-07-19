from typing import List

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

from .config import get_config
from .model import build_model
from .util import get_latest_model, imread
from .dataset.annotated_image_dataset import AutoAnnotatedImagesDataset


class Redactor:
    def __init__(self, res_dir: str = None):

        if res_dir is None:
            res_dir = get_latest_model()
        config = get_config()
        config.merge_from_file(f'{res_dir}/config.yaml')
        config.model.weights = f'{res_dir}/model_state_final.pth'

        self.model = build_model(config)
        self.model.eval()

    def redact_image(self, image):
        with torch.no_grad():
            out = self.model([image])[0]
        npy_img = (image.cpu().numpy()*255).astype('uint8').squeeze()
        for bbox in out['boxes']:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            cv2.rectangle(npy_img, (x0-5, y0-10, x1-x0+5, y1-y0+10), 0, -1)
        return npy_img

    def annot_and_redact(self, fn: str, sz=None):
        rand_image, rand_target = AutoAnnotatedImagesDataset.get_annotated(fn, sz)
        redacted = self.redact_image(rand_image)

        fig, (oax, tax, rax) = plt.subplots(ncols=3, figsize=(15, 5))
        for ax in [oax, tax, rax]:
            plt.sca(ax)
            plt.axis('off')

        orig_img = imread(fn, size=sz)
        oax.imshow(orig_img, cmap='gray')
        tax.imshow(rand_image.numpy().squeeze(), cmap='gray')
        rax.imshow(redacted, cmap='gray')
        plt.tight_layout()
        plt.show()
