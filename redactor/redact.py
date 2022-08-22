import os.path
from typing import List
from zipfile import is_zipfile, ZipFile
from glob import glob
from datetime import datetime

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .config import get_config, CfgNode
from .model import build_model
from .util import get_latest_model, imread
from .dataset.annotated_image_dataset import AutoAnnotatedImagesDataset


class Redactor:
    def __init__(self, config: CfgNode):
        self.device = config.training.device
        self.model = build_model(config)
        self.model.eval()
        self.ds = AutoAnnotatedImagesDataset.from_config(config)

    @classmethod
    def from_res_dir(cls, res_dir: str = None) -> "Redactor":
        if res_dir is None:
            res_dir = get_latest_model()
        config = get_config()
        config.merge_from_file(f'{res_dir}/config.yaml')
        config.model.weights = f'{res_dir}/model_state_final.pth'
        return cls(config)

    @classmethod
    def from_zip(cls, filename) -> "Redactor":
        config = get_config()
        assert is_zipfile(filename), f'File "{filename}" is not a zipfile!'

        print('Loading model...')
        with ZipFile(filename) as f:
            contents = f.filelist
            assert len(contents) == 2, f'Bad zipfile contents: should only have two items: config.yaml and model.pth (got {len(contents)})'
            cfg, state = None, None
            for content in contents:
                if content.filename == 'config.yaml':
                    assert not cfg
                    cfg = content
                else:
                    assert not state
                    state = content

            with f.open(cfg) as f_cfg:
                config.merge_from_other_cfg(CfgNode.load_cfg(f_cfg))

            with f.open(state) as f_state:
                state = torch.load(f_state)

        r = cls(config)
        r.model.load_state_dict(state)
        print('Model loaded.')
        return r

    def redact(self, image_or_list_or_dir_or_pattern: str, output_dir: str = None, redact_filename=False):
        """
        Redact specified images.

        Specified images will not be modified, redacted copies will be placed in $output_dir.

        :param image_or_list_or_dir_or_pattern: What images should be redacted? Can be: (1) a path to a single image (2) a list of paths to images (3) a directory or (4) a glob pattern specifying images.
        :param output_dir: Where to put the redacted images? If left blank, this will default to a creating a new directory with format "REDACTED_{DATE}_{TIME}"
        :param redact_filename: Should the filename of the image be redacted too? By default the output filenames will be the same as the input (with a number appended to prevent collision). If the filename contains sensitive information, this can be omitted by setting this to 'True'.
        """
        if isinstance(image_or_list_or_dir_or_pattern, str):
            if os.path.isdir(image_or_list_or_dir_or_pattern):
                image_or_list_or_dir_or_pattern = os.path.join(image_or_list_or_dir_or_pattern, '*')
            raw_images = glob(image_or_list_or_dir_or_pattern)
        else:
            raw_images = image_or_list_or_dir_or_pattern

        images = []
        for fn in raw_images:
            if os.path.isfile(fn):
                images.append(fn)
            else:
                print(f'"{fn}" is not a file!')

        assert images, f'No images specified (empty list? pattern too specific?)'
        print(f'Redacting {len(images)} images...')

        if output_dir is None:
            output_dir = datetime.now().strftime('REDACTED_%Y-%m-%d_%H-%M-%S')
        os.makedirs(output_dir, exist_ok=True)

        for i, image_fn in enumerate(tqdm(images)):
            redacted = self.redact_image(image_fn)
            bn = os.path.basename(image_fn)
            out_bn = f'{i}_{bn}' if not redact_filename else str(i)
            cv2.imwrite(os.path.join(output_dir, out_bn), redacted)

        print('Done!')
        print(f'Output in dir "{output_dir}"')

    def redact_image(self, image, score_thresh=0.5):
        if isinstance(image, str):
            colour_image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = image[np.newaxis, ...]
            image = image.astype(float)/255.
            image = torch.tensor(image).float()
        else:
            colour_image = (image.cpu().numpy()*255).astype('uint8').squeeze()
            if len(colour_image.shape < 3):
                colour_image = cv2.cvtColour(colour_image, cv2.COLOR_GRAY2BGR)

        with torch.no_grad():
            out = self.model([image])[0]

        for score, bbox in zip(out['scores'], out['boxes']):
            if score >= score_thresh:
                x0, y0, x1, y1 = [int(v) for v in bbox]
                cv2.rectangle(colour_image, (x0, y0, x1-x0, y1-y0), 0, -1)

        return colour_image

    def annot_and_redact(self, fn: str, sz=None):
        rand_image, rand_target = self.ds.get_annotated(fn, sz)
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
