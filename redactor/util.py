from datetime import datetime
from pathlib import Path
from typing import Tuple
from glob import glob
import os

import numpy as np
import torch
import cv2


def get_latest_model() -> str:
    results = glob('training_results/*')
    latest_results = sorted(results, key=lambda fn: os.path.getctime(fn))[-1]
    return latest_results


def imread(fn: str, size: Tuple[int, int] = None) -> torch.Tensor:
    im: np.ndarray = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(f'Error reading image "{fn}"!')
    if size is not None:
        im = cv2.resize(im, size)
    return torch.tensor(im)


def today():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def ensure_dir(dirname: str) -> str:
    Path(dirname).mkdir(parents=True, exist_ok=True)
    return dirname


def gaussian_pdf(x, mu, std):
    pdf = 1./std/np.sqrt(2.*np.pi)*np.exp(-0.5*np.square((x - mu)/std))
    # not always normalised properly? binning issue?
    pdf /= np.trapz(pdf, x)
    return pdf


def geometric_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a*b)**.5


def geometric_midp(v: np.ndarray) -> np.ndarray:
    return geometric_mean(v[1:], v[:-1])


def to_float(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)


def onehot(v, n):
    rv = torch.zeros(n)
    rv[v] = 1
    return rv

