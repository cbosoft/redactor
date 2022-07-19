import torch
import torchvision.transforms as transforms


class AddUniformNoise:

    def __init__(self, magnitude, offset=0.0):
        self.magnitude = magnitude
        self.offset = offset

    def __call__(self, x):
        noise = torch.rand(*x.shape)*self.magnitude + self.offset
        return x + noise


class AddGaussianNoise:

    def __init__(self, std, mean=0.0):
        self.std = std
        self.mean = mean

    def __call__(self, x):
        noise = torch.normal(torch.full_like(x, self.mean), self.std)
        return x + noise


class MultiplyUniformNoise:

    def __init__(self, magnitude, offset=1.0):
        self.magnitude = magnitude
        self.offset = offset

    def __call__(self, x):
        noise = torch.rand(*x.shape)*self.magnitude + self.offset
        return x*noise


class MultiplyGaussianNoise:

    def __init__(self, std, mean=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, x):
        noise = torch.normal(torch.full_like(x, self.mean), self.std)
        return x*noise


def build_augmentations(tfms_src):
    return [eval(tfm_src, dict(transforms=transforms,
                               AddUniformNoise=AddUniformNoise, AddGaussianNoise=AddGaussianNoise,
                               MultiplyUniformNoise=MultiplyUniformNoise, MultiplyGaussianNoise=MultiplyGaussianNoise))
            for tfm_src in tfms_src]
