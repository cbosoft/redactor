from .annotated_image_dataset import AutoAnnotatedImagesDataset


def build_loaders(cfg):
    return AutoAnnotatedImagesDataset.build_loaders(cfg)


def build_dataset(cfg):
    return AutoAnnotatedImagesDataset.from_config(cfg)
