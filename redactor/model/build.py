import random

import torch
import torch.nn as nn
import numpy as np

from ..config import CfgNode
from .faster_rcnn import get_faster_rcnn


def weight_init(m):
    try:
        nn.init.normal_(m.weight.data, 0.0, 0.1)
    except AttributeError:
        pass
    try:
        nn.init.normal_(m.bias.data, 0.0, 0.1)
    except AttributeError:
        pass


def build_model(config: CfgNode):
    torch.manual_seed(config.model.seed)
    random.seed(config.model.seed)
    np.random.seed(config.model.seed)

    model = get_faster_rcnn(config)
    if config.model.weights is not None:
        model.load_state_dict(torch.load(config.model.weights))
    elif config.model.use_custom_weight_init and not config.model.backbone.pretrained:
        model.apply(weight_init)

    return model
