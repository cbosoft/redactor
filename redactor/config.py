import random

from yacs.config import CfgNode
import torch
import numpy as np

from .util import ensure_dir, today


def get_config() -> CfgNode:
    config = CfgNode()

    config.action = 'train'  # can be 'train' or 'kfold'

    config.model = CfgNode()
    config.model.dropout_frac = 0.0
    config.model.seed = None
    config.model.use_custom_weight_init = True
    config.model.weights = None  # Optional[str], optionally load weights at state file specified

    config.model.n_classes = 2  # background + object, always one more than you think!
    config.model.backbone = CfgNode()
    config.model.backbone.kind = 'resnet'
    config.model.backbone.pretrained = True

    # Optional string (python code) defining the layers to be returned by the backbone. Should eval to list of int.
    # e.g. 'list(range(1, 5))' -> [1, 2, 3, 4]
    config.model.backbone.returned_layers = None

    # How many layers should be trainable? The rest are frozen.
    config.model.backbone.trainable_layers = 2

    # ResNet specific backbone settings
    config.model.backbone.resnet = CfgNode()
    config.model.backbone.resnet.n = 18

    config.data = CfgNode()
    config.data.root = 'data'
    config.data.pattern = None
    config.data.frac_train = 0.8

    config.data.images = CfgNode()
    config.data.images.size = 256  # Size of images passed to model. Images are resizes to S by S square where S is this value.
    config.data.images.augmentations = [
        # 'transforms.RandomVerticalFlip()',
    ]
    config.data.images.stack = 5
    config.data.images.lazy_load = False  # whether to keep all images in memory during training or to 'lazy load' them on the fly

    # items related to generating new data.
    config.data.gen = CfgNode()

    # How to decide colour of generated text?
    # - 'grayscale': random grayscale foreground
    # - 'black_or_white_fg': foreground colour is either black or white
    # - 'black_fg', 'white_fg': foreground colour is black or white respectively
    # - 'image_inverse': fg is opposite of (255 minus) mean of image
    # background is always 255 - foreground, or not present
    config.data.gen.colour_method = 'grayscale'

    # Chance that the added text has a solid background
    config.data.gen.bg_chance = 0.5

    config.training = CfgNode()
    config.training.n_epochs = 1_000

    config.training.plot_every = 100
    config.training.checkpoint_every = 100
    config.training.validate_every = 1
    config.training.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config.training.batch_size = 10
    config.training.shuffle_every_epoch = True
    config.training.metrics = [
        'm.CosineSimilarity()',
        'm.RMSE()',
        'm.PearsonCorrcoef()',
        'm.R2Score()',
        # 'm.SpearmanCorrcoef()'
    ]

    config.training.opt = CfgNode()
    config.training.opt.kind = 'Adam'  # ('Adam', 'SGD')

    # Adam optimiser. See pytorch docs for param info:
    # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    config.training.opt.adam = CfgNode()
    config.training.opt.adam.lr = 1e-3
    config.training.opt.adam.betas = (0.9, 0.999)
    config.training.opt.adam.weight_decay = 0.0
    config.training.opt.adam.amsgrad = False

    # SGD optimiser. See pytorch docs for param info:
    # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
    config.training.opt.sgd = CfgNode()
    config.training.opt.sgd.lr = 1e-3
    config.training.opt.sgd.momentum = 0.0
    config.training.opt.sgd.dampening = 0.0
    config.training.opt.sgd.weight_decay = 0.0
    config.training.opt.sgd.nesterov = False
    config.training.opt.sgd.maximize = False

    config.training.sched = CfgNode()
    config.training.sched.kind = 'OneCycle'  # ('OneCycle', 'Linear', 'Step', 'none')

    # OneCycle LR scheduler. See pytorch docs for param info:
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR
    config.training.sched.onecycle = CfgNode()
    config.training.sched.onecycle.max_lr = 0.01
    config.training.sched.onecycle.pct_start = 0.3
    config.training.sched.onecycle.anneal_strategy = 'cos'
    config.training.sched.onecycle.cycle_momentum = True
    config.training.sched.onecycle.base_momentum = 0.85
    config.training.sched.onecycle.max_momentum = 0.95
    config.training.sched.onecycle.div_factor = 25.0
    config.training.sched.onecycle.final_div_factor = 10000.0
    config.training.sched.onecycle.three_phase = False

    # Linear LR scheduler. See pytorch docs for more info:
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR
    config.training.sched.linear = CfgNode()
    config.training.sched.linear.start_factor = 1./3.
    config.training.sched.linear.end_factor = 1.0
    config.training.sched.linear.total_iters = None  # set to value to only decay for that number of steps (batches)

    # Step LR scheduler. See pytorch docs for more info:
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
    config.training.sched.step = CfgNode()
    config.training.sched.step.step_size = 100
    config.training.sched.step.gamma = 0.1

    # Step LR scheduler. See pytorch docs for more info:
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html
    config.training.sched.exponential = CfgNode()
    config.training.sched.exponential.gamma = 0.1

    config.kfold = CfgNode()
    config.kfold.n_folds = 10

    config.output_dir = 'training_results/{grouping}{expname}{architecture}_{today}'
    config.group = ''
    config.expname = ''
    config.debug_mode = False
    return config


def finalise(config: CfgNode):
    if config.group:
        if config.group[-1] != '/':
            config.group = config.group + '/'

    _today = today()

    config.output_dir = ensure_dir(config.output_dir.format(
        today=_today, architecture=f'FasterRCNN-{config.model.backbone.kind}',
        grouping=config.group.format(today=_today),
        expname=config.expname
    ))

    assert config.data.pattern is not None, 'config.data.pattern must be specified.'
    assert isinstance(config.data.pattern, (str, list)), 'config.data.pattern must be str or list.'
    if isinstance(config.data.pattern, str):
        config.data.pattern = [config.data.pattern]

    assert config.model.dropout_frac < 1.0, 'config.model.dropout_frac must be less than one. In addition, it must be greater than zero to actually enable dropout.'

    if config.model.seed is None:
        config.model.seed = random.randint(0, 1_000_000)

    if config.model.weights is not None:
        assert isinstance(config.model.weights, str)

    assert config.data.images.stack > 0, 'config.data.images.stack_size must be greater than zero'

    if config.expname == '':
        config.expname = 'anon-experiment'

    random.seed(config.model.seed)
    np.random.seed(config.model.seed)
    torch.manual_seed(config.model.seed)

    config.freeze()
    return config
