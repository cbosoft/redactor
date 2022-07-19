import os.path
import warnings

from .config import get_config, finalise, CfgNode
from .trainer import Trainer
from .kfold import KFoldCrossValidation
from .progress_bar import progressbar


def run_experiment(config_filename: str, group=None, dry_run=False, exp_prefix=''):
    print(f'Running experiment "{config_filename}"')
    config = get_config()
    config.merge_from_file(config_filename)

    if not config.expname:
        config.expname = exp_prefix + os.path.splitext(os.path.basename(config_filename))[0] + '_'

    run_config(config, group=group, dry_run=dry_run)


def run_config(config: CfgNode, dry_run=False, group=None):

    if group:
        config.group = group

    finalise(config)

    if not dry_run:
        if config.action == 'train':
            with Trainer(config) as trainer:
                trainer.train()
        elif config.action == 'kfold':
            xvalidator = KFoldCrossValidation(config)
            xvalidator.cross_validate()
    else:
        warnings.warn(message='Dry run: not running action.')


def continue_training(results_dir: str):
    config_fn = f'{results_dir}/config.yaml'
    print(f'Continuing experiment "{config_fn}"')
    config = get_config()
    config.merge_from_file(config_fn)
    config.model.weights = f'{config.output_dir}/model_state_final.pth'
    finalise(config)

    assert config.action == 'train', 'Can only continue training, other actions not allowed.'

    with Trainer(config) as trainer:
        trainer.train()


def run_many(config_filenames, group, **kwargs):
    for fn in progressbar(config_filenames, unit='experiments'):
        run_experiment(fn, group, **kwargs)
