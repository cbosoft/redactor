from .trainer import Trainer
from .config import get_config, finalise
from .run import run_experiment, run_config
from .progress_bar import set_is_notebook

from warnings import filterwarnings

filterwarnings(action='ignore', message='Lazy modules are a new feature.*')
