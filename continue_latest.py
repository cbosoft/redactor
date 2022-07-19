from glob import glob
import os

from redactor.run import continue_training


if __name__ == '__main__':
    results = glob('training_results/*')
    latest_results = sorted(results, key=lambda fn: os.path.getctime(fn))[-1]

    continue_training(latest_results)
