# Copyright (c) 2019-2023
# Shell-ML Project
# Pedro Ribeiro Mendes Júnior <pedrormjunior@gmail.com> et al.
# Artificial Intelligence Lab. Recod.ai
# Institute of Computing (IC)
# University of Campinas (Unicamp)
# Campinas, São Paulo, Brazil
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import logging
import logging.handlers
from datetime import datetime


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

try:
    import colored_traceback.auto
except ImportError:
    pass

try:
    import coloredlogs
    coloredlogs.DEFAULT_FIELD_STYLES['levelname']['color'] = 'blue'
    coloredlogs.DEFAULT_FIELD_STYLES['name']['color'] = 'white'
    coloredlogs.DEFAULT_FIELD_STYLES['asctime']['color'] = 'blue'
    coloredlogs.DEFAULT_LEVEL_STYLES['debug']['color'] = 'yellow'
    coloredlogs.DEFAULT_LEVEL_STYLES['warning']['color'] = 'magenta'
    coloredlogs.DEFAULT_LEVEL_STYLES['info']['color'] = 'green'
except ImportError:
    class NoColoredLogs:
        def install(self,
                    logger=None, level=None, fmt=None,
                    *args, **kwargs):
            if logger is not None:
                logger.setLevel(level)
                ch = logging.StreamHandler()
                ch.setLevel(level)
                formatter = logging.Formatter(fmt)
                ch.setFormatter(formatter)
                logger.addHandler(ch)
    coloredlogs = NoColoredLogs()


LOGGERFMT = '%(asctime)s:%(levelname)s:%(message)s'

DATA_PREFIX = './data/'
records_dir = 'records'
logging_dir = 'logs'
checkpoints_dir = 'checkpoints'
features_dir = 'features'
plots_dir = 'plots'
results_dir = 'results'
results_filename = os.path.join(results_dir, 'classification_results')
predicts_filename = os.path.join(results_dir, 'classification_predicts')
partitions_filename = os.path.join(records_dir, 'partitions')
# log_dir = 'tensorboard'
# imgs_dir = 'imgs'
# imgs_cm_dir = os.path.join(imgs_dir, 'cm')  # confusion matrices


def set_visible_devices(devices):
    assert isinstance(devices, tuple), type(devices)
    CUDA_VISIBLE_DEVICES = (str(devices)
                            .replace(',)', ')')
                            .replace(', ', ','))[1:-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES


def loggerGenFileHandler(logger):
    _makedir(logging_dir)
    filename = datetime.now().strftime(
        os.path.join(logging_dir, "%Y-%m-%d--%H-%M-%S-%f.log"))
    handler = logging.handlers.RotatingFileHandler(filename)
    formatter = logging.Formatter(LOGGERFMT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.debug(f'Log file registered: {filename}')
    return filename


def _makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# _makedir(checkpoints_dir)
# _makedir(log_dir)
# _makedir(records_dir)
# _makedir(features_dir)
# _makedir(logging_dir)


def config(gpu_device=-1, tf_log_level=2):
    assert isinstance(gpu_device, (int, list, str)), type(gpu_device)
    assert not isinstance(gpu_device, str) or gpu_device == 'all'
    if isinstance(gpu_device, list):
        assert len(list(set(map(type, gpu_device)))) == 1, gpu_device
        assert isinstance(gpu_device[0], int), gpu_device
    assert isinstance(tf_log_level, int), type(tf_log_level)
    assert tf_log_level in [0, 1, 2, 3], tf_log_level  # https://stackoverflow.com/a/42121886/968131

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    if not (isinstance(gpu_device, str) and gpu_device == 'all'):
        os.environ['CUDA_VISIBLE_DEVICES'] = \
            (str(gpu_device)
             if isinstance(gpu_device, int)
             else ','.join(map(str, gpu_device)))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_log_level)


def tf_suppress_warnings(tf, suppress=True):
    if suppress:

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    else:
        # FIXME: Add the equivalent for avoiding suppress of warnings.
        pass


def tf_set_memory_growth(tf, enable=True):
    # https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, enable)
        except Exception:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
