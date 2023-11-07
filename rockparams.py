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


import torch.optim as optim
from torchvision import transforms


optimizer_params = {
    # Default parameters in PyTorch 1.11.0 implementation
    'Adadelta': {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6, 'weight_decay': 0, },
    'Adagrad': {'lr': 0.01, 'lr_decay': 0, 'weight_decay': 0,
                'initial_accumulator_value': 0, 'eps': 1e-10, },
    'Adam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8,
             'weight_decay': 0, 'amsgrad': False, 'maximize': False, },
    'AdamW': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8,
              'weight_decay': 0.01, 'amsgrad': False, 'maximize': False, },
    'SparseAdam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, },
    'Adamax': {'lr': 0.002, 'betas': (0.9, 0.999), 'eps': 1e-8,
               'weight_decay': 0, },
    'ASGD': {'lr': 0.01, 'lambd': 0.0001, 'alpha': 0.75, 't0': 1000000.0,
             'weight_decay': 0, },
    'LBFGS': {'lr': 1.0, 'max_iter': 20, 'max_eval': None,
              'tolerance_grad': 1e-07, 'tolerance_change': 1e-09,
              'history_size': 100, 'line_search_fn': None, },
    'NAdam': {'lr': 0.002, 'betas': (0.9, 0.999), 'eps': 1e-8,
              'weight_decay': 0, 'momentum_decay': 0.004, },
    'RAdam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8,
              'weight_decay': 0, },
    'RMSprop': {'lr': 0.01, 'alpha': 0.99, 'eps': 1e-8, 'weight_decay': 0,
                'momentum': 0, 'centered': False, },
    'Rprop': {'lr': 0.01, 'etas': (0.5, 1.2), 'step_sizes': (1e-06, 50), },
    'SGD': {'lr': 1e-4, 'momentum': 0, 'dampening': 0, 'weight_decay': 0,
            'nesterov': False, 'maximize': False, },
}

optimizer_dict = {
    'Adadelta': optim.Adadelta,
    'Adagrad': optim.Adagrad,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
    'SparseAdam': optim.SparseAdam,
    'Adamax': optim.Adamax,
    'ASGD': optim.ASGD,
    'LBFGS': optim.LBFGS,
    'NAdam': optim.NAdam,
    'RAdam': optim.RAdam,
    'RMSprop': optim.RMSprop,
    'Rprop': optim.Rprop,
    'SGD': optim.SGD,
}


def compose_transforms(lst):
    return transforms.Compose([x for x in lst if x is not None])


normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

data_transforms_patch_based = {
    'train': compose_transforms([
        # transforms.Resize(input_size),
        # RotationTransform(),
        # transforms.ColorJitter(
        #     brightness=0.15,
        #     contrast=0.1,
        #     saturation=0.1,
        #     hue=0.1,
        # ),
        # NoiseTransform(),
        normalize,
    ]),
    'val': compose_transforms([
        # transforms.Resize(input_size),
        normalize,
    ]),
}
