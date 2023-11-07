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


"""Module that defines the class of classification experiments."""


from typing import Optional
import os
import random
import argparse

import rockdecorators

import envconfig
from rocklib import logger
import parmapper as pm
import rocklongfiles


class RockExperiment:
    """General class for defining the parameters of a experiment."""

    # Store the experiments that are defined with decorator
    # `@rockdecorators.register(experiments)` to keep information of valid
    # experiments.  Each experiment is the name of a module that uses this
    # class for getting the information of parameters for the experiment.
    experiments = []

    def __init__(self, kwargs):
        self._kwargs = kwargs

        self.name = kwargs['name']

        # Parameters that do not influence the experiment.  It means that if
        # those parameters change, the name of the partial storage of results
        # will not change.
        self.device = kwargs['device']
        self._override = kwargs['override']

        self._classes = kwargs['classes']
        self._partition = kwargs['partition']
        self._versions = kwargs['versions']
        self._patch_size_lst = kwargs['patch_size_lst']
        assert len(self._patch_size_lst) == 1, \
            'Using multiple patch size is not DEPRECATED (20210224).'
        self._stride = kwargs['stride']
        self._segmentation = kwargs['segmentation']
        self._model_name = kwargs['model_name']
        self._raw_dataset = kwargs['raw_dataset']
        self._augment_scale = kwargs['augment_scale']
        self._augment_rotations = kwargs['augment_rotations']
        self._num_epochs = kwargs['num_epochs']
        self._class_weight = kwargs['class_weight']
        self._loss_name = kwargs['loss_name']
        self.optimization_metric_lst = kwargs['optimization_metric_lst']
        self._optimizer = kwargs['optimizer']
        self._early_stopping = kwargs['early_stopping']
        self._ssrl = kwargs['ssrl']
        self._layer_name = kwargs['layer_name']
        self._use_pyradiomics = kwargs['use_pyradiomics']
        self._quant_classes_lst = kwargs['quant_classes_lst']
        self._augment_classification = kwargs['augment_classification']
        self._classifier_lst = kwargs['classifier_lst']
        self._topk_lst = kwargs['topk_lst']
        self._normalize = kwargs['normalize']
        self._result_type_lst = kwargs['result_type_lst']
        self._cm_norm_lst = kwargs['cm_norm_lst']
        self._partition_lst = kwargs['partition_lst']

        if not self._raw_dataset and self._augment_scale is not None:
            raise ValueError('Scaling can only be applied when employing raw '
                             'dataset, otherwise it must be None, as it will '
                             'have no effect.')

        if not self._raw_dataset and self._augment_rotations:
            raise ValueError('Rotations can only be applied when employing raw'
                             ' dataset, otherwise it must be False, as it will'
                             ' have no effect.')

        self._init_variables()
        self._experiment = None
        self._params = None
        self._shuffled = False

    def update(self, args):
        """`args` should be from argparse module."""
        assert isinstance(args, argparse.Namespace), type(args)

        experiment = self.experiment
        self.__init__({key: (self._kwargs[key]
                             if key not in args.__dict__
                             else args.__dict__[key])
                       for key in self._kwargs})
        self.experiment = experiment

    def __call__(self, func, mapf: Optional[bool] = None,
                 send_dict: bool = False):
        """Apply the given function to every set of parameters and return a
        list of the returned values.

        send_dict:
            When True, when providing an experiment, it will provide the
            dictionary with all parameters through the key 'experiment'.  It is
            aimed at a smooth transition to having this module providing solely
            the dictionary with all possible parameters instead of providing
            individual parameters.  Further then, each function to run a part
            of the experiment should receive this dictionary and use them as
            appropriate.

        mapf:
            The map function to use, e.g., the builtin `map` function or
            `parmap`.  When, `None` check if each experiment contains a device
            specified (a key 'device') for a different device.  In that case,
            use `parmap`, otherwise, use builtin `map`.

        """
        assert self._experiment, \
            'An experiment should be set before trying to run it.'

        if mapf is None:
            if len(self) > 1 and \
               all('device' in x for x in self) and \
               len(set(x['device'] for x in self)) == len(self):
                mapf = pm.parmap
            else:
                mapf = map

        def compose(func):
            """Generates a new function that accepts a dictionary as argument
            instead of decomposed argments.

            """
            def newfunc(dic):
                newdic = ({**dic, **{'experiment': dic}}
                          if send_dict
                          else dic)
                return func(**newdic)
            return newfunc

        logger.info(self)
        results = list(mapf(compose(func), self))
        return results

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        assert isinstance(value, str), type(value)
        self._name = value

    @property
    def device(self):
        ret = self._device
        if isinstance(ret, (int, str)):
            ret = [ret]
        return ret

    @device.setter
    def device(self, value):
        assert isinstance(value, (list, int, str)), type(value)
        assert isinstance(value, int) \
            or all(isinstance(x, int) for x in value) \
            or (isinstance(value, str) and value in ['cpu', 'auto']) \
            or all(x in ['cpu', 'auto'] for x in value), \
            value
        self._device = value

    @property
    def experiment(self):
        return self._experiment

    @experiment.setter
    def experiment(self, value):
        assert isinstance(value, str), type(value)
        assert value in self.experiments, \
            'Invalid experiment: \'{}\'.  Valid ones are in {}.'.format(
                value, self.experiments,
            )
        self._experiment = value
        self._params = None

    @property
    def optimization_metric_lst(self):
        return self._optimization_metric_lst

    @optimization_metric_lst.setter
    def optimization_metric_lst(self, value):
        assert isinstance(value, (list)), type(value)
        assert all(isinstance(x, str) for x in value), value
        assert all(x.isupper() and x.isalnum() for x in value), value
        self._optimization_metric_lst = value

    @property
    def params(self):
        """Initialize the parameters according to the set `experiment` when it
        is not already computed and available.

        """
        assert self._experiment, \
            'An experiment should be set before using the parameters.'

        if self._params is None:
            self._params = getattr(self, self.experiment)()

            assert isinstance(self._params, list), type(self._params)
            assert all(isinstance(x, dict)
                       for x in self._params), type(self._params[0])
            assert all(all(isinstance(x, str)
                           for x in parameters.keys())
                       for parameters in self._params)
            assert len(self._params) > 0, \
                'It should have parameters of at least one experiment'
            assert len(set(len(x) for x in self._params)) == 1, \
                'All experiments should have the same quantity of parameters'

        return self._params

    def _init_variables(self):
        """Initialize the attributes that can be automatically inferred from
        the dataset than should already be provided by the user.  It is to be
        used in the __init__ of a subclass after attributes declaration.

        It assumes the following variables:
            - self._classes
            - self._versions

        It generates the following varaibles:
            - self._classes_code
            - self._versions_code

        """
        assert isinstance(self._classes, list), type(self._classes)
        assert all(isinstance(x, int) for x in self._classes), self._classes
        assert isinstance(self._versions, list), type(self._versions)
        assert all(isinstance(x, int) for x in self._versions), self._versions
        # TODO: Perform extra assertions for the other variables that
        # must be defined by the subclass.

        patches_prefix = 'patches'
        shelve_extension = '.db'
        network_path_prefix = 'weights'
        network_path_extension = '.pth'
        features_prefix = 'features'
        pyradiomics_prefix = 'pyradiomics'
        key_prefix = 'key'
        keycm_prefix = 'keycm'

        def check_default(key, value, default=None):
            assert isinstance(key, str)
            assert len(key) == 2
            return (''
                    if value == default
                    else f'_{key}:{value}')

        def create_custom(key, value):
            """value: The name of the variable to `format` with later."""
            assert isinstance(key, str)
            assert len(key) == 2
            assert isinstance(value, str)
            return f'_{key}:{{{value}}}'

        self._classes_code = prime_code(self._classes)
        self._versions_code = prime_code(self._versions)
        self._parts = ['train', 'val', 'test']

        filenames_prefix_fmt = (
            check_default('na', self._name, None) +
            check_default('cc', self._classes_code, 0) +
            create_custom('pt', 'partition') +
            check_default('vc', self._versions_code, 0) +
            ''
        )

        filenames_core_fmt = (
            filenames_prefix_fmt +
            create_custom('ps', 'patch_size_str') +
            check_default('st', self._stride, 100) +
            check_default('se', self._segmentation, False) +
            ''
        )

        part_fmt = create_custom('pa', 'part')

        envconfig._makedir(envconfig.records_dir)
        self._patch_file_part_fmt = os.path.join(
            envconfig.records_dir,
            patches_prefix +
            filenames_core_fmt +
            part_fmt +
            shelve_extension +
            ''
        )

        # finetune
        self._network_path_core_fmt = (
            filenames_core_fmt
            .replace('{partition}', '{{partition}}')
            .format(
                patch_size_str=format_patch_size_lst(self._patch_size_lst)
            ) +
            check_default('mn', self._model_name, 'inception') +
            check_default('rd', self._raw_dataset, False) +
            check_default('as', self._augment_scale, None) +
            check_default('ar', self._augment_rotations, False) +
            check_default('ne', self._num_epochs, 200) +
            check_default('cw', self._class_weight, False) +
            check_default('lo', self._loss_name, 'CrossEntropyLoss') +
            check_default('op', self._optimizer, 'SGD') +
            check_default('es', self._early_stopping, 10) +
            check_default('ss',
                          (f'({self._ssrl.split("/")[-1]})'
                           if isinstance(self._ssrl, str)
                           else self._ssrl),
                          None) +
            create_custom('om', 'optimization_metric') +
            ''
        )

        envconfig._makedir(envconfig.checkpoints_dir)
        self._network_path_dict = {
            optimization_metric: os.path.join(
                envconfig.checkpoints_dir,
                network_path_prefix +
                self._network_path_core_fmt.format(
                    partition=self._partition,
                    optimization_metric=optimization_metric,
                ) +
                network_path_extension +
                ''
            )
            for optimization_metric in self.optimization_metric_lst
        }
        self._network_path_partial = os.path.join(
            envconfig.checkpoints_dir,
            network_path_prefix +
            self._network_path_core_fmt.format(
                partition=self._partition,
                optimization_metric=None,
            ) +
            '_P{point}' +
            network_path_extension +
            ''
        )

        def patch_file_part_get(part):
            return [
                self._patch_file_part_fmt.format(
                    partition=self._partition,
                    patch_size_str=format_patch_size(patch_size),
                    part=part,
                )
                for patch_size in self._patch_size_lst
            ]

        assert 'train' in self._parts
        assert 'val' in self._parts
        assert 'test' in self._parts  # for later
        self._patch_file_part_train_lst = patch_file_part_get('train')
        self._patch_file_part_val_lst = patch_file_part_get('val')

        # finetuned_extraction
        self._patch_file_part_lst = [
            self._patch_file_part_fmt.format(
                partition=self._partition,
                patch_size_str=format_patch_size(patch_size),
                part=part,
            )
            for patch_size in self._patch_size_lst
            for part in self._parts
        ]

        self._features_file_core_fmt = (
            filenames_core_fmt +
            check_default('np', f'({self._network_path_core_fmt})') +
            check_default('ln', self._layer_name, None) +
            ''
        )

        envconfig._makedir(envconfig.features_dir)
        self._features_file_part_fmt = os.path.join(
            envconfig.features_dir,
            features_prefix +
            self._features_file_core_fmt +
            part_fmt +
            shelve_extension +
            ''
        )

        # Should match the order of `self._patch_file_part_lst`.
        self._features_file_part_dict_lst = {
            optimization_metric: [
                self._features_file_part_fmt.format(
                    partition=self._partition,
                    patch_size_str=format_patch_size(patch_size),
                    part=part,
                    optimization_metric=optimization_metric,
                )
                for patch_size in self._patch_size_lst
                for part in self._parts
            ]
            for optimization_metric in self.optimization_metric_lst
        }

        # run_pyradiomics
        assert not self._use_pyradiomics or self._segmentation
        assert not self._use_pyradiomics or self._raw_dataset

        self._pyradiomics_file_core_fmt = (
            filenames_prefix_fmt.format(partition=self._partition) +
            ''
        )

        self._pyradiomics_file_part_fmt = os.path.join(
            envconfig.features_dir,
            pyradiomics_prefix +
            self._pyradiomics_file_core_fmt +
            part_fmt +
            shelve_extension +
            ''
        )

        self._pyradiomics_file_part_dict = {
            part: self._pyradiomics_file_part_fmt.format(part=part)
            for part in self._parts
        }

        self._pyradiomics_file_part_lst = [
            self._pyradiomics_file_part_dict[part]
            for part in self._parts
        ]

        # classification
        def features_file_part_get(part):
            return {
                optimization_metric: [
                    self._features_file_part_fmt.format(
                        partition=self._partition,
                        patch_size_str=format_patch_size(patch_size),
                        part=part,
                        optimization_metric=optimization_metric,
                    )
                    for patch_size in self._patch_size_lst
                ]
                for optimization_metric in self.optimization_metric_lst
            }

        self._features_file_part_train_dict_lst = \
            features_file_part_get('train')
        self._features_file_part_val_dict_lst = features_file_part_get('val')
        self._features_file_part_test_dict_lst = features_file_part_get('test')

        # When using `pyradiomics` features, the we will have the same key
        # format for every optimization metric.  As optimization metrics will
        # be ignored in this case, later we will simply use any one.
        self._key_core_dict_fmt = {
            optimization_metric: (
                (
                    self._features_file_core_fmt
                    .replace('{partition}', '{{partition}}')
                    .format(
                        patch_size_str=format_patch_size_lst(
                            self._patch_size_lst
                        ),
                        optimization_metric=optimization_metric,
                    )
                    if not self._use_pyradiomics else
                    self._pyradiomics_file_core_fmt
                ) +
                check_default('ac', self._augment_classification, True) +
                check_default('pr', self._use_pyradiomics, False) +
                check_default('no', self._normalize, False) +
                create_custom('qc', 'quant_classes') +
                create_custom('cl', 'classifier') +
                create_custom('to', 'topk') +
                ''
            )
            for optimization_metric in self.optimization_metric_lst
        }

        self._key_dict_fmt = {
            optimization_metric: (
                key_prefix +
                self._key_core_dict_fmt[optimization_metric] +
                ''
            )
            for optimization_metric in self.optimization_metric_lst
        }

        # plot_confusion_matrices
        self._keycm_dict_fmt = {
            optimization_metric: (
                keycm_prefix +
                self._key_core_dict_fmt[optimization_metric] +
                create_custom('rt', 'result_type') +
                create_custom('cn', 'cm_norm') +
                ''
            )
            for optimization_metric in self.optimization_metric_lst
        }

    def __len__(self):
        """Return the number of parameters of the experiments."""
        return len(self.params)

    def __str__(self):
        strings = [
            '<RockExperiment {} experiments {} parameters>'.format(
                len(self),
                self.quant_parameters(),
            )
            if self._experiment is not None else
            '<RockExperiment '
            '(setting obj.experiment = \'<experiment name>\' is required)>'
        ]
        return '\n'.join(strings)

    def __repr__(self):
        strings = [str(self)]
        if self._experiment is not None:
            for i, parameters in enumerate(self.parameters()):
                strings.append('Exp {}: {}'.format(i, parameters))
        return '\n'.join(strings)

    def __iter__(self):
        return iter(self.parameters())

    def quant_parameters(self):
        """Quantity of parameters for the experiment especifications."""
        return len(self.params[0])

    def shuffle(self):
        """Mark to shuffle the experiments order when providing it."""
        self._shuffled = True

    def parameters(self):
        """Return the content of the parameters of the experiment after
        validation of its correct definition.

        """
        params = self.params

        if self._shuffled:
            params = params.copy()
            random.shuffle(params)

        return params

    def expose_as_global_variables(self, globals_dict, idx=0,
                                   send_dict: bool = False,
                                   ):
        """Exposes the variables of a experiment as global variables.  It
        ignores the effect of shuffling.

        send_dict:
            When True, when providing an experiment, it will provide the
            dictionary with all parameters through the key 'experiment'.  It is
            aimed at a smooth transition to having this module providing solely
            the dictionary with all possible parameters instead of providing
            individual parameters.  Further then, each function to run a part
            of the experiment should receive this dictionary and use them as
            appropriate.

        """
        params = self.params[idx]
        for x in params:
            logger.info(f'Exposing: {x} = {params[x]}')
            globals_dict[x] = params[x]
            if send_dict:
                globals_dict['experiment'] = params

    @rockdecorators.register(experiments, lambda x: x.__name__)
    def generate_patches(self):
        return [
            {
                'classes': self._classes,
                'partition': self._partition,
                'versions': self._versions,
                'patch_size': patch_size,
                'stride': self._stride,
                'segmentation': self._segmentation,
                'parts': self._parts,
                'patch_file_part_lst': [
                    self._patch_file_part_fmt.format(
                        partition=self._partition,
                        patch_size_str=format_patch_size(patch_size),
                        part=part,
                    )
                    for part in self._parts
                ],
                'override': self._override,
            }
            for patch_size in self._patch_size_lst
        ]

    @rockdecorators.register(experiments, lambda x: x.__name__)
    def finetune(self):
        for filename in self._patch_file_part_train_lst:
            assert self._raw_dataset or os.path.exists(filename), filename
        for filename in self._patch_file_part_val_lst:
            assert self._raw_dataset or os.path.exists(filename), filename

        return [
            {
                'classes': self._classes,
                'partition': self._partition,
                'versions': self._versions,
                'model_name': self._model_name,
                'raw_dataset': self._raw_dataset,
                'scale': self._augment_scale,
                'augment_rotations': self._augment_rotations,
                'stride': self._stride,
                'segmentation': self._segmentation,
                'num_epochs': self._num_epochs,
                'class_weight': self._class_weight,
                'loss_name': self._loss_name,
                'optimizer': self._optimizer,
                'early_stopping': self._early_stopping,
                'ssrl': self._ssrl,
                'network_path_dict': self._network_path_dict,
                'network_path_partial': self._network_path_partial,
                'patch_file_part_train_lst': self._patch_file_part_train_lst,
                'patch_file_part_val_lst': self._patch_file_part_val_lst,
                'device': self.device[0],
                'override': self._override,
            }
        ]

    @rockdecorators.register(experiments, lambda x: x.__name__)
    def finetuned_extraction(self):
        assert all(
            os.path.exists(rocklongfiles.shortfile(
                self._network_path_dict[optimization_metric]))
            for optimization_metric in self._network_path_dict), \
            ('Not every network model being tried to be used actually exists. '
             ' The ones that do not exist are the following: {}.  The ones tha'
             't exists are the following {}.').format(
                 [self._network_path_dict[optimization_metric]
                  for optimization_metric in self._network_path_dict
                  if not os.path.exists(
                          self._network_path_dict[optimization_metric]
                  )],
                 [self._network_path_dict[optimization_metric]
                  for optimization_metric in self._network_path_dict
                  if os.path.exists(
                          self._network_path_dict[optimization_metric]
                  )],
             )
        ret = [
            {
                'classes': self._classes,
                'segmentation': self._segmentation,
                'stride': self._stride,
                'model_name': self._model_name,
                'raw_dataset': self._raw_dataset,
                'network_path': self._network_path_dict[optimization_metric],
                'layer_name': self._layer_name,
                'part': part,
                'patch_file_part': patch_file_part,
                'features_file_part': features_file_part,
                'device': self.device[0],
                'override': self._override,
            }
            for optimization_metric in self._features_file_part_dict_lst
            for part, patch_file_part, features_file_part in zip(
                    self._parts,
                    self._patch_file_part_lst,
                    self._features_file_part_dict_lst[optimization_metric],
            )
        ]

        # When the number of experiments is equal to the number of GPU devices,
        # run one experiment on each device.
        if len(ret) > 1 and len(ret) == len(self.device):
            for i in range(len(ret)):
                ret[i]['device'] = self.device[i]

        return ret

    @rockdecorators.register(experiments, lambda x: x.__name__)
    def run_pyradiomics(self):
        ret = [
            {
                'classes': self._classes,
                'versions': self._versions,
                'segmentation': self._segmentation,
                'raw_dataset': self._raw_dataset,
                'parts': self._parts,
                'pyradiomics_file_part_lst': self._pyradiomics_file_part_lst,
                'override': self._override,
            }
        ]

        return ret

    @rockdecorators.register(experiments, lambda x: x.__name__)
    def classification(self):
        return [
            {
                'key': self._key_dict_fmt[optimization_metric].format(
                    partition=self._partition,
                    quant_classes=quant_classes,
                    classifier=classifier,
                    topk=topk,
                ),
                'quant_classes': quant_classes,
                'classes': self._classes,
                'augment_classification': self._augment_classification,
                'classifier': classifier,
                'topk': topk,
                'normalize': self._normalize,
                'features_file_part_train_lst':
                self._features_file_part_train_dict_lst[optimization_metric],
                'features_file_part_val_lst':
                self._features_file_part_val_dict_lst[optimization_metric],
                'features_file_part_test_lst':
                self._features_file_part_test_dict_lst[optimization_metric],
                'override': self._override,
            }
            for quant_classes in self._quant_classes_lst
            for classifier in self._classifier_lst
            for topk in self._topk_lst
            for optimization_metric in self.optimization_metric_lst
        ] if not self._use_pyradiomics else [
            {
                'key': (self._key_dict_fmt[self.optimization_metric_lst[0]]
                        .format(
                            partition=self._partition,
                            quant_classes=quant_classes,
                            classifier=classifier,
                            topk=topk,
                        )),
                'quant_classes': quant_classes,
                'classes': self._classes,
                'augment_classification': self._augment_classification,
                'classifier': classifier,
                'topk': topk,
                'normalize': self._normalize,
                'features_file_part_train_lst': [
                    self._pyradiomics_file_part_dict['train'],
                ],
                'features_file_part_val_lst': [
                    self._pyradiomics_file_part_dict['val'],
                ],
                'features_file_part_test_lst': [
                    self._pyradiomics_file_part_dict['test'],
                ],
                'override': self._override,
            }
            for quant_classes in self._quant_classes_lst
            for classifier in self._classifier_lst
            for topk in self._topk_lst
        ]

    @rockdecorators.register(experiments, lambda x: x.__name__)
    def plot_confusion_matrices(self):
        return [
            {
                'key': self._key_dict_fmt[optimization_metric].format(
                    partition=self._partition,
                    quant_classes=quant_classes,
                    classifier=classifier,
                    topk=topk,
                ),
                'quant_classes': quant_classes,
                'classes': self._classes,
                'result_type': result_type,
                'cm_norm': cm_norm,
                'keycm': self._keycm_dict_fmt[optimization_metric].format(
                    partition=self._partition,
                    quant_classes=quant_classes,
                    classifier=classifier,
                    topk=topk,
                    result_type=result_type,
                    cm_norm=cm_norm,
                ),
                'override': self._override,
                'model_name': self._model_name,
                'partition': self._partition,
            }
            for quant_classes in self._quant_classes_lst
            for classifier in self._classifier_lst
            for topk in self._topk_lst
            for result_type in self._result_type_lst
            for cm_norm in self._cm_norm_lst
            for optimization_metric in self.optimization_metric_lst
        ]

    @rockdecorators.register(experiments, lambda x: x.__name__)
    def gen_table_of_results(self):
        return [
            {
                'key_dict_fmt': self._key_dict_fmt,
                'quant_classes': quant_classes,
                'topk': topk,
                'result_type_lst': self._result_type_lst,
                'classifier_lst': self._classifier_lst,
                'partition_lst': self._partition_lst,
                'optimization_metric_lst': self.optimization_metric_lst,
            }
            for quant_classes in self._quant_classes_lst
            for topk in self._topk_lst
        ]


def primes():
    def isprime(x):
        for y in range(2, x):
            if x % y == 0:
                return False
        return True
    x = 2
    yield x
    while True:
        x += 1
        if isprime(x):
            yield x


def prime_code(values):
    assert isinstance(values, list), type(values)
    assert all(isinstance(x, int) for x in values), values
    assert len(values) == len(set(values)), values
    assert all(x >= 0 for x in values), values
    return sum([(x+1) * y for x, y in zip(sorted(values), primes())])


def format_patch_size(patch_size):
    return f'{patch_size[0]}x{patch_size[1]}'


def format_patch_size_lst(patch_size_lst):
    return ','.join(map(format_patch_size, patch_size_lst))
