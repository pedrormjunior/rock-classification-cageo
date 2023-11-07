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


"""Modules that defines what to experiment with when running the code."""

import rockexperiment


classes = [3, 5, 6, 1, 11, 9, 10]

Experiment = rockexperiment.RockExperiment(
    {
        'name': (
            '20230821'
        ),
        'device': 'auto',
        'override': False,

        # GENERATE_PATCHES
        'classes': classes,
        'partition': 0,
        'versions': [1, 2, 3],
        'patch_size_lst': [
            (224, 224),
        ],
        'stride': 100,

        # 'segmentation': False,
        'segmentation': True,

        # FINAL:
        # 'model_name': "regnet",
        # 'model_name': "vit",
        'model_name': "convnext",
        # 'model_name': "vgg",
        # 'model_name': "wide_resnet",
        # 'model_name': "efficientnet_v2",
        # 'model_name': "swin",
        # 'model_name': "resnext",
        # 'model_name': "efficientnet",
        # 'model_name': "resnet",
        # 'model_name': "alexnet",
        # 'model_name': "densenet",
        # 'model_name': "googlenet",
        # 'model_name': "shufflenet_v2",
        # 'model_name': "mnasnet",
        # 'model_name': "mobilenet_v3",
        # 'model_name': "mobilenet_v2",
        # 'model_name': "squeezenet",
        # 'model_name': "inception_v3",
        # 'model_name': 'fusioncnns',

        # 'raw_dataset': False,
        'raw_dataset': True,

        'augment_scale': None,
        # 'augment_scale': 0.2,

        'augment_rotations': False,
        # 'augment_rotations': True,

        'feature_extract': True,
        # 'feature_extract': False,

        # 'num_epochs': 200,
        'num_epochs': 1000000,

        # 'class_weight': False,
        'class_weight': True,

        'loss_name': 'CrossEntropyLoss',

        'optimization_metric_lst': [
            'LOSS',
            'ACC',
            'NA',
        ],

        'optimizer': (      # Parameters should be modified in ../rockparams.py
            'SGD'
        ),

        'early_stopping': (
            # 10
            100
        ),

        'ssrl': (
            None
        ),

        # FINETUNED_EXTRACTION
        'layer_name': None,
        # 'layer_name': 'flatten',

        # RUN_PYRADIOMICS
        # 'use_pyradiomics': True,
        'use_pyradiomics': False,

        # CLASSIFICATION
        'quant_classes_lst': [len(classes)],

        # 'augment_classification': True,
        'augment_classification': False,

        'classifier_lst': [
            '3nn',
            'rf',
            'svm',
            'softmax',
        ],

        'topk_lst': range(1, 4),

        'normalize': False,
        # 'normalize': True,

        # PLOT_CONFUSION_MATRICES
        'result_type_lst': [
            'img',
            'pat',
        ],
        'cm_norm_lst': [True],

        # GEN_TABLE_OF_RESULTS
        'partition_lst': list(range(8)),
    }
)
