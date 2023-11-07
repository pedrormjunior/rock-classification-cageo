#!/usr/bin/env python3


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


from typing import Union, List, Dict, Optional
import itertools as it
import argparse

import os
import math
import random
import numpy as np

import time
import copy

import rocklib
from rocklib import logger
import rockparams
import rocklongfiles

import experiments.main as rockexperiment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import envconfig
import sklearn
import sklearn.metrics
import collections

import GPUtil


# DETERMINISM
# https://clay-atlas.com/us/blog/2021/08/24/pytorch-en-set-seed-reproduce/
SEED = 1251
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(mode=True)


BATCH_SIZE = (
    # 1024
    # 512
    256                         # for most of the networks
    # 1                           # for `fusioncnns`
    # 128
)
FREEZING_POINTS = (
    None
    # [290, 263, 236, 218, 180, 150, 120, 90, 78, 57, 36, 15, 12, 9, 6, 3, 0, ]
    # [290, ]                     # Inception-V3 last layer
    # [10, ]                      # SimpleCNN last layer
    # [60, ]                     # ResNet last layer
    # [50, ]                     # Squeezenet last layer
    # [362, ]                    # Densenet last layer
    # [159, ]                    # ResNet-50 (BT) last layer
    # [1852, ]                    # FusionCNNs last layers
    # [159, ]                    # ResNeXt last layer
    # [36, ]                    # VGG last layer
    # [150, ]                     # vit_b_16, vit_b_32 last layer
    # [294, ]                     # vit_l_16, vit_l_32 last layer
    # [390, ]                     # vit_h_14 last layer
    # [0, ]
    ### Final:
    # [366, ]                     # regnet
    # [390, ]                     # vit
    # [342, ]                     # convnext
    # [68, ]                      # vgg
    # [312, ]                     # wide_resnet
    # [895, ]                     # efficientnet_v2
    # [327, ]                     # swin
    # [312, ]                     # resnext
    # [709, ]                     # efficientnet
    # [465, ]                     # resnet
    # [14, ]                      # alexnet
    # [482, ]                     # densenet
    # [171, ]                     # googlenet
    # [168, ]                     # shufflenet_v2
    # [156, ]                     # mnasnet
    # [172, ]                     # mobilenet_v3
    # [156, ]                     # mobilenet_v2
    # [50, ]                      # squeezenet
    # [290, ]                     # inception_v3
    # [5591, ]                    # fusioncnns (final)
)
LEARNING_RATES = (
    None
    # [1e-4, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, ]
    # [1e-4, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, ]
    # [1e-4, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, ]
    # [1e-3, 1e-4, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, ]
    # [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, ]
    # [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, ]
)
LOAD_PARTIAL = False

NUM_WORKERS = 10
"""Number of parallel processes/workers for loading the dataset."""

RAW_DATASET_TIMES = 74          # 73.53180433065049
"""Number of times to repeat the training dataset per training epoch.  It only
takes effect when the experiment is to run on the raw dataset.  The value 74 is
estimated to make the raw dataset equivalent to the patch-based one extracted
with stride 101 in terms of the total number of patches.  This value were
estimated based solely on the first 8 partitions, as of July 6, 2022.

"""


def subsample_patches(inputses):
    """Given a sequence of tensors, it is assumed they semantically refer to a
    tensor of patches extracted from the same image.  As `x.size(0)` is
    potentially different, for `x` as each of the given tensors, it returns a
    list of corresponding tensors `y` whose `y.size(0)` are all equal to the
    minimum value for `x.size(0)`.  Furthermore, to avoid bias towards getting
    patches preferably from the edges, the selected patches are randomly
    selected.

    """
    num_patches = min([x.size(0) for x in inputses])

    def get_patches_shuffled(tensor):
        """Return maximum `num_patches` patches randomly selected from
        `tensor`.

        """
        indexes = torch.randperm(tensor.size(0))
        indexes = indexes[:num_patches]
        return tensor[indexes]

    return [get_patches_shuffled(x) for x in inputses]


def train_model_with_freezing_points(
        model,
        freezing_points,
        optimizer_func,
        optimizer_params,
        ssrl: Optional[int],
        model_output_partial=None,
        **kwargs,
):
    """Train the given `model` by setting multiple freezing points, as in
    `freezing_points`.

    model:
        The model to be trained.

    freezing_points:
        A list of points in which to freeze the network before and unfreeze
        after them.  If None, then only the last layer of the given network
        will be unfrozen.

    optimizer_func:
        Function to create an optimizer, given the parameters and the key-based
        arguments.

    optimizer_params:
        A dictionary containing the parameters to be employed along with the
        given `optimizer_func`.

    ssrl:
        Should be `None` when it is a normal network, otherwise an integer
        greater than 0 specifying the size of the occluded patch for
        Self-Supervised Representation Learning (SSRL) with Masked Prediction
        as pretext task.

    model_output_partial:
        A string containing the formatting option '{point}'.  A partial
        computation will be loaded in case it is existent.  If None or not
        specified, it will not try to load partial computation possibly saved
        after the training associated with each freezing point and will not
        save the partial computation.

    **kwargs:
        Should contain the remaining parameters to be passed to `train_model`
        function.  The only necessary parameters to provide are: 'dataloaders',
        'criterion', 'optimizer', 'num_epochs', 'device', 'early_stopping'.

    """
    assert isinstance(freezing_points, (list, type(None))), \
        type(freezing_points)
    if isinstance(freezing_points, type(None)):
        freezing_points = [len(list(model.parameters())) - 2]
    assert not model_output_partial or isinstance(model_output_partial, str), \
        type(model_output_partial)
    assert isinstance(optimizer_params, (dict, list)), type(optimizer_params)
    assert isinstance(optimizer_params, dict) or \
        len(optimizer_params) == len(freezing_points), len(optimizer_params)
    assert 'dataloaders' in kwargs, kwargs.keys()
    assert 'criterion' in kwargs, kwargs.keys()
    assert 'num_epochs' in kwargs, kwargs.keys()
    assert 'device' in kwargs, kwargs.keys()
    assert 'early_stopping' in kwargs, kwargs.keys()
    assert len(kwargs) == 5, kwargs.keys()

    logger.info('finetune.train_model_with_freezing_points(): '
                f'optimizer_params: {optimizer_params}')

    history = []
    best_metric = None
    model_weights = None
    for counter, point in enumerate(freezing_points):
        if model_output_partial and \
           os.path.exists(model_output_partial.format(point=point)):
            logger.info(f'Loading partial model for point {point}.')
            model_weights = torch.load(
                model_output_partial.format(point=point),
                map_location=kwargs['device'],
            )
            model.load_state_dict(model_weights['LOSS'])
            continue
        elif counter > 0:
            model.load_state_dict(model_weights['LOSS'])

        if str(model)[:10] == 'Inception3':
            params_to_update = unfreeze_InceptionV3(model, point)
        else:
            params_to_update = unfreeze(model, point)

        optimizer = optimizer_func(
            params_to_update,
            **(optimizer_params
               if isinstance(optimizer_params, dict)
               else optimizer_params[counter]),
        )

        _, model_weights, best_metric, hist = train_model(
            model,
            kwargs['dataloaders'],
            kwargs['criterion'],
            optimizer,
            ssrl=ssrl,
            num_epochs=kwargs['num_epochs'],
            device=kwargs['device'],
            early_stopping=kwargs['early_stopping'],
            best_metric=best_metric,
            model_weights=model_weights,
        )

        history += hist

        if model_output_partial:
            torch.save(model_weights, model_output_partial.format(point=point))

    return model, model_weights, best_metric, history


def train_model(model, dataloaders, criterion, optimizer,
                ssrl: Optional[int],
                num_epochs: int,
                device: str,
                early_stopping: Optional[int] = None,
                best_metric: Optional[Dict[str, str]] = None,
                model_weights:
                Optional[Dict[str, collections.OrderedDict]] = None,
                ):
    """Train a `model` for the data in `dataloaders` based on the loss function
    `criterion` using the optimizer `optimizer` for a total of `num_epochs`.
    The model as well as the data is loaded on device `device`.

    model:
        The model to be trained upon.

    dataloaders:
        A dictionary that must contain the keys 'train' and 'val' such that
        each of them map to a `torch.utils.data.DataLoader` instance.

    criterion:
        The loss to be employed for training.

    optimizer:
        The optimizer to be employed for training.

    ssrl:
       Should be `None` when it is a normal network, otherwise an integer
       greater than 0 specifying the size of the occluded patch for
       Self-Supervised Representation Learning (SSRL) with Masked Prediction as
       pretext task.

    num_epochs:
        The number of epochs to train the network for.

    device:
        The device on which to put the features and the labels for training.
        Probably the same davice in which the network was sent to.  See
        https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
        for more information.

    early_stopping:
        If None, no early stopping is performed.  Otherwise, it should be an
        integer specifying the maximum number of epochs to tolerate without
        improvement on the model for every metric.

    best_metric:
        If provided, this function will ensure the obtained models perform
        better for those metrics, otherwise the best returned models (for each
        metric) will be the same input model.

    model_weights:
        If provided, it should be a dictionary associating each employed metric
        to the weights of the network, in the same format as the dictionary
        returned by this function itself.  The respective weights will be
        returned in case this function cannot optimize it more for a certain
        metric.

    Return
    ------

    A tuple containing the model, a dictionary with model weights optimized for
    each of the metrics, and the history of the training.

    """
    logger.info(f'finetune.train_model(): DEVICE: {device}')
    logger.debug(f'finetune.train_model(): criterion: {criterion}')
    logger.debug('finetune.train_model(): optimizer: {}'.format(
        str(optimizer).replace('\n', '')
    ))
    logger.debug(f'finetune.train_model(): early_stopping: {early_stopping}')
    since = time.time()

    # Verifications.
    is_barlowtwins = str(model).startswith('BarlowTwins(')
    is_inception = ((str(model).startswith('Inception3(')
                     or str(model).startswith(
                         'DataParallel(\n  (module): Inception3('
                     ))
                    and str(model).endswith(')'))
    is_triplet = 'TripletMargin' in str(criterion)
    is_triplet_cross_entropy = ('TripletMarginWithCrossEntropy'
                                in str(criterion))
    is_contrastive = (str(criterion).startswith('ContrastiveLoss(') and
                      str(criterion).endswith(')'))

    best_metric = (
        {
            'LOSS': float('-inf'),
            'ACC': float('-inf'),
            'NA': float('-inf'),
        } if not isinstance(ssrl, int) and not is_barlowtwins else {
            'LOSS': float('-inf'),
        }
    ) if best_metric is None else best_metric
    assert 'LOSS' in best_metric
    best_epoch = {metric: -1 for metric in best_metric}
    model_weights = model_weights if model_weights else \
        {metric: copy.deepcopy(model.state_dict())
         for metric in best_metric}
    history = []

    stop_early = False
    for epoch in range(num_epochs):
        logger.info('Epoch: {}/{}, Best epoch: {}'.format(
            epoch, num_epochs - 1, best_epoch,
        ))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_counter = 0
            maxdim = 1
            running_cm = sklearn.metrics.confusion_matrix([], [],  # initialize
                                                          labels=range(maxdim))

            # Iterate over data.
            for inputs_labels in envconfig.tqdm(dataloaders[phase]):
                if is_triplet:
                    inputs, inputs_pos, inputs_neg, labels = inputs_labels
                elif is_contrastive:
                    inputs, inputs_other, pairlabels, labels = inputs_labels
                else:
                    inputs, labels = inputs_labels

                is_multiplelabel = len(labels.shape) > 1 and \
                    not isinstance(ssrl, int) and not is_barlowtwins

                raw_dataset_val = len(inputs.shape) == 5
                if raw_dataset_val:
                    assert phase != 'train'
                    assert inputs.shape[0] == 1, \
                        ('When providing a set of patches as with the use of '
                         '`raw_dataset_val`, we expect the batch size to be '
                         '1.')

                    # Here we are considering all patches as a batch, in the
                    # point of view of the inference through the network.
                    # Therefore, we are assuming that all extracted patches can
                    # fit on the GPU memory.  It means we are performing no
                    # verification in this regard.
                    inputs = inputs[0]

                    if is_triplet:
                        assert (inputs_pos.shape[0] == 1
                                and inputs_neg.shape[0] == 1), \
                            ('When providing a set of patches as with the use '
                             'of `raw_dataset_val`, we expect the batch size '
                             'to be 1.')

                        inputs_pos = inputs_pos[0]
                        inputs_neg = inputs_neg[0]
                        inputs, inputs_pos, inputs_neg = \
                            subsample_patches(
                                [inputs, inputs_pos, inputs_neg],
                            )

                    elif is_contrastive:
                        assert inputs_other.shape[0] == 1, \
                            ('When providing a set of patches as with the use '
                             'of `raw_dataset_val`, we expect the batch size '
                             'to be 1.')

                        inputs_other = inputs_other[0]
                        inputs, inputs_other = \
                            subsample_patches([inputs, inputs_other])
                        pairlabels = pairlabels.expand(inputs.size(0))

                    labels = labels.expand(inputs.size(0))

                inputs = inputs.to(device)
                if is_triplet:
                    inputs_pos = inputs_pos.to(device)
                    inputs_neg = inputs_neg.to(device)
                elif is_contrastive:
                    inputs_other = inputs_other.to(device)
                    pairlabels = pairlabels.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an
                    # auxiliary output. In train mode we calculate the loss by
                    # summing the final output and the auxiliary output but in
                    # testing we only consider the final output.

                    def regularize_triplet(batch_fvs):
                        return batch_fvs / torch.norm(batch_fvs,
                                                      dim=1,
                                                      keepdim=True)

                    if is_inception and phase == 'train':
                        # From
                        # https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        if is_triplet:
                            outputs_pos, aux_outputs_pos = model(inputs_pos)
                            outputs_neg, aux_outputs_neg = model(inputs_neg)

                            outputs_pos, aux_outputs_pos = \
                                regularize_triplet(outputs_pos), \
                                regularize_triplet(aux_outputs_pos)
                            outputs_neg, aux_outputs_neg = \
                                regularize_triplet(outputs_neg), \
                                regularize_triplet(aux_outputs_neg)

                            if is_triplet_cross_entropy:
                                loss1 = criterion(
                                    outputs, outputs_pos, outputs_neg, labels,
                                )
                                loss2 = criterion(
                                    aux_outputs, aux_outputs_pos,
                                    aux_outputs_neg, labels,
                                )
                            else:
                                loss1 = criterion(
                                    outputs, outputs_pos, outputs_neg,
                                )
                                loss2 = criterion(
                                    aux_outputs, aux_outputs_pos,
                                    aux_outputs_neg,
                                )

                        elif is_contrastive:
                            outputs_other, aux_outputs_other = \
                                model(inputs_other)

                            # TODO: Regularize for contrastive loss.

                            loss1 = criterion(
                                outputs,
                                outputs_other,
                                pairlabels,
                            )
                            loss2 = criterion(
                                aux_outputs,
                                aux_outputs_other,
                                pairlabels,
                            )

                        else:   # not `is_triplet` nor `is_contrastive`
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)

                        loss = loss1 + 0.4*loss2

                    else:       # not `is_inception` or `phase != 'train'`
                        if is_barlowtwins:
                            outputs = model(inputs, labels)
                        else:
                            outputs = model(inputs)

                        if is_triplet:
                            outputs_pos = model(inputs_pos)
                            outputs_neg = model(inputs_neg)

                            outputs_pos = regularize_triplet(outputs_pos)
                            outputs_neg = regularize_triplet(outputs_neg)

                            if is_triplet_cross_entropy:
                                loss = criterion(
                                    outputs, outputs_pos, outputs_neg, labels,
                                )
                            else:
                                loss = criterion(
                                    outputs, outputs_pos, outputs_neg,
                                )

                        elif is_contrastive:
                            outputs_other = model(inputs_other)
                            loss = criterion(
                                outputs, outputs_other, pairlabels,
                            )

                        else:
                            if is_barlowtwins:
                                loss = criterion(outputs)
                            else:
                                loss = criterion(outputs, labels)

                    if not is_multiplelabel:
                        if not isinstance(ssrl, int) and not is_barlowtwins:
                            _, preds = torch.max(outputs, 1)
                        else:
                            preds = outputs
                    else:
                        preds = (outputs > 0).int()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                labels = labels.to('cpu')
                preds = preds.to('cpu')

                # fusion of patches on a single prediction
                if raw_dataset_val:
                    assert len(labels.unique()) == 1, labels
                    labels = labels[:1]

                    # max voting
                    indexes, counts = preds.unique(return_counts=True)
                    preds = indexes[counts.argmax()].unsqueeze(-1)

                running_loss += loss.item() * inputs.size(0)
                if not isinstance(ssrl, int) and not is_barlowtwins:
                    running_corrects += torch.sum(preds == labels.data)
                    running_counter += preds.shape[0]

                    maxdim = max(max(labels.max(), preds.max()) + 1, maxdim) \
                        if not is_multiplelabel else labels.size(1)
                    cm_func = sklearn.metrics.confusion_matrix \
                        if not is_multiplelabel \
                        else sklearn.metrics.multilabel_confusion_matrix
                    cm = cm_func(
                        labels,
                        preds,
                        labels=range(maxdim),
                    )
                    cm[:running_cm.shape[0], :running_cm.shape[1]] += \
                        running_cm
                    running_cm = cm

            epoch_loss = running_loss / running_counter
            epoch_metric = {
                'LOSS': -epoch_loss,
            }
            if not isinstance(ssrl, int) and not is_barlowtwins:
                epoch_metric['ACC'] = (
                    (running_corrects.double() / running_counter).item() /
                    (1 if not is_multiplelabel else running_cm.shape[0])
                )
                epoch_metric['NA'] = (
                    (running_cm.diagonal() /
                     running_cm.sum(axis=1)).mean()
                    if not is_multiplelabel else
                    sum([(aux.diagonal() / aux.sum(axis=1)).mean()
                         for aux in running_cm]) / running_cm.shape[0]
                )
            assert (all(x in best_metric for x in epoch_metric) and
                    all(x in epoch_metric for x in best_metric)), \
                (best_metric.keys(), epoch_metric.keys())

            logger.info('({}) Metrics: {}'.format(phase, epoch_metric))

            if phase == 'val':
                history.append(epoch_metric)

                # deep copy the model
                for metric in best_metric:
                    if epoch_metric[metric] > best_metric[metric]:
                        best_metric[metric] = epoch_metric[metric]
                        model_weights[metric] = \
                            copy.deepcopy(model.state_dict())
                        best_epoch[metric] = epoch

                LOSS_ONLY_BASED_EARLY_STOPPING = True
                """When True, the period of tolerance is based on the last
                improvement according solely to LOSS metric.  Otherwise, the
                tolerance is based on the improvement of any of the considered
                metrics, i.e., if any metric improves, the counter for the
                tolerance zeroes.

                """
                if early_stopping is not None and (
                        (epoch - best_epoch['LOSS'] > early_stopping)
                        if LOSS_ONLY_BASED_EARLY_STOPPING else
                        (epoch - max(best_epoch.values()) > early_stopping)
                ):
                    stop_early = True

            if phase == 'val' and (not is_triplet and
                                   not is_contrastive and
                                   not is_barlowtwins or
                                   is_triplet_cross_entropy):
                if not is_multiplelabel:
                    if not isinstance(ssrl, int):
                        logger.debug('Confusion matrix of the epoch:')
                        for line in (running_cm /
                                     running_cm.sum(axis=1, keepdims=True)):
                            logger.debug('  ' ' '.join(['{:.2f}'.format(x)
                                                        for x in line]))
                else:
                    logger.debug('Confusion matrices of the epoch:')
                    for cm in [cm / cm.sum(axis=1, keepdims=True)
                               for cm in running_cm]:
                        for line in cm:
                            logger.debug('  ' ' '.join(['{:.2f}'.format(x)
                                                        for x in line]))
                        logger.debug('-----------')

        if stop_early:
            logger.info(f'Early stop at {epoch}.')
            break

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60,
        time_elapsed % 60,
    ))
    for metric in best_metric:
        logger.info('Best {}: {}, Best epoch: {}'.format(
            metric, best_metric[metric], best_epoch[metric],
        ))

    logger.info('finetune.train_model(): RETURN')
    return model, model_weights, best_metric, history


def model_extract(model,
                  dataloader,
                  device: str,
                  layer_name: Optional[str] = None,
                  ):
    """Use the given model to extract the data from the given Data Loader.

    model:
        The model to be used for feature extraction.

    dataloader:
        A `torch.utils.data.DataLoader` that provides the image or a set of
        patches from which to extract the features.  When the data provided
        (not considering the provided label) with the Data Loader has 5
        dimensions, this functions assumes a raw dataset is being used.  In
        this case, it asserts for the provided batch size to be 1.

    device:
        The device on which to put the features and the labels for training.
        Probably the same device in which the network was sent to.  See
        https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
        for more information.

    layer_name:
        Name of the layer from which extract features.  When not provided or
        None, then extract last layer.

    Return
    ------

    features:
        For a normal patch-based dataset.  A bi-dimensional tensor containing
        the feature vectors of each of the patches.

    (features, indexes):
        For a raw dataset.  `features` is the same as above.  `indexes` is a
        list containing information of the features in `features` that refer to
        each of the images.  For instance, `features[indexes[i-1]:indexes[i]]`
        are the features of the patches of the `i`-th image.  This list has an
        appended 0 at the end in order to make the indexing work for `i = 0`.
        Hence, it should one needs care on measuring its length for indexing
        purpose, as the quantity of images is determined by `len(indexes) - 1`.

    """
    logger.debug('finetune.model_extract()')
    logger.info(f'DEVICE: {device}')
    since = time.time()

    model.eval()                # Set model to evaluate mode

    if layer_name is not None:
        model = models.feature_extraction.create_feature_extractor(
            model, [layer_name])

    all_outputs = []
    all_sizes = []              # Used and returned only for raw datasets
    for inputs, labels in envconfig.tqdm(dataloader):
        raw_dataset = len(inputs.shape) == 5
        if raw_dataset:
            assert inputs.shape[0] == 1, \
                ('When providing a set of patches as with the use of '
                 '`raw_dataset`, we expect the batch size to be 1.')

            # Here we are considering all patches as a batch, in the
            # point of view of the inference through the network.
            # Therefore, we are assuming that all extracted patches can
            # fit on the GPU memory.  It means we are performing no
            # verification in this regard.
            inputs = inputs[0]
            all_sizes.append(inputs.shape[0])

        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            if layer_name is not None:
                outputs = outputs[layer_name]
            outputs = outputs.to('cpu')
            all_outputs = [torch.cat(all_outputs + [outputs])]

    all_outputs = all_outputs[0]

    time_elapsed = time.time() - since
    logger.info('Prediction complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60,
        time_elapsed % 60,
    ))

    # Assuming all examples would have the same `raw_dataset` property, we are
    # using here the `raw_dataset` property for the last example for the
    # features were extracted.
    if raw_dataset:
        all_sizes = list(it.accumulate(all_sizes)) + [0]
        return all_outputs, all_sizes
    else:
        assert ((len(dataloader) - 1) * BATCH_SIZE
                < len(all_outputs)
                <= len(dataloader) * BATCH_SIZE), \
            (len(all_outputs), len(dataloader))
        return all_outputs


def set_parameter_requires_grad(model, feature_extracting):
    """From a PyTorch tutorial."""
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class SimpleCNN(nn.Module):
    # https://tomroth.com.au/pytorch-cnn/
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # we use the maxpool multiple times, but define it once
        self.pool = nn.MaxPool2d(2, 2)
        # in_channels = 6 because self.conv1 output 6 channel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 71*71 comes from the dimension of the last convnet layer
        self.fc1 = nn.Linear(16*71*71, 120)
        self.fc2 = nn.Linear(120, 84)
        self.batch_norm_84 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(dropout)
        # self.batch_norm_6 = nn.BatchNorm2d(6)
        # self.batch_norm_16 = nn.BatchNorm2d(16)
        # self.batch_norm_120 = nn.BatchNorm1d(120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*71*71)
        x = F.relu(self.fc1(x))
        x = F.relu(self.batch_norm_84(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # no activation on final layer
        return x


# https://github.com/facebookresearch/barlowtwins/blob/main/main.py
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def initialize_model(model_name, num_classes, feature_extract,
                     use_pretrained=True, network_path=None):
    """Get base PyTorch model.

    network_path:
        Only used for FusionCNNs, as the weights of the pretrained backbones
        need to be loaded.

    """
    # Initialize these variables which will be set in this if statement. Each
    # of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "alexnet":
        model_ft = models.alexnet(
            weights=(models.AlexNet_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "convnext":
        model_ft = models.convnext_large(
            weights=(models.ConvNeXt_Large_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[2].in_features
        model_ft.classifier[2] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "densenet":
        model_ft = models.densenet161(
            weights=(models.DenseNet161_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "efficientnet":
        model_ft = models.efficientnet_b7(
            weights=(models.EfficientNet_B7_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "efficientnet_v2":
        model_ft = models.efficientnet_v2_l(
            weights=(models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "googlenet":
        model_ft = models.googlenet(
            weights=(models.GoogLeNet_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name in ["inception", "inception_v3"]:
        # Be careful, expects (299,299) sized images and has auxiliary output.
        model_ft = models.inception_v3(
            weights=(models.Inception_V3_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif model_name == "mnasnet":
        model_ft = models.mnasnet1_3(
            weights=(models.MNASNet1_3_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mobilenet_v2":
        model_ft = models.mobilenet_v2(
            weights=(models.MobileNet_V2_Weights.IMAGENET1K_V2
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mobilenet_v3":
        model_ft = models.mobilenet_v3_large(
            weights=(models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "regnet":
        model_ft = models.regnet_y_128gf(
            weights=(models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet":
        model_ft = models.resnet152(
            weights=(models.ResNet152_Weights.IMAGENET1K_V2
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnext":
        model_ft = models.resnext101_64x4d(
            weights=(models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "shufflenet_v2":
        model_ft = models.shufflenet_v2_x2_0(
            weights=(models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_1(
            weights=(models.SqueezeNet1_1_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        # Maybe change it here to use flatten instead of a convolution
        model_ft.classifier[1] = nn.Conv2d(512, num_classes,
                                           kernel_size=(1, 1),
                                           stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "swin":
        model_ft = models.swin_b(
            weights=(models.Swin_B_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        model_ft = models.vgg19_bn(
            weights=(models.VGG19_BN_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vit":
        model_ft = models.vit_h_14(
            weights=(models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
                     if use_pretrained else None),
            # weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "wide_resnet":
        model_ft = models.wide_resnet101_2(
            weights=(models.Wide_ResNet101_2_Weights.IMAGENET1K_V2
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "cnn":
        model_ft = SimpleCNN(num_classes, dropout=0.0)
        # model_ft = SimpleCNN(num_classes)
        input_size = 299

    elif model_name == "resnet50bt":
        # https://github.com/facebookresearch/barlowtwins
        model_ft = torch.hub.load('facebookresearch/barlowtwins:main',
                                  'resnet50')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif model_name == "ran2019":
        """Architecture proposed in Ran, X.; Xue, L.; Zhang, Y.; Liu, Z.; Sang,
        X. and He, J.; "Rock Classification from Field Image Patches Analyzed
        Using a Deep Convolutional Neural Network", Mathematics, 7(8):755,
        August (2019). DOI: 10.3390/math7080755."""
        class Ran2019(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 5, padding='same')
                # we use the `maxpool` multiple times, but define it once
                self.maxpool = nn.MaxPool2d(3, 2, padding=1)
                self.conv2 = nn.Conv2d(64, 64, 5, padding='same')
                self.fc_in_size = 64 * 24 * 24
                self.fc1 = nn.Linear(self.fc_in_size, 384)
                self.fc2 = nn.Linear(384, 192)
                self.fc3 = nn.Linear(192, num_classes)

            def forward(self, x):
                x = self.maxpool(F.relu(self.conv1(x)))
                x = self.maxpool(F.relu(self.conv2(x)))
                x = x.view(-1, self.fc_in_size)
                x = F.relu(self.fc1(x))  # they do not mention ReLU application
                x = F.relu(self.fc2(x))  # they do not mention ReLU application
                x = self.fc3(x)
                return x
        model_ft = Ran2019(num_classes)
        # input_size = 128
        input_size = 96

    elif model_name == "fusioncnns":
        class FusionCNNs(nn.Module):
            def __init__(self):
                super().__init__()

                def get_network(name):
                    network, _, finetuned = load_pretrained_model(
                        name, num_classes, feature_extract=feature_extract,
                        # TODO: need adjustment below for "rocklongfiles" for
                        # those names, just in case
                        network_path=network_path.replace('_mn:fusioncnns_',
                                                          f'_mn:{name}_'),
                    )
                    assert finetuned, name
                    return network

                self.squeezenet = get_network('squeezenet')
                self.mobilenet_v2 = get_network('mobilenet_v2')
                self.mobilenet_v3 = get_network('mobilenet_v3')
                self.mnasnet = get_network('mnasnet')
                self.shufflenet_v2 = get_network('shufflenet_v2')
                self.googlenet = get_network('googlenet')
                # self.inception_v3 = get_network('inception_v3')
                self.densenet = get_network('densenet')
                self.alexnet = get_network('alexnet')
                self.resnet = get_network('resnet')
                self.efficientnet = get_network('efficientnet')
                self.resnext = get_network('resnext')
                self.swin = get_network('swin')
                self.efficientnet_v2 = get_network('efficientnet_v2')
                self.wide_resnet = get_network('wide_resnet')
                self.vgg = get_network('vgg')
                self.convnext = get_network('convnext')
                self.vit = get_network('vit')
                self.regnet = get_network('regnet')

                self.networks = [
                    self.squeezenet,
                    self.mobilenet_v2,
                    self.mobilenet_v3,
                    self.mnasnet,
                    self.shufflenet_v2,
                    self.googlenet,
                    # self.inception_v3,
                    self.densenet,
                    self.alexnet,
                    self.resnet,
                    self.efficientnet,
                    self.resnext,
                    self.swin,
                    self.efficientnet_v2,
                    self.wide_resnet,
                    self.vgg,
                    self.convnext,
                    self.vit,
                    self.regnet,
                ]

                self.fc = nn.Sequential(
                    nn.Linear(len(self.networks) * num_classes, 30),
                    nn.ReLU(),
                    nn.Linear(30, num_classes),
                )

            def forward(self, x):
                x = torch.cat([network(x) for network in self.networks], dim=1)
                x = self.fc(x)
                return x

        model_ft = FusionCNNs()
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224

    elif model_name == "barlowtwins":
        class BarlowTwins(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = models.resnet50(zero_init_residual=True)
                self.backbone.fc = nn.Identity()

                # projector
                sizes = [2048, 8192, 8192, 8192]
                layers = []
                for i in range(len(sizes) - 2):
                    layers.append(nn.Linear(sizes[i],
                                            sizes[i + 1],
                                            bias=False))
                    layers.append(nn.BatchNorm1d(sizes[i + 1]))
                    layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
                self.projector = nn.Sequential(*layers)

                # normalization layer for the representations z1 and z2
                self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

            def forward(self, y1, y2):
                z1 = self.projector(self.backbone(y1))
                z2 = self.projector(self.backbone(y2))

                # empirical cross-correlation matrix
                c = self.bn(z1).T @ self.bn(z2)

                # sum the cross-correlation matrix between all gpus
                c.div_(BATCH_SIZE)
                # torch.distributed.all_reduce(c)

                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                lambd = 1.0
                loss = on_diag + lambd * off_diag
                return loss

        model_ft = BarlowTwins()
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224

    elif model_name == 'vit_b_16':
        model_ft = models.vit_b_16(
            weights=(models.ViT_B_16_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
            # weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
            # weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'vit_b_32':
        model_ft = models.vit_b_32(
            weights=(models.ViT_B_32_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'vit_l_16':
        model_ft = models.vit_l_16(
            weights=(models.ViT_L_16_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
            # weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1,
            # weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'vit_l_32':
        model_ft = models.vit_l_32(
            weights=(models.ViT_L_32_Weights.IMAGENET1K_V1
                     if use_pretrained else None),
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vit-pytorch":
        import vit_pytorch

        model_ft = vit_pytorch.ViT(
            image_size=256,
            patch_size=32,
            num_classes=1000,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.mlp_head[1].in_features
        model_ft.mlp_head[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 256

    else:
        logger.error("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def load_pretrained_model(model_name,
                          num_classes,
                          ssrl: Optional[Union[int, str]] = None,
                          feature_extract: Optional[bool] = None,
                          network_path=None):
    """Load previously saved model from `network_path`.

    ssrl:
       Should be `None` when it is a normal network, otherwise an integer
       greater than 0 specifying the size of the occluded patch for
       Self-Supervised Representation Learning (SSRL) with Masked Prediction as
       pretext task.

    """

    ssrlnum = int(ssrl.split('_ss:')[1].split('_')[0]) \
        if isinstance(ssrl, str) else ssrl

    model_ft, input_size = initialize_model(
        model_name,
        num_classes if ssrlnum is None else 3*ssrlnum*ssrlnum,
        feature_extract=(feature_extract
                         if feature_extract is not None
                         else True),
        use_pretrained=True,
        network_path=network_path,
    )

    finetuned = False  # whether the model have already been fined before by us
    if network_path is not None and os.path.exists(network_path):
        logger.info(f'Loading weights from: {network_path}.')
        m_state_dict = torch.load(network_path)
        model_ft.load_state_dict(m_state_dict)
        finetuned = True
    elif isinstance(ssrl, str) and os.path.exists(ssrl):
        logger.info(f'Loading SSRL weights from: {ssrl}.')
        m_state_dict = torch.load(ssrl)
        model_ft.load_state_dict(m_state_dict)
        if model_name == 'cnn':
            model_ft.fc3 = nn.Linear(model_ft.fc3.in_features, num_classes)
        elif model_name == 'inception':
            model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
            model_ft.AuxLogits.fc = nn.Linear(
                model_ft.AuxLogits.fc.in_features, num_classes)
        else:
            raise Exception(f'Not implemented for model {model_name}.')
    assert ssrl is None or not finetuned, (ssrl, finetuned)

    return model_ft, input_size, finetuned


def get_data_transforms(input_size: int,
                        segmentation: bool,
                        raw_dataset: bool = False,
                        stride: Optional[int] = None,
                        scale: Optional[float] = None,
                        augment_rotations: Optional[bool] = False,
                        feature_extraction: bool = False):
    """Return a dictionary with corresponding data transforms for each of the
    part of the dataset (i.e., train, val, and test).

    input_size:
        The input size of the network that will use the data transformed by the
        transforms provided through this function.

    raw_dataset:
        Whether the dataset being used is a raw dataset (with raw images
        instead of patches).

    stride:
        Only required when `raw_dataset` is True, otherwise it is ignored.  It
        is used for patch extraction for both val and test datasets.

    scale:
        The scale amount to each extracted patch can be randomly scaled.  Each
        patch will receive a scaling by a random factor in [1-`scale`,
        1+`scale`].  Default is no scaling.  When provided, this value should
        be greater than 0.

    augment_rotations:
        Whether to perform rotation transformations as augmentation.

    feature_extraction:
        Only required when `raw_dataset` is True, otherwise it is ignored.
        Whether to force patch extraction for the training set as well.  Set
        True when the training features should be extracted to fit a
        classifier.

    """
    assert scale is None or scale > 0.0, scale
    assert isinstance(augment_rotations, bool), type(augment_rotations)

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    class RotationTransform:
        """Rotate by one of the given angles."""

        def __init__(self, angles=[0, 90, 180, 270]):
            self.angles = angles

        def __call__(self, x):
            angle = random.choice(self.angles)
            return transforms.functional.rotate(x, angle)

    class RandomCropMasked(transforms.RandomCrop):
        """It returns only crops inside the mask."""
        def __init__(self, size):
            logger.debug(f'RandomCropMasked({size}) constructed')
            assert isinstance(size, int), type(size)
            self._size = (size, size)

        def example(self, image, mask):
            valid_mask = False
            counter_tries = 0
            while not valid_mask:
                i, j, h, w = self.get_params(image, self._size)
                masknp = 1 - rocklib.binarize_mask(np.asarray(
                    transforms.functional.crop(mask, i, j, h, w)
                ))
                valid_mask = masknp.sum() == 0
                counter_tries += 1
            if counter_tries > 200:  # magic number
                logger.warning('Taking {} tries to get a valid patch'.format(
                    counter_tries))
            image = transforms.functional.crop(image, i, j, h, w)
            return image

        def __call__(self, image_mask):
            image, mask = image_mask
            image = self.example(image, mask)
            return image

    def compose(lst):
        return transforms.Compose([x for x in lst if x is not None])

    if raw_dataset:
        assert stride is not None

        patchextraction = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            rocklib.PatchExtraction(size=input_size, stride=stride),
        ])

        assert scale is None or not segmentation
        if scale is not None:
            scaled_random_crop = transforms.Compose([
                # The idea of this transform is to randomly crop a patch of
                # `input_size` with random scaling.  It does so by ensuring
                # that when `scale` < 1.0, we will have no black borders around
                # from padding.
                transforms.RandomCrop(math.ceil(input_size / (1 - scale))),
                transforms.RandomAffine(0, scale=(1 - scale, 1 + scale)),
                transforms.CenterCrop(input_size),
            ])
        else:
            scaled_random_crop = (RandomCropMasked(input_size)
                                  if segmentation else
                                  transforms.RandomCrop(input_size))

        randomcrop = compose([
            scaled_random_crop,
            RotationTransform() if augment_rotations else None,
            transforms.ToTensor(),
            normalize,
        ])

        data_transforms = {
            'train': randomcrop if not feature_extraction else patchextraction,
            'val': patchextraction,
            'test': patchextraction,
        }

    else:
        simple_read_patch = compose([
            transforms.ToTensor(),
            transforms.Resize(input_size, antialias=None),  # Just in case
            transforms.CenterCrop(input_size),  # Just in case
            RotationTransform() if augment_rotations else None,
            normalize,
        ])

        data_transforms = {
            'train': simple_read_patch,
            'val': simple_read_patch,
            'test': simple_read_patch,
        }

    return data_transforms


def get_torch_device(device):
    """Format the PyTorch device.  When there is not GPU available, "cpu" is
    always returned.  Otherwise, if `device` is already a string, it is
    returned as is if it (hopefully) specified a PyTorch device already.  When
    `device` is an interger, a PyTorch-formatted string device is returned
    specifying the corresponding device.  If `device` is the string "auto", it
    will use `GPUtil` package to select the GPU with the most memory available.

    """
    if not torch.cuda.is_available():
        logger.warning('CUDA is not available!')
        
    torchdevice = torch.device((f'cuda:{device}' if isinstance(device, int)
                                else 'cuda:{}'.format(
                                        GPUtil.getAvailable(
                                            order='memory',
                                            maxLoad=1.00001,
                                            maxMemory=1.00001,
                                        )[0]
                                ) if device == "auto"
                                else device)
                               if torch.cuda.is_available()
                               else "cpu")
    return torchdevice


def unfreeze_InceptionV3(model, point):
    """It freezes the network up to the `point` and unfreeze the network from
    the `point` on.  This function is assumed to be used along with
    Inception-V3, so it also ensures the output layer of `AuxLogits` is always
    unfrozen.

    """
    assert isinstance(point, int), type(point)
    parameters = list(model.parameters())
    for param in parameters[:point]:
        param.requires_grad = False
    for param in parameters[point:]:
        param.requires_grad = True
    for index in range(210, 218):  # `AuxLogits` of Inception-V3.
        parameters[index].requires_grad = True
    params_to_update = [p for p in parameters if p.requires_grad]
    logger.debug('unfreeze: point, len(params_to_update): '
                 f'{point}, {len(params_to_update)}')
    return params_to_update


def unfreeze(model, point):
    """It freezes the network up to the `point` and unfreeze the network from
    the `point` on.

    """
    assert isinstance(point, int), type(point)
    parameters = list(model.parameters())
    assert point >= 0 and point < len(parameters), point
    for param in parameters[:point]:
        param.requires_grad = False
    for param in parameters[point:]:
        param.requires_grad = True
    params_to_update = [p for p in parameters if p.requires_grad]
    logger.debug('unfreeze: point, len(params_to_update): '
                 f'{point}, {len(params_to_update)}')
    return params_to_update


def finetune(classes: List[int],
             partition: int,
             versions: List[int],
             model_name: str,
             network_path_dict: Dict[str, str],
             network_path_partial: Optional[str],
             raw_dataset: bool,
             scale: Optional[float],
             augment_rotations: bool,
             stride: int,
             segmentation: bool,
             num_epochs: int,
             class_weight: bool,
             loss_name: str,
             optimizer: str,
             early_stopping: int,
             ssrl: Optional[str],
             patch_file_part_train_lst: List[str],
             patch_file_part_val_lst: List[str],
             device: Union[str, int],
             override: bool,
             experiment: Dict[str, str],
             ):
    """Fine-tune the specified model with the specified parameters.

    model_name:
        Name of the model as one of ['resnet', 'alexnet', 'vgg', 'squeezenet',
        'densenet', 'inception', 'cnn', ].

    network_path_partial:
        A string containing the formatting option '{point}'.  It will be passed
        to `train_model_with_freezing_points`.

    class_weight:
        Whether to weight the classes when calculating the loss.  Weighting is
        useful for unbalanced problems.  The weights are calculated through
        `compute_class_weight` of scikit-learn.

    loss_name:
        The name of the loss to be employed.  Accepted losses:
        'CrossEntropyLoss', 'TripletMarginLoss', and 'ContrastiveLoss'.

    optimizer:
        The name of the optmizer to be employed.  Accepted optmizers:
        'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'NAdam',
        'RAdam', 'RMSprop', 'Rprop', 'SGD'.

    early_stopping:
        The number of epochs to tolerate no improvement before stopping the
        training phase.

    ssrl:
       Should be `None` when it is a normal network, otherwise an integer
       greater than 0 specifying the size of the occluded patch for
       Self-Supervised Representation Learning (SSRL) with Masked Prediction as
       pretext task.

    device:
        When `device` is an integer, it will try go use the corresponsing GPU
        device.  To run on CPU, `device` should be provided as a string "cpu".
        If "auto" is provided, it will try to use the GPU with more memory
        available.  Devices can also be specified in PyTorch's way, i.e.,
        "cuda:0", "cuda:1", etc., which is assumed to be the case when the
        provided device is an string.  When GPUs are not available, this
        argument is ignored and device is set to be "auto".

    override : str
        Whether to override previous computation in case it was already
        performed.

    raw_dataset:
        Whether to use the raw dataset of images instead of the predefined
        patches.

    scale:
        The scale amount to each extracted patch can be randomly scaled.  Each
        patch will receive a scaling by a random factor in [1-`scale`,
        1+`scale`].  Default is no scaling.  When provided, this value should
        be greater than 0.

    augment_rotations:
        Whether to perform random rotations of 0, 90, 180, or 270 degrees as a
        form of augmentation.

    experiment:
        A dictionary containing information of all other parameters received by
        this function.

    """

    assert isinstance(loss_name, str), type(experiment['loss_name'])
    assert (experiment['loss_name']
            in ['CrossEntropyLoss',
                'TripletMarginLoss',
                'ContrastiveLoss',
                'TripletMarginWithCrossEntropyLoss',
                'MSELoss',
                'L1Loss',
                'Identity',
                ]), \
        experiment['loss_name']

    if experiment['loss_name'] not in ['CrossEntropyLoss',
                                       'TripletMarginWithCrossEntropyLoss',]:
        assert not experiment['class_weight'], \
            '{} loss does not support class weight.'.format(
                experiment['loss_name'],
            )

    if experiment['loss_name'] not in ['CrossEntropyLoss',
                                       'MSELoss',
                                       'L1Loss',
                                       'Identity',]:
        assert experiment['raw_dataset'], \
            '{} loss is implemented only for raw dataset.'.format(
                experiment['loss_name'],
            )

    assert not isinstance(experiment['ssrl'], int) or (
        experiment['loss_name'] in ['MSELoss',
                                    'L1Loss',]
    ), (
        f'ssrl: {experiment["ssrl"]}, '
        f'loss_name: {experiment["loss_name"]}'
    )

    assert (not isinstance(experiment['ssrl'], int) or
            not experiment['raw_dataset']), \
        'SSRL is not implemented for raw dataset.'

    assert isinstance(experiment['optimizer'], str), \
        type(experiment['optimizer'])

    assert 'LOSS' in experiment['network_path_dict'], \
        ('We expect the LOSS to be always provided when training, so that the '
         'final model optimized for it can be used.  It will be useful in the '
         'future when we add the functionality of fine-tuning a network from '
         'a checkpoint from previous fine-tuning.')

    is_triplet = 'TripletMargin' in loss_name
    is_alternative_loss = (experiment['loss_name'] == 'TripletMarginLoss' or
                           (loss_name.startswith('ContrastiveLoss(')
                            and loss_name.endswith(')')))

    logger.info(f'Network will be written in: {network_path_dict}')
    logger.debug(f'Experiment: {experiment}')

    if all(os.path.exists(network_path_dict[x])
           for x in network_path_dict
           if not is_alternative_loss or x == 'LOSS'):
        if override:
            logger.warning('OVERRIDE: New model(s) will override old one(s).')
        else:
            logger.warning('Networks were already trained before.  '
                           'So, skipping a new training.')
            return
    elif any(os.path.exists(network_path_dict[x])
             for x in network_path_dict):
        logger.warning('Not all but some models trained before will be '
                       'overridden.')

    num_classes = len(classes)

    model_ft, input_size, finetuned = load_pretrained_model(
        model_name,
        num_classes,
        ssrl=ssrl,
        network_path=rocklongfiles.shortfile(network_path_dict['LOSS'])
    )

    data_transforms = get_data_transforms(
        input_size,
        segmentation=experiment['segmentation'],
        stride=experiment['stride'],
        scale=experiment['scale'],
        augment_rotations=experiment['augment_rotations'],
        raw_dataset=experiment['raw_dataset'],
    )
    data_transforms_patches = get_data_transforms(
        input_size,
        segmentation=experiment['segmentation'],
        stride=experiment['stride'],
        scale=experiment['scale'],
        augment_rotations=experiment['augment_rotations'],
        raw_dataset=False,
    )

    # Preparing the dataset.
    assert len(patch_file_part_train_lst) == 1, \
        ('TODO: It needs the implementation for multi-scale training, however '
         'it is deprecated, as a better approach seems to be implementing '
         "multi-scale through PyTorch's data loader.  See "
         'https://pytorch.org/tutorials/beginner/finetuning_torchvision_models'
         '_tutorial.html#load-data '
         'for more details.')
    assert len(patch_file_part_val_lst) == 1, patch_file_part_val_lst
    map_file = {
        'train': patch_file_part_train_lst[0],
        'val': patch_file_part_val_lst[0],
    }

    image_datasets = {
        part: (
            rocklib.RockDatasetRaw(
                partition=partition,
                part=part,
                segmentation=experiment['segmentation'],
                loss_type=('triplet'
                           if is_triplet
                           else 'contrastive'
                           if experiment['loss_name'] == 'ContrastiveLoss'
                           else None),
                transforms=data_transforms[part],
                classes=experiment['classes'],
                keep_in_memory=True,
            )
            if experiment['raw_dataset'] and part == 'train' else
            rocklib.RockDataset(
                filename=map_file[part],
                classes=experiment['classes'],
                transforms=data_transforms_patches[part],
            )
            if not isinstance(ssrl, int) and model_name != 'barlowtwins' else
            rocklib.RockDatasetSSRL(
                filename=map_file[part],
                p=ssrl,
                classes=experiment['classes'],
                transforms=data_transforms_patches[part],
            )
            if isinstance(ssrl, int) else
            rocklib.RockDatasetBarlowTwins(
                filename=map_file[part],
                classes=experiment['classes'],
                transforms=data_transforms_patches[part],
            )
        )
        for part in map_file
    }
    del map_file

    logger.debug(f'Datasets: {image_datasets}')
    logger.debug('Length of datasets: {}'.format({
        part: len(image_datasets[part])
        for part in image_datasets
    }))

    # Create training and validation `torch.utils.data.DataLoader`s.  When
    # running for the raw dataset, we use a sampler that repeats over the
    # dataset a certain number of times.  This is so because raw dataset
    # provides fewer examples for a epoch and we want to extend the duration of
    # a training epoch before evaluating the model for the validation set.  In
    # this case, shuffle parameter for the data loader must be None.
    # Furthermore, when working for the validation set, which contains patches,
    # when the raw dataset is employed, we want the batch size to be 1.
    dataloaders_dict = {
        part: torch.utils.data.DataLoader(
            image_datasets[part],
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
            shuffle=((part == 'train')
                     if not experiment['raw_dataset']
                     else None),
            sampler=(
                rocklib.RepeaterSampler(
                    size=len(image_datasets[part]),
                    times=RAW_DATASET_TIMES,
                    shuffle=(part == 'train'),
                )
                if part == 'train' and experiment['raw_dataset']
                else None
            ),
        )
        for part in image_datasets
    }

    torchdevice = get_torch_device(device)
    model_ft = model_ft.to(torchdevice)

    # Setup the loss fxn
    if class_weight:
        logger.debug('Calculating weights for loss')
        labels = image_datasets['train'].labels()
        weight = sklearn.utils.class_weight.compute_class_weight(
            'balanced', classes=sorted(set(labels)), y=labels,
        )
        logger.debug(f'Weights: {weight}')
        weight = torch.from_numpy(weight).float()
        weight = weight.to(torchdevice)
        del labels
    else:
        weight = None

    if loss_name == 'TripletMarginLoss':
        criterion = nn.TripletMarginLoss()
    elif loss_name == 'ContrastiveLoss':
        criterion = rocklib.ContrastiveLoss()
    elif loss_name == 'TripletMarginWithCrossEntropyLoss':
        criterion = rocklib.TripletMarginWithCrossEntropyLoss(
            weight=weight,
        )
    elif loss_name == 'MSELoss':
        criterion = nn.MSELoss()
    elif loss_name == 'L1Loss':
        criterion = nn.L1Loss()
    elif loss_name == 'Identity':
        criterion = nn.Identity()
    else:
        criterion = nn.CrossEntropyLoss(weight)
    del weight

    rockparams.optimizer_params['SGD']['momentum'] = 0.9
    # rockparams.optimizer_params['SGD']['lr'] = 1e-5
    # rockparams.optimizer_params['SGD']['nesterov'] = True
    # rockparams.optimizer_params['Adadelta']['lr'] = 0.5
    # rockparams.optimizer_params['Adam']['lr'] = 1e-4
    # rockparams.optimizer_params['AdamW']['lr'] = 1e-4

    # Train and evaluate
    _, model_weights, best_metric, hist = train_model_with_freezing_points(
            model=model_ft,
            freezing_points=FREEZING_POINTS,
            optimizer_func=rockparams.optimizer_dict[optimizer],
            optimizer_params=([{**rockparams.optimizer_params[optimizer],
                                'lr': lr} for lr in LEARNING_RATES]
                              if LEARNING_RATES
                              else rockparams.optimizer_params[optimizer]),
            ssrl=ssrl,
            model_output_partial=(experiment['network_path_partial']
                                  if LOAD_PARTIAL else None),
            dataloaders=dataloaders_dict,
            criterion=criterion,
            num_epochs=num_epochs,
            device=torchdevice,
            early_stopping=experiment['early_stopping'],
        )

    for optimization_metric in network_path_dict:
        if is_alternative_loss and optimization_metric != 'LOSS':
            logger.warning(
                f'Best model for {optimization_metric} is not being saved, as '
                f'this metric is not useful along with {loss_name} loss '
                'function.'
            )
            continue

        logger.debug(f'Saving weights for {optimization_metric}.')
        assert optimization_metric in model_weights, \
            (optimization_metric, model_weights.keys())
        path = rocklongfiles.shortfile(network_path_dict[optimization_metric])
        logger.debug(f'Saving in: {path}')
        torch.save(model_weights[optimization_metric], path)

    if not experiment['raw_dataset']:
        for part in image_datasets:
            image_datasets[part].close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='[rocklib] finetuned_extraction.py')
    parser.add_argument(
        '--partition',
        type=int,
        help='Partition to run feature extraction for.  Should be an integer.',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--device',
        type=int,
        help='Device to run feature extraction for.  Should be an integer.',
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    envconfig.loggerGenFileHandler(logger)
    logger.debug(f'Batch size: {BATCH_SIZE}')
    Experiment = rockexperiment.Experiment
    Experiment.experiment = 'finetune'
    Experiment.update(args)

    assert len(Experiment) == 1, \
        ('Experiments definition for this module should contain a single '
         'experiment, as it will potentially use GPUs ({} experiments).'
         .format(len(Experiment)))

    Experiment(finetune, send_dict=True)
