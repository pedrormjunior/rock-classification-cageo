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


"""Perform classification on the extracted features."""


from typing import List, Dict
import random
import argparse

import itertools as it
import numpy as np

import sklearn.ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import parmapper as pm
import rocklib
import rocklibdata
import rockmeasures
import rocklongfiles

import rockresults
from rocklib import logger
import envconfig

import experiments.main as rockexperiment


SAVE_PREDICTS = True
SAVE_RESULTS = True

envconfig._makedir(envconfig.results_dir)
res = rockresults.Results(envconfig.results_filename)
predres = rockresults.Results(envconfig.predicts_filename)


class FeatureDimensionError(ValueError):
    """Exception pretty much for Softmax classifier."""


def softmax(vector):
    """Simply calculates the Softmax function."""
    e = np.exp(vector)
    ret = e / e.sum(axis=-1, keepdims=True)
    return ret


class Softmax():
    """Softmax classifier."""

    def __init__(self, classes, quant_classes):
        self.classes = classes
        self.quant_classes = quant_classes
        self.labels_to_use = classes[:quant_classes]

    def fit(self, features, _):
        """Fit the classifier.  Fitting soft Softmax means doing nothing, just
        asserting given data is as expected for a classifier.

        """
        if features.shape[1] > len(self.classes):
            raise FeatureDimensionError((features.shape,
                                         self.quant_classes))

    def predict_proba(self, features):
        """Predict probabilities."""
        if features.shape[1] > len(self.classes):
            raise FeatureDimensionError((features.shape,
                                         self.quant_classes))

        indexes = np.array([self.classes.index(x)
                            for x in self.labels_to_use])
        ret = features[:, indexes]
        ret = softmax(ret)
        return ret


def get_classifier_model(classifier: str,
                         classes: List[int],
                         quant_classes: int,
                         ):
    """`classes` and `quant_classes` are required only for Softmax classifier,
    which needs know which portion of the feature vector on which to perform
    the Softmax calculation.

    """
    assert isinstance(classifier, str), type(classifier)

    if classifier.endswith('nn') and classifier[:-2].isnumeric():
        # Using the nearest neighbor classifier.
        n_neighbors = int(classifier[:-2])
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm='brute',
            metric='euclidean',
        )

    elif classifier == 'svm':
        model = SVC(
            gamma='auto',
            probability=True,
            decision_function_shape='ovo',
        )

    elif classifier == 'rf':
        model = sklearn.ensemble.RandomForestClassifier()

    elif classifier == 'softmax':
        model = Softmax(classes, quant_classes)

    else:
        raise Exception('Unknown classifier: {}'.format(classifier))

    return model


def normalize_features(data, scale=None):
    assert isinstance(data, np.ndarray), type(data)
    assert len(data.shape) == 2, data.shape
    assert scale is None or (isinstance(scale, tuple) and len(scale) == 2), \
        scale

    if scale:
        datamin, diff = scale
    else:
        datamin = data.min(axis=0)
        diff = data.max(axis=0) - datamin

    ret = (data - datamin) / diff

    if scale:
        return ret
    else:
        return ret, (datamin, diff)


def classify(key,
             quant_classes,
             classes,
             augment_classification,
             classifier,
             topk,
             normalize,
             features_file_part_train_lst,
             features_file_part_val_lst,
             features_file_part_test_lst,
             override: bool,
             experiment: Dict[str, str],
             ):
    """Given the parameters of a classification experiment, perform it and save
    the results.

    """
    assert isinstance(quant_classes, int), type(quant_classes)
    assert quant_classes >= 2, quant_classes
    assert isinstance(classes, list), type(classes)
    assert all(isinstance(x, int) for x in classes), classes
    assert isinstance(augment_classification, bool), \
        type(augment_classification)

    assert isinstance(classifier, str), type(classifier)
    assert isinstance(topk, int), type(topk)
    assert topk > 0, topk
    assert isinstance(normalize, bool), type(normalize)
    assert isinstance(override, bool), type(override)

    logger.info('Results will be stored with KEY on '
                f'{envconfig.results_filename}.  KEY: {key}')

    if key in res and key in predres:
        if override:
            logger.warning('OVERRIDE previous key.')
        else:
            logger.warning('This classification were performed before.  '
                            'So, skipping a new one.')
            return

    # Preparing the dataset.
    assert len(features_file_part_train_lst) == 1, \
        ('TODO: It needs the implementation for multi-scale training, however '
         'it is deprecated, as a better approach seems to be implementing '
         "multi-scale through PyTorch's data loader.  See "
         'https://pytorch.org/tutorials/beginner/finetuning_torchvision_models'
         '_tutorial.html#load-data '
         'for more details.')
    assert len(features_file_part_val_lst) == 1, features_file_part_val_lst
    assert len(features_file_part_test_lst) == 1, features_file_part_test_lst
    shelve_extension = '.db'
    assert features_file_part_train_lst[0].endswith(shelve_extension)
    assert features_file_part_val_lst[0].endswith(shelve_extension)
    assert features_file_part_test_lst[0].endswith(shelve_extension)
    map_file = {
        'train': rocklongfiles.shortfile(
            features_file_part_train_lst[0][:-len(shelve_extension)]) + shelve_extension,
        'val': rocklongfiles.shortfile(
            features_file_part_val_lst[0][:-len(shelve_extension)]) + shelve_extension,
        'test': rocklongfiles.shortfile(
            features_file_part_test_lst[0][:-len(shelve_extension)]) + shelve_extension,
    }
    del shelve_extension

    image_datasets = {
        part: rocklib.RockDataset(
            filename=map_file[part],
            classes=classes,
            datatype='features',
        )
        for part in map_file
    }
    del map_file

    features = {
        part: [image_datasets[part][i]
               for i in range(len(image_datasets[part]))]
        for part in image_datasets
    }

    def get_data_as_numpy(lst):
        def get_info(features_metainfo):
            features, label, metainfo = features_metainfo
            filepath = metainfo['path']
            # label = rocklib.filename_to_label(filepath)
            well = rocklibdata.well_name_to_num[
                rocklib.filename_to_well(filepath)]
            depth = rocklib.filename_to_depth(filepath)
            dirtype = rocklib.filename_to_dirtype(filepath)
            return features, label, well, depth, dirtype

        return {
            key: data
            for key, data in zip(
                    ['features', 'label', 'well', 'depth', 'dirtype'],
                    map(np.array, zip(*map(get_info, lst))),
            )
        }

    features_info = {
        part: get_data_as_numpy(features[part])
        for part in features
    }
    # Now, limiting the experiment to the first `quant_classes`, i.e., use only
    # data (feature vectors, etc.) referring to those classes.
    for part in features_info:
        indexes = features_info[part]['label'] < quant_classes
        for dictkey in features_info[part]:
            features_info[part][dictkey] = \
                features_info[part][dictkey][indexes]

    model = get_classifier_model(classifier, classes, quant_classes)
    logger.debug(f'Model: {model}')

    if experiment['normalize']:
        logger.debug('Normalizing features.')
        features_info['train']['features'], scale = normalize_features(
            features_info['train']['features']
        )
        features_info['val']['features'] = normalize_features(
            features_info['val']['features'],
            scale,
        )
        features_info['test']['features'] = normalize_features(
            features_info['test']['features'],
            scale,
        )

    logger.debug('Fitting classifier.')
    try:
        model.fit(
            features_info['train']['features'],
            features_info['train']['label'],
        )
    except FeatureDimensionError:
        # Occurs for softmax when the dimensionality of the feature
        # vectors is bigger than the number of classes.
        pass

    logger.debug('Predicting for train, val, and test.')
    pred_probs = {
        part: model.predict_proba(features_info[part]['features'])
        for part in features_info
    }
    logger.debug('Predicting done.')

    logger.debug('Performing top-k analysis')
    labels_to_use = np.array(classes[:quant_classes])

    def top_k(probs_matrix):
        def top_k_per_vector(probs):
            assert topk >= 1, topk
            ret = [x[1] for x in sorted(zip(probs, range(len(probs))),
                                        reverse=True)][:topk]
            ret = labels_to_use[ret]
            return ret
        return np.array([top_k_per_vector(probs) for probs in probs_matrix])

    pred_top = {
        part: top_k(pred_probs[part])
        for part in pred_probs
    }

    # At this point, predicted labels are the original labels instead of the
    # index-based labels (i.e., 0, ..., N-1).  Then should also convert the
    # ground-truth labels to the original labels.
    for part in features_info:
        features_info[part]['label'] = \
            labels_to_use[features_info[part]['label']]

    # The data on the internal `zip` will be used to group feature vectors from
    # the same image.  The combination of those four keys is enough information
    # to separate each image.
    logger.debug('Grouping feature vectors for fusion.')
    data = {
        part: list(zip(zip(features_info[part]['well'],
                           features_info[part]['depth'],
                           features_info[part]['dirtype'],
                           features_info[part]['label']),
                       pred_top[part]))
        for part in pred_top
    }

    def group_features(data):
        groups = []
        uniquekeys = []
        data = [(x[0], list(x[1])) for x in data]
        data_sorted = sorted(data)
        data_sorted = [(x[0], np.array(x[1])) for x in data_sorted]
        for key, group in it.groupby(data_sorted, lambda x: x[0]):
            groups.append(np.array([x[1] for x in group]))
            uniquekeys.append(key)
        return {'groups': groups,
                'uniquekeys': uniquekeys}

    data_groups = {
        part: group_features(data[part])
        for part in data
    }

    def max_vote(group):
        assert len(group) >= 1, len(group)
        group = group.flatten()
        counts = sorted([((group == x).sum(), x) for x in set(group)],
                        reverse=True)
        ret = [x[1] for x in counts[:topk]]
        return ret

    logger.debug('Performing max-voting fusion.')
    votes = {
        part: {
            'votes': np.array([
                max_vote(group)
                for group in data_groups[part]['groups']
            ]),
            'labels': np.array([
                label
                for _, _, _, label in data_groups[part]['uniquekeys']
            ]),
        }
        for part in data_groups
    }

    for part in votes:
        assert len(votes[part]['votes']) == len(votes[part]['labels'])
    for part in pred_top:
        assert len(pred_top[part]) == len(features_info[part]['label'])
    preds = {
        part: {
            'img': {'true': votes[part]['labels'],
                    'pred': votes[part]['votes']},
            'pat': {'true': features_info[part]['label'],
                    'pred': pred_top[part]},
        }
        for part in votes
    }

    results_dict = {
        part: {
            metric: {
                imgpat: func(preds[part][imgpat]['true'],
                             preds[part][imgpat]['pred'])
                for imgpat in preds[part]
            }
            for metric, func in zip(['acc', 'na'],
                                    [rockmeasures.acc, rockmeasures.na])
        }
        for part in preds
    }

    if SAVE_PREDICTS:
        logger.debug('Saving predictions.')
        predres[key] = preds
    else:
        logger.warning('Predictions are not being saved.')

    if SAVE_RESULTS:
        logger.debug('Saving results.')
        res[key] = results_dict
    else:
        logger.warning('Results are not being saved.')

    logger.info(f'Results for {key}: {results_dict}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[rocklib] classification.py')
    parser.add_argument(
        '--partition',
        type=int,
        help='''Partition to run classification for.  Should be an integer.''',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--model_name',
        type=str,
        help='''Name of the network model used for feature extraction.  Should
        be a string.''',
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    random.seed(0.8640890499471714)
    Experiment = rockexperiment.Experiment
    Experiment.experiment = 'classification'
    Experiment.update(args)
    Experiment.shuffle()
    Experiment(classify, mapf=pm.parmap, send_dict=True)
    # Experiment(classify, mapf=map, send_dict=True)
    logger.info('READY')
