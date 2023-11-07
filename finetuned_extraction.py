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


from typing import Union, Dict
import argparse

import os
import rocklib
import shelve
import envconfig
import rocklongfiles

import experiments.main as rockexperiment

import finetune
import torch
from rocklib import logger


def extraction(classes,
               segmentation: bool,
               stride,
               model_name,
               raw_dataset: bool,  # DEPRECATED
               network_path,
               layer_name,
               part,
               patch_file_part,
               features_file_part,
               device: Union[int, str],
               override: bool,
               experiment: Dict[str, str],
               ) -> None:

    assert isinstance(classes, list), type(classes)
    assert isinstance(segmentation, bool), type(segmentation)
    assert all(isinstance(x, int) for x in classes), classes
    assert isinstance(model_name, str), type(model_name)
    assert isinstance(network_path, str), type(network_path)
    assert isinstance(layer_name, (str, type(None))), type(layer_name)
    assert isinstance(raw_dataset, bool), type(raw_dataset)
    assert isinstance(patch_file_part, str), type(patch_file_part)
    assert part in ['train', 'val', 'test', ], part
    assert isinstance(features_file_part, str), type(features_file_part)

    logger.info(f'Feature vectors will be written in: {features_file_part}')
    logger.debug(f'device: {device}')

    if os.path.exists(features_file_part):
        if override:
            logger.warning('OVERRIDE previous features.')
        else:
            logger.warning('These feature vectors were generated before.  '
                           'So, skipping a new generation.')
            return

    model_ft, input_size, finetuned = finetune.load_pretrained_model(
        model_name,
        num_classes=len(classes),
        feature_extract=True,
        network_path=rocklongfiles.shortfile(network_path),
    )
    assert finetuned, \
        ('The network model tried to be used were not fine-tuned before for '
         'this specific problem.  Aborting execution!')

    model_ft.eval()

    data_transforms = finetune.get_data_transforms(
        input_size=input_size,
        segmentation=experiment['segmentation'],
        raw_dataset=False,
        stride=experiment['stride'],
        feature_extraction=True,  # Not considerd when raw_dataset=False
    )

    image_dataset = rocklib.RockDataset(
        filename=experiment['patch_file_part'],
        classes=experiment['classes'],
        transforms=data_transforms[part],
    )

    logger.debug(f'Length of dataset: {len(image_dataset)}')

    dataloader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=finetune.BATCH_SIZE,
        shuffle=False,
        num_workers=10,
    )

    torchdevice = finetune.get_torch_device(device)
    model_ft = model_ft.to(torchdevice)

    features = finetune.model_extract(
        model_ft,
        dataloader,
        torchdevice,
        layer_name,
    )

    # The most important information in the `metainfo` is the path for the
    # original file, that can be used later to acquire information of well,
    # depth, label, etc.
    metainfo = image_dataset.get_metainfo()
    assert len(features) == len(metainfo)

    image_dataset.close()

    shelve_extension = '.db'
    assert features_file_part.endswith(shelve_extension), features_file_part
    with shelve.open(
            rocklongfiles.shortfile(
                features_file_part[:-len(shelve_extension)])) as fd:
        for i in envconfig.tqdm(range(len(metainfo))):
            fd[f'{i}'] = {**metainfo[i],
                          **{'features': features[i].numpy()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[rocklib] finetuned_extraction.py')
    parser.add_argument(
        '--partition',
        type=int,
        help='''Partition to run feature extraction for.  Should be an integer.''',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--device',
        type=int,
        help='''Device to run feature extraction for.  Should be an integer.''',
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    envconfig.loggerGenFileHandler(logger)
    Experiment = rockexperiment.Experiment
    Experiment.experiment = 'finetuned_extraction'
    Experiment.update(args)
    Experiment(extraction, send_dict=True)
