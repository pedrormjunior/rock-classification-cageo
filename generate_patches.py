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


from typing import List, Tuple, Dict
import argparse

import os

import numpy as np

import rocklib
import rocklibdata
from rocklib import logger

import shelve
import torch
from PIL import Image

import envconfig

import split_dataset as split
import experiments.main as rockexperiment


envconfig.config()

SAVE_PATCHES: bool = True
THRESHOLD_PIXELS: int = 1700


def generate(classes: List[int],
             partition: int,
             versions: List[int],
             patch_size: Tuple[int, int],
             stride: int,
             segmentation: bool,
             parts: List[str],
             patch_file_part_lst: List[str],
             override: bool,
             experiment: Dict[str, str],
             ) -> None:
    """Generate the shelve dictionaries with patches from the images that are
    being considered.

    Parameters
    ----------
    classes:
        List of integers referring to the classes to select for usage.  See
        `rocklib.py` for the indexes of the classes.

    partition:
        An integer specifying the partition of the dataset to use.

    versions:
        List of integers referring to the datasets to read data from.  See
        `rocklib.py` for the datasets available.

    patch_size:
        A tuple of size 2 specifying the size the patches should be extracted
        on, in pixels.

    stride:
        The stride to use on the sliding window through the process of patch
        extraction.

    segmentation:
        Whether to perform segmentation prior to patch extraction.  The
        segmentation have the purpose of removing the background borders of the
        core plug in order to avoid having patches that contains background
        pixels.  For some images the segmentation can be very aggressive,
        making most of the patches of the images to be ignored.

    parts:
        Should be a list containing, e.g., ['train', 'val', 'test'].

    patch_file_part_list : str
        A list of shelve file names on which to write the generated patches for
        each part in "parts".

    override : bool
        Whether to override previous computation in case it was already
        performed.

    """

    assert isinstance(classes, list), type(classes)
    assert all(isinstance(x, int) for x in classes), classes
    assert all(x >= 0 for x in classes), classes
    assert isinstance(partition, int), type(partition)
    assert isinstance(versions, list), type(versions)
    assert all(isinstance(x, int) for x in versions), versions
    assert versions == sorted(versions), versions
    assert isinstance(stride, int), type(stride)
    assert isinstance(patch_size, tuple), type(patch_size)
    assert isinstance(segmentation, bool), type(segmentation)
    assert isinstance(parts, list), type(parts)
    assert isinstance(patch_file_part_lst, list), type(patch_file_part_lst)
    assert isinstance(override, bool), type(override)

    logger.info('Patches will be saved in: {}'.format(
        {part: file_part
         for part, file_part in zip(parts, patch_file_part_lst)}
    ))

    if all(os.path.exists(patch_file_part)
           for part, patch_file_part
           in zip(parts, patch_file_part_lst)):
        if override:
            logger.warning('OVERRIDE: All previous files will be overridden.')
        else:
            logger.warning('All files were already generated.  '
                           'So, skipping a new generation.')
            return

    shelve_extension = '.db'
    assert all(filename.endswith(shelve_extension)
               for filename in patch_file_part_lst)
    features_part = {
        part: shelve.open(patch_file_part[:-len(shelve_extension)])
        for part, patch_file_part
        in zip(parts, patch_file_part_lst)
    }
    del shelve_extension

    if not SAVE_PATCHES:
        logger.warning('Not saving patches')

    counters = {part: 0 for part in features_part}
    for filename in split.partitions['filenames']:
        logger.debug(filename)
        img = Image.open(filename).convert('RGB')
        img_np = np.asarray(img)
        assert sum(img_np.shape[:2]) >= THRESHOLD_PIXELS

        well = rocklib.filename_to_well(filename)
        wellnum = rocklibdata.well_name_to_num[well]
        label = rocklib.filename_to_label(filename)
        depth = rocklib.filename_to_depth(filename)
        key = (wellnum, label, depth)
        part = split.partitions['partitions'][partition][key]

        img = torch.from_numpy(img_np.copy())  # A copy to supress a warning

        missinglabel = (rocklibdata.labels[well] and
                        depth not in rocklibdata.labels[well])
        assert not missinglabel
        del missinglabel
        assert label in classes

        imgseg = 1 - rocklib.load_mask_from_filename(filename)
        imgseg = torch.from_numpy(imgseg)
        assert img_np.shape[:2] == imgseg.shape, (img_np.shape, imgseg.shape)

        def unfold(img):
            return (img
                    .unfold(0, patch_size[0], stride)
                    .unfold(1, patch_size[1], stride))

        patches = unfold(img)
        masks = unfold(imgseg)
        assert patches.shape[:2] == masks.shape[:2]
        assert patches.shape[-2:] == masks.shape[-2:]

        assert part is not None, (filename, well, label, depth)
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                mask = masks[i][j]
                if segmentation and mask.sum() > 0:
                    continue
                patch = patches[i][j]

                if SAVE_PATCHES:
                    features_part[part][f'{counters[part]}'] = {
                        # 'dataset': idx,
                        'path': filename,
                        'patch': patch.numpy(),
                    }
                counters[part] += 1

    logger.debug(counters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='[rocklib] generate_patches.py')
    parser.add_argument(
        '--partition',
        type=int,
        help='''Partition to run classification for.  Should be an integer.''',
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    Experiment = rockexperiment.Experiment
    Experiment.experiment = 'generate_patches'
    Experiment.update(args)
    Experiment(generate, send_dict=True)
