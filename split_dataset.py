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


import os
import sys
import rocklib
import rocklibdata
import torchvision
import numpy as np
import random
import shelve
import copy
import envconfig


VERSIONS = [1, 2, 3]
CLASSES = [3, 5, 6, 1, 11, 9, 10]
THRESHOLD_PIXELS: int = 1700
QUANT_PARTITIONS: int = 16
PERC_TRAIN = 0.6
PERC_TEST = 0.24                # VAL: 0.16
envconfig._makedir(envconfig.records_dir)
PARTITIONS_FILENAME = envconfig.partitions_filename


if __name__ == '__main__':
    random.seed(0.5699905559313857)

    datasets = {
        version: torchvision.datasets.ImageFolder(folder)
        for version, folder in rocklib.get_dataset_folders(VERSIONS).items()
    }

    filenames_employed = []
    keys_employed = []
    counters = {
        'threshold': 0,
        'label': 0,
        'ignored': 0,
    }
    for idx in datasets:
        # logger.debug(f'Dataset {idx}')
        for filename, data in (
                zip(datasets[idx].imgs, datasets[idx])
        ):
            filename, data = filename[0], np.array(data[0])
            if filename.endswith('_mask.png'):
                continue

            if sum(data.shape[:2]) < THRESHOLD_PIXELS:
                rocklib.logger.debug(f'THRESHOLD: Image {filename} '
                                     f'{data.shape} is too small')
                counters['threshold'] += 1
                continue

            key = rocklib.filename_to_key(filename)
            _, label, _ = key

            if label not in CLASSES:
                rocklib.logger.debug(f'LABEL: Image {filename} has non valid '
                                     f'label {label}')
                counters['label'] += 1
                continue

            # 20221010
            if filename in rocklibdata.ignored_filepaths:
                # Those are the images that get accepted by the size criterion
                # checked above, however, those images are of broken cores.
                # Based on the segmentation mask for those images, no patch can
                # be extracted for them which make them useless for
                # experimentation.  So they are being ignored.
                rocklib.logger.debug(f'IGNORED: Image {filename} is ignored '
                                     'due to its mask')
                counters['ignored'] += 1
                continue

            keys_employed.append(key)
            filenames_employed.append(filename)

    rocklib.logger.debug(f'Not used: {counters}')
    rocklib.logger.debug(f'keys_employed: {len(keys_employed)}')
    rocklib.logger.debug(f'filenames_employed: {len(filenames_employed)}')

    def generate_partition():
        keys = copy.deepcopy(keys_employed)
        keys = list(set(keys))
        random.shuffle(keys)
        quant_train = int(PERC_TRAIN * len(keys))
        quant_test = int(PERC_TEST * len(keys))
        train = keys[:quant_train]
        test = keys[quant_train:quant_train+quant_test]
        val = keys[quant_train+quant_test:]
        assert len(train) + len(test) + len(val) == len(keys), \
            (len(train), len(test), len(val), len(keys))
        assert len(set(train + test + val)) == len(keys)
        train_labels = [label for _, label, _ in train]
        test_labels = [label for _, label, _ in test]
        val_labels = [label for _, label, _ in val]

        def print_labels(string, labels):
            lst = [labels.count(label) for label in CLASSES]
            print(f'{string} {lst} '
                  f'(min={min(lst)}) '
                  f'(sum={sum(lst)}) ')
        print('Quant core plugs:')
        print_labels('train', train_labels)
        print_labels('test', test_labels)
        print_labels('val', val_labels)
        return {**{x: 'train' for x in train},
                **{x: 'test' for x in test},
                **{x: 'val' for x in val}}

    partitions = shelve.open(PARTITIONS_FILENAME, flag='n', writeback=True)
    partitions['filenames'] = filenames_employed
    partitions['partitions'] = {}
    for partition_num in range(QUANT_PARTITIONS):
        # print(str(partition_num))
        partition = generate_partition()

        partitions['partitions'][partition_num] = {}
        filenames_count = {part: {label: 0 for label in CLASSES}
                           for part in ['train', 'test', 'val']}
        for filename in filenames_employed:
            key = rocklib.filename_to_key(filename)
            part = partition[key]
            partitions['partitions'][partition_num][key] = part
            _, label, _ = key
            filenames_count[part][label] += 1
            del key, part, label, filename

        filenames_count = {part: [filenames_count[part][label]
                                  for label in CLASSES]
                           for part in filenames_count}
        print('Quant images:')
        for part in filenames_count:
            print(f'{part} {filenames_count[part]} '
                  f'(min={min(filenames_count[part])}) '
                  f'(sum={sum(filenames_count[part])})')
        print()

    partitions.close()
    # logger.info(f'Shelve file of partitions: {PARTITIONS_FILENAME}')

    partitions = shelve.open(PARTITIONS_FILENAME)
    dic = {}
    for partition_num in partitions['partitions']:
        for filename in partitions['filenames']:
            if filename not in dic:
                dic[filename] = []
            key = rocklib.filename_to_key(filename)
            dic[filename].append(partitions['partitions'][partition_num][key])
    partitions.close()

    for filename in dic:
        set_parts = set(dic[filename])
        if len(set_parts) == 1:
            print(filename)
            print(set_parts)

else:
    assert os.path.basename(sys.argv[0]) == os.path.basename(__file__) or \
        os.path.exists(PARTITIONS_FILENAME + '.db'), \
        (f"Before importing this module ({__file__}), you need to first "
         f"generate the file {PARTITIONS_FILENAME + '.db'}; run `make "
         "split_dataset` for this purpose.")
    if os.path.exists(PARTITIONS_FILENAME + '.db'):
        fd = shelve.open(PARTITIONS_FILENAME)
        partitions = dict(fd)
        fd.close()
