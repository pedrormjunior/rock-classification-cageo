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
import csv
import envconfig


FILENAME_CLASSES = os.path.join(envconfig.DATA_PREFIX, 'CLASSES.txt')
FILENAME_WELLS = os.path.join(envconfig.DATA_PREFIX, 'WELLS.txt')
FILENAME_IGNORED_FILEPATHS = os.path.join(
    envconfig.DATA_PREFIX, 'IGNORED_FILEPATHS.txt')
FILENAME_NUMBER_TO_WELL = os.path.join(
    envconfig.DATA_PREFIX, 'NUMBER_TO_WELL.csv')
FILENAME_DATASET_VERSIONS = os.path.join(
    envconfig.DATA_PREFIX, 'DATASET_VERSIONS.csv')
FILENAME_LABELS_ASSOC = os.path.join(
    envconfig.DATA_PREFIX, 'LABELS_ASSOCIATION.csv')
FILENAME_LABELS_MANUAL = os.path.join(
    envconfig.DATA_PREFIX, 'LABELS_MANUAL.csv')


"""List of rock labels we are dealing with so far.  This list should be kept in
the same order in future modifications, so the data that have already been
generated in the past continues to be valid.  See also `rock_label_to_num` and
`rock_num_to_label` below.

"""
rock_association = [x.strip() for x in open(FILENAME_CLASSES).readlines()]

rock_label_to_num = {label: num for num, label in enumerate(rock_association)}
rock_num_to_label = {num: label for label, num in rock_label_to_num.items()}


"""List of wells we are dealing with so far.  This list should be kept in order
in future modifications, i.e., if a new well is going to be added herein, it
must be appended at the end, so the data that have already been generated in
the past continues to be valid.  See also `well_name_to_num` and
`well_num_to_name` below.

"""
wells = [x.strip() for x in open(FILENAME_WELLS).readlines()]

well_name_to_num = {name: num for num, name in enumerate(wells)}
well_num_to_name = {num: name for name, num in well_name_to_num.items()}


"""List of filepaths for the images to be ignored."""
ignored_filepaths = [
    x.strip() for x in open(FILENAME_IGNORED_FILEPATHS).readlines()]


def read_csv(filename, header):
    with open(filename) as fd:
        content = list(csv.reader(fd))
    assert content[0] == header, content[0]
    content = content[1:]
    return content


# Some filenames in the dataset contains some numbers associated to the well.
# The file FILENAME_NUMBER_TO_WELL specifies this association, that is useful
# later for getting the class of an example based on its filepath.
number_to_well = {
    number: well
    for number, well in read_csv(
            FILENAME_NUMBER_TO_WELL,
            ['number', 'well'],
    )
}


# We received the dataset in multiple parts, that we name here as "versions".
# Each part/version is located at one different directory inside `./data/`.
# The association between the version and the directory is specified in
# FILENAME_DATASET_VERSIONS.
dataset_versions = {
    int(version): dirname
    for version, dirname in read_csv(
            FILENAME_DATASET_VERSIONS,
            ['version', 'dirname'],
    )
}


# Reading the labels association.  The FILENAME_LABELS_ASSOC should be a CSV
# containing the well, the depth the photos was taken and the corresponding
# label.  From these data, we create here a `labels_association` and is a
# dictionary which keys are the wells and the values are a second type of
# dictionary.  This second type of dictionary maps the depth as key to the
# label number specified by `rock_label_to_num` as value.
labels_association = {}
for well, depth, label_name in read_csv(
        FILENAME_LABELS_ASSOC,
        ['well', 'depth', 'label'],
):
    if well not in labels_association:
        labels_association[well] = {}
    labels_association[well][int(depth)] = (rock_label_to_num[label_name]
                                            if label_name else None)

labels = labels_association

for well in wells:
    if well not in labels:
        labels[well] = None


# It associates each filename marked above as 'test' into a number
# order.  This information is mainly used for human classification.
assoc_manual_classification = {
    filepath: int(index)
    for filepath, index in read_csv(
            FILENAME_LABELS_MANUAL,
            ['filepath', 'index'],
    )
}
