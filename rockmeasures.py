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


"""The functions here accept `pred` to be two dimensional, in case multiple
labels are provided for a top k evaluation.

"""


import numpy as np


def acc(labels, pred):
    assert isinstance(labels, np.ndarray), type(labels)
    assert isinstance(pred, np.ndarray), type(pred)
    assert labels.shape[0] == pred.shape[0], (labels.shape, pred.shape)
    if len(pred.shape) > 1:
        ret = sum([x in y for x, y in zip(labels, pred)]) / labels.shape[0]
    else:
        ret = (labels == pred).sum() / labels.shape[0]
    return ret


def na(labels, pred):
    assert isinstance(labels, np.ndarray), type(labels)
    assert isinstance(pred, np.ndarray), type(pred)
    assert labels.shape[0] == pred.shape[0], (labels.shape, pred.shape)
    accuracies = []
    for label in set(labels):
        pos = labels == label
        v = sum([label in p
                 if isinstance(p, np.ndarray)
                 else label == p
                 for label, p in zip(labels[pos], pred[pos])]) / pos.sum()
        accuracies.append(v)
    # print(accuracies)
    ret = np.mean(accuracies)
    return ret
