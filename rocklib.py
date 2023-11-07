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


"""Core code containing main components."""


from typing import Optional, Any, List

import random
import os
import shelve
import torch
import envconfig
import logging
from PIL import Image
import split_dataset as split
import numpy as np

import rocklibdata


SHELVE_EXTENSION = '.db'


# Logger
logger = logging.getLogger('RockLogger')
envconfig.coloredlogs.install(
    logger=logger,
    level='DEBUG',
    fmt=envconfig.LOGGERFMT,
)


class RepeaterSampler(torch.utils.data.Sampler):
    """A Sampler for repeating the pass through the dataset multiple times.
    Useful when the training dataset is too small for an epoch, so it is better
    to train over it multiple times before evaluating on the validation set.

    """

    def __init__(self,
                 size: int,
                 times: int,
                 shuffle: bool = False,
                 generator=None):
        """Initializer for the RepeaterSampler.

        size:
            The size of the dataset to be sampled.

        times:
            How many times to repeat over the dataset.

        shuffle:
            Whether to shuffle the provided indexes.

        generator:
            Used when shuffle=True.  Used for randomly generating the shuffle.
        """

        assert isinstance(size, int), type(size)
        assert size > 0, size
        assert isinstance(times, int), type(times)
        assert times > 0, times
        assert isinstance(shuffle, bool), type(shuffle)

        self._size = size
        self._times = times
        self._shuffle = shuffle
        self._generator = generator

    def __iter__(self):
        return ((v
                 for v in torch.randint(
                         high=self._size,
                         size=(len(self),),
                         dtype=torch.int64,
                         generator=self._generator,
                 ).tolist())
                if self._shuffle else
                (v % self._size
                 for v in range(len(self))))

    def __len__(self):
        return self._size * self._times


class PatchExtraction(object):
    """Uniformly extract patches from the given torch image based on a provided
stride.  It assumes format is CHW.

    Args:
        stride (int): stride to use on patch extraction.

    """

    def __init__(self, size, stride):
        assert isinstance(size, (int, tuple)), type(size)
        assert isinstance(stride, int), type(stride)
        assert stride > 0, stride
        if isinstance(size, int):
            self._size = (size, size)
        else:
            assert len(size) == 2, size
            self._size = size
        self._stride = stride

    def __call__(self, img):
        """Args:
            img (Tensor): Image from which to extract patches.

        Returns: 4D
            Tensor: Format NCST, in which N is the number of patches that could
            be extracted and S and T are the size provided to the initializer.

        """

        patches = (img
                   .unfold(1, self._size[0], self._stride)
                   .unfold(2, self._size[1], self._stride)
                   .contiguous()
                   .view(3, -1, self._size[0], self._size[1])
                   .permute(1, 0, 2, 3))
        return patches

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(size={self._size}, stride={self._stride})'


# TODO: Implement a general Dataset class that could instantiate either the
# `RockDataset` or the `RockDatasetRaw`.  It will avoid repeating conditionals
# on `finetune` and `finetuned_extraction` modules.
class Dataset(torch.utils.data.Dataset):
    """Implement the base features of a dataset for rock image classification.

    """

    def __init__(self,
                 classes: List[int],
                 transforms: Optional[Any] = None,
                 ):
        """Initialize a base Dataset for rock classification.

        classes:
            Classes should be a list containing all labels within dataset
            `filename`; it is used to shift the labels to the range [0, N-1]
            for training networks, in which N is the number of classes on the
            dataset.

        transforms:
            A PyTorch structure to transform the main data extracted from the
            dataset before providing it.

        """

        self.classes = classes
        self.transforms = transforms

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value):
        assert isinstance(value, list), type(value)
        assert len(value) > 0, value
        assert all(isinstance(x, int) and x >= 0
                   for x in value)
        assert len(value) == len(set(value)), \
            (value, set(value))
        self._classes = value

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, value):
        logger.debug(f'Type of transforms value: {type(value)}')
        self._transforms = value


class RockDatasetRaw(Dataset):
    """Implement a dataset consisting of the raw full-resolution images made
    available by Shell as is.  It does not provide patches but the
    variable-sized original images.  It is intended to be used along with
    on-the-fly random crops processing during training of networks through
    transforms (see documentation for `rocklib.RockDatasetRaw.__init__`).

    """

    def __init__(self,
                 partition: int,
                 part: str,
                 segmentation: bool,
                 loss_type: Optional[str] = None,
                 threshold: int = 1700,
                 keep_in_memory: bool = False,
                 **kwargs):
        """Initialize a RockDataset of full-resolution images.

        The initialization parses all datasets returned by
        `rocklib.get_dataset_folders` filtering out the images that are too
        small (according to `threshold`), the images for which we have no
        label, or the images that refer to a `part` other than the specified
        one.  Then, it prepares the index for the valid images.

        partition:
            An integer specifying the partition of the dataset to use.

        part:
            One of ['train', 'val', 'test'].

        segmentation : bool
            Whether to consider drill cores' masks.

        loss_type: One of ['triplet', 'contrastive'] or `None`.  Specifies the
            type of data to be provided, according to each type of loss.

        threshold:
            On the minimum size a image should have to be considered on the
            dataset.  It is given in number of pixels and is applied on H + W,
            in which H and W are respectively the height and width of the
            image.

        keep_in_memory:
            Whether to keep the loaded images in memory so that successive use
            of them will be faster.  It will work well for not-very-large
            datasets on a machine with reasonable amount of memory.

        **kwargs:
            Remaining arguments should be as required by `Dataset`.

        """

        super().__init__(**kwargs)
        self.part = part
        self._segmentation = segmentation
        self.threshold = threshold
        self.keep_in_memory = keep_in_memory
        self.loss_type = loss_type

        self._indexes = []
        """Keep the indexes of the filtered in images for O(1) access by
        `__getitem__`.  It will keep a tuple of the form (filename,
        torch_label, idx), in which idx is the index of the dataset.

        """

        self._labels = []
        """Keep the labels of the filtered in images for O(1) access by
        `__getitem__`.  It is mainly useful for obtaining negative or positive
        examples for both types of losses: triplet and contrastive.  At the
        end, it will be converted to Tensor.

        """

        if self.keep_in_memory:
            self._images = []
            if self._segmentation:
                self._masks = []

        for filename in split.partitions['filenames']:
            img = Image.open(filename).convert('RGB')

            assert sum(img.size[:2]) >= self.threshold
            key = filename_to_key(filename)
            well, label, depth = key
            assert label in self.classes, (label, self.classes)

            imgpart = split.partitions['partitions'][partition][key]
            if imgpart != self.part:
                continue

            torch_label = self.classes.index(label)
            self._indexes.append((filename, torch_label, ))
            self._labels.append(torch_label)
            if self.keep_in_memory:
                self._images.append(img)
                if self._segmentation:
                    mask = load_PIL_mask_from_filename(filename)
                    self._masks.append(mask)

        self._labels = torch.tensor(self._labels)

    def example(self, index):
        """It provides a valid image example on the index `idx` of one of the
        datasets.  For information of what is considered a "valid" image, see
        documentation for `rocklib.RockDatasetRaw.__init__`.

        index:
            The index of the valid image to get.

        Return
        ------

        A valid image with unknown height and width, as it is the raw dataset.

        """

        filename, label = self._indexes[index]
        if self.keep_in_memory:
            data = self._images[index]
            if self._segmentation:
                mask = self._masks[index]
        else:
            data = Image.open(filename).convert('RGB')
            if self._segmentation:
                mask = load_PIL_mask_from_filename(filename)

        if not self._segmentation:
            # TODO: Move this magic number to inside a transform instead.
            magic_number = 93       # Just remove the legend on the bottom part
            data = data.crop((0, 0, data.size[0], data.size[1] - magic_number))

        if self.transforms:
            if self._segmentation:
                data = self.transforms([data, mask])
            else:
                data = self.transforms(data)

        return data, label

    def _random_positive(self, label, index):
        """Randomly getting an example, other than the one at `index`, with
        label equals to `label`, i.e., a positive example.

        """

        mask = self._labels == label
        mask[index] = False
        index_pos = int(random.choice(torch.where(mask)[0]))
        datapos, _ = self.example(index_pos)
        return datapos

    def _random_negative(self, label):
        """Randomly getting an example with label different than `label`, i.e.,
        a negative example.

        """

        index_neg = int(random.choice(
            torch.where(self._labels != label)[0]
        ))
        dataneg, labelneg = self.example(index_neg)
        return dataneg, labelneg

    def __getitem__(self,
                    index: int,
                    ):
        """It provides an image on the `index` of one of the datasets or a
        triplet of images for which the images at `index` is the anchor or two
        images for the use with contrastive loss.  When `self.loss_type ==
        'triplet'`, it provides the triplet such that the first image on the
        provided tuple is the anchor, the second is the positive, and the third
        is the negative.  When `self.loss_type == 'contrastive'`, it provides
        two examples and the labels.  The last element of the provided tuple
        for both cases is the label of the image at `index`.  See documentation
        for `self.example` for more information.

        """

        data, label = self.example(index)

        if self.loss_type == 'triplet':
            datapos = self._random_positive(label, index)
            dataneg, _ = self._random_negative(label)

            return data, datapos, dataneg, label

        elif self.loss_type == 'contrastive':
            pairlabel = int(random.random() > 0.5)
            if pairlabel:
                data_other = self._random_positive(label, index)
            else:
                data_other, _ = self._random_negative(label)

            return data, data_other, pairlabel, label

        else:
            return data, label

    def __len__(self):
        """Return the length of the valid images."""
        return len(self._indexes)

    def labels(self):
        """Return the respective labels of the valid images."""
        return [label for _, label in self._indexes]

    def get_metainfo(self):
        """Return the respective meta information (original filepath, etc.) of
        the valid images.

        """

        def get_dict(i):
            dic = {
                # Those information is decided to be provided because they are
                # the same as the ones provided by `generate_patches`.  Of
                # those, pretty much only the 'path' will be useful later on.
                # 'dataset': self._indexes[i][2],
                'path': self._indexes[i][0],
            }
            return dic
        return [get_dict(i) for i in envconfig.tqdm(range(len(self._indexes)))]

    @property
    def part(self):
        return self._part

    @part.setter
    def part(self, value):
        assert isinstance(value, str), type(value)
        assert value in ['train', 'val', 'test'], value
        self._part = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        assert isinstance(value, int), type(value)
        assert value > 0, value
        self._threshold = value

    @property
    def keep_in_memory(self):
        return self._keep_in_memory

    @keep_in_memory.setter
    def keep_in_memory(self, value):
        assert isinstance(value, bool), type(value)
        self._keep_in_memory = value

    @property
    def loss_type(self):
        return self._loss_type

    @loss_type.setter
    def loss_type(self, value):
        assert isinstance(value, (str, type(None))), type(value)
        self._loss_type = value


class RockDataset(Dataset):
    """Implement a dataset consisting of the patches previously extracted and
    saved on a `shelve` structure.  It can also be used for features instead of
    patches if with the proper initialization (see documentation for
    `rocklib.RockDataset.__init__`).

    """

    def __init__(self,
                 filename: str,
                 datatype: str = 'patch',
                 **kwargs
                 ):
        """Initialize a RockDataset.

        filename:
            The dataset pointed through `filename` should be a `shelve` dataset
            containing either patches of the images or features extracted from
            patches.  `datatype` specifies which case to be assumed.

        datatype:
            Should be one of ['patch', 'features'], which is the key of the
            shelve dataset on which the main data is stored on.

        **kwargs:
            Remaining arguments should be as required by `Dataset`.

        """

        super().__init__(**kwargs)
        self.filename = filename
        self.shelve = shelve.open(self.filename)
        self.datatype = datatype

    def __getitem__(self, index):
        """It provides only the patch, when `self.datatype` is 'patch', or the
        feature vectors when `self.datatype` is 'features'.  When providing the
        patches, it ensure to return it in HWC mode, as it will probably be
        used with a transformer that changes from HWC to CHW.

        """

        data_dict = self.shelve[f'{index}']
        data = data_dict[self.datatype]
        if self.datatype == 'patch':
            data = data.transpose((1, 2, 0))
        if self.transforms:
            data = self.transforms(data)
        label = filename_to_label(data_dict['path'])
        label = self.classes.index(label)

        if self.datatype == 'patch':
            ret = (data, label)
        else:
            data_dict.pop(self.datatype)
            ret = (data, label, data_dict)

        return ret

    def __len__(self):
        return len(self.shelve)

    def labels(self):
        """Return a list of the labels of the dataset."""
        return [self[i][1] for i in range(len(self))]

    def get_metainfo(self):
        def get_dict(i):
            dic = self.shelve[f'{i}']
            dic.pop(self.datatype)
            return dic
        return [get_dict(i)
                for i in envconfig.tqdm(range(len(self.shelve)))]

    def close(self):
        self.shelve.close()
        self.shelve = None

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        assert isinstance(value, str), type(value)
        assert value.endswith(SHELVE_EXTENSION), value
        assert os.path.exists(value), value
        self._filename = value[:-len(SHELVE_EXTENSION)]

    @property
    def shelve(self):
        assert self._shelve is not None, \
            ('It seems someone closed the dataset and is still trying to use '
             'it.')
        return self._shelve

    @shelve.setter
    def shelve(self, value):
        assert isinstance(value, (shelve.DbfilenameShelf, type(None))), \
            type(value)
        self._shelve = value

    @property
    def datatype(self):
        return self._datatype

    @datatype.setter
    def datatype(self, value):
        self._valid_datatype_lst = ['patch', 'features']
        assert isinstance(value, str), type(value)
        assert value in self._valid_datatype_lst, value
        self._datatype = value


def filename_to_depth(filename):
    basename = os.path.basename(filename)
    depth = basename.split('_')[1].split('m')[0].strip()
    if len(depth.split('.')) == 1:
        depth += '.00'
    assert len(depth.split('.')[1]) in [1, 2], (depth, filename)
    if len(depth.split('.')[1]) == 1:
        depth += '0'
    assert len(depth.split('.')[1]) == 2
    depth = int(''.join(depth.split('.')))
    return depth


def filename_to_well(filename):
    assert isinstance(filename, str), type(filename)
    basename = os.path.basename(filename)
    number = basename.split('_')[0]
    assert number.isnumeric(), number
    if number not in rocklibdata.number_to_well:
        return None
    else:
        return rocklibdata.number_to_well[number]


def filename_to_label(filename):
    """Return the label starting from 0, or -1 when there is no label
    registered for `filename`.

    """

    well = filename_to_well(filename)
    depth = filename_to_depth(filename)
    ret = (rocklibdata.labels[well][depth]
           if well in rocklibdata.labels
           and rocklibdata.labels[well]
           and depth in rocklibdata.labels[well]
           and rocklibdata.labels[well][depth] is not None
           else -1)
    return ret


def filename_to_dirtype(filename):
    dirtype = ('top'
               if (filename.endswith('top.jpg') or
                   filename.endswith('top (2).jpg'))
               else 'elong')
    dirtype = {
        'elong': 0,
        'top': 1,
    }[dirtype]
    return dirtype


def filename_to_key(filename):
    well = rocklibdata.well_name_to_num[filename_to_well(filename)]
    label = filename_to_label(filename)
    depth = filename_to_depth(filename)
    return (well, label, depth)


def get_mask_filename(filename):
    assert filename.lower().endswith('.jpg')
    filenamemask = filename[:-4] + '_mask.png'
    return filenamemask


def binarize_mask(mask):
    # Assume a numpy array (or similar?)
    return (mask > 127).all(axis=2).astype('uint8')


def load_PIL_mask(filenamemask):
    return Image.open(filenamemask).convert('RGB')


def load_PIL_mask_from_filename(filename):
    filenamemask = get_mask_filename(filename)
    return load_PIL_mask(filenamemask)


def load_mask(filenamemask):
    mask = np.asarray(load_PIL_mask(filenamemask))
    mask = binarize_mask(mask)
    return mask


def load_mask_from_filename(filename):
    filenamemask = get_mask_filename(filename)
    mask = load_mask(filenamemask)
    return mask


def get_dataset_folders(versions=None):
    ret = {
        key: os.path.join(envconfig.DATA_PREFIX,
                          rocklibdata.dataset_versions[key])
        for key in rocklibdata.dataset_versions
    }
    return ret if versions is None else \
        {version: ret[version] for version in versions}
