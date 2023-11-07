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


"""This module provides a dictionary-like structure named Results that persist
the data.  It is heavily based on `shelve` module and behaves in a similar
fashion, however, with two main advantages:

        1) It is thread safe as each reading or writing employs the `filelock`
           module to open the persisted `shelve` file.

        2) It accept `tuple`s as keys.  Behind the scenes, it simply transform
           the `tuple` key to string and then employs this string as `shelve`
           key.

Usage example:

        1) Writing a single result:

                >>> from results import Results
                >>> res = Results('filename_with_results')
                >>> var1 = 1; var2 = 0; res[(var1, var2)] = \
                        {'result1': 1.0, 'result2': 0.9}

        2) Reading a single result:

                >>> from results import Results
                >>> res = Results('filename_with_results')
                >>> var1 = 1; var2 = 0; res_dict = res[(var1, var2)]

"""


import shelve
from filelock import FileLock
import os


use_locker = True


class Results:
    def __init__(self, filename):
        assert isinstance(filename, str), type(filename)
        assert not filename.endswith('.lock'), filename

        self.filenamelock = filename + '.lock'
        self.filename = filename

        if not os.path.exists(self.filenamelock):
            res = shelve.open(self.filename)
            res.close()

    def __len__(self):
        if use_locker:
            with FileLock(self.filenamelock):
                res = shelve.open(self.filename)
                ret = len(res)
                res.close()
        else:
            res = shelve.open(self.filename)
            ret = len(res)
            res.close()

        return ret

    def __setitem__(self, key, results):
        assert isinstance(key, str), type(key)
        assert isinstance(results, dict), type(results)

        if use_locker:
            with FileLock(self.filenamelock):
                res = shelve.open(self.filename)
                res[key] = results
                res.close()
        else:
            res = shelve.open(self.filename)
            res[key] = results
            res.close()

    def __getitem__(self, key):
        assert isinstance(key, str), type(key)

        if use_locker:
            with FileLock(self.filenamelock):
                res = shelve.open(self.filename)
                ret = res[key]
                res.close()
        else:
            res = shelve.open(self.filename)
            ret = res[key]
            res.close()

        return ret

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key):
        assert isinstance(key, str), type(key)

        if use_locker:
            with FileLock(self.filenamelock):
                res = shelve.open(self.filename)
                ret = key in res
                res.close()
        else:
            res = shelve.open(self.filename)
            ret = key in res
            res.close()

        return ret

    def save_DEPRECATED_20200526(self, key, results):
        assert isinstance(key, tuple), type(key)
        assert isinstance(results, dict), type(results)
        assert all(not isinstance(x, list) for x in key), key

        if use_locker:
            with FileLock(self.filenamelock):
                res = shelve.open(self.filename)
                res[str(key)] = results
                res.close()
        else:
            res = shelve.open(self.filename)
            res[str(key)] = results
            res.close()

    def remove_DEPRECATED_20200526(self, key):
        assert isinstance(key, tuple), type(key)

        if use_locker:
            with FileLock(self.filenamelock):
                res = shelve.open(self.filename)
                results = res.pop(str(key))
                res.close()
        else:
            res = shelve.open(self.filename)
            results = res.pop(str(key))
            res.close()

        return results

    def dict(self):
        """Return a dictionary structure with the saved data."""
        if use_locker:
            with FileLock(self.filenamelock):
                res = shelve.open(self.filename)
                items = [(x[0], x[1]) for x in list(res.items())]
                res.close()
        else:
            res = shelve.open(self.filename)
            items = [(eval(x[0]), x[1]) for x in list(res.items())]
            res.close()

        dic = dict(items)
        return dic

    def keys(self):
        if use_locker:
            with FileLock(self.filenamelock):
                res = shelve.open(self.filename)
                keys = list(res.keys())
                res.close()
        else:
            res = shelve.open(self.filename)
            keys = list(res.keys())
            res.close()

        return keys

    def get_processed_DEPRECATED_20200526(self):
        dic = self.dict()
        assert len(list(dic.keys())[0]) == 8, list(dic.keys())[0]
        dic_processed = {
            (
                key[0],         # version
                key[1],         # number of classes employed
                key[2],         # if there is augmentation
                eval(key[3]),   # if there is normalization
                key[4],         # color space
                key[5],         # stride
                'Top' if int(key[6]) == 1 else 'Elong',  # rock type
                key[7].upper()                           # classifier
            ): dic[key]
            for key in dic
        }
        return dic_processed

    def clear(self):
        if use_locker:
            with FileLock(self.filenamelock):
                res = shelve.open(self.filename)
                res.clear()
                res.close()
        else:
            res = shelve.open(self.filename)
            res.clear()
            res.close()
