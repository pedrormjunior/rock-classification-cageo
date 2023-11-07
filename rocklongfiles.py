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


import shelve
import os
import hashlib
from filelock import FileLock

SHELVE_LONGFILES = 'Shell_longfiles'
SHELVE_LONGFILES_LOCK = SHELVE_LONGFILES + '.lock'


def shortfile(filename):
    """This function returns the given filename itself when its size is
    supported by the operating system, otherwise it generated and stores in
    `SHELVE_LONGFILES` a shorter filename.

    """
    slash_pos = filename.rfind('/')
    file_part = filename[slash_pos+1:]
    if len(file_part) <= 255:   # OK on Linux
        return filename
    else:
        directory = filename[:slash_pos+1]
        extension_pos = file_part.rfind('.')
        extension = (file_part[extension_pos:]
                     if extension_pos >= 0
                     else '')
        extension = extension if len(extension) <= 4 else ''
        file_part_new = (hashlib.sha256(file_part.encode()).hexdigest() +
                         extension)

        with FileLock(SHELVE_LONGFILES_LOCK):
            with shelve.open(SHELVE_LONGFILES) as fd:
                fd[file_part_new] = file_part

        return os.path.join(directory, file_part_new)
