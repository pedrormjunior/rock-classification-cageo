# -*- mode: makefile; -*-

# Copyright (c) 2019-2023
# Shell-ML Project
# Pedro Ribeiro Mendes Júnior <pedrormjunior@gmail.com> et al.
# Artificial Intelligence Lab. Recod.ai
# Institute of Computing (IC)
# University of Campinas (Unicamp)
# Campinas, São Paulo, Brazil
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

export SHELL = /bin/bash
COLORS_SCRIPT ?= ./colors.sh

define print_vars
	. ${COLORS_SCRIPT}; \
	if [ -z "${2}" -a -z "${3}" ]; then \
		echo -e $${IPur}$@$${Reg}:$${IGre}DATE$${Reg}:$${IYel}$$(date +%Y/%m/%d-%H:%M:%S)$${Reg}; \
		$(foreach var,${1},echo -e $${IPur}$@$${Reg}:$${IGre}${var}$${Reg}:$${IYel}${${var}}$${Reg};) \
	else \
		echo -en $${IPur}$@; \
		echo -en $${Reg}:$${IGre}DATE$${Reg}:$${IYel}$$(date +%Y/%m/%d-%H:%M:%S)$${Reg}; \
		$(foreach var,${1},echo -en $${Reg}:$${IGre}${var}$${Reg}:$${IYel}${${var}}$${Reg};) \
		echo -e $${ICya}${2}$${IRed}${3}$${Reg}; \
	fi;
endef
