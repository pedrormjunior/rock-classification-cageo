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


"""Module for generating the final table of results that are included in the
Slides.

"""


from rocklib import logger
import rockresults
import envconfig
import experiments.main as rockexperiment
import statistics
import math


res = rockresults.Results(envconfig.results_filename)


def gen_table(key_dict_fmt,
              quant_classes,
              topk,
              result_type_lst,
              classifier_lst,
              partition_lst,
              optimization_metric_lst,
              experiment,
              ):
    """Generate table of results.

    Table is organized as following:

                    Network optimized for
                    LOSS    ACC     NA
    img     3nn     ??      ??      ??
            rf      ??      ??      ??
            svm     ??      ??      ??
            softmax ??      ??      ??
    pat     3nn     ??      ??      ??
            rf      ??      ??      ??
            svm     ??      ??      ??
            softmax ??      ??      ??

    """
    # Just fixing for the order in the Slides.
    assert result_type_lst == ['img', 'pat',], result_type_lst
    # assert classifier_lst == ['3nn', 'rf', 'svm', 'softmax',], \
    #     classifier_lst
    assert optimization_metric_lst == ['LOSS', 'ACC', 'NA',], \
        optimization_metric_lst

    def std_error(lst):
        return statistics.stdev(lst) / math.sqrt(len(lst))

    logger.debug(key_dict_fmt)
    logger.debug(f'topk: {topk}')
    for result_type in result_type_lst:
        for classifier in classifier_lst:
            if classifier == 'rf':
                # This is for the case in which 3nn is not being considered and
                # a blank line is being printed instead of the results for 3nn.
                print()
            for optimization_metric in optimization_metric_lst:
                values = [
                    res[
                        key_dict_fmt[optimization_metric].format(
                            partition=partition,
                            quant_classes=quant_classes,
                            classifier=classifier,
                            topk=topk,
                        )
                    ]['test']['na'][result_type]
                    for partition in partition_lst
                ]
                print(f'{statistics.mean(values):.3f}', end='  ')
                print(f'{std_error(values):.3f}', end='    ')
            print()


if __name__ == '__main__':
    Experiment = rockexperiment.Experiment
    Experiment.experiment = 'gen_table_of_results'
    Experiment(gen_table, send_dict=True)
