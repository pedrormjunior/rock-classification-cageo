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
from matplotlib import pyplot as plt
import itertools as it
import textwrap

import envconfig


plt.rcParams.update({
    'text.usetex': True,
})

figsize = plt.gcf().get_size_inches()
inches_per_pt = 1 / 72.27       # https://jwalton.info/Matplotlib-latex-PGF/
textwidth = 390                 # Got with \showthe\textwidth in the document.
plt.gcf().set_size_inches((inches_per_pt * textwidth,
                           inches_per_pt * textwidth
                           * figsize[1] / figsize[0]))
plt.plot()
plt.close()               # Necessary to normalize the font size in plot (bug?)
del figsize, inches_per_pt, textwidth

DARK_BACKGROUND = (
    True
    # False
)

if DARK_BACKGROUND:
    plt.style.use('dark_background')
    COLORS = {
        'LOSS': 'red',
        'ACC': '#00ff00',
        'NA': 'cyan',
    }
    WHITE_COLOR = 'white'
    YELLOW_COLOR = 'yellow'

else:
    COLORS = {
        'LOSS': 'red',
        'ACC': 'green',
        'NA': 'blue',
    }
    WHITE_COLOR = 'black'
    YELLOW_COLOR = 'magenta'

MINMAX = {
    'LOSS': min,
    'ACC': max,
    'NA': max,
}
KEYS = (
    COLORS.keys()
    # ['LOSS']
)
LW = 0.868


FILENAMES = (
    # 20230615: From now on, containing the experiments performed with the code
    # transferred to Github.  All the experiments are running with early
    # stopping with tolerance of 10 epochs.

    # # paper: inception_v3 (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_inception_v3_rdTrue']),
    #     [
    #     ],
    # )) +

    # # paper: squeezenet (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_squeezenet_rdTrue']),
    #     [
    #         'logs/2023-07-07--23-41-35-961918.log',  # 0
    #         'logs/2023-07-07--23-41-40-104321.log',  # 1
    #         'logs/2023-07-07--23-41-41-781808.log',  # 2
    #         'logs/2023-07-07--23-41-44-798714.log',  # 3
    #         'logs/2023-07-07--23-41-48-423877.log',  # 4
    #         'logs/2023-07-07--23-41-51-985424.log',  # 5
    #         'logs/2023-07-07--23-41-55-572965.log',  # 6
    #         'logs/2023-07-07--23-41-59-793436.log',  # 7
    #     ],
    # )) +

    # # paper: mobilenet_v2 (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_mobilenet_v2_rdTrue']),
    #     [
    #         'logs/2023-07-07--23-38-24-003332.log',  # 0
    #         'logs/2023-07-07--23-38-28-621511.log',  # 1
    #         'logs/2023-07-07--23-38-31-347637.log',  # 2
    #         'logs/2023-07-07--23-38-33-902719.log',  # 3
    #         'logs/2023-07-07--23-38-38-179683.log',  # 4
    #         'logs/2023-07-07--23-38-41-520391.log',  # 5
    #         'logs/2023-07-07--23-38-45-118025.log',  # 6
    #         'logs/2023-07-07--23-38-50-496370.log',  # 7
    #     ],
    # )) +

    # # paper: mobilenet_v3 (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_mobilenet_v3_rdTrue']),
    #     [
    #         'logs/2023-07-07--13-26-51-881758.log',  # 0
    #         'logs/2023-07-07--13-27-48-902823.log',  # 1
    #         'logs/2023-07-07--14-23-42-012322.log',  # 2
    #         'logs/2023-07-07--16-07-31-321716.log',  # 3
    #         'logs/2023-07-07--16-07-42-631233.log',  # 4
    #         'logs/2023-07-07--16-07-57-606399.log',  # 5
    #         'logs/2023-07-07--16-46-05-915915.log',  # 6
    #         'logs/2023-07-07--23-37-25-774930.log',  # 7
    #     ],
    # )) +

    # # paper: mnasnet (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_mnasnet_rdTrue']),
    #     [
    #         'logs/2023-07-07--01-22-33-563233.log',  # 0
    #         'logs/2023-07-07--01-23-17-368486.log',  # 1
    #         'logs/2023-07-07--10-29-10-923395.log',  # 2
    #         'logs/2023-07-07--10-29-16-956728.log',  # 3
    #         'logs/2023-07-07--10-29-23-969514.log',  # 4
    #         'logs/2023-07-07--10-29-30-597348.log',  # 5
    #         'logs/2023-07-07--10-29-38-372030.log',  # 6
    #         'logs/2023-07-07--10-29-56-120389.log',  # 7
    #     ],
    # )) +

    # # paper: shufflenet_v2 (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_shufflenet_v2_rdTrue']),
    #     [
    #         'logs/2023-07-06--14-41-23-026853.log',  # 0
    #         'logs/2023-07-06--15-20-23-653116.log',  # 1
    #         'logs/2023-07-07--01-16-42-892401.log',  # 2
    #         'logs/2023-07-07--01-17-10-533417.log',  # 3
    #         'logs/2023-07-07--01-18-22-014722.log',  # 4
    #         'logs/2023-07-07--01-18-35-183861.log',  # 5
    #         'logs/2023-07-07--01-20-15-247828.log',  # 6
    #         'logs/2023-07-07--01-20-54-564148.log',  # 7
    #     ],
    # )) +

    # # paper: googlenet (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_googlenet_rdTrue']),
    #     [
    #         'logs/2023-07-06--12-14-34-888520.log',  # 0
    #         'logs/2023-07-06--00-52-14-195480.log',  # 1
    #         'logs/2023-07-06--00-54-02-918674.log',  # 2
    #         'logs/2023-07-06--12-16-38-024229.log',  # 3
    #         'logs/2023-07-06--14-31-38-046621.log',  # 4
    #         'logs/2023-07-06--14-31-48-264722.log',  # 5
    #         'logs/2023-07-06--14-34-23-711181.log',  # 6
    #         'logs/2023-07-06--14-37-04-363944.log',  # 7
    #     ],
    # )) +

    # # paper: densenet (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_densenet_rdTrue']),
    #     [
    #         'logs/2023-06-26--15-59-41-466992.log',  # 0
    #         'logs/2023-06-26--16-06-40-144584.log',  # 1
    #         'logs/2023-06-26--16-26-04-939170.log',  # 2
    #         'logs/2023-06-26--16-29-05-253465.log',  # 3
    #         'logs/2023-06-26--16-33-09-830895.log',  # 4
    #         'logs/2023-06-26--16-36-31-890415.log',  # 5
    #         'logs/2023-06-28--18-01-55-050360.log',  # 6
    #         'logs/2023-06-28--18-01-58-750491.log',  # 7
    #     ],
    # )) +

    # # paper: alexnet (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_alexnet_rdTrue']),
    #     [
    #         'logs/2023-06-15--16-42-20-070410.log',  # 0
    #         'logs/2023-06-15--16-42-22-827390.log',  # 1
    #         'logs/2023-06-15--16-42-26-892859.log',  # 2
    #         'logs/2023-06-15--16-42-31-004487.log',  # 3
    #         'logs/2023-06-15--16-42-34-559808.log',  # 4
    #         'logs/2023-06-15--16-42-39-598418.log',  # 5
    #         'logs/2023-06-15--16-42-44-418084.log',  # 6
    #         'logs/2023-06-15--16-42-49-194751.log',  # 7
    #     ],
    # )) +

    # # paper: resnet (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_resnet_rdTrue']),
    #     [
    #         'logs/2023-06-26--14-51-01-480603.log',  # 0
    #         'logs/2023-06-26--14-54-28-133983.log',  # 1
    #         'logs/2023-06-26--15-07-32-688965.log',  # 2
    #         'logs/2023-06-26--15-13-17-891412.log',  # 3
    #         'logs/2023-06-26--15-19-38-589511.log',  # 4
    #         'logs/2023-06-26--15-22-38-833350.log',  # 5
    #         'logs/2023-06-26--15-30-21-568384.log',  # 6
    #         'logs/2023-06-26--15-56-49-362922.log',  # 7
    #     ],
    # )) +

    # # paper: efficientnet (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_efficientnet_rdTrue']),
    #     [
    #         'logs/2023-07-04--09-37-27-948733.log',  # 0
    #         'logs/2023-07-04--09-38-29-993456.log',  # 1
    #         'logs/2023-07-05--11-59-04-002379.log',  # 2
    #         'logs/2023-07-05--11-59-53-742459.log',  # 3
    #         'logs/2023-07-05--12-00-41-913671.log',  # 4
    #         'logs/2023-07-05--12-01-14-805934.log',  # 5
    #         'logs/2023-07-06--00-37-13-988180.log',  # 6
    #         'logs/2023-07-06--00-38-34-099400.log',  # 7
    #     ],
    # )) +

    # # paper: resnext (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_resnext_rdTrue']),
    #     [
    #         'logs/2023-07-03--04-38-11-330314.log',  # 0
    #         'logs/2023-07-03--04-38-52-343015.log',  # 1
    #         'logs/2023-07-03--04-39-17-506713.log',  # 2
    #         'logs/2023-06-26--14-13-30-932002.log',  # 3
    #         'logs/2023-07-03--04-48-41-853360.log',  # 4
    #         'logs/2023-07-04--09-33-32-309181.log',  # 5
    #         'logs/2023-07-04--09-34-11-919974.log',  # 6
    #         'logs/2023-07-04--09-34-41-473889.log',  # 7
    #     ],
    # )) +

    # # paper: swin (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_swin_rdTrue']),
    #     [
    #         'logs/2023-06-26--13-45-35-636775.log',  # 0
    #         'logs/2023-07-03--04-41-17-273018.log',  # 1
    #         'logs/2023-06-26--13-49-02-647604.log',  # 2
    #         'logs/2023-07-03--04-42-20-203841.log',  # 3
    #         'logs/2023-07-03--04-43-39-996615.log',  # 4
    #         'logs/2023-06-26--13-57-40-653397.log',  # 5
    #         'logs/2023-07-03--04-44-20-196919.log',  # 6
    #         'logs/2023-06-26--14-01-56-657015.log',  # 7
    #     ],
    # )) +

    # # paper: efficientnet_v2 (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_efficientnet_v2_rdTrue']),
    #     [
    #         'logs/2023-06-22--10-07-52-955431.log',  # 0
    #         'logs/2023-06-22--10-10-45-447719.log',  # 1
    #         'logs/2023-06-22--10-14-13-305573.log',  # 2
    #         'logs/2023-06-22--10-14-23-325556.log',  # 3
    #         'logs/2023-06-22--10-18-25-696062.log',  # 4
    #         'logs/2023-06-22--10-18-31-625723.log',  # 5
    #         'logs/2023-06-22--10-22-54-197761.log',  # 6
    #         'logs/2023-06-22--10-22-58-232151.log',  # 7
    #     ],
    # )) +

    # # paper: wide_resnet (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_wide_resnet_rdTrue']),
    #     [
    #         'logs/2023-06-22--09-13-49-206363.log',  # 0
    #         'logs/2023-06-22--09-16-02-307538.log',  # 1
    #         'logs/2023-06-22--09-16-49-674297.log',  # 2
    #         'logs/2023-06-22--09-16-56-609238.log',  # 3
    #         'logs/2023-06-22--09-18-32-511230.log',  # 4
    #         'logs/2023-06-22--09-18-49-506471.log',  # 5
    #         'logs/2023-06-22--09-19-04-237136.log',  # 6
    #         'logs/2023-06-22--09-19-57-897200.log',  # 7
    #     ],
    # )) +

    # # paper: vgg (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_vgg_rdTrue']),
    #     [
    #         'logs/2023-06-21--11-37-59-829302.log',  # 0
    #         'logs/2023-06-21--11-40-09-683585.log',  # 1
    #         'logs/2023-06-21--11-44-55-766297.log',  # 2
    #         'logs/2023-06-21--11-47-09-015899.log',  # 3
    #         'logs/2023-06-21--11-47-33-569717.log',  # 4
    #         'logs/2023-06-21--13-08-53-421907.log',  # 5
    #         'logs/2023-06-21--13-12-42-393719.log',  # 6
    #         'logs/2023-06-21--16-32-38-496965.log',  # 7
    #     ],
    # )) +

    # # paper: convnext (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_convnext_rdTrue']),
    #     [
    #         'logs/2023-06-15--17-38-28-635219.log',  # 0
    #         'logs/2023-06-15--17-38-40-114223.log',  # 1
    #         'logs/2023-06-15--17-38-46-634990.log',  # 2
    #         'logs/2023-06-15--17-38-51-567169.log',  # 3
    #         'logs/2023-06-15--17-39-12-848191.log',  # 4
    #         'logs/2023-06-15--17-39-21-457973.log',  # 5
    #         'logs/2023-06-15--17-39-25-819868.log',  # 6
    #         'logs/2023-06-15--17-39-30-633671.log',  # 7
    #     ],
    # )) +

    # # paper: vit (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_vit_rdTrue']),
    #     [
    #         'logs/2023-06-20--04-40-34-057548.log',  # 0
    #         'logs/2023-06-20--04-40-55-383140.log',  # 1
    #         'logs/2023-06-20--15-19-15-148395.log',  # 2
    #         'logs/2023-06-21--00-55-52-623661.log',  # 3
    #         'logs/2023-06-21--00-58-33-710837.log',  # 4
    #         'logs/2023-06-21--01-02-23-928759.log',  # 5
    #         'logs/2023-06-21--01-02-36-228628.log',  # 6
    #         'logs/2023-06-21--01-02-52-572096.log',  # 7
    #     ],
    # )) +

    # # paper: regnet (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_regnet_rdTrue']),
    #     [
    #         'logs/2023-06-16--15-57-53-604305.log',  # 0
    #         'logs/2023-06-16--16-02-39-092836.log',  # 1
    #         'logs/2023-06-16--16-03-12-265090.log',  # 2
    #         'logs/2023-06-18--23-23-38-586963.log',  # 3
    #         'logs/2023-06-19--00-21-05-313554.log',  # 4
    #         'logs/2023-06-19--12-37-57-705715.log',  # 5
    #         'logs/2023-06-19--12-38-42-110454.log',  # 6
    #         'logs/2023-06-20--04-39-26-266571.log',  # 7
    #     ],
    # )) +

    # # paper: convnext100 (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_convnext100_rdTrue']),
    #     [
    #         'logs/2023-07-19--17-05-53-123758.log',  # 0
    #         'logs/2023-07-19--17-06-46-985893.log',  # 1
    #         'logs/2023-07-19--17-07-00-849037.log',  # 2
    #         'logs/2023-07-19--17-07-16-473103.log',  # 3
    #         'logs/2023-07-24--14-25-03-409884.log',  # 4
    #         'logs/2023-07-19--17-07-47-888059.log',  # 5
    #         'logs/2023-07-19--17-08-06-313647.log',  # 6
    #         'logs/2023-07-19--17-08-22-853721.log',  # 7
    #     ],
    # )) +

    # # paper: fusioncnns (raw_dataset=True)
    # list(zip(
    #     it.cycle(['202306_fusioncnns_rdTrue']),
    #     [
    #     ],
    # )) +

    []
)
plot_prefix, filename = FILENAMES[0]


def line_to_dict(line):
    return eval(line.split('Metrics:')[1])


def get_metric(dic, metric):
    return [x[metric] * (-1 if MINMAX[metric] == min else 1)
            for x in dic]


def plot_metric(dic, metric, label, **kwargs):
    plt.plot(get_metric(dic, metric),
             # label=f'{metric} ({label})',
             color=COLORS[metric],
             lw=LW,
             **kwargs)


def axvline_best(dic, metric):
    lst = get_metric(dic, metric)
    best_value = MINMAX[metric](lst)
    best = lst.index(best_value)
    plt.axvline(
        best,
        color=COLORS[metric],
        ls='dashdot',
        lw=LW,
        # label='Best {}\n({}; {:.03})'.format(
        #     metric,
        #     best,
        #     best_value,
        # )
    )
    return best, best_value


def process(plot_prefix, filename):
    with open(filename) as fd:
        lines = fd.readlines()

    # line = [line for line in lines if line.find('optimizer:') >= 0][0]
    # def get_param(string):
    #     ret = ' '.join(line[line.find(string):].split(' ')[:2])
    #     return ret if ret != '\n' else ''
    # # title = ', '.join([get_param('lr:'), get_param('eps:'), get_param('rho:')])
    # title = ', '.join([get_param('optimizer:'), get_param('lr:'), get_param('momentum:')])
    # del line

    line = [line for line in lines if line.find('Experiment:') >= 0][0]
    experiment = eval(line[line.find('{'):line.rfind('}')+1])
    title = f'{experiment["model_name"]} — Partition {experiment["partition"]}'

    # title = lines[1].split('_')
    # title = f'Optimizer: {title[3]}, USE_PRETRAINED: {title[4]}'

    # title = lines[0].split(' ')[-1].split('/')[1][:-1]  # log filename

    def string_indexes(string):
        """Return the indexes of occurence of the `string` in the plot."""
        string_lines = [line for line in lines
                        if line.find('Metrics:') >= 0
                        or line.find(string) >= 0]
        string_occurrences = [line for line in string_lines
                              if line.find(string) >= 0]
        string_indexes = [
            int((string_lines.index(string_occurrences[i]) - i)/2)
            for i in range(len(string_occurrences))
        ]
        return string_indexes

    early_stop = string_indexes('Early stop at ')
    end_training = string_indexes('Training complete in ')
    assert all([x in end_training for x in early_stop]), \
        (early_stop, end_training)
    end_training = [x for x in end_training if x not in early_stop]

    train = [line for line in lines
             if line.find('(train)') >= 0
             and line.find('Loading dataset') == -1]
    val = [line for line in lines
           if line.find('(val)') >= 0
           and line.find('Loading dataset') == -1]

    train = [line_to_dict(line) for line in train]
    val = [line_to_dict(line) for line in val]

    for metric in KEYS:
        plot_metric(val, metric, label='val')

    for metric in KEYS:
        plot_metric(train, metric, label='train', ls='dashed')

    values = []
    for metric in KEYS:
        v = axvline_best(val, metric)
        values.append(v)
        del v
    values_str = '\n'.join(['{}: {:.03}'.format(v[0], v[1]) for v in values])

    for metric in KEYS:
        plt.plot([], [], color=COLORS[metric], label=metric, lw=7)
    plt.plot([], [], color=WHITE_COLOR, label='val', lw=LW)
    plt.plot([], [], color=WHITE_COLOR, label='train', ls='dashed', lw=LW)
    plt.plot([], [], color=WHITE_COLOR, label=f'Best val\n{values_str}',
             ls='dashdot', lw=LW)
    del values, values_str

    def plot_points(points, color, label):
        assert len(points) <= 1, points
        points_str = ('\n'.join(textwrap.wrap(str(points[0]), 16))
                      if len(points) > 0 else '')
        for i, x in enumerate(points):
            if i == 0:
                plt.axvline(x,
                            color=color,
                            ls='dotted',
                            label=f'{label}\n{points_str}',
                            lw=LW,)
            else:
                plt.axvline(x,
                            color=color,
                            ls='dotted',
                            lw=LW,)

    plot_points(early_stop, color=WHITE_COLOR, label='Early stop')
    plot_points(end_training, color=YELLOW_COLOR, label='End training')

    # plt.title(title, fontname='monospace')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    # plt.ylim([0.07, 0.25])
    # plt.ylim([0, 1])
    # plt.ylim([0, 0.62])
    plt.ylim([0, 2])
    plt.axhline(1, lw=LW)

    envconfig._makedir(envconfig.plots_dir)
    filenameout = os.path.join(
        envconfig.plots_dir,
        (plot_prefix + '__' +
         title
         .replace(':', '_')
         .replace(' ', '')
         .replace(',', '__') + '__' +
         filename.split('/')[-1].split('.')[0] +
         '.{}')
    )
    for ext in [
            'png',
            # 'pdf',
            # 'pgf',
    ]:
        plt.savefig(filenameout.format(ext), bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':
    for plot_prefix, filename in FILENAMES:
        print(plot_prefix, filename)
        process(plot_prefix, filename)
