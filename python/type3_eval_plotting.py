import sys

from annotation import *
from ontology import *
from gene_ontology import *
from mat_pred import *
from mat_ontology import *
from graph import *

import pandas as pd
from collections import defaultdict
from sortedcontainers import SortedList
from itertools import starmap
import heapq
from heapq import heappush, heappop
import random
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle

import glob
import os
import traceback

asp_map = {'mfo': 'F', 'bpo': 'P', 'cco': 'C'}

# def plot_ont_top10(ont_top10, ont, ont_bl=None, metric='fmax'):
#     plt.figure(figsize=[8, 8])
#     plt.set_cmap('tab20b')
#     if metric == 'fmax':
#         curve_field = 'prrc_curve'
#         tauidx_field = 'fmax_tauidx'
#     elif metric == 'smin':
#         curve_field = 'rumi_curve'
#         tauidx_field = 'smin_tauidx'
#     curves = np.array(ont_top10[ont][curve_field])
#     markpos = np.array(ont_top10[ont][tauidx_field])
#     #     markpos = np.stack(markpos,0).reshape([10, 1])
#     #     print(markpos)
#     lines = []
#     for i in range(len(ont_top10[ont])):
#         p, = plt.plot(curves[i][:, 1], curves[i][:, 0])
#         lines.append(p)
#         plt.scatter(curves[i][markpos[i], 1], curves[i][markpos[i], 0], edgecolors=p.get_color(), s=80,
#                     facecolors='none', marker='o', )
#     legends = list(ont_top10[ont].pi)
#     if ont_bl:
#         curves = np.array(ont_bl[ont]['prrc_curve'])
#         markpos = np.array(ont_bl[ont][tauidx_field])
#         # curves = np.stack(curves,0)
#         for i in range(2):
#             p, = plt.plot(curves[i][:, 1], curves[i][:, 0], '--')
#             lines.append(p)
#             plt.scatter(curves[i][markpos[i], 1], curves[i][markpos[i], 0], edgecolors=p.get_color(), s=80,
#                         facecolors='none', marker='o', )
#         legends += list(ont_bl[ont].teamname)
#
#     plt.legend(lines, legends)
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')

mcolor = [0.6, 0.6, 0.6]  # regular models
bcolor = [[196 / 255, 48 / 255, 43 / 255], [0 / 255, 83 / 255, 159 / 255]]  # baseline models


def convert_color_format(colors):
    ret = []
    for color in colors:
        r, g, b = color
        r = '%02x' % int(r * 255)
        g = '%02x' % int(g * 255)
        b = '%02x' % int(b * 255)
        ret.append('#%s%s%s' % (r, g, b))
    return ret


def get_bounding_box(marks):
    xmax, ymax = marks.max(0)
    xmin, ymin = marks.min(0)
    return [[xmin - 1, xmax + 1], [ymin - 1, ymax + 1]]


circle_size = 9
dot_size = 5.5
circle_mark = 'o'
dot_mark = 'o'


def plot_ont_top10(ont_top10, ont, ont_bl=None, metric='fmax'):
    assert (metric in ['fmax', 'smin'])
    if metric == 'fmax':
        curve_field = 'prrc_curve'
        tauidx_field = 'fmax_tauidx'
    elif metric == 'smin':
        curve_field = 'rumi_curve'
        tauidx_field = 'smin_tauidx'
    fig = plt.figure(figsize=[8, 8])
    plt.set_cmap('tab20b')
    ax = plt.gca()
    # ax.set_aspect(1)
    curves = list(ont_top10[ont][curve_field])
    markpos = np.array(ont_top10[ont][tauidx_field])
    lines = []
    marks = []
    for i in range(len(ont_top10[ont])):
        p, = plt.plot(curves[i][:, 1], curves[i][:, 0], linewidth=1.5)
        lines.append(p)
        marks.append((curves[i][markpos[i], 1], curves[i][markpos[i], 0]))
        plt.scatter(curves[i][markpos[i], 1], curves[i][markpos[i], 0], edgecolors=p.get_color(), s=np.pi * (circle_size)**2 / 2,
                    facecolors='none', marker=circle_mark, )
        plt.scatter(curves[i][markpos[i], 1], curves[i][markpos[i], 0], edgecolors='none', s=np.pi * (dot_size)**2 / 2,
                    facecolors=p.get_color(), marker=dot_mark, )
        legends = list(ont_top10[ont].pi)
    if ont_bl:
        curves = list(ont_bl[ont][curve_field])
        markpos = np.array(ont_bl[ont][tauidx_field])
        for i in range(2):
            if metric == 'smin':
                curves[i] = curves[:, [1, 0]]  # swap axis for ru-mi
            p, = plt.plot(curves[i][:, 1], curves[i][:, 0], '--', color=(bcolor[i]), linewidth=3)
            lines.append(p)
            marks.append((curves[i][markpos[i], 1], curves[i][markpos[i], 0]))
            plt.scatter(curves[i][markpos[i], 1], curves[i][markpos[i], 0], edgecolors=p.get_color(), s=np.pi * (circle_size)**2 / 2,
                        facecolors='none', marker=circle_mark, )
            plt.scatter(curves[i][markpos[i], 1], curves[i][markpos[i], 0], edgecolors='none', s=np.pi * (dot_size)**2 / 2,
                        facecolors=p.get_color(), marker=dot_mark, )
        legends += list(ont_bl[ont].teamname)
    marks = np.array(marks)

    if metric == 'smin':
        # plt.xlim([0, 14])
        # plt.ylim([0, 14])
        box = get_bounding_box(marks)
        plt.xlim([max(0, box[0][0]), box[0][1]])
        plt.ylim(box[1])
    plt.legend(lines, legends)
    if metric == 'fmax':
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    else:
        plt.xlabel('Remaining Uncertainty')
        plt.ylabel('Misinformation')

    if metric == 'fmax':
        x = np.arange(0.05, 1, 0.01)
        y = np.arange(0.05, 1, 0.01)
        X, Y = np.meshgrid(x, y)
        Z = 2 * X * Y / (X + Y)

        # plt.legend(tag, 'Fontsize', 10, 'Interpreter', 'none', 'Position', [0.65, 0.25, 0.30, 0.50]);
        # plt.contour(X, Y, Z, 'ShowText', 'on', 'LineColor', np.array([1, 1, 1]) * 0.5, 'LineStyle', ':', 'LabelSpacing', 288)
        CS = plt.contour(X, Y, Z, np.arange(0.1, 1, 0.1), colors=np.array([[1, 1, 1, 1]]) * 0.5, linestyles='dotted')
        ax.clabel(CS, inline=True, fontsize=10, fmt=lambda x: "%.1f" % x, inline_spacing=20)
    else:
        x = np.arange(box[0][0], box[0][1], np.diff(box[0]) / 100)
        y = np.arange(box[1][0], box[1][1], np.diff(box[1]) / 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sqrt(X ** 2 + Y ** 2)
        zmin, zmax = int(Z.min()), int(Z.max())
        CS = plt.contour(X, Y, Z, np.arange(zmin, zmax, (zmax - zmin) / 9), colors=np.array([[1, 1, 1, 1]]) * 0.5, linestyles='dotted')
        ax.clabel(CS, inline=True, fontsize=10, fmt=lambda x: "%.1f" % x, inline_spacing=20)


bar_w = 0.7


def plot_ont_top10_bar(ont_top10, ont, ont_bl=None, metric='fmax', withbst=False):
    assert (metric in ['fmax', 'smin'])
    plt.figure(figsize=[8, 4.5], dpi=350)
    data = ont_top10[ont].copy()
    bars = ont_top10[ont][metric].copy()
    barcolors = [mcolor] * 10
    teams = list(ont_top10[ont].pi)
    if ont_bl:
        plt.axhline(ont_bl[ont][metric]['BN4S'], c=bcolor[0], linestyle='--')
        plt.axhline(ont_bl[ont][metric]['BB4S'], c=bcolor[1], linestyle='--')

        bars = pd.concat([bars, ont_bl[ont][metric].loc[['BN4S', 'BB4S']].copy()])
        data = pd.concat([data, ont_bl[ont].loc[['BN4S', 'BB4S']].copy()])
        barcolors += bcolor
        teams += list(ont_bl[ont].teamname)
    bars.plot.bar(color=convert_color_format(barcolors), rot=45, width=0.8)

    if withbst:
        for i in range(len(teams)):
            print(data.iloc[i][f'{metric}_q05'])
            plt.plot([i, i], [data.iloc[i][f'{metric}_q05'], data.iloc[i][f'{metric}_q95']], color='k')
            plt.plot([i - bar_w / 4, i + bar_w / 4], [data.iloc[i][f'{metric}_q05'], data.iloc[i][f'{metric}_q05']],
                     color='k')
            plt.plot([i - bar_w / 4, i + bar_w / 4], [data.iloc[i][f'{metric}_q95'], data.iloc[i][f'{metric}_q95']],
                     color='k')

    #     print(list(ont_top10[ont]['cover_rate']))
    cov = ['C=%.2f' % c for c in ont_top10[ont]['cover_rate']]
    if ont_bl:
        cov += ['C=%.2f' % c for c in ont_bl[ont]['cover_rate']]
    for i in range(len(teams)):
        plt.text(i, 0.01, cov[i], rotation='vertical', ha='center', va='bottom', c='white')
    plt.xticks(range(len(teams)), labels=teams, ha='right')
    plt.ylabel(metric.capitalize())
    plt.title(ont.upper())

    if metric == 'smin':
        plt.ylim([bars.min() - 2, bars.max() + 1])
