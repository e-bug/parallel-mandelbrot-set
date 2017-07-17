# -*- coding: utf-8 -*-
"""
Created by e-bug on 24/03/17.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib2tikz import save as tikz_save
import numpy as np


# colorblind palette
colorblind_palette_dict = {'black': (0,0,0), 
                           'orange': (0.9,0.6,0), 
                           'sky_blue': (0.35,0.7,0.9), 
                           'bluish_green': (0,0.6,0.5), 
                           'yellow': (0.95,0.9,0.25), 
                           'blue': (0,0.45,0.7), 
                           'vermillion': (0.8,0.4,0), 
                           'reddish_purple': (0.8,0.6,0.7)}
palette_order = {0: 'vermillion', 1: 'bluish_green', 2: 'sky_blue', 3: 'orange', 
                 4: 'black', 5: 'yellow', 6: 'blue', 7: 'reddish_purple'}
palette_order2 = {0: 'blue', 1: 'orange', 2: 'bluish_green', 3: 'yellow', 
                 4: 'reddish_purple', 5: 'sky_blue', 6: 'vermillion', 7: 'black'}
n_colors = len(colorblind_palette_dict)

# ideal line
ideal_color = (0.5,0.5,0.5)

# markers
markers = ['o', '^', 's', 'D', '*', 'h', '.', '+']
n_markers = len(markers)

# linestyles
linestyles = [':', '-.', '--', '-']
n_linestyles = len(linestyles)


def plot_tts(n_nodes, lines, labels=None, legend_title='Problem size', xlabel='Number of nodes', xscale='log2',
             ylabel='Time to solution [s]', yscale='log', cmap_name=None, filename=None, saveas='tikz',
             figureheight = '\\figureheight', figurewidth = '\\figurewidth'):
    """
    Plots the time to solution as a function of the number of nodes.
    :param n_nodes: values in x-axis (i.e. number of nodes)
    :param lines: list of lists, each with y values for each x value
    :param labels: labels of the lines
    :param legend_title: title of the legend
    :param xlabel: label of x-axis
    :param xscale: scale of x-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param ylabel: label of y-axis
    :param yscale: scale of y-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param cmap_name: name of colormap to be used (see: http://matplotlib.org/examples/color/colormaps_reference.html).
                      If None, colorblind palette is used
    :param saveas:
    """

    plt.figure(figsize=(12,8))
    plt.grid()

    # colormap
    n_lines = len(lines)
    line_colors = []
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        line_colors = cmap(np.linspace(0.25, 0.9, n_lines))
    else:
        line_colors = [colorblind_palette_dict[palette_order[i%n_colors]] for i in range(n_lines)]

    # plot lines
    for i,tts in enumerate(lines):
        plt.plot(n_nodes, tts, 
                 color=line_colors[i], linestyle=linestyles[i%n_linestyles], 
                 marker=markers[i%n_markers], markerfacecolor=line_colors[i], markersize=7)
    # x-axis
    if xscale == 'log2':
        plt.xscale('log', basex=2)
    elif xscale == 'log':
        plt.xscale('log')
    plt.xticks(n_nodes, fontsize='large')
    plt.xlabel(xlabel, fontsize='x-large')

    # y-axis
    if yscale == 'log2':
        plt.yscale('log', basex=2)
    elif yscale == 'log':
        plt.yscale('log')
    plt.yticks(fontsize='large')
    plt.ylabel(ylabel, fontsize='x-large')

    # legend
    if labels is not None:
        if len(labels) == n_lines:
            legend = plt.legend(labels, loc='upper right', bbox_to_anchor=[1, 1], 
                                ncol=min(n_lines,4), shadow=False, fancybox=True,
                                title=legend_title, fontsize='large')
            plt.setp(legend.get_title(),fontsize='x-large')
        else:
            raise ValueError('Number of labels does not match number of lines')

    # ticks formatting
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    # ax.yaxis.set_major_formatter(ScalarFormatter())

    # save figure
    if (saveas is None) or (filename is None):
        plt.show()
    elif saveas == 'tikz':
        tikz_save(filename + '.' + saveas, figureheight = figureheight, figurewidth = figurewidth)
    else:
        plt.savefig(filename + '.' + saveas)
        


def plot_speedup(n_nodes, lines, labels=None, legend_title='Problem size', xlabel='Number of nodes', xscale='log2',
                 ylabel='Speedup', yscale='log2', plot_ideal=True, cmap_name=None, filename=None, saveas='tikz',
                 figureheight = '\\figureheight', figurewidth = '\\figurewidth'):
    """
    Plots the speedup as a function of the number of nodes.
    :param n_nodes: values in x-axis (i.e. number of nodes)
    :param lines: list of lists, each with y values for each x value
    :param labels: labels of the lines
    :param legend_title: title of the legend
    :param xlabel: label of x-axis
    :param xscale: scale of x-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param ylabel: label of y-axis
    :param yscale: scale of y-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param plot_ideal: if True, plots ideal speedup line
    :param cmap_name: name of colormap to be used (see: http://matplotlib.org/examples/color/colormaps_reference.html).
                      If None, colorblind palette is used
    :param saveas:
    """

    plt.figure(figsize=(12,8))
    plt.grid()

    # colormap
    n_lines = len(lines)
    line_colors = []
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        line_colors = cmap(np.linspace(0.25, 0.9, n_lines))
    else:
        line_colors = [colorblind_palette_dict[palette_order[i%n_colors]] for i in range(n_lines)]

    # plot lines
    for i,tts in enumerate(lines):
        plt.plot(n_nodes, tts, 
                 color=line_colors[i], linestyle=linestyles[i%n_linestyles], 
                 marker=markers[i%n_markers], markerfacecolor=line_colors[i], markersize=7)

    if plot_ideal:
        plt.plot(n_nodes, n_nodes, color=ideal_color)
        plt.text(n_nodes[-2]+1, n_nodes[-2]+5, 'ideal', fontsize='x-large')

    # x-axis
    if xscale == 'log2':
        plt.xscale('log', basex=2)
    elif xscale == 'log':
        plt.xscale('log')
    plt.xticks(n_nodes, fontsize='large')
    plt.xlabel(xlabel, fontsize='x-large')

    # y-axis
    if yscale == 'log2':
        plt.yscale('log', basex=2)
    elif yscale == 'log':
        plt.yscale('log')
    plt.yticks(n_nodes, fontsize='large')
    plt.ylabel(ylabel, fontsize='x-large')

    # legend
    if labels is not None:
        if len(labels) == n_lines:
            legend = plt.legend(labels, loc='upper left',
                                ncol=min(n_lines,4), shadow=False, fancybox=True,
                                title=legend_title, fontsize='large')
            plt.setp(legend.get_title(),fontsize='x-large')
        else:
            raise ValueError('Number of labels does not match number of lines')

    # ticks formatting
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    # save figure
    if (saveas is None) or (filename is None):
        plt.show()
    elif saveas == 'tikz':
        tikz_save(filename + '.' + saveas, figureheight = figureheight, figurewidth = figurewidth)
    else:
        plt.savefig(filename + '.' + saveas)


def plot_efficiency(n_nodes, lines, labels=None, legend_title='Problem size', xlabel='Number of nodes', xscale='log2',
                    ylabel='Efficiency', yscale=None, plot_ideal=True, cmap_name=None, filename=None, saveas='tikz',
                    figureheight = '\\figureheight', figurewidth = '\\figurewidth'):
    """
    Plots the efficiency as a function of the number of nodes.
    :param n_nodes: values in x-axis (i.e. number of nodes)
    :param lines: list of lists, each with y values for each x value
    :param labels: labels of the lines
    :param legend_title: title of the legend
    :param xlabel: label of x-axis
    :param xscale: scale of x-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param ylabel: label of y-axis
    :param yscale: scale of y-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param plot_ideal: if True, plots ideal speedup line
    :param cmap_name: name of colormap to be used (see: http://matplotlib.org/examples/color/colormaps_reference.html).
                      If None, colorblind palette is used
    :param saveas:
    """

    plt.figure(figsize=(12,8))
    plt.grid()

    # colormap
    n_lines = len(lines)
    line_colors = []
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        line_colors = cmap(np.linspace(0.25, 0.9, n_lines))
    else:
        line_colors = [colorblind_palette_dict[palette_order[i%n_colors]] for i in range(n_lines)]

    # plot lines
    for i,tts in enumerate(lines):
        plt.plot(n_nodes, tts, 
                 color=line_colors[i], linestyle=linestyles[i%n_linestyles], 
                 marker=markers[i%n_markers], markerfacecolor=line_colors[i], markersize=7)

    if plot_ideal:
        plt.plot(n_nodes, np.ones(len(n_nodes)), color=ideal_color)
        plt.text(n_nodes[-1]-10, 0.96, 'ideal', fontsize='x-large')

    # x-axis
    if xscale == 'log2':
        plt.xscale('log', basex=2)
    elif xscale == 'log':
        plt.xscale('log')
    plt.xticks(n_nodes, fontsize='large')
    plt.xlabel(xlabel, fontsize='x-large')

    # y-axis
    if yscale == 'log2':
        plt.yscale('log', basex=2)
    elif yscale == 'log':
        plt.yscale('log')
    plt.yticks(fontsize='large')
    plt.ylabel(ylabel, fontsize='x-large')

    # legend
    if labels is not None:
        if len(labels) == n_lines:
            legend = plt.legend(labels, loc='lower left',
                                ncol=min(n_lines,4), shadow=False, fancybox=True,
                                title=legend_title, fontsize='large')
            plt.setp(legend.get_title(),fontsize='x-large')
        else:
            raise ValueError('Number of labels does not match number of lines')

    # ticks formatting
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #ax.yaxis.set_major_formatter(ScalarFormatter())

    # save figure
    if (saveas is None) or (filename is None):
        plt.show()
    elif saveas == 'tikz':
        tikz_save(filename + '.' + saveas, figureheight = figureheight, figurewidth = figurewidth)
    else:
        plt.savefig(filename + '.' + saveas)


def plot_tts_bar(machines, values_lists, labels=None, width=0.35, legend_title='Program', xlabel='Machine',
                 ylabel='Time to solution [s]', yscale=None, cmap_name=None, filename=None, saveas='tikz',
                 figureheight = '\\figureheight', figurewidth = '\\figurewidth'):
    """
    Plots the time to solution as a function of the number of nodes.
    :param machines: list of strings for each machine
    :param values_lists: list of lists of values corresponding to the passed machines and programs
    :param labels: labels of the programs
    :param width: the width of the bars
    :param legend_title: title of the legend
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param yscale: scale of y-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param cmap_name: name of colormap to be used (see: http://matplotlib.org/examples/color/colormaps_reference.html).
                      If None, colorblind palette is used
    :param saveas:
    """
    # TODO -- fix plotted values: see *O.png

    plt.figure(figsize=(12,8))
    plt.grid()

    # colormap
    n_labels = len(values_lists[0])
    bar_colors = []
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        bar_colors = cmap(np.linspace(0.25, 0.9, n_labels))
    else:
        bar_colors = [colorblind_palette_dict[palette_order2[i%n_colors]] for i in range(n_labels)]

    # plot bars -- label by label
    n_machines = len(machines)
    x_values = np.arange(1, n_machines+1)
    max_value = max(max(values_lists))
    for i in range(n_labels):
        values = [val_list[i] for val_list in values_lists]
        plt.bar(x_values+i*width, values, width, align='center', color=bar_colors[i])
        for idx, v in enumerate(sorted(values_lists[i], reverse=True)):
            plt.text(idx+1+i*width, v+max_value/100, str(v), fontsize='large', horizontalalignment='center')
    
    # x-axis
    plt.xticks(x_values+(n_labels-1)*width/2, machines, fontsize='large')
    plt.xlabel(xlabel, fontsize='x-large')

    # y-axis
    if yscale == 'log2':
        plt.yscale('log', basex=2)
    elif yscale == 'log':
        plt.yscale('log')
    plt.yticks(fontsize='large')
    plt.ylabel(ylabel, fontsize='x-large')

    # legend
    if labels is not None:
        if n_labels == len(labels):
            legend = plt.legend(labels, loc='upper right', bbox_to_anchor=[1, 1], 
                                ncol=min(n_labels,4), shadow=False, fancybox=True,
                                title=legend_title, fontsize='large')
            plt.setp(legend.get_title(),fontsize='x-large')
        else:
            raise ValueError('Number of labels does not match number of lines')

    # ticks formatting
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    # ax.yaxis.set_major_formatter(ScalarFormatter())

    # save figure
    if (saveas is None) or (filename is None):
        plt.show()
    elif saveas == 'tikz':
        tikz_save(filename + '.' + saveas, figureheight = figureheight, figurewidth = figurewidth)
    else:
        plt.savefig(filename + '.' + saveas)




