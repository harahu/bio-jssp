from mpso_jssp import Problem
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np


def read_mpso_problem(filename):
    with open(filename, 'r') as f:
        jssp = [line.split() for line in f]
        n, m = int(jssp[0][0]), int(jssp[0][1])

        jssp = jssp[1:]
        jobs = [[] for _ in range(n)]

        for i in range(n):
            line = jssp[i]
            for j in range(m):
                machine = int(line[2 * j])
                time = int(line[2 * j + 1])
                jobs[i].append((machine, time))
    return Problem(n, m, jobs)


def solution_plotter(solution, fname):
    cm = plt.get_cmap('gist_rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=solution.problem.n - 1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    ylabels = ['Machine {}'.format(i+1) for i in range(len(solution.schedule))]
    ilen = len(ylabels)
    pos = np.arange(0.5, ilen * 0.5 + 0.5, 0.5)
    task_times = {}
    for i, task in enumerate(ylabels):
        task_times[task] = solution.schedule[i]
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)
    for i in range(len(ylabels)):
        for job, op_num, start, end in task_times[ylabels[i]]:
            width = end - start
            ax.barh(pos[i], width, left=start,
                    height=0.3, align='center', edgecolor='lightgreen',
                    color=scalarMap.to_rgba(job), alpha=0.8)
            rot = 'horizontal'
            if width < solution.makespan/70:
                continue
            elif width < solution.makespan/35:
                rot = 'vertical'
            xloc = start + width / 2
            ax.text(xloc, pos[i], '{}/{}'.format(op_num+1, job+1), horizontalalignment='center',
                    verticalalignment='center', color='black', weight='bold', rotation=rot)

    locsy, labelsy = plt.yticks(pos, ylabels)
    plt.setp(labelsy, fontsize=14)
    #ax.axis('tight')
    ax.set_ylim(ymin=-0.1, ymax=ilen * 0.5 + 0.5)
    ax.grid(color='g', linestyle=':')
    ax.invert_yaxis()
    plt.xlabel('Time')
    plt.title(fname)
    plt.savefig('gantt.svg')
    plt.show()
