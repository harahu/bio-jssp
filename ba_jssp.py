import random, collections
import jssp_io
import numpy as np


Parameters = collections.namedtuple('Parameters',
                                    ['max_iteration', 'ns', 'ne',
                                     'nb', 'nre', 'nrb', 'init_hyperbox',
                                     'shrink', 'stlim'])


class Patch(object):

    def __init__(self, jssp_problem, alg_params, coordinates):
        self._problem = jssp_problem
        self._coordinates = np.copy(coordinates)
        self._solution = Solution(jssp_problem, self._coordinates)
        self._patch_lim = alg_params.init_hyperbox
        self._patch_lim_max = alg_params.init_hyperbox
        self._patch_shrink = alg_params.shrink
        self._stagnation_count = 0
        self._stlim = alg_params.stlim

    @property
    def coordinates(self):
        return np.copy(self._coordinates)

    @property
    def makespan(self):
        return self._solution.makespan

    def _random_local(self):
        rand = np.copy(self._coordinates)
        for i in range(len(rand)):
            diff = random.uniform(-self._patch_lim, self._patch_lim)
            rand[i] += diff
        return rand

    def _random_global(self):
        dim = self._problem.n * self._problem.m
        rand = np.array([random.uniform(0, dim) for _ in range(dim)])
        return rand

    def global_search(self):
        self._coordinates = self._random_global()
        self._solution = Solution(self._problem, self._coordinates)
        self._stagnation_count = 0
        self._patch_lim = self._patch_lim_max

    def local_search(self, n_bees):
        search_results = []
        for i in range(n_bees):
            search_coord = self._random_local()
            search_results.append((search_coord, Solution(self._problem, search_coord)))
        search_results.sort(key=lambda r: r[1].makespan)
        if search_results[0][1].makespan < self._solution.makespan:
            self._coordinates = search_results[0][0]
            self._solution = search_results[0][1]
            self._stagnation_count = 0
            self._patch_lim = self._patch_lim_max
        else:
            self._stagnation_count += 1
            self._patch_lim *= self._patch_shrink
        if self._stagnation_count > self._stlim:
            self.global_search()


class Solution(object):

    def __init__(self, jssp_problem, coordinates):
        self.problem = jssp_problem
        int_series = sorted(range(len(coordinates)),
                            key=lambda index: coordinates[index])
        operations = [job % jssp_problem.n for job in int_series]
        self._schedule, self._makespan = self._schedule(operations)

    @property
    def makespan(self):
        return self._makespan

    @property
    def schedule(self):
        return self._schedule

    def _schedule(self, operations):
        job_operation_tracker = [0 for _ in range(self.problem.n)]
        job_end = [0 for _ in range(self.problem.n)]
        schedule = [[] for _ in range(self.problem.m)]
        for op in operations:
            operation = self.problem.jobs[op][job_operation_tracker[op]]
            machine = operation[0]
            time_rec = operation[1]
            machine_free = 0
            if not len(schedule[machine]) == 0:
                machine_free = schedule[machine][-1][3]
            start = max(job_end[op], machine_free)
            end = start + time_rec
            schedule[machine].append((op, job_operation_tracker[op], start, end))
            job_operation_tracker[op] += 1
            job_end[op] = end
        makespan = max(job_end)
        return schedule, makespan


class Problem(object):

    def __init__(self, n, m, jobs):
        self._n = n
        self._m = m
        self._jobs = jobs

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m

    @property
    def jobs(self):
        return self._jobs


def bees_algorithms(jssp_problem, alg_params, points, return_list):
    # Initialize patches
    flower_patches = []
    for point in points:
        flower_patches.append(Patch(jssp_problem, alg_params, point))
    while len(flower_patches) < alg_params.ns:
        dim = jssp_problem.n * jssp_problem.m
        coordinates = np.array(
            [random.uniform(0, dim) for _ in range(dim)])
        flower_patches.append(Patch(jssp_problem, alg_params, coordinates))
    # Initialize global best
    flower_patches.sort(key=lambda p: p.makespan)
    global_best = flower_patches[0].coordinates
    global_best_val = flower_patches[0].makespan
    # Run search
    for _ in range(alg_params.max_iteration):
        for i in range(alg_params.ns):
            n_bees = 1
            if i < alg_params.ne:
                n_bees = alg_params.nre
            elif i < alg_params.nb:
                n_bees = alg_params.nrb
            if n_bees > 1:
                flower_patches[i] = flower_patches[i].local_search(n_bees)
            else:
                flower_patches[i] = flower_patches[i].global_search()
        flower_patches.sort(key=lambda p: p.makespan)
        if flower_patches[0].makespan < global_best_val:
            global_best = flower_patches[0].coordinates
            global_best_val = flower_patches[0].makespan
            print(global_best_val)
    return_list.append(global_best)


if __name__ == '__main__':
    problem_1 = jssp_io.read_mpso_problem('test_data/1.txt')
    algorithm_parameters = Parameters(max_iteration=500,
                                      ns=50, ne=3, nb=10,
                                      nre=7, nrb=3,
                                      init_hyperbox=2,
                                      shrink=0.8, stlim=10)
    ret_list = []
    bees_algorithms(problem_1, algorithm_parameters, [], ret_list)
    jssp_io.solution_plotter(Solution(problem_1, ret_list[0]))
