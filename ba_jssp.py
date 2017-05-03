import random, collections, multiprocessing
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

    def _swap(self):
        mutated = np.copy(self._coordinates)
        swap_positions = random.sample(range(len(self._coordinates)), 2)
        p, q = swap_positions[0], swap_positions[1]
        mutated[p], mutated[q] = mutated[q], mutated[p]
        return mutated

    def _insert(self):
        mutated = np.copy(self._coordinates)
        swap_positions = random.sample(range(len(self._coordinates)), 2)
        p, q = swap_positions[0], swap_positions[1]
        element = mutated[p]
        mutated = np.append(mutated[:p], mutated[p+1:])
        mutated = np.insert(mutated, q, element)
        return mutated

    def _inverse(self):
        mutated = np.copy(self._coordinates)
        swap_positions = sorted(random.sample(range(len(self._coordinates)), 2))
        p, q = swap_positions[0], swap_positions[1]
        segment = mutated[p:q+1]
        segment = segment[::-1]
        mutated = np.append(np.append(mutated[:p], segment), mutated[q+1:])
        return mutated

    def _long(self):
        mutated = np.copy(self._coordinates)
        swap_positions = sorted(random.sample(range(len(self._coordinates)), 3))
        if random.random() <= 0.5:
            p, q, r = swap_positions[0], swap_positions[1], swap_positions[2]
        else:
            r, p, q = swap_positions[0], swap_positions[1], swap_positions[2]
        segment = mutated[p:q+1]
        mutated = np.append(mutated[:p], mutated[q+1:])
        mutated = np.append(np.append(mutated[:r], segment), mutated[r:])
        return mutated

    def _mutate_local(self):
        prob_s = 0.4
        prob_i = 0.4
        prob_inv = 0.1
        q = random.random()
        if 0 <= q <= prob_s:
            return self._swap()
        elif prob_s < q <= prob_s + prob_i:
            return self._insert()
        elif prob_s + prob_i < q <= prob_s + prob_i + prob_inv:
            return self._inverse()
        else:
            return self._long()

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
            if random.random() < 0.8:
                search_coord = self._random_local()
            else:
                search_coord = self._mutate_local()
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


def bees_algorithm(jssp_problem, alg_params, points, return_list):
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
        for i in range(len(flower_patches)):
            n_bees = 1
            if i < alg_params.ne:
                n_bees = alg_params.nre
            elif i < alg_params.nb:
                n_bees = alg_params.nrb
            if n_bees > 1:
                flower_patches[i].local_search(n_bees)
            else:
                flower_patches[i].global_search()
        flower_patches.sort(key=lambda p: p.makespan)
        if flower_patches[0].makespan < global_best_val:
            global_best = flower_patches[0].coordinates
            global_best_val = flower_patches[0].makespan
            print(global_best_val)
    return_list.append(global_best)


def parallel_ba(jssp_problem, alg_params, points):
    n_jobs = min(multiprocessing.cpu_count(), int(alg_params.ns/2))
    manager = multiprocessing.Manager()
    return_list = manager.list()
    jobs = []
    for i in range(n_jobs):
        p = multiprocessing.Process(target=bees_algorithm, args=(jssp_problem, alg_params, points, return_list))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    return return_list


if __name__ == '__main__':
    fname = '6.txt'
    problem_1 = jssp_io.read_mpso_problem('test_data/'+fname)
    dim = problem_1.n * problem_1.m
    algorithm_parameters = Parameters(max_iteration=100,
                                      ns=200, ne=3, nb=10,
                                      nre=dim, nrb=int(dim/4),
                                      init_hyperbox=3,
                                      shrink=0.80, stlim=15)
    solutions = []
    i = 0
    end = int(input('Rounds: '))
    while i < end:
        print('Round {}: ---------------'.format(i))
        solutions = parallel_ba(problem_1, algorithm_parameters, solutions)
        i += 1
        if i == end:
            print('Continue?')
            end = int(input('End round: '))

    solutions = [Solution(problem_1, coordinates) for coordinates in solutions]
    ba_min = min(solutions, key=lambda p: p.makespan)
    jssp_io.solution_plotter(ba_min, fname)
