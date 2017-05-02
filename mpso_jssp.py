import random, math, collections
import numpy as np
from jssp_io import read_mpso_problem

Parameters = collections.namedtuple('Parameters',
                                    ['max_iteration', 'pop_size',
                                     'max_omega', 'min_omega',
                                     'prob_mie',
                                     'tf', 'beta',
                                     'prob_s', 'prob_i', 'prob_inv',
                                     'c1', 'c2',
                                     'v_max'])


class Particle(object):

    def __init__(self, jssp_problem):
        self._problem = jssp_problem
        dim = jssp_problem.n * jssp_problem.m
        self._coordinates = np.array(
            [random.uniform(0, dim) for _ in range(dim)])
        self._velocity = np.array(
            [random.random() for _ in range(dim)])
        self._solution = Solution(jssp_problem, self._coordinates)
        self._local_best = self._coordinates[:]
        self._local_best_val = self._solution.makespan

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def solution(self):
        return self._solution

    @property
    def makespan(self):
        return self._solution.makespan

    def _refresh(self):
        self._solution = Solution(self._problem, self._coordinates)
        if self._solution.makespan < self._local_best_val:
            self._local_best = self._coordinates[:]
            self._local_best_val = self._solution.makespan

    def update_velocity(self, omega, global_best, alg_params):
        self._velocity *= omega
        self._velocity += alg_params.c1 * random.random() * (self._local_best - self._coordinates)
        self._velocity += alg_params.c2 * random.random() * (global_best - self._coordinates)
        # Truncate speed for each dimension to v_max
        for i in range(len(self._velocity)):
            if abs(self._velocity[i]) > alg_params.v_max:
                if self._velocity[i] < 0:
                    self._velocity[i] = -alg_params.v_max
                self._velocity[i] = alg_params.v_max

    def move(self):
        self._coordinates += self._velocity
        self._refresh()

    def _swap(self):
        mutated = self._coordinates[:]
        swap_positions = random.sample(range(len(self._coordinates)), 2)
        p, q = swap_positions[0], swap_positions[1]
        mutated[p], mutated[q] = mutated[q], mutated[p]
        return mutated

    def _insert(self):
        mutated = self._coordinates[:]
        swap_positions = random.sample(range(len(self._coordinates)), 2)
        p, q = swap_positions[0], swap_positions[1]
        element = mutated[p]
        del mutated[p]
        mutated.insert(q, element)
        return mutated

    def _inverse(self):
        mutated = self._coordinates[:]
        swap_positions = random.sample(range(len(self._coordinates)), 2).sort()
        p, q = swap_positions[0], swap_positions[1]
        mutated = mutated[:p] + reversed(mutated[p:q+1]) + mutated[q+1:]
        return mutated

    def _long(self):
        mutated = self._coordinates[:]
        swap_positions = random.sample(range(len(self._coordinates)), 2).sort()
        p, q = swap_positions[0], swap_positions[1]
        segment = mutated[p:q+1]
        del mutated[p:q+1]
        r = random.sample(range(len(self._coordinates)), 1)
        mutated = mutated[:r] + segment + mutated[r:]
        return mutated

    def _mutate(self, alg_params):
        q = random.random()
        if 0 <= q <= alg_params.prob_s:
            return self._swap()
        elif alg_params.prob_s < q <= alg_params.prob_s + alg_params.prob_i:
            return self._insert()
        elif alg_params.prob_s + alg_params.prob_i < q <= alg_params.prob_s + alg_params.prob_i + alg_params.prob_inv:
            return self._inverse()
        else:
            return self._long()

    def enhance(self, alg_params, global_best_val):
        t = self.makespan - global_best_val
        while t > alg_params.tf:
            p_prime_coordinates = self._mutate(alg_params)
            p_prime = Solution(self._problem, p_prime_coordinates)
            delta = p_prime.makespan - self._solution.makespan
            if delta < 0 or random.random() < min(1, math.exp(-(delta/t))):
                self._coordinates = p_prime_coordinates
                self._refresh()
            t *= alg_params.beta


class Solution(object):

    def __init__(self, jssp_problem, coordinates):
        self.problem = jssp_problem
        int_series = sorted(range(len(coordinates)),
                            key=lambda i: coordinates[i])
        operations = [i % jssp_problem.n + 1 for i in int_series]
        self.schedule, self._makespan = self._schedule(operations)

    @property
    def makespan(self):
        return self._makespan

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
                machine_free = schedule[machine][-1][1]
            start = max(job_end[op], machine_free)
            end = start + time_rec
            schedule[machine].append((start, end))
            job_operation_tracker[op] += 1
            job_end = end
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


def mpso(jssp_problem, alg_params):
    # Initialize swarm
    swarm = [Particle(jssp_problem) for _ in range(alg_params.pop_size)]
    # Initialize global best
    swarm_min = min(swarm, key=lambda x: x.makespan)
    global_best = swarm_min.coordinates[:]
    global_best_val = swarm_min.makespan
    for i in range(alg_params.max_iteration):
        # Perform local search for particles
        for particle in swarm:
            if random.random() <= alg_params.prob_mie:
                particle.enhance(alg_params, global_best_val)
        # Update global best
        swarm_min = min(swarm, key=lambda x: x.makespan)
        if swarm_min.makespan < global_best_val:
            global_best = swarm_min.coordinates[:]
            global_best_val = swarm_min.makespan
        print(global_best_val)
        # Update omega
        omega = alg_params.max_omega - i * (alg_params.max_omega - alg_params.min_omega) / alg_params.max_iteration
        # Move particles
        for particle in swarm:
            particle.update_velocity(omega, global_best, alg_params)
            particle.move()
    return min(swarm, key=lambda x: x.makespan).solution


if __name__ == '__main__':
    problem_1 = read_mpso_problem('test_data/1.txt')
    algorithm_parameters = Parameters(max_iteration=300, pop_size=30,
                                      max_omega=1.4, min_omega=0.4,
                                      prob_mie=0.01, prob_s=0.4,
                                      prob_i=0.4, prob_inv=0.1,
                                      tf=0.1, beta=0.97,
                                      c1=2, c2=2,
                                      v_max=problem_1.n * problem_1.m * 0.1)
    solution = mpso(problem_1, algorithm_parameters)

