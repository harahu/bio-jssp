import random, math
import numpy as np

class Particle(object):

    def __init__(self, problem):
        self._problem = problem
        dim = problem.n * problem.m
        self._coordinates = np.array(
            [random.uniform(0, dim) for _ in range(dim)])
        self._velocity = np.array(
            [random.random() for _ in range(dim)])
        self._solution = Solution(problem, self._coordinates)
        self._local_best = self._coordinates[:]
        self._local_best_val = self._solution.makespan

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def makespan(self):
        return self._solution.makespan

    def _refresh(self):
        self._solution = Solution(self._problem, self._coordinates)
        if self._solution.makespan < self._local_best_val:
            self._local_best = self._coordinates[:]
            self._local_best_val = self._solution.makespan

    def update_velocity(self, omega, c1, c2, global_best, v_max):
        self._velocity *= omega
        self._velocity += c1 * random.random() * (self._local_best - self._coordinates)
        self._velocity += c2 * random.random() * (global_best - self._coordinates)
        for i in range(len(self._velocity)):
            if self._velocity[i] > v_max:
                self._velocity[i] = v_max

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
        pass

    def _long(self):
        pass

    def _mutate(self, prob_s, prob_i, prob_inv):
        q = random.random()
        if 0 <= q <= prob_s:
            return self._swap()
        elif prob_s < q <= prob_s + prob_i:
            return self._insert()
        elif prob_s + prob_i < q <= prob_s + prob_i + prob_inv:
            return self._inverse()
        else:
            return self._long()

    def enhance(self, t, tf, beta):
        while t > tf:
            p_prime_coordinates = #algorithm2
            p_prime = Solution(self._problem, p_prime_coordinates)
            delta = p_prime.makespan - self._solution.makespan
            if delta < 0 or random.random() < min(1, math.exp(-(delta/T))):
                self._coordinates = p_prime_coordinates
                self._refresh()
            t *= beta


class Solution(object):

    def __init__(self, problem, coordinates):
        self.problem = problem
        int_series = sorted(range(len(coordinates)),
                            key=lambda i: coordinates[i])
        operations = [i % problem.n + 1 for i in int_series]
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


def mpso(max_iteration, max_omega, min_omega, prob_mie, c1, c2, v_max, problem):
    # Initialize swarm
    swarm = [Particle(problem) for _ in range(30)]
    # Initialize global best
    swarm_min = min(swarm, key=lambda x: x.makespan)
    global_best = swarm_min.coordinates[:]
    global_best_val = swarm_min.makespan
    for i in range(max_iteration):
        for particle in swarm:
            if random.random() <= prob_mie:
                particle.enhance()
        # Update global best
        swarm_min = min(swarm, key=lambda x: x.makespan)
        if swarm_min.makespan < global_best_val:
            global_best = swarm_min.coordinates[:]
            global_best_val = swarm_min.makespan
        # Update omega
        omega = max_omega - i * (max_omega - min_omega) / max_iteration
        for particle in swarm:
            particle.update_velocity(omega, c1, c2, global_best, v_max)
            particle.move()
