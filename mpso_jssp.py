import random, math, collections, multiprocessing
import numpy as np
import jssp_io

Parameters = collections.namedtuple('Parameters',
                                    ['max_iteration', 'pop_size',
                                     'max_omega', 'min_omega',
                                     'prob_mie',
                                     'tf', 'beta',
                                     'prob_s', 'prob_i', 'prob_inv',
                                     'c1', 'c2',
                                     'v_max'])


class Particle(object):

    def __init__(self, jssp_problem, coordinates):
        self._problem = jssp_problem
        self._coordinates = coordinates
        self._velocity = np.array(
            [random.random() for _ in range(len(coordinates))])
        self._solution = Solution(jssp_problem, self._coordinates[:])
        self._makespan = self._solution.makespan
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
        for i in range(len(self._velocity)):
            self._velocity[i] *= omega
            self._velocity[i] += alg_params.c1 * random.random() * (self._local_best[i] - self._coordinates[i])
            self._velocity[i] += alg_params.c2 * random.random() * (global_best[i] - self._coordinates[i])
            # Truncate speed to v_max
            if self._velocity[i] > alg_params.v_max:
                self._velocity[i] = alg_params.v_max
            elif self._velocity[i] < -alg_params.v_max:
                self._velocity[i] = -alg_params.v_max

    def move(self):
        self._coordinates += self._velocity
        self._refresh()

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


def mpso(jssp_problem, alg_params, particles, return_list):
    # Initialize swarm
    swarm = particles
    while len(swarm) < alg_params.pop_size:
        dim = jssp_problem.n * jssp_problem.m
        coordinates = np.array(
            [random.uniform(0, dim) for _ in range(dim)])
        swarm.append(Particle(jssp_problem, coordinates))
    # Initialize global best
    swarm_min = min(swarm, key=lambda p: p.makespan)
    global_best = np.copy(swarm_min.coordinates)
    global_best_val = swarm_min.makespan
    for i in range(alg_params.max_iteration):
        # Perform local search for particles
        for particle in swarm:
            if random.random() <= alg_params.prob_mie:
                particle.enhance(alg_params, global_best_val)
        # Update global best
        swarm_min = min(swarm, key=lambda p: p.makespan)
        if swarm_min.makespan < global_best_val:
            global_best = np.copy(swarm_min.coordinates)
            global_best_val = swarm_min.makespan
            print(global_best_val)
        # Update omega
        omega = alg_params.max_omega - i * (alg_params.max_omega - alg_params.min_omega) / alg_params.max_iteration
        # Move particles
        for particle in swarm:
            particle.update_velocity(omega, global_best, alg_params)
            particle.move()
    return_list.append(global_best)


def parallel_mpso(jssp_problem, alg_params, particles):
    n_jobs = min(multiprocessing.cpu_count(), int(alg_params.pop_size/2))
    manager = multiprocessing.Manager()
    return_list = manager.list()
    jobs = []
    for i in range(n_jobs):
        p = multiprocessing.Process(target=mpso, args=(jssp_problem, alg_params, particles, return_list))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    return return_list


if __name__ == '__main__':
    problem_1 = jssp_io.read_mpso_problem('test_data/6.txt')
    algorithm_parameters = Parameters(max_iteration=300, pop_size=30,
                                      max_omega=1.4, min_omega=0.4,
                                      prob_mie=0.01, prob_s=0.4,
                                      prob_i=0.4, prob_inv=0.1,
                                      tf=0.1, beta=0.97, c1=2, c2=2,
                                      v_max=problem_1.n * problem_1.m * 0.1)
    particles_p = []
    i = 0
    end = int(input('Rounds: '))
    while i < end:
        print('Round {}: ---------------'.format(i))
        solutions = parallel_mpso(problem_1, algorithm_parameters, particles_p)
        particles_p = [Particle(problem_1, coord) for coord in solutions]
        particles_p = [min(particles_p, key=lambda p: p.makespan)]
        i += 1
        if i == end:
            print('Continue?')
            end = int(input('End round: '))

    mpso_min = min(particles_p, key=lambda p: p.makespan)
    jssp_io.solution_plotter(mpso_min.solution)
