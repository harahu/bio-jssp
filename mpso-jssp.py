import random

class Particle(object):

    def __init__(self, dim):
        self.coordinates = [random.uniform(0, dim) for _ in range(dim)]

    def makespan(self):
        pass

    def enhance(self):
        # Update local best
        pass


def mpso(max_iteration, n, m, prob_mie):
    swarm = [Particle(n * m) for _ in range(30)]
    for _ in range(max_iteration):
        for particle in swarm:
            if random.random() <= prob_mie:
                particle.enhance()
        # Update swarm global best
        # update omega
        for particle in swarm:
            # Update position
            pass

def individual_enhancement_scheme(particle):
    """Individual enhancement local search, based on simulated annealing"""
    while T > T_f:
        #Select operation and create new individual (algorithm 2)
        delta = difference_in_makespan
        if delta > 0 #p' is worse than p