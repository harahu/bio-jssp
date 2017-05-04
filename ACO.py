from sys import argv
import multiprocessing
import random as rand
import numpy as np

import jssp_io
import mpso_jssp


class Node:
    _ID = 0

    def __init__(self, job, machine, time, total_nodes):
        self._id = Node._ID
        Node._ID += 1
        
        self._job = job
        self._machine = machine
        self._time = time
        
        if self._id == 0: self.child = []
        else: self.child = None
        
        self.trail = {}
        for i in range(total_nodes):
            if i+1 != self._id: self.trail[i+1] = 0.5
            
    def set_child(self, child):
        if self._id == 0:
            self.child.append(child)
        else:
            self.child = child
            
    def __repr__(self):
        return "ID: " + str(self.id) + ", job: " + str(self.job) + ", machine: " + str(self.machine) + ", time: " + str(self.time) + "." + " Child(" + str(self.child) + ")"
           
    @property
    def id(self):
        return self._id
    
    @property
    def job(self):
        return self._job
        
    @property
    def machine(self):
        return self._machine
        
    @property
    def time(self):
        return self._time
        
    def reset_trail(self, total_nodes):
        for i in range(total_nodes):
            if i+1 != self.id: self.trail[i+1] = 1.0


def build_tree(problem_spec):
    total_nodes = problem_spec.n*problem_spec.m
    root = Node(None, None, 1, total_nodes)
    
    nodes = [root]
    
    for i in range(problem_spec.n):
        prev_node = root
        
        for j in range(problem_spec.m):
            node = Node(i, problem_spec.jobs[i][j][0], problem_spec.jobs[i][j][1], total_nodes)
            prev_node.set_child(node)
            
            prev_node = node
            nodes.append(node)
            
    return nodes


def encode_to_local_search(route, jobs, machines):
    encoded = []
    for item in route:
        encoded_val = (item-1) // jobs
        encoded.append(encoded_val)
        
    return encoded


def decode_to_trail(encoded, jobs, machines):
    job_head = [0 for i in range(jobs)]
    
    route = []
    for item in encoded:
        decoded_val = item*jobs + job_head[item] + 1
        route.append(decoded_val)
        job_head[item] += 1
        
    return route


def HACO_Single(nodes, ant_amt, heur_const=10, alpha=1, beta=1):
    routes = []
    for i in range(ant_amt):
        route = []
        open = [node for node in nodes[0].child]
        
        curr_node = nodes[0]
        while(open):
            trail_heur_sum = 0
            choice_prob = []
            for node in open:
                trail_heur = curr_node.trail[node.id]**alpha * (heur_const/curr_node.time)**beta
                trail_heur_sum += trail_heur
                choice_prob.append(trail_heur_sum)
        
            choice_prob = [i/trail_heur_sum for i in choice_prob]
        
            choice = None
            r = rand.random()
            for i in range(len(open)):
                if r <= choice_prob[i]: 
                    choice = open[i]
                    break
            
            open.remove(choice)
            
            route.append(choice)
            
            if choice.child: open.append(choice.child)
            
            curr_node = choice
            
        routes.append(route)
        
    routes = [[node.id for node in route] for route in routes]
           
    return routes


def f_mmas(x, t_min = 0.001, t_max = 1.999):
    if x < t_min: return t_min
    if x > t_max: return t_max
    return x


def swap(encoded_route):
    mutated = np.copy(encoded_route)
    swap_positions = rand.sample(range(len(encoded_route)), 2)
    p, q = swap_positions[0], swap_positions[1]
    mutated[p], mutated[q] = mutated[q], mutated[p]
    return mutated


def insert(encoded_route):
    mutated = np.copy(encoded_route)
    swap_positions = rand.sample(range(len(encoded_route)), 2)
    p, q = swap_positions[0], swap_positions[1]
    element = mutated[p]
    mutated = np.append(mutated[:p], mutated[p+1:])
    mutated = np.insert(mutated, q, element)
    return mutated


def inverse(encoded_route):
    mutated = np.copy(encoded_route)
    swap_positions = sorted(rand.sample(range(len(encoded_route)), 2))
    p, q = swap_positions[0], swap_positions[1]
    segment = mutated[p:q+1]
    segment = segment[::-1]
    mutated = np.append(np.append(mutated[:p], segment), mutated[q+1:])
    return mutated


def long(encoded_route):
    mutated = np.copy(encoded_route)
    swap_positions = sorted(rand.sample(range(len(encoded_route)), 3))
    if rand.random() <= 0.5:
        p, q, r = swap_positions[0], swap_positions[1], swap_positions[2]
    else:
        r, p, q = swap_positions[0], swap_positions[1], swap_positions[2]
    segment = mutated[p:q+1]
    mutated = np.append(mutated[:p], mutated[q+1:])
    mutated = np.append(np.append(mutated[:r], segment), mutated[r:])
    return mutated


def mutate(encoded_route, prob_s, prob_i, prob_inv):
        q = rand.random()
        if q <= prob_s:
            return swap(encoded_route)
        elif q <= prob_s + prob_i:
            return insert(encoded_route)
        elif q <= prob_s + prob_i + prob_inv:
            return inverse(encoded_route)
        else:
            return long(encoded_route)


def local_search(route, prob_s, prob_i, prob_inv, local_search_size, problem_spec):
    encoded = encode_to_local_search(route, jobs, machines)
    sol = mpso_jssp.Solution(problem_spec, encoded)
    search_space = [(sol.makespan, encoded)]
    for j in range(local_search_size):
        mutant = mutate(encoded, prob_s, prob_i, prob_inv)
        mut_sol = mpso_jssp.Solution(problem_spec, mutant)
        search_space.append((mut_sol.makespan, mutant))
    search_space.sort(key=lambda x: x[0])
    decoded = decode_to_trail(search_space[0][1], jobs, machines)
    
    return search_space[0][0], decoded


def HACO(problem_spec, nodes, jobs, machines, target, return_list, generations = 100, evaporation = 0.01, trail_constant = 10.0, local_search_size = 200):
    ants = max(10, machines // 10)
    prob_s=0.4
    prob_i=0.4
    prob_inv=0.1
    
    best = []
    best_makespan = 10000000000
    for g in range(generations):
        if g % 10 == 0:
            print("Generation"+str(g))
        routes = HACO_Single(nodes, ants)
        
        for i in range(len(routes)):
            makespan, improved = local_search(routes[i], prob_s, prob_i, prob_inv, local_search_size, problem_spec)
            if g in [0, 1, 2, 3]:
                sd = 30
            else:
                sd = 1
            for j in range(sd):
                makespan, improved = local_search(improved, prob_s, prob_i, prob_inv, local_search_size, problem_spec)
            
            routes[i] = (makespan, improved)
            
            if makespan < best_makespan:
                best_makespan = makespan
                best = routes[i][1]
        
        for node in nodes:
            for key in node.trail:
                node.trail[key] *= (1-evaporation)
                
        for route_str in routes:
            for i in range(len(route_str[1])-1):
                start = route_str[1][i]
                end = route_str[1][i+1]
                
                t_start_end = nodes[start].trail[end]
                t_start_end += trail_constant / route_str[0]
                t_start_end = f_mmas(t_start_end)
                
                nodes[start].trail[end] = t_start_end
                
        routes = [route[1] for route in routes]

        if best_makespan <= target: break
        
        # for node in nodes: print(node.trail)

    return_list.append(best)
    return best


def parallel_HACO(prob, nodes, jobs, machines, target):
    n_jobs = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    return_list = manager.list()
    jobs = []
    for i in range(n_jobs):
        p = multiprocessing.Process(target=HACO, args=(prob, nodes, jobs, machines, target, return_list))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    return return_list
       
if __name__ == "__main__":
    filename = "./test_data/" + str(6) + ".txt"
    target = int(1000)
    prob = jssp_io.read_mpso_problem(filename)
    
    jobs = prob.n
    machines = prob.m
    
    prob_s=0.4
    prob_i=0.4
    prob_inv=0.1
    
    nodes = build_tree(prob)
    
    best = parallel_HACO(prob, nodes, jobs, machines, target)
    solutions = []
    for b in best:
        enc = encode_to_local_search(b, jobs, machines)
        sol = mpso_jssp.Solution(prob, enc)
        solutions.append(sol)
    solutions.sort(key=lambda s: s.makespan)
    sol = solutions[0]
    jssp_io.solution_plotter(sol, filename)
    '''
    routes = HACO_Single(nodes, 10)
    
    test_route = routes[0]
    test_route = encode_to_local_search(test_route, jobs, machines)
    test_solution = mpso_jssp.Solution(prob, test_route)
    
    print("Initial makespan: " + str(test_solution.makespan))
    
    all_routes = [(test_solution.makespan, test_route, test_solution)]
    for i in range(500):
        print("Local run " + str(i))
        for j in range(1000):
            mutant = mutate(all_routes[0][1], prob_s, prob_i, prob_inv)
            mutant = list(mutant)
            mut_sol = mpso_jssp.Solution(prob, mutant)
            all_routes.append((mut_sol.makespan, mutant, mut_sol))
            #rint(mut_sol.makespan)
            
        all_routes.sort(key=lambda x: x[0])
        all_routes = [all_routes[0]]
        print(all_routes[0][0])
    
    jssp_io.solution_plotter(all_routes[0][2], filename)
    '''