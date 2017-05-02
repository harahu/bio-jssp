from mpso_jssp import Problem


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
