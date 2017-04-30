def read_jssp(filename):
    with open(filename, 'r') as f:
        jssp = [line.split() for line in f]
        n, m = jssp[0][0], jssp[0][1]


if __name__ == '__main__':
    read_jssp('test_data/1.txt')