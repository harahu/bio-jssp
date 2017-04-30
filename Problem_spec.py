
def file_to_spec(filename):
    spec = []
    
    with open(filename, 'r') as file:
        content = file.readlines()
        
        content = [x.split() for x in content]
        
        jobs = int(content[0][0])
        machines = int(content[0][1])
        
        content = content[1:]
        
        for i in range(jobs):
            job = content[i]
            line = []
            for j in range(machines):
                line.append(int(job[2*j]))
                line.append(int(job[2*j + 1]))
                
            spec.append(line)
            
    
    return jobs, machines, spec
    
    
    
if __name__ == '__main__':
    x = file_to_spec('./Test Data/2.txt')
    print(x)