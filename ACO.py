
import Aco_jssp_node as AJN
import Problem_spec as PS

class Root_node:
    def __init__(self):
        self.children = []
        
    def __repr__(self):
        return "Root Node with children: " + str(self.children)
        
    def set_child(self, child):
        self.children.append(child)
        
def build_tree(spec, jobs, machines):
    root = Root_node()
    total_nodes = jobs*machines
    for i in range(len(spec)):
        job = spec[i]
        prev_node = root
        for j in range(len(job) // 2):
            node = AJN.Node(i, job[2*j], job[2*j+1], total_nodes)
            prev_node.set_child(node)
            prev_node = node
            
    return root
    
def ACO(spec, ant_amt):
    root = build_tree(spec)
    

if __name__ == '__main__':
    _,_,sp = PS.file_to_spec('./Test Data/1.txt')
    root = build_tree(sp)
    print(root)
    