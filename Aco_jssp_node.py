
class Node:
    _id = 0
    def __init__(self, job, machine, time, total_nodes):
        self.id = self.__class__._id
        self.__class__._id += 1
        self.job = job
        self.machine = machine
        self.time = time
        
        self.trail = [0 for i in range(total_nodes)]
        
        self.child = None
        
    def __repr__(self):
        ret_str = "ID: " + str(self.id) + ". Node for job " + str(self.job) + " on machine " + str(self.machine) + " using " + str(self.time) + " time. "
        if self.child: ret_str += "Child: [" + str(self.child) + "]"
        else: ret_str += "Last part of job"
        return ret_str
        
    def set_child(self, child):
        self.child = child