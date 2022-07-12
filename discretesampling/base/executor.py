import multiprocessing

class Executor(object):
    def map(self, f, *inputs):
        return map(f, *inputs)

class Executor_MP(Executor):
    def __init__(self, num_cores):
        self.num_cores = num_cores
    def map(self, f, *inputs):
        with multiprocessing.Pool(self.num_cores) as pool:
            return pool.map(f, *inputs)