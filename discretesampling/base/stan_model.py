import subprocess
import re
import math
import os
import logging

class stan_model(object):
    def __init__(self, model_file, redding_stan_path):
        self.model_file = os.path.abspath(model_file)
        self.model_filename = os.path.basename(self.model_file)
        self.model_path = os.path.dirname(self.model_file)
        self.exec_name = self.model_filename.replace(".stan", "")
        self.exec_path = os.path.join(self.model_path,self.exec_name)
        self.data_file = self.exec_name + ".data.R"
        self.redding_stan_path = redding_stan_path
        self.compiled = False

    def compile(self):
        model_path = os.path.join(self.model_path, self.exec_name)
        logging.info("Compiling Stan model ", self.exec_name, "with redding-stan...")
        p = subprocess.Popen(["make", model_path], cwd = self.redding_stan_path)
        p.wait()
        logging.info("Done.")
        self.compiled = True


    def eval(self, data, params):

        if not self.compiled:
            self.compile()
        

        self.prepare_data(data)

        logprob = -math.inf
        
        data_command = "load " + self.data_file + "\n"
        
        
        eval_command = "eval "
        for i in range(len(params)):
            eval_command = eval_command + str(params[i])
            if i < len(params)-1:
                eval_command = eval_command + ","
        eval_command = eval_command + "\n"
        
        
        proc = subprocess.Popen(self.exec_path, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

        proc.stdin.write(data_command.encode())    
        proc.stdin.flush()
        proc.stdin.write(eval_command.encode())
        proc.stdin.flush()
        
        prompt = 0
        
        result = []
        while True:
            output  = proc.stdout.readline()
            
            #find first prompt
            if re.match("\[redding\]\$", output.decode()) is not None:
                prompt = prompt + 1
            if prompt > 1:
                #Found second prompt, what follows is the output we care about
                #Grab three lines of output
                result.append(output)
                for i in range(2):
                    output = proc.stdout.readline()
                    
                    result.append(output)
                break

        proc.stdin.write('quit\n'.encode())
        proc.kill()
        
        text_results = [s.decode().strip() for s in result]
        text_results[0] = text_results[0].replace("[redding]$ ", "")

        logprob = float(text_results[0])
        gradient_strings = text_results[1].split()
        gradients = [float(x) for x in gradient_strings]
        exec_time = float(text_results[2])
        return logprob

    def prepare_data(self, data):
        #Write params to data file
        with open(self.data_file, "w") as f:
            for d in data:
                param_name = str(d[0])
                param_value = ""
                if type(d[1]) is list:
                    param_value = "c("
                    for i in range(len(d[1])):
                        param_value = param_value + str(d[1][i])
                        if i < len(d[1])-1:
                            param_value = param_value + ","
                    param_value = param_value + ")"
                else:
                    param_value = str(d[1])
                f.write(param_name + " <- " + param_value + "\n")
            f.close()