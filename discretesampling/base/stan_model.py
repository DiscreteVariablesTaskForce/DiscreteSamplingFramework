import subprocess
import math
import os
import logging
import bridgestan.python.bridgestan as bs
from cmdstanpy import write_stan_json
import numpy as np

class stan_model(object):
    def __init__(self, model_file, bridgestan_path, cmdstan_path):
        self.model_file = os.path.abspath(model_file)
        self.model_filename = os.path.basename(self.model_file)
        self.model_path = os.path.dirname(self.model_file)
        self.exec_name = self.model_filename.replace(".stan", "")
        self.exec_path = os.path.join(self.model_path,self.exec_name)
        self.data_file = self.exec_name + ".data.json"
        self.bridgestan_path = os.path.abspath(bridgestan_path)
        self.cmdstan_path = os.path.abspath(cmdstan_path)
        self.compiled = False
        self.model = None
        self.lib = None
        self.data = None

    def compile(self):
        model_path = os.path.join(self.model_path, self.exec_name)
        logging.info("Compiling Stan model ", self.exec_name, "with bridge-stan...")
        self.lib = os.path.join(self.model_path, self.exec_name + "_model.so")
        cmdstan_cmd = "CMDSTAN=" + self.cmdstan_path + "/"

        p = subprocess.Popen(["make", cmdstan_cmd, self.lib], cwd = self.bridgestan_path)
        p.wait()
        logging.info("Done.")
        self.compiled = True

    def eval(self, data, params):

        if not self.compiled:
            self.compile()

        self.prepare_data(data)
        self.model = bs.StanModel(self.lib, self.data_file)

        logprob = -math.inf
        gradients = None

        try:
            if self.model.param_unc_num() != len(params):
                raise ValueError("Array of incorrect length passed to log_density; expected {x}, found {y}".format(x=self.model.param_unc_num(), y=len(params)))
            
            tmp_params = np.array([0.0]*self.model.param_unc_num())
            logprob = self.model.log_density(theta_unc=tmp_params)                

        except RuntimeError as err:
            logging.error("Something went wrong when trying to evaluate the stan model")

        return logprob, gradients

    def prepare_data(self, data):
        #Write params to data file
        write_stan_json(self.data_file, dict(data))
        
