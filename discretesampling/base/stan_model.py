import math
import os
import logging
import bridgestan as bs
from cmdstanpy import write_stan_json
import numpy as np


class StanModel:
    def __init__(self, model_file):
        self.model_file = os.path.abspath(model_file)
        self.model_filename = os.path.basename(self.model_file)
        self.model_path = os.path.dirname(self.model_file)
        self.exec_name = self.model_filename.replace(".stan", "")
        self.exec_path = os.path.join(self.model_path, self.exec_name)
        self.data_file = self.exec_name + ".data.json"
        self.compiled = False
        self.model = None
        self.lib = None
        self.data = None

    def compile(self):
        if not self.compiled:
            logging.info("Compiling Stan model ", self.exec_name, "with bridge-stan...")
            self.model = bs.StanModel.from_stan_file(self.model_file, self.data_file)
            logging.info("Finished compiling Stan model ", self.exec_name, ".")
            self.compiled = True

    def num_unconstrained_parameters(self, data):
        self.prepare_data(data)
        self.compile()
        return self.model.param_unc_num()

    def eval(self, data, params):
        self.prepare_data(data)
        self.compile()

        logprob = -math.inf
        gradients = None

        try:
            if self.model.param_unc_num() != len(params):
                raise ValueError("Array of incorrect length passed to log_density;"
                                 + " expected {x}, found {y}".format(x=self.model.param_unc_num(), y=len(params)))

            logprob, gradients = self.model.log_density_gradient(theta_unc=np.array(params))

        except RuntimeError:
            logging.error("Something went wrong when trying to evaluate the stan model")

        return logprob, gradients

    def prepare_data(self, data):
        if self.data != data:
            # Cache data
            self.data = data
            # Write params to data file
            write_stan_json(self.data_file, dict(data))
            # If the data changes, the model also needs to be recompiled before it's used
            self.compiled = False
