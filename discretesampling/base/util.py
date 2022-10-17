from scipy.special import logit, expit


# transforms a continuous unconstrained parameter to the constrained space for a given set of limit values
def u2c(unconstrained_param, min_val, max_val):
    return min_val+(max_val-min_val)*expit(unconstrained_param)


# transforms a continuous constrained parameter to the unconstrained space for a given set of limit values
def c2u(constrained_param, min_val, max_val):
    return logit((constrained_param - min_val) / (max_val - min_val))


# transforms arrays of continuous unconstrained parameters to constrained parameters for a given set of limit values
def u2c_array(unconstrained_params, min_vals, max_vals):
    constrained_params = []
    for param_array, min, max in zip(unconstrained_params, min_vals, max_vals):
        constrained_params.append(min+(max-min)*expit(param_array))
    return constrained_params


# transforms arrays of continuous constrained parameters to unconstrained parameters for a given set of limit values
def c2u_array(constrained_params, min_vals, max_vals):
    unconstrained_params = []
    for param_array, min, max in zip(constrained_params, min_vals, max_vals):
        unconstrained_params.append(logit((param_array-min)/(max-min)))
    return unconstrained_params
