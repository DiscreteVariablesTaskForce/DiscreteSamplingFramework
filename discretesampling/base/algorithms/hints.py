import math
import copy
import numpy as np
from tqdm.auto import tqdm
from discretesampling.base.random import RNG


class DiscreteVariableHINTS():

    def __init__(self, args, fn):
        self.args = args
        self.fn = fn
        if 'NUM_SCENARIOS' in args:
            print("WARNING: NUM_SCENARIOS not specified, using default value of 1")
        args.NUM_SCENARIOS = 1 if ('NUM_SCENARIOS' not in args) else args.NUM_SCENARIOS
        if 'NUM_SAMPLES' in args:
            print("WARNING: HINTS design not specified, using NUM_SENARIOS")
        self.design = args.design if ('design' in args) else np.array([args.NUM_SCENARIOS])
        self.levels = self.design.shape[0] - 1
        self.ns = np.cumprod(self.design)  # number of scenarios at each level
        self.N = self.ns[self.levels]  # total number of scenarios

        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.proposal = proposal
        if proposal is None:
            self.proposal = self.proposalType()
        self.initialProposal = initialProposal
        self.target = target

    def scenarios(self, level, index):
        return ([self.fixed_scenarios[self.ns[level] * index + i] for i in range(self.ns[level])])

    def shuffle(self):
        self.rng.shuffle(self.fixed_scenarios)

    def get_child_index(self, parent_level, parent_index, branch):  # indexing of child nodes
        return (parent_index * self.design[parent_level] + branch)

    def get_parent_index(self, child_level, child_index):  # indexing of parent nodes
        parent_level = child_level+1
        return (child_index//self.design[parent_level])

    def metropolis_accept(self, exp_diff):
        return (1.0 if (exp_diff > 0.0) else (0.0 if (exp_diff < -50.0) else np.exp(exp_diff)))

    def hints(self, state, level, index=0, always_accept=False, adaptive_proposal_params={}, verbose=False):
        if (level == 0):
            return (self.primitive_move(state, index, always_accept, adaptive_proposal_params, verbose=verbose))  # this method is only separated so we can override
        scenarios = self.scenarios(level, index)
        correction = 0.0
        if 'duplicate' in dir(state):
            current = state.duplicate()  # if it's a class, we want to make a deep copy
        else:
            current = state  # we hold state at all levels in the hierarchy
        branches = list(range(self.design[level]))
        if self.shuffle_as_we_go:  # alternative to full shuffle - useful for retaining contiguity
            self.rng.shuffle(branches)
        # loop thru child nodes
        for bi, b in enumerate(branches):
            if (bi % self.downsample) == 0:
                current, delta_correction = self.hints(current, level-1, self.get_child_index(level, index, b), False, adaptive_proposal_params, verbose=verbose)  # recursive call
                correction += delta_correction
        #
        # now do composite evaluations AFTER primitive ones, in case primitive ones needed gradients
        no_change = False
        if self.equality_test is not None:
            no_change = self.equality_test(state, current)
        #
        if no_change:
            # print("Same state: count as reject", state, current)
            accept = False
            aprob = 0.0  # for logging
            v = v_prime = np.nan  # for logging
            vdiff = 0.0  # no need to evaluate
        else:
            if verbose:
                print("HINTS: COMPARISON USING ", min(scenarios), len(scenarios), " at level, index = ", level, index)
            kwargs = {}
            if self.pass_parent_scenarios and (level < self.levels):  # parent scenarios are used to make proxy evaluations on the parent (sub)set
                kwargs = {'parent_scenarios': self.scenarios(level+1, self.get_parent_index(level, index))}
                if verbose:
                    print("first and last parent scenario = ", kwargs['parent_scenarios'][0], kwargs['parent_scenarios'][-1])
            if self.proxy and (level < self.fast_below):  # top level always uses actual evaluations (except in hybrid); lower levels can be proxy only
                kwargs['proxy_only'] = True
            v_prime = self.fn(current, scenarios,  **kwargs)
            v = self.fn(state, scenarios,  **kwargs)
            vdiff = (v_prime - v)/self.Ts[level]  # these are cached evaluations, no side effects
            aprob = self.rule(vdiff - correction) if (level < self.levels) else self.metropolis_accept(vdiff-correction)
            accept = True if always_accept else (self.rng.uniform() < aprob)
        #
        # logging and diagnostics:
        #
        (self.acceptances if accept else self.rejections)[level] += 1
        if (self.execution_trace == "all") or ((self.execution_trace == "root") and (level == self.levels)):
            ldict = {'iteration': self.iteration, 'level': level, 'state': state, 'proposal': current, 'v': v, 'v_prime': v_prime, 'vdiff': vdiff, 'correction': correction, 'accept': accept, 'requests': self.fn.total_counter, 'evaluations': self.fn.counter, 'aprob': aprob}
            if 'logging' in dir(state):  # custom logging
                ldict['state'] = state.logging()
                ldict['proposal'] = current.logging()
            # use shallow copies to save memory
            self.history.append(ldict)
        # output
        return ((current, vdiff) if accept else (state, 0.0))

    def primitive_move(self, state, index=0, always_accept=False, adaptive_proposal_params={}, verbose=False):
        scenarios = self.scenarios(0, index)  # level 0
        if verbose:
            print("leaf index, scenarios = ", index, scenarios)
        kwargs = {}
        if self.pass_parent_scenarios and (self.levels >= 1):
            kwargs = {'parent_scenarios': self.scenarios(1, self.get_parent_index(0, index))}
            if verbose:
                print("first and last parent scenario = ", kwargs['parent_scenarios'][0], kwargs['parent_scenarios'][-1])
        if self.proxy and (0 < self.fast_below):
            kwargs['proxy_only'] = True
        v = self.fn(state, scenarios, **kwargs)  # could put a gradient into state as a side effect for HMC
        current, correction = self.fn.proposal(state, index, **adaptive_proposal_params, rng=self.rng)
        v_prime = self.fn(current, scenarios, **kwargs)
        vdiff = (v_prime - v)/self.Ts[0]  # these are cached evaluations, no side effects
        aprob = self.rule(vdiff - correction)
        accept = True if always_accept else (self.rng.uniform() < aprob)  # MH rule
        #
        # logging and diagnostics:
        (self.acceptances if accept else self.rejections)[0] += 1
        if (self.execution_trace == "all") or ((self.execution_trace == "root") and (level == self.levels)):
            ldict = {'iteration': self.iteration, 'level': 0, 'state': state, 'proposal': current, 'v': v, 'v_prime': v_prime, 'vdiff': vdiff, 'correction': correction, 'accept': accept, 'requests': self.fn.total_counter, 'evaluations': self.fn.counter, 'aprob': aprob}
            if 'logging' in dir(state):  # custom logging
                ldict['state'] = state.logging()
                ldict['proposal'] = current.logging()
            self.history.append(ldict)
        return ((current, vdiff) if accept else (state, 0.0))

    def sample(self, run=0, iterations=None, verbose=False):
        if verbose:
            print("WARNING verbose = True creates a lot of output")
        if run != self.rnum:
            self.reset(run)
        state = self.current_state  # stored state of the chain
        # this can be wrapped in an adaptive control loop
        if iterations is None:
            iterations = self.iterations - self.iteration  # remaining iterations
        #
        # MAIN LOOP
        progress_bar = tqdm(total=iterations, desc="HINTS sampling", disable=verbose)
        for i in range(iterations):
            self.iteration += 1  # cumulative iteration count from multiple calls
            if self.strict and (self.fn.counter > (self.args.max_evaluations//2)):  # if strict, make exact for second half of run
                self.fn.frozen = True
            # if (self.iteration % 10) == 0:
            #    print("Run", run, "iteration", self.iteration)
            counter_before = self.fn.counter
            adp = self.adaptor.get_adaptive_proposal_params() if self.adapt else {}
            state, correction = self.hints(state, self.levels, adaptive_proposal_params=adp, verbose=verbose)
            cost = (self.fn.counter - counter_before)/self.fn.N  # for adaptive strategies that are cost sensitive (ESJD)
            # optional adaptive control
            if self.adapt and not (self.fn.frozen):
                self.adaptor.step(self.iteration, self.history[-1], cost)  # exactly same rule as MCMC   
            if self.proxy & (self.acceptances[-1] >= 8):  # don't allow first few updates into proxy
                top_row = self.history[-1]
                if not (np.equal(top_row['proposal'], top_row['state']).all()):  # (not) no new info
                    self.fn.step_proxy(top_row)
            if not self.shuffle_as_we_go:
                self.shuffle() 
            if ((self.iteration % self.top_level_seed_T) == 0):  # the seed for stochastic likelihoods
                new_seed = self.iteration//self.top_level_seed_T
                print("new top level seed at iteration ", self.iteration, " = ", new_seed)
                self.fn.lockdown_randomness(new_seed)  # this seed could vary by parallel run but may not matter as start state is different anyway
            progress_bar.update(1)

        progress_bar.close()
        self.current_state = state  # backup in case we want to do more sampling
        return (self.history)
