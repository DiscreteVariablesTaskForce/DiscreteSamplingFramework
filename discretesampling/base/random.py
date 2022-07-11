from random import randint
import random
import numpy as np


class Random(object):
    def eval(self):
        return random.random()


class RandomInt(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def eval(self):
        return randint(self.low, self.high)


class RandomChoice(object):
    def __init__(self, choices):
        self.choices = choices

    def eval(self):
        return random.choice(self.choices)


class RandomChoices(object):
    def __init__(self, population, weights=None, cum_weights=None, k=1):
        self.population = population
        self.weights = weights
        self.cum_weights = cum_weights
        self.k = k

    def eval(self):
        return random.choices(population=self.population, weights=self.weights, cum_weights=self.cum_weights, k=self.k)


class Dice(object):
    def __init__(self, probabilities, outcomes):
        assert len(outcomes) == len(probabilities), "Invalid PMF specified, x and p" +\
             " of different lengths"
        probabilities = np.array(probabilities)
        tolerance = np.sqrt(np.finfo(np.float64).eps)
        assert abs(1 - sum(probabilities)) < tolerance, "Invalid PMF specified," +\
            " sum of probabilities !~= 1.0"
        assert all(probabilities >= 0.), "Invalid PMF specified, all probabilities" +\
            " must be > 0"
        self.probabilities = probabilities
        self.outcomes = outcomes
        self.pmf = probabilities
        self.cmf = np.cumsum(probabilities)
        self.randomiser = Random()

    def eval(self):
        q = self.randomiser.eval()
        x = self.outcomes[np.argmax(self.cmf >= q)]

        while callable(x):
            x = x()
        return x


def set_seed(seed):
    """
    :param seed: random seed
    """
    random.seed(seed)


set_seed(42)
