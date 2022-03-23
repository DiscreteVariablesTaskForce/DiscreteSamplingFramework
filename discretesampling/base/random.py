from random import randint
import random


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


def set_seed(seed):
    """
    :param seed: random seed
    """
    random.seed(seed)


set_seed(42)
