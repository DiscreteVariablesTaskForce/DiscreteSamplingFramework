import numpy as np


class RNG():
    """Wrapper class around :class:`numpy.random.default_rng`:
    """

    def __init__(self, seed: int = 0):
        """Constructor method.

        Parameters
        ----------
        seed : int, optional
            Random seed, by default 0
        """
        self.seed = seed
        self.nprng = np.random.default_rng(seed)

    def random(self) -> float:
        """Generate uniform random number from U(0,1)

        Returns
        -------
        float
            Generated random number from (0,1)
        """
        return self.nprng.random()

    def randomInt(self, low: int, high: int) -> int:
        """Generate a random integer uniformly from the closed interval [low, high]

        Parameters
        ----------
        low : int
            Minimum value of the interval.
        high : int
            Maximum value of the interval.

        Returns
        -------
        int
            Generated integer
        """
        if high == low:
            return low

        return self.nprng.integers(low=low, high=high+1)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Generate a random float from the half open interval [low,high)

        Parameters
        ----------
        low : float, optional
            Minimum value, by default 0.0
        high : float, optional
            Maximum value, by default 1.0

        Returns
        -------
        float
            Generated float.
        """
        if high == low:
            return low
        return self.nprng.uniform(low=low, high=high)

    def randomChoice(self, choices: list):
        """Generate a random choice from list of items.

        Parameters
        ----------
        choices : list
            Values to sample from.

        Returns
        -------
        Any
            Randomly selected item from `choices`
        """
        return self.nprng.choice(choices)

    def randomChoices(
        self,
        population: list,
        weights: list[float] | None = None,
        k: int = 1
    ) -> np.array:
        """Sample with replacement from list.

        Parameters
        ----------
        population : list
            List of values to sample from
        weights : list[float] | None, optional
            Sample weights, by default None, meaning all values are equally weighted
        k : int, optional
            Number of samples to generate, by default 1

        Returns
        -------
        :class:`numpy.ndarray`:
            Array of sampled values.
        """
        return self.nprng.choice(population, size=k, replace=True, p=weights)
