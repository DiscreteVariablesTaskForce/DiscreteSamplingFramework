from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.random import RNG
from discretesampling.base.stan_model import StanModel


class ContinuousSampler(ABC):
    """Abstract base class for continuous samplers

        Parameters
        ----------
        model : StanModel
            Stan model desribing continuous posterior
        data_function : Callable
            Function which generates Stan model data from a :class:`DiscreteVariable`
        """

    def __init__(self, model: StanModel, data_function: Callable, **kwargs):
        """Constructor method
        """
        self.model = model
        self.data_function = data_function

    @abstractmethod
    def sample(
        self,
        current_continuous: np.ndarray,
        current_discrete: DiscreteVariable,
        rng=RNG(),
        ** kwargs
    ) -> np.ndarray:
        """Sample vector of parameters in continuous space

        Parameters
        ----------
        current_continuous : np.ndarray
            Current array of parameters in continuous space
        current_discrete : DiscreteVariable
            Current value in discrete space

        Returns
        -------
        np.ndarray
            Sampled parameters in continuous space
        """
        pass

    @abstractmethod
    def eval(
        self,
        current_continuous: np.ndarray,
        current_discrete: DiscreteVariable,
        proposed_continuous: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        pass
