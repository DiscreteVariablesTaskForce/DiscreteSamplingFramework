import math
from typing import Union
from pickle import loads, dumps
import numpy as np
from discretesampling.base.random import RNG
from discretesampling.base.kernel import DiscreteVariableOptimalLKernel


class DiscreteVariable:
    """Base class representing discrete variables.
    """

    def __init__(self):
        """Constructor method
        """
        pass

    @classmethod
    def getProposalType(self):
        """Get the relevant proposal distribution type for this variable type

        Returns
        -------
        Type[DiscreteVariableProposal]
            Proposal distribution type for this variable type
        """
        return DiscreteVariableProposal

    @classmethod
    def getTargetType(self):
        """Get the relevant target distribution type for this variable type

        Returns
        -------
        Type[DiscreteVariableTarget]
            Target distribution type for this variable type
        """
        return DiscreteVariableTarget

    @classmethod
    def getLKernelType(self):
        """Get the relevant L-kernel distribution type for this variable type

        Returns
        -------
        Type[DiscreteVariableProposal]
            L-kernel distribution type for this variable type
        """
        # Forward proposal
        return self.getProposalType()

    @classmethod
    def getOptimalLKernelType(self):
        """Get the relevant Optimal L-kernel distribution type for this variable
        type

        Returns
        -------
        Type[DiscreteVariableProposal]
            Optimal L-kernel distribution type for this variable type
        """
        return DiscreteVariableOptimalLKernel

    @classmethod
    def encode(self, x: 'DiscreteVariable') -> np.ndarray:
        """Encode an instance of :class:`DiscreteVariable` to a :class:`numpy.ndarray`

        Parameters
        ----------
        x : DiscreteVariable
            DiscreteVariable object to encode.

        Returns
        -------
        :class:`numpy.ndarray`
            Encoded DiscreteVariable as :class:`numpy.ndarray`

        Notes
        -----
        Default implementation uses `pickle.dumps` to encode the object.
        Classes which inherit from :class:`DiscreteVariable` can optionally implement a more efficient method of
        encoding/decoding.
        """
        encoded = np.array(bytearray(dumps(x)))
        return encoded

    @classmethod
    def decode(self, x: np.ndarray, particle: 'DiscreteVariable') -> 'DiscreteVariable':
        """Decode an encoded :class:`DiscreteVariable`.

        Decodes an encoded :class:`DiscreteVariable` (for example, encoded by
        `encode`) back to a :class:`DiscreteVariable`.

        Parameters
        ----------
        x : np.ndarray
            Encoded variable
        particle : DiscreteVariable
            Instance of DiscreteVariable to use as template for decoding.

        Returns
        -------
        list

        """
        pickle_stopcode = 0x2e
        end_of_pickle_data = np.argwhere(x == pickle_stopcode)[-1][0] + 1
        encoded = np.array(x[0:end_of_pickle_data], dtype=np.uint8)
        decoded = loads(bytes(encoded))
        return decoded


class DiscreteVariableProposal:
    """Base class representing proposal distribution over DiscreteVariables

        Parameters
        ----------
        values : list[DiscreteVariable]
            List of possible values that could be return by the proposal distribution
        probs : list[float]
            List of associated probabilities for the values given in `values`
        rng : RNG, optional
            :class:`RNG` for random number generation, by default RNG()
    """

    def __init__(self, values: list[DiscreteVariable], probs: list[float], rng: RNG = RNG()):
        """Constructor method
        """
        # Check dims and probs are valid
        assert len(values) == len(probs), "Invalid PMF specified, x and p" +\
            " of different lengths"
        probs = np.array(probs)
        tolerance = np.sqrt(np.finfo(np.float64).eps)
        assert abs(1 - sum(probs)) < tolerance, "Invalid PMF specified," +\
            " sum of probabilities !~= 1.0"
        assert all(probs > 0), "Invalid PMF specified, all probabilities" +\
            " must be > 0"
        self.x = values
        self.pmf = probs
        self.cmf = np.cumsum(probs)
        self.rng = rng

    @classmethod
    def norm(self, x: DiscreteVariable) -> float:
        """Calculate the norm metric of a discrete variable

        Parameters
        ----------
        x : DiscreteVariable
            _description_

        Returns
        -------
        float
            _description_

        See Also
        --------
        heuristic : Calculate the possibility of proposing values between two norms

        Notes
        -----
        The norm metric should give some notion of "position" within the space of discrete variables, such that that position
        is meaningful when describing the possible range of values that might be generated by the proposal distribution.
        For example, for :class:`discretesampling.domain.decision_tree.Tree`, the `norm` is simply
        the number of nodes in the tree. 
        """
        return 1.0

    @classmethod
    def heuristic(self, x: Union[int, float, list, tuple], y: Union[int, float, list, tuple]) -> bool:
        """Calculate heuristic for proposal between norms generated from two :class:`DiscreteVariable`s

        Parameters
        ----------
        x : Union[int, float, list, tuple]
            _description_
        y : Union[int, float, list, tuple]
            _description_

        Returns
        -------
        bool
            Heuristic value between :class:`DiscreteVariable`s with norms x and y

        See Also
        --------
        norm : Calculate norm value for :class:`DiscreteVariable`

        Notes
        -----
        Should return true if proposal is possible between x and y (and possibly at other times) where x and y are
        norm values generated by `norm`, but will only return false if
        the proposal between x and y is definitely impossible.

        """
        return True

    def sample(self, target: 'DiscreteVariableTarget' = None) -> DiscreteVariable:
        """Generate a sample from the proposal distribution.

        Parameters
        ----------
        target : DiscreteVariableTarget, optional
            Target distribution for weighting, by default None

        Returns
        -------
        DiscreteVariable
            Sampled value
        """
        q = self.rng.random()  # random unif(0,1)
        return self.x[np.argmax(self.cmf >= q)]

    def eval(self, y: DiscreteVariable, target: 'DiscreteVariableTarget' = None) -> float:
        """Evaluate proposal distribution at value

        Parameters
        ----------
        y : DiscreteVariable
            Value to evaluate proposal distribution at
        target : DiscreteVariableTarget, optional
            Target distribution for weighting, by default None

        Returns
        -------
        float
            Log-probability of proposal distribution at value `y`
        """
        try:
            i = self.x.index(y)
            logp = math.log(self.pmf[i])
        except ValueError:
            print("Warning: value " + str(y) + " not in pmf")
            logp = -math.inf
        return logp


class DiscreteVariableInitialProposal():
    """Base class representing discrete variable initial proposal distribution

    Parameters
        ----------
        values : list[DiscreteVariable]
            List of possible values that could be return by the proposal distribution
        probs : list[float]
            List of associated probabilities for the values given in `values`
    """

    def __init__(self, values, probs):
        """Constructor method
        """
        # Check dims and probs are valid
        assert len(values) == len(probs), "Invalid PMF specified, x and p" +\
            " of different lengths"
        probs = np.array(probs)
        tolerance = np.sqrt(np.finfo(np.float64).eps)
        assert abs(1 - sum(probs)) < tolerance, "Invalid PMF specified," +\
            " sum of probabilities !~= 1.0"
        assert all(probs > 0), "Invalid PMF specified, all probabilities" +\
            " must be > 0"
        self.x = values
        self.pmf = probs
        self.cmf = np.cumsum(probs)

    def sample(self, rng: RNG = RNG(), target: 'DiscreteVariableTarget' = None) -> DiscreteVariable:
        """Generate a sample from initial proposal distribution

        Parameters
        ----------
        rng : RNG, optional
            RNG for random number generation, by default RNG()
        target : DiscreteVariableTarget, optional
            Target distribution for weighting, by default None

        Returns
        -------
        DiscreteVariable
            Sampled value
        """
        q = rng.random()  # random unif(0,1)
        return self.x[np.argmax(self.cmf >= q)]

    def eval(self, y: DiscreteVariable, target: 'DiscreteVariableTarget' = None) -> float:
        """Base class representing discrete variables

        Parameters
        ----------
        y : DiscreteVariable
            Value at which distribution should be evaluated
        target : DiscreteVariableTarget, optional
            Target distribution for weighting, by default None

        Returns
        -------
        float
            Log-probability of distribution at value `y`
        """
        try:
            i = self.x.index(y)
            logp = math.log(self.pmf[i])
        except ValueError:
            print("Warning: value " + str(y) + " not in pmf")
            logp = -math.inf
        return logp


class DiscreteVariableTarget:
    """Base class representing target (or posterior) distributions for
    DiscreteVariables.
    """

    def __init__(self):
        pass

    def eval(self, x: DiscreteVariable) -> float:
        """Evaluate the target distribution.

        Parameters
        ----------
        x : DiscreteVariable
            value at which the target distribution should be evaluated

        Returns
        -------
        float
           log-probability of target distribution at value `x`
        """
        logprob = -math.inf
        logPrior = self.evaluatePrior(x)
        logprob += logPrior
        return logprob

    def evaluatePrior(self, x: DiscreteVariable) -> float:
        logprob = -math.inf
        return logprob
