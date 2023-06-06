from abc import ABC, abstractmethod


class BaseClassMlModel(ABC):
    """Base class for all ML models.

    Enforces the implementation of the fit and predict methods.
    """

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass