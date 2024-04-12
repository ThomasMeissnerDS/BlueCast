import numpy as np


def one_c(y_hat: np.ndarray):
    """
    Calculate proportion of singleton sets among all prediction sets.
    :param y_hat: Predicted probabilities of shape (n_samples, 1) where each row is a set of classes.
    """
    singleton_counts = [len(ps[0]) == 1 for ps in y_hat]
    num_singletons = sum(singleton_counts)
    return num_singletons / y_hat.shape[0]
