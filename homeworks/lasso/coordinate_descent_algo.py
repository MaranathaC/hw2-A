from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    """Precalculate a vector. You should only call this function once.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.

    Returns:
        np.ndarray: An (d, ) array, which contains a corresponding `a` value for each feature.
    """
    return 2 * np.sum(X ** 2, axis=0)


@problem.tag("hw2-A")
def step(
        X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> Tuple[np.ndarray, float]:
    """Single step in coordinate gradient descent.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        a (np.ndarray): An (d,) array. Respresents precalculated value a that shows up in the algorithm.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
            Bias should be calculated using input weight to this function (i.e. before any updates to weight happen).

    Note:
        When calculating weight[k] you should use entries in weight[0, ..., k - 1] that have already been calculated and updated.
        This has no effect on entries weight[k + 1, k + 2, ...]
    """
    bias = np.mean(y - X.dot(weight))

    for k in range(X.shape[1]):
        except_k = [i for i in range(X.shape[1]) if i != k]
        prediction_wo_k = X[:, except_k].dot(weight[except_k]) + bias
        c_k = X[:, k].dot(y - prediction_wo_k)
        c_k = 2 * np.sum(c_k)

        if c_k < -_lambda:
            weight[k] = (c_k + _lambda) / a[k]
        elif c_k > _lambda:
            weight[k] = (c_k - _lambda) / a[k]
        else:
            weight[k] = 0

    return weight, float(bias)


@problem.tag("hw2-A")
def loss(
        X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    residual = np.sum((y - X.dot(weight) - bias) ** 2)
    return residual + _lambda * np.sum(np.abs(weight))


@problem.tag("hw2-A", start_line=4)
def train(
        X: np.ndarray,
        y: np.ndarray,
        _lambda: float = 0.01,
        convergence_delta: float = 1e-4,
        start_weight: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    a = precalculate_a(X)

    while True:
        old_w = start_weight
        start_weight, bias = step(X, y, start_weight, a, _lambda)
        if convergence_criterion(start_weight, old_w, convergence_delta):
            break

    return start_weight, bias


@problem.tag("hw2-A")
def convergence_criterion(
        weight: np.ndarray, old_w: np.ndarray, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    delta = weight - old_w
    return False if np.max(np.abs(delta)) >= convergence_delta else True


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    X = np.random.normal(0, 1, (500, 1000))
    w = np.zeros(1000)
    k = 100
    for j in range(k):
        w[j] = (j + 1) / k
    y = X.dot(w) + np.random.normal(0, 1, 500)

    def helper(fdr, tpr, xs, ys, _lambda):
        nw = np.copy(w)
        train(X, y, _lambda, 1e-4, nw)
        fdr.append(np.count_nonzero(nw[100:]) / max(1, np.count_nonzero(nw)))
        tpr.append(np.count_nonzero(nw[:100]) / 100)
        xs.append(_lambda)
        ys.append(np.count_nonzero(nw))

    fdr, tpr = [], []
    xs, ys = [], []

    for _lambda in range(1500, 1, -50):
        helper(fdr, tpr, xs, ys, _lambda)

    _lambda = 1
    for i in range(4):
        helper(fdr, tpr, xs, ys, _lambda)
        _lambda -= 0.2

    helper(fdr, tpr, xs, ys, 0)

    plt.plot(xs, ys)
    plt.xlabel('Lambda')
    plt.xscale('log')
    plt.ylabel('Non-zeros')
    plt.show()

    plt.plot(fdr, tpr)
    plt.xlabel('False Discovery Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


if __name__ == "__main__":
    main()
