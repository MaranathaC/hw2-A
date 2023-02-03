if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")
    X = df_train.drop(columns=["ViolentCrimesPerPop"], axis=1).values
    y = df_train["ViolentCrimesPerPop"].values
    w = np.zeros(X.shape[1])
    xs, ys = [], []

    def helper(_lambda):
        nw = np.copy(w)
        train(X, y, _lambda, 0.1, nw)
        xs.append(_lambda)
        ys.append(np.count_nonzero(nw))

    for _lambda in range(1500, 1, -2):
        helper(_lambda)

    _lambda = 1
    for i in range(4):
        helper(_lambda)
        _lambda -= 0.2

    helper(0)

    plt.plot(xs, ys)
    plt.xlabel('Lambda')
    plt.xscale('log')
    plt.ylabel('Non-zeros')
    plt.show()


if __name__ == "__main__":
    main()
