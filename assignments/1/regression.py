import matplotlib.pyplot as plt
import numpy as np


# assuming
# dimensions of x: N x d
# dimensions of y: N x 1
def linear_regression(x: np.ndarray, y: np.ndarray):
    n = np.shape(x)[0]
    x = np.c_[x, np.ones(n)]
    wt = np.linalg.solve(x.T @ x, x.T @ y)
    w = wt[:-1]
    b = wt[-1]
    return w, b


def main():
    data_file_path = "PCB.dt"
    data = np.loadtxt(data_file_path)

    x = data[:, 0]
    y = data[:, 1]
    w, b = linear_regression(x, y)
    yp = w.T * x + b
    plt.plot(x, y, "x")
    plt.plot(x, yp)
    plt.show()


if __name__ == "__main__":
    main()
