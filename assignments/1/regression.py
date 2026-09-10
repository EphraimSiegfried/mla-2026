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


def mean_squared_error(y, yr):
    return np.mean((y - yr) ** 2)


def determination(y, yr):
    return 1 - np.sum((y - yr) ** 2) / np.sum(y - np.mean(y))


def main():
    data_file_path = "PCB.dt"
    data = np.loadtxt(data_file_path)

    # x = data[:, 0]
    # y = data[:, 1]
    # w, b = linear_regression(x, y)
    # yp = w.T * x + b
    # plt.plot(x, y, "x")
    # plt.plot(x, yp)
    # plt.show()

    x = data[:, 0]
    y = data[:, 1]
    x_dense = np.arange(np.min(x), np.max(x), 0.5)
    y_prime = np.log(y)
    w, b = linear_regression(x, y_prime)
    yr = np.exp(w.T * x + b)
    # 3.2
    mse = mean_squared_error(y, yr)
    print(f"h=exp(ax+b): {w[0]=} {b=} {mse=}")

    # 3.4
    fig, ax = plt.subplots()
    ax.plot(x, y, "+", label="data")
    ax.plot(x_dense, np.exp(w * x_dense + b), "-", label="Model h")
    ax.set_yscale("log")
    ax.set_xlabel("Age")
    ax.set_xlabel("PCB residues (ln)")
    ax.set_title("Linear regression of PCB data with $h(x)=exp(ax + b)$")
    ax.legend()
    fig.savefig("figures/regression-h1.png")

    # 3.5
    r2 = determination(y, yr)
    print(f"Coefficient of determination: {r2}")

    # 3.6
    fig2, ax2 = plt.subplots()
    w, b = linear_regression(np.sqrt(x), np.log(y))
    yr = np.exp(w * np.sqrt(x) + b)
    ax2.plot(x, y, "+", label="data")
    ax2.plot(x_dense, np.exp(w * np.sqrt(x_dense) + b), "-", label="Model h")
    mse = mean_squared_error(y, yr)
    print(f"h=exp(a*sqrt(x)+b): {w[0]=} {b=} {mse=}")
    ax2.set_yscale("log")
    ax2.set_xlabel("Age")
    ax2.set_xlabel("PCB residues (ln)")
    ax2.set_title("Linear regression of PCB data with $h(x)=exp(a\sqrt{x} + b)$")
    ax2.legend()
    fig2.savefig("figures/regression-h2.png")
    plt.show()
    r2 = determination(y, yr)
    print(f"Coefficient of determination: {r2}")


if __name__ == "__main__":
    main()
