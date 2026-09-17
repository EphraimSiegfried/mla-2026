import matplotlib.pyplot as plt
import numpy as np


def main():
    n = 1000000
    m = 20
    p = 0.35
    variance = p * (1 - p) / m
    trials = np.random.binomial(n=1, p=p, size=(n, m))
    avg = 1 / m * np.sum(trials, axis=1)
    thresholds = np.arange(0.5, 1.05, 0.05)
    markov_bound = p / thresholds
    eps = thresholds - p
    chebyshev_bound = variance / eps**2
    hoeffding_bound = np.exp(-2 * m * eps**2)

    freqs = [np.sum(avg >= t) / n for t in thresholds]
    plt.plot(thresholds, freqs, label="Empirical Frequency")
    plt.xlabel("$\\alpha$")
    plt.ylabel("Frequency")
    plt.title("Comparison of Empirical Frequency and Theoretical Bounds")
    plt.savefig("figures/3_1.png")

    plt.plot(thresholds, markov_bound, label="Markov Bound")
    plt.legend()
    plt.savefig("figures/3_3.png")

    plt.plot(thresholds, chebyshev_bound, label="Chebyshev Bound")
    plt.legend()
    plt.savefig("figures/3_4.png")

    plt.plot(thresholds, hoeffding_bound, label="Hoeffding Bound")
    plt.legend()
    plt.savefig("figures/3_5.png")
    plt.show()


if __name__ == "__main__":
    main()
