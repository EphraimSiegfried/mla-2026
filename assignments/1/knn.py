import matplotlib.pyplot as plt
import numpy as np


def knn(
    training_points: np.ndarray,
    training_labels: np.ndarray,
    test_points: np.ndarray,
    test_labels: np.ndarray,
):
    _, m = np.shape(training_points)
    _, n = np.shape(test_points)
    distances = (
        np.outer(np.diagonal(training_points.T @ training_points), np.ones(n))
        - 2 * training_points.T @ test_points
        + np.outer(np.ones(m), np.diagonal(test_points.T @ test_points))
    )
    distances_argsort = np.argsort(distances, axis=0)

    labels_ext = np.tile(training_labels, (n, 1)).T
    labels_sorted = np.take_along_axis(labels_ext, distances_argsort, axis=0)
    predictions = np.cumsum(labels_sorted, axis=0)
    predictions = np.where(predictions > 0, 1, -1)

    error = predictions != test_labels
    avg_error = np.average(error, axis=1)
    return avg_error


def transform_labels(labels):
    return np.array([-1 if x == 5 else 1 for x in labels])


def main():
    # training_points = np.array([[2, 3, 4], [1, 2, 3], [0, 4, 19], [2, 20, 5]])
    # training_labels = np.array([5, 5, 6])
    # test_points = np.array([[2, 6, 4], [8, 2, 3], [9, 4, 6], [10, 4, 18]])
    # test_labels = np.array([6, 5, 5])
    data_file_path = "MNIST-5-6-Subset/MNIST-5-6-Subset.txt"
    data_matrix = np.loadtxt(data_file_path).reshape(1877, 784)
    labels_file_path = "MNIST-5-6-Subset/MNIST-5-6-Subset-Labels.txt"
    labels = np.loadtxt(labels_file_path)
    labels = transform_labels(labels)

    ns = [10, 20, 40, 80]
    m = 50
    ks = np.array([i for i in range(1, m + 1)])
    fig2, ax2 = plt.subplots()

    for n in ns:
        split_indices = [m + (i - 1) * n for i in range(1, 7)]
        data_split = np.split(data_matrix, split_indices)[:-1]
        labels_split = np.split(labels, split_indices)[:-1]
        training_points = data_split[0]
        training_labels = labels_split[0]
        fig, ax = plt.subplots()

        avg_errors = []
        for i in range(1, 6):
            test_points = data_split[i]
            test_labels = labels_split[i]
            avg_error = knn(
                training_points.T, training_labels, test_points.T, test_labels
            )
            avg_errors.append(avg_error)
            ax.plot(ks, avg_error, label=f"Validation Set {i}")

        variances = np.var(np.array(avg_errors), axis=0)
        ax2.plot(ks, variances, label=f"n = {n}")

        ax.set_xlabel("k (Nearest Neighbors)")
        ax.set_ylabel(f"Average error over {n} samples")
        ax.set_title(f"KNN Error Rate vs. k (n = {n})")
        ax.legend()
        fig.savefig(f"figures/knn-error-rate-n={n}.png")

    ax2.set_xlabel("k (Nearest Neighbors)")
    ax2.set_ylabel("Variance")
    ax2.set_title("KNN Variance over validation sets vs. k ")
    ax2.legend()
    fig2.savefig("figures/knn-error-variance.png")


if __name__ == "__main__":
    main()
