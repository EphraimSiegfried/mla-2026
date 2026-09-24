import numpy as np
import sklearn


def main():
    X_train = np.genfromtxt("data/X_train.csv", delimiter=",")
    y_train = np.genfromtxt("data/y_train.csv", delimiter=",")
    X_test = np.genfromtxt("data/X_test.csv", delimiter=",")
    y_test = np.genfromtxt("data/y_test.csv", delimiter=",")

    # 3.1
    n = np.shape(y_train)
    unique, counts = np.unique(y_train, return_counts=True)
    freq = counts / n
    print(np.array([unique, freq]))

    # # 3.2
    # logreg = sklearn.linear_model.LogisticRegression(max_iter=10000)
    # logreg.fit(X_train, y_train)
    # train_loss = sklearn.metrics.zero_one_loss(y_train, logreg.predict(X_train))
    # test_loss = sklearn.metrics.zero_one_loss(y_test, logreg.predict(X_test))
    # print(
    #     f"Logistic Regression: test loss: {test_loss:.2f}, train loss: {train_loss:.2f}"
    # )
    #
    # # 3.3
    # for n in [50, 100, 200]:
    #     rf = sklearn.ensemble.RandomForestClassifier(n_estimators=n, oob_score=True)
    #     rf.fit(X_train, y_train)
    #     train_loss = sklearn.metrics.zero_one_loss(y_train, rf.predict(X_train))
    #     test_loss = sklearn.metrics.zero_one_loss(y_test, rf.predict(X_test))
    #     print(
    #         f"Random Forest (n={n}): test loss: {test_loss:.2f}, train loss: {train_loss:.2f}, oob score: {rf.oob_score_:.4f}"
    #     )

    # 3.4
    knn = sklearn.neighbors.KNeighborsClassifier()
    param_grid = {"n_neighbors": np.arange(1, 50)}
    knn_gscv = sklearn.model_selection.GridSearchCV(knn, param_grid, cv=5)
    knn_gscv.fit(X_train, y_train)
    train_loss = sklearn.metrics.zero_one_loss(y_train, knn_gscv.predict(X_train))
    test_loss = sklearn.metrics.zero_one_loss(y_test, knn_gscv.predict(X_test))
    print(
        f"KNN: test loss: {test_loss:.2f}, train loss: {train_loss:.2f}, K: {knn_gscv.best_params_}"
    )


if __name__ == "__main__":
    main()
