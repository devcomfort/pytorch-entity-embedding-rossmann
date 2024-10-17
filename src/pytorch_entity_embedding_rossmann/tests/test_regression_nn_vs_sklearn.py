import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_diabetes

from ..EENNRegression import EntEmbNNRegression
from ..eval_utils import eval_regression


def test_regression_nn_vs_sklearn():
    """
    Compares results from sklearn and eenn Multilayer Perceptron
    """

    # Load the diabetes dataset (from scikit-learn)
    diabetes = load_diabetes()

    # Split the data into training/testing sets
    X_train = pd.DataFrame(diabetes.data[:-20])
    X_test = pd.DataFrame(diabetes.data[-20:])

    # Split the targets into training/testing sets
    y_train = pd.Series(diabetes.target[:-20]) / diabetes.target.max()
    y_test = pd.Series(diabetes.target[-20:]) / diabetes.target.max()

    params = {
        "dense_layers": [10],
        "act_func": "relu",
        "alpha": 0.001,
        "batch_size": 64,
        "lr": 0.001,
        "epochs": 100,
        "rand_seed": 1,
    }

    eenn_model = EntEmbNNRegression(
        cat_emb_dim={},
        dense_layers=params["dense_layers"],
        act_func=params["act_func"],
        alpha=params["alpha"],
        batch_size=params["batch_size"],
        lr=params["lr"],
        epochs=params["epochs"],
        drop_out_layers=[0.0, 0.0],
        drop_out_emb=0.0,
        loss_function="MSELoss",
        train_size=1.0,
        allow_cuda=False,
        verbose=True,
    )

    eenn_model.fit(X_train, y_train)

    eenn_y_pred = eenn_model.predict(X_test)
    eenn_report = eval_regression(y_true=y_test, y_pred=eenn_y_pred)

    sk_model = MLPRegressor(
        hidden_layer_sizes=params["dense_layers"],
        activation=params["act_func"],
        alpha=params["alpha"],
        batch_size=params["batch_size"],
        learning_rate_init=params["lr"],
        max_iter=params["epochs"],
        solver="adam",
        learning_rate="constant",
        validation_fraction=0.0,
        verbose=True,
        momentum=False,
        early_stopping=False,
        epsilon=1e-8,
    )

    sk_model.fit(X_train, y_train)
    sk_y_pred = sk_model.predict(X_test)

    sk_report = eval_regression(y_true=y_test, y_pred=sk_y_pred)

    print(eenn_report[["mean_absolute_error", "mean_squared_error"]])
    print(sk_report[["mean_absolute_error", "mean_squared_error"]])
