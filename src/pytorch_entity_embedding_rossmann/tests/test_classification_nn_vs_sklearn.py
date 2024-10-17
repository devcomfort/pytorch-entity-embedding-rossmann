import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ..eval_utils import classification_report
from ..EENNBinary import EntEmbNNBinary


def test_classification_nn_vs_sklearn():
    """
    Compares results from sklearn and Entity Embedding NeuralNet Multilayer Perceptron
    """

    X, y = make_classification(n_samples=1000, weights=[0.1, 0.9])

    X, y = pd.DataFrame(X), pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    params = {
        "dense_layers": [10],
        "act_func": "relu",
        "alpha": 0.001,
        "batch_size": 64,
        "lr": 0.001,
        "epochs": 100,
        "rand_seed": 1,
    }

    eenn_model = EntEmbNNBinary(
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

    eenn_model.fit(X=X_train, y=y_train)

    eenn_y_pred = eenn_model.predict(X_test)
    eenn_y_score = eenn_model.predict_proba(X_test)

    eenn_report = classification_report(
        y_true=y_test, y_pred=eenn_y_pred, y_score=eenn_y_score
    )

    sk_model = MLPClassifier(
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
    sk_y_score = sk_model.predict_proba(X_test)

    sk_report = classification_report(
        y_true=y_test, y_pred=sk_y_pred, y_score=sk_y_score
    )

    print(eenn_report)
    print(sk_report)
