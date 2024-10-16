#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:14:58 2018

@author: lsanchez
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from .eval_utils import MAPE, eval_regression, classification_report
from .EENNRegression import EntEmbNNRegression
from .EENNBinary import EntEmbNNBinary

# 데이터셋 모듈 호출
from sklearn.datasets import load_diabetes
from .datasets import get_X_train_test_data


def test_rossman():
    """
    Reproduces rossman results.
    """

    X, y, X_test, y_test = get_X_train_test_data()

    for data in [X, X_test]:
        data.drop("Open", inplace=True, axis=1)

    for data in [X, X_test]:
        for f in data.columns:
            data[f] = data[f].cat.codes

    y = np.log(y) / np.log(41551)
    y_test = np.log(y_test) / np.log(41551)

    models = []
    for random_seed in range(5):
        self = EntEmbNNRegression(
            cat_emb_dim={
                "Store": 10,
                "DayOfWeek": 6,
                "Promo": 1,
                "Year": 2,
                "Month": 6,
                "Day": 10,
                "State": 6,
            },
            alpha=0,
            epochs=10,
            dense_layers=[1000, 500],
            drop_out_layers=[0.0, 0.0],
            drop_out_emb=0.0,
            loss_function="L1Loss",
            output_sigmoid=True,
            lr=0.001,
            train_size=1.0,
            random_seed=random_seed,
        )

        self.fit(X, y)
        models.append(self)
        print("\n")

    test_y_pred = np.array([model.predict(X_test) for model in models])
    test_y_pred = test_y_pred.mean(axis=0)

    print(
        "Ent.Emb. Neural Net MAPE: %s"
        % MAPE(
            y_true=np.exp(y_test.values.flatten() * np.log(41551)),
            y_pred=np.exp(test_y_pred * np.log(41551)),
        )
    )


def test_regression_pure_neural_net_vs_sklearn():
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


def test_classification_pure_neural_net_vs_sklearn():
    """
    Compares results from sklearn and eenn Multilayer Perceptron
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
