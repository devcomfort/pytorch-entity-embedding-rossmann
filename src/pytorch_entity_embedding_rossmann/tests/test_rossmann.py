import numpy as np
import torch

from ..datasets import get_rossmann
from ..eval_utils import MAPE
from ..EENNRegression import EntEmbNNRegression

import os

torch.set_num_threads(os.cpu_count())  # 사용할 스레드 수 설정


def test_rossman():
    """
    Reproduces rossman results.
    """

    X, y, X_test, y_test = get_rossmann()

    for data in [X, X_test]:
        data.drop("Open", inplace=True, axis=1)

    for data in [X, X_test]:
        for f in data.columns:
            data[f] = data[f].cat.codes

    y = np.log(y) / np.log(41551)
    y_test = np.log(y_test) / np.log(41551)

    models = []
    for random_seed in range(5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            verbose=True,
        )

        self.to(device)

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
