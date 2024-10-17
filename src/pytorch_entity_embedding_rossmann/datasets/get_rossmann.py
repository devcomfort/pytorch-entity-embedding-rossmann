#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:57:36 2018

@author: raul

NOTE: hand-configure DATAPATH var. in this script to the folder
where files will be downloaded.

Original codes from:
    https://github.com/entron/entity-embedding-rossmann/blob/master/extract_csv_files.py
    https://github.com/entron/entity-embedding-rossmann/blob/master/prepare_features.py
    https://github.com/entron/entity-embedding-rossmann/blob/master/train_test_model.py
"""

import os
import csv
import random
import pandas as pd
import numpy as np

from .download_rossmann import download_rossmann
from .csv2dicts import csv2dicts
from .set_nan_as_string import set_nan_as_string
from .env import DATAPATH, ROSSMANN_PATH


random.seed(42)
np.random.seed(123)


def get_rossmann(simulate_sparsity: bool = True):
    """
    Computes the dataset used in:

    @article{guo2016entity,
      title={Entity embeddings of categorical variables},
      author={Guo, Cheng and Berkhahn, Felix},
      journal={arXiv preprint arXiv:1604.06737},
      year={2016}
    }

    After first computation files are saved and cached as HDF.

    Args:
        simulate_sparsity (bool): 희소 데이터 여부를 결정합니다. True인 경우, 훈련 데이터셋을 언더샘플링하여 사용합니다

    Return:
        X_train: pandas.DataFrame
        y_train: pandas.Series
        X_test: pandas.DataFrame
        y_test: pandas.Series
    """

    # Rossmann 파일이 다운로드 되지 않았다면 다운로드 하기
    if not os.path.exists(ROSSMANN_PATH):
        download_rossmann()

    # HDF 데이터 저장 경로 생성
    if simulate_sparsity:
        dataset_output_path = "%s/X_train_test_sparse.hdf" % DATAPATH
    else:
        dataset_output_path = "%s/X_train_test_no_sparse.hdf" % DATAPATH

    if not os.path.exists(dataset_output_path):
        # 해당 경로에 파일이 없다면, csv 파일을 읽어 HDF로 변환, 저장합니다

        train_data_path = "%s/train.csv" % DATAPATH
        with open(train_data_path) as csvfile:
            train_data = csv.reader(csvfile, delimiter=",")
            train_data = csv2dicts(train_data)
            train_data = train_data[::-1]

        store_data_path = "%s/store.csv" % DATAPATH
        with open(store_data_path) as csvfile:
            store_data = csv.reader(csvfile, delimiter=",")
            store_data = csv2dicts(store_data)

        store_states_path = "%s/store_states.csv" % DATAPATH
        with open(store_states_path) as csvfile2:
            state_data = csv.reader(csvfile2, delimiter=",")
            state_data = csv2dicts(state_data)

        set_nan_as_string(store_data)
        for index, val in enumerate(store_data):
            state = state_data[index]
            val["State"] = state["State"]
            store_data[index] = val

        """
        prepare_features.py
        """
        train_data = pd.DataFrame(train_data)
        is_open_defined = train_data["Open"] != ""
        has_sales = train_data["Sales"] != "0"

        train_data = train_data[(has_sales) & (is_open_defined)]

        """
        The following lines does what the function:
            
            feature_list(record, store_data)
            
        used to do in script prepare_features.py, but in pandas-like
        operations.
        """
        train_data["Date"] = pd.to_datetime(train_data["Date"])
        train_data["Store"] = train_data["Store"].astype(int)
        train_data["Year"] = train_data["Date"].dt.year.values
        train_data["Month"] = train_data["Date"].dt.month.values
        train_data["Day"] = train_data["Date"].dt.day.values
        train_data["DayOfWeek"] = train_data["DayOfWeek"].astype(int)
        train_data["Open"] = train_data["Open"].astype(int)
        train_data["Promo"] = train_data["Promo"].astype(int)
        train_data["State"] = train_data["Store"].apply(
            lambda x: store_data[x - 1]["State"]
        )

        cols = ["Open", "Store", "DayOfWeek", "Promo", "Year", "Month", "Day", "State"]
        train_data_X = train_data[cols]

        train_data_y = train_data["Sales"].astype(int)

        for c in train_data_X.columns:
            train_data_X[c] = train_data_X[c].astype("category").cat.as_ordered()

        # Train/test split 수행

        train_ratio = 0.9  # 훈련 데이터셋 비율 (기본값: 0.9 = 90%)
        train_size = int(train_ratio * train_data_X.shape[0])

        X_train = train_data_X[:train_size]
        y_train = train_data_y[:train_size]

        X_test = train_data_X[train_size:]
        y_test = train_data_y[train_size:]

        X_train.head(10)
        X_train.tail(10)
        # Simulate data sparsity
        if simulate_sparsity:
            # TODO: 이 언더 샘플링 샘플 수(size)가 실제 샘플 수보다 많은지, 적은지 검토하고 어떤 영향을 줄 지 생각해보아야 함
            size = 200000

            idx = np.random.randint(X_train.shape[0], size=size)

            X_train = X_train.iloc[idx]
            y_train = y_train.iloc[idx]

        X_train.to_hdf(dataset_output_path, key="X_train", format="table")
        X_test.to_hdf(dataset_output_path, key="X_test", format="table")
        y_train.to_hdf(dataset_output_path, key="y_train", format="table")
        y_test.to_hdf(dataset_output_path, key="y_test", format="table")
    else:
        # HDF로 저장된 데이터가 있는 경우, 해당 데이터를 로드하여 반환합니다

        X_train = pd.read_hdf(dataset_output_path, key="X_train")
        X_test = pd.read_hdf(dataset_output_path, key="X_test")
        y_train = pd.read_hdf(dataset_output_path, key="y_train")
        y_test = pd.read_hdf(dataset_output_path, key="y_test")

    return X_train, y_train, X_test, y_test
