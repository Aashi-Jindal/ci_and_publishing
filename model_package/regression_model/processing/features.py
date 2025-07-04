from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class MedianMapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str]):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables
        self.variables_map: Dict[str, float] = dict()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        for var in self.variables:
            med_val = X[~X[var].isna()][var].median()
            self.variables_map[var] = med_val

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        for var in self.variables:
            med_val = self.variables_map[var]
            X.loc[X[var].isna(), var] = med_val

        return X


class FrequentMapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str]):

        if not isinstance(variables, list):

            raise ValueError("variables should be a list")

        self.variables = variables
        self.variables_map: Dict[str, str] = dict()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        for var in self.variables:
            mode_ = X[var].mode()[0]
            self.variables_map[var] = mode_

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        # print("Inside Transform")
        for var in self.variables:
            mode_ = self.variables_map[var]
            X.loc[X[var].isna(), var] = mode_

        return X


class MissingMapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str]):

        if not isinstance(variables, list):

            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        for var in self.variables:
            X.loc[X[var].isna(), var] = "Not Applicable"

        return X


class TemporalMapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str], ref_var: str):

        if not isinstance(variables, list):

            raise ValueError("variables should be a list")

        self.variables = variables
        self.ref_var = ref_var

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
       
        for var in self.variables:
            X.loc[:, var] = X.loc[:, self.ref_var] - X.loc[:, var]

        return X


class DropFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variable: str):

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X.drop(self.variable, axis=1, inplace=True)

        return X


class LogMapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str]):

        if not isinstance(variables, list):

            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        for var in self.variables:
            X[var] = X[var].astype(float)
            X.loc[:, var] = np.log(X[var])

        return X


class OrdinalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str], variable_mapping: Dict[str, int]):

        if not isinstance(variables, list):

            raise ValueError("variables should be a list")

        self.variables = variables
        self.variables_map = variable_mapping

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        for var in self.variables:
            X.loc[:, var] = X[var].map(self.variables_map)

        return X


class RareLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str]):

        self.variables = variables
        self.rare_variables_map: Dict[Any, Dict[Any, Any]] = dict()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        for var in self.variables:
            self.rare_variables_map[var] = dict()
            for val in X[var].unique():
                if X[var].value_counts()[val] / X.shape[0] < 0.01:
                    self.rare_variables_map[var][val] = "rare"
                else:
                    self.rare_variables_map[var][val] = val

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        for var in self.variables:
            print(var)
            for val in X[var].unique():
                if self.rare_variables_map[var][val] != val:
                    X.loc[X[var] == val, var] = "rare"

        return X


class LabelEncoderDef(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str]):

        if not isinstance(variables, list):

            raise ValueError("variables should be a list")

        self.variables = variables
        self.variables_map: Dict[Any, Dict[Any, int]] = dict()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        for var in self.variables:
            le = LabelEncoder()
            le.fit_transform(X[var])
            self.variables_map[var] = {k: v for v, k in enumerate(le.classes_)}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        for var in self.variables:
            X.loc[:, var] = X[var].map(self.variables_map[var])

        return X


class PresentNa(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str]):

        self.variables = [a for a in variables if a != "YrSold"]
        self.variables_na: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        return self

    def transform(self, X: pd.DataFrame):

        for var in self.variables:
            if X[var].isna().sum() > 0:
                self.variables_na.append(var)
                print(X.loc[X[var].isna(), var])

        # print(self.variables_na)
        return X
