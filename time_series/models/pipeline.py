# -*- coding: utf-8 -*-
__author__ = 'Jonas Van Der Donckt, Jeroen Van Der Donckt'

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

from typing import List, Dict


class Clipper(BaseEstimator, TransformerMixin):
    """
    Percentile clipper for dataframes in sklearn-pipeline.

    Clipper clips the given columns based on the given (lower and/or upper) percentages.
    This is a wrapper class for sklearn-pipeline which outputs a DataFrame instead of a numpy array.
    """

    def __init__(self, lower_pct: float = None, upper_pct: float = None, selected_cols: List[str] = None):
        """
        :param lower_pct: the lower percentage that will be used to calculate the quantiles to clip the columns.
        :param upper_pct: the upper percentage that will be used to calculate the quantiles to clip the columns.
        :param selected_cols: the columns that need to be clipped.
        """
        self.lower = lower_pct
        self.upper = upper_pct
        self.selected_cols = selected_cols
        self.lower_quantiles = None  # List containing the quantiles for the selected_cols
        self.upper_quantiles = None  # List containing the quantiles for the selected_cols

    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, pd.DataFrame)
        selected_cols = X.columns.values if self.selected_cols is None else self.selected_cols
        if self.lower is not None:
            self.lower_quantiles = X[selected_cols].quantile(self.lower)
        if self.upper is not None:
            self.upper_quantiles = X[selected_cols].quantile(self.upper)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame)
        result_df = X.copy()
        selected_cols = X.columns.values if self.selected_cols is None else self.selected_cols
        result_df[selected_cols] = result_df[selected_cols].clip(lower=self.lower_quantiles, upper=self.upper_quantiles, axis=1)
        return result_df


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Column selector for dataframes in sklearn-pipeline.

    ColumnSelector selects the given columns.
    This is a wrapper class for sklearn-pipeline which outputs a DataFrame instead of a numpy array.
    """

    def __init__(self, columns: List[str]):
        """
        :param columns: the columns that needs to be selected.
        """
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

class ColumnRenamer(BaseEstimator, TransformerMixin):
    """
    Column renamer for dataframes in sklearn-pipeline.

    ColumnRenamer renames the given columns.
    This is a wrapper class for sklearn-pipeline which outputs a DataFrame instead of a numpy array.
    """

    def __init__(self, column_mapping: Dict[str, str]):
        """
        :param column_mapping: the mapping of the original column names to their new names.
        """
        self.column_mapping = column_mapping

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame)

        try:
            return X.rename(columns=self.column_mapping)
        except KeyError:
            cols_error = list(set(self.column_mapping.keys()) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class DataFrameOperator(BaseEstimator, TransformerMixin):
    """
    Dataframe operator for dataframes in sklearn-pipeline.

    DataFrameOperator applies the given operation on the given columns.
    This is a wrapper class for sklearn-pipeline which outputs a DataFrame instead of a numpy array.
    """

    def __init__(self, wrapped_operator, operator_cols: List[str] = None):
        """
        :param wrapped_operator: method that is applied to the columns of the dataframe.
        :param operator_cols: the columns to which the wrapped_operator is applied.
        """
        self.operator_cols = operator_cols
        self.wrapped_operator = wrapped_operator

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        operator_cols = X.columns.values if self.operator_cols is None else self.operator_cols
        self.wrapped_operator.fit(X[operator_cols], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame)
        operator_cols = X.columns.values if self.operator_cols is None else self.operator_cols
        try:
            result_df = pd.DataFrame(index=X.index, data=self.wrapped_operator.transform(X[operator_cols]), columns=operator_cols)
            # Add the unused columns back to the resulting dataframe
            unused_columns = list(set(X.columns).difference(operator_cols))
            result_df[unused_columns] = X[unused_columns]
            return result_df
        except:
            cols_error = list(set(operator_cols).difference(X.columns.values))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class GroupbyDataFrameOperator(DataFrameOperator):
    """
    Groupby Dataframe operator dataframes in sklearn-pipeline.

    GroupbyDataFrameOperator applies the given operation on the different groups for the given columns.
    This is a wrapper class for sklearn-pipeline which outputs a DataFrame instead of a numpy array.
    """

    def __init__(self, wrapped_operator, groupby_col: str, operator_cols: List[str] = None):
        """
        :param wrapped_operator: method that is applied to the grouped columns of the dataframe.
        :param groupby_col: the column name for which the groups are constructed.
        :param operator_cols: the columns to which the wrapped_operator is applied.
        """
        super().__init__(wrapped_operator, operator_cols)
        self.groupby_col = groupby_col
        self.fitted_operators = {}

    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, pd.DataFrame)
        operator_cols = X.columns.values if self.operator_cols is None else self.operator_cols
        operators = X.groupby(self.groupby_col).apply(
            lambda group: self.wrapped_operator.__class__().fit(group[operator_cols]))
        self.fitted_operators = {idx: operator for idx, operator in zip(operators.index, operators.values)}
        return self

    def _transform_group(self, X: pd.DataFrame, group_id):
        assert isinstance(X, pd.DataFrame)
        operator_cols = X.columns.values if self.operator_cols is None else self.operator_cols
        if group_id in self.fitted_operators.keys():
            return self.fitted_operators[group_id].transform(X[operator_cols])
        return self.wrapped_operator.__class__().fit_transform(X[operator_cols])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame)
        operator_cols = X.columns.values if self.operator_cols is None else self.operator_cols
        transformed_df = pd.DataFrame(index=X.index, columns=operator_cols)
        for group_id, group_idxs in X.groupby(self.groupby_col).groups.items():
            assert all(transformed_df[transformed_df.index.isin(group_idxs)].index == group_idxs), 'Order is different!'
            transformed_df.loc[transformed_df.index.isin(group_idxs)] = self._transform_group(
                X[X.index.isin(group_idxs)], group_id)
        # Add the unused columns back to the resulting dataframe
        unused_columns = list(set(X.columns).difference(operator_cols))
        transformed_df[unused_columns] = X[unused_columns]
        return transformed_df


class LogNorm(BaseEstimator, TransformerMixin):
    """
    Log normalization for dataframes in sklearn-pipeline.

    LogNorm log normalizes the given columns.
    This is a wrapper class for sklearn-pipeline which outputs a DataFrame instead of a numpy array.
    """

    def __init__(self, selected_cols: List[str] = None):
        """
        :param selected_cols: the columns that need to be clipped.
        """
        self.selected_cols = selected_cols

    def fit(self, X, y=None):
        # self.min = np.min(X, axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame)
        result_df = X.copy()
        selected_cols = X.columns.values if self.selected_cols is None else self.selected_cols
        result_df[selected_cols] = np.log1p(X[selected_cols] - np.min(X[selected_cols], axis=0))
        return result_df
