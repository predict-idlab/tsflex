# -*- coding: utf-8 -*-
__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt"

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


from typing import List, Any, Optional


def to_dataframe_operator(operator: Any, operator_cols: Optional[List[str]] = None):
    """Convert the given (transformer) operator to a DataFrameOperator.

    Parameters
    ----------
    operator: Any
        The operator that should be transformed to a `DataFrameOperator`.
    operator_cols: List[str], optional
        The list of column names on which the operator should operate, by default None.

    Returns
    -------
    A ``DataFrameOperator`` from the given operator when eligable (i.e, an operator
    that is instance of ``TransformerMixin`` and is not yet ``DataFrameOperator``),
    otherwise the operator is just returned.

    """
    if isinstance(operator, DataFrameOperator):
        return operator
    elif isinstance(operator, TransformerMixin):
        return DataFrameOperator(operator, operator_cols)
    else:
        return operator


class DataFrameOperator(BaseEstimator, TransformerMixin):
    """Dataframe operator (transformer) for using dataframes in sklearn-pipeline.

    This is a wrapper class for `sklearn.pipeline.Pipeline` which outputs a DataFrame
    instead of a numpy array. The wrapped operator applies the given operation on
    the given operator_columns (when passed to the constructor).

    Parameters
    ---------
    wrapped_operator: TransformerMixin
        Operator (transformer) that needs to be wrapped.
    operator_cols:  List[str], optional
        The list of column names on which the operator should operate, by default None.

    """

    def __init__(
        self,
        wrapped_operator: TransformerMixin,
        operator_cols: Optional[List[str]] = None,
    ):
        """Create a dataframeoperator."""
        self.wrapped_operator = wrapped_operator
        self.operator_cols = operator_cols

    def fit(self, X: pd.DataFrame, y=None):
        # TODO: document?
        assert isinstance(X, pd.DataFrame)
        operator_cols = (
            X.columns.values if self.operator_cols is None else self.operator_cols
        )
        print(self.wrapped_operator, operator_cols)
        self.wrapped_operator.fit(X[operator_cols], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # TODO: document?
        assert isinstance(X, pd.DataFrame)
        operator_cols = (
            X.columns.values if self.operator_cols is None else self.operator_cols
        )
        try:
            data = self.wrapped_operator.transform(X[operator_cols])
            result_df = pd.DataFrame(index=X.index, data=data, columns=operator_cols)
            # Add the unused columns back to the resulting dataframe
            unused_columns = list(set(X.columns).difference(operator_cols))
            result_df[unused_columns] = X[unused_columns]
            return result_df
        except:
            cols_error = list(set(operator_cols).difference(X.columns.values))
            raise KeyError(f"The DataFrame does not include the columns: {cols_error}")
