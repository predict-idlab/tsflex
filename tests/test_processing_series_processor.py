"""Tests for the processing functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pytest
import pandas as pd
import numpy as np

from tsflex.processing import dataframe_func
from tsflex.processing import SeriesProcessor

from typing import List
from pandas.testing import assert_index_equal, assert_series_equal
from .utils import dummy_data, dataframe_to_series_dict, series_to_series_dict


## Function wrappers


def test_dataframe_func_decorator(dummy_data):
    # Create undecorated dataframe function
    def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    # Create decorated dataframe function
    @dataframe_func
    def drop_nans_decorated(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    # Insert some NANs into the dummy_data
    assert not np.any(pd.isna(dummy_data))  # Check that there are no NANs present
    dummy_data.iloc[:10] = pd.NA

    # Undecorated datframe function
    dataframe_f = drop_nans
    assert np.any(pd.isna(dummy_data))  # Check if there are some NANs present
    res = dataframe_f(dummy_data)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (dummy_data.shape[0] - 10,) + dummy_data.shape[1:]
    assert not np.any(pd.isna(res))  # Check that there are no NANs present

    # Decorated datframe function
    decorated_dataframe_f = drop_nans_decorated
    assert np.any(pd.isna(dummy_data))  # Check if there are some NANs present
    res = decorated_dataframe_f(dummy_data["EDA"], dummy_data["TMP"])
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (dummy_data.shape[0] - 10,) + (2,)
    assert not np.any(pd.isna(res))  # Check that there are no NANs present


## Various output types


def test_dataframe_output(dummy_data):
    # Create dataframe output function
    def duplicate_with_offset(series, offset: float) -> pd.DataFrame:
        offset = abs(offset)
        df = pd.DataFrame()
        df["TMP" + "+" + str(offset)] = series + offset
        df["TMP" + "-" + str(offset)] = series - offset
        return df

    dataframe_f = duplicate_with_offset
    offset = 0.5

    inp = dummy_data["TMP"]
    assert isinstance(inp, pd.Series)
    series_dict = series_to_series_dict(inp)
    assert isinstance(series_dict, dict)

    # Raw series function
    res = dataframe_f(inp, offset)
    assert isinstance(res, pd.DataFrame)
    assert (res.shape[0] == len(dummy_data["TMP"])) & (res.shape[1] == 2)
    assert np.all(res[f"TMP+{offset}"] == inp + offset)
    assert np.all(res[f"TMP-{offset}"] == inp - offset)

    # SeriesProcessor with series function
    processor = SeriesProcessor(
        series_names=["TMP"], function=dataframe_f, offset=offset
    )
    res = processor(series_dict)
    assert isinstance(res, dict)
    assert len(res.values()) == 2
    assert all([res_val.shape[0] == len(dummy_data["TMP"]) for res_val in res.values()])
    assert np.all(res[f"TMP+{offset}"] == inp + offset)
    assert np.all(res[f"TMP-{offset}"] == inp - offset)


def test_series_output(dummy_data):
    # Create series output function
    def to_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series)

    single_series_f = to_numeric

    inp = dummy_data["TMP"].astype(str)
    assert isinstance(inp, pd.Series)

    # Undecorated series function
    assert not np.issubdtype(inp, np.number)
    res = single_series_f(inp)
    assert isinstance(res, pd.Series)
    assert res.shape == dummy_data["TMP"].shape
    assert np.issubdtype(res, np.number)

    # Decorated series function
    processor = SeriesProcessor(series_names=["TMP"], function=single_series_f)
    series_dict = series_to_series_dict(inp)
    assert not np.issubdtype(series_dict["TMP"], np.number)
    res = processor(series_dict)
    assert isinstance(res, dict)
    assert res.keys() == series_dict.keys()
    assert isinstance(res["TMP"], pd.Series)
    assert res["TMP"].shape == dummy_data["TMP"].shape
    assert np.issubdtype(res["TMP"], np.number)


def test_series_list_output(dummy_data):
    # Create series output function
    def to_numeric_abs(series: pd.Series) -> List[pd.Series]:
        return [
            pd.to_numeric(series),
            pd.Series(np.abs(pd.to_numeric(series)), name=f"{series.name}_abs"),
        ]

    series_list_f = to_numeric_abs

    inp = dummy_data["TMP"].astype(str)
    assert isinstance(inp, pd.Series)

    # Undecorated series function
    assert not np.issubdtype(inp, np.number)
    res = to_numeric_abs(inp)
    assert isinstance(res, list)
    assert len(res) == 2
    assert [s.name for s in res] == ["TMP", "TMP_abs"]
    assert all([isinstance(s, pd.Series) for s in res])
    assert all([s.shape == dummy_data["TMP"].shape for s in res])
    assert np.issubdtype(res[0], np.number)
    assert all(res[1] > 0)

    # Decorated series function
    processor = SeriesProcessor(series_names=["TMP"], function=series_list_f)
    series_dict = series_to_series_dict(inp)
    assert not np.issubdtype(series_dict["TMP"], np.number)
    res = processor(series_dict)
    assert isinstance(res, dict)
    assert len(res) == 2
    assert res.keys() == set(["TMP", "TMP_abs"])
    assert all([isinstance(s, pd.Series) for s in res.values()])
    assert all([s.shape == dummy_data["TMP"].shape for s in res.values()])
    assert np.issubdtype(res["TMP"], np.number)
    assert all(res["TMP_abs"] > 0)


def test_numpy_func(dummy_data):
    # Create numpy function
    def numpy_is_close_med(sig: np.ndarray) -> np.ndarray:
        return np.isclose(sig, np.median(sig))

    numpy_f = numpy_is_close_med

    inp = dummy_data["TMP"]

    # Undecorated numpy function
    numpy_f = numpy_is_close_med
    assert isinstance(inp.values, np.ndarray)
    res = numpy_f(inp.values)
    assert isinstance(res, np.ndarray)
    assert res.shape == dummy_data["TMP"].shape
    assert res.dtype == np.bool8
    assert sum(res) > 0  # Check if at least 1 value is True

    # Decorated series function
    processor = SeriesProcessor(series_names=["TMP"], function=numpy_f)
    series_dict = series_to_series_dict(inp)
    assert isinstance(inp, pd.Series)
    res = processor(series_dict)
    assert isinstance(res, dict)
    assert res.keys() == series_dict.keys()
    assert isinstance(res["TMP"], pd.Series)
    assert res["TMP"].shape == dummy_data["TMP"].shape
    assert np.issubdtype(res["TMP"], np.bool8)
    assert sum(res["TMP"]) > 0  # Check if at least 1 value is True


def test_raw_numpy_func(dummy_data):
    numpy_f = np.abs
    inp = dummy_data["TMP"]

    # Raw numpy function
    assert isinstance(inp.values, np.ndarray)
    res = numpy_f(inp.values)
    assert isinstance(res, np.ndarray)
    assert res.shape == dummy_data["TMP"].shape
    assert all(res >= 0)

    # Decorated series function
    processor = SeriesProcessor(series_names=["TMP"], function=numpy_f)
    series_dict = series_to_series_dict(inp)
    assert isinstance(inp, pd.Series)
    res = processor(series_dict)
    assert isinstance(res, dict)
    assert res.keys() == series_dict.keys()
    assert isinstance(res["TMP"], pd.Series)
    assert res["TMP"].shape == dummy_data["TMP"].shape
    assert all(res["TMP"] >= 0)


def test_series_numpy_func(dummy_data):
    # Create series numpy function
    def normalized_freq_scale(series: pd.Series) -> np.ndarray:
        # NOTE: this is a really useless function, but it highlights a legit use case
        sr = 1 / pd.to_timedelta(pd.infer_freq(series.index)).total_seconds()
        return np.interp(series, (series.min(), series.max()), (0, sr))

    series_numpy_f = normalized_freq_scale

    inp = dummy_data["TMP"]

    # Undecorated numpy function
    assert isinstance(inp, pd.Series)
    res = series_numpy_f(inp)
    assert isinstance(res, np.ndarray)
    assert res.shape == dummy_data["TMP"].shape
    assert res.dtype == np.float64
    assert (min(res) == 0) & (max(res) > 0)

    # # Decorated series function
    processor = SeriesProcessor(series_names=["TMP"], function=series_numpy_f)
    series_dict = series_to_series_dict(inp)
    assert isinstance(inp, pd.Series)
    res = processor(series_dict)
    assert isinstance(res, dict)
    assert res.keys() == series_dict.keys()
    assert isinstance(res["TMP"], pd.Series)
    assert res["TMP"].shape == dummy_data["TMP"].shape
    assert np.issubdtype(res["TMP"], np.number)
    assert (min(res["TMP"]) == 0) & (max(res["TMP"]) > 0)


## SeriesProcessor

### Test 'normal' use-cases


def test_single_signal_series_processor(dummy_data):
    def to_binary(series, thresh_value):
        return series.map(lambda eda: eda > thresh_value)

    thresh = 0.6
    series_processor = SeriesProcessor(
        series_names=["EDA"], function=to_binary, thresh_value=thresh
    )
    assert series_processor.name == "to_binary"
    assert series_processor.get_required_series() == ["EDA"]
    series_dict = dataframe_to_series_dict(dummy_data)
    res = series_processor(series_dict)

    assert isinstance(res, dict)
    assert res.keys() == set(["EDA"])
    assert isinstance(res["EDA"], pd.Series)
    assert res["EDA"].shape == dummy_data["EDA"].shape

    assert all(res["EDA"][dummy_data["EDA"] <= thresh] == False)
    assert all(res["EDA"][dummy_data["EDA"] > thresh] == True)


def test_multi_signal_series_processor(dummy_data):
    def percentile_clip(series, l_perc=0.01, h_perc=0.99):
        # Note: this func is useless in ML (data leakage; percentiles are not fitted)
        l_thresh = np.percentile(series, l_perc * 100)
        h_thresh = np.percentile(series, h_perc * 100)
        return series.clip(l_thresh, h_thresh)

    lower = 0.02
    upper = 0.99  # The default value => do not pass
    series_processor = SeriesProcessor(
        series_names=["EDA", "TMP"], function=percentile_clip, l_perc=lower
    )
    assert series_processor.name == "percentile_clip"
    assert len(series_processor.get_required_series()) == 2
    assert set(series_processor.get_required_series()) == set(["EDA", "TMP"])
    series_dict = dataframe_to_series_dict(dummy_data)
    res = series_processor(series_dict)

    assert isinstance(res, dict)
    assert res.keys() == set(["EDA", "TMP"])
    assert isinstance(res["EDA"], pd.Series)
    assert isinstance(res["TMP"], pd.Series)
    assert res["EDA"].shape == dummy_data["EDA"].shape
    assert res["TMP"].shape == dummy_data["TMP"].shape

    assert min(res["EDA"]) == dummy_data["EDA"].quantile(lower)
    assert max(res["EDA"]) == dummy_data["EDA"].quantile(upper)
    assert min(res["TMP"]) == dummy_data["TMP"].quantile(lower)
    assert max(res["TMP"]) == dummy_data["TMP"].quantile(upper)


def test_numpy_replace_series_processor(dummy_data):
    def scale(sig: pd.Series) -> np.ndarray:
        return np.array(sig - np.mean(sig)) / np.std(sig)

    # Check if scale returns a numpy array
    scaled_tmp = scale(dummy_data["TMP"])
    assert isinstance(scaled_tmp, np.ndarray)

    series_processor = SeriesProcessor(
        series_names=["ACC_x", "ACC_y", "ACC_z"],
        function=scale,
    )
    series_dict = dataframe_to_series_dict(dummy_data)
    res = series_processor(series_dict)

    # Check if the series get replaced correctly
    assert res.keys() == set(["ACC_x", "ACC_y", "ACC_z"])
    for key in ["ACC_x", "ACC_y", "ACC_z"]:
        res_series = res[key]
        assert isinstance(res_series, pd.Series)
        assert_index_equal(res_series.index, dummy_data[key].index)
        assert_series_equal(
            res_series,
            (dummy_data[key] - np.mean(dummy_data[key])) / np.std(dummy_data[key]),
        )


def test_series_name_replace_series_processor(dummy_data):
    def scale(sig: pd.Series) -> pd.Series:
        return (sig - np.mean(sig)) / np.std(sig)

    # Check if scale returns a series wit a name
    scaled_tmp = scale(dummy_data["TMP"])
    assert isinstance(scaled_tmp, pd.Series)
    assert scaled_tmp.name == "TMP"

    series_processor = SeriesProcessor(
        series_names=["ACC_x", "ACC_y", "ACC_z"],
        function=scale,
    )
    series_dict = dataframe_to_series_dict(dummy_data)
    res = series_processor(series_dict)

    # Check if the series get replaced correctly
    assert res.keys() == set(["ACC_x", "ACC_y", "ACC_z"])
    for key in ["ACC_x", "ACC_y", "ACC_z"]:
        res_series = res[key]
        assert isinstance(res_series, pd.Series)
        assert_index_equal(res_series.index, dummy_data[key].index)
        assert_series_equal(
            res_series,
            (dummy_data[key] - np.mean(dummy_data[key])) / np.std(dummy_data[key]),
        )


def test_series_no_name_replace_series_processor(dummy_data):
    # A series gets no name when you perform multi-series operations (e.g, subtraction)
    def scale(sig: pd.Series) -> pd.Series:
        return pd.Series((sig - np.mean(sig)) / np.std(sig)).rename()

    # Check if scale returns a series without a name
    scaled_tmp = scale(dummy_data["TMP"])
    assert isinstance(scaled_tmp, pd.Series)
    assert scaled_tmp.name is None

    series_processor = SeriesProcessor(
        series_names=["ACC_x", "ACC_y", "ACC_z"],
        function=scale,
    )
    series_dict = dataframe_to_series_dict(dummy_data)
    res = series_processor(series_dict)

    # Check if the series get replaced correctly
    assert res.keys() == set(["ACC_x", "ACC_y", "ACC_z"])
    for key in ["ACC_x", "ACC_y", "ACC_z"]:
        res_series = res[key]
        assert isinstance(res_series, pd.Series)
        assert_index_equal(res_series.index, dummy_data[key].index)
        assert_series_equal(
            res_series,
            (dummy_data[key] - np.mean(dummy_data[key])) / np.std(dummy_data[key]),
        )


def test_new_output_series_processor(dummy_data):
    def abs_diff(sig1, sig2):
        return pd.Series(np.abs(sig1 - sig2), name=f"abs_diff_{sig1.name}-{sig2.name}")

    series_processor = SeriesProcessor(
        series_names=[("EDA", "TMP")],
        function=abs_diff,
    )
    series_dict = dataframe_to_series_dict(dummy_data)
    res = series_processor(series_dict)

    assert res.keys() == set(["abs_diff_EDA-TMP"])
    assert_index_equal(res["abs_diff_EDA-TMP"].index, dummy_data.index)
    assert_series_equal(
        res["abs_diff_EDA-TMP"],
        np.abs(dummy_data["EDA"] - dummy_data["TMP"]),
        check_names=False,
    )


def test_new_multi_output_series_processor(dummy_data):
    def sum_diff(sig1, sig2):
        sum_sigs = sig1 + sig2
        diff_sigs = sig1 - sig2
        return pd.DataFrame(
            {
                f"sum_{sig1.name}_{sig2.name}": sum_sigs,
                f"diff_{sig1.name}_{sig2.name}": diff_sigs,
            },
        )

    series_processor = SeriesProcessor(
        series_names=[("EDA", "TMP")],
        function=sum_diff,
    )
    series_dict = dataframe_to_series_dict(dummy_data)
    res = series_processor(series_dict)

    assert res.keys() == set(["diff_EDA_TMP", "sum_EDA_TMP"])
    assert_index_equal(res["diff_EDA_TMP"].index, dummy_data.index)
    assert_series_equal(
        res["diff_EDA_TMP"],
        dummy_data["EDA"] - dummy_data["TMP"],
        check_names=False,
    )
    assert_index_equal(res["sum_EDA_TMP"].index, dummy_data.index)
    assert_series_equal(
        res["sum_EDA_TMP"],
        dummy_data["EDA"] + dummy_data["TMP"],
        check_names=False,
    )


### Test 'error' use-cases


def test_error_dataframe_no_time_index_series_processor(dummy_data):
    def absx2(sig):
        return pd.DataFrame(np.abs(sig.values * 2), columns=[sig.name])

    # Check if abs_diff returns a series without a name
    absx2_tmp = absx2(dummy_data["TMP"])
    assert isinstance(absx2_tmp, pd.DataFrame)
    assert absx2_tmp.columns.values == ["TMP"]
    assert not isinstance(absx2_tmp.index, pd.DatetimeIndex)

    series_processor = SeriesProcessor(
        series_names=["EDA", "TMP"],
        function=absx2,
    )
    series_dict = dataframe_to_series_dict(dummy_data)

    with pytest.raises(Exception):
        _ = series_processor(series_dict)


def test_error_dataframe_no_name_single_input_series_processor(dummy_data):
    # If you perform multi-series operation on series with different / no names
    # => the output will also have no name
    def absx2(sig):
        return pd.DataFrame(np.abs(sig ** 2 - sig.rename("")))

    # Check if abs_diff returns a series without a name
    absx2_tmp = absx2(dummy_data["TMP"])
    assert isinstance(absx2_tmp, pd.DataFrame)
    assert absx2_tmp.columns.values == [0]

    series_processor = SeriesProcessor(
        series_names=["EDA", "TMP"],
        function=absx2,
    )
    series_dict = dataframe_to_series_dict(dummy_data)

    with pytest.raises(Exception):
        _ = series_processor(series_dict)


def test_error_dataframe_no_name_multi_input_series_processor(dummy_data):
    # If you perform multi-series operation on series with different / no names
    # => the output will also have no name
    def abs_diff(sig1, sig2):
        return pd.DataFrame(np.abs(sig1 - sig2))

    # Check if abs_diff returns a series without a name
    abs_diff_tmp_eda = abs_diff(dummy_data["TMP"], dummy_data["EDA"])
    assert isinstance(abs_diff_tmp_eda, pd.DataFrame)
    assert abs_diff_tmp_eda.columns.values == [0]

    series_processor = SeriesProcessor(
        series_names=[("EDA", "TMP")],
        function=abs_diff,
    )
    series_dict = dataframe_to_series_dict(dummy_data)

    with pytest.raises(Exception):
        _ = series_processor(series_dict)


def test_error_series_no_time_index_series_processor(dummy_data):
    def absx2(sig):
        return pd.Series(np.abs(sig.values * 2), name=sig.name)

    # Check if abs_diff returns a series without a name
    absx2_tmp = absx2(dummy_data["TMP"])
    assert isinstance(absx2_tmp, pd.Series)
    assert absx2_tmp.name == "TMP"
    assert not isinstance(absx2_tmp.index, pd.DatetimeIndex)

    series_processor = SeriesProcessor(
        series_names=["EDA", "TMP"],
        function=absx2,
    )
    series_dict = dataframe_to_series_dict(dummy_data)

    with pytest.raises(Exception):
        _ = series_processor(series_dict)


def test_error_series_no_name_multi_input_series_processor(dummy_data):
    def abs_diff(sig1, sig2):
        return pd.Series(np.abs(sig1 - sig2)).rename()

    # Check if abs_diff returns a series without a name
    abs_diff_tmp_eda = abs_diff(dummy_data["TMP"], dummy_data["EDA"])
    assert isinstance(abs_diff_tmp_eda, pd.Series)
    assert abs_diff_tmp_eda.name is None

    series_processor = SeriesProcessor(
        series_names=[("EDA", "TMP")],
        function=abs_diff,
    )
    series_dict = dataframe_to_series_dict(dummy_data)

    with pytest.raises(Exception):
        _ = series_processor(series_dict)


def test_error_numpy_array_multi_input_series_processor(dummy_data):
    def abs_diff(sig1, sig2):
        return np.array(np.abs(sig1 - sig2))

    abs_diff_tmp_eda = abs_diff(dummy_data["TMP"], dummy_data["EDA"])
    assert isinstance(abs_diff_tmp_eda, np.ndarray)

    series_processor = SeriesProcessor(
        series_names=[("EDA", "TMP")],
        function=abs_diff,
    )
    series_dict = dataframe_to_series_dict(dummy_data)

    with pytest.raises(Exception):
        _ = series_processor(series_dict)


def test_error_numpy_array_different_length_single_input_series_processor(dummy_data):
    def absx2(sig):
        return np.abs(sig.values * 2)[:-2]

    absx2_tmp = absx2(dummy_data["TMP"])
    assert isinstance(absx2_tmp, np.ndarray)
    assert len(absx2_tmp) == len(dummy_data) - 2

    series_processor = SeriesProcessor(
        series_names=["EDA"],
        function=absx2,
    )
    series_dict = dataframe_to_series_dict(dummy_data)

    with pytest.raises(Exception):
        _ = series_processor(series_dict)


def test_error_series_list_same_name_series_processor(dummy_data):
    # If you perform multi-series operation on series with different / no names
    # => the output will also have no name
    def abs_diff_sum(sig1, sig2):
        return [
            pd.Series(np.abs(sig1 - sig2), name="S1"),
            pd.Series(np.abs(sig1 + sig2), name="S1"),
        ]

    # Check if abs_diff returns a series without a name
    abs_diff_sum_tmp_eda = abs_diff_sum(dummy_data["TMP"], dummy_data["EDA"])
    assert isinstance(abs_diff_sum_tmp_eda, list)
    assert len(abs_diff_sum_tmp_eda) == 2
    assert [isinstance(el, pd.Series) for el in abs_diff_sum_tmp_eda]
    assert [el.name == "S1" for el in abs_diff_sum_tmp_eda]

    series_processor = SeriesProcessor(
        series_names=[("EDA", "TMP")],
        function=abs_diff_sum,
    )
    series_dict = dataframe_to_series_dict(dummy_data)

    with pytest.raises(Exception):
        _ = series_processor(series_dict)


def test_error_series_list_no_time_index_series_processor(dummy_data):
    def absx2(sig):
        return [pd.Series(np.abs(sig.values * 2), name="S1")]

    # Check if abs_diff returns a series without a name
    absx2_eda = absx2(dummy_data["TMP"])
    assert isinstance(absx2_eda, list)
    assert len(absx2_eda) == 1
    assert isinstance(absx2_eda[0], pd.Series)
    assert absx2_eda[0].name == "S1"
    assert not isinstance(absx2_eda[0].index, pd.DatetimeIndex)

    series_processor = SeriesProcessor(
        series_names=["EDA" "TMP"],
        function=absx2,
    )
    series_dict = dataframe_to_series_dict(dummy_data)

    with pytest.raises(Exception):
        _ = series_processor(series_dict)


def test_error_series_list_no_valid_output_series_processor(dummy_data):
    def absx2(sig):
        return [i for i in range(len(sig))]

    # Check if abs_diff returns a series without a name
    absx2_eda = absx2(dummy_data["TMP"])
    assert isinstance(absx2_eda, list)
    assert len(absx2_eda) == len(dummy_data)

    series_processor = SeriesProcessor(
        series_names=["EDA", "TMP"],
        function=absx2,
    )
    series_dict = dataframe_to_series_dict(dummy_data)

    with pytest.raises(TypeError):
        _ = series_processor(series_dict)


### Test output types (np.float32)
