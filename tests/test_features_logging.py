"""Tests for the feature extraction functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import os
import warnings

import numpy as np

from tsflex.features import (
    FeatureCollection,
    FeatureDescriptor,
    MultipleFeatureDescriptors,
    get_feature_logs,
    get_function_stats,
    get_series_names_stats,
)
from tsflex.utils.data import flatten

from .utils import dummy_data, logging_file_path

test_path = os.path.abspath(os.path.dirname(__file__))


def test_simple_features_logging_time_index(dummy_data, logging_file_path):
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window="5s",
        stride="12s",
    )
    fc = FeatureCollection(feature_descriptors=fd)
    fc.add(
        MultipleFeatureDescriptors(
            np.min, series_names=["TMP", "ACC_x"], windows="5s", strides="12s"
        )
    )
    fc.add(FeatureDescriptor(np.min, series_name=("EDA",), window="5s", stride="12s"))

    assert set(fc.get_required_series()) == set(["EDA", "TMP", "ACC_x"])
    assert len(fc.get_required_series()) == 3

    assert not os.path.exists(logging_file_path)

    # Sequential (n_jobs <= 1), otherwise file_path gets cleared
    _ = fc.calculate(dummy_data, logging_file_path=logging_file_path, n_jobs=1)

    assert os.path.exists(logging_file_path)
    logging_df = get_feature_logs(logging_file_path)

    assert all(
        logging_df.columns.values
        == ["log_time", "function", "series_names", "window", "stride", "duration"]
    )

    assert len(logging_df) == 4
    assert logging_df.select_dtypes(include=[np.datetime64]).columns.values == [
        "log_time"
    ]
    assert logging_df.select_dtypes(include=[np.timedelta64]).columns.values == [
        "duration"
    ]

    assert set(logging_df["function"].values) == set(["amin", "sum"])
    assert set(logging_df["series_names"].values) == set(
        ["(EDA,)", "(ACC_x,)", "(TMP,)"]
    )
    assert all(logging_df["window"] == "5s")
    assert all(logging_df["stride"] == ("12s",))

    function_stats_df = get_function_stats(logging_file_path)
    assert len(function_stats_df) == 2
    assert set(function_stats_df.index) == set(
        [(s, "5s", ("12s",)) for s in ["sum", "amin"]]
    )
    assert all(function_stats_df["duration"]["mean"] >= 0)
    assert function_stats_df["duration"]["count"].sum() == 4

    series_names_df = get_series_names_stats(logging_file_path)
    assert len(series_names_df) == 3
    assert set(series_names_df.index) == set(
        [(s, "5s", ("12s",)) for s in ["(EDA,)", "(TMP,)", "(ACC_x,)"]]
    )
    assert all(series_names_df["duration"]["mean"] >= 0)
    assert series_names_df["duration"]["count"].sum() == 4


def test_simple_features_logging_sequence_index(dummy_data, logging_file_path):
    dummy_data = dummy_data.reset_index(drop=True)
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window=50,
        stride=120,
    )
    fc = FeatureCollection(feature_descriptors=fd)
    fc.add(
        MultipleFeatureDescriptors(
            np.min, series_names=["TMP", "ACC_x"], windows=50, strides=120
        )
    )
    fc.add(FeatureDescriptor(np.min, series_name=("EDA",), window=50, stride=120))

    assert set(fc.get_required_series()) == set(["EDA", "TMP", "ACC_x"])
    assert len(fc.get_required_series()) == 3

    assert not os.path.exists(logging_file_path)

    # Sequential (n_jobs <= 1), otherwise file_path gets cleared
    _ = fc.calculate(dummy_data, logging_file_path=logging_file_path, n_jobs=1)

    assert os.path.exists(logging_file_path)
    logging_df = get_feature_logs(logging_file_path)

    assert all(
        logging_df.columns.values
        == ["log_time", "function", "series_names", "window", "stride", "duration"]
    )

    assert len(logging_df) == 4
    assert logging_df.select_dtypes(include=[np.datetime64]).columns.values == [
        "log_time"
    ]
    assert logging_df.select_dtypes(include=[np.timedelta64]).columns.values == [
        "duration"
    ]

    assert set(logging_df["function"].values) == set(["amin", "sum"])
    assert set(logging_df["series_names"].values) == set(
        ["(EDA,)", "(ACC_x,)", "(TMP,)"]
    )
    assert all(logging_df["window"] == 50)
    assert all(logging_df["stride"] == (120,))

    function_stats_df = get_function_stats(logging_file_path)
    assert len(function_stats_df) == 2
    assert set(function_stats_df.index) == set(
        [(s, 50, (120,)) for s in ["sum", "amin"]]
    )
    assert all(function_stats_df["duration"]["mean"] >= 0)
    assert function_stats_df["duration"]["count"].sum() == 4

    series_names_df = get_series_names_stats(logging_file_path)
    assert len(series_names_df) == 3
    assert set(series_names_df.index) == set(
        [(s, 50, (120,)) for s in ["(EDA,)", "(TMP,)", "(ACC_x,)"]]
    )
    assert all(series_names_df["duration"]["mean"] >= 0)
    assert series_names_df["duration"]["count"].sum() == 4


def test_simple_features_logging_segment_start_idxs_no_stride(
    dummy_data, logging_file_path
):
    # Add no stride
    dummy_data = dummy_data.reset_index(drop=True)
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window=50,
    )
    fc = FeatureCollection(feature_descriptors=fd)
    fc.add(
        MultipleFeatureDescriptors(np.min, series_names=["TMP", "ACC_x"], windows=50)
    )
    fc.add(FeatureDescriptor(np.min, series_name=("EDA",), window=50))
    for fd in flatten(fc._feature_desc_dict.values()):
        assert fd.stride is None

    assert set(fc.get_required_series()) == set(["EDA", "TMP", "ACC_x"])
    assert len(fc.get_required_series()) == 3

    assert not os.path.exists(logging_file_path)

    # Sequential (n_jobs <= 1), otherwise file_path gets cleared
    segment_start_idxs = np.arange(0, 1300, 130)
    _ = fc.calculate(
        dummy_data,
        segment_start_idxs=segment_start_idxs,
        logging_file_path=logging_file_path,
        n_jobs=1,
    )

    assert os.path.exists(logging_file_path)
    logging_df = get_feature_logs(logging_file_path)

    assert all(
        logging_df.columns.values
        == ["log_time", "function", "series_names", "window", "stride", "duration"]
    )

    assert len(logging_df) == 4
    assert logging_df.select_dtypes(include=[np.datetime64]).columns.values == [
        "log_time"
    ]
    assert logging_df.select_dtypes(include=[np.timedelta64]).columns.values == [
        "duration"
    ]

    assert set(logging_df["function"].values) == set(["amin", "sum"])
    assert set(logging_df["series_names"].values) == set(
        ["(EDA,)", "(ACC_x,)", "(TMP,)"]
    )
    assert all(logging_df["window"] == 50)
    assert all(logging_df["stride"] == "manual")

    function_stats_df = get_function_stats(logging_file_path)
    assert len(function_stats_df) == 2
    assert set(function_stats_df.index) == set(
        [(s, 50, "manual") for s in ["sum", "amin"]]
    )
    assert all(function_stats_df["duration"]["mean"] >= 0)
    assert function_stats_df["duration"]["count"].sum() == 4

    series_names_df = get_series_names_stats(logging_file_path)
    assert len(series_names_df) == 3
    assert set(series_names_df.index) == set(
        [(s, 50, "manual") for s in ["(EDA,)", "(TMP,)", "(ACC_x,)"]]
    )
    assert all(series_names_df["duration"]["mean"] >= 0)
    assert series_names_df["duration"]["count"].sum() == 4


def test_simple_features_logging_segment_end_idxs_no_stride(
    dummy_data, logging_file_path
):
    # Add no stride
    dummy_data = dummy_data.reset_index(drop=True)
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window=50,
    )
    fc = FeatureCollection(feature_descriptors=fd)
    fc.add(
        MultipleFeatureDescriptors(np.min, series_names=["TMP", "ACC_x"], windows=50)
    )
    fc.add(FeatureDescriptor(np.min, series_name=("EDA",), window=50))
    for fd in flatten(fc._feature_desc_dict.values()):
        assert fd.stride is None

    assert set(fc.get_required_series()) == set(["EDA", "TMP", "ACC_x"])
    assert len(fc.get_required_series()) == 3

    assert not os.path.exists(logging_file_path)

    # Sequential (n_jobs <= 1), otherwise file_path gets cleared
    segment_end_idxs = np.arange(50, 1350, 130)
    _ = fc.calculate(
        dummy_data,
        segment_end_idxs=segment_end_idxs,
        logging_file_path=logging_file_path,
        n_jobs=1,
    )

    assert os.path.exists(logging_file_path)
    logging_df = get_feature_logs(logging_file_path)

    assert all(
        logging_df.columns.values
        == ["log_time", "function", "series_names", "window", "stride", "duration"]
    )

    assert len(logging_df) == 4
    assert logging_df.select_dtypes(include=[np.datetime64]).columns.values == [
        "log_time"
    ]
    assert logging_df.select_dtypes(include=[np.timedelta64]).columns.values == [
        "duration"
    ]

    assert set(logging_df["function"].values) == set(["amin", "sum"])
    assert set(logging_df["series_names"].values) == set(
        ["(EDA,)", "(ACC_x,)", "(TMP,)"]
    )
    assert all(logging_df["window"] == 50)
    assert all(logging_df["stride"] == "manual")

    function_stats_df = get_function_stats(logging_file_path)
    assert len(function_stats_df) == 2
    assert set(function_stats_df.index) == set(
        [(s, 50, "manual") for s in ["sum", "amin"]]
    )
    assert all(function_stats_df["duration"]["mean"] >= 0)
    assert function_stats_df["duration"]["count"].sum() == 4

    series_names_df = get_series_names_stats(logging_file_path)
    assert len(series_names_df) == 3
    assert set(series_names_df.index) == set(
        [(s, 50, "manual") for s in ["(EDA,)", "(TMP,)", "(ACC_x,)"]]
    )
    assert all(series_names_df["duration"]["mean"] >= 0)
    assert series_names_df["duration"]["count"].sum() == 4


def test_simple_features_logging_segment_start_idxs_overrule_stride(
    dummy_data, logging_file_path
):
    # Add no stride
    dummy_data = dummy_data.reset_index(drop=True)
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window=50,
        stride=120,
    )
    fc = FeatureCollection(feature_descriptors=fd)
    fc.add(
        MultipleFeatureDescriptors(
            np.min, series_names=["TMP", "ACC_x"], windows=50, strides=20
        )
    )
    fc.add(FeatureDescriptor(np.min, series_name=("EDA",), window=50, stride=34))
    for fd in flatten(fc._feature_desc_dict.values()):
        assert fd.stride is not None

    assert set(fc.get_required_series()) == set(["EDA", "TMP", "ACC_x"])
    assert len(fc.get_required_series()) == 3

    assert not os.path.exists(logging_file_path)

    # Sequential (n_jobs <= 1), otherwise file_path gets cleared
    segment_start_idxs = np.arange(0, 1200, 120)
    _ = fc.calculate(
        dummy_data,
        segment_start_idxs=segment_start_idxs,
        logging_file_path=logging_file_path,
        n_jobs=1,
    )

    assert os.path.exists(logging_file_path)
    logging_df = get_feature_logs(logging_file_path)

    assert all(
        logging_df.columns.values
        == ["log_time", "function", "series_names", "window", "stride", "duration"]
    )

    assert len(logging_df) == 4
    assert logging_df.select_dtypes(include=[np.datetime64]).columns.values == [
        "log_time"
    ]
    assert logging_df.select_dtypes(include=[np.timedelta64]).columns.values == [
        "duration"
    ]

    assert set(logging_df["function"].values) == set(["amin", "sum"])
    assert set(logging_df["series_names"].values) == set(
        ["(EDA,)", "(ACC_x,)", "(TMP,)"]
    )
    assert all(logging_df["window"] == 50)
    assert all(logging_df["stride"] == "manual")

    function_stats_df = get_function_stats(logging_file_path)
    assert len(function_stats_df) == 2
    assert set(function_stats_df.index) == set(
        [(s, 50, "manual") for s in ["sum", "amin"]]
    )
    assert all(function_stats_df["duration"]["mean"] >= 0)
    assert function_stats_df["duration"]["count"].sum() == 4

    series_names_df = get_series_names_stats(logging_file_path)
    assert len(series_names_df) == 3
    assert set(series_names_df.index) == set(
        [(s, 50, "manual") for s in ["(EDA,)", "(TMP,)", "(ACC_x,)"]]
    )
    assert all(series_names_df["duration"]["mean"] >= 0)
    assert series_names_df["duration"]["count"].sum() == 4


def test_simple_features_logging_segment_end_idxs_overrule_stride(
    dummy_data, logging_file_path
):
    # Add no stride
    dummy_data = dummy_data.reset_index(drop=True)
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window=50,
        stride=120,
    )
    fc = FeatureCollection(feature_descriptors=fd)
    fc.add(
        MultipleFeatureDescriptors(
            np.min, series_names=["TMP", "ACC_x"], windows=50, strides=20
        )
    )
    fc.add(FeatureDescriptor(np.min, series_name=("EDA",), window=50, stride=34))
    for fd in flatten(fc._feature_desc_dict.values()):
        assert fd.stride is not None

    assert set(fc.get_required_series()) == set(["EDA", "TMP", "ACC_x"])
    assert len(fc.get_required_series()) == 3

    assert not os.path.exists(logging_file_path)

    # Sequential (n_jobs <= 1), otherwise file_path gets cleared
    segment_end_idxs = np.arange(50, 1250, 120)
    _ = fc.calculate(
        dummy_data,
        segment_end_idxs=segment_end_idxs,
        logging_file_path=logging_file_path,
        n_jobs=1,
    )

    assert os.path.exists(logging_file_path)
    logging_df = get_feature_logs(logging_file_path)

    assert all(
        logging_df.columns.values
        == ["log_time", "function", "series_names", "window", "stride", "duration"]
    )

    assert len(logging_df) == 4
    assert logging_df.select_dtypes(include=[np.datetime64]).columns.values == [
        "log_time"
    ]
    assert logging_df.select_dtypes(include=[np.timedelta64]).columns.values == [
        "duration"
    ]

    assert set(logging_df["function"].values) == set(["amin", "sum"])
    assert set(logging_df["series_names"].values) == set(
        ["(EDA,)", "(ACC_x,)", "(TMP,)"]
    )
    assert all(logging_df["window"] == 50)
    assert all(logging_df["stride"] == "manual")

    function_stats_df = get_function_stats(logging_file_path)
    assert len(function_stats_df) == 2
    assert set(function_stats_df.index) == set(
        [(s, 50, "manual") for s in ["sum", "amin"]]
    )
    assert all(function_stats_df["duration"]["mean"] >= 0)
    assert function_stats_df["duration"]["count"].sum() == 4

    series_names_df = get_series_names_stats(logging_file_path)
    assert len(series_names_df) == 3
    assert set(series_names_df.index) == set(
        [(s, 50, "manual") for s in ["(EDA,)", "(TMP,)", "(ACC_x,)"]]
    )
    assert all(series_names_df["duration"]["mean"] >= 0)
    assert series_names_df["duration"]["count"].sum() == 4


def test_simple_features_logging_segment_start_and_end_idxs_overrule_stride_and_window(
    dummy_data, logging_file_path
):
    # Add no stride
    dummy_data = dummy_data.reset_index(drop=True)
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window=50,
        stride=120,
    )
    fc = FeatureCollection(feature_descriptors=fd)
    fc.add(
        MultipleFeatureDescriptors(
            np.min, series_names=["TMP", "ACC_x"], windows=50, strides=20
        )
    )
    fc.add(FeatureDescriptor(np.min, series_name=("EDA",), window=50, stride=34))
    for fd in flatten(fc._feature_desc_dict.values()):
        assert fd.stride is not None

    assert set(fc.get_required_series()) == set(["EDA", "TMP", "ACC_x"])
    assert len(fc.get_required_series()) == 3

    assert not os.path.exists(logging_file_path)

    # Sequential (n_jobs <= 1), otherwise file_path gets cleared
    segment_start_idxs = np.arange(0, 1200, 120)
    segment_end_idxs = segment_start_idxs + 50
    _ = fc.calculate(
        dummy_data,
        segment_start_idxs=segment_start_idxs,
        segment_end_idxs=segment_end_idxs,
        logging_file_path=logging_file_path,
        n_jobs=1,
    )

    assert os.path.exists(logging_file_path)
    logging_df = get_feature_logs(logging_file_path)

    assert all(
        logging_df.columns.values
        == ["log_time", "function", "series_names", "window", "stride", "duration"]
    )

    assert len(logging_df) == 4
    assert logging_df.select_dtypes(include=[np.datetime64]).columns.values == [
        "log_time"
    ]
    assert logging_df.select_dtypes(include=[np.timedelta64]).columns.values == [
        "duration"
    ]

    assert set(logging_df["function"].values) == set(["amin", "sum"])
    assert set(logging_df["series_names"].values) == set(
        ["(EDA,)", "(ACC_x,)", "(TMP,)"]
    )
    assert all(logging_df["window"] == "manual")
    assert all(logging_df["stride"] == "manual")

    function_stats_df = get_function_stats(logging_file_path)
    assert len(function_stats_df) == 2
    assert set(function_stats_df.index) == set(
        [(s, "manual", "manual") for s in ["sum", "amin"]]
    )
    assert all(function_stats_df["duration"]["mean"] >= 0)
    assert function_stats_df["duration"]["count"].sum() == 4

    series_names_df = get_series_names_stats(logging_file_path)
    assert len(series_names_df) == 3
    assert set(series_names_df.index) == set(
        [(s, "manual", "manual") for s in ["(EDA,)", "(TMP,)", "(ACC_x,)"]]
    )
    assert all(series_names_df["duration"]["mean"] >= 0)
    assert series_names_df["duration"]["count"].sum() == 4


def test_file_warning_features_logging(dummy_data, logging_file_path):
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window="5s",
        stride="2.5s",
    )
    fc = FeatureCollection(feature_descriptors=fd)

    assert not os.path.exists(logging_file_path)

    with warnings.catch_warnings(record=True) as w:
        with open(logging_file_path, "w"):
            pass
        assert os.path.exists(logging_file_path)
        # Sequential (n_jobs <= 1), otherwise file_path gets cleared
        _ = fc.calculate(dummy_data, logging_file_path=logging_file_path, n_jobs=1)
        assert os.path.exists(logging_file_path)
        assert len(w) == 1
        assert all([issubclass(warn.category, RuntimeWarning) for warn in w])
        assert "already exists" in str(w[0])
        # CLEANUP
        os.remove(logging_file_path)
