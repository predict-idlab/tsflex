import os

import numpy as np
import pytest

from tsflex.features.feature import FeatureDescriptor, MultipleFeatureDescriptors
from tsflex.features.feature_collection import FeatureCollection

from ..utils import dummy_data, dummy_group_data  # noqa: F401

FUNCS = [np.sum, np.min, np.max, np.mean, np.median, np.std, np.var]
MAX_CPUS = os.cpu_count() or 2
NB_CORES = [1, int(MAX_CPUS / 2), MAX_CPUS]
WINDOWS = ["10s", "30s", "60s", "120s"]
STRIDES = ["5s", "15s", "30s", "60s"]
SERIES_NAMES = ["EDA", "TMP"]


@pytest.mark.benchmark(group="single descriptor")
@pytest.mark.parametrize("func", FUNCS)
@pytest.mark.parametrize("n_cores", NB_CORES)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("stride", STRIDES)
def test_single_series_feature_collection(
    benchmark, func, n_cores, window, stride, dummy_data  # noqa: F811
):
    fd = FeatureDescriptor(
        function=func, series_name="EDA", window=window, stride=stride
    )

    fc = FeatureCollection(feature_descriptors=fd)

    benchmark(fc.calculate, dummy_data, n_jobs=n_cores)


@pytest.mark.benchmark(group="multiple descriptors")
@pytest.mark.parametrize("n_cores", NB_CORES)
def test_single_series_feature_collection_multiple_descriptors(
    benchmark, n_cores, dummy_data  # noqa: F811
):
    mfd = MultipleFeatureDescriptors(
        functions=FUNCS,
        series_names=SERIES_NAMES,
        windows=[
            w for w in WINDOWS
        ],  # gives error when just passing WINDOWS for some reason, same with STRIDES
        strides=[s for s in STRIDES],
    )

    fc = FeatureCollection(mfd)

    benchmark(fc.calculate, dummy_data, n_jobs=n_cores)


@pytest.mark.benchmark(group="group_by collection")
@pytest.mark.parametrize("n_cores", NB_CORES)
@pytest.mark.parametrize("func", FUNCS)
@pytest.mark.parametrize("group_by", ["group_by_all", "group_by_consecutive"])
def test_single_series_feature_collection_group_by_consecutive(
    benchmark, n_cores, func, group_by, dummy_group_data  # noqa: F811
):
    fd = FeatureDescriptor(function=func, series_name="number_sold")

    fc = FeatureCollection(feature_descriptors=fd)

    benchmark(fc.calculate, dummy_group_data, n_jobs=n_cores, **{group_by: "store"})
