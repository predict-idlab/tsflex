# Feature extraction guide

The following sections will explain the feature extraction module in detail.

<div style="text-align: center;">
<h3><b><a href="#header-submodules">Jump to API reference</a></b></h3>
</div>
<br>

## Working example ✅

_tsflex_ is built to be intuitive, so we encourage you to copy-paste this code and toy with some parameters! <br>

This executable example creates a feature-collection that contains 2 features (skewness and minimum). Note that _tsflex_ does not make any assumptions about the sampling rate of the time-series data.

```python
import pandas as pd; import scipy.stats as ss; import numpy as np
from tsflex.features import FeatureDescriptor, FeatureCollection, FuncWrapper

# 1. -------- Get your time-indexed data --------
# Data contains 1 column; ["TMP"]
url = "https://github.com/predict-idlab/tsflex/raw/main/examples/data/empatica/"
data = pd.read_parquet(url + "tmp.parquet").set_index("timestamp")

# 2 -------- Construct your feature collection --------
fc = FeatureCollection(
    feature_descriptors=[
        FeatureDescriptor(
            function=FuncWrapper(func=ss.skew, output_names="skew"),
            series_name="TMP", 
            window="5min", stride="2.5min",
        )
    ]
)
# -- 2.1. Add features to your feature collection
# NOTE: tsflex allows features to have different windows and strides
fc.add(FeatureDescriptor(np.min, "TMP", '2.5min', '2.5min'))

# 3 -------- Calculate features --------
fc.calculate(data=data, return_df=True)  # which outputs:
```
| timestamp                 |   TMP__amin__w=1m_s=30s |   TMP__skew__w=2m_s=1m |
|:--------------------------|------------------------:|-----------------------:|
| 2017-06-13 14:23:13+02:00 |                   27.37 |            nan         |
| 2017-06-13 14:23:43+02:00 |                   27.37 |            nan         |
| 2017-06-13 14:24:13+02:00 |                   27.43 |             10.8159    |
| 2017-06-13 14:24:43+02:00 |                   27.81 |            nan         |
| 2017-06-13 14:25:13+02:00 |                   28.23 |             -0.0327893 |
|                       ... |                     ... |                    ... |
<br>

!!!tip 
    More advanced feature-extraction examples can be found [in these example notebooks](https://github.com/predict-idlab/tsflex/tree/main/examples)

<br>

## Getting started 🚀

The feature-extraction functionality of _tsflex_ is provided by a `FeatureCollection` that contains `FeatureDescriptor`s. The features are calculated (in a parallel manner) on the data that is passed to the feature collection.

### Components
![features uml](https://raw.githubusercontent.com/predict-idlab/tsflex/main/docs/_static/features_uml.png)

As shown above, there are 3 relevant classes for feature-extraction.

1. [FeatureCollection](/tsflex/features/#tsflex.features.FeatureCollection): serves as a registry, withholding the to-be-calculated _features_
2. [FeatureDescriptor](/tsflex/features/#tsflex.features.FeatureDescriptor): an instance of this class describes a _feature_.
    <br>Features are defined by:
      * `series_name`: the names of the input series on which the feature-function will operate
      * `function`: the _Callable_ feature-function - e.g. _np.mean_
      * `window`: the _sample_ or _time-based_ window -  e.g. _200_ or _"2days"_
      * `stride`: the _sample_ or _time-based_ stride - e.g. _15_ or _"1hour"_
3. [FuncWrapper](/tsflex/features/#tsflex.features.FuncWrapper): wraps _Callable_ feature functions, and is intended for feature function configuration.
    <br>FuncWrappers are defined by:
    * `func`: The wrapped feature-function
    * `output_names`: set custom and/or multiple feature output names
    * `input_type`: define the feature its datatype; e.g., a pd.Series or np.array
    * _**kwargs_: additional keyword argument which are passed to `func`

The snippet below shows how the `FeatureCollection` & `FeatureDescriptor` components work together:

```python
import numpy as np; import scipy.stats as ss
from tsflex.features import FeatureDescriptor, FeatureCollection

# The FeatureCollection takes a List[FeatureDescriptor] as input
# There is no need for using a FuncWrapper when dealing with simple feature functions
fc = FeatureCollection(feature_descriptors=[
        FeatureDescriptor(np.mean, "series_a", "1hour", "15min"),
        FeatureDescriptor(ss.skew, "series_b", "3hours", "5min")
    ]
)

# We can still add features after instantiating.
fc.add(features=[FeatureDescriptor(np.std, "series_a", "1hour", "15min")])

# Calculate the features
fc.calculate(...)
```

### Feature functions

A feature function needs to match this prototype:

    function(*series: Union[np.ndarray, pd.Series], **kwargs)
        -> Union[Any, List[Any]]

Hence, feature functions should take one (or multiple) arrays as first input. This can be followed by some keyword arguments.<br>
The output of a feature function can be rather versatile (e.g., a float, integer, string, bool, ... or a list thereof). <br><br>
Note that the feature function may also take more than one series as input. In this case, the feature function should be wrapped in a ``FuncWrapper``, with the `input_type` argument set to `pd.Series`.

In the [advanced usage](#advanced-usage) section, more info is given on these feature-function.

### Multiple feature descriptors

Sometimes it can get overly verbose when the same feature is shared over multiple series, windows and/or strides. To solve this problem, we introduce the `MultipleFeatureDescriptors`. This component allows to **create multiple feature descriptors for all** the ``function - series_name(s) - window - stride`` **combinations**.

As shown in the example below, a `MultipleFeatureDescriptors` instance can be added a `FeatureCollection`.

```python
import numpy as np; import scipy.stats as ss
from tsflex.features import FeatureDescriptor, FeatureCollection
from tsflex.features import MultipleFeatureDescriptors

# There is no need for using a FuncWrapper when dealing with simple feature functions
fc = FeatureCollection(feature_descriptors=[
        FeatureDescriptor(np.mean, "series_a", "1hour", "15min"),
        FeatureDescriptor(ss.skew, "series_b", "3hours", "5min"),
        # Expands to a feature-descriptor list, withholding the combination of all 
        # The feature-window-stride arguments above.
        MultipleFeatureDescriptors(
            functions=[np.min, np.max, np.std, ss.skew],
            series_names=["series_a", "series_b", "series_c"],
            windows=["5min", "15min"],
            strides=["1min","2min","3min"]
        )
    ]
)

# Calculate the features
fc.calculate(...)
```

### Output format
The output of the `FeatureCollection` its `FeatureCollection.calculate()` is a (list of) **`sequence-indexed pd.DataFrames`** with column names:

> **`<SERIES-NAME>__<FEAT-NAME>__w=<WINDOW>__s=<STRIDE>`**.

The column-name for the first feature defined in the snippet above will thus be `series_a__std__w=1h__s=15m`.

When the windows and strides are defined in a sample based-manner (which is mandatory for non datetime-indexed data), a possible output column would be `series_a__std__w=100_s=15`, where the window and stride are defined in samples and thus not in time-strings.

!!!note
    You can find more information about the **input data-formats** in [this section](/tsflex/#data-formats) and read more about the (obvious) **limitations** in the next section.

<br>

## Limitations ⚠️

It is important to note that there a still some, albeit logical, **limitations** regarding the supported [data format](/tsflex/#data-formats).

These limitations are:

1. Each [`ts`](/tsflex/#data-formats) must be in the **<span style="color:darkred">flat/wide</span> data format** and they all need to have the **<span style="color:darkred">same sequence index dtype</span>**, which needs to be **sortable**.
    - It just doesn't make sense to have a mix of different sequence index dtypes.<br>
    Imagine a FeatureCollection to which a `ts` with a `pd.DatetimeIndex` is passed, but a `ts` with a `pd.RangeIndex` is also passed. Both indexes aren't comparable, which thus is counterintuitive.
2. tsflex has **no support** for <b style="color:darkred">multi-indexes & multi-columns</b>
3. tsflex assumes that each `ts` has a unique name. Hence <b style="color:darkred">no duplicate</b> `ts` <b style="color:darkred">names</b> are allowed
      - **Countermeasure**: rename your `ts`

<br>


## Important notes 📢

We support various data-types. e.g. (np.float32, string-data, time-based data). However, it is the end-users responsibility to use a function which interplays nicely with the data its format.

<br>

## Advanced usage 👀

Also take a look at the `FeatureCollection.reduce()` and `FeatureCollection.serialize()` methods.

### Versatile functions

As [explained above](#feature-functions) _tsflex_ is rather versatile in terms of function input and output.

_tsflex_ does not just allow ``one-to-one`` processing functions, but also ``many-to-one``, ``one-to-many``, and ``many-to-many`` functions are supported in a convenient way:

<!-- 
- `one-to-one`; the **feature function** should
    - take a single series as input
    - output a single value

    The function should (usually) not be wrapped in a ``FuncWrapper``. 

    Example
```python
def abs_sum(s: np.array) -> float:
    return np.sum(np.abs(s))

fd = FeatureDescriptor(
    abs_sum, series_name="series_1", window="5m", stride="2m30s",
)
``` 
-->

- `many-to-one`; the **feature function** should
    - take multiple series as input
    - output a single value

    The function should (usually) not be wrapped in a ``FuncWrapper``. <br> 
    Note that now the `series_name` argument requires a tuple of the ordered input series names.

    Example:
```python
def abs_sum_diff(s1: np.array, s2: np.array) -> float:
    min_len = min(len(s1), len(s2))
    return np.sum(np.abs(s1[:min_len] - s2[:min_len]))

fd = FeatureDescriptor(
    abs_sum_diff, series_name=("series_1", "series_2"), 
    window="5m", stride="2m30s",
)
```

- `one-to-many`; the **feature function** should
    - take a single series as input
    - output multiple values

    The function should be wrapped in a ``FuncWrapper`` to log its multiple output names.

    Example:
```python
def abs_stats(s: np.array) -> Tuple[float]:
    s_abs = np.abs(s)
    return np.min(s_abs), np.max(s_abs), np.mean(s_abs), np.std(s_abs)

output_names = ["abs_min", "abs_max", "abs_mean", "abs_std"]
fd = FeatureDescriptor(
    FuncWrapper(abs_stats, output_names=output_names),
    series_name="series_1", window="5m", stride="2m30s",
)
```

- `many-to-many`; the **feature function** should
    - take multiple series as input
    - output multiple values

    The function should be wrapped in a ``FuncWrapper`` to log its multiple output names.

    Example:
```python
def abs_stats_diff(s1: np.array, s2: np.array) -> Tuple[float]:
    min_len = min(len(s1), len(s2))
    s_abs_diff = np.sum(np.abs(s1[:min_len] - s2[:min_len]))
    return np.min(s_abs_diff), np.max(s_abs_diff), np.mean(s_abs_diff)

output_names = ["abs_diff_min", "abs_diff_max", "abs_diff_mean"]
fd = FeatureDescriptor(
    FuncWrapper(abs_stats_diff, output_names=output_names),
    series_name=("series_1", "series_2"), window="5m", stride="2m30s",
)
```

!!!note
    As visible in the [feature function prototype](#feature-functions), both `np.array` and `pd.Series` are supported function input types.
    If your feature function requires `pd.Series` as input (instead of the default `np.array`), the function should be wrapped in a ``FuncWrapper`` with the `input_type` argument set to `pd.Series`.

An example of a function that leverages the pd.Series datatype:

```python
def linear_trend_timewise(s: pd.Series):
    # Get differences between each timestamp and the first timestamp in hour float
    # Then convert to hours and reshape for linear regression
    times_hours = np.asarray((s.index - s.index[0]).total_seconds() / 3600)
    linReg = linregress(times_hours, s.values)
    return linReg.slope, linReg.intercept, linReg.rvalue

fd = FeatureDescriptor(
     FuncWrapper(
        linear_trend_timewise,
        ["twise_regr_slope", "twise_regr_intercept", "twise_regr_r_value"],
        input_type=pd.Series, 
    ),
)
```
<!-- TODO: review Jeroen! -->
### Multivariate-data

There are no assumptions made about the `data` its `sequence-ranges`. However, the end-user must take some things in consideration.

* By using the `bound_method` argument of `FeatureCollection.calculate`, the end-user can specify whether the "inner" or "outer" data-bounds will be used for generating the slice-ranges.
* All `ts` must-have the same data-index dtype. this makes them comparable and allows for generating same-range slices on multivariate data.

### Irregularly sampled data

Strided-rolling feature extraction on irregularly sampled data results in varying feature-segment sizes.<br>

When using multivariate data, with either different sample rates or with an irregular data-rate, <span style="color:darkred">you cannot make the assumption that all windows will have the same length</span>. Your feature extraction method should thus be:
>
* robust for varying length windows
* robust for (possible) empty windows

!!!tip
    For conveniently creating such **robust features** we suggest using the [``make_robust``](integrations#tsflex.features.integrations.make_robust) function.

!!!note
    A `warning` will be raised when irregular sampled data is observed. <br>
    In order to avoid this warning, the user should explicitly approve that there may be sparsity in the data by setting the **`approve_sparsity`** flag to True in the ``FeatureCollection.calculate`` method.

### Logging

When a `logging_file_path` is passed to the ``FeatureCollection`` its `FeatureCollection.calculate` method, the execution times of the feature functions will be logged.

This is especially useful to identify which feature functions take a long time to compute.

[More info about logging](#tsflex.features.get_feature_logs).

<br>

---