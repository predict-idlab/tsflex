# Feature extraction guide

The following sections will explain the feature extraction module in detail.

<!-- <div style="text-align: center;"> -->
<h3><b><a href="#header-submodules">Jump to API reference</a></b></h3>
<!-- </div> -->
<br>

## Working example ✅

_tsflex_ is built to be intuitive, so we encourage you to copy-paste this code and toy with some parameters! <br>

This executable example creates a feature-collection that contains 2 features (skewness and minimum). <br>
**Note**: we do not make any assumptions about the sampling rate of the time-series data.

```python
import pandas as pd; import scipy.stats as ss; import numpy as np
from tsflex.features import FeatureDescriptor, FeatureCollection, NumpyFuncWrapper

# 1. -------- Get your time-indexed data --------
# Data contains 1 column; ["TMP"]
url = "https://github.com/predict-idlab/tsflex/raw/main/examples/data/empatica/"
data = pd.read_parquet(url + "tmp.parquet").set_index("timestamp")

# 2 -------- Construct your feature collection --------
fc = FeatureCollection(
    feature_descriptors=[
        FeatureDescriptor(
            function=NumpyFuncWrapper(func=ss.skew, output_names="skew"),
            series_name="TMP", 
            window="5min", stride="2.5min",
        )
    ]
)
# -- 2.1. Add features to your feature collection
fc.add(FeatureDescriptor(np.min, "TMP", '2.5min', '2.5min'))

# 3 -------- Calculate features --------
fc.calculate(data=data, return_df=True)
# which outputs:
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
2. [FeatureDescriptor](/tsflex/features/#tsflex.features.FeatureDescriptor): an instance of this class describes a _feature_. <br>Features are defined by:
      * `series_name`: the names of the signal(s) which this feature will use. 
      * `function`: the _Callable_ feature-function - e.g. _np.mean_
      * `window`: the _time-based_ window -  e.g. _"1hour"_
      * `stride`: the _time-based_ stride - e.g. _"2days"_
3. [NumpyFuncWrapper](/tsflex/features/#tsflex.features.NumpyFuncWrapper): a wrapper around _Callable_ functions, intended for advanced feature function definitions, such as:
    * features with multiple output columns
    * passing _**kwargs_ to feature functions

The snippet below shows how the `FeatureCollection` & `FeatureDescriptor` components work together:

```python
import numpy as np; import scipy.stats as ss
from tsflex.features import FeatureDescriptor, FeatureCollection

# The FeatureCollection takes a List[FeatureDescriptor] as input
fc = FeatureCollection(feature_descriptors=[
        # There is no need for NumpyFuncWrapper when using "simple" features
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

The function that processes the series should match this prototype:

    function(*series: np.ndarray, **kwargs)
        -> Union[Any, List[Any]]

<!-- TODO: waarom geen pd.Series?? -->

Hence, the feature function should take one (or multiple) arrays as input, these may be followed by some keyword arguments. The output of a feature function can be rather versatile (e.g., a float, an integer, a string, a bool, ... or a list thereof).

In [this section](#advanced-usage) you can find more info on advanced usage of feature functions.

### Multiple feature descriptors

Sometimes it can get overly verbose when the same feature is shared over multiple series, windows and/or strides. To solve this proble, we introduce the `MultipleFeatureDescriptors`, this component allows to **create multiple feature descriptors for all** the ``function - series_name(s) - window - stride`` **combinations**.

A `MultipleFeatureDescriptors` instance can be added a `FeatureCollection`.

Example
```python
import numpy as np; import scipy.stats as ss
from tsflex.features import FeatureDescriptor, FeatureCollection
from tsflex.features import MultipleFeatureDescriptors

# The FeatureCollection takes a List[FeatureDescriptor] as input
fc = FeatureCollection(feature_descriptors=[
        # There is no need for NumpyFuncWrapper when using "simple" features
        FeatureDescriptor(np.mean, "series_a", "1hour", "15min"),
        FeatureDescriptor(ss.skew, "series_b", "3hours", "5min"),
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
The output of the `FeatureCollection` its `calculate` method is a (list of) **`time-indexed pd.DataFrame`** with column names<br>

> **`<SERIES-NAME>__<FEAT-NAME>__w=<WINDOW>__s=<STRIDE>`**.

The column-name for the feature defined on the penultimate line in the snipped above will thus be `series_a__std__w=1h__s=15m`.

!!!note
    You can find more information about the **input data-formats** in [this section](/tsflex/#data-formats) and read more about the (obvious) **limitations** in the next section.

<br>

## Limitations ⚠️

It is important to note that there a still some, albeit logical, **limitations** regarding the supported [data format](/tsflex/#data-formats).

These limitations are:

1. Each [`ts`](/tsflex/#data-formats) must have a <b style="color:red">`pd.DatetimeIndex` that increases monotonically</b>
      - **Countermeasure**: Apply _[sort_index()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_index.html)_ on your not-monotonically increasing data
2. <b style="color:red">No duplicate</b> `ts` <b style="color:red">names</b> are allowed
      - **Countermeasure**: rename your `ts`

<br>


## Important notes 📢

- We support various data-types. e.g. (np.float32, string-data, time-based data). However, it is the end-users responsibility to use a function which interplays nicely with the data its format.


<br>

## Advanced usage 👀

### Versatile functions

`TODO` 

<!-- hier NumpyFuncWrapper shillen -->

<!-- TODO: tot hier geraakt -->

### Multivariate-data
There are no assumptions made about the `data` its `time-ranges`.<br>
However, the end-user must take some things in consideration.

### Multiple time series

* functions that work on **multiple time series**: see the `tsflex.chunking` module for more info.


### Irregularly sampled data

This case may cause that not all windows on which features are calculated have the same amount of samples.<br>
When using multivariate data, with either different sample rates or with an irregular data-rate, you cannot make the assumption that all windows will have the same length. Your feature extraction method will thus 
  * will the s 

### Logging

When a `logging_file_path` is passed to the `FeatureCollection` its `calculate` method, the execution times of the feature functions will be logged.

[More info](#tsflex.features.get_feature_logs)



<br>

---
