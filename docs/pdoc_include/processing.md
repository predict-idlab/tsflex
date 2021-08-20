# Processing guide

The following sections will explain the processing module in detail.

<!-- <div style="text-align: center;"> -->
<h3><b><a href="#header-submodules">Jump to API reference</a></b></h3>
<!-- </div> -->
<br>

## Working example ✅

_tsflex_ is built to be intuitive, so we encourage you to copy-paste this code and toy with some parameters! <br>

This executable example creates a processing pipeline that contains 3 processing steps (abs, median filter, and detreding, each applied on a different subset of series). <br>

```python
import pandas as pd; import scipy.signal as ss; import numpy as np
from tsflex.processing import SeriesProcessor, SeriesPipeline

# 1. -------- Get your time-indexed data --------
# Data contains 3 columns; ["ACC_x", "ACC_y", "ACC_z"]
url = "https://github.com/predict-idlab/tsflex/raw/main/examples/data/empatica/"
data = pd.read_parquet(url + "acc.parquet").set_index("timestamp")

# 2 -------- Construct your processing pipeline --------
processing_pipe = SeriesPipeline(
    processors=[
        SeriesProcessor(function=np.abs, series_names=["ACC_x", "ACC_y"]),
        SeriesProcessor(ss.medfilt, ["ACC_y", "ACC_z"], kernel_size=5) 
    ]
)
# -- 2.1. Append processing steps to your processing pipeline
processing_pipe.append(SeriesProcessor(ss.detrend, ["ACC_x", "ACC_z"]))

# 3 -------- Process the data --------
processing_pipe.process(data=data, return_df=True)
# which outputs:
```
| timestamp                        |    ACC_x |   ACC_y |   ACC_z |
|:---------------------------------|---------:|--------:|--------:|
| 2017-06-13 14:22:13+02:00        | -32.8736 |  5.0000 | 51.1051 |
| 2017-06-13 14:22:13.031250+02:00 | -32.8737 |  5.0000 | 51.1051 |
| 2017-06-13 14:22:13.062500+02:00 | -32.8738 |  5.0000 | 51.105  |
| 2017-06-13 14:22:13.093750+02:00 | -32.8739 |  5.0000 | 51.105  |
| 2017-06-13 14:22:13.125000+02:00 | -32.8740 |  5.0000 | 51.1049 |
| ...                              | ...      | ...     | ...     |
<br>

!!!tip 
    More advanced processing examples can be found [in these example notebooks](https://github.com/predict-idlab/tsflex/tree/main/examples)

<br>

## Getting started 🚀

The processing functionality of _tsflex_ is provided by a `SeriesPipeline` that contains `SeriesProcessor` steps. The processing steps are applied sequentially on the data that is passed to the processing pipeline.

### Components
![processing uml](https://raw.githubusercontent.com/predict-idlab/tsflex/main/docs/_static/series_uml.png)

As shown above, there are 2 relevant classes for processing.

1. [SeriesPipeline](/tsflex/processing/#tsflex.processing.SeriesPipeline): serves as a pipeline, withholding the to-be-applied _processing steps_
2. [SeriesProcessor](/tsflex/processing/#tsflex.processing.SeriesProcessor): an instance of this class describes a _processing step_. <br>Processors are defined by:
      * `function`: the _Callable_ processing-function - e.g. _scipy.signal.detrend_
      * `series_names`: the _name(s)_ of the series on which the processing function should be applied
      * `**kwargs`: the _keyword arguments_ for the `function`.

The snippet below shows how the `SeriesPipeline` & `SeriesProcessor` components work:

```python
import numpy as np; import scipy.signal as ss
from tsflex.processing import SeriesProcessor, SeriesPipeline

# The SeriesPipeline takes a List[SeriesProcessor] as input
processing_pipe = SeriesPipeline(processors=[
        SeriesProcessor(np.abs, ["series_a", "series_b"]),
        SeriesProcessor(ss.medfilt, "series_b", kernel_size=5) # (with kwargs)
    ]
)
# We can still append processing steps after instantiating.
processing_pipe.append(processor=SeriesProcessor(ss.detrend, "series_a"))

# Apply the processing steps
processing_pipe.process(...)
```

### Processing functions

The function that processes the series should match this prototype:

    function(*series: pd.Series, **kwargs)
        -> Union[np.ndarray, pd.Series, pd.DataFrame, List[pd.Series]]

Hence, the processing function should take one (or multiple) series as input, these may be followed by some keyword arguments. The output of a processing function can be rather versatile.

.. note::
    A function that processes a ``np.ndarray`` instead of ``pd.Series``
    should work just fine.


In [this section](#advanced-usage) you can find more info on advanced usage of processing functions.

<br>

## Important notes 📢

<!-- As processing steps (`SeriesProcessor`s) are applied sequentially in the processing pipeline (`SeriesPipeline`), the order of these steps (might) affect the output. -->

In a `SeriesPipeline` it is common behavior that series are **transformed** (i.e., replaced).
Hence, it is important to keep the following principles in mind when:

- Applying processing functions that take 1 series as input will (generally) <b style="color:red">transform (i.e., replace) the input series</b> .
    - **Countermeassure**: this behavior does not occur when the processing function returns a `pd.Series` _with a different name_.
- The order of steps (`SeriesProcessor`s) in the processing pipeline (`SeriesPipeline`) might affect the output.


<br>

## Advanced usage 👀

### Versatile processing functions

As [explained above](#processing-functions) _tsflex_ is rather versatile in terms of function input and output.

_tsflex_ does not just allow ``one-to-one`` processing functions, but also ``many-to-one``, ``one-to-many``, and ``many-to-many`` functions are supported in a convenient way:

- ``many-to-one``; the **processing function** should 
    - take multiple series as input 
    - output a single array or (named!) series / dataframe with 1 column

    Example
```python
def abs_diff(s1: pd.Series, s2: pd.Series) -> pd.Series:
    return pd.Series(np.abs(s1-s2), name=f"abs_diff_{s1.name}-{s2.name}")
```
- ``one-to-many``; the **processing function** should
    - take a single series as input  
    - output a list of (named!) series or a dataframe with multiple columns
    
    Example
```python
def abs_square(s1: pd.Series) -> List[pd.Series]:
    s1_abs = pd.Series(np.abs(s1), name=f"abs_{s1.name}")
    s1_square = pd.Series(np.square(s1), name=f"square_{s1.name}")
    return [s1_abs, s1_square]
```

- ``many-to-many``; _(combination of the above)_ the **processing function** should
    - take multiple series as input
    - output a list of (named!) series or a dataframe with multiple columns

    Example
```python
def abs_square_diff(s1: pd.Series, s2: pd.Series) -> List[pd.Series]:
    s_abs = pd.Series(np.abs(s1-s2), name=f"abs_{s1.name}-{s2.name}")
    s_square = pd.Series(np.square(s1-s2), name=f"square_{s1.name}-{s2.name}")
    return [s_abs, s_square]
```

### DataFrame decorator

In some (rare) cases a processing function requires a ``pd.DataFrame`` as input. 
For these cases we provide the [dataframe_func decorator](#tsflex.processing.dataframe_func). This decorator wraps the processing function in the `SeriesPipeline`, provided a ``pd.DataFrame`` as input instead of multiple ``pd.Series``.

!!!note
    In most cases series arguments are sufficient; you can perform column-based operations on multiple `pd.Series` (e.g., subtract 2 series). Only when row-based operations are required (e.g., `df.dropna(axis=0)`), a `pd.DataFrame` is unavoidable.

### Logging

When a `logging_file_path` is passed to the `SeriesPipeline` its `process` method, the execution times of the processing steps will be logged.

[More info](#tsflex.processing.get_processor_logs)

<br>

---
