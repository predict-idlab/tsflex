# <p align="center"> <a href="https://predict-idlab.github.io/tsflex"><img alt="tsflex" src="https://raw.githubusercontent.com/predict-idlab/tsflex/main/docs/_static/logo.png" height="100"></a></p>

[![PyPI Latest Release](https://img.shields.io/pypi/v/tsflex.svg)](https://pypi.org/project/tsflex/)
[![Documentation](https://github.com/predict-idlab/tsflex/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/predict-idlab/tsflex/actions/workflows/deploy-docs.yml)
[![Testing](https://github.com/predict-idlab/tsflex/actions/workflows/test.yml/badge.svg)](https://github.com/predict-idlab/tsflex/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/predict-idlab/tsflex/branch/main/graph/badge.svg)](https://codecov.io/gh/predict-idlab/tsflex)
[![Code quality](https://img.shields.io/lgtm/grade/python/g/predict-idlab/tsflex.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/predict-idlab/tsflex/context:python)
[![Downloads](https://pepy.tech/badge/tsflex)](https://pepy.tech/project/tsflex)

*tsflex* is a toolkit for _**flex**ible **t**ime-**s**eries_ **[processing](https://predict-idlab.github.io/tsflex/processing) & [feature extraction](https://predict-idlab.github.io/tsflex/features)**, making few assumptions about input data. 

#### Useful links

- [Documentation](https://predict-idlab.github.io/tsflex/)
- [Example notebooks](https://github.com/predict-idlab/tsflex/tree/main/examples)

## Installation

If you are using [**pip**](https://pypi.org/project/tsflex/), just execute the following command:

```sh
pip install tsflex
```

## Why tsflex? ✨

* flexible;
    * handles multi-variate time-series
    * versatile function support  
      => **integrates natively** with many packages for processing (e.g., [scipy.signal](https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html)) & feature extraction (e.g., [numpy](https://numpy.org/doc/stable/reference/routines.html), [scipy.stats](https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html))
    * feature-extraction handles **multiple strides & window sizes**
* efficient view-based operations  
  => extremely **low memory peak & fast execution** times ([see benchmarks]())
    <!-- * faster than any existing library (single- & multi-core)
    * lower memory peak than any existing library (single- & multi-core) -->
* maintains the **time-index** of the data
* makes **little to no assumptions** about the time-series data

## Usage

_tsflex_ is built to be intuitive, so we encourage you to copy-paste this code and toy with some parameters!


### <a href="https://predict-idlab.github.io/tsflex/processing/#getting-started">Series processing</a>

```python
import pandas as pd; import scipy.signal as ssig; import numpy as np
from tsflex.processing import SeriesProcessor, SeriesPipeline

# 1. -------- Get your time-indexed data --------
# Data contains 3 columns; ["ACC_x", "ACC_y", "ACC_z"]
url = "https://github.com/predict-idlab/tsflex/raw/main/examples/data/empatica/acc.parquet"
data = pd.read_parquet(url).set_index("timestamp")

# 2 -------- Construct your processing pipeline --------
processing_pipe = SeriesPipeline(
    processors=[
        SeriesProcessor(function=np.abs, series_names=["ACC_x", "ACC_y", "ACC_z"]),
        SeriesProcessor(ssig.medfilt, ["ACC_x", "ACC_y", "ACC_z"], kernel_size=5)  # (with kwargs!)
    ]
)
# -- 2.1. Append processing steps to your processing pipeline
processing_pipe.append(SeriesProcessor(ssig.detrend, ["ACC_x", "ACC_y", "ACC_z"]))

# 3 -------- Process the data --------
processing_pipe.process(data=data)
```

### <a href="https://predict-idlab.github.io/tsflex/features/#getting-started">Feature extraction</a>

```python
import pandas as pd; import scipy.stats as ssig; import numpy as np
from tsflex.features import FeatureDescriptor, FeatureCollection, NumpyFuncWrapper

# 1. -------- Get your time-indexed data --------
# Data contains 1 column; ["TMP"]
url = "https://github.com/predict-idlab/tsflex/raw/main/examples/data/empatica/tmp.parquet"
data = pd.read_parquet(url).set_index("timestamp")

# 2 -------- Construct your feature collection --------
fc = FeatureCollection(
    feature_descriptors=[
        FeatureDescriptor(
            function=NumpyFuncWrapper(func=ssig.skew, output_names="skew"),
            series_name="TMP", 
            window="5min",  # Use 5 minutes 
            stride="2.5min",  # With steps of 2.5 minutes
        )
    ]
)
# -- 2.1. Add features to your feature collection
fc.add(FeatureDescriptor(np.min, "TMP", '2.5min', '2.5min'))

# 3 -------- Calculate features --------
fc.calculate(data=data)
```

### Scikit-learn integration

`TODO`

<br>

---

<p align="center">
👤 <i>Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost</i>
</p>


