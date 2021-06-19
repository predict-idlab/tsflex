# <p align="center"> <a href="https://tsflex.github.io/tsflex"><img alt="tsflex" src="https://raw.githubusercontent.com/tsflex/tsflex/main/docs/_static/logo.png" height="100"></a></p>

[![PyPI Latest Release](https://img.shields.io/pypi/v/tsflex.svg)](https://pypi.org/project/tsflex/)
[![Documentation](https://github.com/tsflex/tsflex/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/tsflex/tsflex/actions/workflows/deploy-docs.yml)
[![Testing](https://github.com/tsflex/tsflex/actions/workflows/test.yml/badge.svg)](https://github.com/tsflex/tsflex/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/tsflex/tsflex/branch/main/graph/badge.svg)](https://codecov.io/gh/tsflex/tsflex)
[![Downloads](https://pepy.tech/badge/tsflex)](https://pepy.tech/project/tsflex)

*tsflex* stands for: _**flex**ible **t**ime-**s**eries operations_<br>

It is a `time-series first` toolkit for **processing & feature extraction**, making few assumptions about input data. 

#### Useful links

- [Documentation](https://tsflex.github.io/tsflex/)
- [Example notebooks](https://github.com/tsflex/tsflex/tree/main/examples)

## Installation

If you are using [**pip**](https://pypi.org/project/tsflex/), just execute the following command:

```sh
pip install tsflex
```

## Usage

_tsflex_ is built to be intuitive, so we encourage you to copy-paste this code and toy with some parameters!


### Series processing

```python
import pandas as pd; import scipy.signal as ssig; import numpy as np
from tsflex.processing import SeriesProcessor, SeriesPipeline

# 1. -------- Get your time-indexed data --------
series_size = 10_000
series_name="lux"

data = pd.Series(
    data=np.random.random(series_size), 
    index=pd.date_range("2021-07-01", freq="1h", periods=series_size)
).rename(series_name)


# 2 -------- Construct your processing pipeline --------
processing_pipe = SeriesPipeline(
    processors=[
        SeriesProcessor(np.abs, series_names=series_name),
        SeriesProcessor(ssig.medfilt, series_name, kernel_size=5)  # (with kwargs!)
    ]
)
# -- 2.1. Append processing steps to your processing pipeline
processing_pipe.append(SeriesProcessor(ssig.detrend, series_name))

# 3 -------- Calculate features --------
processing_pipe.process(data=data)
```

### Feature extraction

```python
import pandas as pd; import scipy.stats as sstats; import numpy as np
from tsflex.features import FeatureDescriptor, FeatureCollection, NumpyFuncWrapper

# 1. -------- Get your time-indexed data --------
series_size = 10_000
series_name="lux"

data = pd.Series(
    data=np.random.random(series_size), 
    index=pd.date_range("2021-07-01", freq="1h", periods=series_size)
).rename(series_name)
# -- 1.1 drop some data, as we don't make frequency assumptions
data = data.drop(np.random.choice(data.index, 200, replace=False))


# 2 -------- Construct your feature collection --------
fc = FeatureCollection(
    feature_descriptors=[
        FeatureDescriptor(
            function=NumpyFuncWrapper(func=sstats.skew, output_names="skew"),
            series_name=series_name, 
            window="1day", stride="6hours"
        )
    ]
)
# -- 2.1. Add multiple features to your feature collection
fc.add(FeatureDescriptor(np.min, series_name, '2days', '1day'))


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


