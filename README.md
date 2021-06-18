# <p align="center"><img alt="tsflex" src="./docs/_static/logo.png" height="100"></p>

*tsflex* stands for: _**flex**ible **t**ime-**s**eries operations_<br>

It is a `time-series first` toolkit for **processing & feature extraction**, making few assumptions about input data. 

* [example notebooks](examples/)

## Table of contents
  - [Installation](#installation)
  - [Usage](#usage)
    - [Series processing](#series-processing)
    - [Feature extraction](#feature-extraction)
  - [Documentation](#documentation)


## Installation

If you are using **pip**, just execute the following command:

```sh
pip install tsflex
```

## Usage

_tsflex_ is built to be intuitive, so we encourage you to copy-paste this code and toy with some parameters!


### Series processing

`:WIP:`

### Feature extraction

```python
import pandas as pd; import scipy.stats as ss; import numpy as np
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
            function=NumpyFuncWrapper(func=ss.skew, output_names="skew"),
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

## Documentation

To see the documentation locally, install [pdoc](https://github.com/pdoc3/pdoc) and execute the succeeding command from this folder location.

```sh
pdoc3 --template-dir docs/pdoc_template/ --http :8181 tsflex
```

<br>

---

<p align="center">
👤 <i>Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost</i>
</p>


