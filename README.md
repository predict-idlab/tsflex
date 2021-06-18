# <p align="center"><img alt="tsflex" src="./docs/_static/logo.png" height="100"></p>

*tsflex* stands for: _**flex**ible **t**ime-**s**eries operations_<br>

It is a `time-series first` toolkit for **processing & feature extraction**, making few assumptions about input data. 

* [example notebooks](examples/)

## Table of contents
  - [Installation](#installation)
  - [Usage](#usage)
    - [Series processing](#series-processing)
    - [Feature extraction](#feature-extraction-1)
  - [Documentation](#documentation)


## Installation

Installing tsflex is as straightforward as any other Python package. If you are using **pip**, just execute the following command in your environment.

```sh
pip install tsflex
```

## Advantages of tsflex

*tsflex* has multiple selling points, for example

`todo`: create links to example benchmarking notebooks

* it is efficient
  * execution time -> multiprocessing / vectorized
  * memory -> view based operations
* it is flexible:  
  **feature extraction**:
     * multiple series, signal & stride combinations are possible
     * no frequency requirements, just a datetime index
* it has logging capabilities to improve feature extraction speed.  
* it is field & unit tested
* it has a comprehensive documentation
* it is compatible with sklearn (w.i.p. for gridsearch integration), pandas and numpy

## Usage

### Series processing

### Feature extraction

The only data assumptions made by tsflex are:
* the data has a `pd.DatetimeIndex` & this index is `monotonically_increasing`
* the data's series names must be unique


```python
import pandas as pd; import scipy.stats as ss; import numpy as np
from tsflex.features import FeatureDescriptor, FeatureCollection, NumpyFuncWrapper
from tsflex.features import MultipleFeatureDescriptors

# 1. Construct the collection in which you add all your features
fc = FeatureCollection(
    feature_descriptors=[
        FeatureDescriptor(
            function=NumpyFuncWrapper(func=ss.skew,output_names="skew"),
            series_name="lux", window="1day", stride="6hours"
        )
    ]
)

# -- 1.1. Add multiple features to your feature collection
fc.add(FeatureDescriptor(np.min, 'lux', '2days', '1day'))
fc.add(MultipleFeatureDescriptors(
    functions=[ 
        np.mean, np.std,
        NumpyFuncWrapper(func=lambda x: np.sum(np.abs(x)), output_names="abssum") 
    ],
    series_names="lux", windows=["1day", "2days", "3hours"],
    strides=["3hours"]
))

# 2. Get your time-indexed data
## TODO -> look into wget time series public
data = pd.Series(data=np.random.random(10_000), 
    index=pd.date_range("2021-07-01", freq="1h", periods=10_000)).rename('lux')
# -- 2.1 drop some data, as we don't make frequency assumptions
data = data.drop(np.random.choice(data.index, 200, replace=False))

# 3. Calculate the feature on some data
fc.calculate(data=data, n_jobs=1, return_df=True)
# which outputs an outer merged dataframe with content
```
|      index          |  **lux__skew__w=1D_s=12h**  |   **lux__amin__w=2D_s=1D** |  **lux__...** |
|:-------------------:|:-------------------------------|:------------------------------|:---|
| 2021-07-02 00:00:00 |                     -0.0607221 |                   nan         |   ... |
| 2021-07-02 12:00:00 |                     -0.142407  |                   nan         |  ... |
| 2021-07-03 00:00:00 |                     -0.283447  |                     0.042413  | ... |
| 2021-07-03 12:00:00 |                     -0.353314  |                   nan         | ... |
| 2021-07-04 00:00:00 |                     -0.188953  |                     0.0011865 | ... |
| 2021-07-04 12:00:00 |                      0.259685  |                   nan         | ... |
| 2021-07-05 00:00:00 |                      0.726858  |                     0.0011865 | ... |
| ... |                      ...  |                     ... | ... |


## Documentation

Too see the documentation locally, install [pdoc](https://github.com/pdoc3/pdoc) and execute the succeeding command from this folder location.

```sh
pdoc3 --template-dir docs/pdoc_template/ --http :8181 tsflex
```

<br>



---
<p align="center">
👤 <i>Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost</i>
</p>


