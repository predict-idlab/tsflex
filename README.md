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

`:WIP: - not yet published to pypi`

```sh
pip install tsflex
```

## Advantages of tsflex

*tsflex* has multiple selling points, for example

`todo: create links to example benchmarking notebooks`

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

```python
import pandas as pd
import scipy.stats
import numpy as np

from tsflex.processing import SeriesProcessor, SeriesPipeline


```


### Feature extraction

The only data assumptions made by tsflex are:
* the data has a `pd.DatetimeIndex` & this index is `monotonically_increasing`
* the data's series names must be unique


```python
import pandas as pd
import scipy.stats
import numpy as np

from tsflex.features import FeatureDescriptor, FeatureCollection

# 1. Construct the collection in which you add all your features
fc = FeatureCollection(
    feature_descriptors=[
        FeatureDescriptor(
            function=scipy.stats.skew,
            series_name="myseries",
            window="1day",
            stride="6hours"
        )
    ]
)
# -- 1.1 Add another feature to the feature collection
fc.add(FeatureDescriptor(np.min, 'myseries', '2days', '1day'))

# 2. Get your time-indexed data
data = pd.Series(
    data=np.random.random(10_000), 
    index=pd.date_range("2021-07-01", freq="1h", periods=10_000),
).rename('myseries')
# -- 2.1 drop some data, as we don't make frequency assumptions
data = data.drop(np.random.choice(data.index, 200, replace=False))

# 3. Calculate the feature on some data
fc.calculate(data=data, n_jobs=1, return_df=True)
# which outputs: a pd.DataFrame with content:
```
|      index               |   **myseries__skew__w=1D_s=12h**  |    **myseries__amin__w=2D_s=1D** |
|:--------------------|-------------------------------:|------------------------------:|
| 2021-07-02 00:00:00 |                     -0.0607221 |                   nan         |
| 2021-07-02 12:00:00 |                     -0.142407  |                   nan         |
| 2021-07-03 00:00:00 |                     -0.283447  |                     0.042413  |
| 2021-07-03 12:00:00 |                     -0.353314  |                   nan         |
| 2021-07-04 00:00:00 |                     -0.188953  |                     0.0011865 |
| 2021-07-04 12:00:00 |                      0.259685  |                   nan         |
| 2021-07-05 00:00:00 |                      0.726858  |                     0.0011865 |
| ... |                      ...  |                     ... |


## Documentation

`:WIP:`

Too see the documentation locally, install [pdoc](https://github.com/pdoc3/pdoc) and execute the succeeding command from this folder location.

```sh
pdoc3 --template-dir docs/pdoc_template/ --http :8181 tsflex
```

<br>



---
<p align="center">
👤 <i>Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost</i>
</p>


