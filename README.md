# <div style="text-align: center;"><img alt="tsflex" src="./docs/_static/logo.png" height="60"></div>

*tsflex* is an abbreviation for: ***flex**ible **t**ime-**s**eries operations*

It is a `time-series first toolkit` for **processing & feature extraction**, making few assumptions about input data.

## Feature extraction

Using time-series data, the most classical way to extract features is by employing a **strided-window** approach.

* assume `stride of 1`
* assume `data is fixed frequency`
* 

---
The only data assumptions made by tsflex are:
* the data has a `pd.DatetimeIndex` & this index is `monotonically_increasing`
* the data's series names must be unique

### Feature extraction





### Advantages of tsflex

*fslex* has multiple selling points, for example

* it is efficient
  * execution time -> multiprocessing / vectorized
  * memory -> view based operations
* it is flexible:  
  **feature extraction**:
     * multiple series, signal & stride combinations are possible
     * no frequency requirements, just a datetime index
* it is field & unit tested
* it has a comprehensive documentation
* it is compatible with sklearn (w.i.p. for gridsearch integration), pandas and numpy

## Installation

```sh
pip install tsflex
```

### Feature extraction

```python
import pandas as pd
import scipy.stats

from tsflex import FeatureDescriptor, FeatureCollection

# get your time-indexed data
data = pd.Series(
    data=np.random.random(1000),
    index=pd.date_range("2021-07-01", freq="1h", periods=1000),
).rename('series_key')

# drop some data, as we don't make frequency assumptions
data = data.drop(np.random.choice(data.index, 200, replace=False))


# construct the collection in which you add all your features
fc = FeatureCollection(
    feature_descriptors=[
        FeatureDescriptor(
            function=scipy.stats.skew,
            series_name='series_key',
            window='1day',
            stride='6hours'
        )
    ]
)

fc.add(FeatureDescriptor(np.min, 'series_key', '2days', '1day'))
fc.calculate(data=data, n_jobs=1, return_df=True)
```
|                     |   **series_key__skew__w=1D_s=12h** |   **series_key__amin__w=2D_s=1D** |
|:--------------------|-------------------------------:|------------------------------:|
| 2021-07-02 00:00:00 |                     -0.0607221 |                   nan         |
| 2021-07-02 12:00:00 |                     -0.142407  |                   nan         |
| 2021-07-03 00:00:00 |                     -0.283447  |                     0.042413  |
| 2021-07-03 12:00:00 |                     -0.353314  |                   nan         |
| 2021-07-04 00:00:00 |                     -0.188953  |                     0.0011865 |
| 2021-07-04 12:00:00 |                      0.259685  |                   nan         |
| 2021-07-05 00:00:00 |                      0.726858  |                     0.0011865 |



<br>

---
👤 _Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost_



