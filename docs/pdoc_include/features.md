# Feature extraction example

The only data assumptions made by tsflex are:

* the data has a `pd.DatetimeIndex` & this index is _monotonically increasing_
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
            series_name="s_name",
            window="1day",
            stride="6hours"
        )
    ]
)
# -- 1.1 Add another feature to the feature collection
fc.add(FeatureDescriptor(np.min, 's_name', '2days', '1day'))

# 2. Get your time-indexed data
data = pd.Series(
    data=np.random.random(10_000), 
    index=pd.date_range("2021-07-01", freq="1h", periods=10_000),
).rename('s_name')
# -- 2.1 drop some data, as we don't make frequency assumptions
data = data.drop(np.random.choice(data.index, 200, replace=False))

# 3. Calculate the feature on some data
fc.calculate(data=data, n_jobs=1, return_df=True)
# which outputs:
```
|      index               |   **s_name__skew__w=1D_s=12h**  |    **s_name__amin__w=2D_s=1D** |
|:--------------------|-------------------------------:|------------------------------:|
| 2021-07-02 00:00:00 |                     -0.0607221 |                   nan         |
| 2021-07-02 12:00:00 |                     -0.142407  |                   nan         |
| 2021-07-03 00:00:00 |                     -0.283447  |                     0.042413  |
| 2021-07-03 12:00:00 |                     -0.353314  |                   nan         |
| 2021-07-04 00:00:00 |                     -0.188953  |                     0.0011865 |
| 2021-07-04 12:00:00 |                      0.259685  |                   nan         |
| 2021-07-05 00:00:00 |                      0.726858  |                     0.0011865 |


