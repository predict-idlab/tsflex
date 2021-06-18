This is the documentation of **tsflex**, which is a `time-series first` Python toolkit for 
**processing & feature extraction**, making few assumptions about input data.

This makes _tsflex_ suitable for use-cases such as inference on streaming data, performing operations on irregularly sampled time-series, and dealing with time-gaps.

> ~ _With great flexibility comes great responsiblity, read our docs!_ ~ - the tsflex devs

## Contents

The following sections will explain the tsflex package in detail 

  - [Installation](#installation)
  - [Getting started](#getting-started)
    - [Series processing](/tsflex/processing)
    - [Feature extraction](/tsflex/features)
  - [Data formats](#data-formats) 
  - [Advanced usage](#advanced-usage)
     - [Data chunking](/tsflex/chunking)
  - [API reference](#header-submodules) 

<br>

## Installation ⚙️

Installing tsflex is as straightforward as any other Python package. If you are using **pip**, just execute the following command in your environment.

```shell
pip install tsflex
```

<br>

## Getting started

*tsflex* serves 2 main functionalities; _signal processing_ and _feature extraction_:

* The [processing](/tsflex/processing) module withholds a `SeriesPipeline` in which uni- and multivariate signal processing operations can be defined.
* The [feature extraction](/tsflex/features) module defines a `FeatureCollection` which does the bookkeeping of the defined features for the data.

<br>

## Data formats 🗄️

*tsflex* leverages the flexibility and convenience that [Pandas](https://pandas.pydata.org/docs/index.html) delivers. This has the consequence that your input should always be either one or more `pd.Series/pd.DataFrame`. Using type-hinting, the input-data can be defined as:

```python
import pandas as pd; from typing import Union, List
data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
```

For brevity, we call an item from `data`, i.e., series or dataframe-colum, a time-series (`ts`). 

> tsflex was mainly intended to work on flat data such as a list of series or a wide-dataframe.

For more infromation, read the secion below. 

### Wide vs. Long Data
![image](../_static/long_wide.png)

Time series data is often stored in 2 data-types:

1. `Wide` time-series data, also known as flat data, is the most common variant. Each column represents a data-modality and the **index** is the **shared time**.<br>
    **Note**: As shown in the example above, not all data-modalities might have values for the shared (~union) index. This circumstance introduces Not a Number (`NaN`) entries.
2. `Long` time-series data, which consists of 3 columns:
      * A _non-index_ `time`-column, which thus can withhold duplicates
      * A `kind` column, defining the `ts` its name.
      * A `value` column, withholding the value for the corresponding _kind-time_ combination

> **_tsflex_ was built to support `wide` & `series-list`data**

!!!tip
    If you use long data, you might want to convert this to other modalities.<br>
    Remark that it is not recommended to transform `long->wide` as this might introduce `NaN`s, which may cause uwanted processing or feature-extraction behavior.<br>
    The snippet below serves the functionality for the `long->series-list` transformation.

```python
import pandas as pd; from typing import List

def long_dataframe_to_series_list(
    long_df: pd.DataFrame, time_col: str, value_col: str, kind_col: str
) -> List[pd.Series]:
    codes, uniques = pd.factorize(long_df[kind_col])
    series_list = []
    for idx, unique in enumerate(uniques):
        series_list.append(
            pd.Series(
                data=long_df.loc[codes == idx, value_col].values,
                index=long_df.loc[codes == idx, time_col],
                name=unique,
            )
        )
    return series_list
```
### Supported data-types

`Todo`