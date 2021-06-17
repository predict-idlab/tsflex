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
  - [Advanced usage](#advanced-usage)
     - [Data chunking](/tsflex/chunking)
  - [API reference](#header-submodules) 

<br>
___ 

## Installation

Installing tsflex is as straightforward as any other Python package. If you are using **pip**, just execute the following command in your environment.

```shell
pip install tsflex
```
<br>
When using  **conda**, just run:

```shell
conda install tsflex
```

## Getting started

*tsflex* serves 2 main functionalities; _signal processing_ and _feature extraction_:

* The [processing](/tsflex/processing) module withholds a `SeriesPipeline` in which uni- and multivariate signal processing operations can be defined.
* The [feature extraction](/tsflex/features) module defines a `FeatureCollection` which does the bookkeeping of the defined features for the data.



## Data Formats

_tsflex_ was mainly intended to work on [flat data]()

The only data assumptions made by tsflex are:

* the data has a `pd.DatetimeIndex` & this index is _monotonically increasing_
* the data's series names must be unique
