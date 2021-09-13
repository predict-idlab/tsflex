# <p align="center"> <a href="https://predict-idlab.github.io/tsflex"><img alt="tsflex" src="https://raw.githubusercontent.com/predict-idlab/tsflex/main/docs/_static/logo.png" height="100"></a></p>

[![PyPI Latest Release](https://img.shields.io/pypi/v/tsflex.svg)](https://pypi.org/project/tsflex/)
[![Conda Latest Release](https://img.shields.io/conda/vn/conda-forge/tsflex?label=conda)](https://anaconda.org/conda-forge/tsflex)
[![support-version](https://img.shields.io/pypi/pyversions/tsflex)](https://img.shields.io/pypi/pyversions/tsflex)
[![codecov](https://img.shields.io/codecov/c/github/predict-idlab/tsflex?logo=codecov)](https://codecov.io/gh/predict-idlab/tsflex)
[![Code quality](https://img.shields.io/lgtm/grade/python/github/predict-idlab/tsflex?label=code%20quality&logo=lgtm)](https://lgtm.com/projects/g/predict-idlab/tsflex/context:python)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=black)
[![Downloads](https://pepy.tech/badge/tsflex)](https://pepy.tech/project/tsflex)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?)](http://makeapullrequest.com) 
[![Documentation](https://github.com/predict-idlab/tsflex/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/predict-idlab/tsflex/actions/workflows/deploy-docs.yml)
[![Testing](https://github.com/predict-idlab/tsflex/actions/workflows/test.yml/badge.svg)](https://github.com/predict-idlab/tsflex/actions/workflows/test.yml)

<!-- ![Downloads](https://img.shields.io/conda/dn/conda-forge/tsflex?logo=anaconda) -->

*tsflex* is a toolkit for _**flex**ible **t**ime **s**eries_ **[processing](https://predict-idlab.github.io/tsflex/processing) & [feature extraction](https://predict-idlab.github.io/tsflex/features)**, making few assumptions about input data. 

#### Useful links

- [Documentation](https://predict-idlab.github.io/tsflex/)
- [Example notebooks](https://github.com/predict-idlab/tsflex/tree/main/examples)

## Installation

If you are using [**pip**](https://pypi.org/project/tsflex/), just execute the following command:

```sh
pip install tsflex
```

Or, if you are using [**conda**](https://anaconda.org/conda-forge/tsflex), then execute this command:

```sh
conda install -c conda-forge tsflex
```

## Usage

_tsflex_ is built to be intuitive, so we encourage you to copy-paste this code and toy with some parameters!

### <a href="https://predict-idlab.github.io/tsflex/features/#getting-started">Feature extraction</a>

```python
import pandas as pd; import numpy as np; import scipy.stats as ss
from tsflex.features import MultipleFeatureDescriptors, FeatureCollection

# 1. -------- Get your time-indexed data --------
url = "https://github.com/predict-idlab/tsflex/raw/main/examples/data/empatica/"
# Contains 1 column; ["TMP"] - 4 Hz sampling rate
data_tmp = pd.read_parquet(url+"tmp.parquet").set_index("timestamp")
# Contains 3 columns; ["ACC_x", "ACC_y", "ACC_z"] - 32 Hz sampling rate
data_acc = pd.read_parquet(url+"acc.parquet").set_index("timestamp")

# 2. -------- Construct your feature collection --------
fc = FeatureCollection(
    MultipleFeatureDescriptors(
          functions=[np.min, np.max, np.mean, np.std, np.median, ss.skew, ss.kurtosis],
          series_names=["TMP", "ACC_x", "ACC_y"], # Use 3 multimodal signals 
          windows=["5min", "7.5min"],  # Use 5 minutes and 7.5 minutes 
          strides="2.5min",  # With steps of 2.5 minutes
    )
)

# 3. -------- Calculate features --------
fc.calculate(data=[data_tmp, data_acc])
```

### More examples

For processing [look here](https://predict-idlab.github.io/tsflex/processing/index.html#working-example)    
Other examples can be found [here](https://github.com/predict-idlab/tsflex/tree/main/examples)

## Why tsflex? ✨

* flexible;
    * handles multivariate/multimodal time series
    * versatile function support  
      => **integrates natively** with many packages for processing (e.g., [scipy.signal](https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html), [statsmodels.tsa](https://www.statsmodels.org/stable/tsa.html#time-series-filters)) & feature extraction (e.g., [numpy](https://numpy.org/doc/stable/reference/routines.html), [scipy.stats](https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html), [seglearn](https://dmbee.github.io/seglearn/feature_functions.html)¹, [tsfresh](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)¹, [tsfel](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html)¹)
    * feature-extraction handles **multiple strides & window sizes**
* efficient view-based operations  
  => extremely **low memory peak & fast execution times** ([see benchmarks](https://github.com/predict-idlab/tsflex-benchmarking))
    <!-- * faster than any existing library (single- & multi-core)
    * lower memory peak than any existing library (single- & multi-core) -->
* maintains the **time-index** of the data
* makes **little to no assumptions** about the time series data

¹ These integrations are shown in [integration-example notebooks](https://github.com/predict-idlab/tsflex/tree/main/examples).

## Future work 🔨

* scikit-learn integration for both processing and feature extraction<br>
  **note**: is actively developed upon [sklearn integration](https://github.com/predict-idlab/tsflex/tree/sklearn_integration) branch.
* support time series segmentation (exposing under the hood strided-rolling functionality)<br>
  **note**: [see more here](https://github.com/predict-idlab/tsflex/issues/15).
* support for multi-indexed dataframes

## Contributing 👪

We are thrilled to see your contributions to further enhance `tsflex`.<br>
See [this guide](CONTRIBUTING.md) for more instructions on how to contribute.

## Referencing our package

If you use `tsflex` in a scientific publication, we would highly appreciate citing us as:

```bibtex
@article{vanderdonckt2021tsflex,
    author = {Van Der Donckt, Jonas and Van Der Donckt, Jeroen and Deprost, Emiel and Van Hoecke, Sofie},
    title = {tsflex: flexible time series processing \& feature extraction},
    journal = {SoftwareX},
    year = {2021},
    url = {https://github.com/predict-idlab/tsflex},
    publisher={Elsevier}
}
```

---

<p align="center">
👤 <i>Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost</i>
</p>
