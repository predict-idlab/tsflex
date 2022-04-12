# <p align="center"> <a href="https://predict-idlab.github.io/tsflex"><img alt="tsflex" src="https://raw.githubusercontent.com/predict-idlab/tsflex/main/docs/_static/logo.png" width="66%"></a></p>

[![PyPI Latest Release](https://img.shields.io/pypi/v/tsflex.svg)](https://pypi.org/project/tsflex/)
[![Conda Latest Release](https://img.shields.io/conda/vn/conda-forge/tsflex?label=conda)](https://anaconda.org/conda-forge/tsflex)
[![support-version](https://img.shields.io/pypi/pyversions/tsflex)](https://img.shields.io/pypi/pyversions/tsflex)
[![codecov](https://img.shields.io/codecov/c/github/predict-idlab/tsflex?logo=codecov)](https://codecov.io/gh/predict-idlab/tsflex)
[![Code quality](https://img.shields.io/lgtm/grade/python/github/predict-idlab/tsflex?label=code%20quality&logo=lgtm)](https://lgtm.com/projects/g/predict-idlab/tsflex/context:python)
[![Downloads](https://pepy.tech/badge/tsflex)](https://pepy.tech/project/tsflex)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?)](http://makeapullrequest.com)
[![Documentation](https://github.com/predict-idlab/tsflex/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/predict-idlab/tsflex/actions/workflows/deploy-docs.yml)
[![Testing](https://github.com/predict-idlab/tsflex/actions/workflows/test.yml/badge.svg)](https://github.com/predict-idlab/tsflex/actions/workflows/test.yml)

<!-- ![Downloads](https://img.shields.io/conda/dn/conda-forge/tsflex?logo=anaconda) -->

> *tsflex* is a toolkit for _**flex**ible **t**ime **s**eries_ [processing](https://predict-idlab.github.io/tsflex/processing) & [feature extraction](https://predict-idlab.github.io/tsflex/features), that is efficient and makes few assumptions about sequence data.

#### Useful links

- [Paper](https://www.sciencedirect.com/science/article/pii/S2352711021001904)
- [Documentation](https://predict-idlab.github.io/tsflex/)
- [Example (machine learning) notebooks](https://github.com/predict-idlab/tsflex/tree/main/examples)

#### Installation

| | command|
|:--------------|:--------------|
| [**pip**](https://pypi.org/project/tsflex/) | `pip install tsflex` | 
| [**conda**](https://anaconda.org/conda-forge/tsflex) | `conda install -c conda-forge tsflex` |

## Usage

_tsflex_ is built to be intuitive, so we encourage you to copy-paste this code and toy with some parameters!

### <a href="https://predict-idlab.github.io/tsflex/features/#getting-started">Feature extraction</a>

```python
import pandas as pd; import numpy as np; import scipy.stats as ss
from tsflex.features import MultipleFeatureDescriptors, FeatureCollection
from tsflex.utils.data import load_empatica_data

# 1. Load sequence-indexed data (in this case a time-index)
df_tmp, df_acc, df_ibi = load_empatica_data(['tmp', 'acc', 'ibi'])

# 2. Construct your feature extraction configuration
fc = FeatureCollection(
    MultipleFeatureDescriptors(
          functions=[np.min, np.mean, np.std, ss.skew, ss.kurtosis],
          series_names=["TMP", "ACC_x", "ACC_y", "IBI"],
          windows=["15min", "30min"],
          strides="15min",
    )
)

# 3. Extract features
fc.calculate(data=[df_tmp, df_acc, df_ibi], approve_sparsity=True)
```

Note that the feature extraction is performed on multivariate data with varying sample rates.
| signal | columns | sample rate |
|:-------|:-------|------------------:|
| df_tmp | ["TMP"]| 4Hz |
| df_acc | ["ACC_x", "ACC_y", "ACC_z" ]| 32Hz |
| df_ibi | ["IBI"]| irregularly sampled |

### <a href="https://predict-idlab.github.io/tsflex/processing/index.html#getting-started">Processing</a>
[Working example in our docs](https://predict-idlab.github.io/tsflex/processing/index.html#working-example)

## Why tsflex? ✨

* `Flexible`:
    * handles multivariate/multimodal time series
    * versatile function support
      => **integrates** with many packages for:
      * processing (e.g., [scipy.signal](https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html), [statsmodels.tsa](https://www.statsmodels.org/stable/tsa.html#time-series-filters))
      * feature extraction (e.g., [numpy](https://numpy.org/doc/stable/reference/routines.html), [scipy.stats](https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html), [seglearn](https://dmbee.github.io/seglearn/feature_functions.html)¹, [tsfresh](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)¹, [tsfel](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html)¹)
    * feature extraction handles **multiple strides & window sizes**
* `Efficient`:<br>
  * view-based operations for processing & feature extraction => extremely **low memory peak** & **fast execution time**<br>
    * see: [feature extraction benchmark visualization](https://predict-idlab.github.io/tsflex/#benchmark)
* `Intuitive`:<br>
  * maintains the sequence-index of the data
  * feature extraction constructs interpretable output column names
  * intuitive API
* `Few assumptions` about the sequence data:
  * no assumptions about sampling rate
  * able to deal with multivariate asynchronous data<br>i.e. data with small time-offsets between the modalities
* `Advanced functionalities`:
  * apply [FeatureCollection.**reduce**](https://predict-idlab.github.io/tsflex/features/index.html#tsflex.features.FeatureCollection.reduce) after feature selection for faster inference
  * use **function execution time logging** to discover processing and feature extraction bottlenecks
  * embedded [SeriesPipeline](http://predict-idlab.github.io/tsflex/processing/#tsflex.processing.SeriesPipeline.serialize) & [FeatureCollection](https://predict-idlab.github.io/tsflex/features/index.html#tsflex.features.FeatureCollection.serialize) **serialization**
  * time series [**chunking**](https://predict-idlab.github.io/tsflex/chunking/index.html)

¹ These integrations are shown in [integration-example notebooks](https://github.com/predict-idlab/tsflex/tree/main/examples).
## Future work 🔨

* scikit-learn integration for both processing and feature extraction<br>
  **note**: is actively developed upon [sklearn integration](https://github.com/predict-idlab/tsflex/tree/sklearn_integration) branch.
* Support time series segmentation (exposing under the hood strided-rolling functionality) - [see this issue](https://github.com/predict-idlab/tsflex/issues/15)
* Support for multi-indexed dataframes

=> Also see the [enhancement issues](https://github.com/predict-idlab/tsflex/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement+)
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

Link to the paper: https://www.sciencedirect.com/science/article/pii/S2352711021001904

---

<p align="center">
👤 <i>Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost</i>
</p>
