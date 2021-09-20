# tsflex examples

This folder contains several examples, indicating (1.) the cross-domain applicability of tsflex and (2.) how tsflex integrates seamless with other data science packages.

## 0. general examples

**Paper example**: [tsflex_paper.ipynb](https://github.com/predict-idlab/tsflex/blob/jeroen_examples/examples/tsflex_paper.ipynb)  
Example used in the tsflex paper. The example elaborates shows how tsflex can be applied for processing & feature extraction on multivariate (and even irregularly sampled) time series data.
<!-- TODO: add link to the paper -->

**Verbose example**: [verbose_example.ipynb](https://github.com/predict-idlab/tsflex/blob/jeroen_examples/examples/verbose_example.ipynb)  
Example that elaborates in great detail (very verbose) the various functionalities of tsflex. In addition to processing & feature extraction, this example shows how to use the logging functionality, serialization, and chunking.


## 1. tsflex cross-domain examples

tsflex is a domain independent package for time series processing & feature extraction;

| Domain | Dataset | ML Notebook |
| --- | --- | --- |
| Climate modelling | [Ozone level detection](https://archive.ics.uci.edu/ml/datasets/Ozone%20Level%20Detection) | [example_ozone_level_detection.ipynb](https://github.com/predict-idlab/tsflex/blob/main/examples/example_ozone_level_detection.ipynb) |  
| Household data | [Electric power consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption) | [example_power_consumption_estimation.ipynb](example_power_consumption_estimation.ipynb)

<!-- | Wearable data | [WESAD - Wearable stress & affect detection](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) | [verbose_example.ipynb](verbose_example.ipynb) - <br>`TODO` create a notebook that uses the whole wesad dataset -->

<!-- ML notebooks with sklearn, tslearn, sktime -->

## 2. tsflex integration examples

With existing popular data-science packages tsflex integrates natively:
* **Processing**: e.g., [scipy.signal](https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html), [statsmodels.tsa](https://www.statsmodels.org/stable/tsa.html#time-series-filters).
* **Feature extraction**: e.g., [numpy](https://numpy.org/doc/stable/reference/routines.html), [scipy.stats](https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html).


### Feature extraction

We highlight how tsflex integrates conveniently with popular time series feature extraction packages:

| package | example notebook |
| --- | --- |
| [seglearn](https://dmbee.github.io/seglearn/feature_functions.html) | [seglearn_integration.ipynb](https://github.com/predict-idlab/tsflex/blob/main/examples/seglearn_integration.ipynb)
| [tsfresh](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html) | [tsfresh_integration.ipynb](https://github.com/predict-idlab/tsflex/blob/main/examples/tsfresh_integration.ipynb) |
| [tsfel](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html) | [tsfel_integration.ipynb](https://github.com/predict-idlab/tsflex/blob/main/examples/tsfel_integration.ipynb) |


> As some of these time series feature extraction packages use different formats for their feature function, a wrapper function might be required to enable a convenient integration.  

*We encourage users to add example notebooks for other feature extraction packages (and if necessary, add the required wrapper function in the [`tsflex.features.integrations`](https://github.com/predict-idlab/tsflex/blob/main/tsflex/features/integrations.py) file.*  
=> More info on **contributing** can be found [here](https://github.com/predict-idlab/tsflex/blob/main/CONTRIBUTING.md).
