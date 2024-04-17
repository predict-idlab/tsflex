"""

FeatureDescriptor and MultipleFeatureDescriptors class for creating time-series
features.

"""

import itertools
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from ..utils.argument_parsing import parse_time_arg
from ..utils.attribute_parsing import AttributeParser, DataType
from ..utils.classes import FrozenClass
from ..utils.data import to_list, to_tuple
from .function_wrapper import FuncWrapper


class FeatureDescriptor(FrozenClass):
    """A FeatureDescriptor object, containing all feature information.

    Parameters
    ----------
    function : Union[FuncWrapper, Callable]
        The function that calculates the feature(s).
        The prototype of the function should match: \n

            function(*series: Union[np.array, pd.Series])
                -> Union[Any, List[Any]]

        Note that when the input type is ``pd.Series``, the function should be wrapped
          in a `FuncWrapper` with `input_type` = ``pd.Series``.

    series_name : Union[str, Tuple[str, ...]]
        The names of the series on which the feature function should be applied.
        This argument should match the `function` its input; \n
        * If `series_name` is a string (or tuple of a single string), then
            `function` should require just one series as input.
        * If `series_name` is a tuple of strings, then `function` should
            require `len(tuple)` series as input **and in exactly the same order**

    window : Union[float, str, pd.Timedelta], optional
        The window size. By default None. This argument supports multiple types: \n
        * If None, the `segment_start_idxs` and `segment_end_idxs` will need to be
          passed.
        * If the type is an `float` or an `int`, its value represents the series
            - its window **range** when a **non time-indexed** series is passed.
            - its window in **number of samples**, when a **time-indexed** series is
              passed (must then be and `int`)
        * If the window's type is a `pd.Timedelta`, the window size represents
          the window-time-range. The passed data **must have a time-index**.
        * If a `str`, it must represents a window-time-range-string. The **passed data
          must have a time-index**.
        .. Note::
            - When the `segment_start_idxs` and `segment_end_idxs` are both passed to
              the `FeatureCollection.calculate` method, this window argument is ignored.
              Note that this is the only case when it is allowed to pass None for the
              window argument.
            - When the window argument is None, than the stride argument should be None
              as well (as it makes no sense to pass a stride value when the window is
              None).

    stride : Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]], optional
        The stride size(s). By default None. This argument supports multiple types: \n
        * If None, the stride will need to be passed to `FeatureCollection.calculate`.
        * If the type is an `float` or an `int`, its value represents the series
            - its stride **range** when a **non time-indexed** series is passed.
            - the stride in **number of samples**, when a **time-indexed** series
              is passed (must then be and `int`)
        * If the stride's type is a `pd.Timedelta`, the stride size represents
          the stride-time delta. The passed data **must have a time-index**.
        * If a `str`, it must represent a stride-time-delta-string. The **passed data
          must have a time-index**. \n
        * If a `List[Union[float, str, pd.Timedelta]]`, then the set intersection,of the
          strides will be used (e.g., stride=[2,3] -> index: 0, 2, 3, 4, 6, 8, 9, ...)
        .. Note::
            - The stride argument of `FeatureCollection.calculate` takes precedence over
              this value when set (i.e., not None value for `stride` passed to the
              `calculate` method).
            - The stride argument should be None when the window argument is None (as it
              makes no sense to pass a stride value when the window is None).

    .. Note::
        As described above, the `window-stride` argument can be sample-based (when using
        time-index series and int based arguments), but we
        do **not encourage** using this for `time-indexed` sequences. As we make the
        implicit assumption that the time-based data is sampled at a fixed frequency
        So only, if you're 100% sure that this is correct, you can safely use such
        arguments.

    Notes
    -----
    * The `window` and `stride` argument should be either **both** numeric or
      ``pd.Timedelta`` (depending on de index datatype) - when `stride` is not None.
    * For each `function` - `input`(-series) - `window` - stride combination, one needs
      to create a distinct `FeatureDescriptor`. Hence it is more convenient to
      create a `MultipleFeatureDescriptors` when `function` - `window` - `stride`
      **combinations** should be applied on various input-series (combinations).
    * When `function` takes **multiple series** (i.e., arguments) as **input**, these
      are joined (based on the index) before applying the function. If the indexes of
      these series are not exactly the same, it might occur that not all series have
      exactly the same length! Hence,  make sure that the `function` can deal with
      this!
    * For more information about the str-based time args, look into:
      [pandas time delta](https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html#parsing){:target="_blank"}

    Raises
    ------
    TypeError
        * Raised when the `function` is not an instance of Callable or FuncWrapper.
        * Raised when `window` and `stride` are not of exactly the same type (when
          `stride` is not None).

    See Also
    --------
    StridedRolling: As the window-stride sequence conversion takes place there.

    """

    def __init__(
        self,
        function: Union[FuncWrapper, Callable],
        series_name: Union[str, Tuple[str, ...]],
        window: Optional[Union[float, str, pd.Timedelta]] = None,
        stride: Optional[
            Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]]
        ] = None,
    ):
        strides = sorted(set(to_list(stride)))  # omit duplicate stride values
        if window is None:
            assert strides == [None], "stride must be None if window is None"
        self.series_name: Tuple[str, ...] = to_tuple(series_name)
        self.window = parse_time_arg(window) if isinstance(window, str) else window
        if strides == [None]:
            self.stride = None
        else:
            self.stride = [
                parse_time_arg(s) if isinstance(s, str) else s for s in strides
            ]

        # Verify whether window and stride are either both sequence or time based
        dtype_set = set(
            AttributeParser.determine_type(v)
            for v in [self.window] + to_list(self.stride)
        ).difference([DataType.UNDEFINED])
        if len(dtype_set) > 1:
            raise TypeError(
                f"a combination of window ({self.window} type={type(self.window)}) and"
                f" stride ({self.stride}) is not supported!"
            )

        # Order of if statements is important (as FuncWrapper also is a Callable)!
        if isinstance(function, FuncWrapper):
            self.function: FuncWrapper = function
        elif isinstance(function, Callable):  # type: ignore[arg-type]
            self.function: FuncWrapper = FuncWrapper(function)  # type: ignore[no-redef]
        else:
            raise TypeError(
                "Expected feature function to be `Callable` or `FuncWrapper` but is a"
                f" {type(function)}."
            )

        # Construct a function-string
        f_name = str(self.function)
        self._func_str: str = f"{self.__class__.__name__} - func: {f_name}"

        self._freeze()

    def get_required_series(self) -> List[str]:
        """Return all required series names for this feature descriptor.

        Return the list of series names that are required in order to execute the
        feature function.

        Returns
        -------
        List[str]
            List of all the required series names.

        """
        return list(set(self.series_name))

    def get_nb_output_features(self) -> int:
        """Return the number of output features of this feature descriptor.

        Returns
        -------
        int
            Number of output features.

        """
        return len(self.function.output_names)

    def __repr__(self) -> str:
        """Representation string of Feature."""
        return (
            f"{self.__class__.__name__}({self.series_name}, {self.window}, "
            f"{self.stride})"
        )


class MultipleFeatureDescriptors:
    """Create a MultipleFeatureDescriptors object.

    Create a list of features from **all** combinations of the given parameter
    lists. Total number of created `FeatureDescriptor`s will be:

        len(func_inputs)*len(functions)*len(windows)*len(strides).

    Parameters
    ----------
    functions : Union[FuncWrapper, Callable, List[Union[FuncWrapper, Callable]]]
        The functions, can be either of both types (even in a single array).
    series_names : Union[str, Tuple[str, ...], List[str], List[Tuple[str, ...]]]
        The names of the series on which the feature function should be applied.

        * If `series_names` is a (list of) string (or tuple of a single string),
          then each `function` should require just one series as input.
        * If `series_names` is a (list of) tuple of strings, then each `function` should
          require `len(tuple)` series as input.

        A `list` implies that multiple multiple series (combinations) will be used to
        extract features from; \n
        * If `series_names` is a string or a tuple of strings, then `function` will
          be called only once for the series of this argument.
        * If `series_names` is a list of either strings or tuple of strings, then
          `function` will be called for each entry of this list.

        .. Note::
            when passing a list as `series_names`, all items in this list should
            have the same type, i.e, either \n
            * all a str
            * or, all a tuple _with same length_.\n
            And perfectly match the func-input size.

    windows : Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]]
        All the window sizes.
    strides : Union[float, str, pd.Timedelta, None, List[Union[float, str, pd.Timedelta]]], optional
        All the strides. By default None.

    Note
    ----
    The `windows` and `strides` argument should be either both numeric or
    ``pd.Timedelta`` (depending on de index datatype) - when `strides` is not None.

    """

    def __init__(
        self,
        functions: Union[FuncWrapper, Callable, List[Union[FuncWrapper, Callable]]],
        series_names: Union[str, Tuple[str, ...], List[str], List[Tuple[str, ...]]],
        windows: Optional[
            Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]]
        ] = None,
        strides: Optional[
            Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]]
        ] = None,
    ):
        # Cast functions to FuncWrapper, this avoids creating multiple
        # FuncWrapper objects for the same function in the FeatureDescriptor
        def to_func_wrapper(f: Callable):
            return f if isinstance(f, FuncWrapper) else FuncWrapper(f)

        functions = [to_func_wrapper(f) for f in to_list(functions)]
        # Convert the series names to list of tuples
        series_names = [to_tuple(names) for names in to_list(series_names)]
        # Assert that function inputs (series) all have the same length
        assert all(
            len(series_names[0]) == len(series_name_tuple)
            for series_name_tuple in series_names
        )
        # Convert the other types to list
        windows = to_list(windows)

        self.feature_descriptions: List[FeatureDescriptor] = []
        # Iterate over all combinations
        combinations = [functions, series_names, windows]
        for function, series_name, window in itertools.product(*combinations):  # type: ignore[call-overload]
            self.feature_descriptions.append(
                FeatureDescriptor(function, series_name, window, strides)
            )
