# Comparison 🔎

The table below positions _tsflex_ among other relative packages.

<style type="text/css">
.tg  {border-collapse:collapse;border-color:#ccc;border-spacing:0;}
.tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:0px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:0px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-p7qa{border-color:inherit;font-size:100%;position:-webkit-sticky;position:sticky;text-align:center;top:-1px;
  vertical-align:top;will-change:transform}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-70uo{border-color:inherit;font-size:100%;position:-webkit-sticky;position:sticky;text-align:left;top:-1px;
  vertical-align:top;will-change:transform}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
@media screen and (max-width: 767px) {.tg {width: auto !important;}.tg col {width: auto !important;}.tg-wrap {overflow-x: auto;-webkit-overflow-scrolling: touch;}}</style>
<div class="tg-wrap"><table class="tg">
<thead>
  <tr>
    <th class="tg-70uo"></th>
    <th class="tg-p7qa"><b>tsflex</b> </th>
    <th class="tg-p7qa"><b><a href="https://github.com/dmbee/seglearn">seglearn</a></b> <a href="https://pypi.org/project/seglearn/1.2.3/">v1.2.3</a> </th>
    <th class="tg-p7qa"><b><a href="https://tsfresh.readthedocs.io/en/v0.18.0/">tsfresh</a></b> <a href="https://github.com/blue-yonder/tsfresh/releases/tag/v0.18.0">v.0.18.0</a></th>
    <th class="tg-p7qa"><b><a href="https://tsfel.readthedocs.io/en/latest/">TSFEL</a></b><a href="https://github.com/fraunhoferportugal/tsfel/releases/tag/v0.1.4"> v0.1.4</a><br></th>
    <th class="tg-p7qa"><a href=""><b>Kats</b></a> <a href="https://github.com/facebookresearch/Kats/releases/tag/v0.1"> v0.1</a></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow"><span style="font-weight:bold">General</span></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-0pky">Time column requirements</td>
    <td class="tg-c3ow">datetime-index</td>
    <td class="tg-c3ow">Any - assumes is sorted<br></td>
    <td class="tg-c3ow">Any - sortable</td>
    <td class="tg-c3ow">Any - assumes is sorted</td>
    <td class="tg-c3ow">datetiem index</td>
  </tr>
  <tr>
    <td class="tg-0pky">Multivariate time-series</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
  </tr>
  <tr>
    <td class="tg-0pky">Unevenly sampled data</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">✔️</td>
  </tr>
  <tr>
    <td class="tg-0pky">Time column maintenance</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
  </tr>
  <tr>
    <td class="tg-0pky">Retains output names</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
  </tr>
  <tr>
    <td class="tg-0pky">Multiprocessing</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
  </tr>
  <tr>
    <td class="tg-0pky">Operation Execution time logging</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
  </tr>
  <tr>
    <td class="tg-0pky">Chunking (multiple) time-series</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
  </tr>
  <tr>
    <td class="tg-7btt">Feature extraction</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-0pky">Strided-window definition format</td>
    <td class="tg-c3ow">time-based</td>
    <td class="tg-c3ow">sample-based</td>
    <td class="tg-c3ow">sample-based</td>
    <td class="tg-c3ow">sample-based</td>
    <td class="tg-c3ow">sample-based</td>
  </tr>
  <tr>
    <td class="tg-0pky">Strided-window feature extraction</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
  </tr>
  <tr>
    <td class="tg-0pky">Multiple stride-window combinations</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
  </tr>
  <tr>
    <td class="tg-0pky">Custom Features</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
  </tr>
  <tr>
    <td class="tg-0pky">One-to-one functions</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
  </tr>
  <tr>
    <td class="tg-0pky">One-to-many functions</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
  </tr>
  <tr>
    <td class="tg-0pky">Many-to-one functions</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
  </tr>
  <tr>
    <td class="tg-0pky">Many-to-many functions</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
  </tr>
  <tr>
    <td class="tg-0pky">Categorical data</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
  </tr>
  <tr>
    <td class="tg-0pky">Input data datatype retention</td>
    <td class="tg-c3ow">✔️</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">❌</td>
  </tr>
</tbody>
</table></div>

<br><br>

# Benchmark 📊

The visualization below compares `tsflex v0.1.2.3` against other eligible packages for the strided-rolling feature extraction use-case.<br>
For reference, the in-memory data size when loaded in RAM was 96.4MB.

The figure is constructed by using the [github.com/idlab-predict/tsflex-benchmarking](https://github.com/idlab-predict/tsflex-benchmarking) repo, we further refer to this repository for more details.

<iframe src="https://datapane.com/u/jonasvdd/reports/dkjVy5k/tsflex-benchmark-v2/embed/" width="100%" height="540px" style="border: none;" allowfullscreen>IFrame not supported</iframe><br>