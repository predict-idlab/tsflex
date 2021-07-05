# How to contribute

First of all, thank you for considering contributing to `tsflex`.<br>
It's people like you that will help make `tsflex` a great toolkit.

As usual, contributions are managed through GitHub Issues and Pull Requests.

We are welcoming contributions in the following forms:
* **Bug reports**: when filing an issue to report a bug, please use the search tool to ensure the bug hasn't been reported yet;
* **New feature suggestions**: if you think `tsflex` should include a new , please open an issue to ask for it (of course, you should always check that the feature has not been asked for yet :). Think about linking to a pdf version of the paper that first proposed the method when suggesting a new algorithm. 
* **Bugfixes and new feature implementations**: if you feel you can fix a reported bug/implement a suggested feature yourself, do not hesitate to:
  1. fork the project;
  2. implement your bugfix;
  3. submit a pull request referencing the ID of the issue in which the bug was reported / the feature was suggested;
    
When submitting code, please think about code quality, adding proper docstrings including doctests with high code coverage.

## More details on Pull requests

The preferred workflow for contributing to tsflex is to fork the
[main repository](https://github.com/predict-idlab/tsflex) on
GitHub, clone, and develop on a branch. Steps:

1. Fork the [project repository](https://github.com/predict-idlab/tsflex)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the tsflex repo from your GitHub account to your local disk:

   ```bash
   $ git clone git@github.com:YourLogin/tsflex.git
   $ cd tsflex
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never work on the ``master`` branch!

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

5. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork)
to create a pull request from your fork. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)

### Pull Request Checklist

We recommended that your contribution complies with the
following rules before you submit a pull request:

-  Follow the PEP8 Guidelines.

-  If your pull request addresses an issue, please use the pull request title
   to describe the issue and mention the issue number in the pull request description. 
   This will make sure a link back to the original issue is created.

-  All public methods should have informative *numpy* docstrings with sample
   usage presented as doctests when appropriate. Validate whether the generated 
   documentation is properly formatted (see below). 

-  Please prefix the title of your pull request with `[MRG]` (Ready for
   Merge), if the contribution is complete and ready for a detailed review.
   An incomplete contribution -- where you expect to do more work before
   receiving a full review -- should be prefixed `[WIP]` (to indicate a work
   in progress) and changed to `[MRG]` when it matures. WIPs may be useful
   to: indicate you are working on something to avoid duplicated work,
   request broad review of functionality or API, or seek collaborators.
   WIPs often benefit from the inclusion of a
   [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
   in the PR description.

-  When adding additional functionality, provide at least one
   example notebook in the ``tsflex/examples/`` folder or add the functionality in an 
   existing noteboo. Have a look at other examples for reference. 
   Examples should demonstrate why the new functionality is useful in practice and, 
   if possible, benchmark/integrate with other packages.

-  Documentation and high-coverage tests are necessary for enhancements to be
   accepted. Bug-fixes or new features should be provided with 
   [non-regression tests](https://en.wikipedia.org/wiki/Non-regression_testing).
   These tests verify the correct behavior of the fix or feature. In this
   manner, further modifications on the code base are granted to be consistent
   with the desired behavior.
   For the Bug-fixes case, at the time of the PR, this tests should fail for
   the code base in master and pass for the PR code.

You can also check for common programming errors with the following
tools:

- Check whether the hosted-documentation will look fine:

```bash
$ pdoc3 --template-dir docs/pdoc_template/ --http :8181 tsflex/
# you will be able to see the documentation locally on localhost:8181
```

- Validate tests & Code coverage:

```bash
 $ pytest tests --cov-report term-missing --cov=tsflex tests
```

Bonus points for contributions that include a performance analysis with
a benchmark script and profiling output (please report on the mailing
list or on the GitHub issue).

our favored profiling tool is [VizTracer](https://github.com/gaogaotiantian/viztracer)
which can be used as:

```python
from viztracer import VizTracer

with VizTracer(
    log_gc=False,
    log_async=True,
    output_file=f"<SAVE_PATH>/<save_name>.json",
    max_stack_depth=0,
    # make sure to monitor memory and cpu usage
    plugins=["vizplugins.cpu_usage", "vizplugins.memory_usage"],
):
    # the code that is improved in either memory usage / speed
    ...
```
