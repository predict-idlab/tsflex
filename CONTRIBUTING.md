# Contributing to `tsflex`

First of all, thank you for considering contributing to `tsflex`.<br>
It's people like you that will help make `tsflex` a great toolkit! 🤝

As usual, contributions are managed through GitHub Issues and Pull Requests.  
We invite you to use GitHub's [Issues](https://github.com/predict-idlab/tsflex/issues) to report bugs, request features, or ask questions about the project. To ask use-specific questions, please use the [Discussions](https://github.com/predict-idlab/tsflex/discussions) instead.

If you are new to GitHub, you can read more about how to contribute [here](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).

## How to develop locally

*Note: this guide is tailored to developers using linux*

The following steps assume that your console is at the root folder of this repository.

### Create a new (poetry) Python environment

It is best practice to use a new Python environment when starting on a new project.

We describe two options; 

<details>
<summary><i>Advised option</i>: using <code>poetry shell</code></summary>
For dependency management we use poetry (read more below).<br>
Hence, we advise to use poetry shell to create a Python environment for this project.

1. Install poetry: https://python-poetry.org/docs/#installation <br>
   (If necessary add poetry to the PATH)
2. Create & activate a new python environment: <code>poetry shell</code>

After the poetry shell command your python environment is activated.
</details>

<details>
<summary><i>Alternative option</i>: using <code>python-venv</code></summary>
As alternative option, you can create a Python environment by using python-venv

1. Create a new Python environment: <code>python -m venv venv</code>
2. Activate this environment; <code>source venv/bin/activate</code>
</details>

Make sure that this environment is activated when developing (e.g., installing dependencies, running tests).

### Installing & building the dependencies

We use [`poetry`](https://python-poetry.org/) as dependency manager for this project. 
- The dependencies for installation & development are written in the [`pyproject.toml`](pyproject.toml) file (which is quite similar to a requirements.txt file). 
- To ensure that package versions are consistent with everyone who works on this project poetry uses a [`poetry.lock`](poetry.lock) file (read more [here](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock)).

To install the requirements
```sh
pip install poetry  # install poetry (if you do use the venv option)
poetry install      # install all the dependencies
poetry build        # build the underlying C code
```

### Formatting the code

We use [`black`](https://github.com/psf/black) and [`ruff`](https://github.com/charliermarsh/ruff) to format the code.

To format the code, run the following command (more details in the [`Makefile`](Makefile)):
```sh
make format
```

### Checking the linting

We use [`ruff`](https://github.com/charliermarsh/ruff) to check the linting.

To check the linting, run the following command (more details in the [`Makefile`](Makefile)):
```sh
make lint
```

### Running the tests (& code coverage)

You can run the tests with the following code (more details in the [`Makefile`](Makefile)):

```sh
make test
```

## Documentation

We use [`pdoc`](https://pdoc3.github.io/pdoc/) to generate the documentation.

To generate the documentation and view it locally, run the following command:
```bash
$ pdoc3 --template-dir docs/pdoc_template/ --http :8181 tsflex/
# you will be able to see the documentation locally on localhost:8181
```

---

## Bonus points

Bonus points for contributions that include a performance analysis with a benchmark script and profiling output 👀

<details>
<summary><i>Details on how we profiled <code>tsflex</code></i> </summary>
Our favored profiling tool is <a href="https://github.com/gaogaotiantian/viztracer"><code>VizTracer</code></a> 
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
</details>
