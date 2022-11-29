[![Python package](https://github.com/inductiva/molecules-binding/actions/workflows/python-package.yml/badge.svg)](https://github.com/inductiva/molecules-binding/actions/workflows/python-package.yml)

# Geometric Deep Learning for Molecules Binding

This repository contains the source code and documentation related with Sofia Guerreiro's 
research on Geometric Deep Learning for molecules. In particular, we will study the interaction
and binding between pairs of molecules.

## Python

We love Python! To make our lives easier and write good code we follow a couple
of common best practices:

* For local development we recommend using Python's virtual environments. This
  way you can keep a separate environment for each project without polluting
  your operating system with additional Python packages. We also recommend using
  `pip` for managing those packages.

  To create a new virtual environment (only once):

  ```bash
  python3 -m venv .env
  ```

  To activate your virtual environment (every time you want to work on the
  current project):

  ```bash
  source .env/bin/activate 
  ```

  It's always a good practice to upgrade `pip` when you create the virtual environment, before installing other packages. In the past we've noticed some
  bugs with older versions of `pip`, causing the installation of incompatible
  versions of several packages.

  ```bash
  pip install --upgrade pip
  ```

  To install the project's required packages (whenever dependencies must be
  updated):

  ```bash
  pip install -r requirements.txt
  ```

* We follow
  [Google's style guide](https://google.github.io/styleguide/pyguide.html).
  Specifically, we format code with `yapf`, and check for bugs and style
  problems with `pylint`.

  Usually, there's no need to run these tools manually. We added continuous
  integration (CI) through GitHub, so every time your pull request has errors
  you'll be alerted via email. Over time you'll get used to these rules and
  coding consistently will become second nature.

  However, if you want to run `yapf` and `pylint` locally, simply install them
  via `pip`:

  ```bash
  pip install yapf pylint
  ```

* We write tests using `pytest`. See [`example_test.py`](example_test.py) for an
  example and delete if example after you copied the template to your new
  project.

  Tests are also run automatically in GitHub via continuous integration. If you
  want to run them locally, install `pytest`:

  ```bash
  pip install pytest
  ```

  and run:

  ```bash
  pytest
  ```

  in your base directory.

  `pytest` is smart enough to discover your tests automatically when you're
  following
  [the right conventions](https://docs.pytest.org/en/stable/goodpractices.html#conventions-for-python-test-discovery). At Inductiva we'd prefer that any
  test file (e.g. `example_test.py`) stays in the same directory next to the
  component it tests (e.g. `example.py`).


## Conda Environments

To automate all the setup commands above, we have created a Conda
environment config file that you can use for your convenience.

Make sure the first line of the conda.yml file contains a name related to 
your project, such as:

```
name: binding_env
```

Note: If you are starting a new project from this template repo, it might work 
automatically for you, because we use the placeholder `name: binding_env`
that gets auto-filled with the repo name.

Now, if you have conda installed, just run:

```bash
conda env create -f conda.yml
```

This will install python 3.9, pylint, yapf, pytest, sphinx, and all the 
package requirements, and you can now activate that environment with:

```bash
conda activate binding_env
```

And you are good to go!
