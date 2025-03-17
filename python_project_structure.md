# Setting Up Python Projects to Avoid Explicit src Imports

Yes, you can set up your project to access modules in `src` without explicitly writing `import src.brainseg...`. Here's how you can do it:

## Option 1: Install your package in development mode

The most standard approach is to install your package in development mode:

1. Create a `setup.py` file in your project root:
```python
from setuptools import setup, find_packages

setup(
    name="brainseg",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
```

2. Install your package in development mode:
```bash
pip install -e .
```

This makes your package importable from anywhere as if it were installed, but changes to the source code take effect immediately without reinstalling.

## Option 2: Use `conftest.py` for pytest

For pytest specifically, you can create a `conftest.py` file in your tests directory:
```python
# tests/conftest.py
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

This will modify the Python path for all tests, allowing you to import directly with:
```python
from brainseg.nifti_utils import save_nifti_from_torch
```

## Option 3: Use `pytest.ini` configuration

You can also create a `pytest.ini` file in your project root:
```ini
[pytest]
pythonpath = src
```

This tells pytest to add the `src` directory to the Python path for all tests.

## Recommended Python Project Layout

For a project with multiple modules in the `src` directory, a good layout would be:
```
project_root/
│
├── src/
│   ├── module1/
│   │   ├── __init__.py
│   │   ├── file1.py
│   │   └── file2.py
│   │
│   ├── module2/
│   │   ├── __init__.py
│   │   ├── file1.py
│   │   └── file2.py
│   │
│   └── __init__.py
│
├── tests/
│   ├── conftest.py
│   ├── test_module1/
│   │   ├── test_file1.py
│   │   └── test_file2.py
│   │
│   └── test_module2/
│       ├── test_file1.py
│       └── test_file2.py
│
├── setup.py
├── pyproject.toml  # For modern Python packaging
├── README.md
└── .gitignore
```

With this structure and the development installation, you can import your modules directly:
```python
# In your tests or other code
from module1.file1 import some_function
from module2.file2 import another_function
```

This approach is clean, follows Python best practices, and works well with tools like pytest, mypy, and IDEs.