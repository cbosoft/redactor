"""
an __init__.py file in a directory turns the directory into a python module

so it can be used like:
```
from model import build_model

model = build_model(...)
```
"""

from .build import build_model
