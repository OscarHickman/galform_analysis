# galform_analysis

[![Build Status](https://github.com/OscarHickman/galform_analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/OscarHickman/galform_analysis/actions/workflows/ci.yml)

Tools for analysing GALFORM HDF5 outputs on COSMA.

## Setup

```bash
pip install -r requirements.txt
```

## Quick start

Add `src/` to your Python path, then import directly:

```python
import sys
sys.path.insert(0, '/cosma/apps/durham/dc-hick2/galform_analysis/src')

from config import set_base_dir
from readers.loaders import read_snapshot_data

set_base_dir('/cosma5/data/durham/dc-hick2/Galform_Out/L800/lc16')
data = read_snapshot_data('iz271', 0)
```

Notebooks in `examples/` add `src/` to the path automatically at the top of their first import cell.

## Running tests

```bash
pytest tests -q
```

## Lint

```bash
ruff check src
```

## Author

Oscar Hickman
