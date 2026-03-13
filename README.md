# galform_analysis

Tools for analyzing GALFORM outputs, plus a SLURM submitter for running GALFORM on COSMA.

## Setup

```bash
cd galform_analysis
pip install -r requirements.txt
```

Optional (for imports in notebooks/scripts):

```bash
pip install -e .
```

## Analysis Package Quick Start

```python
from galform_analysis.config import set_base_dir
from galform_analysis import read_snapshot_data

set_base_dir('/cosma5/data/durham/dc-hick2/Galform_Out/L800/gp14')
data = read_snapshot_data('iz271', 0)
```

Examples live in `examples/`.

## GALFORM Submission (COSMA)

Main script:

```bash
python src/galform_execution/submit_galform_job.py --help
```

Typical run:

```bash
python src/galform_execution/submit_galform_job.py --nbody-sim Mill2 --model lc16 --iz 40 --nvol 1-64 --output-folder-name Galform_Test
```

Dry run:

```bash
python src/galform_execution/submit_galform_job.py --nbody-sim Mill2 --model lc16 --iz 40 --nvol 1-64 --dry-run
```

## Execution Config Files

GALFORM execution config is stored in JSON under:

- `src/galform_execution/config/simulations/*.json`
- `src/galform_execution/config/models.json`
- `src/galform_execution/config/dust_params.json`
- `src/galform_execution/config/run_flags.json`

Edit these files to change defaults without touching Python code.

## Development

Run tests:

```bash
pytest tests -q
```

Lint:

```bash
ruff check src/galform_analysis src/galform_execution
```

## Author

Oscar Hickman
